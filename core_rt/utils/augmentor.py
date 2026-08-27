import numpy as np
import random
import warnings
import os
import time
from glob import glob
from skimage import color, io
from PIL import Image

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter, functional, Compose
import torch.nn.functional as F


# -------- Robust helpers to normalize config-like ranges --------
def _is_iterable_but_not_str(x):
    return hasattr(x, "__iter__") and not isinstance(x, (str, bytes, dict))


def _parse_numbers_from_str(s):
    try:
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        return [float(p) for p in parts]
    except Exception:
        return []


def _normalize_saturation_range(x, default=0.0):
    """
    Normalize saturation_range to a type accepted by torchvision ColorJitter:
    - single float (e.g., 1.3) or
    - 2-tuple (low, high) (e.g., (0.7, 1.3))
    Accepts OmegaConf ListConfig, list/tuple, numpy arrays, strings like "0.7,1.3", or single numbers.
    """
    if x is None:
        return float(default)

    if isinstance(x, (int, float)):
        return float(x)

    if isinstance(x, str):
        arr = _parse_numbers_from_str(x)
        if len(arr) == 0:
            return float(default)
        if len(arr) == 1:
            return float(arr[0])
        return (float(arr[0]), float(arr[1]))

    if _is_iterable_but_not_str(x):
        seq = list(x)
        if len(seq) == 0:
            return float(default)
        if len(seq) == 1:
            return float(seq[0])
        return (float(seq[0]), float(seq[1]))

    return float(default)


def _normalize_gamma(x, default=(1.0, 1.0, 1.0, 1.0)):
    """
    Normalize gamma config to four scalars required by AdjustGamma(*gamma):
    (gamma_min, gamma_max, gain_min, gain_max)
    Accepts:
    - single number g -> (g, g, 1.0, 1.0)
    - 2 numbers [g_min, g_max] -> (g_min, g_max, 1.0, 1.0)
    - 4 numbers [g_min, g_max, gain_min, gain_max] -> as is (first four)
    - strings like "0.8,1.2,1.0,1.0"
    - OmegaConf ListConfig / list / tuple / numpy arrays
    """
    if x is None:
        return tuple(default)

    if isinstance(x, (int, float)):
        g = float(x)
        return (g, g, 1.0, 1.0)

    if isinstance(x, str):
        arr = _parse_numbers_from_str(x)
        if len(arr) >= 4:
            return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
        if len(arr) == 2:
            return (float(arr[0]), float(arr[1]), 1.0, 1.0)
        if len(arr) == 1:
            g = float(arr[0])
            return (g, g, 1.0, 1.0)
        return tuple(default)

    if _is_iterable_but_not_str(x):
        seq = list(x)
        if len(seq) >= 4:
            return (float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3]))
        if len(seq) == 2:
            return (float(seq[0]), float(seq[1]), 1.0, 1.0)
        if len(seq) == 1:
            g = float(seq[0])
            return (g, g, 1.0, 1.0)
        return tuple(default)

    return tuple(default)


# ---------------------------------------------------------------


def get_middlebury_images():
    root = "datasets/Middlebury/MiddEval3"
    with open(os.path.join(root, "official_train.txt"), 'r') as f:
        lines = f.read().splitlines()
    return sorted([os.path.join(root, 'trainingQ', f'{name}/im0.png') for name in lines])


def get_eth3d_images():
    return sorted(glob('datasets/ETH3D/two_view_training/*/im0.png'))


def get_kitti_images():
    return sorted(glob('datasets/KITTI/training/image_2/*_10.png'))


def transfer_color(image, style_mean, style_stddev):
    reference_image_lab = color.rgb2lab(image)
    reference_stddev = np.std(reference_image_lab, axis=(0, 1), keepdims=True)  # + 1
    reference_mean = np.mean(reference_image_lab, axis=(0, 1), keepdims=True)

    reference_image_lab = reference_image_lab - reference_mean
    lamb = style_stddev / reference_stddev
    style_image_lab = lamb * reference_image_lab
    output_image_lab = style_image_lab + style_mean
    l, a, b = np.split(output_image_lab, 3, axis=2)
    l = l.clip(0, 100)
    output_image_lab = np.concatenate((l, a, b), axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        output_image_rgb = color.lab2rgb(output_image_lab) * 255
        return output_image_rgb


class AdjustGamma(object):

    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        self.gamma_min, self.gamma_max, self.gain_min, self.gain_max = gamma_min, gamma_max, gain_min, gain_max

    def __call__(self, sample):
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return functional.adjust_gamma(sample, gamma, gain)

    def __repr__(self):
        return f"Adjust Gamma {self.gamma_min}, ({self.gamma_max}) and Gain ({self.gain_min}, {self.gain_max})"


class FlowAugmentor:
    def __init__(self, crop_size, spatial_scale=True, min_scale=-0.2, max_scale=0.5, do_flip=True, yjitter=False,
                 saturation_range=[0.6, 1.4], gamma=[1, 1, 1, 1]):
        # spatial augmentation params
        self.crop_size = crop_size
        self.spatial_scale = spatial_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 1.0
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.yjitter = yjitter
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params (robustly normalized)
        sat_norm = _normalize_saturation_range(saturation_range, default=0.0)
        gmin, gmax, gain_min, gain_max = _normalize_gamma(gamma, default=(1.0, 1.0, 1.0, 1.0))
        self.photo_aug = Compose([
            ColorJitter(brightness=0.4, contrast=0.4, saturation=sat_norm, hue=0.5 / 3.14),
            AdjustGamma(gmin, gmax, gain_min, gain_max)
        ])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        if self.spatial_scale:
            ht, wd = img1.shape[:2]
            if (ht < self.crop_size[0]) or (wd < self.crop_size[1]):
                scale = np.maximum(
                    (self.crop_size[0] + 8) / float(ht),
                    (self.crop_size[1] + 8) / float(wd))
                img1 = cv2.resize(img1, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                img2 = cv2.resize(img2, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                flow = cv2.resize(flow, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                flow = flow * [scale, scale]

            min_scale = np.maximum(
                (self.crop_size[0] + 8) / float(ht),
                (self.crop_size[1] + 8) / float(wd))

            scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
            scale_x = scale
            scale_y = scale
            if np.random.rand() < self.stretch_prob:
                scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
                scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

            scale_x = np.clip(scale_x, min_scale, None)
            scale_y = np.clip(scale_y, min_scale, None)

            if (np.random.rand() < self.spatial_aug_prob) or (ht < self.crop_size[0]) or (wd < self.crop_size[1]):
                # rescale the images
                img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf':  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h':  # h-flip for stereo
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v':  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]
        if self.yjitter:
            y0 = np.random.randint(2, img1.shape[0] - self.crop_size[0] - 2)
            x0 = np.random.randint(2, img1.shape[1] - self.crop_size[1] - 2)

            y1 = y0 + np.random.randint(-2, 2 + 1)
            img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            img2 = img2[y1:y1 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

            img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, flow

    def __call__(self, img1, img2, flow):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow


class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False, yjitter=False,
                 saturation_range=[0.7, 1.3], gamma=[1, 1, 1, 1]):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params (robustly normalized)
        sat_norm = _normalize_saturation_range(saturation_range, default=0.0)
        gmin, gmax, gain_min, gain_max = _normalize_gamma(gamma, default=(1.0, 1.0, 1.0, 1.0))
        self.photo_aug = Compose([
            ColorJitter(brightness=0.3, contrast=0.3, saturation=sat_norm, hue=0.3 / 3.14),
            AdjustGamma(gmin, gmax, gain_min, gain_max)
        ])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if (np.random.rand() < self.spatial_aug_prob) or (ht < self.crop_size[0]) or (wd < self.crop_size[1]):
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf':  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h':  # h-flip for stereo
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v':  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, flow, valid

    def __call__(self, img1, img2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid
