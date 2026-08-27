import sys

sys.path.append('core_rt')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core_rt.gio_stereo import GIOStereo
from core_rt.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import logging
import os
import skimage.io
import cv2

DEVICE = 'cuda'

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def _fix_state_dict_prefix_for_dp(model, state_dict):
    model_keys = list(model.state_dict().keys())
    model_is_dp = len(model_keys) > 0 and all(k.startswith('module.') for k in model_keys)

    ckpt_keys = list(state_dict.keys())
    ckpt_has_module = len(ckpt_keys) > 0 and all(k.startswith('module.') for k in ckpt_keys)

    if model_is_dp and not ckpt_has_module:
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    elif (not model_is_dp) and ckpt_has_module:
        state_dict = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint_into_model(model, ckpt_path, strict=True):
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    ckpt = _fix_state_dict_prefix_for_dp(model, ckpt)
    try:
        model.load_state_dict(ckpt, strict=strict)
        logging.info(f"Loaded checkpoint (strict={strict}) from {ckpt_path}")
    except RuntimeError as e:
        if strict:
            logging.warning(f"Strict load failed: {e}\nRetry with strict=False ...")
            mi_un = model.load_state_dict(ckpt, strict=False)
            missing = getattr(mi_un, 'missing_keys', [])
            unexpected = getattr(mi_un, 'unexpected_keys', [])
            logging.warning(f"Missing keys (<=10 shown): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
            logging.warning(f"Unexpected keys (<=10 shown): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")
        else:
            raise


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def demo(args):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    model = torch.nn.DataParallel(GIOStereo(args), device_ids=[0])
    load_checkpoint_into_model(model, args.restore_ckpt, strict=True)

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp)
            file_stem = os.path.join(output_directory, imfile1.split('/')[-1])
            disp = disp.cpu().numpy().squeeze()
            if args.save_png:
                disp_16 = np.round(disp * 256).astype(np.uint16)
                skimage.io.imsave(file_stem, disp_16)
            # plt.imsave(file_stem, disp, cmap='jet')

            if args.save_numpy:
                np.save(file_stem.replace('.png', '.npy'), disp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint",
                        default='checkpoints/kitti2015.pth')
    parser.add_argument('--save_png', action='store_true', default=True, help='save output as gray images')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames",
                       default="/mnt/9c69d5cd-01cb-4603-b7c6-06d924734d0c/CYJ/dataset/kitti2015/testing/image_2/*_10.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames",
                       default="/mnt/9c69d5cd-01cb-4603-b7c6-06d924734d0c/CYJ/dataset/kitti2015/testing/image_3/*_10.png")
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames",
    #                     default="/mnt/9c69d5cd-01cb-4603-b7c6-06d924734d0c/CYJ/dataset/kitti2012/testing/colored_0/*_10.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames",
    #                     default="/mnt/9c69d5cd-01cb-4603-b7c6-06d924734d0c/CYJ/dataset/kitti2012/testing/colored_1/*_10.png")
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames",
    #                     default="/data/StereoDatasets/kitti/2012/testing/colored_0/*_10.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames",
    #                     default="/data/StereoDatasets/kitti/2012/testing/colored_1/*_10.png")
    parser.add_argument('--output_directory', help="directory to save output",
                        default="output_kitti2015_disp_0")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'],
                        help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=8, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=96, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp range")

    args = parser.parse_args()

    demo(args)
