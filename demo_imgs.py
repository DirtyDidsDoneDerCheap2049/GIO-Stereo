import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core_rt.gio_stereo import GIOStereo
from core_rt.utils.utils import InputPadder
from PIL import Image
import os
import cv2
import logging
import matplotlib.pyplot as plt

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
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    else:
        img = img[..., :3]
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
        assert len(left_images) == len(right_images), "left/right image counts do not match"
        logging.info(f"Found {len(left_images)} image pairs. Saving files to {output_directory}/")

        for imfile1, imfile2 in tqdm(zip(left_images, right_images), total=len(left_images)):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            disp_out = model(image1, image2, iters=args.valid_iters, test_mode=True)
            if isinstance(disp_out, (list, tuple)):
                disp_out = disp_out[-1]

            disp = padder.unpad(disp_out).cpu().numpy().squeeze()  # (H,W)

            # stem = os.path.splitext(os.path.basename(imfile1))[0]
            stem = os.path.basename(os.path.dirname(imfile1))
            out_png = output_directory / f"{stem}.png"

            plt.imsave(out_png, disp, cmap='jet')

            if args.save_numpy:
                np.save(output_directory / f"{stem}.npy", disp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint",
                        default='checkpoints/middlebury.pth')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames",
    #                 default="/mnt/9c69d5cd-01cb-4603-b7c6-06d924734d0c/CYJ/dataset/kitti2012/testing/colored_0/*.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames",
    #                     default="/mnt/9c69d5cd-01cb-4603-b7c6-06d924734d0c/CYJ/dataset/kitti2012/testing/colored_1/*.png")
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames",
    #                     default="/mnt/9c69d5cd-01cb-4603-b7c6-06d924734d0c/CYJ/dataset/kitti2015/testing/image_2/*.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames",
    #                     default="/mnt/9c69d5cd-01cb-4603-b7c6-06d924734d0c/CYJ/dataset/kitti2015/testing/image_3/*.png")
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames",
                       default="/mnt/9c69d5cd-01cb-4603-b7c6-06d924734d0c/CYJ/dataset/Middlebury/testF/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames",
                        default="/mnt/9c69d5cd-01cb-4603-b7c6-06d924734d0c/CYJ/dataset/Middlebury/testF/*/im1.png")
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames",
    #                     default="/mnt/9c69d5cd-01cb-4603-b7c6-06d924734d0c/wjy/dataset/Middleburys/2021/*/im0.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames",
    #                     default="/mnt/9c69d5cd-01cb-4603-b7c6-06d924734d0c/wjy/dataset/Middleburys/2021/*/im1.png")
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames",
    #                    default="/mnt/9c69d5cd-01cb-4603-b7c6-06d924734d0c/CYJ/dataset/ETH3D/two_view_training/*/im0.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames",
    #                     default="/mnt/9c69d5cd-01cb-4603-b7c6-06d924734d0c/CYJ/dataset/ETH3D/two_view_training/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="TSET_OUTPUT")
    parser.add_argument('--mixed_precision', action='store_true', default=True, help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type')
    parser.add_argument('--valid_iters', type=int, default=8, help='number of updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=96, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--max_disp', type=int, default=416, help="max disp range")

    args = parser.parse_args()
    demo(args)
