import sys

sys.path.append('core_rt')
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from core_rt.gio_stereo import GIOStereo, autocast
import core_rt.stereo_datasets as datasets
from core_rt.utils.utils import InputPadder
from PIL import Image
import torch.utils.data as data
from pathlib import Path
from matplotlib import pyplot as plt


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _fix_state_dict_prefix_for_dp(model, state_dict):
    model_keys = list(model.state_dict().keys())
    model_is_dp = all(k.startswith('module.') for k in model_keys)

    ckpt_keys = list(state_dict.keys())
    ckpt_has_module = all(k.startswith('module.') for k in ckpt_keys)

    if model_is_dp and not ckpt_has_module:
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    elif (not model_is_dp) and ckpt_has_module:
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    return state_dict


def load_checkpoint_into_model(model, ckpt_path, strict=True):
    assert ckpt_path.endswith('.pth') and os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    logging.info("Loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']

    ckpt = _fix_state_dict_prefix_for_dp(model, ckpt)

    try:
        model.load_state_dict(ckpt, strict=strict)
        logging.info("Done loading checkpoint (strict=True).")
    except RuntimeError as e:
        logging.warning(f"Strict load failed: {e}")
        missing_unexpected = model.load_state_dict(ckpt, strict=False)
        missing = getattr(missing_unexpected, 'missing_keys', [])
        unexpected = getattr(missing_unexpected, 'unexpected_keys', [])
        logging.warning(f"Loaded with strict=False. Missing keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
        logging.warning(f"Unexpected keys: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
        logging.info("Done loading checkpoint (strict=False).")


@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()

        epe_flattened = epe.flatten()

        occ_mask = Image.open(GT_file.replace('disp0GT.pfm', 'mask0nocc.png'))
        occ_mask = np.ascontiguousarray(occ_mask).flatten()

        val = (valid_gt.flatten() >= 0.5) & (occ_mask == 255)
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(
            f"ETH3D {val_id + 1} out of {len(val_dataset)}. EPE {round(image_epe, 4)} D1 {round(image_out, 4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    return {'eth3d-epe': epe, 'eth3d-d1': d1}


@torch.no_grad()
def validate_kitti_2012(model, iters=32, mixed_prec=False):
    # Peform validation using the KITTI-2012 (train) split
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI_2012(aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end - start)

        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()
        epe_flattened = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id + 1) % 10 == 0:
            logging.info(
                f"KITTI Iter {val_id + 1} out of {len(val_dataset)}. EPE {round(image_epe, 4)} D1 {round(image_out, 4)}. Runtime: {format(end - start, '.3f')}s ({format(1 / (end - start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(
        f"Validation KITTI_2012: EPE {epe}, D1 {d1}, {format(1 / avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'kitti-epe': epe, 'kitti-d1': d1}


@torch.no_grad()
def validate_kitti_2015(model, iters=32, mixed_prec=False):
    # Peform validation using the KITTI-2015 (train) split
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI_2015(aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end - start)

        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()
        epe_flattened = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id + 1) % 10 == 0:
            logging.info(
                f"KITTI Iter {val_id + 1} out of {len(val_dataset)}. EPE {round(image_epe, 4)} D1 {round(image_out, 4)}. Runtime: {format(end - start, '.3f')}s ({format(1 / (end - start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(
        f"Validation KITTI_2015: EPE {epe}, D1 {d1}, {format(1 / avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'kitti-epe': epe, 'kitti-d1': d1}


@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False):
    # Peform validation using the KITTI-2015 (train) split
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end - start)

        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()
        epe_flattened = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id + 1) % 10 == 0:
            logging.info(
                f"KITTI Iter {val_id + 1} out of {len(val_dataset)}. EPE {round(image_epe, 4)} D1 {round(image_out, 4)}. Runtime: {format(end - start, '.3f')}s ({format(1 / (end - start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(f"Validation KITTI: EPE {epe}, D1 {d1}, {format(1 / avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'kitti-epe': epe, 'kitti-d1': d1}


@torch.no_grad()
def validate_sceneflow(model, iters=32, mixed_prec=False):
    """ Peform validation using the Scene Flow (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end - start)

        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        epe = torch.abs(flow_pr - flow_gt).flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        if np.isnan(epe[val].mean().item()):
            continue

        out = (epe > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe[val].mean().item()
        if val_id < 9 or (val_id + 1) % 10 == 0:
            logging.info(
                f"Scene Flow Iter {val_id + 1} out of {len(val_dataset)}. EPE {round(image_epe, 4)} D1 {round(image_out, 4)}. Runtime: {format(end - start, '.3f')}s ({format(1 / (end - start), '.2f')}-FPS)")

        epe_list.append(image_epe)
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(
        f"Validation Scene Flow: EPE {epe}, D1 {d1}, {format(1 / avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'scene-disp-epe': epe, 'scene-disp-d1': d1}


@torch.no_grad()
def validate_middlebury(model, iters=32, resolution='F', mixed_prec=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, resolution=resolution)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()
        epe_flattened = epe.flatten()

        occ_mask = Image.open(imageL_file.replace('im0.png', 'mask0nocc.png')).convert('L')
        occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32).flatten()

        val = (valid_gt.reshape(-1) >= 0.5) & (flow_gt[0].reshape(-1) < 192) & (occ_mask == 255)
        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(
            f"Middlebury Iter {val_id + 1} out of {len(val_dataset)}. EPE {round(image_epe, 4)} D1 {round(image_out, 4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"Validation Middlebury{resolution}: EPE {epe}, D1 {d1}")
    return {f'middlebury{resolution}-epe': epe, f'middlebury{resolution}-d1': d1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint",
                        default='checkpoints/sceneflow.pth')
    parser.add_argument('--dataset', help="dataset for evaluation", default='middlebury_Q',
                        choices=["eth3d", "kitti", "sceneflow", "kitti_2012", "kitti_2015"] +  [f"middlebury_{s}" for s
                                                                                               in 'FHQ'])
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float16',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=8,
                        help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=96,
                        help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg",
                        help="correlation volume implementation")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--max_disp', type=int, default=768, help="max disp range")
    args = parser.parse_args()

    model = torch.nn.DataParallel(GIOStereo(args), device_ids=[0])

    if not hasattr(args, 'mixed_precision'):
        args.mixed_precision = (args.precision_dtype in ('float16', 'bfloat16'))

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        load_checkpoint_into_model(model, args.restore_ckpt, strict=True)

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model) / 1e6, '.2f')}M learnable parameters.")

    use_mixed_precision = args.corr_implementation.endswith("_cuda")

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
    elif args.dataset == 'kitti_2012':
        validate_kitti_2012(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
    elif args.dataset == 'kitti_2015':
        validate_kitti_2015(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, resolution=args.dataset[-1], mixed_prec=use_mixed_precision)
    elif args.dataset == 'sceneflow':
        validate_sceneflow(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
