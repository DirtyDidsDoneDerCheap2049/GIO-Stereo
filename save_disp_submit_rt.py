import sys

sys.path.append('core_rt')

import argparse
import logging
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from core_rt.gio_stereo import GIOStereo
from core_rt.utils.frame_utils import writePFM
from core_rt.utils.utils import InputPadder


IMAGE_EXTS = ('.png', '.pgm', '.ppm', '.jpg', '.jpeg')


def _fix_state_dict_prefix_for_dp(model, state_dict):
    """Match checkpoint keys to whether the current model uses DataParallel."""
    model_keys = list(model.state_dict().keys())
    model_is_dp = len(model_keys) > 0 and all(k.startswith('module.') for k in model_keys)

    ckpt_keys = list(state_dict.keys())
    ckpt_has_module = len(ckpt_keys) > 0 and all(k.startswith('module.') for k in ckpt_keys)

    if model_is_dp and not ckpt_has_module:
        return {'module.' + k: v for k, v in state_dict.items()}
    if not model_is_dp and ckpt_has_module:
        return {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint_into_model(model, ckpt_path, strict=True):
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    ckpt = _fix_state_dict_prefix_for_dp(model, ckpt)

    try:
        model.load_state_dict(ckpt, strict=strict)
        logging.info("Loaded checkpoint (strict=%s) from %s", strict, ckpt_path)
    except RuntimeError as err:
        if not strict:
            raise
        logging.warning("Strict load failed: %s\nRetry with strict=False ...", err)
        info = model.load_state_dict(ckpt, strict=False)
        missing = getattr(info, 'missing_keys', [])
        unexpected = getattr(info, 'unexpected_keys', [])
        logging.warning("Missing keys (<=10 shown): %s%s", missing[:10], ' ...' if len(missing) > 10 else '')
        logging.warning("Unexpected keys (<=10 shown): %s%s", unexpected[:10], ' ...' if len(unexpected) > 10 else '')


def build_model(args, device):
    if device.type == 'cuda':
        model = torch.nn.DataParallel(GIOStereo(args), device_ids=[0])
        load_checkpoint_into_model(model, args.restore_ckpt, strict=True)
        model = model.module
    else:
        model = GIOStereo(args)
        load_checkpoint_into_model(model, args.restore_ckpt, strict=True)

    model.to(device)
    model.eval()
    return model


def resolve_splits(splits):
    if not splits:
        return ['training', 'test']

    resolved = []
    for split in splits:
        if split == 'all':
            resolved.extend(['training', 'test'])
        else:
            resolved.append(split)

    deduped = []
    for split in resolved:
        if split not in ('training', 'test'):
            raise ValueError(f"Unsupported split: {split}. Use training, test, or all.")
        if split not in deduped:
            deduped.append(split)
    return deduped


def find_image(scene_dir, stem):
    for ext in IMAGE_EXTS:
        candidate = scene_dir / f'{stem}{ext}'
        if candidate.is_file():
            return candidate
    return None


def iter_middlebury_pairs(dataset_root, resolution, splits):
    for split in splits:
        split_tag = f'{split}{resolution}'
        split_dir = dataset_root / split_tag
        if not split_dir.is_dir():
            logging.warning("Missing Middlebury split directory: %s", split_dir)
            continue

        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            left = find_image(scene_dir, 'im0')
            right = find_image(scene_dir, 'im1')
            if left is None or right is None:
                logging.warning("Skip %s because im0/im1 was not found", scene_dir)
                continue
            yield {
                'benchmark': 'middlebury',
                'split': split,
                'split_tag': split_tag,
                'scene': scene_dir.name,
                'left': left,
                'right': right,
            }


def iter_eth3d_pairs(dataset_root, splits):
    used_split_dirs = set()
    for split in splits:
        candidates = []
        if dataset_root.name == f'two_view_{split}':
            candidates.append(dataset_root)
        candidates.append(dataset_root / f'two_view_{split}')

        for split_dir in candidates:
            split_dir = split_dir.resolve()
            if split_dir in used_split_dirs:
                continue
            used_split_dirs.add(split_dir)

            if not split_dir.is_dir():
                logging.warning("Missing ETH3D split directory: %s", split_dir)
                continue

            for scene_dir in sorted(split_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                left = find_image(scene_dir, 'im0')
                right = find_image(scene_dir, 'im1')
                if left is None or right is None:
                    logging.warning("Skip %s because im0/im1 was not found", scene_dir)
                    continue
                yield {
                    'benchmark': 'eth3d',
                    'split': split,
                    'scene': scene_dir.name,
                    'left': left,
                    'right': right,
                }


def load_image(imfile, device):
    img = np.array(Image.open(imfile).convert('RGB')).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)


def cuda_synchronize_if_needed(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def predict_disparity(model, pair, args, device):
    image1 = load_image(pair['left'], device)
    image2 = load_image(pair['right'], device)

    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)

    cuda_synchronize_if_needed(device)
    start_time = time.perf_counter()
    disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
    cuda_synchronize_if_needed(device)
    runtime = time.perf_counter() - start_time

    disp = padder.unpad(disp)
    disp = disp.detach().cpu().numpy().squeeze().astype(np.float32)
    return disp, runtime


def middlebury_output_paths(pair, args):
    output_root = Path(args.output_root) if args.output_root else Path(args.data_root)
    scene_dir = output_root / pair['split_tag'] / pair['scene']
    scene_dir.mkdir(parents=True, exist_ok=True)
    return scene_dir / f'disp0{args.method_name}.pfm', scene_dir / f'time{args.method_name}.txt'


def eth3d_output_paths(pair, args):
    output_root = Path(args.output_root) if args.output_root else Path(f'output_eth3d_{args.method_name}')
    output_dir = output_root / 'low_res_two_view'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{pair['scene']}.pfm", output_dir / f"{pair['scene']}.txt"


def create_middlebury_zip(args, pairs):
    zip_path = Path(args.zip_name)
    logging.info('Creating Middlebury archive: %s', zip_path)
    file_count = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for pair in pairs:
            disp_file, runtime_file = middlebury_output_paths(pair, args)
            if not disp_file.is_file() or not runtime_file.is_file():
                raise RuntimeError(f'Missing result files for {pair["split_tag"]}/{pair["scene"]}: {disp_file}, {runtime_file}')

            arc_dir = Path(pair['split_tag']) / pair['scene']
            zf.write(disp_file, arc_dir / disp_file.name)
            zf.write(runtime_file, arc_dir / runtime_file.name)
            file_count += 2
    logging.info('Middlebury archive created: %s (%d files)', zip_path, file_count)


def create_eth3d_zip(args, pairs):
    zip_path = Path(args.zip_name)
    logging.info('Creating ETH3D archive: %s', zip_path)
    file_count = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for pair in pairs:
            disp_file, runtime_file = eth3d_output_paths(pair, args)
            if not disp_file.is_file() or not runtime_file.is_file():
                raise RuntimeError(f'Missing result files for {pair["scene"]}: {disp_file}, {runtime_file}')

            arc_dir = Path('low_res_two_view')
            zf.write(disp_file, arc_dir / disp_file.name)
            zf.write(runtime_file, arc_dir / runtime_file.name)
            file_count += 2
    logging.info('ETH3D archive created: %s (%d files)', zip_path, file_count)
    validate_eth3d_zip(zip_path)


def validate_eth3d_zip(zip_path):
    allowed_suffixes = {'.pfm', '.txt'}
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        txt_contents = {
            name: zf.read(name).decode('utf-8').strip()
            for name in names
            if Path(name).suffix == '.txt'
        }

    if not names:
        raise RuntimeError(f'ETH3D archive is empty: {zip_path}')

    for name in names:
        path = Path(name)
        parts = path.parts
        if len(parts) != 2 or parts[0] != 'low_res_two_view':
            raise RuntimeError(
                f'Invalid ETH3D archive entry: {name}. '
                'Archive root must contain only low_res_two_view/*.pfm and low_res_two_view/*.txt.')
        if path.suffix not in allowed_suffixes:
            raise RuntimeError(f'Invalid ETH3D archive file type: {name}')

    pfm_stems = {Path(name).stem for name in names if Path(name).suffix == '.pfm'}
    txt_stems = {Path(name).stem for name in names if Path(name).suffix == '.txt'}
    missing_txt = sorted(pfm_stems - txt_stems)
    missing_pfm = sorted(txt_stems - pfm_stems)
    if missing_txt or missing_pfm:
        raise RuntimeError(
            f'ETH3D archive has unmatched metadata files. '
            f'Missing txt for: {missing_txt}; missing pfm for: {missing_pfm}')

    for name, text in txt_contents.items():
        fields = text.split()
        if len(fields) != 2 or fields[0] != 'runtime':
            raise RuntimeError(f'Invalid ETH3D runtime metadata in {name}: {text!r}')
        try:
            float(fields[1])
        except ValueError as err:
            raise RuntimeError(f'Invalid ETH3D runtime value in {name}: {fields[1]!r}') from err

    logging.info('ETH3D archive format check passed: %s', zip_path)


def write_runtime(runtime_file, runtime, benchmark):
    if benchmark == 'middlebury':
        text = f'{runtime:.6f}\n'
    elif benchmark == 'eth3d':
        text = f'runtime {runtime:.6f}\n'
    else:
        raise ValueError(f'Unsupported benchmark: {benchmark}')
    runtime_file.write_text(text)


def get_pairs(args):
    data_root = Path(args.data_root)
    splits = resolve_splits(args.splits)

    if args.benchmark == 'middlebury':
        return list(iter_middlebury_pairs(data_root, args.resolution, splits))
    if args.benchmark == 'eth3d':
        return list(iter_eth3d_pairs(data_root, splits))
    raise ValueError(f'Unsupported benchmark: {args.benchmark}')


def validate_args(args):
    if any(sep in args.method_name for sep in ('/', '\\')):
        raise ValueError('method_name must not contain path separators')
    if args.benchmark == 'middlebury' and args.resolution not in ('Q', 'H', 'F'):
        raise ValueError('Middlebury resolution must be Q, H, or F')


def run(args):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    validate_args(args)

    if args.cuda_visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

    requested_device = torch.device(args.device)
    if requested_device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')

    pairs = get_pairs(args)
    if len(pairs) == 0:
        raise RuntimeError('No image pairs found. Check --data_root, --benchmark, --resolution, and --splits.')

    model = build_model(args, requested_device)
    logging.info("Found %d image pairs for %s", len(pairs), args.benchmark)

    with torch.no_grad():
        for pair in tqdm(pairs):
            if args.benchmark == 'middlebury':
                disp_file, runtime_file = middlebury_output_paths(pair, args)
            else:
                disp_file, runtime_file = eth3d_output_paths(pair, args)

            if args.skip_existing and disp_file.is_file() and runtime_file.is_file():
                logging.info("Skip existing result: %s", disp_file)
                continue

            disp, runtime = predict_disparity(model, pair, args, requested_device)
            writePFM(str(disp_file), disp)
            write_runtime(runtime_file, runtime, args.benchmark)

    if args.zip_name:
        if args.benchmark == 'middlebury':
            create_middlebury_zip(args, pairs)
        else:
            create_eth3d_zip(args, pairs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export GIO-Stereo disparities for Middlebury v3 or ETH3D low-res two-view submission.')
    parser.add_argument('--benchmark', default='eth3d', choices=['middlebury', 'eth3d'])

    # ---- paths ----
    parser.add_argument('--data_root', default='/mnt/9c69d5cd-01cb-4603-b7c6-06d924734d0c/CYJ/dataset/ETH3D')
    parser.add_argument('--output_root', default='output_eth3d_GIO')
    parser.add_argument('--zip_name', default='eth3d_low_res_two_view_GIO.zip')
    parser.add_argument('--restore_ckpt', default='checkpoints/eth3d.pth')
    parser.add_argument('--method_name', default='GIO')
    parser.add_argument('--resolution', default='Q', choices=['Q', 'H', 'F'])
    parser.add_argument('--splits', default=['all'], nargs='+') # ← 'training', 'test', or ['all']
    parser.add_argument('--skip_existing', action='store_true', default=True) # ← skip scenes that already have .pfm + .txt
    parser.add_argument('--device', default='cuda') # ← 'cuda' or 'cpu'
    parser.add_argument('--cuda_visible_devices', default='1') # ← '0', '1', etc.
    parser.add_argument('--mixed_precision', action='store_true', default=False)
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--valid_iters', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=96)
    parser.add_argument('--corr_levels', type=int, default=2)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--max_disp', type=int, default=192)

    run(parser.parse_args())
