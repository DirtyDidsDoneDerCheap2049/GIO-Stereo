import os
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "pretrained"
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("TIMM_HOME", str(CACHE_DIR))
try:
    from timm.models.hub import set_model_cache_dir

    set_model_cache_dir(str(CACHE_DIR))
except Exception:
    pass

import hydra
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
import numpy as np
import matplotlib
import wandb
import imageio.v2 as imageio
from PIL import Image

from core_rt.gio_stereo import GIOStereo
import core_rt.stereo_datasets as datasets
from core_rt.utils.utils import InputPadder


def gray_2_colormap_np(img, cmap='rainbow', max=None):
    img = img.detach().float().cpu().numpy().squeeze()
    assert img.ndim == 2
    img[img < 0] = 0
    mask_invalid = img < 1e-10
    if max is None:
        img = img / (img.max() + 1e-8)
    else:
        img = img / (max + 1e-8)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.colormaps.get_cmap(cmap)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (mapper.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    colormap[mask_invalid] = 0
    return colormap


def tensor_to_uint8_img(t):
    # BCHW -> HWC uint8
    t = t.detach().float().cpu()
    if t.dim() == 4:
        t = t[0]
    t = t.clamp(min=t.min().item(), max=t.max().item())
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)
    t = (t * 255.0).byte().numpy()
    if t.shape[0] == 1:
        t = np.repeat(t, 3, axis=0)
    t = np.transpose(t, (1, 2, 0))
    return t


def sequence_loss(agg_pred, iter_preds, disp_gt, valid, loss_gamma=0.9, max_disp=192):
    n_predictions = len(iter_preds)
    assert n_predictions >= 1

    mag = torch.sum(disp_gt ** 2, dim=1).sqrt()
    mask = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert mask.shape == disp_gt.shape, [mask.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[mask.bool()]).any()

    loss = 1.0 * F.smooth_l1_loss(agg_pred[mask.bool()], disp_gt[mask.bool()], reduction='mean')

    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = (iter_preds[i] - disp_gt).abs()
        assert i_loss.shape == mask.shape, [i_loss.shape, mask.shape, disp_gt.shape, iter_preds[i].shape]
        loss = loss + i_weight * i_loss[mask.bool()].mean()

    epe = torch.sum((iter_preds[-1] - disp_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[mask.view(-1)]

    metrics = {
        'train/epe': epe.mean(),
        'train/1px': (epe < 1).float().mean(),
        'train/3px': (epe < 3).float().mean(),
        'train/5px': (epe < 5).float().mean(),
    }
    return loss, metrics


def fetch_optimizer(cfg, model):
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wdecay, eps=1e-8)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, cfg.lr, cfg.total_step + 100,
        pct_start=0.01, cycle_momentum=False, anneal_strategy='linear'
    )
    return optimizer, scheduler


def maybe_visualize(cfg, accelerator, step, tensors, save_dir):
    viz_cfg = getattr(cfg, 'viz', None)
    if viz_cfg is None:
        return

    from omegaconf import OmegaConf as _OC
    if not isinstance(viz_cfg, dict):
        viz_cfg = _OC.to_container(viz_cfg, resolve=True)

    enabled = bool(viz_cfg.get('enabled', False))
    if not enabled or not accelerator.is_main_process:
        return

    every = int(viz_cfg.get('every', 1000))
    if step % every != 0:
        return

    items = viz_cfg.get('items', [])
    if not isinstance(items, (list, tuple)):
        items = [items]

    dest = str(viz_cfg.get('to', 'disk')).lower()
    max_samples = int(viz_cfg.get('max_samples', 1))

    out_dir = None
    if dest in ('disk', 'both'):
        out_dir = Path(save_dir) / 'vis'
        out_dir.mkdir(parents=True, exist_ok=True)

    imgs_to_log = {}
    for name in items:
        if name not in tensors:
            continue
        t = tensors[name]
        b = min(t.shape[0], max_samples)

        if name in ('disp_pred', 'disp_gt'):
            arrs = [gray_2_colormap_np(t[i].squeeze()) for i in range(b)]
        else:
            arrs = [tensor_to_uint8_img(t[i].unsqueeze(0)) for i in range(b)]
        imgs_to_log[name] = arrs

    if dest in ('wandb', 'both'):
        log_dict = {}
        for k, arrs in imgs_to_log.items():
            for idx, arr in enumerate(arrs):
                log_dict[f'{k}_{idx}'] = wandb.Image(arr, caption=f'step:{step} {k}[{idx}]')
        if len(log_dict) > 0:
            accelerator.log(log_dict, step)

    if dest in ('disk', 'both') and out_dir is not None:
        for k, arrs in imgs_to_log.items():
            for idx, arr in enumerate(arrs):
                fpath = out_dir / f'step_{step: 07d}_{k}_{idx}.png'
                imageio.imwrite(fpath, arr)


@hydra.main(version_base=None, config_path='config', config_name='train_zero_shot')
def main(cfg):
    set_seed(cfg.seed)
    logger = get_logger(__name__)
    Path(cfg.save_path).mkdir(exist_ok=True, parents=True)

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    mp_map = {'float16': 'fp16', 'bfloat16': 'bf16', 'float32': 'no'}
    acc_mp = mp_map.get(str(cfg.get('precision_dtype', 'bfloat16')), 'bf16')

    accelerator = Accelerator(
        mixed_precision=acc_mp,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        log_with='wandb',
        kwargs_handlers=[kwargs],
        step_scheduler_with_optimizer=False
    )

    if cfg.wandb.get('mode', '') != 'offline':
        cfg.wandb['mode'] = 'offline'
    accelerator.init_trackers(
        project_name=cfg.project_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        init_kwargs={'wandb': cfg.wandb}
    )

    OmegaConf.set_struct(cfg, False)
    cfg.mixed_precision = (acc_mp != 'no')

    global_batch = int(cfg.batch_size)
    world_size = int(accelerator.num_processes)
    if global_batch % world_size != 0:
        raise ValueError(
            f"global batch_size {global_batch} must be divisible by world_size {world_size}"
        )
    per_process_batch = global_batch // world_size
    cfg.global_batch_size = global_batch
    cfg.batch_size = per_process_batch
    logger.info(
        f"Using global batch_size={global_batch} (per_process={per_process_batch}, world_size={world_size})"
    )

    train_loader = datasets.fetch_dataloader(cfg)

    val_loaders = {}
    for dataset_name in cfg.val_dataset:
        logger.info(f"Preparing validation dataset: {dataset_name}")
        try:
            dataset_class = getattr(datasets, dataset_name)
        except AttributeError:
            raise ValueError(f"Dataset class {dataset_name} not found")

        if dataset_name == 'Middlebury':
            val_dataset = dataset_class(aug_params=None, resolution=cfg.resolution)
        else:
            val_dataset = dataset_class(aug_params=None)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=4, drop_last=False
        )
        val_loaders[dataset_name] = val_loader

    model = GIOStereo(cfg)

    resume_step = 0
    resume_ckpt = getattr(cfg, 'resume', None)
    if isinstance(resume_ckpt, str) and resume_ckpt.endswith(".pth") and os.path.exists(resume_ckpt):
        ckpt_stem = Path(resume_ckpt).stem
        try:
            resume_step = int(ckpt_stem)
        except ValueError:
            logger.warning(f"Could not parse step from checkpoint name {resume_ckpt}, starting from step 0")

        logger.info(f"Loading resume checkpoint from {resume_ckpt} (resuming at step {resume_step})")
        checkpoint = torch.load(resume_ckpt, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint)
        clean_state = {}
        for k, v in state.items():
            nk = k.replace('module.', '')
            clean_state[nk] = v
        missing, unexpected = model.load_state_dict(clean_state, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading: {unexpected}")
        del checkpoint, state, clean_state
        logger.info("Checkpoint loaded successfully.")
    elif isinstance(cfg.restore_ckpt, str) and cfg.restore_ckpt.endswith(".pth") and os.path.exists(cfg.restore_ckpt):
        logger.info(f"Loading checkpoint from {cfg.restore_ckpt}")
        checkpoint = torch.load(cfg.restore_ckpt, map_location='cpu')

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state = checkpoint['model']
            logger.info(f"Loaded training checkpoint at step {checkpoint.get('total_step', 'unknown')}")
        else:
            state = checkpoint.get('state_dict', checkpoint)
            logger.info("Loaded model weights")

        clean_state = {}
        for k, v in state.items():
            nk = k.replace('module. ', '')
            clean_state[nk] = v
        missing, unexpected = model.load_state_dict(clean_state, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading:  {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading: {unexpected}")
        del checkpoint, state, clean_state
        logger.info("Checkpoint loaded successfully.")

    optimizer, lr_scheduler = fetch_optimizer(cfg, model)

    if resume_step > 0:
        lr_scheduler.last_epoch = resume_step
        logger.info(f"LR schedule set to step {resume_step}")

    train_loader, model, optimizer, lr_scheduler = accelerator.prepare(
        train_loader, model, optimizer, lr_scheduler
    )
    val_loaders = {
        name: accelerator.prepare(loader)
        for name, loader in val_loaders.items()
    }
    model.to(accelerator.device)
    logger.info(f"Using {accelerator.num_processes} GPUs")

    total_step = resume_step
    should_keep_training = True

    while should_keep_training:
        model.train()
        try:
            (model.module if hasattr(model, 'module') else model).freeze_bn()
        except Exception:
            pass

        pbar = tqdm(train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process)

        for data in pbar:
            _, left, right, disp_gt, valid = [x for x in data]

            with accelerator.autocast():
                agg_pred, disp_preds = model(left, right, iters=cfg.train_iters)

            loss, metrics = sequence_loss(
                agg_pred=agg_pred, iter_preds=disp_preds, disp_gt=disp_gt, valid=valid, max_disp=cfg.max_disp
            )

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_step += 1

            loss_r = accelerator.reduce(loss.detach(), reduction='mean')
            metrics_r = accelerator.reduce(metrics, reduction='mean')
            accelerator.log({'train/loss': loss_r, 'train/learning_rate': optimizer.param_groups[0]['lr']}, total_step)
            accelerator.log(metrics_r, total_step)

            if accelerator.is_main_process:
                try:
                    epe_val = float(metrics_r['train/epe'])
                except Exception:
                    epe_val = float(metrics_r.get('train/epe', 0.0))
                pbar.set_postfix(loss=float(loss_r), epe=epe_val, lr=optimizer.param_groups[0]['lr'])

            tensors_for_viz = {
                'disp_pred': disp_preds[-1],
                'disp_gt': disp_gt,
                'left': left,
                'right': right,
            }
            maybe_visualize(cfg, accelerator, total_step, tensors_for_viz, cfg.save_path)

            if (total_step > 0) and (total_step % cfg.save_frequency == 0) and accelerator.is_main_process:
                save_path = Path(cfg.save_path) / f"{total_step}.pth"
                model_save = accelerator.unwrap_model(model)
                torch.save(model_save.state_dict(), save_path)
                del model_save

            if (total_step > 0) and (total_step % cfg.val_frequency == 0):
                torch.cuda.empty_cache()
                model.eval()

                for name, val_loader in val_loaders.items():
                    logger.info(f"\nEvaluating on {name}...")
                    elem_num, total_epe, total_out = 0, 0.0, 0.0

                    threshold = 3.0
                    if name == 'Middlebury':
                        threshold = 2.0
                    elif name == 'ETH3D':
                        threshold = 1.0

                    val_pbar = tqdm(val_loader, dynamic_ncols=True, disable=not accelerator.is_main_process)

                    with torch.no_grad():
                        for vdata in val_pbar:
                            (imageL_file, imageR_file, GT_file), lval, rval, disp_gt_v, valid_v = [x for x in vdata]
                            padder = InputPadder(lval.shape, divis_by=32)
                            lval, rval = padder.pad(lval, rval)

                            disp_pred_v = model(lval, rval, iters=cfg.valid_iters, test_mode=True)
                            disp_pred_v = padder.unpad(disp_pred_v)

                            assert disp_pred_v.shape == disp_gt_v.shape
                            epe = torch.abs(disp_pred_v - disp_gt_v)
                            out = (epe > threshold).float()
                            epe = torch.squeeze(epe, dim=1)
                            out = torch.squeeze(out, dim=1)

                            if name == 'ETH3D':
                                try:
                                    occ_mask = Image.open(GT_file[0].replace('disp0GT.pfm', 'mask0nocc.png'))
                                    occ_mask = np.ascontiguousarray(occ_mask)
                                    occ_mask = torch.from_numpy(occ_mask).to(valid_v.device)
                                    valid_v = (valid_v >= 0.5) & (occ_mask == 255)
                                except Exception:
                                    valid_v = (valid_v >= 0.5)
                            elif name == 'Middlebury':
                                try:
                                    occ_mask = Image.open(imageL_file[0].replace('im0.png', 'mask0nocc.png')).convert(
                                        'L')
                                    occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32)
                                    occ_mask = torch.from_numpy(occ_mask).to(valid_v.device)
                                    valid_v = (valid_v >= 0.5) & (occ_mask == 255)
                                except Exception:
                                    valid_v = (valid_v >= 0.5)

                            valid_mask = valid_v >= 0.5
                            if valid_mask.sum() > 0:
                                local_epe = epe[valid_mask].mean()
                                local_out = out[valid_mask].mean()

                                epe_gathered, out_gathered = accelerator.gather_for_metrics((local_epe, local_out))

                                if accelerator.is_main_process:
                                    if epe_gathered.dim() == 0:
                                        elem_num += 1
                                        total_epe += float(epe_gathered)
                                        total_out += float(out_gathered)
                                    else:
                                        elem_num += epe_gathered.numel()
                                        total_epe += float(epe_gathered.sum())
                                        total_out += float(out_gathered.sum())

                    if accelerator.is_main_process:
                        if elem_num > 0:
                            avg_epe = total_epe / elem_num
                            avg_d1 = 100 * total_out / elem_num
                            logger.info(f"{name} - EPE: {avg_epe:.3f}, D1: {avg_d1:.3f}")
                            accelerator.log({
                                f'{name}/val_epe': avg_epe,
                                f'{name}/val_d1': avg_d1
                            }, total_step)
                        else:
                            logger.warning(f"{name} - No valid samples found!")

                model.train()
                try:
                    (model.module if hasattr(model, 'module') else model).freeze_bn()
                except Exception:
                    pass

            if total_step >= cfg.total_step:
                should_keep_training = False
                break

    if accelerator.is_main_process:
        save_path = Path(cfg.save_path) / 'final.pth'
        model_save = accelerator.unwrap_model(model)
        torch.save(model_save.state_dict(), save_path)
        del model_save

    accelerator.end_training()


if __name__ == '__main__':
    main()
