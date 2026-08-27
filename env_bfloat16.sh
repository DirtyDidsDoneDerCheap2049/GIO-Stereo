#!/usr/bin/env bash
set -euo pipefail

# PyTorch stack for CUDA 11.7.
conda install -y \
  pytorch==2.0.1 \
  torchvision==0.15.2 \
  torchaudio==2.0.2 \
  pytorch-cuda=11.7 \
  -c pytorch -c nvidia

# Direct runtime dependencies used by training, evaluation, demos, and export.
python -m pip install \
  accelerate==1.0.1 \
  hydra-core==1.3.2 \
  omegaconf==2.3.0 \
  wandb==0.17.6 \
  numpy==1.24.3 \
  scipy==1.10.1 \
  Pillow==10.4.0 \
  imageio==2.35.1 \
  opencv-python==4.9.0.80 \
  scikit-image==0.21.0 \
  matplotlib==3.7.4 \
  timm==0.5.4 \
  tqdm==4.66.5 \
  tensorboard==2.14.0

# Report incompatible or missing package requirements.
python -m pip check
