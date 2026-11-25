#!/usr/bin/env bash
# Fresh local setup, sanity check, then full FastViT-SA12 ADE20K training.
set -euo pipefail
# Run from the repo root (this script lives in segmentation/).
cd "$(dirname "$0")/.."

echo "== 1/4 creating .venv"
rm -rf segmentation/.venv
python3 -m venv segmentation/.venv
source segmentation/.venv/bin/activate
pip install --upgrade pip

echo "== 2/4 installing dependencies"
# The ~770 MB torch wheel gets truncated on this network, so fetch it with
# curl (retries + resume), verify the hash, and install from the local file.
WHEEL_DIR="segmentation/cache/wheels"; mkdir -p "$WHEEL_DIR"
TORCH_WHL="torch-2.6.0+cu124-cp310-cp310-linux_x86_64.whl"
TORCH_SHA="7f2ba7f7c0459320a521696f6b5bccc187f59890b23c9dfb6c49b0b87c6bfc97"
until echo "$TORCH_SHA  $WHEEL_DIR/$TORCH_WHL" | sha256sum -c --quiet 2>/dev/null; do
  curl -L --retry 20 --retry-all-errors --retry-delay 5 -C - -o "$WHEEL_DIR/$TORCH_WHL" \
    "https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp310-cp310-linux_x86_64.whl" || true
  echo "$TORCH_SHA  $WHEEL_DIR/$TORCH_WHL" | sha256sum -c --quiet 2>/dev/null || { echo "hash mismatch, re-downloading"; rm -f "$WHEEL_DIR/$TORCH_WHL"; }
done
pip install --retries 10 --timeout 60 "$WHEEL_DIR/$TORCH_WHL" torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
pip install --retries 10 --timeout 60 timm datasets scipy pandas matplotlib ipykernel
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

echo "== 3/4 sanity check (20 iters, downloads ADE20K on first run)"
python segmentation/train_seg.py --model fastvit_sa12 --checkpoint segmentation/checkpoints/fastvit_sa12.pth.tar \
  --iters 20 --batch-size 2 --accum-steps 1 --train-samples 64 --eval-interval 20 --eval-samples 10 \
  --final-eval-samples 10 --log-interval 5 --workers 8 --amp --output segmentation/runs/debug_run

echo "== 4/4 training"
RESUME=""
[ -f segmentation/runs/fastvit_sa12/last.pth ] && RESUME="--resume segmentation/runs/fastvit_sa12/last.pth"
mkdir -p segmentation/runs/fastvit_sa12
python segmentation/train_seg.py --model fastvit_sa12 --checkpoint segmentation/checkpoints/fastvit_sa12.pth.tar $RESUME \
  --iters 40000 --batch-size 8 --accum-steps 4 --amp --eval-interval 2000 --eval-samples 500 \
  --save-interval 500 --workers 8 --output segmentation/runs/fastvit_sa12 2>&1 | tee -a segmentation/runs/fastvit_sa12/train.log
