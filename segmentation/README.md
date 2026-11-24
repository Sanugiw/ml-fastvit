# FastViT semantic segmentation on ADE20K

This fine-tunes the ImageNet-pretrained FastViT checkpoints from the main README for
semantic segmentation on ADE20K, using the setup from the paper (Sec. 4.4,
Table 9): a FastViT backbone with a **Semantic FPN** decoder, 512×512 crops,
and mIoU on the 2,000-image validation set.

It's written in plain PyTorch, so you don't need to install mmcv or mmsegmentation.

## Files

| File | Purpose |
| --- | --- |
| `model.py` | FastViT backbone (`fork_feat=True`, which outputs features at strides 4/8/16/32) + FPN neck + Semantic FPN head. Loads the **unfused** ImageNet checkpoint and drops the classifier head. |
| `ade20k.py` | Loads ADE20K (150 classes) from the Hugging Face dataset `shijli/ade20k` (config `segmentation`). Applies the mmseg-style augmentations: random rescale 0.5–2.0, 512 crop, flip, and photometric distortion. |
| `train_seg.py` | Training loop: AdamW, poly LR, gradient accumulation, AMP, periodic mIoU evaluation, checkpoints, and `log.csv`. |
| `eval_seg.py` | mIoU / aAcc / mAcc, a **reparameterized** (fused) model check, latency benchmark, and visualizations. |

## Setup (Colab / Kaggle, GPU runtime)

```bash
git clone <your fork of this repo> ml-fastvit && cd ml-fastvit
pip install timm datasets
```

The first run downloads the dataset (about 1 GB) into the HF cache and the
ImageNet checkpoint into the torch hub cache.

## Train

Paper recipe: batch 32, lr 2e-4, weight decay 1e-4, poly schedule, 40k iterations.
A single GPU can't fit batch 32, so use gradient accumulation (8 × 4 = 32):

```bash
python segmentation/train_seg.py --model fastvit_sa12 \
    --batch-size 8 --accum-steps 4 --iters 40000 --amp \
    --eval-interval 4000 --eval-samples 500 --output runs/sa12
```

* `--checkpoint` defaults to the README URL of the **unfused** ImageNet model for
  `--model`. To use a file you already downloaded, pass its path:
  `--checkpoint /path/to/fastvit_sa12.pth.tar`. Don't use `*_reparam.pth.tar`.
  Fused checkpoints can't be trained, and the script rejects them.
* With limited GPU time, shorten the schedule, e.g. `--iters 8000`. Report
  the change in your write-up, because mIoU will be lower than in the paper.
* Colab sessions time out. Resume with `--resume runs/sa12/last.pth`, which is
  saved at every evaluation. Lower `--eval-interval` if you want more frequent saves.
* For a quick debug run, use
  `--iters 20 --batch-size 2 --accum-steps 1 --train-samples 64 --eval-interval 20 --eval-samples 10`.

Paper results for reference (Table 9, Semantic FPN, ADE20K val): FastViT-SA12
38.0 mIoU, SA24 41.0, SA36 42.9, MA36 44.6. The paper says it uses "the same
settings as PoolFormer [65]" and labels the schedule "80k". 40k iterations at
batch 32 sees the same number of images as 80k at batch 16.

## Evaluate and reparameterize

```bash
python segmentation/eval_seg.py --checkpoint runs/sa12/best.pth \
    --reparam --benchmark --vis 8
```

This prints mIoU for the trained (multi-branch) model and for the fused model.
The two should match up to floating-point noise, which is the point of structural
reparameterization. It also reports latency for both, and saves
image | ground-truth | prediction panels to `runs/sa12/vis/`. Results go
to `eval_results.json`.

## Note: change to `models/fastvit.py`

`fork_feat` previously hard-coded `out_indices = [0, 2, 4, 6]`. For the
SA/MA variants, a RepCPE module sits before the last stage, so index 6 was the
RepCPE output, and the final **attention stage was skipped** in dense
prediction. The indices are now taken from the actual stage positions
(`[0, 2, 4, 7]` for SA/MA, unchanged for T/S models).
