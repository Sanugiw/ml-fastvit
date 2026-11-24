"""Fine-tune an ImageNet-pretrained FastViT + Semantic FPN on ADE20K.

Paper / PoolFormer recipe (8 GPUs): 512x512 crops, batch 32, AdamW
lr 2e-4, weight decay 1e-4, poly LR (power 0.9), 40k iterations.
On a single GPU use --batch-size N --accum-steps M with N*M = 32 to keep the
same effective batch, and reduce --iters if you are short on compute.

Example (Colab / Kaggle, from the repo root):
    python segmentation/train_seg.py --model fastvit_sa12 \
        --checkpoint https://docs-assets.developer.apple.com/ml-research/models/fastvit/image_classification_models/fastvit_sa12.pth.tar \
        --batch-size 8 --accum-steps 4 --iters 40000 --amp --output runs/sa12
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmentation.ade20k import (  # noqa: E402
    IGNORE_INDEX, NUM_CLASSES, CLASSES, ADE20KSeg, load_hf_split,
)
from segmentation.model import (  # noqa: E402
    CHECKPOINT_URLS, VARIANTS, FastViTSemanticFPN,
)


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", default="fastvit_sa12", choices=VARIANTS)
    p.add_argument("--checkpoint", default=None,
                   help="Path/URL of the UNFUSED ImageNet checkpoint. "
                        "Default: the README url for --model. "
                        "Pass 'none' to train from scratch.")
    p.add_argument("--output", default="runs/fastvit_seg")
    p.add_argument("--iters", type=int, default=40000,
                   help="number of optimizer steps")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum-steps", type=int, default=4,
                   help="gradient accumulation; effective batch = "
                        "batch-size * accum-steps (paper: 32)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-iters", type=int, default=0)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--amp", action="store_true", help="mixed precision")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--eval-interval", type=int, default=4000)
    p.add_argument("--eval-samples", type=int, default=None,
                   help="evaluate on a subset of val during training "
                        "(full val is used at the end unless "
                        "--final-eval-samples is set)")
    p.add_argument("--final-eval-samples", type=int, default=None,
                   help="val images for the final evaluation "
                        "(default: all 2000; set small for debug runs)")
    p.add_argument("--train-samples", type=int, default=None,
                   help="use a subset of train (for quick debugging)")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=500,
                   help="save last.pth every N iters (for Colab disconnects)")
    p.add_argument("--resume", default=None, help="path to last.pth")
    p.add_argument("--cache-dir", default=None, help="HF datasets cache dir")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def poly_lr(base_lr, it, total, warmup, power=0.9, min_lr=0.0):
    if warmup and it < warmup:
        return base_lr * (it + 1) / warmup
    return (base_lr - min_lr) * (1 - it / total) ** power + min_lr


@torch.no_grad()
def evaluate(model, dataset, device, amp=False, max_samples=None):
    """Whole-image inference, batch size 1. Returns aAcc, mIoU, mAcc and
    per-class IoU, computed on the original-resolution labels."""
    model.eval()
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    hist = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64,
                       device=device)
    t0 = time.time()
    print(f"evaluating on {n} val images...", flush=True)
    for i in range(n):
        if i and i % 250 == 0:
            print(f"  eval {i}/{n} ({time.time() - t0:.0f}s)", flush=True)
        img, lbl = dataset[i]
        img = img.unsqueeze(0).to(device)
        lbl = lbl.to(device)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits = model(img)
        logits = F.interpolate(logits.float(), size=lbl.shape, mode="bilinear",
                               align_corners=False)
        pred = logits.argmax(1)[0]
        valid = lbl != IGNORE_INDEX
        hist += torch.bincount(
            lbl[valid] * NUM_CLASSES + pred[valid],
            minlength=NUM_CLASSES ** 2,
        ).view(NUM_CLASSES, NUM_CLASSES)
    hist = hist.double()
    tp = hist.diag()
    iou = tp / (hist.sum(0) + hist.sum(1) - tp)
    acc = tp / hist.sum(1)
    model.train()
    # classes absent from GT and predictions give NaN and are ignored (mmseg)
    return {
        "aAcc": (tp.sum() / hist.sum()).item() * 100,
        "mIoU": torch.nanmean(iou).item() * 100,
        "mAcc": torch.nanmean(acc).item() * 100,
        "per_class_iou": {c: (v * 100 if not math.isnan(v) else None)
                          for c, v in zip(CLASSES, iou.tolist())},
        "num_images": n,
    }


def param_groups(model, weight_decay):
    """No weight decay on norms, biases, layer-scale and positional params."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "layer_scale" in name or name.endswith(".gamma"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt = args.checkpoint or CHECKPOINT_URLS[args.model]
    if ckpt.lower() == "none":
        ckpt = None
    model = FastViTSemanticFPN(args.model, NUM_CLASSES, pretrained=ckpt,
                               drop_path_rate=args.drop_path).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{args.model} + Semantic FPN: {n_params:.2f}M params")

    train_set = ADE20KSeg(load_hf_split("train", args.cache_dir), train=True,
                          crop_size=args.crop_size,
                          max_samples=args.train_samples)
    val_set = ADE20KSeg(load_hf_split("val", args.cache_dir), train=False)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True,
                        drop_last=True,
                        persistent_workers=args.workers > 0)
    print(f"train images: {len(train_set)}, val images: {len(val_set)}")

    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and
                                  device.type == "cuda")
    start_it, best = 0, 0.0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_it, best = state["iter"], state.get("best_miou", 0.0)
        print(f"resumed from {args.resume} at iter {start_it}")

    log_path = os.path.join(args.output, "log.csv")
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if start_it == 0:
        log.writerow(["iter", "loss", "lr", "mIoU", "aAcc", "mAcc", "time_s"])

    def save(name, it):
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(), "iter": it,
                    "best_miou": best, "variant": args.model,
                    "num_classes": NUM_CLASSES},
                   os.path.join(args.output, name))

    data = infinite(loader)
    model.train()
    t0, running = time.time(), 0.0
    for it in range(start_it, args.iters):
        lr = poly_lr(args.lr, it, args.iters, args.warmup_iters)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum_steps):
            img, lbl = next(data)
            img = img.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                logits = model(img)
                loss = F.cross_entropy(logits.float(), lbl,
                                       ignore_index=IGNORE_INDEX)
            scaler.scale(loss / args.accum_steps).backward()
            running += loss.item() / args.accum_steps
        scaler.step(opt)
        scaler.update()

        step = it + 1
        if step % args.log_interval == 0:
            avg = running / args.log_interval
            el = time.time() - t0
            eta = el / (step - start_it) * (args.iters - step) / 3600
            print(f"iter {step}/{args.iters} loss {avg:.4f} lr {lr:.2e} "
                  f"elapsed {el / 60:.1f}m eta {eta:.2f}h", flush=True)
            log.writerow([step, f"{avg:.4f}", f"{lr:.3e}", "", "", "",
                          f"{el:.0f}"])
            log_f.flush()
            running = 0.0

        if step % args.save_interval == 0:
            save("last.pth", step)

        if step % args.eval_interval == 0 or step == args.iters:
            final = step == args.iters
            m = evaluate(model, val_set, device, args.amp,
                         args.final_eval_samples if final
                         else args.eval_samples)
            print(f"[eval @ {step}] mIoU {m['mIoU']:.2f} aAcc {m['aAcc']:.2f} "
                  f"mAcc {m['mAcc']:.2f} ({m['num_images']} images)",
                  flush=True)
            log.writerow([step, "", "", f"{m['mIoU']:.2f}", f"{m['aAcc']:.2f}",
                          f"{m['mAcc']:.2f}", f"{time.time() - t0:.0f}"])
            log_f.flush()
            if m["mIoU"] > best:
                best = m["mIoU"]
                save("best.pth", step)
            if final:
                with open(os.path.join(args.output, "final_metrics.json"),
                          "w") as f:
                    json.dump(m, f, indent=2)
            save("last.pth", step)

    log_f.close()
    print(f"done. best mIoU {best:.2f}. checkpoints in {args.output}")


if __name__ == "__main__":
    main()
