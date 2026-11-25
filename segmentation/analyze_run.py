"""Analyze a finished segmentation run: data leakage, errors and curves.

Writes to ``<run>/analysis/``:
    * ``summary.json``: train/val image overlap, mIoU per val quarter,
      top confusions, per-class IoU and pixel frequency.
    * ``curves.png``: training loss and validation mIoU.
    * ``per_class.png``: IoU vs. class pixel frequency, best and worst classes.

Example (from the repo root):
    python segmentation/analyze_run.py --run segmentation/runs/fastvit_sa12
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmentation.ade20k import (  # noqa: E402
    CLASSES, IGNORE_INDEX, NUM_CLASSES, ADE20KSeg, load_hf_split,
)
from segmentation.model import FastViTSemanticFPN  # noqa: E402


def image_hashes(split):
    """Return the MD5 of every encoded image in a Hugging Face split.

    Args:
        split: Hugging Face ``Dataset`` with an ``image`` column.

    Returns:
        List of hex digests, in dataset order.
    """
    from datasets import Image as HFImage
    raw = split.cast_column("image", HFImage(decode=False))
    return [hashlib.md5(r["bytes"]).hexdigest()
            for r in raw.select_columns(["image"])["image"]]


def miou(hist):
    """Compute mIoU (%) from a confusion matrix, ignoring absent classes."""
    tp = np.diag(hist)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = tp / (hist.sum(0) + hist.sum(1) - tp)
    return float(np.nanmean(iou) * 100), iou


@torch.no_grad()
def confusion(model, dataset, device, parts=4):
    """Run whole-image inference and accumulate one confusion matrix per part.

    Args:
        model: Segmentation model in eval mode.
        dataset: Validation :class:`ADE20KSeg`.
        device: Inference device.
        parts: Number of equal, consecutive parts to split the dataset into.

    Returns:
        Array of shape ``(parts, NUM_CLASSES, NUM_CLASSES)`` (rows = ground
        truth, columns = prediction).
    """
    n = len(dataset)
    hists = np.zeros((parts, NUM_CLASSES, NUM_CLASSES), np.int64)
    for i in range(n):
        img, lbl = dataset[i]
        logits = model(img.unsqueeze(0).to(device))
        pred = F.interpolate(logits.float(), size=lbl.shape, mode="bilinear",
                             align_corners=False).argmax(1)[0].cpu()
        valid = lbl != IGNORE_INDEX
        h = torch.bincount(lbl[valid] * NUM_CLASSES + pred[valid],
                           minlength=NUM_CLASSES ** 2)
        hists[i * parts // n] += h.view(NUM_CLASSES, NUM_CLASSES).numpy()
        if i and i % 500 == 0:
            print(f"  {i}/{n}", flush=True)
    return hists


def plot_curves(run, out):
    """Plot training loss and validation mIoU from ``log.csv``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    loss, ev = {}, {}
    with open(os.path.join(run, "log.csv")) as f:
        for r in csv.DictReader(f):
            if r["loss"]:
                loss[int(r["iter"])] = float(r["loss"])  # resume: keep last
            if r["mIoU"]:
                ev[int(r["iter"])] = float(r["mIoU"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    it = sorted(loss)
    ax[0].plot(it, [loss[i] for i in it], lw=1)
    ax[0].set(title="Training loss", xlabel="iteration", ylabel="cross-entropy")
    it = sorted(ev)
    ax[1].plot(it[:-1], [ev[i] for i in it[:-1]], "o-", ms=3,
               label="500 val images")
    ax[1].plot(it[-1:], [ev[it[-1]]], "s", ms=6, label="all 2,000 (final)")
    ax[1].axhline(38.0, ls="--", c="gray", lw=1, label="paper (38.0)")
    ax[1].set(title="Validation mIoU (%)", xlabel="iteration")
    ax[1].legend(frameon=False)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "curves.png"), dpi=150)


def plot_per_class(iou, freq, out, k=15):
    """Plot IoU against pixel frequency, plus the best and worst classes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = ~np.isnan(iou)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].scatter(freq[ok] * 100, iou[ok] * 100, s=10)
    ax[0].set_xscale("log")
    ax[0].set(title="Per-class IoU vs. share of val pixels",
              xlabel="share of labeled val pixels (%)", ylabel="IoU (%)")
    ax[0].grid(alpha=0.3)
    order = np.argsort(np.where(ok, iou, np.inf))
    worst = order[:k]
    best = np.argsort(np.where(ok, -iou, np.inf))[:k][::-1]
    idx = np.concatenate([worst, best])
    ax[1].barh(range(len(idx)), iou[idx] * 100,
               color=["#c0504d"] * k + ["#4f81bd"] * k)
    ax[1].set_yticks(range(len(idx)), [CLASSES[i] for i in idx], fontsize=7)
    ax[1].set(title=f"{k} worst and {k} best classes", xlabel="IoU (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "per_class.png"), dpi=150)


def main():
    """Run all analyses for one training run."""
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run directory with last.pth")
    p.add_argument("--checkpoint", default="last.pth")
    args = p.parse_args()
    out = os.path.join(args.run, "analysis")
    os.makedirs(out, exist_ok=True)
    summary = {}

    train_split, val_split = load_hf_split("train"), load_hf_split("val")
    print("hashing images...", flush=True)
    th, vh = image_hashes(train_split), image_hashes(val_split)
    overlap = sorted(set(th) & set(vh))
    summary["leakage"] = {
        "train_images": len(th), "val_images": len(vh),
        "unique_train": len(set(th)), "unique_val": len(set(vh)),
        "val_images_also_in_train": sum(h in set(th) for h in vh),
        "identical_overlap_hashes": len(overlap),
    }
    print(summary["leakage"], flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(os.path.join(args.run, args.checkpoint),
                       map_location="cpu", weights_only=False)
    model = FastViTSemanticFPN(state["variant"], NUM_CLASSES)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    print("evaluating...", flush=True)
    hists = confusion(model, ADE20KSeg(val_split, train=False), device)
    hist = hists.sum(0)
    np.save(os.path.join(out, "confusion.npy"), hist)

    total, iou = miou(hist)
    summary["mIoU"] = total
    summary["mIoU_per_quarter"] = [miou(h)[0] for h in hists]
    freq = hist.sum(1) / hist.sum()
    summary["classes"] = {
        c: {"iou": None if np.isnan(v) else float(v * 100),
            "pixel_share": float(f * 100)}
        for c, v, f in zip(CLASSES, iou, freq)}
    # Most frequent errors as a share of all labeled pixels.
    off = hist.astype(np.float64).copy()
    np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(-off.ravel())[:15],
                                     off.shape))[0]
    summary["top_confusions"] = [
        {"true": CLASSES[t], "predicted": CLASSES[q],
         "share_of_true_class": float(off[t, q] / hist[t].sum() * 100),
         "share_of_all_pixels": float(off[t, q] / hist.sum() * 100)}
        for t, q in top]
    # IoU by class frequency tercile.
    order = np.argsort(-freq)
    terc = np.array_split(order, 3)
    summary["mIoU_by_frequency_tercile"] = {
        name: float(np.nanmean(iou[t]) * 100)
        for name, t in zip(["frequent", "medium", "rare"], terc)}

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_curves(args.run, out)
    plot_per_class(iou, freq, out)
    print(json.dumps({k: v for k, v in summary.items() if k != "classes"},
                     indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
