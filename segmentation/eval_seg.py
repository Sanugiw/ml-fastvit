"""Evaluate a trained FastViT segmentation checkpoint on ADE20K val.

Also demonstrates the paper's key idea: structural reparameterization. With
--reparam the backbone's train-time branches are fused into single convs;
mIoU should be unchanged while inference gets faster.

    python segmentation/eval_seg.py --checkpoint runs/sa12/best.pth --reparam \
        --benchmark --vis 8
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmentation.ade20k import (  # noqa: E402
    NUM_CLASSES, ADE20KSeg, colorize, denormalize, load_hf_split,
)
from segmentation.model import FastViTSemanticFPN, reparameterize  # noqa: E402
from segmentation.train_seg import evaluate  # noqa: E402


@torch.no_grad()
def benchmark(model, device, size=512, warmup=10, runs=50):
    """Measure the average forward-pass latency on a random input.

    Args:
        model: Model to benchmark.
        device: Device to run on.
        size: Side of the square input image. Default: ``512``
        warmup: Number of untimed warmup runs. Default: ``10``
        runs: Number of timed runs. Default: ``50``

    Returns:
        Mean latency per forward pass, in milliseconds (batch size 1).
    """
    model.eval()
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / runs * 1000


def main():
    """Evaluate a trained segmentation checkpoint on ADE20K val.

    Optionally re-parameterizes the model, benchmarks latency and saves
    visualizations. Results are written to ``eval_results.json``.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="best.pth / last.pth")
    p.add_argument("--reparam", action="store_true",
                   help="also evaluate the reparameterized (fused) model")
    p.add_argument("--benchmark", action="store_true",
                   help="measure latency at 512x512, unfused vs fused")
    p.add_argument("--samples", type=int, default=None,
                   help="evaluate on first N val images only")
    p.add_argument("--vis", type=int, default=0,
                   help="save N image | ground truth | prediction panels")
    p.add_argument("--output", default=None)
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = FastViTSemanticFPN(state["variant"], state.get("num_classes",
                                                           NUM_CLASSES))
    model.load_state_dict(state["model"])
    model.to(device).eval()
    out_dir = args.output or os.path.dirname(os.path.abspath(args.checkpoint))

    val_set = ADE20KSeg(load_hf_split("val", args.cache_dir), train=False)
    results = {"variant": state["variant"]}
    m = evaluate(model, val_set, device, max_samples=args.samples)
    print(f"[unfused] mIoU {m['mIoU']:.2f} aAcc {m['aAcc']:.2f} "
          f"mAcc {m['mAcc']:.2f}")
    results["unfused"] = m

    fused = None
    if args.reparam or args.benchmark:
        fused = reparameterize(model).to(device).eval()
        n_u = sum(p.numel() for p in model.parameters()) / 1e6
        n_f = sum(p.numel() for p in fused.parameters()) / 1e6
        print(f"params: unfused {n_u:.2f}M -> fused {n_f:.2f}M")
        results["params_M"] = {"unfused": n_u, "fused": n_f}
    if args.reparam:
        mf = evaluate(fused, val_set, device, max_samples=args.samples)
        print(f"[fused]   mIoU {mf['mIoU']:.2f} aAcc {mf['aAcc']:.2f} "
              f"mAcc {mf['mAcc']:.2f}")
        results["fused"] = mf
    if args.benchmark:
        lu, lf = benchmark(model, device), benchmark(fused, device)
        print(f"latency @512x512 on {device}: unfused {lu:.1f} ms, "
              f"fused {lf:.1f} ms")
        results["latency_ms"] = {"device": str(device), "unfused": lu,
                                 "fused": lf}

    if args.vis:
        vis_dir = os.path.join(out_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        net = fused or model
        with torch.no_grad():
            for i in range(args.vis):
                img, lbl = val_set[i]
                logits = net(img.unsqueeze(0).to(device))
                pred = torch.nn.functional.interpolate(
                    logits, size=lbl.shape, mode="bilinear",
                    align_corners=False).argmax(1)[0].cpu().numpy()
                rgb = np.array(Image.fromarray(denormalize(img)).resize(
                    lbl.shape[::-1], Image.BILINEAR))
                panel = np.concatenate(
                    [rgb, colorize(lbl.numpy()), colorize(pred)], axis=1)
                Image.fromarray(panel).save(os.path.join(vis_dir, f"{i}.png"))
        print(f"saved {args.vis} visualisations to {vis_dir}")

    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
