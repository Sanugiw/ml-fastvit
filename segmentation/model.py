"""FastViT backbone + Semantic FPN decoder for semantic segmentation.

The FastViT paper (Sec. 4.4, Table 9) evaluates segmentation on ADE20K with a Semantic
FPN decoder (Kirillov et al., 2019), following the PoolFormer recipe. This is a
plain-PyTorch version of that setup so it runs without mmcv / mmsegmentation.
"""
import os
import sys
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow `python segmentation/train_seg.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import fastvit as fastvit_lib  # noqa: E402
from models.modules.mobileone import reparameterize_model  # noqa: E402

VARIANTS = [
    "fastvit_t8",
    "fastvit_t12",
    "fastvit_s12",
    "fastvit_sa12",
    "fastvit_sa24",
    "fastvit_sa36",
    "fastvit_ma36",
]

# Unfused ImageNet checkpoints from the README (the ones meant for fine-tuning).
CHECKPOINT_URLS = {
    v: f"https://docs-assets.developer.apple.com/ml-research/models/fastvit/"
    f"image_classification_models/{v}.pth.tar"
    for v in VARIANTS
}
DISTILLED_CHECKPOINT_URLS = {
    v: f"https://docs-assets.developer.apple.com/ml-research/models/fastvit/"
    f"image_classification_distilled_models/{v}.pth.tar"
    for v in VARIANTS
}


def build_backbone(variant: str, drop_path_rate: float = 0.0) -> nn.Module:
    """FastViT that returns the 4 stage feature maps (strides 4, 8, 16, 32)."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant}. Choose from {VARIANTS}")
    # Call the constructor directly instead of timm.create_model: recent timm
    # versions ship their own (different) `fastvit_*` models under the same
    # names, so going through the registry is ambiguous.
    return getattr(fastvit_lib, variant)(
        fork_feat=True, drop_path_rate=drop_path_rate
    )


def _read_state_dict(path_or_url: str) -> Dict[str, torch.Tensor]:
    if path_or_url.startswith("http"):
        ckpt = torch.hub.load_state_dict_from_url(
            path_or_url, map_location="cpu", check_hash=False
        )
    else:
        ckpt = torch.load(path_or_url, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "state_dict_ema"):
        if isinstance(ckpt, dict) and key in ckpt:
            ckpt = ckpt[key]
            break
    return {k.replace("module.", "", 1): v for k, v in ckpt.items()}


def load_imagenet_weights(backbone: nn.Module, path_or_url: str) -> None:
    """Load an *unfused* ImageNet classification checkpoint into the backbone.

    The classifier head (conv_exp, head) is dropped; the per-stage output norms
    (norm*) are new and stay randomly initialised.
    """
    state_dict = _read_state_dict(path_or_url)
    if any("reparam_conv" in k for k in state_dict):
        raise ValueError(
            f"{path_or_url} is a re-parameterized (fused) checkpoint. Use the "
            "*unfused* checkpoint (file name without '_reparam') for training."
        )
    model_sd = backbone.state_dict()
    filtered = {
        k: v for k, v in state_dict.items()
        if k in model_sd and v.shape == model_sd[k].shape
    }
    missing, _ = backbone.load_state_dict(filtered, strict=False)
    skipped = sorted(set(state_dict) - set(filtered))
    unexpected_missing = [k for k in missing if not k.startswith("norm")]
    print(
        f"[backbone] loaded {len(filtered)}/{len(model_sd)} tensors from "
        f"{path_or_url}"
    )
    print(f"[backbone] skipped from checkpoint (classifier head): "
          f"{len(skipped)} tensors, e.g. {skipped[:3]}")
    if unexpected_missing:
        raise RuntimeError(
            "Backbone weights missing from checkpoint (wrong variant?): "
            f"{unexpected_missing[:10]}"
        )


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class FPNNeck(nn.Module):
    """Feature Pyramid Network (same as mmseg `FPN`, no extra levels)."""

    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        )
        self.fpn = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels
        )

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        for i in range(len(lat) - 1, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(
                lat[i], size=lat[i - 1].shape[-2:], mode="nearest"
            )
        return [conv(x) for conv, x in zip(self.fpn, lat)]


class SemanticFPNHead(nn.Module):
    """Semantic FPN head (mmseg `FPNHead`): upsample every level to stride 4,
    sum, then classify."""

    def __init__(self, in_channels: int = 256, channels: int = 128,
                 num_classes: int = 150, num_levels: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.scale_heads = nn.ModuleList()
        for i in range(num_levels):
            layers = []
            for j in range(max(1, i)):  # level i is 2**i times smaller
                layers.append(ConvBNReLU(in_channels if j == 0 else channels,
                                         channels))
                if i > 0:
                    layers.append(nn.Upsample(scale_factor=2, mode="bilinear",
                                              align_corners=False))
            self.scale_heads.append(nn.Sequential(*layers))
        self.dropout = nn.Dropout2d(dropout)
        self.cls = nn.Conv2d(channels, num_classes, 1)
        nn.init.normal_(self.cls.weight, 0, 0.01)
        nn.init.zeros_(self.cls.bias)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        out = self.scale_heads[0](feats[0])
        for i in range(1, len(feats)):
            out = out + F.interpolate(
                self.scale_heads[i](feats[i]), size=out.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        return self.cls(self.dropout(out))


class FastViTSemanticFPN(nn.Module):
    def __init__(self, variant: str = "fastvit_sa12", num_classes: int = 150,
                 pretrained: str = None, drop_path_rate: float = 0.0):
        super().__init__()
        self.variant = variant
        self.backbone = build_backbone(variant, drop_path_rate)
        if pretrained:
            load_imagenet_weights(self.backbone, pretrained)
        with torch.no_grad():
            chans = [f.shape[1] for f in
                     self.backbone.eval()(torch.zeros(1, 3, 64, 64))]
        self.backbone.train()
        self.neck = FPNNeck(chans, 256)
        self.decode_head = SemanticFPNHead(256, 128, num_classes, len(chans))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits at the input resolution: (B, num_classes, H, W)."""
        logits = self.decode_head(self.neck(self.backbone(x)))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear",
                             align_corners=False)


def reparameterize(model: nn.Module) -> nn.Module:
    """Fuse FastViT's train-time branches (structural reparameterization).
    Returns a new model; call on a model in eval mode."""
    return reparameterize_model(model)
