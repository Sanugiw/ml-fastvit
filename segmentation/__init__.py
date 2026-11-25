"""Semantic segmentation (ADE20K) with FastViT backbones."""
import os

# Keep the ADE20K download inside segmentation/cache/hf. Set before anything
# imports huggingface_hub (timm does), since it reads HF_HOME at import time.
# An HF_HOME already set in the environment takes precedence.
os.environ.setdefault(
    "HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "hf"))
