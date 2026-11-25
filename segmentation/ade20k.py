"""ADE20K (SceneParsing, 150 classes) from the Hugging Face Hub.

Uses `shijli/ade20k` (config `segmentation`): parquet copy of the official
ADEChallengeData2016 release, 20,210 train / 2,000 val images. The mask uses
0 = "other/ignore" and 1..150 = classes, so we shift by -1 and map 0 -> 255
(the `reduce_zero_label` convention used by mmsegmentation and the paper).

The original `scene_parse_150` repo relies on a loading script, which newer
versions of `datasets` no longer execute, hence the parquet mirror.
"""
import math
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

HF_DATASET = "shijli/ade20k"
HF_CONFIG = "segmentation"
IGNORE_INDEX = 255
NUM_CLASSES = 150
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASSES = (
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair", "car", "water",
    "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp",
    "bathtub", "railing", "cushion", "base", "box", "column", "signboard",
    "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen island", "computer",
    "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel",
    "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth",
    "television receiver", "airplane", "dirt track", "apparel", "pole",
    "land", "bannister", "escalator", "ottoman", "bottle", "buffet", "poster",
    "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer",
    "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall",
    "tent", "bag", "minibike", "cradle", "oven", "ball", "food", "step",
    "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake",
    "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase",
    "traffic light", "tray", "ashcan", "fan", "pier", "crt screen", "plate",
    "monitor", "bulletin board", "shower", "radiator", "glass", "clock",
    "flag",
)


def load_hf_split(split: str, cache_dir: str = None):
    """Load one split of ADE20K from the Hugging Face Hub.

    Args:
        split: ``train``, ``val`` or ``validation``.
        cache_dir: Hugging Face datasets cache directory. ``None`` uses the
            default cache. Default: ``None``

    Returns:
        Hugging Face ``Dataset`` with ``image`` and ``mask`` columns.
    """
    from datasets import load_dataset
    split = {"val": "validation"}.get(split, split)
    return load_dataset(HF_DATASET, HF_CONFIG, split=split, cache_dir=cache_dir)


def _rescale_keep_ratio(w: int, h: int, scale=(2048, 512)):
    """Compute a keep-ratio resize, as mmseg's ``Resize(keep_ratio=True)``.

    The image is scaled so that its long side is at most ``max(scale)`` and
    its short side at most ``min(scale)``.

    Args:
        w: Image width.
        h: Image height.
        scale: Target ``(long, short)`` side limits. Default: ``(2048, 512)``

    Returns:
        New ``(width, height)``.
    """
    long_s, short_s = max(scale), min(scale)
    factor = min(long_s / max(w, h), short_s / min(w, h))
    return int(w * factor + 0.5), int(h * factor + 0.5)


def _convert(x: np.ndarray, alpha: float = 1.0, beta: float = 0.0):
    """Scale and shift a ``uint8`` array, clipping to ``[0, 255]``."""
    return np.clip(x.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def photometric_distortion(img: np.ndarray, brightness_delta: int = 32,
                           contrast_range=(0.5, 1.5),
                           saturation_range=(0.5, 1.5),
                           hue_delta: int = 18) -> np.ndarray:
    """Port of mmseg's ``PhotoMetricDistortion`` with its default arguments.

    Each step is applied with probability 0.5, in mmseg's order: brightness,
    contrast (either here or last), saturation, hue, contrast. Hue is shifted
    by ``randint(-hue_delta, hue_delta)`` OpenCV units (2 degrees each),
    converted to PIL's 0-255 hue scale.

    Args:
        img: ``uint8`` RGB image of shape ``(H, W, 3)``.
        brightness_delta: Maximum brightness shift. Default: ``32``
        contrast_range: Range of the contrast factor. Default: ``(0.5, 1.5)``
        saturation_range: Range of the saturation factor.
            Default: ``(0.5, 1.5)``
        hue_delta: Maximum hue shift in OpenCV units. Default: ``18``

    Returns:
        Distorted ``uint8`` RGB image.
    """
    def contrast(x):
        if random.randint(0, 1):
            return _convert(x, alpha=random.uniform(*contrast_range))
        return x

    if random.randint(0, 1):
        img = _convert(img, beta=random.uniform(-brightness_delta,
                                                brightness_delta))
    contrast_first = random.randint(0, 1)
    if contrast_first:
        img = contrast(img)
    hsv = np.array(Image.fromarray(img).convert("HSV"))
    if random.randint(0, 1):
        hsv[..., 1] = _convert(hsv[..., 1],
                               alpha=random.uniform(*saturation_range))
    if random.randint(0, 1):
        shift = random.randint(-hue_delta, hue_delta - 1)  # np.randint range
        shift = round(shift * 2 * 256 / 360)  # OpenCV units -> PIL units
        hsv[..., 0] = (hsv[..., 0].astype(np.int64) + shift) % 256
    img = np.array(Image.fromarray(hsv, "HSV").convert("RGB"))
    if not contrast_first:
        img = contrast(img)
    return img


class ADE20KSeg(Dataset):
    """ADE20K semantic segmentation dataset with mmseg-style transforms.

    Training samples: random rescale, random crop, horizontal flip,
    photometric distortion, normalization and padding to ``crop_size``.
    Validation samples: keep-ratio resize (sides rounded to multiples of
    ``size_divisor``) and normalization of the image only;
    the label is returned at its original resolution.
    """

    def __init__(self, hf_split, train: bool, crop_size: int = 512,
                 scale=(2048, 512), ratio_range=(0.5, 2.0),
                 max_samples: int = None, photometric: str = "mmseg",
                 size_divisor: int = 32) -> None:
        """Build ADE20K dataset.

        Args:
            hf_split: Hugging Face split from :func:`load_hf_split`, or any
                indexable of dicts with ``image`` and ``mask`` PIL images.
            train: Apply training augmentations if ``True``.
            crop_size: Side of the square training crop. Default: ``512``
            scale: Base ``(long, short)`` resize limits. Default: ``(2048, 512)``
            ratio_range: Range of the random rescale factor applied to
                ``scale`` during training. Default: ``(0.5, 2.0)``
            max_samples: Use only the first ``max_samples`` images. ``None``
                uses all of them. Default: ``None``
            photometric: Color augmentation for training. ``mmseg`` is an
                exact port of mmseg's ``PhotoMetricDistortion`` (as in the
                paper); ``jitter`` is the earlier torchvision ``ColorJitter``
                approximation. Default: ``mmseg``
            size_divisor: Validation images are resized so both sides are
                multiples of this, as PoolFormer's ``AlignResize``. ``1``
                disables it. Default: ``32``
        """
        if photometric not in ("mmseg", "jitter"):
            raise ValueError(f"Unknown photometric mode {photometric}")
        self.ds = hf_split
        self.n = len(hf_split) if max_samples is None else min(
            max_samples, len(hf_split))
        self.train = train
        self.crop = crop_size
        self.scale = scale
        self.ratio_range = ratio_range
        self.photometric = photometric
        self.size_divisor = size_divisor
        self.jitter = ColorJitter(brightness=32 / 255, contrast=0.5,
                                  saturation=0.5, hue=18 / 360)

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.n

    @staticmethod
    def _to_label(mask: Image.Image) -> np.ndarray:
        """Convert an ADE20K mask to training labels (``reduce_zero_label``).

        Args:
            mask: Mask where ``0`` is "other" and ``1..150`` are classes.

        Returns:
            ``uint8`` label map with classes ``0..149`` and ``IGNORE_INDEX``
            for "other".
        """
        m = np.array(mask, dtype=np.int64)
        if m.ndim == 3:
            m = m[..., 0]
        m = m - 1
        m[m < 0] = IGNORE_INDEX
        return m.astype(np.uint8)

    def _random_crop(self, img, lbl, cat_max_ratio=0.75, tries=10):
        """Randomly crop image and label, as mmseg's ``RandomCrop``.

        Crops dominated by a single class are re-drawn, up to ``tries`` times.

        Args:
            img: Image array of shape ``(H, W, 3)``.
            lbl: Label array of shape ``(H, W)``.
            cat_max_ratio: Maximum fraction of the crop's labeled pixels a
                single class may occupy. Default: ``0.75``
            tries: Maximum number of crops to draw. Default: ``10``

        Returns:
            Cropped ``(img, lbl)``, at most ``crop_size`` on each side.
        """
        h, w = lbl.shape
        ch, cw = min(self.crop, h), min(self.crop, w)
        for _ in range(tries):
            y = random.randint(0, h - ch)
            x = random.randint(0, w - cw)
            crop_lbl = lbl[y:y + ch, x:x + cw]
            cls, cnt = np.unique(crop_lbl, return_counts=True)
            cnt = cnt[cls != IGNORE_INDEX]
            if len(cnt) > 1 and cnt.max() / cnt.sum() < cat_max_ratio:
                break
        return img[y:y + ch, x:x + cw], crop_lbl

    def __getitem__(self, idx):
        """Load and transform one sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of image tensor ``(3, H, W)`` (float, normalized) and label
            tensor ``(H, W)`` (int64). For training both are
            ``crop_size x crop_size``; for validation the label keeps the
            original resolution.
        """
        row = self.ds[idx]
        img = row["image"].convert("RGB")
        lbl = self._to_label(row["mask"])
        orig_label = lbl

        if self.train:
            r = random.uniform(*self.ratio_range)
            scale = (int(self.scale[0] * r), int(self.scale[1] * r))
        else:
            scale = self.scale
        nw, nh = _rescale_keep_ratio(*img.size, scale)
        if not self.train and self.size_divisor > 1:
            d = self.size_divisor  # PoolFormer AlignResize
            nw, nh = math.ceil(nw / d) * d, math.ceil(nh / d) * d
        img = img.resize((nw, nh), Image.BILINEAR)

        if self.train:
            lbl = np.array(Image.fromarray(lbl).resize((nw, nh), Image.NEAREST))
            img = np.asarray(img)
            img, lbl = self._random_crop(img, lbl)
            if random.random() < 0.5:
                img, lbl = img[:, ::-1], lbl[:, ::-1]
            img = np.ascontiguousarray(img)
            if self.photometric == "mmseg":
                img = photometric_distortion(img)
            else:
                img = np.asarray(self.jitter(Image.fromarray(img)))
            img = (img.astype(np.float32) / 255.0 - MEAN) / STD
            # pad to crop size (image 0, label ignore)
            h, w = lbl.shape
            pad_img = np.zeros((self.crop, self.crop, 3), np.float32)
            pad_lbl = np.full((self.crop, self.crop), IGNORE_INDEX, np.uint8)
            pad_img[:h, :w] = img
            pad_lbl[:h, :w] = lbl
            return (torch.from_numpy(pad_img).permute(2, 0, 1),
                    torch.from_numpy(pad_lbl.astype(np.int64)))

        # Validation: keep-ratio resize of the image only; metrics are
        # computed against the original-resolution label (as in mmseg).
        img = (np.asarray(img).astype(np.float32) / 255.0 - MEAN) / STD
        return (torch.from_numpy(img).permute(2, 0, 1).contiguous(),
                torch.from_numpy(orig_label.astype(np.int64)))


# Fixed random palette for visualisations.
_rng = np.random.RandomState(0)
PALETTE = _rng.randint(0, 255, size=(256, 3)).astype(np.uint8)
PALETTE[IGNORE_INDEX] = 0


def colorize(label: np.ndarray) -> np.ndarray:
    """Map a label map to RGB colors for visualization.

    Args:
        label: Label map of shape ``(H, W)``.

    Returns:
        ``uint8`` RGB image of shape ``(H, W, 3)``.
    """
    return PALETTE[label.astype(np.int64)]


def denormalize(img: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization for visualization.

    Args:
        img: Normalized image tensor of shape ``(3, H, W)``.

    Returns:
        ``uint8`` RGB image of shape ``(H, W, 3)``.
    """
    x = img.permute(1, 2, 0).cpu().numpy() * STD + MEAN
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)
