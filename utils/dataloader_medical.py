"""VOC RGB datasets with binary/index masks and nested or flat split lists."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .augmentation import augment_train

IMAGE_SIZE = 512
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def voc_root(path):
    root = Path(path)
    if (root / "VOC2012").is_dir():
        root = root / "VOC2012"
    if not (root / "JPEGImages").is_dir() or not (root / "SegmentationClass").is_dir():
        raise FileNotFoundError(f"Expected VOC2012/JPEGImages and SegmentationClass under {path}")
    return root


def split_paths(root, split):
    """Supported manifest locations, in conventional VOC priority order."""
    if split not in ("train", "val", "test"):
        raise ValueError("Use independent train/val/test lists, not trainval")
    root = Path(root)
    return (root / "ImageSets" / "Segmentation" / f"{split}.txt",
            root / "ImageSets" / f"{split}.txt")


def read_ids(root, split):
    candidates = split_paths(root, split)
    paths = [p for p in candidates if p.is_file()]
    if not paths:
        raise FileNotFoundError("Split list not found; checked: " + ", ".join(map(str, candidates)))
    selected = None
    for path in paths:
        # Each line is one full stem: preserve internal spaces and parentheses.
        ids = [x.strip() for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Empty or duplicate split IDs: {path}")
        if any("/" in x or "\\" in x or "\x00" in x or x in (".", "..") for x in ids):
            raise ValueError(f"Expected a filename stem without path separators or NUL: {path}")
        if selected is not None and ids != selected:
            raise ValueError(f"Conflicting split lists: {paths[0]} and {path}; "
                             "keep one copy or make IDs and order identical")
        selected = ids
    return selected


def validate_splits(root):
    """Require train/val; check test disjointness when provided, without reading test pixels."""
    result = {split: read_ids(root, split) for split in ("train", "val")}
    if any(p.is_file() for p in split_paths(root, "test")):
        result["test"] = read_ids(root, "test")
    keys = list(result)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            overlap = set(result[a]) & set(result[b])
            if overlap:
                raise ValueError(f"Split leakage {a}/{b}: {sorted(overlap)[:5]}")
    return result


def index_images(root):
    index = {}
    for path in sorted((root / "JPEGImages").iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in index:
                raise ValueError(f"Duplicate image stem: {path.stem}")
            index[path.stem] = path
    return index


def image_tensor(image):
    x = torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32)
                         .transpose(2, 0, 1).copy()) / 255.
    return (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]


def decode_mask(path, num_classes=2, mask_encoding="binary255"):
    if not 2 <= num_classes <= 256:
        raise ValueError("Require 2..256 total classes")
    if mask_encoding not in ("auto", "binary255", "index"):
        raise ValueError("Use auto, binary255 or index mask encoding")
    with Image.open(path) as image:
        mask = np.array(image)  # preserve indices of palette PNG, not displayed colors
    if mask.ndim == 3 and mask.shape[2] in (3, 4):
        if not (np.array_equal(mask[..., 0], mask[..., 1])
                and np.array_equal(mask[..., 0], mask[..., 2])):
            raise ValueError(f"Colored mask requires explicit mapping: {path}")
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask: {path}")
    values = set(np.unique(mask))
    if mask_encoding == "binary255":
        if num_classes != 2 or not values.issubset({0, 255}):
            raise ValueError(f"binary255 requires two classes and only 0/255: {path}")
        return (mask == 255).astype(np.uint8)
    if mask_encoding == "auto" and num_classes == 2 and 255 in values:
        if not values.issubset({0, 255}):
            raise ValueError(f"Mixed 0/1/255 labels are ambiguous: {path}")
        return (mask == 255).astype(np.uint8)
    if not values.issubset(set(range(num_classes))):
        raise ValueError(f"Expected class IDs 0..{num_classes-1}; got {sorted(values)}: {path}")
    return mask.astype(np.uint8)



class PlantVOCDataset(Dataset):
    def __init__(self, data_path, split="train", augmentation=False, num_classes=2, mask_encoding="binary255"):
        self.root = voc_root(data_path)
        self.ids = read_ids(self.root, split)
        self.augmentation = augmentation
        self.num_classes = num_classes
        self.mask_encoding = mask_encoding
        images = index_images(self.root)
        self.pairs = []
        for name in self.ids:
            mask = self.root / "SegmentationClass" / f"{name}.png"
            if name not in images or not mask.is_file():
                raise FileNotFoundError(f"Missing image or mask for {name}")
            self.pairs.append((images[name], mask))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as file:
            image = file.convert("RGB")
        mask = decode_mask(mask_path, self.num_classes, self.mask_encoding)
        if image.size != (IMAGE_SIZE, IMAGE_SIZE) or mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Expected 512x512 image AND mask: {image_path}, {mask_path}")
        label = Image.fromarray(mask)
        if self.augmentation:
            image, label = augment_train(image, label)
        return image_tensor(image), torch.from_numpy(np.array(label, dtype=np.int64)), self.ids[index]

