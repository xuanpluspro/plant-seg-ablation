"""Validate all VOC lists, 512px shapes and class-index or binary masks before training."""
import argparse
from utils.dataloader_medical import PlantVOCDataset, validate_splits, voc_root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default="VOCdevkit")
    p.add_argument("--num-classes", type=int, default=1, help="Foreground classes excluding background")
    p.add_argument("--mask-encoding", choices=("auto", "index", "binary255"), default="auto")
    args = p.parse_args()
    if not 1 <= args.num_classes <= 255:
        p.error("--num-classes must be 1..255")
    root = voc_root(args.data_path)
    splits = validate_splits(root)
    for split in splits:
        dataset = PlantVOCDataset(root, split, num_classes=args.num_classes+1, mask_encoding=args.mask_encoding)
        foreground = total = 0
        for _, target, _ in dataset:
            foreground += int((target > 0).sum())
            total += target.numel()
        print(f"{split}: {len(dataset)} pairs, plant pixel fraction={foreground/total:.6f}")
    print("PASS: disjoint IDs, matched images/masks, 512x512, valid labels for the requested encoding and class count.")


if __name__ == "__main__":
    main()
