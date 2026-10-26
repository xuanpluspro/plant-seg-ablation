"""Evaluate a checkpoint on another VOC dataset; export metrics only.

Unlike val.py, this script intentionally permits a different dataset/split.
--mask-encoding describes the TARGET labels, not the training labels.
"""
import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from utils.dataloader_medical import PlantVOCDataset
from utils.results import write_results
from utils.runtime import load_model, resolve_device
from utils.train_and_eval import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-path", required=True, help="VOC root, not test.txt")
    parser.add_argument("--mask-encoding", choices=("index", "binary255", "auto"),
                        required=True, help="Encoding of the TARGET dataset masks")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers nonnegative")

    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new or empty result directory: {output}")

    device = resolve_device(args.device)
    model, checkpoint = load_model(args.weights, device)
    num_classes = checkpoint["model_config"]["num_classes"]
    dataset = PlantVOCDataset(
        args.data_path, args.split, augmentation=False,
        num_classes=num_classes, mask_encoding=args.mask_encoding,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False, pin_memory=device.type == "cuda",
    )
    print(f"Checkpoint: {args.weights}", flush=True)
    print(f"Target dataset: {dataset.root}; split={args.split}; samples={len(dataset)}",
          flush=True)
    print(f"Target mask encoding: {args.mask_encoding}; total classes={num_classes}",
          flush=True)
    print("Independent target evaluation; no training or prediction-mask export.",
          flush=True)

    metrics = evaluate(model, loader, device, desc=f"Target {args.split}")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "checkpoint": str(Path(args.weights).resolve()),
        "best_epoch": checkpoint["epoch"] + 1,
        "arch": "BiMSeg",
        "encoder": "resnet50",
        "seed": checkpoint.get("training_config", {}).get("seed"),
        "evaluation_type": "independent target dataset; no fine-tuning",
        "evaluation_dataset": str(dataset.root.resolve()),
        "evaluation_split": args.split,
        "evaluation_samples": len(dataset),
        "evaluation_ids": dataset.ids,
        "parameters": parameters,
        "params_m": parameters / 1e6,
        "model_config": checkpoint["model_config"],
        "source_data_config": checkpoint.get("data_config", {}),
        "data_config": {"num_classes": num_classes,
                        "mask_encoding": args.mask_encoding},
        "metric_scale": "0-1",
        "metric_units": {"region_scores": "0-1", "Boundary F1": "0-1",
                         "HD95": "pixels", "ASSD": "pixels"},
        "aggregation": {"region": "dataset confusion matrix",
                        "boundary": "per-image foreground-class macro, then eligible-image mean"},
        "benchmark": None,
        args.split: metrics,
    }
    if args.split == "test":
        result["test_status"] = "evaluated"
    write_results(result, output)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Results saved: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

