"""Train a BiMSeg ablation on 512px binary or multiclass VOC masks."""
import argparse
import json
import csv
from time import perf_counter
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from model import OSSSeg
from model.losses import SegmentationLoss
from ablation_config import canonical_model_config, describe_ablation
from utils.checkpoint_storage import inference_checkpoint
from utils.dataloader_medical import PlantVOCDataset, voc_root, validate_splits
from utils.train_and_eval import train_one_epoch, evaluate
from utils.augmentation import AUGMENTATION_CONFIG
from utils.checkpoint_selection import SELECTION_METRIC, validation_score, resume_best_score
from utils.runtime import (FORMAT, seed_everything, seed_worker, resolve_device,
                           save_checkpoint, read_checkpoint, rng_state, restore_rng)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default="VOCdevkit")
    p.add_argument("--output", default="run/train/ossseg")
    init = p.add_mutually_exclusive_group()
    init.add_argument("--backbone-weights", help="Local ResNet-50 weights or full U-Net checkpoint with resnet.* backbone")
    init.add_argument("--from-scratch", action="store_true", help="Explicitly train without pretraining")
    init.add_argument("--resume", help="Resume this project's last.pth with optimizer/RNG state")
    p.add_argument("--num-classes", type=int, default=1, help="Foreground class count excluding background: 1 for binary, 4 for IDs 0..4")
    p.add_argument("--mask-encoding", choices=["auto", "binary255", "index"], default="auto")
    p.add_argument("--dynamic-fusion", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--msca", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drcf", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dcam", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--freeze-backbone-bn", action=argparse.BooleanOptionalAction, default=False,
                   help="Freeze backbone BN running statistics (default: update during training)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dice-weight", type=float, default=1.)
    p.add_argument("--aux-weight", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--benchmark-warmup", type=int, default=50)
    p.add_argument("--benchmark-iterations", type=int, default=200)
    return p.parse_args(argv)


def resolve_initialization(args):
    if not args.backbone_weights and not args.from_scratch and not args.resume:
        args.backbone_weights = str(Path(__file__).resolve().parent / "pretrained" / "resnet50.pth")
    if args.backbone_weights and not Path(args.backbone_weights).is_file():
        raise FileNotFoundError(
            f"Pretrained backbone missing: {args.backbone_weights}. "
            "Run python download_pretrained.py, or supply --backbone-weights PATH. "
            "Random initialization requires explicit --from-scratch.")


def train(args):
    if not getattr(args, "dynamic_fusion", True):
        raise ValueError("Use --no-dcam for the baseline replacement; --no-dynamic-fusion selects an unrelated legacy module")
    resolve_initialization(args)
    if not 1 <= args.num_classes <= 255 or args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("Require 1..255 foreground classes, positive epochs/batch-size, nonnegative workers")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer settings")
    if getattr(args, "benchmark_warmup", 50) < 0 or getattr(args, "benchmark_iterations", 200) < 1:
        raise ValueError("Invalid benchmark repetitions")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    root = voc_root(args.data_path)
    splits = validate_splits(root)
    data_config = dict(num_classes=args.num_classes + 1, mask_encoding=getattr(args, "mask_encoding", "auto"))
    model_config = dict(num_classes=args.num_classes + 1, width=args.width, freeze_backbone_bn=args.freeze_backbone_bn,
                        use_dynamic_fusion=True, use_msca=getattr(args, "msca", True),
                        use_drcf=getattr(args, "drcf", True), use_dcam=getattr(args, "dcam", True))
    ablation = describe_ablation(model_config)
    training_config = {key: getattr(args, key) for key in (
        "epochs", "batch_size", "workers", "lr", "weight_decay", "dice_weight",
        "aux_weight", "seed", "amp")}
    output = Path(args.output)
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new --output or --resume: {output}")
    if args.resume and Path(args.resume).resolve().parent != output.resolve():
        raise ValueError("--output must be the run directory containing the resume checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.workers,
                         pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
    train_loader = DataLoader(PlantVOCDataset(root, "train", augmentation=True, **data_config),
                              shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(PlantVOCDataset(root, "val", augmentation=False, **data_config),
                            shuffle=False, **loader_kwargs)
    model = OSSSeg(**model_config)
    initialization = args.backbone_weights or "random"
    if args.backbone_weights:
        report = model.load_backbone(args.backbone_weights)
        print(f"Loaded pretrained ResNet-50 backbone: {args.backbone_weights}; "
              f"namespace={report['namespace']}; loaded_tensors={report['loaded_tensors']}; "
              f"ignored_entries={report['ignored_entries']}", flush=True)
    if args.from_scratch:
        print("Explicit random initialization; no weights will be downloaded.", flush=True)
    model.to(device)
    criterion = SegmentationLoss(args.dice_weight, args.aux_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start, best, history = 0, -1., []
    if args.resume:
        checkpoint = read_checkpoint(args.resume)
        missing = [key for key in ("optimizer", "scheduler", "scaler", "rng", "history") if key not in checkpoint]
        if missing:
            raise ValueError("This checkpoint is inference-only; resume training from last.pth")
        saved_data = checkpoint.get("data_config", dict(num_classes=2, mask_encoding="binary255"))
        saved_config = canonical_model_config(checkpoint["model_config"])
        if (saved_data != data_config or saved_config != model_config
                or checkpoint["training_config"] != training_config
                or checkpoint["splits"] != splits):
            raise ValueError("Resume requires identical model/training settings and split IDs, including total epochs")
        if checkpoint.get("augmentation_config") != AUGMENTATION_CONFIG:
            raise ValueError("Resume requires the same augmentation policy; start a new --output for the new augmentation settings")
        best = resume_best_score(checkpoint)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"], generator)
        start = checkpoint["epoch"] + 1
        history = checkpoint["history"]
        initialization = checkpoint["initialization"]
    config = dict(model=model_config, training=training_config, initialization=initialization,
                  data_path=str(root.resolve()), data=data_config, selection_metric=SELECTION_METRIC,
                  augmentation_config=AUGMENTATION_CONFIG, ablation=ablation)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(f"Ablation: {ablation}", flush=True)
    print(f"device={device}; train={len(train_loader.dataset)} val={len(val_loader.dataset)}; params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    for epoch in range(start, args.epochs):
        begin = perf_counter()
        loss = train_one_epoch(model, optimizer, train_loader, device, criterion, scaler,
                               desc=f"Epoch {epoch+1}/{args.epochs} Train")
        metrics = evaluate(model, val_loader, device, criterion,
                           desc=f"Epoch {epoch+1}/{args.epochs} Val")
        selection_metric = SELECTION_METRIC
        score = validation_score(metrics)
        improved = score > best
        best = max(best, score)
        history.append(dict(epoch=epoch + 1, train_loss=loss, seconds=perf_counter()-begin,
                            lr=optimizer.param_groups[0]["lr"], **metrics))
        scheduler.step()
        checkpoint = dict(
            format=FORMAT, model_config=model_config, training_config=training_config,
            initialization=initialization, splits=splits, data_config=data_config, epoch=epoch,
            selection_metric=selection_metric, best_score=best,
            augmentation_config=AUGMENTATION_CONFIG, ablation=ablation,
            model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
            rng=rng_state(generator), history=history)
        if improved:
            save_checkpoint(inference_checkpoint(checkpoint), output / "best.pth")
        save_checkpoint(checkpoint, output / "last.pth")
        (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
        with (output / "history.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            fields = list(dict.fromkeys(k for h in history for k, v in h.items() if not isinstance(v, (list, dict))))
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch+1}/{args.epochs} | loss={loss:.4f} | val loss={metrics['Loss']:.4f} | val mIoU={metrics['Mean IoU']:.4f} "
              f"| Dice={metrics['Dice']:.4f} | best {selection_metric}={best:.4f} "
              f"| time={history[-1]['seconds']:.1f}s", flush=True)
    if history:
        from utils.results import export_final_results
        best_checkpoint = read_checkpoint(output / "best.pth")
        model.load_state_dict(best_checkpoint["model"], strict=True)
        export_final_results(model, best_checkpoint, output / "best.pth", root, output,
                             device, args.batch_size, args.workers,
                             getattr(args, "benchmark", True),
                             getattr(args, "benchmark_warmup", 50),
                             getattr(args, "benchmark_iterations", 200))
        from utils.plot_results import plot_training_curves
        plot_training_curves([h["train_loss"] for h in history],
                             [h["Loss"] for h in history], history, str(output))


if __name__ == "__main__":
    train(parse_args())

