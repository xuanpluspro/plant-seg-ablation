"""Run the eight module configurations sequentially with one common protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from ablation_config import VARIANTS, PROTOCOL
from utils.experiment_runner import run_logged
from utils.ablation_results import collect_results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--resume", action="store_true", help="Skip completed jobs and resume interrupted last.pth")
    args, extra = parser.parse_known_args(argv)
    reserved = ("--output", "--seed", "--msca", "--no-msca", "--drcf", "--no-drcf", "--dcam", "--no-dcam", "--dynamic-fusion", "--no-dynamic-fusion")
    if any(token.split("=", 1)[0] in reserved for token in extra):
        parser.error("Output, seeds and module flags are controlled by the sweep; use --output-root/--seeds/--variants")
    if len(set(args.variants)) != len(args.variants) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate variants/seeds are not allowed")
    # Validate the exact single-run arguments and the shared data before any job.
    from train_medical import parse_args, resolve_initialization
    from utils.dataloader_medical import voc_root, validate_splits
    from utils.augmentation import AUGMENTATION_CONFIG
    common = parse_args(extra)
    if args.resume:
        if not common.backbone_weights and not common.from_scratch:
            common.backbone_weights = str(Path(__file__).resolve().parent / "pretrained" / "resnet50.pth")
    else:
        resolve_initialization(common)
    root = Path(args.output_root).resolve()
    data_root = voc_root(common.data_path).resolve()
    splits = validate_splits(data_root)
    common_config = vars(common).copy()
    for key in ("output", "seed", "resume", "msca", "drcf", "dcam"):
        common_config.pop(key, None)
    common_config["data_path"] = str(data_root)
    implementation = hashlib.sha256()
    for path in sorted((Path(__file__).parent / "model").glob("*.py")):
        implementation.update(path.name.encode()); implementation.update(path.read_bytes())
    expected = dict(protocol=PROTOCOL, variants=args.variants, seeds=args.seeds,
                    common=common_config, splits=splits, augmentation=AUGMENTATION_CONFIG,
                    model_code_sha256=implementation.hexdigest())
    plan = root / "sweep_config.json"
    if args.resume:
        if not plan.is_file() or json.loads(plan.read_text()) != expected:
            raise ValueError("Resume requires the same sweep settings, module code, augmentation and split IDs")
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new --output-root or --resume")
    root.mkdir(parents=True, exist_ok=True)
    plan.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    logs = root / "logs"; logs.mkdir(exist_ok=True)
    statuses = []
    for variant in args.variants:
        for seed in args.seeds:
            name = f"{variant}_seed{seed}"
            output = root / name
            marker = output / "completed.json"
            if args.resume and marker.is_file() and (output / "metrics.json").is_file():
                statuses.append(dict(variant=variant, seed=seed, status=0, skipped_completed=True))
                continue
            flags = ["--" + ("" if enabled else "no-") + module
                     for module, enabled in zip(("msca", "drcf", "dcam"), VARIANTS[variant])]
            command = [sys.executable, "-u", str(Path(__file__).with_name("train_medical.py")),
                       *extra, "--output", str(output), "--seed", str(seed), *flags]
            if args.resume and (output / "last.pth").is_file():
                # Full initialization is restored from last.pth; do not reload pretraining.
                clean = []
                iterator = iter(command)
                for token in iterator:
                    if token == "--from-scratch":
                        continue
                    if token == "--backbone-weights":
                        next(iterator); continue
                    if token.startswith("--backbone-weights="):
                        continue
                    clean.append(token)
                command = [*clean, "--resume", str(output / "last.pth")]
            print(f"\nAblation {name}: {VARIANTS[variant]}", flush=True)
            status = run_logged(command, logs / f"{name}.log")
            if status == 0 and not (output / "metrics.json").is_file():
                status = 1
            if status == 0:
                marker.write_text(json.dumps(dict(protocol=PROTOCOL, variant=variant, seed=seed)))
            statuses.append(dict(variant=variant, seed=seed, status=status, output=str(output)))
            (root / "sweep_status.json").write_text(json.dumps(statuses, indent=2))
            collect_results(root)
    (root / "sweep_status.json").write_text(json.dumps(statuses, indent=2))
    collect_results(root)
    print(f"Ablation tables saved to {root}", flush=True)
    if any(row["status"] for row in statuses):
        raise SystemExit(1)
    return root


if __name__ == "__main__":
    main()
