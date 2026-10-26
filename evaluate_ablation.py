"""Re-evaluate saved ablation best.pth files, preserving training directories."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from utils.ablation_results import collect_results
from utils.experiment_runner import run_logged


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark-warmup", type=int, default=50)
    parser.add_argument("--benchmark-iterations", type=int, default=200)
    args = parser.parse_args(argv)
    root = Path(args.runs_root).resolve()
    paths = [root / "best.pth"] if (root / "best.pth").is_file() else sorted(root.glob("*/best.pth"))
    if not paths:
        raise FileNotFoundError(f"No best.pth found under {root}")
    output = Path(args.output_root).resolve() if args.output_root else root / f"evaluation_{datetime.now():%Y%m%d_%H%M%S_%f}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new or empty --output-root")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"; logs.mkdir()
    statuses = []
    for weights in paths:
        name = weights.parent.name
        destination = output / name
        command = [sys.executable, "-u", str(Path(__file__).with_name("summarize.py")),
            "--data-path", args.data_path, "--weights", str(weights), "--output", str(destination),
            "--device", args.device, "--batch-size", str(args.batch_size), "--workers", str(args.workers),
            "--benchmark" if args.benchmark else "--no-benchmark",
            "--benchmark-warmup", str(args.benchmark_warmup), "--benchmark-iterations", str(args.benchmark_iterations)]
        status = run_logged(command, logs / f"{name}.log")
        statuses.append(dict(name=name, checkpoint=str(weights), status=status))
        (output / "evaluation_status.json").write_text(json.dumps(statuses, indent=2))
        collect_results(output)
    print(f"Evaluation tables: {output}", flush=True)
    if any(row["status"] for row in statuses):
        raise SystemExit(1)
    return output


if __name__ == "__main__":
    main()
