"""Generate final results for an existing trained checkpoint without retraining."""
import argparse
from pathlib import Path
from utils.runtime import load_model, resolve_device
from utils.dataloader_medical import voc_root
from utils.results import export_final_results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--output', help='Result directory; defaults to checkpoint directory')
    p.add_argument('--benchmark', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--benchmark-warmup', type=int, default=50)
    p.add_argument('--benchmark-iterations', type=int, default=200)
    args = p.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.benchmark_warmup < 0 or args.benchmark_iterations < 1:
        p.error('Invalid batch size, workers or benchmark repetitions')
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.weights, device)
    output = Path(args.output) if args.output else Path(args.weights).parent
    export_final_results(model, checkpoint, args.weights, voc_root(args.data_path), output,
                         device, args.batch_size, args.workers, args.benchmark,
                         args.benchmark_warmup, args.benchmark_iterations)


if __name__ == '__main__': main()
