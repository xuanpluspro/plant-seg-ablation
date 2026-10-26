"""Profile a saved model without a dataset: Params, FLOPs and GPU FP32 speed."""
import argparse
import json
from pathlib import Path
from utils.runtime import load_model, resolve_device
from utils.benchmark import benchmark_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--output', help='Defaults to checkpoint directory/benchmark.json')
    args = p.parse_args()
    if args.size < 64 or args.size % 32:
        p.error('--size must be >=64 and divisible by 32')
    if args.warmup < 0 or args.iterations < 1:
        p.error('Require nonnegative warmup and positive iterations')
    model, checkpoint = load_model(args.weights, resolve_device(args.device))
    report = benchmark_model(model, args.size, args.warmup, args.iterations)
    report.update(checkpoint=args.weights, epoch=checkpoint['epoch'] + 1,
                  model_config=checkpoint['model_config'])
    output = Path(args.output) if args.output else Path(args.weights).parent / 'benchmark.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
