"""Regenerate overall and per-class ablation tables; no model or GPU required."""
import argparse
from utils.ablation_results import collect_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output", help="Defaults to runs-root")
    args = parser.parse_args()
    result = collect_results(args.runs_root, args.output)
    print(f"Summarized {result['completed_result_files']} result files")


if __name__ == "__main__":
    main()
