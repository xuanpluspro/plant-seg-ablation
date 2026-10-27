"""Strip training state from an existing checkpoint; never overwrite the source."""
import argparse
from pathlib import Path
from utils.runtime import read_checkpoint, save_checkpoint
from utils.checkpoint_storage import inference_checkpoint

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.weights).resolve(), Path(args.output).resolve()
    if source == output or output.exists():
        raise FileExistsError("Choose a new output file; the source is preserved")
    checkpoint = inference_checkpoint(read_checkpoint(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(checkpoint, output)
    print(f"Saved {output}: {source.stat().st_size / 2**20:.1f} -> {output.stat().st_size / 2**20:.1f} MiB")
    print("FP32/model tensor values unchanged. Resume training from the original last.pth.")

if __name__ == "__main__":
    main()
