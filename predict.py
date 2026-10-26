"""Export 512px class masks, probabilities and overlays."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from utils.dataloader_medical import IMAGE_SIZE, IMAGE_EXTENSIONS, image_tensor
from utils.runtime import load_model, resolve_device


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", "--data_path", dest="data_path", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--output", default="run/predict/ossseg")
    p.add_argument("--device", default="auto")
    p.add_argument("--mask-output-encoding", choices=("index", "binary255"), default="binary255",
                   help="Binary export defaults to 0/255; multiclass always exports class IDs")
    args = p.parse_args()
    device = resolve_device(args.device)
    model, _ = load_model(args.weights, device)
    source = Path(args.data_path)
    paths = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS) if source.is_dir() else [source]
    if not paths:
        raise ValueError("No input images")
    # Avoid collisions between foo.jpg and foo.png in the same directory.
    destinations = set()
    for path in paths:
        relative = path.relative_to(source) if source.is_dir() else Path(path.name)
        name = relative.with_suffix("").as_posix()
        if name in destinations:
            raise ValueError(f"Duplicate output stem: {name}")
        destinations.add(name)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Prediction output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        with Image.open(path) as file:
            image = file.convert("RGB")
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Expected 512x512 input: {path}; no implicit resizing")
        probabilities = model(image_tensor(image)[None].to(device)).float().softmax(1)[0].cpu().numpy()
        labels = probabilities.argmax(0).astype(np.uint8)
        binary = probabilities.shape[0] == 2
        probability = probabilities[1] if binary else probabilities
        plant = labels > 0
        relative = path.relative_to(source) if source.is_dir() else Path(path.name)
        destination = output / relative.parent
        destination.mkdir(parents=True, exist_ok=True)
        name = relative.stem
        Image.fromarray((labels * 255).astype(np.uint8) if binary and args.mask_output_encoding == "binary255" else labels).save(destination / f"{name}_mask.png")
        np.save(destination / f"{name}_prob.npy", probability.astype(np.float32))
        overlay = np.array(image, dtype=np.float32)
        palette = np.array([[0, 0, 0]] + [[(37*i)%256, (97*i+158)%256, (173*i+83)%256] for i in range(1, 256)])
        palette[1] = [0, 255, 0]
        overlay[plant] = overlay[plant] * 0.6 + palette[labels[plant]] * 0.4
        Image.fromarray(overlay.astype(np.uint8)).save(destination / f"{name}_overlay.png")
    print(f"Saved {len(paths)} mask/probability/overlay sets to {output}")


if __name__ == "__main__":
    main()
