"""Download hash-checked official torchvision ResNet-50 ImageNet V2 weights."""
import argparse
import hashlib
from pathlib import Path
import re
import torch
from torchvision.models import ResNet50_Weights

DEFAULT_PATH = Path(__file__).resolve().parent / 'pretrained' / 'resnet50.pth'


def check_hash(path, prefix):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if not digest.hexdigest().startswith(prefix):
        raise ValueError(f'Weight checksum mismatch: {path}')


def download(output=DEFAULT_PATH):
    path = Path(output)
    url = ResNet50_Weights.IMAGENET1K_V2.url
    prefix = re.search(r'-([0-9a-f]+)\.pth$', url).group(1)
    if path.exists():
        check_hash(path, prefix)
        print(f'Existing official weights verified: {path}')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part')
    try:
        torch.hub.download_url_to_file(url, str(temporary), hash_prefix=prefix, progress=True)
        check_hash(temporary, prefix)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Saved torchvision ResNet-50 IMAGENET1K_V2 weights: {path}')
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_PATH)
    download(p.parse_args().output)


if __name__ == '__main__':
    main()
