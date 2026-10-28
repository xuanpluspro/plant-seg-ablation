"""Rebuild VOC split TXT files with deterministic class/area stratification.
No torch required. Defaults: GWFSS, train/val/test=70/10/20, seed=42.
Uses only IDs in the existing manifests; never adds unlisted images.
Exact decoded-image duplicates stop the operation for manual review.
Same-source/near-duplicate grouping requires acquisition metadata and is NOT inferred.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image

SPLITS = ("train", "val", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_lists(root):
    result, paths = {}, []
    for split in SPLITS:
        candidates = [root / "ImageSets" / f"{split}.txt",
                      root / "ImageSets" / "Segmentation" / f"{split}.txt"]
        existing = [p for p in candidates if p.is_file()]
        if not existing:
            raise ValueError(f"Missing {split}.txt")
        versions = []
        for p in existing:
            ids = [s.strip() for s in p.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
            if not ids or len(ids) != len(set(ids)):
                raise ValueError(f"Empty or duplicate IDs: {p}")
            if any("/" in s or "\\" in s or s in (".", "..") or "\x00" in s for s in ids):
                raise ValueError(f"Expected filename stems: {p}")
            versions.append(ids)
        if any(v != versions[0] for v in versions[1:]):
            raise ValueError(f"Conflicting flat/nested lists for {split}; resolve first")
        result[split] = versions[0]
        paths.extend(existing)
    all_ids = sum(result.values(), [])
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Existing train/val/test overlap; resolve before resplitting")
    return result, paths


def inspect(root, ids, classes):
    images = {}
    for p in (root / "JPEGImages").iterdir():
        if p.is_file() and p.suffix.lower() in EXTENSIONS:
            if p.stem in images:
                raise ValueError(f"Ambiguous image stem: {p.stem}")
            images[p.stem] = p
    counts, hashes, seen = [], [], {}
    for i, name in enumerate(ids):
        if name not in images:
            raise ValueError(f"Missing image: {name}")
        mask_path = root / "SegmentationClass" / (name + ".png")
        with Image.open(images[name]) as im:
            rgb = np.asarray(im.convert("RGB"))
        with Image.open(mask_path) as im:
            mask = np.asarray(im)
        if mask.ndim != 2 or mask.shape != rgb.shape[:2]:
            raise ValueError(f"Invalid mask shape or image/mask mismatch: {name}")
        if not np.isin(mask, np.arange(classes)).all():
            raise ValueError(f"Expected indexed mask 0..{classes-1}: {name}")
        digest = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
        if digest in seen:
            raise ValueError(f"Duplicate decoded image: {seen[digest]} and {name}; "
                             "group/deduplicate explicitly before splitting")
        seen[digest] = name
        hashes.append(hashlib.sha256((digest + hashlib.sha256(mask.tobytes()).hexdigest()).encode()).hexdigest())
        counts.append(np.bincount(mask.ravel(), minlength=classes))
        if (i + 1) % 200 == 0:
            print(f"Audited {i+1}/{len(ids)}", flush=True)
    return np.asarray(counts), hashes


def features(counts):
    presence = counts[:, 1:] > 0
    columns, names, weights = [], [], []
    def add(name, x, weight):
        if np.any(x):
            names.append(name); columns.append(np.asarray(x, float)); weights.append(weight)
    def bins(name, rows, values, number, weight):
        rows = np.asarray(rows)
        order = np.argsort(values, kind="stable")
        for k, positions in enumerate(np.array_split(order, min(number, len(order)))):
            x = np.zeros(len(counts))
            x[rows[positions]] = 1
            add(f"{name}_bin{k+1}", x, weight)
    total = counts.sum(1)
    for c in range(1, counts.shape[1]):
        rows = np.flatnonzero(counts[:, c] > 0)
        if len(rows) < 3:
            raise ValueError(f"Class {c} occurs in fewer than 3 images")
        add(f"class{c}_presence", counts[:, c] > 0, 5.)
        bins(f"class{c}_relative_area", rows, counts[rows, c] / total[rows], 4, 4.)
    bins("foreground_fraction", np.arange(len(counts)), counts[:, 1:].sum(1)/total, 5, 3.)
    for k in np.unique(presence.sum(1)):
        add(f"foreground_class_count{k}", presence.sum(1) == k, 1.)
    return np.stack(columns, axis=1), np.asarray(weights), names


def assign(x, weights, ratios, seed):
    n = len(x)
    sizes = np.floor(n * ratios).astype(int)
    for s in np.argsort(-(n * ratios - sizes), kind="stable")[:n-sizes.sum()]:
        sizes[s] += 1
    if min(sizes) == 0:
        raise ValueError("Dataset too small for requested split")
    rng = np.random.default_rng(seed)
    totals = x.sum(0)
    target = ratios[:, None] * totals
    scale = np.maximum(target, 1.)
    def score(actual):
        return float((((actual-target)/scale)**2 * weights).sum())
    # Multiple seeded random candidates, followed by fixed-size pair swaps.
    best, best_score = None, float("inf")
    for _ in range(64):
        order = rng.permutation(n)
        a = np.empty(n, dtype=int)
        a[order] = np.repeat(np.arange(3), sizes)
        actual = np.stack([x[a == s].sum(0) for s in range(3)])
        value = score(actual)
        if value < best_score:
            best, best_score = a.copy(), value
    a = best
    actual = np.stack([x[a == s].sum(0) for s in range(3)])
    for _ in range(max(20000, n * 100)):
        i, j = rng.integers(n, size=2)
        u, v = a[i], a[j]
        if u == v:
            continue
        delta = x[j] - x[i]
        candidate = actual.copy()
        candidate[u] += delta; candidate[v] -= delta
        value = score(candidate)
        if value < best_score - 1e-12:
            actual, best_score = candidate, value
            a[i], a[j] = v, u
    return a, sizes, actual


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default="data/GWFSSnclass")
    p.add_argument("--ratios", nargs=3, type=float, default=[.7, .1, .2],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-classes", type=int, default=4, help="TOTAL classes including background")
    p.add_argument("--apply", action="store_true", help="Back up and replace TXT files; otherwise dry run")
    args = p.parse_args()
    root = Path(args.data_path).resolve()
    ratios = np.asarray(args.ratios)
    if not np.isfinite(ratios).all() or min(ratios) <= 0 or not np.isclose(ratios.sum(), 1.):
        p.error("Three positive finite ratios must sum to 1")
    old, paths = read_lists(root)
    ids = sorted(sum(old.values(), []))
    counts, hashes = inspect(root, ids, args.num_classes)
    x, weights, names = features(counts)
    assignment, sizes, actual = assign(x, weights, ratios, args.seed)
    new = {s: [name for i, name in enumerate(ids) if assignment[i] == k]
           for k, s in enumerate(SPLITS)}
    for k, s in enumerate(SPLITS):
        if not (counts[assignment == k, 1:].sum(0) > 0).all():
            raise ValueError(f"Missing foreground class in {s}")
    report = dict(strategy="class_presence_relative_area_stratified_v1", seed=args.seed,
                  ratios=dict(zip(SPLITS, ratios.tolist())), classes=args.num_classes,
                  old_counts={s:len(old[s]) for s in SPLITS},
                  new_counts={s:len(new[s]) for s in SPLITS},
                  inventory_sha256=hashlib.sha256("\n".join(n+":"+h for n,h in zip(ids,hashes)).encode()).hexdigest(),
                  features=names,
                  feature_counts={s:actual[k].tolist() for k,s in enumerate(SPLITS)},
                  class_pixels={s:counts[assignment == k].sum(0).tolist() for k,s in enumerate(SPLITS)},
                  warning="No acquisition/site/near-duplicate grouping; no images or masks changed")
    print(json.dumps(report, indent=2))
    if not args.apply:
        print("DRY RUN only. Add --apply to back up and replace existing TXT files.")
        return
    backup_root = root / "split_backups"
    backup_root.mkdir(exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=backup_root))
    for path in paths:
        dest = backup / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    (backup / "new_split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Refresh every existing supported location, preventing conflicting duplicate lists.
    staged = []
    for path in paths:
        tmp = path.with_name(path.name + ".resplit.tmp")
        tmp.write_text("".join(name+"\n" for name in new[path.stem]), encoding="utf-8")
        staged.append((tmp, path))
    try:
        for tmp, path in staged:
            os.replace(tmp, path)
        read_lists(root)
    except BaseException:
        for path in paths:
            shutil.copy2(backup / path.relative_to(root), path)
        raise
    print(f"APPLIED. Original TXT backup: {backup}")
    print("Start a NEW output-root. Do not resume checkpoints trained with old lists.")


if __name__ == "__main__":
    main()
