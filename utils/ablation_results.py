"""Aggregate eight ablation configurations without loading checkpoints."""
import csv
import json
from pathlib import Path

from ablation_config import describe_ablation, PROTOCOL
from segmentation_metrics import RESULT_FIELDS, CLASS_FIELDS, result_rows


def _write(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def collect_results(runs_root, output=None):
    root = Path(runs_root).resolve()
    output = Path(output).resolve() if output else root
    paths = [root / "metrics.json"] if (root / "metrics.json").is_file() else sorted(root.glob("*/metrics.json"))
    output.mkdir(parents=True, exist_ok=True)
    records, classes, seen, data_roots = [], None, set(), set()
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        config_path = path.parent / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        model_config = result.get("model_config", config.get("model", {}))
        ablation = describe_ablation(model_config)
        seed = result.get("seed", config.get("training", {}).get("seed"))
        identity = (ablation["variant"], seed)
        if identity in seen:
            raise ValueError(f"Duplicate ablation/seed {identity}; select one sweep")
        seen.add(identity)
        data_path = result.get("data_path", config.get("data_path"))
        if data_path:
            data_roots.add(str(Path(data_path).resolve()))
        meta = dict(variant=ablation["variant"], msca=ablation["use_msca"],
                    drcf=ablation["use_drcf"], dcam=ablation["use_dcam"], seed=seed,
                    run=str(path.parent), metrics_file=str(path))
        for split in ("val", "test"):
            if split not in result:
                continue
            per_class = result[split].get("per_class", [])
            mapping = tuple((r["class_id"], r.get("class_name", f"class_{r['class_id']}")) for r in per_class)
            if not mapping:
                raise ValueError(f"Missing per_class: {path}")
            if classes is not None and classes != mapping:
                raise ValueError("Class mappings differ; summarize only one dataset")
            classes = mapping
        records.append((result, meta))
    if len(data_roots) > 1:
        raise ValueError("Dataset roots differ; summarize only one dataset")
    meta_fields = ("variant", "msca", "drcf", "dcam", "seed")
    rows, class_rows = [], []
    for result, meta in records:
        for row in result_rows(result):
            rows.append(dict(row, **meta))
        for split in ("val", "test"):
            for row in result.get(split, {}).get("per_class", []):
                class_rows.append(dict(row, split=split, **meta))
    overall_fields = tuple(dict.fromkeys((*meta_fields, *RESULT_FIELDS, "run", "metrics_file")))
    _write(output / "ablation_results.csv", overall_fields, rows)
    _write(output / "all_class_metrics.csv", (*meta_fields, *CLASS_FIELDS, "run", "metrics_file"), class_rows)
    class_cols = [(cid, f"iou_{cid}_{name}") for cid, name in (classes or ())]
    wide_fields = (*meta_fields, "split", "miou", "foreground_miou", "dice", "macc", "pixacc",
                   *[name for _, name in class_cols], "params_m", "gflops", "fps", "run")
    for split in ("val", "test"):
        wide = []
        for result, meta in records:
            if split not in result:
                continue
            row = next(r for r in result_rows(result) if r["split"] == split)
            scores = {r["class_id"]: r.get("iou", r.get("IoU")) for r in result[split]["per_class"]}
            row.update({name: scores[cid] for cid, name in class_cols})
            row.update(meta); wide.append(row)
        _write(output / f"{split}_class_iou.csv", wide_fields, wide)
    manifest = dict(ablation_protocol=PROTOCOL, completed_result_files=len(records),
        source_files=[str(path) for path in paths], class_mapping=list(classes or ()),
        units="region scores 0-1; HD95/ASSD pixels; params_m millions; FPS model forward only",
        note="Each seed is a separate row; no best-seed picking or cross-seed averaging")
    (output / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
