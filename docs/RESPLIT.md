# Rebuild GWFSS split manifests

Run from repository root, with no training using these lists in progress:

```bash
python resplit_dataset.py --data-path data/GWFSSnclass
python resplit_dataset.py --data-path data/GWFSSnclass --apply
```

Defaults: train/val/test 0.7/0.1/0.2, split seed 42, four TOTAL indexed classes (including background). No torch or GPU needed; requires NumPy/Pillow. Change ratios with `--ratios 0.7 0.1 0.2`. The split seed is independent of the training seed.

Only the union of the existing train/val/test IDs is eligible. Images and masks are not changed. Class presence, per-class relative pixel area quartiles, total foreground fraction quintiles and foreground class count are balanced using a deterministic weighted search. This is a custom stratification, not the original GWFSS split. Pixel totals are reported but are not directly optimized. Rare strata may not be perfectly balanced. Image hashes detect identical decoded RGB content. All copies are retained and bound to the same split, even when masks differ; differing-mask groups are reported without stopping. All-background masks (class ID 0 only) are retained and counted globally and per split. Group-aware assignment preserves exact rounded split sizes where feasible; infeasible group capacities stop before modifying TXT files. This does not detect near duplicates or enforce field/site/original-image grouping; use acquisition metadata separately when needed.

`--apply` backs up all existing supported TXT locations into `data/GWFSSnclass/split_backups/<timestamp_unique>/`, with a new split report recording counts, features, seed and inventory hash. Both flat ImageSets and nested ImageSets/Segmentation are updated when present. Conflicting originals, overlapping IDs, invalid/missing indexed masks and missing foreground classes stop execution. Each TXT replacement is atomic, but the three-file update is not a cross-file transaction: do not train or shut down while applying; backups permit recovery.

Use a NEW training output directory, for example `run/ablation/gwfss_712_bs32_ep30`. Do not use --resume on old 8/1/1 runs or relabel their scores with the new split. Use identical new manifests for every ablation and comparison model. Reusing old test samples as training changes the evaluation protocol and is not a fresh untouched holdout for prior model decisions.

To restore: copy backed-up ImageSets/train.txt, val.txt, test.txt (and nested copies if present) back to their original locations. Keep backups and reports with experiment records.
