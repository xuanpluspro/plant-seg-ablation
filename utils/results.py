"""Final checkpoint evaluation and accuracy/complexity result exports."""
import json
from pathlib import Path
from torch.utils.data import DataLoader
from .dataloader_medical import PlantVOCDataset, validate_splits
from .train_and_eval import evaluate
from .benchmark import benchmark_model
from .checkpoint_selection import checkpoint_selection_metric
from segmentation_metrics import write_run_results
from ablation_config import describe_ablation


def write_results(result, output):
    write_run_results(result, output)


def export_final_results(model, checkpoint, checkpoint_path, root, output, device,
                         batch_size=2, workers=0, benchmark=True, warmup=50, iterations=200):
    splits = validate_splits(root)
    if splits != checkpoint['splits']:
        raise ValueError('Final evaluation requires the same recorded train/val/test IDs')
    parameters = sum(p.numel() for p in model.parameters())
    result = dict(checkpoint=str(checkpoint_path), best_epoch=checkpoint['epoch'] + 1,
                  arch='BiMSeg', encoder='resnet50', seed=checkpoint.get('training_config', {}).get('seed'),
                  selection_metric='validation ' + checkpoint_selection_metric(checkpoint),
                  parameters=parameters, params_m=parameters / 1e6,
                  ablation=checkpoint.get('ablation') or describe_ablation(checkpoint['model_config']),
                  data_path=str(Path(root).resolve()), model_config=checkpoint['model_config'], data_config=checkpoint.get('data_config', {}), metric_scale='0-1',
                  metric_units={'region_scores': '0-1', 'Boundary F1': '0-1', 'HD95': 'pixels', 'ASSD': 'pixels'},
                  aggregation={'region': 'dataset confusion matrix', 'boundary': 'per-image foreground-class macro, then eligible-image mean'},
                  test_status='evaluated' if 'test' in splits else 'not provided at training time',
                  benchmark=None)
    for split in ('val', 'test'):
        if split not in splits: continue
        print(f'Final {split} evaluation using checkpoint epoch {result["best_epoch"]}...', flush=True)
        loader = DataLoader(PlantVOCDataset(root, split, augmentation=False, **checkpoint.get("data_config", {})),
                            batch_size=batch_size, num_workers=workers, shuffle=False,
                            pin_memory=device.type == 'cuda')
        result[split] = evaluate(model, loader, device, desc=f'Final {split}')
        values = result[split]
        summary = ' | '.join(f'{key}={values[key]:.4f}' if values[key] is not None else f'{key}=N/A'
                             for key in ('Foreground IoU', 'Mean IoU', 'Dice', 'Precision', 'Recall', 'Pixel Accuracy'))
        print(f'{split}: {summary}', flush=True)
    # Accuracy results survive any later profiling failure.
    write_results(result, output)
    if benchmark:
        print('Measuring Params/GFLOPs/GPU FPS: batch=1, 512x512, FP32...', flush=True)
        try:
            report = benchmark_model(model, 512, warmup, iterations)
        except Exception as exc:
            report = dict(parameters=parameters, params_m=parameters / 1e6,
                          gflops=None, fps=None, latency_ms=None,
                          benchmark_error=f'{type(exc).__name__}: {exc}')
        report.update(checkpoint=str(checkpoint_path), best_epoch=result['best_epoch'],
                      model_config=checkpoint['model_config'])
        result['benchmark'] = report
        write_results(result, output)
    print(f'Final results saved: {Path(output) / "metrics.json"} and {Path(output) / "results.csv"}', flush=True)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


