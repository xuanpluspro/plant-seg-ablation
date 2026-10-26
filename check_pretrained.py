"""Check local backbone weights and a synthetic train/eval pass before training."""
import argparse
from pathlib import Path
import sys
import torch
from model import OSSSeg
from model.losses import SegmentationLoss
from utils.runtime import resolve_device, seed_everything


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', default='pretrained/unet_resnet_voc.pth')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=1, help='Use training batch size to check approximate memory needs')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--num-classes', type=int, default=1, help='Foreground classes, excluding background (same as train_medical.py)')
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--freeze-backbone-bn', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--dynamic-fusion', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args(argv)


def check(args):
    path = Path(args.weights)
    if not path.is_file():
        raise FileNotFoundError(f'Weight file not found: {path.resolve()}')
    if args.batch_size < 1 or args.size < 64 or args.size % 32:
        raise ValueError('batch-size must be positive; size must be >=64 and divisible by 32')
    if not 1 <= args.num_classes <= 255:
        raise ValueError('num-classes must be 1..255 foreground classes, excluding background')
    seed_everything(11)
    device = resolve_device(args.device)
    print(f'[1/3] Strict backbone loading: {path.resolve()}', flush=True)
    model = OSSSeg(num_classes=args.num_classes + 1, width=args.width,
                   freeze_backbone_bn=args.freeze_backbone_bn,
                   use_dynamic_fusion=args.dynamic_fusion)
    report = model.load_backbone(path)
    for name, value in model.backbone.state_dict().items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f'Non-finite pretrained tensor (NaN/Inf): {name}')
        if name.endswith('running_var') and (value < 0).any():
            raise ValueError(f'Negative BN running variance: {name}')
    total = sum(p.numel() for p in model.parameters())
    print(f"  PASS: namespace={report['namespace']}, loaded tensors={report['loaded_tensors']}, "
          f"skipped entries={report['ignored_entries']}", flush=True)
    print(f"  Backbone={report['parameters']:,}; full model={total:,}; "
          f"backbone share={report['parameters']/total:.1%}", flush=True)
    model.to(device).train()
    amp = args.amp and device.type == 'cuda'
    print(f'[2/3] Synthetic forward/backward: device={device}, AMP={amp}, '
          f'batch={args.batch_size}, RGB={args.size}x{args.size}, '
          f'total classes={args.num_classes+1}, freeze_backbone_bn={args.freeze_backbone_bn}', flush=True)
    image = torch.randn(args.batch_size, 3, args.size, args.size, device=device)
    target = torch.randint(args.num_classes + 1, (args.batch_size, args.size, args.size), device=device)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    with torch.autocast(device_type=device.type, enabled=amp):
        output = model(image, return_aux=True)
        expected = (args.batch_size, args.num_classes + 1, args.size, args.size)
        if tuple(output['logits'].shape) != expected:
            raise ValueError(f'Wrong output shape: {tuple(output["logits"].shape)} != {expected}')
        if not all(torch.isfinite(t).all() for t in output.values()):
            raise ValueError('Non-finite training output')
        loss = SegmentationLoss()(output, target)
    if not torch.isfinite(loss):
        raise ValueError('Non-finite synthetic loss')
    # An optimizer is only used for AMP unscaling; no optimizer step is taken.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if model.backbone.conv1.weight.grad is None:
        raise ValueError('No gradient reached the backbone')
    invalid = [name for name, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
    if invalid:
        raise ValueError(f'Non-finite gradients: {invalid[:8]}')
    print(f'  PASS: logits={expected}, loss={loss.item():.6f}, gradients finite', flush=True)
    model.zero_grad(set_to_none=True)
    del output, loss, target, optimizer
    print('[3/3] Evaluation forward (eval mode)', flush=True)
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
        prediction = model(image)
    if tuple(prediction.shape) != expected or not torch.isfinite(prediction).all():
        raise ValueError('Invalid evaluation shape or non-finite evaluation output')
    print('  PASS: evaluation shape and values valid', flush=True)
    print('CHECK PASSED: weights are load-compatible and synthetic train/eval passes succeeded.\n'
          'No optimizer update, dataset training or checkpoint write was performed.\n'
          'This does not guarantee dataset validity, accuracy or full-training memory capacity.', flush=True)
    return report


def main(argv=None):
    args = parse_args(argv)
    try:
        check(args)
    except Exception as error:
        print(f'CHECK FAILED: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        if 'out of memory' in str(error).lower():
            print('Reduce --batch-size or stop other GPU jobs, then check again.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
