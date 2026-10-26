"""Contracts and numerical behavior of the two candidate replacement modules."""
import tempfile
import unittest
from pathlib import Path
import torch
from model import OSSSeg
from model.dynamic_fusion import CrossScaleDynamicModulation
from utils.runtime import FORMAT, load_model

torch.set_num_threads(2)


class Replacements(unittest.TestCase):
    def test_three_scales_affine_and_deep_residual(self):
        for h, w in [(64, 96), (32, 48), (16, 24)]:
            m = CrossScaleDynamicModulation(16)
            low = torch.randn(1, 16, h * 2, w * 2, requires_grad=True)
            high = torch.randn(1, 16, h, w, requires_grad=True)
            with torch.no_grad():
                aligned = m.align_low(low)
                t = m.project_high(high)
                torch.testing.assert_close(m(low, high), t + m.refine(m.detail(aligned)))
            # Nonzero conditioning exposes the discrepancy gradient path as well.
            torch.nn.init.normal_(m.modulator.weight, std=.01)
            y = m(low, high)
            self.assertEqual(y.shape, high.shape)
            y.square().mean().backward()
            self.assertGreater(low.grad.abs().sum().item(), 0)
            self.assertGreater(high.grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_dynamic_fusion_modes_checkpoint(self):
        x = torch.randn(1, 3, 64, 96)
        for fusion in (False, True):
            m = OSSSeg(width=16, use_dynamic_fusion=fusion).eval()
            with torch.no_grad(): expected = m(x)
            config = dict(m.model_config)
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / 'model.pth'
                torch.save(dict(format=FORMAT, model_config=config, model=m.state_dict()), p)
                restored, _ = load_model(p, torch.device('cpu'))
                with torch.no_grad(): torch.testing.assert_close(expected, restored(x))


if __name__ == '__main__': unittest.main()
