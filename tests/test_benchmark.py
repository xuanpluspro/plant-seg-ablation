import unittest
from unittest.mock import patch
import torch
from torch import nn
from utils.benchmark import benchmark_model


class BenchmarkContracts(unittest.TestCase):
    def test_known_conv_count_and_restore(self):
        model = nn.Sequential(nn.Conv2d(3, 2, 1, bias=False), nn.BatchNorm2d(2))
        model.train()
        model[1].eval()
        modes = [m.training for m in model.modules()]
        state = {k: v.clone() for k, v in model.state_dict().items()}
        report = benchmark_model(model, size=8, warmup=0, iterations=1)
        self.assertEqual(report['parameters'], 10)
        self.assertGreaterEqual(report['counted_gflops'], 3 * 2 * 8 * 8 / 1e9)
        self.assertIsNone(report['fps'])
        self.assertIsNone(report['latency_ms'])
        self.assertEqual(modes, [m.training for m in model.modules()])
        for key, value in model.state_dict().items(): torch.testing.assert_close(value, state[key])

    def test_partial_flops_are_not_full(self):
        class FakeAnalysis:
            def __init__(self, *a): pass
            def unsupported_ops_warnings(self, *a): return self
            def uncalled_modules_warnings(self, *a): return self
            def total(self): return 123
            def unsupported_ops(self): return {'aten::example': 1}
        with patch('fvcore.nn.FlopCountAnalysis', FakeAnalysis):
            report = benchmark_model(nn.Conv2d(3, 2, 1), 8, 0, 1)
        self.assertFalse(report['flops_complete'])
        self.assertIsNone(report['gflops'])
        self.assertEqual(report['counted_gflops'], 123 / 1e9)


if __name__ == '__main__': unittest.main()
