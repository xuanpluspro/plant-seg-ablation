import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from model import OSSSeg
from model.mpm import MPM
from utils.runtime import FORMAT, load_model
from train_medical import parse_args


class MPMContracts(unittest.TestCase):
    def test_shapes_branch_sharing_and_gradients(self):
        torch.set_num_threads(2)
        model = MPM(32)
        inputs, fused, cam = [], [], []
        handles = [model.conv5.register_forward_hook(lambda m,i,o: inputs.append(i[0].shape)),
                   model.Conv.register_forward_pre_hook(lambda m,i: fused.append(i[0].shape)),
                   model.ca.register_forward_pre_hook(lambda m,i: cam.append(i[0].shape))]
        x = torch.randn(2,32,16,24,requires_grad=True)
        y = model(x)
        for h in handles: h.remove()
        self.assertEqual(inputs,[torch.Size([2,8,16,24])]*2)
        self.assertEqual(fused,[torch.Size([2,64,16,24])])
        self.assertEqual(cam,[x.shape])
        self.assertEqual(y.shape,x.shape)
        self.assertEqual(model.conv5[0].groups,1)
        self.assertEqual(model.conv5[0].kernel_size,(5,5))
        self.assertEqual(model.conv3[0].kernel_size,(3,3))
        self.assertEqual(model.conv1[0].kernel_size,(1,1))
        self.assertEqual(model.Conv.act.weight.shape,(32,1,7,7))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_default_checkpoint(self):
        model = OSSSeg(width=16).eval()
        self.assertIsInstance(model.mpm,MPM)
        x = torch.randn(1,3,64,96)
        with torch.no_grad(): expected = model(x)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'mpm.pth'
            torch.save(dict(format=FORMAT,model_config=model.model_config,model=model.state_dict()),path)
            restored,_=load_model(path,torch.device('cpu'))
            with torch.no_grad(): torch.testing.assert_close(expected,restored(x),rtol=0,atol=0)

    def test_cli_has_no_model_variants(self):
        with patch('sys.argv',['train_medical.py','--from-scratch']):
            args=parse_args()
        self.assertFalse(hasattr(args,'cssp_module'))
        self.assertFalse(hasattr(args,'mpm_variant'))
        self.assertFalse(hasattr(args,'adpm'))


class SparseAttentionContracts(unittest.TestCase):
    def test_sources_normalization_and_residual(self):
        torch.set_num_threads(2)
        model = MPM(32).eval()
        seen = {}
        def record(name):
            def hook(module, inputs, output):
                seen[name] = output.detach().clone()
            return hook
        def record_input(name):
            def hook(module, inputs):
                seen[name] = inputs[0].detach().clone()
            return hook
        handles = [model.Conv.register_forward_hook(record('fused')),
                   model.ca.register_forward_hook(record('cam')),
                   model.attention.q.register_forward_pre_hook(record_input('q_input')),
                   model.attention.kv.register_forward_pre_hook(record_input('kv_input')),
                   model.attention.register_forward_hook(record('attention'))]
        x = torch.randn(2, 32, 8, 12)
        with torch.no_grad(): actual = model(x)
        for h in handles: h.remove()
        torch.testing.assert_close(seen['q_input'], seen['fused'])
        expected_kv_input = model.attention.norm(seen['cam'].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        torch.testing.assert_close(seen['kv_input'], expected_kv_input)
        torch.testing.assert_close(actual, seen['fused'] + seen['attention'])
        # A zero attention output must leave exactly the fused feature, not CAM.
        torch.nn.init.zeros_(model.attention.proj.weight)
        torch.nn.init.zeros_(model.attention.proj.bias)
        with torch.no_grad(): torch.testing.assert_close(model(x), seen['fused'])

    def test_attention_against_per_query_reference(self):
        from model.mpm import SparseAttention
        # Independent per-query reference verifies selected value aggregation,
        # floor-based keep counts and the two unconstrained learned coefficients.
        torch.manual_seed(7)
        m = SparseAttention(8, num_heads=2).eval()
        x, y = torch.randn(1, 8, 2, 3), torch.randn(1, 8, 2, 3)
        with torch.no_grad():
            m.attn1.fill_(0.7)
            m.attn2.fill_(-0.2)
            q = m.q(x).flatten(2).transpose(1, 2)[0]
            kv = m.kv(m.norm(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
            k, v = kv.chunk(2, dim=1)
            k, v = k.flatten(2).transpose(1, 2)[0], v.flatten(2).transpose(1, 2)[0]
            reference = torch.zeros(6, 8)
            for head in range(2):
                sl = slice(head * 4, (head + 1) * 4)
                for i in range(6):
                    scores = (k[:, sl] * q[i, sl]).sum(-1) / 2
                    for divisor, coefficient in [(2, 0.7), (6, -0.2)]:
                        ids = scores.argsort(descending=True)[:max(1, 6 // divisor)]
                        reference[i, sl] += coefficient * (scores[ids].softmax(0)[:, None] * v[ids, sl]).sum(0)
            reference = m.proj(reference.T.reshape(1, 8, 2, 3))
            torch.testing.assert_close(m(x, y), reference)

    def test_small_rectangular_and_autocast_backward(self):
        from model.mpm import SparseAttention
        for h, w in [(1, 1), (2, 3), (8, 12)]:
            m = SparseAttention(16)
            x = torch.randn(2, 16, h, w, requires_grad=True)
            y = torch.randn_like(x, requires_grad=True)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                out = m(x, y)
                loss = out.float().square().mean()
            self.assertEqual(out.shape, x.shape)
            self.assertTrue(torch.isfinite(out).all())
            loss.backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue(torch.isfinite(y.grad).all())
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_configuration_and_old_checkpoint_rejected(self):
        from model.mpm import SparseAttention
        from utils.runtime import read_checkpoint
        with self.assertRaises(ValueError): SparseAttention(12)
        with self.assertRaises(ValueError): SparseAttention(16, k2=0)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'old.pth'
            torch.save({'format': 'plant_seg_lab_mpm5531_drm_cbm_v3'}, p)
            with self.assertRaisesRegex(ValueError, 'v4'): read_checkpoint(p)
