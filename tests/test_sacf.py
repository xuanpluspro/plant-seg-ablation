"""DRM/CBM placement, gate granularity and gradients at all fusion scales."""
import unittest
import torch
from model.ssaf import DynamicRelationModule, ContextBridgeModule, SpectralSpatialAdaptiveFusion


class SACFContracts(unittest.TestCase):
    def test_drm_identity_gate_and_learned_path(self):
        torch.set_num_threads(2)
        m = DynamicRelationModule(16)
        x = torch.randn(2,16,8,12,requires_grad=True)
        torch.testing.assert_close(m(x),x,rtol=0,atol=0)
        weights=m.attention_net(x)
        self.assertEqual(weights.shape,(2,2,1,1))
        torch.testing.assert_close(weights.sum(1),torch.ones(2,1,1))
        m(x).square().mean().backward()
        self.assertGreater(m.gamma.grad.abs().sum().item(),0)
        m.zero_grad(set_to_none=True)
        with torch.no_grad(): m.gamma.fill_(.1)
        m(x).square().mean().backward()
        for module in (m.dilated_convs,m.attention_net,m.fusion):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()),0)

    def test_cbm_gates_original_input(self):
        m=ContextBridgeModule(16)
        x=torch.randn(2,16,12,16,requires_grad=True)
        weights=m.global_path(x)
        self.assertEqual(weights.shape,(2,16,1,1))
        expected=x+m.out_proj(m.local_path(x)+x*weights)
        torch.testing.assert_close(m(x),expected,rtol=0,atol=0)
        m(x).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_three_pairs_placement_and_two_input_gradients(self):
        for h,w in [(16,16),(32,32),(64,64)]:
            m=SpectralSpatialAdaptiveFusion(16)
            with torch.no_grad(): m.drm.gamma.fill_(.1)
            high=torch.randn(1,16,h,w,requires_grad=True)
            shallow=torch.randn(1,16,2*h,2*w,requires_grad=True)
            seen=[]
            hooks=[m.drm.register_forward_pre_hook(lambda mod,inp:seen.append(inp[0].shape)),
                   m.cbm.register_forward_pre_hook(lambda mod,inp:seen.append(inp[0].shape))]
            out=m(high,shallow)
            for hook in hooks: hook.remove()
            self.assertEqual(seen,[high.shape,shallow.shape])
            self.assertEqual(out.shape,shallow.shape)
            out.square().mean().backward()
            for grad in (high.grad,shallow.grad):
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum().item(),0)
