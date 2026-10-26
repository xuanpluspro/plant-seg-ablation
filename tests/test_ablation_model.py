"""Actual CPU tests for full-model equivalence and every module configuration."""
import gc
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from model import OSSSeg
from model.basic_fusion import BasicTopDownFusion, BasicBottomUpFusion
from model.dynamic_fusion import CrossScaleDynamicModulation
from model.ssaf import SpectralSpatialAdaptiveFusion
from model.losses import SegmentationLoss
from ablation_config import VARIANTS, variant_flags, describe_ablation
from utils.runtime import FORMAT, LEGACY_FORMAT, load_model

torch.set_num_threads(2)


class AblationModels(unittest.TestCase):
    def test_full_matches_pinned_upstream_exactly_and_loads_legacy(self):
        source = Path(__file__).with_name("upstream_ossseg_reference.py").read_text()
        namespace = {"__name__": "model.upstream_reference", "__package__": "model"}
        exec(compile(source, "pinned_upstream", "exec"), namespace)
        torch.manual_seed(11)
        original = namespace["OSSSeg"](width=16, num_classes=4).eval()
        torch.manual_seed(11)
        full = OSSSeg(width=16, num_classes=4, **variant_flags("full")).eval()
        self.assertEqual(set(original.state_dict()), set(full.state_dict()))
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, full.state_dict()[key], atol=0, rtol=0)
        full.load_state_dict(original.state_dict(), strict=True)
        x = torch.randn(1,3,64,96)
        with torch.inference_mode():
            expected = original(x, return_aux=True)
            actual = full(x, return_aux=True)
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], atol=0, rtol=0)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"old_best.pth"
            torch.save(dict(format=LEGACY_FORMAT, model_config=original.model_config,
                            model=original.state_dict()), path)
            restored, _ = load_model(path, torch.device("cpu"))
            with torch.inference_mode():
                torch.testing.assert_close(expected["logits"], restored(x), atol=0, rtol=0)

    def test_all_eight_variants_backward_for_binary_four_and_six_classes(self):
        for classes in (2,4,6):
            for name in VARIANTS:
                with self.subTest(classes=classes, variant=name):
                    model = OSSSeg(width=8, num_classes=classes, **variant_flags(name)).train()
                    x = torch.randn(2,3,64,96)
                    target = torch.randint(classes, (2,64,96))
                    output = model(x, return_aux=True)
                    self.assertEqual(output["logits"].shape, (2,classes,64,96))
                    self.assertEqual(output["aux_logits"].shape, (2,1,16,24))
                    loss = SegmentationLoss()(output,target)
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for p in model.parameters() if p.requires_grad))
                    self.assertEqual(describe_ablation(model.model_config)["variant"],name)
                    self.assertEqual(isinstance(model.mpm, nn.Identity), not VARIANTS[name][0])
                    self.assertTrue(all(isinstance(m, SpectralSpatialAdaptiveFusion if VARIANTS[name][1] else BasicTopDownFusion)
                                        for m in model.ssaf))
                    self.assertTrue(all(isinstance(m, CrossScaleDynamicModulation if VARIANTS[name][2] else BasicBottomUpFusion)
                                        for m in model.fem))
                    del model, output, loss
                    gc.collect()

    def test_each_configuration_checkpoint_restores_without_switch_arguments(self):
        x = torch.randn(1,3,64,96)
        for name in VARIANTS:
            with self.subTest(variant=name), tempfile.TemporaryDirectory() as d:
                model = OSSSeg(width=8,num_classes=4, **variant_flags(name)).eval()
                with torch.inference_mode():expected = model(x)
                path = Path(d)/"best.pth"
                torch.save(dict(format=FORMAT, model_config=model.model_config, model=model.state_dict()),path)
                del model
                restored, checkpoint = load_model(path,torch.device("cpu"))
                self.assertEqual(describe_ablation(checkpoint["model_config"])["variant"],name)
                with torch.inference_mode():torch.testing.assert_close(expected,restored(x),atol=0,rtol=0)
                del restored
                gc.collect()

    def test_disabled_modules_remove_their_parameters_and_preserve_shapes(self):
        full = OSSSeg(width=8)
        count = sum(p.numel() for p in full.parameters())
        del full
        base = OSSSeg(width=8, **variant_flags("baseline"))
        self.assertLess(sum(p.numel() for p in base.parameters()),count)
        shallow = torch.randn(2,8,16,24);deep = torch.randn(2,8,8,12)
        self.assertEqual(BasicTopDownFusion(8)(deep,shallow).shape,shallow.shape)
        self.assertEqual(BasicBottomUpFusion(8)(shallow,deep).shape,deep.shape)
        with self.assertRaises(ValueError): BasicTopDownFusion(8)(deep,shallow[:,:,:15])


if __name__ == "__main__": unittest.main()
