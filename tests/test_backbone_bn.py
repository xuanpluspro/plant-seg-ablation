"""BN defaults, running statistics and checkpoint configuration contracts."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from model import OSSSeg
from train_medical import parse_args
from utils.runtime import FORMAT, load_model


class BackboneBN(unittest.TestCase):
    def test_cli_default_and_explicit_flags(self):
        for flag, expected in [(None, False), ('--freeze-backbone-bn', True),
                               ('--no-freeze-backbone-bn', False)]:
            argv = ['train_medical.py'] + ([flag] if flag else [])
            with patch('sys.argv', argv):
                self.assertEqual(parse_args().freeze_backbone_bn, expected)

    def test_running_stats_follow_mode_without_freezing_parameters(self):
        torch.set_num_threads(2)
        default = OSSSeg(width=16)
        self.assertFalse(default.model_config['freeze_backbone_bn'])
        for freeze in (False, True):
            model = default if not freeze else OSSSeg(width=16, freeze_backbone_bn=True)
            model.train()
            backbone_bn = [m for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d)]
            self.assertTrue(all(m.training == (not freeze) for m in backbone_bn))
            self.assertTrue(model.mpm.ca.bn1.training)
            self.assertTrue(model.backbone.conv1.weight.requires_grad)
            self.assertTrue(all(m.weight.requires_grad and m.bias.requires_grad for m in backbone_bn))
            bn = model.backbone.bn1
            before = bn.running_mean.clone()
            count = bn.num_batches_tracked.clone()
            bn(torch.ones(2, 64, 4, 4))
            self.assertEqual(torch.equal(before, bn.running_mean), freeze)
            self.assertEqual(int(bn.num_batches_tracked-count), 0 if freeze else 1)
            model.eval()
            before = bn.running_mean.clone()
            bn(torch.ones(2, 64, 4, 4) * 3)
            torch.testing.assert_close(before, bn.running_mean)
            self.assertTrue(all(not m.training for m in backbone_bn))

    def test_saved_bn_setting_survives_default_change(self):
        for freeze in (False, True):
            model = OSSSeg(width=16, freeze_backbone_bn=freeze)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model.pth'
                torch.save(dict(format=FORMAT, model_config=model.model_config,
                                model=model.state_dict()), path)
                restored, _ = load_model(path, torch.device('cpu'))
                self.assertEqual(restored.freeze_backbone_bn, freeze)
                restored.train()
                self.assertEqual(restored.backbone.bn1.training, not freeze)
