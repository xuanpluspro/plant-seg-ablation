import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from torchvision.models import resnet50
import check_pretrained


class Preflight(unittest.TestCase):
    def test_full_unet_weights_train_and_eval_pass(self):
        torch.set_num_threads(2)
        encoder = resnet50(weights=None)
        encoder.fc = nn.Identity()
        state = {'resnet.' + k: v for k, v in encoder.state_dict().items()}
        state['unet.decoder.weight'] = torch.ones(2, 3, 1, 1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'unet.pth'
            torch.save(state, path)
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                result = check_pretrained.main(['--weights', str(path), '--device', 'cpu',
                                                '--size', '64', '--width', '16'])
            self.assertEqual(result, 0)
            self.assertIn('CHECK PASSED', capture.getvalue())
            self.assertIn('skipped entries=1', capture.getvalue())

    def test_invalid_file_fails_before_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'bad.pth'
            torch.save({'resnet.conv1.weight': torch.zeros(64, 3, 7, 7)}, path)
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(check_pretrained.main(['--weights', str(path), '--device', 'cpu']), 1)
                self.assertEqual(check_pretrained.main(['--weights', str(path / 'missing')]), 1)

    def test_nonfinite_backbone_is_rejected(self):
        def corrupt(model, path):
            with torch.no_grad(): model.backbone.conv1.weight.fill_(float('nan'))
            return dict(namespace='resnet.', loaded_tensors=318, ignored_entries=0, parameters=23508032)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'dummy.pth'
            path.touch()
            capture = io.StringIO()
            with patch('model.OSSSeg.load_backbone', corrupt), contextlib.redirect_stderr(capture), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(check_pretrained.main(['--weights', str(path), '--device', 'cpu']), 1)
            self.assertIn('Non-finite pretrained tensor', capture.getvalue())
