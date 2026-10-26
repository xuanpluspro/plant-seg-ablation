import argparse
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from train_medical import resolve_initialization
from download_pretrained import download, check_hash


class PretrainedContracts(unittest.TestCase):
    def test_default_is_required_pretraining(self):
        args = argparse.Namespace(backbone_weights=None, from_scratch=False, resume=None)
        with patch('train_medical.Path.is_file', return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, 'download_pretrained'):
                resolve_initialization(args)
        self.assertTrue(args.backbone_weights.endswith('pretrained/resnet50.pth'))
        for scratch, resume in [(True, None), (False, 'last.pth')]:
            args = argparse.Namespace(backbone_weights=None, from_scratch=scratch, resume=resume)
            resolve_initialization(args)
            self.assertIsNone(args.backbone_weights)

    def test_download_hash_and_reuse_without_network(self):
        payload = b'weight fixture'; prefix = hashlib.sha256(payload).hexdigest()[:8]
        weights = type('Weights', (), {'IMAGENET1K_V2': type('V2', (), {'url': 'https://example.test/resnet50-' + prefix + '.pth'})})
        def transfer(url, filename, **kwargs): Path(filename).write_bytes(payload)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'resnet50.pth'
            with patch('download_pretrained.ResNet50_Weights', weights), patch('torch.hub.download_url_to_file', side_effect=transfer) as fetch:
                download(path); download(path)
                self.assertEqual(fetch.call_count, 1)
                self.assertFalse(path.with_name(path.name + '.part').exists())
                path.write_bytes(b'bad')
                with self.assertRaises(ValueError): download(path)
            with self.assertRaises(ValueError): check_hash(path, prefix)


if __name__ == '__main__': unittest.main()


class BackboneExtraction(unittest.TestCase):
    def setUp(self):
        import torch
        from torchvision.models import resnet50
        from torch import nn
        torch.set_num_threads(2)
        self.torch = torch
        self.source = resnet50(weights=None)
        self.source.fc = nn.Identity()

    def test_full_unet_and_wrappers_ignore_only_other_namespaces(self):
        from model import OSSSeg
        torch = self.torch
        state = {'module.resnet.' + k: v for k, v in self.source.state_dict().items()}
        state['module.up_concat4.conv.weight'] = torch.randn(3, 5, 1, 1)
        model = OSSSeg(width=16)
        decoder_before = model.segmentation_head[-1].weight.detach().clone()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'unet.pth'
            for wrapper in (None, 'state_dict', 'model_state_dict', 'model', 'net'):
                torch.save(state if wrapper is None else {wrapper: state, 'epoch': 12}, path)
                report = model.load_backbone(path)
                self.assertEqual(report['namespace'], 'resnet.')
                self.assertEqual(report['ignored_entries'], 1)
                self.assertEqual(report['loaded_tensors'], len(self.source.state_dict()))
                for k, v in self.source.state_dict().items():
                    torch.testing.assert_close(model.backbone.state_dict()[k], v, rtol=0, atol=0)
        torch.testing.assert_close(model.segmentation_head[-1].weight, decoder_before, rtol=0, atol=0)

    def test_plain_and_backbone_prefixed_weights_with_classifier(self):
        from model.pretrained import load_resnet_backbone
        torch = self.torch
        state = dict(self.source.state_dict())
        state.update({'fc.weight': torch.randn(1000, 2048), 'fc.bias': torch.randn(1000)})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'resnet.pth'
            for prefix in ('', 'module.backbone.'):
                torch.save({prefix + k: v for k, v in state.items()}, path)
                report = load_resnet_backbone(self.source, path)
                self.assertEqual(report['ignored_entries'], 2)
                self.assertEqual(report['loaded_tensors'], len(self.source.state_dict()))

    def test_invalid_backbone_is_rejected_before_any_tensor_changes(self):
        from model.pretrained import load_resnet_backbone
        torch = self.torch
        good = dict(self.source.state_dict())
        broken = dict(good)
        broken['conv1.weight'] = torch.zeros_like(good['conv1.weight'])
        broken['layer4.2.conv3.weight'] = torch.zeros(8, 8, 1, 1)
        missing = {k: v for k, v in good.items() if k != 'layer1.0.conv3.weight'}
        ambiguous = {'resnet.' + k: v for k, v in good.items()}
        ambiguous.update({'backbone.' + k: v for k, v in good.items()})
        before = self.source.conv1.weight.detach().clone()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'bad.pth'
            for state in (broken, missing, ambiguous, {'decoder.weight': torch.zeros(1)}):
                torch.save(state, path)
                with self.assertRaises(ValueError): load_resnet_backbone(self.source, path)
                torch.testing.assert_close(self.source.conv1.weight, before, rtol=0, atol=0)
