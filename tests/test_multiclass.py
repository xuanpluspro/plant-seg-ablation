import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from model import OSSSeg
from model.losses import SegmentationLoss
from utils.dataloader_medical import decode_mask
from utils.metrics import SegmentationMetrics


class MultiClass(unittest.TestCase):
    def test_encodings_and_palette(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'mask.png'
            for raw in ([0, 1], [0, 255]):
                Image.fromarray(np.array([raw], dtype=np.uint8)).save(p)
                self.assertEqual(decode_mask(p, 2, 'auto').tolist(), [[0, 1]])
            image = Image.fromarray(np.array([[0,1,2,3,4]], dtype=np.uint8)).convert('P')
            image.putpalette([255,0,0]*256)
            image.save(p)
            self.assertEqual(decode_mask(p, 5, 'index').tolist(), [[0,1,2,3,4]])
            Image.fromarray(np.array([[0,1,255]], dtype=np.uint8)).save(p)
            for k in (2,5):
                with self.assertRaises(ValueError): decode_mask(p, k, 'auto')

    def test_class_confusion_is_penalized(self):
        target = torch.tensor([[[0,1,2,3,4]]])
        pred = torch.tensor([[[0,2,1,4,3]]])
        metric = SegmentationMetrics(5)
        metric.update(torch.nn.functional.one_hot(pred,5).permute(0,3,1,2).float(),target)
        result = metric.compute()
        self.assertEqual(result['Foreground IoU'],0.)
        self.assertEqual(result['Union Foreground IoU'],1.)
        self.assertEqual(result['Mean IoU'],.2)
        self.assertEqual(result['Dice'],0.)
        self.assertEqual(len(result['per_class']),5)

    def test_five_class_backward(self):
        torch.set_num_threads(2)
        model = OSSSeg(num_classes=5,width=16)
        target = (torch.arange(512)[None,None,:].expand(1,512,512)%5).long()
        output = model(torch.randn(1,3,512,512),return_aux=True)
        self.assertEqual(output['logits'].shape,(1,5,512,512))
        loss = SegmentationLoss()(output,target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.segmentation_head[-1].weight.grad.abs().sum().item(),0)
        self.assertGreater(model.foreground_head[-1].weight.grad.abs().sum().item(),0)

    def test_multiclass_training_and_prediction(self):
        import argparse
        import json
        from unittest.mock import patch
        import train_medical
        import predict
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'VOC'
            for folder in ('JPEGImages','SegmentationClass','ImageSets/Segmentation'):
                (root/folder).mkdir(parents=True)
            for split in ('train','val','test'):
                Image.fromarray(np.full((512,512,3),100,np.uint8)).save(root/'JPEGImages'/f'{split}.png')
                mask = np.broadcast_to(np.arange(512,dtype=np.uint16)%5,(512,512)).astype(np.uint8)
                Image.fromarray(mask).save(root/'SegmentationClass'/f'{split}.png')
                (root/'ImageSets/Segmentation'/f'{split}.txt').write_text(split+'\n')
            output = Path(d)/'run'
            args = argparse.Namespace(data_path=str(root),output=str(output),backbone_weights=None,
                from_scratch=True,resume=None,num_classes=4,mask_encoding='index',width=16,
                freeze_backbone_bn=True,epochs=1,batch_size=1,workers=0,lr=1e-4,
                weight_decay=1e-4,dice_weight=1.,aux_weight=.2,seed=11,device='cpu',amp=False,
                benchmark=False)
            with patch('utils.plot_results.plot_training_curves'):
                train_medical.train(args)
            report = json.loads((output/'metrics.json').read_text())
            self.assertEqual(report['data_config'],dict(num_classes=5,mask_encoding='index'))
            self.assertEqual(report['selection_metric'],'validation Mean IoU')
            self.assertEqual(len(report['val']['per_class']),5)
            self.assertEqual(len((output/'class_metrics.csv').read_text().splitlines()),11)
            destination = Path(d)/'pred'
            with patch('sys.argv',['predict.py','--data-path',str(root/'JPEGImages/test.png'),
                     '--weights',str(output/'best.pth'),'--device','cpu','--output',str(destination)]):
                predict.main()
            self.assertTrue(set(np.unique(np.array(Image.open(destination/'test_mask.png')))).issubset(set(range(5))))
            probability = np.load(destination/'test_prob.npy')
            self.assertEqual(probability.shape,(5,512,512))
            np.testing.assert_allclose(probability.sum(0),1.,atol=1e-6)

