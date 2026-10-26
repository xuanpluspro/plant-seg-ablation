import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from utils.results import export_final_results


class FinalResults(unittest.TestCase):
    def test_no_test_and_profiling_failure_keep_accuracy(self):
        splits = dict(train=['a'], val=['b'])
        ck = dict(epoch=1, model_config={}, splits=splits)
        metrics = {'Foreground IoU': .8, 'Mean IoU': .85, 'Dice': .9,
                   'Precision': .9, 'Recall': .9, 'Pixel Accuracy': .95}
        with tempfile.TemporaryDirectory() as d:
            with patch('utils.results.validate_splits', return_value=splits), \
                 patch('utils.results.PlantVOCDataset', return_value=[]), \
                 patch('utils.results.evaluate', return_value=metrics) as evaluation, \
                 patch('utils.results.benchmark_model', side_effect=RuntimeError('profiling failed')):
                result = export_final_results(nn.Conv2d(3, 2, 1), ck, 'best.pth', d, d,
                                              torch.device('cpu'), benchmark=True)
            self.assertEqual(evaluation.call_count, 1)
            self.assertNotIn('test', result)
            saved = json.loads((Path(d)/'metrics.json').read_text())
            self.assertEqual(saved['val']['Foreground IoU'], .8)
            self.assertIn('benchmark_error', saved['benchmark'])
            self.assertEqual(len((Path(d)/'results.csv').read_text(encoding='utf-8-sig').splitlines()), 2)

    def test_changed_splits_rejected_before_evaluation(self):
        with patch('utils.results.validate_splits', return_value={'train':['x'],'val':['b']}), \
             patch('utils.results.evaluate') as evaluation:
            with self.assertRaises(ValueError):
                export_final_results(nn.Conv2d(3, 2, 1),
                                     dict(epoch=0, splits={'train':['a'],'val':['b']}),
                                     'best.pth', '.', '.', torch.device('cpu'))
            evaluation.assert_not_called()


if __name__ == '__main__': unittest.main()
