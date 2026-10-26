"""Nested/flat manifests must share discovery, validation and test registration."""
import tempfile
import unittest
from pathlib import Path
from utils.dataloader_medical import read_ids, validate_splits, split_paths


class SplitLayouts(unittest.TestCase):
    def write(self, root, split, text, nested=False):
        path = split_paths(root, split)[0 if nested else 1]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        return path

    def test_flat_nested_and_mixed_discover_optional_test(self):
        for nested in (False, True):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                self.write(root, 'train', 'a\nb\n', nested)
                self.write(root, 'val', 'c\n', nested)
                self.assertEqual(validate_splits(root), dict(train=['a','b'], val=['c']))
                self.write(root, 'test', 'd\n', not nested)
                self.assertEqual(validate_splits(root)['test'], ['d'])

    def test_identical_copies_allowed_but_conflicts_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.write(root, 'train', '\ufeffa\nb\n')
            nested = self.write(root, 'train', 'a\nb\n', True)
            self.assertEqual(read_ids(root, 'train'), ['a','b'])
            nested.write_text('b\na\n')
            with self.assertRaisesRegex(ValueError, 'Conflicting split lists'):
                read_ids(root, 'train')

    def test_flat_lists_keep_leakage_and_id_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            train = self.write(root, 'train', 'a\n')
            self.write(root, 'val', 'b\n')
            test = self.write(root, 'test', 'a\n')
            with self.assertRaisesRegex(ValueError, 'Split leakage'):
                validate_splits(root)
            test.unlink()
            for text in ('', 'a\na\n', '../a\n', 'a\x00b\n', 'a\\b\n'):
                train.write_text(text)
                with self.assertRaises(ValueError): read_ids(root, 'train')

    def test_missing_lists_report_both_paths(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError) as caught:
                read_ids(Path(d), 'val')
            for path in split_paths(Path(d), 'val'):
                self.assertIn(str(path), str(caught.exception))


    def test_spaces_parentheses_and_multiclass_pairs(self):
        import numpy as np
        from PIL import Image
        from utils.dataloader_medical import PlantVOCDataset
        for nested in (False, True):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                (root / 'JPEGImages').mkdir()
                (root / 'SegmentationClass').mkdir()
                names = ['ARVALIS_ (1)', 'ARVALIS_  (10)', '小麦 sample (100)']
                for split, name in zip(('train', 'val', 'test'), names):
                    self.write(root, split, name + '\n', nested)
                    Image.fromarray(np.zeros((512,512,3), dtype=np.uint8)).save(root / 'JPEGImages' / (name + '.png'))
                    mask = np.tile(np.arange(512, dtype=np.uint16) % 4, (512,1)).astype(np.uint8)
                    Image.fromarray(mask).save(root / 'SegmentationClass' / (name + '.png'))
                splits = validate_splits(root)
                for split, name in zip(('train', 'val', 'test'), names):
                    self.assertEqual(splits[split], [name])
                    data = PlantVOCDataset(root, split, num_classes=4, mask_encoding='index')
                    image, label, sample_id = data[0]
                    self.assertEqual(sample_id, name)
                    self.assertEqual(tuple(image.shape), (3,512,512))
                    self.assertEqual(set(label.unique().tolist()), {0,1,2,3})
