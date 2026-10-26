import random
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from utils.augmentation import augment_train, random_scale_crop


class Augmentation(unittest.TestCase):
    def test_zoom_out_geometry_and_background_padding(self):
        image = Image.new("RGB", (512, 512), (20, 190, 30))
        mask = Image.new("L", (512, 512), 5)
        with patch("utils.augmentation.random.randint", return_value=20):
            x, y = random_scale_crop(image, mask, .8)
        self.assertEqual(x.size, (512, 512))
        self.assertEqual(y.size, (512, 512))
        a, b = np.asarray(x), np.asarray(y)
        np.testing.assert_array_equal(a[b == 5], np.tile([20, 190, 30], ((b == 5).sum(), 1)))
        self.assertEqual(int((b == 5).sum()), round(512 * .8) ** 2)
        self.assertEqual(set(np.unique(b)), {0, 5})

    def test_zoom_in_preserves_class_indices_and_shared_crop(self):
        mask = Image.fromarray(np.broadcast_to(np.arange(512) % 6, (512, 512)).astype("uint8"))
        image = mask.convert("RGB")
        with patch("utils.augmentation.random.randint", return_value=17):
            x, y = random_scale_crop(image, mask, 1.2)
        expected = mask.resize((614, 614), Image.Resampling.NEAREST).crop((17, 17, 529, 529))
        np.testing.assert_array_equal(np.asarray(y), np.asarray(expected))
        self.assertEqual(x.size, (512, 512))
        self.assertTrue(set(np.unique(y)).issubset(set(range(6))))

    def test_photometric_changes_image_only(self):
        image = Image.new("RGB", (512, 512), (100, 100, 100))
        mask = Image.new("L", (512, 512), 3)
        with patch("utils.augmentation.random.random", side_effect=[1., 1., 1., 0.]), \
             patch("utils.augmentation.random.randrange", return_value=0), \
             patch("utils.augmentation.random.uniform", return_value=1.1):
            x, y = augment_train(image, mask)
        self.assertNotEqual(x.getpixel((0, 0)), image.getpixel((0, 0)))
        np.testing.assert_array_equal(np.asarray(y), np.asarray(mask))

    def test_seed_reproducibility_binary_and_multiclass(self):
        for classes in (2, 4, 6):
            mask = Image.fromarray(np.broadcast_to(np.arange(512) % classes, (512, 512)).astype("uint8"))
            image = mask.convert("RGB")
            random.seed(11)
            x1, y1 = augment_train(image, mask)
            random.seed(11)
            x2, y2 = augment_train(image, mask)
            np.testing.assert_array_equal(np.asarray(x1), np.asarray(x2))
            np.testing.assert_array_equal(np.asarray(y1), np.asarray(y2))
            self.assertTrue(set(np.unique(y1)).issubset(set(range(classes))))


if __name__ == "__main__":
    unittest.main()
