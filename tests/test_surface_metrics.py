import unittest
import numpy as np
from utils.surface_metrics import surface_scores


class Surfaces(unittest.TestCase):
    def test_identical_empty_and_one_empty(self):
        a = np.zeros((16, 16), bool); a[4:8, 4:8] = True
        self.assertEqual(surface_scores(a, a), (0., 0., 1.))
        empty = np.zeros_like(a)
        self.assertIsNone(surface_scores(empty, empty))
        hd, assd, f1 = surface_scores(empty, a)
        self.assertAlmostEqual(hd, np.hypot(16, 16)); self.assertEqual(hd, assd)
        self.assertEqual(f1, 0.)

    def test_shifted_single_pixel(self):
        a = np.zeros((16,16), bool); b = a.copy()
        a[4,4] = True; b[4,7] = True
        self.assertEqual(surface_scores(a, b), (3., 3., 0.))
        self.assertEqual(surface_scores(a, b, tolerance=3), (3., 3., 1.))


if __name__ == '__main__': unittest.main()

