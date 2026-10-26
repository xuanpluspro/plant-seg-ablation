import unittest
from utils.checkpoint_selection import (
    checkpoint_selection_metric, resume_best_score, validation_score,
)


class CheckpointSelection(unittest.TestCase):
    def test_conflicting_foreground_and_miou_rankings(self):
        epochs = [
            {"Foreground IoU": .90, "Mean IoU": .85},
            {"Foreground IoU": .89, "Mean IoU": .865},
        ]
        self.assertEqual(max(range(2), key=lambda i: validation_score(epochs[i])), 1)

    def test_new_binary_and_multiclass_resume(self):
        for k in (2, 4, 6):
            ck = {"model_config": {"num_classes": k},
                  "selection_metric": "Mean IoU", "best_score": .8}
            self.assertEqual(resume_best_score(ck), .8)
            self.assertEqual(checkpoint_selection_metric(ck), "Mean IoU")

    def test_legacy_binary_not_relabelled_or_resumed(self):
        for metadata in ({}, {"selection_metric": "Foreground IoU", "best_score": .9}):
            ck = {"model_config": {"num_classes": 2}, "best_foreground_iou": .9, **metadata}
            self.assertEqual(checkpoint_selection_metric(ck), "Foreground IoU")
            with self.assertRaisesRegex(ValueError, "Cannot resume"):
                resume_best_score(ck)

    def test_legacy_multiclass_resume(self):
        self.assertEqual(resume_best_score({"model_config": {"num_classes": 6},
                                           "best_foreground_iou": .7}), .7)

    def test_undefined_and_nonfinite_rejected(self):
        for value in (None, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validation_score({"Mean IoU": value})


if __name__ == "__main__":
    unittest.main()
