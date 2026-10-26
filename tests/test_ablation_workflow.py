import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ablation_config import VARIANTS, variant_flags, describe_ablation
from segmentation_metrics import DatasetSegmentationMetrics, write_run_results
from utils.ablation_results import collect_results
import batch_ablation


def sample_result(config, seed=11):
    meter = DatasetSegmentationMetrics(4)
    labels = np.arange(4).reshape(1,2,2)
    meter.update(labels, labels)
    return dict(seed=seed, best_epoch=1, model_config=config,
                ablation=describe_ablation(config), parameters=100, params_m=.0001,
                test=meter.compute(), benchmark=None)


class AblationWorkflow(unittest.TestCase):
    def data_root(self, directory):
        root = Path(directory)/"data"
        for name in ("JPEGImages", "SegmentationClass", "ImageSets"):
            (root/name).mkdir(parents=True)
        for split in ("train","val","test"):
            (root/"ImageSets"/f"{split}.txt").write_text(split+"\n")
        return root

    def arguments(self, root, output):
        return ["--output-root",str(output),"--data-path",str(root),"--from-scratch",
            "--num-classes","3","--mask-encoding","index","--width","8",
            "--epochs","1","--batch-size","1","--workers","0","--device","cpu","--no-benchmark"]

    def fake_train(self, command, log_path):
        output = Path(command[command.index("--output")+1])
        seed = int(command[command.index("--seed")+1])
        config = dict(num_classes=4, width=8, freeze_backbone_bn=False, use_dynamic_fusion=True)
        config.update({f"use_{name}":f"--{name}" in command for name in ("msca","drcf","dcam")})
        output.mkdir(parents=True, exist_ok=True)
        (output/"config.json").write_text(json.dumps(dict(model=config,training=dict(seed=seed))))
        write_run_results(sample_result(config,seed),output)
        Path(log_path).write_text("synthetic runner log")
        return 0

    def test_eight_unique_definitions(self):
        self.assertEqual(len(set(VARIANTS.values())),8)
        for name in VARIANTS:
            self.assertEqual(describe_ablation(variant_flags(name))["variant"],name)

    def test_batch_resumes_completed_jobs_without_retraining(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.data_root(d); output = Path(d)/"sweep"
            args = self.arguments(root,output)
            with patch("batch_ablation.run_logged",side_effect=self.fake_train) as worker:
                batch_ablation.main(args)
                self.assertEqual(worker.call_count,8)
            with (output/"test_class_iou.csv").open(encoding="utf-8-sig") as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),8)
            self.assertEqual({r["variant"] for r in rows},set(VARIANTS))
            with (output/"all_class_metrics.csv").open(encoding="utf-8-sig") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),32)
            with patch("batch_ablation.run_logged") as worker:
                batch_ablation.main([*args,"--resume"])
                worker.assert_not_called()

    def test_resume_changed_budget_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.data_root(d);output=Path(d)/"sweep"
            args=self.arguments(root,output)+["--variants","baseline"]
            with patch("batch_ablation.run_logged",side_effect=self.fake_train):batch_ablation.main(args)
            changed=list(args);changed[changed.index("--epochs")+1]="2"
            with self.assertRaises(ValueError):batch_ablation.main([*changed,"--resume"])

    def test_reserved_switch_override_rejected(self):
        with self.assertRaises(SystemExit):
            batch_ablation.main(["--output-root","unused","--no-msca"])

    def test_failed_job_is_recorded_and_remaining_jobs_run(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.data_root(d);output=Path(d)/"sweep"
            args=self.arguments(root,output)+["--variants","baseline","full"]
            calls=[]
            def worker(command,log):
                calls.append(command)
                return 5 if len(calls)==1 else self.fake_train(command,log)
            with patch("batch_ablation.run_logged",side_effect=worker):
                with self.assertRaises(SystemExit):batch_ablation.main(args)
            self.assertEqual(len(calls),2)
            statuses=json.loads((output/"sweep_status.json").read_text())
            self.assertEqual([r["status"] for r in statuses],[5,0])

    def test_pure_summary_preserves_files_and_keeps_seeds_separate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for seed in (11,12):
                write_run_results(sample_result(variant_flags("full"),seed),root/f"full_seed{seed}")
            before=(root/"full_seed11/metrics.json").read_bytes()
            report=collect_results(root)
            self.assertEqual(report["completed_result_files"],2)
            self.assertEqual((root/"full_seed11/metrics.json").read_bytes(),before)

    def test_interrupted_job_resumes_last_checkpoint_without_initialization_flag(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.data_root(d);output=Path(d)/"sweep"
            args=self.arguments(root,output)+["--variants","baseline"]
            def interrupt(command,log):
                destination=Path(command[command.index("--output")+1]);destination.mkdir()
                (destination/"last.pth").write_bytes(b"saved state")
                return 1
            with patch("batch_ablation.run_logged",side_effect=interrupt):
                with self.assertRaises(SystemExit):batch_ablation.main(args)
            def resume(command,log):
                self.assertIn("--resume",command)
                self.assertNotIn("--from-scratch",command)
                self.assertEqual(command[-1],str(output/"baseline_seed11/last.pth"))
                return self.fake_train(command,log)
            with patch("batch_ablation.run_logged",side_effect=resume):
                batch_ablation.main([*args,"--resume"])


if __name__ == "__main__": unittest.main()
