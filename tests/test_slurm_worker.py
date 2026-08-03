from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scruffy.slurm import build_srun_argv, cancel_step, live_steps, load_inventory
from scruffy.worker import execute_assignment, find_node_assignment


class SlurmArgumentTests(unittest.TestCase):
    def test_multi_node_srun_argv_is_exact(self) -> None:
        assignment_file = Path("/shared/scruffy/jobs/job-1/assignment.json")

        argv = build_srun_argv(
            slurm_job_id="240292",
            name="scruffy-launch-token",
            assignment_file=assignment_file,
            node_names=["gpu-3", "gpu-5"],
            cpus_per_node=28,
            memory_gb_per_node=256,
            wait_seconds=45,
        )

        self.assertEqual(
            [
                "srun",
                "--jobid=240292",
                "--job-name=scruffy-launch-token",
                "--overlap",
                "--exact",
                "--nodes=2",
                "--nodelist=gpu-3,gpu-5",
                "--ntasks=2",
                "--ntasks-per-node=1",
                "--cpus-per-task=28",
                "--mem=256G",
                "--kill-on-bad-exit=1",
                "--wait=45",
                "--wait-for-children",
                "--label",
                sys.executable,
                "-m",
                "scruffy.worker",
                str(assignment_file),
            ],
            argv,
        )

    def test_srun_requires_an_outer_allocation_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "Slurm job ID"):
            build_srun_argv(
                slurm_job_id="",
                name="scruffy-launch-token",
                assignment_file=Path("assignment.json"),
                node_names=["gpu-3"],
                cpus_per_node=1,
                memory_gb_per_node=1,
            )

    def test_srun_default_waits_for_descendants_without_rank_timeout(self) -> None:
        argv = build_srun_argv(
            slurm_job_id="240292",
            name="scruffy-token",
            assignment_file=Path("assignment.json"),
            node_names=["gpu-3"],
            cpus_per_node=1,
            memory_gb_per_node=1,
        )

        self.assertIn("--wait=0", argv)
        self.assertIn("--wait-for-children", argv)


class InventoryTests(unittest.TestCase):
    def _load(self, document: object):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "inventory.json"
            source.write_text(json.dumps(document), encoding="utf-8")
            return load_inventory(source)

    def test_inventory_document_must_be_an_object(self) -> None:
        for document in ([], None):
            with self.subTest(document=document):
                with self.assertRaisesRegex(ValueError, "JSON object"):
                    self._load(document)

    def test_inventory_rejects_inner_names_and_hostname_aliases(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain 'name'"):
            self._load(
                {
                    "nodes": {
                        "gpu-3": {
                            "name": "gpu-5",
                            "gpu_ids": [0],
                            "cpus": 1,
                            "memory_gb": 1,
                        }
                    }
                }
            )
        with self.assertRaisesRegex(ValueError, "short node names"):
            self._load(
                {
                    "nodes": {
                        "gpu-3": {"gpu_ids": [0], "cpus": 1, "memory_gb": 1},
                        "gpu-3.cluster": {
                            "gpu_ids": [1],
                            "cpus": 1,
                            "memory_gb": 1,
                        },
                    }
                }
            )


class SlurmReconciliationTests(unittest.TestCase):
    def test_live_steps_parses_structured_scontrol_output(self) -> None:
        result = mock.Mock(
            stdout=json.dumps(
                {
                    "steps": [
                        {
                            "id": "240292.17",
                            "name": "scruffy-token",
                            "nodes": "gpu-[3,5]",
                        }
                    ],
                    "errors": [],
                }
            )
        )
        with mock.patch("scruffy.slurm.subprocess.run", return_value=result) as run:
            steps = live_steps("240292")

        self.assertEqual("240292.17", steps[0].step_id)
        self.assertEqual("scruffy-token", steps[0].name)
        run.assert_called_once_with(
            ["scontrol", "--json", "show", "step", "240292"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_scontrol_errors_are_unknown_not_absence(self) -> None:
        result = mock.Mock(stdout=json.dumps({"steps": [], "errors": ["down"]}))
        with (
            mock.patch("scruffy.slurm.subprocess.run", return_value=result),
            self.assertRaisesRegex(RuntimeError, "reported errors"),
        ):
            live_steps("240292")

    def test_cancel_accepts_only_an_exact_numeric_step(self) -> None:
        with mock.patch("scruffy.slurm.subprocess.run") as run:
            cancel_step("240292", "240292.17")
        run.assert_called_once_with(
            ["scancel", "--ctld", "--quiet", "240292.17"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        for unsafe in ("240292", "240292.batch", "999.17", ""):
            with self.assertRaisesRegex(ValueError, "unsafe Slurm step ID"):
                cancel_step("240292", unsafe)


class WorkerPlacementTests(unittest.TestCase):
    def test_multi_node_worker_selects_its_own_reservation(self) -> None:
        document = {
            "assignment": [
                {"node": "gpu-3.cluster", "gpu_ids": [0, 1]},
                {"node": "gpu-5.cluster", "gpu_ids": [4, 6]},
            ]
        }

        placement = find_node_assignment(document, "gpu-5")

        self.assertEqual("gpu-5.cluster", placement["node"])
        self.assertEqual([4, 6], placement["gpu_ids"])

    def test_exact_hostname_wins_over_an_earlier_short_name_match(self) -> None:
        document = {
            "assignment": [
                {"node": "gpu-3.alpha.example", "gpu_ids": [0]},
                {"node": "gpu-3.beta.example", "gpu_ids": [1]},
            ]
        }

        placement = find_node_assignment(document, "gpu-3.beta.example")

        self.assertEqual("gpu-3.beta.example", placement["node"])

    def test_ambiguous_short_hostname_is_rejected(self) -> None:
        document = {
            "assignment": [
                {"node": "gpu-3.alpha.example", "gpu_ids": [0]},
                {"node": "gpu-3.beta.example", "gpu_ids": [1]},
            ]
        }

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            find_node_assignment(document, "gpu-3")

    def test_worker_overrides_submitted_gpu_visibility(self) -> None:
        document = {
            "root": "/shared/scruffy",
            "job_id": "job-1",
            "argv": ["python", "train.py"],
            "cwd": "/work/job-1",
            "env": {
                "CUDA_VISIBLE_DEVICES": "99",
                "CUDA_DEVICE_ORDER": "FASTEST_FIRST",
                "SCRUFFY_JOB_ID": "spoofed-job",
                "SCRUFFY_ROOT": "/spoofed/root",
                "SCRUFFY_EVENT_DIR": "/spoofed/events",
                "SCRUFFY_NODE": "spoofed-node",
                "USER_SETTING": "kept",
            },
            "assignment": [
                {"node": "gpu-3.cluster", "gpu_ids": [0]},
                {"node": "gpu-5.cluster", "gpu_ids": [4, 6]},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "assignment.json"
            source.write_text(json.dumps(document), encoding="utf-8")
            with (
                mock.patch.dict(os.environ, {"BASE_SETTING": "base"}, clear=True),
                mock.patch("scruffy.worker.current_node", return_value="gpu-5.cluster"),
                mock.patch("scruffy.worker.os.chdir") as change_directory,
                mock.patch("scruffy.worker.os.execvpe") as execvpe,
            ):
                execute_assignment(source)

        change_directory.assert_called_once_with("/work/job-1")
        executable, argv, environment = execvpe.call_args.args
        self.assertEqual("python", executable)
        self.assertEqual(["python", "train.py"], argv)
        self.assertEqual("base", environment["BASE_SETTING"])
        self.assertEqual("kept", environment["USER_SETTING"])
        self.assertEqual("/shared/scruffy", environment["SCRUFFY_ROOT"])
        self.assertEqual("job-1", environment["SCRUFFY_JOB_ID"])
        self.assertEqual(
            "/shared/scruffy/reports/job-1", environment["SCRUFFY_EVENT_DIR"]
        )
        self.assertEqual("gpu-5.cluster", environment["SCRUFFY_NODE"])
        self.assertEqual("PCI_BUS_ID", environment["CUDA_DEVICE_ORDER"])
        self.assertEqual("4,6", environment["CUDA_VISIBLE_DEVICES"])


if __name__ == "__main__":
    unittest.main()
