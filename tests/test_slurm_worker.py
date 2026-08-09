from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scruffy.slurm import (
    build_srun_argv,
    build_srun_environment,
    cancel_step,
    completed_step,
    discover_slurm_inventory,
    live_steps,
    load_inventory,
)
from scruffy.worker import current_node, execute_assignment, find_node_assignment


class SlurmArgumentTests(unittest.TestCase):
    def test_multi_node_srun_argv_is_exact(self) -> None:
        assignment_file = Path("/shared/scruffy/jobs/job-1/assignment.json")

        argv = build_srun_argv(
            slurm_job_id="240292",
            name="scruffy-launch-token",
            assignment_file=assignment_file,
            stdout_file=Path("/shared/scruffy/jobs/job-1/stdout.log"),
            stderr_file=Path("/shared/scruffy/jobs/job-1/stderr.log"),
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
                "--cpu-bind=none",
                "--mem=256G",
                "--kill-on-bad-exit=1",
                "--wait=45",
                "--wait-for-children",
                "--label",
                "--output=/shared/scruffy/jobs/job-1/stdout.log",
                "--error=/shared/scruffy/jobs/job-1/stderr.log",
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
                stdout_file=Path("stdout.log"),
                stderr_file=Path("stderr.log"),
                node_names=["gpu-3"],
                cpus_per_node=1,
                memory_gb_per_node=1,
            )

    def test_srun_default_waits_for_descendants_without_rank_timeout(self) -> None:
        argv = build_srun_argv(
            slurm_job_id="240292",
            name="scruffy-token",
            assignment_file=Path("assignment.json"),
            stdout_file=Path("stdout.log"),
            stderr_file=Path("stderr.log"),
            node_names=["gpu-3"],
            cpus_per_node=1,
            memory_gb_per_node=1,
        )

        self.assertIn("--wait=0", argv)
        self.assertIn("--wait-for-children", argv)

    def test_srun_does_not_inherit_controller_placement(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "SCRUFFY_NODE": "stale-node",
                "SLURM_CPU_BIND": "verbose,mask_cpu:0x1",
                "SLURM_CPU_BIND_LIST": "0x1",
                "SLURM_CPU_BIND_TYPE": "mask_cpu",
                "SLURM_CPU_BIND_VERBOSE": "verbose",
                "KEPT": "yes",
            },
            clear=True,
        ):
            environment = build_srun_environment()

        self.assertNotIn("SCRUFFY_NODE", environment)
        self.assertFalse(
            any(name.startswith("SLURM_CPU_BIND") for name in environment)
        )
        self.assertEqual("yes", environment["KEPT"])


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

    def test_discovers_uniform_capacity_from_the_slurm_allocation(self) -> None:
        allocation = mock.Mock(
            stdout=json.dumps(
                {
                    "errors": [],
                    "jobs": [
                        {
                            "nodes": "gpu-[0,5,8,13]",
                            "tres_alloc_str": (
                                "cpu=448,mem=8160440M,node=4,gres/gpu=32"
                            ),
                        }
                    ],
                }
            )
        )
        hostnames = mock.Mock(stdout="gpu-0\ngpu-5\ngpu-8\ngpu-13\n")
        with mock.patch("scruffy.slurm.subprocess.run", side_effect=[allocation, hostnames]):
            inventory = discover_slurm_inventory(slurm_job_id="263105")

        self.assertEqual({"gpu-0", "gpu-5", "gpu-8", "gpu-13"}, inventory.keys())
        for node in inventory.values():
            self.assertEqual(tuple(range(8)), node.gpu_ids)
            self.assertEqual(112, node.cpus)
            self.assertEqual(1992, node.memory_gb)

    def test_explicit_resource_values_are_caps_on_discovered_capacity(self) -> None:
        allocation = mock.Mock(
            stdout=json.dumps(
                {
                    "errors": [],
                    "jobs": [
                        {
                            "nodes": "gpu-0",
                            "tres_alloc_str": "cpu=112,mem=2040110M,gres/gpu:h100=8",
                        }
                    ],
                }
            )
        )
        hostnames = mock.Mock(stdout="gpu-0\n")
        with mock.patch("scruffy.slurm.subprocess.run", side_effect=[allocation, hostnames]):
            inventory = discover_slurm_inventory(
                slurm_job_id="263105",
                gpus_per_node=4,
                cpus_per_node=56,
                memory_gb_per_node=512,
            )

        node = inventory["gpu-0"]
        self.assertEqual(tuple(range(4)), node.gpu_ids)
        self.assertEqual(56, node.cpus)
        self.assertEqual(512, node.memory_gb)

    def test_resource_cap_cannot_exceed_the_allocation(self) -> None:
        allocation = mock.Mock(
            stdout=json.dumps(
                {
                    "errors": [],
                    "jobs": [
                        {
                            "nodes": "gpu-0",
                            "tres_alloc_str": "cpu=112,mem=2040110M,gres/gpu=8",
                        }
                    ],
                }
            )
        )
        hostnames = mock.Mock(stdout="gpu-0\n")
        with (
            mock.patch(
                "scruffy.slurm.subprocess.run", side_effect=[allocation, hostnames]
            ),
            self.assertRaisesRegex(ValueError, "exceeds the Slurm allocation"),
        ):
            discover_slurm_inventory(slurm_job_id="263105", gpus_per_node=9)


class SlurmReconciliationTests(unittest.TestCase):
    def test_completed_step_reads_the_exact_accounting_row(self) -> None:
        result = mock.Mock(
            stdout=(
                "240292|RUNNING|0:0\n"
                "240292.17|FAILED|2:0\n"
                "240292.17.0|FAILED|2:0\n"
            )
        )
        with mock.patch("scruffy.slurm.subprocess.run", return_value=result) as run:
            step = completed_step("240292.17")

        self.assertIsNotNone(step)
        self.assertEqual("FAILED", step.state)
        self.assertEqual(2, step.returncode)
        run.assert_called_once_with(
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--jobs=240292.17",
                "--format=JobIDRaw,State,ExitCode",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )

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
        for unsafe in ("240292", "240292.batch", "240292.١٧", "999.17", ""):
            with self.assertRaisesRegex(ValueError, "unsafe Slurm step ID"):
                cancel_step("240292", unsafe)


class WorkerPlacementTests(unittest.TestCase):
    def test_controller_node_identity_wins_inside_an_existing_slurm_step(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"SCRUFFY_NODE": "inventory-node", "SLURMD_NODENAME": "outer-node"},
            clear=True,
        ):
            self.assertEqual("inventory-node", current_node())

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

    def test_worker_exec_appends_directly_to_job_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            stdout = directory / "stdout.log"
            stderr = directory / "stderr.log"
            stdout.write_text("existing stdout\n", encoding="utf-8")
            stderr.write_text("existing stderr\n", encoding="utf-8")
            source = directory / "assignment.json"
            source.write_text(
                json.dumps(
                    {
                        "root": str(directory),
                        "job_id": "job-1",
                        "project_id": "tests",
                        "argv": [
                            sys.executable,
                            "-c",
                            (
                                "import os; "
                                "os.write(1, b'worker stdout\\n'); "
                                "os.write(2, b'worker stderr\\n')"
                            ),
                        ],
                        "cwd": str(directory),
                        "env": {},
                        "assignment": [{"node": "gpu-5", "gpu_ids": []}],
                        "logs": {
                            "stdout": str(stdout),
                            "stderr": str(stderr),
                        },
                    }
                ),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["SCRUFFY_NODE"] = "gpu-5"

            result = subprocess.run(
                [sys.executable, "-m", "scruffy.worker", str(source)],
                check=False,
                capture_output=True,
                env=environment,
                timeout=10,
            )

            self.assertEqual(0, result.returncode, result.stderr.decode())
            self.assertEqual(b"", result.stdout)
            self.assertEqual(b"", result.stderr)
            self.assertEqual(
                "existing stdout\nworker stdout\n",
                stdout.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                "existing stderr\nworker stderr\n",
                stderr.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
