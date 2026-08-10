from __future__ import annotations

import json
import os
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
            gpus_per_node=2,
            cpus_per_node=28,
            memory_gb_per_node=256,
            wait_seconds=45,
        )

        self.assertEqual(
            [
                "srun",
                "--jobid=240292",
                "--job-name=scruffy-launch-token",
                "--exact",
                "--nodes=2",
                "--nodelist=gpu-3,gpu-5",
                "--ntasks=2",
                "--ntasks-per-node=1",
                "--gpus-per-node=2",
                "--cpus-per-task=28",
                "--cpu-bind=none",
                "--mem=256G",
                "--kill-on-bad-exit=1",
                "--wait=45",
                "--wait-for-children",
                "--export=ALL",
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
                gpus_per_node=1,
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
            gpus_per_node=1,
            cpus_per_node=1,
            memory_gb_per_node=1,
        )

        self.assertIn("--wait=0", argv)
        self.assertIn("--wait-for-children", argv)

    def test_partial_step_requests_gpu_tres_without_overlap(self) -> None:
        argv = build_srun_argv(
            slurm_job_id="240292",
            name="scruffy-token",
            assignment_file=Path("assignment.json"),
            stdout_file=Path("stdout.log"),
            stderr_file=Path("stderr.log"),
            node_names=["gpu-3"],
            gpus_per_node=1,
            cpus_per_node=14,
            memory_gb_per_node=128,
        )

        self.assertIn("--gpus-per-node=1", argv)
        self.assertNotIn("--overlap", argv)
        self.assertIn("--exact", argv)

    def test_full_node_step_requests_all_gpu_tres(self) -> None:
        argv = build_srun_argv(
            slurm_job_id="240292",
            name="scruffy-token",
            assignment_file=Path("assignment.json"),
            stdout_file=Path("stdout.log"),
            stderr_file=Path("stderr.log"),
            node_names=["gpu-3"],
            gpus_per_node=8,
            cpus_per_node=112,
            memory_gb_per_node=1024,
        )

        self.assertIn("--gpus-per-node=8", argv)
        self.assertIn("--cpus-per-task=112", argv)
        self.assertIn("--mem=1024G", argv)

    def test_srun_does_not_inherit_controller_placement(self) -> None:
        source = {
            "PATH": "/usr/bin",
            "SCRUFFY_NODE": "stale-node",
            "SLURMD_NODENAME": "controller-node",
            "SRUN_CONTAINER": "/stale/controller.sqsh",
            "SRUN_EXPORT_ENV": "NONE",
            "SLURM_JOB_ID": "263105",
            "SLURM_JOBID": "263105",
            "SLURM_JOB_NODELIST": "gpu-[0,5,8,13]",
            "SLURM_JOB_NODES": "gpu-[0,5,8,13]",
            "SLURM_JOB_NUM_NODES": "4",
            "SLURM_JOB_CPUS_PER_NODE": "112(x4)",
            "SLURM_JOB_GPUS": "0,1,2,3,4,5,6,7",
            "SLURM_JOB_END_TIME": "1786150800",
            "SLURM_CLUSTER_NAME": "tokyo",
            "SLURM_CONF": "/etc/slurm/slurm.conf",
            "SLURM_SUBMIT_DIR": "/shared/run",
            "SLURM_SUBMIT_HOST": "login-0",
            "SLURM_CPUS_PER_TASK": "112",
            "SLURM_CPUS_PER_GPU": "14",
            "SLURM_CPUS_ON_NODE": "112",
            "SLURM_CPU_FREQ_REQ": "high",
            "SLURM_CORE_SPEC": "2",
            "SLURM_HINT": "nomultithread",
            "SLURM_THREAD_SPEC": "2",
            "SLURM_GPUS": "8",
            "SLURM_GPUS_PER_TASK": "1",
            "SLURM_GPUS_ON_NODE": "8",
            "SLURM_GPU_FREQ": "high",
            "SLURM_GRES": "gpu:h100:8",
            "SLURM_GRES_FLAGS": "enforce-binding",
            "SLURM_SHARDS_ON_NODE": "8",
            "SLURM_MEM_PER_CPU": "1024",
            "SLURM_MEM_PER_GPU": "128G",
            "SLURM_MEM_PER_NODE": "1T",
            "SLURM_STEP_ID": "3",
            "SLURM_STEP_NODELIST": "gpu-13",
            "SLURM_STEP_NUM_TASKS": "1",
            "SLURM_STEPID": "3",
            "SLURM_PROCID": "0",
            "SLURM_LOCALID": "0",
            "SLURM_NODEID": "0",
            "SLURM_NTASKS": "1",
            "SLURM_NTASKS_PER_NODE": "1",
            "SLURM_NPROCS": "1",
            "SLURM_TASKS_PER_NODE": "1",
            "SLURM_THREADS_PER_CORE": "1",
            "SLURM_CPU_BIND": "quiet,mask_cpu:0x1",
            "SLURM_CPU_BIND_LIST": "0x1",
            "SLURM_CPU_BIND_TYPE": "mask_cpu:",
            "SLURM_CPU_BIND_VERBOSE": "quiet",
            "SLURM_MEM_BIND": "local",
            "SLURM_GPU_BIND": "closest",
            "SLURM_TRES_BIND": "gres/gpu:per_task:1",
            "SLURM_TRES_PER_TASK": "cpu=112",
            "SLURM_TRES_FREQ": "gpu=high",
            "SLURM_DISTRIBUTION": "block",
            "SLURM_CONSTRAINT": "h100",
            "SLURM_EXCLUSIVE": "1",
            "SLURM_OVERCOMMIT": "1",
            "SLURM_SPREAD_JOB": "1",
            "SLURM_EXPORT_ENV": "NONE",
            "SLURM_GTIDS": "0",
            "SLURM_TASK_PID": "12345",
            "SLURM_TASK_PROLOG": "/stale/task-prolog",
            "SLURM_SRUN_COMM_HOST": "10.0.0.1",
            "SLURM_CONTAINER": "/stale/controller.sqsh",
            "SLURM_JOB_UNRECOGNIZED": "unsafe-by-default",
        }
        original = dict(source)

        environment = build_srun_environment(source)

        self.assertEqual(original, source)
        self.assertEqual(
            {
                "PATH": "/usr/bin",
                "SLURM_JOB_ID": "263105",
                "SLURM_JOBID": "263105",
                "SLURM_JOB_NODELIST": "gpu-[0,5,8,13]",
                "SLURM_JOB_NODES": "gpu-[0,5,8,13]",
                "SLURM_JOB_NUM_NODES": "4",
                "SLURM_JOB_CPUS_PER_NODE": "112(x4)",
                "SLURM_JOB_GPUS": "0,1,2,3,4,5,6,7",
                "SLURM_JOB_END_TIME": "1786150800",
                "SLURM_CLUSTER_NAME": "tokyo",
                "SLURM_CONF": "/etc/slurm/slurm.conf",
                "SLURM_SUBMIT_DIR": "/shared/run",
                "SLURM_SUBMIT_HOST": "login-0",
            },
            environment,
        )

    def test_srun_strips_exact_v20_cpu_environment(self) -> None:
        mask = "0x0000000FFFFFFF0000000FFFFFFF0000000FFFFFFF0000000FFFFFFF"

        environment = build_srun_environment(
            {
                "SLURM_JOB_ID": "263105",
                "SLURM_CPUS_PER_TASK": "112",
                "SLURM_CPU_BIND": f"quiet,mask_cpu:{mask}",
                "SLURM_CPU_BIND_LIST": mask,
                "SLURM_CPU_BIND_TYPE": "mask_cpu:",
                "SLURM_CPU_BIND_VERBOSE": "quiet",
            }
        )

        self.assertEqual({"SLURM_JOB_ID": "263105"}, environment)

    def test_srun_sanitizes_the_default_process_environment(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "PATH": "/usr/bin",
                "SLURM_JOB_ID": "263105",
                "SLURM_GRES": "gpu:h100:8",
                "SLURM_GPUS_ON_NODE": "8",
                "SRUN_EXPORT_ENV": "NONE",
            },
            clear=True,
        ):
            environment = build_srun_environment()

        self.assertEqual(
            {"PATH": "/usr/bin", "SLURM_JOB_ID": "263105"}, environment
        )


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

    def test_slurm_worker_preserves_selected_gpu_mapping_and_records_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            source = Path(temporary) / "assignment.json"
            document = {
                "root": str(root),
                "job_id": "job-1",
                "project_id": "science",
                "launcher": "slurm",
                "slurm_job_id": "263105",
                "gpus_per_node": 2,
                "argv": ["python", "train.py"],
                "cwd": "/work/job-1",
                "env": {
                    "CUDA_VISIBLE_DEVICES": "99",
                    "CUDA_DEVICE_ORDER": "FASTEST_FIRST",
                    "SLURM_STEP_GPUS": "99",
                    "SLURM_STEP_ID": "spoofed",
                    "SLURM_JOB_GPUS": "99",
                    "SCRUFFY_GPU_IDS": "99",
                },
                "assignment": [
                    {
                        "node": "gpu-5.cluster",
                        "gpu_ids": [4, 6],
                        "runtime_placement": "jobs/job-1/runtime-placement-0.json",
                    }
                ],
            }
            source.write_text(json.dumps(document), encoding="utf-8")
            slurm_environment = {
                "CUDA_VISIBLE_DEVICES": "0,1",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "SLURM_JOB_GPUS": "0,1,2,3,4,5,6,7",
                "SLURM_STEP_GPUS": "2,7",
                "SLURM_JOB_ID": "263105",
                "SLURM_STEP_ID": "42",
            }
            with (
                mock.patch.dict(os.environ, slurm_environment, clear=True),
                mock.patch("scruffy.worker.current_node", return_value="gpu-5.cluster"),
                mock.patch("scruffy.worker.os.chdir"),
                mock.patch("scruffy.worker.os.execvpe") as execvpe,
            ):
                execute_assignment(source)

            environment = execvpe.call_args.args[2]
            self.assertEqual("0,1", environment["CUDA_VISIBLE_DEVICES"])
            self.assertEqual("2,7", environment["SLURM_STEP_GPUS"])
            self.assertEqual("0,1,2,3,4,5,6,7", environment["SLURM_JOB_GPUS"])
            self.assertEqual("2,7", environment["SCRUFFY_GPU_IDS"])
            self.assertEqual("4,6", environment["SCRUFFY_RESERVED_GPU_IDS"])
            self.assertEqual("263105", environment["SCRUFFY_SLURM_JOB_ID"])
            self.assertEqual("42", environment["SCRUFFY_SLURM_STEP_ID"])
            placement_file = root.resolve() / "jobs/job-1/runtime-placement-0.json"
            self.assertEqual(str(placement_file), environment["SCRUFFY_RUNTIME_PLACEMENT"])
            self.assertEqual(
                {
                    "schema": 1,
                    "job_id": "job-1",
                    "node": "gpu-5.cluster",
                    "requested_gpus": 2,
                    "ledger_gpu_ids": [4, 6],
                    "slurm_job_id": "263105",
                    "slurm_step_id": "42",
                    "slurm_step_gpus": ["2", "7"],
                    "cuda_visible_devices": ["0", "1"],
                    "cuda_device_order": "PCI_BUS_ID",
                },
                json.loads(placement_file.read_text(encoding="utf-8")),
            )

    def test_slurm_worker_rejects_missing_or_wrong_gpu_mapping(self) -> None:
        document = {
            "root": "/shared/scruffy",
            "job_id": "job-1",
            "launcher": "slurm",
            "slurm_job_id": "263105",
            "gpus_per_node": 2,
            "argv": ["true"],
            "cwd": "/tmp",
            "env": {},
            "assignment": [
                {
                    "node": "gpu-5",
                    "gpu_ids": [4, 6],
                    "runtime_placement": "jobs/job-1/runtime-placement-0.json",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "assignment.json"
            source.write_text(json.dumps(document), encoding="utf-8")
            for inherited in (
                {
                    "SLURM_JOB_ID": "263105",
                    "SLURM_STEP_ID": "42",
                    "SLURM_STEP_GPUS": "2,7",
                },
                {
                    "SLURM_JOB_ID": "263105",
                    "SLURM_STEP_ID": "42",
                    "SLURM_STEP_GPUS": "2,7",
                    "CUDA_VISIBLE_DEVICES": "0",
                },
            ):
                with (
                    self.subTest(inherited=inherited),
                    mock.patch.dict(os.environ, inherited, clear=True),
                    mock.patch("scruffy.worker.current_node", return_value="gpu-5"),
                    mock.patch("scruffy.worker.os.execvpe") as execvpe,
                ):
                    with self.assertRaisesRegex(
                        ValueError, "CUDA_VISIBLE_DEVICES|GPU mapping"
                    ):
                        execute_assignment(source)
                    execvpe.assert_not_called()

    def test_slurm_worker_rejects_a_different_outer_allocation(self) -> None:
        document = {
            "root": "/shared/scruffy",
            "job_id": "job-1",
            "launcher": "slurm",
            "slurm_job_id": "263105",
            "gpus_per_node": 1,
            "argv": ["true"],
            "cwd": "/tmp",
            "env": {},
            "assignment": [
                {
                    "node": "gpu-5",
                    "gpu_ids": [4],
                    "runtime_placement": "jobs/job-1/runtime-placement-0.json",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "assignment.json"
            source.write_text(json.dumps(document), encoding="utf-8")
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "SLURM_JOB_ID": "999999",
                        "SLURM_STEP_ID": "42",
                        "SLURM_STEP_GPUS": "2",
                        "CUDA_VISIBLE_DEVICES": "0",
                    },
                    clear=True,
                ),
                mock.patch("scruffy.worker.current_node", return_value="gpu-5"),
                mock.patch("scruffy.worker.os.execvpe") as execvpe,
            ):
                with self.assertRaisesRegex(ValueError, "allocation differs"):
                    execute_assignment(source)
                execvpe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
