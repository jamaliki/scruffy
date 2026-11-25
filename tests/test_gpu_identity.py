from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

from scruffy._compat import UTC
from scruffy.controller import _initialize_controller
from scruffy.gpu_identity import run_probe
from scruffy.lifecycle import _finish_job, _ingest_gpu_identity
from scruffy.mcp_server import QUIET_EVENT_KINDS
from scruffy.models import Assignment, NodeInventory, NodeReservation, ResourceRequest
from scruffy.runtime import RunningProcess
from scruffy.slurm import AllocationIncarnation
from scruffy.storage import _archived_job, atomic_write_json, read_events, utc_now
from scruffy.summary import job_identity, job_view
from scruffy.worker import execute_assignment

UUID_A = "GPU-5d9dfde5-b05d-eb77-1b88-bf46dab160a4"
UUID_B = "GPU-0c1b2a39-1111-2222-3333-444455556666"
REQUEST = ResourceRequest(1, 2, 1, 1)


def _device(uuid: str, index: int, cuda_index: int) -> dict[str, object]:
    return {
        "cuda_index": cuda_index,
        "uuid": uuid,
        "name": "NVIDIA H100 80GB HBM3",
        "pci_bus_id": f"00000000:{index:02X}:00.0",
        "index": index,
    }


class WorkerIdentityTests(unittest.TestCase):
    def _run(self, probe: dict[str, object]) -> tuple[Path, dict[str, str]]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "queue"
        source = Path(temporary.name) / "assignment.json"
        document = {
            "root": str(root),
            "job_id": "job-1",
            "project_id": "science",
            "launcher": "slurm",
            "launch_token": "scruffy-token",
            "allocation_incarnation_sha256": "a" * 64,
            "runtime_placement_contract": 1,
            "slurm_job_id": "263105",
            "gpus_per_node": 2,
            "argv": ["python", "train.py"],
            "cwd": "/work/job-1",
            "env": {"SCRUFFY_GPU_UUIDS": "spoofed"},
            "assignment": [
                {
                    "node": "gpu-6",
                    "gpu_ids": [4, 6],
                    "runtime_placement": "jobs/job-1/runtime-placement-0.json",
                }
            ],
        }
        source.write_text(json.dumps(document), encoding="utf-8")
        slurm_environment = {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "SLURM_STEP_GPUS": "2,7",
            "SLURM_JOB_ID": "263105",
            "SLURM_STEP_ID": "42",
        }
        with (
            mock.patch.dict(os.environ, slurm_environment, clear=True),
            mock.patch("scruffy.worker.current_node", return_value="gpu-6"),
            mock.patch("scruffy.worker.run_probe", return_value=probe) as run_probe_mock,
            mock.patch("scruffy.worker.os.chdir"),
            mock.patch("scruffy.worker.os.execvpe") as execvpe,
            mock.patch("sys.stderr"),
        ):
            execute_assignment(source)
        # The probe sees the step's own device environment, not submitted values.
        probe_environment = run_probe_mock.call_args.args[0]
        self.assertEqual("0,1", probe_environment["CUDA_VISIBLE_DEVICES"])
        execvpe.assert_called_once()
        return root.resolve() / "jobs/job-1/gpu-identity-0.json", execvpe.call_args.args[2]

    def test_worker_records_identity_and_exports_uuids(self) -> None:
        identity_file, environment = self._run(
            {"devices": [_device(UUID_A, 2, 0), _device(UUID_B, 7, 1)], "error": None}
        )

        self.assertEqual(f"{UUID_A},{UUID_B}", environment["SCRUFFY_GPU_UUIDS"])
        record = json.loads(identity_file.read_text(encoding="utf-8"))
        self.assertEqual("job-1", record["job_id"])
        self.assertEqual("gpu-6", record["node"])
        self.assertEqual("scruffy-token", record["launch_token"])
        self.assertEqual("42", record["slurm_step_id"])
        self.assertEqual(["2", "7"], record["slurm_step_gpus"])
        self.assertIsNone(record["error"])
        self.assertEqual([UUID_A, UUID_B], [item["uuid"] for item in record["devices"]])
        self.assertEqual([2, 7], [item["index"] for item in record["devices"]])

    def test_probe_failure_is_recorded_and_never_blocks_the_job(self) -> None:
        identity_file, environment = self._run(
            {"devices": [_device(UUID_A, 2, 0)], "error": None}
        )

        self.assertNotIn("SCRUFFY_GPU_UUIDS", environment)
        record = json.loads(identity_file.read_text(encoding="utf-8"))
        self.assertIn("found 1 devices", record["error"])

        identity_file, environment = self._run(
            {"devices": [], "error": "load_driver: missing"}
        )
        self.assertNotIn("SCRUFFY_GPU_UUIDS", environment)
        self.assertEqual(
            "load_driver: missing",
            json.loads(identity_file.read_text(encoding="utf-8"))["error"],
        )


class ProbeRunnerTests(unittest.TestCase):
    def test_run_probe_parses_the_child_document(self) -> None:
        completed = subprocess.CompletedProcess(
            [], 0, stdout=json.dumps({"devices": [_device(UUID_A, 3, 0)], "error": None})
        )
        with mock.patch("scruffy.gpu_identity.subprocess.run", return_value=completed) as run:
            result = run_probe({"CUDA_VISIBLE_DEVICES": "0"})

        self.assertEqual(UUID_A, result["devices"][0]["uuid"])
        self.assertEqual({"CUDA_VISIBLE_DEVICES": "0"}, run.call_args.kwargs["env"])

    def test_run_probe_reports_failures_without_raising(self) -> None:
        crashed = subprocess.CompletedProcess([], 1, stdout="", stderr="Traceback: boom")
        with mock.patch("scruffy.gpu_identity.subprocess.run", return_value=crashed):
            self.assertIn("boom", run_probe({})["error"])
        with mock.patch(
            "scruffy.gpu_identity.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["probe"], 30),
        ):
            result = run_probe({})
        self.assertEqual([], result["devices"])
        self.assertIn("probe", result["error"])


class ControllerIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"
        inventory = (NodeInventory("gpu-6", (0, 1), 4, 4),)
        self.controller = _initialize_controller(
            root=self.root,
            inventory=inventory,
            launcher="slurm",
            allocation_id="240292",
            slurm_job_id="240292",
            allocation_incarnation=AllocationIncarnation("240292", 0, inventory),
            poll_interval=0.1,
            cancel_grace=0,
            gpu_health_mode="off",
        )
        self.addCleanup(lambda: self.controller.journal.close())

    def _job(self, *, started_at: str | None = None) -> dict[str, object]:
        job = {
            "id": "job-1",
            "name": "timing",
            "state": "running",
            "submitted_at": utc_now(),
            "started_at": started_at or utc_now(),
            "queue_order": 1,
            "argv": ["true"],
            "cwd": "/tmp",
            "env": {},
            "request": REQUEST.to_dict(),
            "assignment": Assignment(
                "job-1", REQUEST, (NodeReservation("gpu-6", (0, 1), 1, 1),)
            ).to_dict(),
            "launch_token": "scruffy-token",
            "slurm_step_id": "240292.7",
            "runtime_placement_contract": 1,
            "runtime_placement_files": ["jobs/job-1/runtime-placement-0.json"],
        }
        self.controller.state["jobs"]["job-1"] = job
        return job

    def _write_identity(self, **updates: object) -> None:
        record = {
            "schema": 1,
            "job_id": "job-1",
            "node": "gpu-6",
            "launch_token": "scruffy-token",
            "slurm_job_id": "240292",
            "slurm_step_id": "7",
            "slurm_step_gpus": ["2", "7"],
            "cuda_visible_devices": ["0", "1"],
            "recorded_at": utc_now(),
            "devices": [_device(UUID_A, 2, 0), _device(UUID_B, 7, 1)],
            "error": None,
        }
        record.update(updates)
        atomic_write_json(self.root / "jobs/job-1/gpu-identity-0.json", record)

    def _identified_events(self) -> list[dict[str, object]]:
        return [
            event for event in read_events(self.root) if event["kind"] == "job.gpu_identified"
        ]

    def test_running_job_records_identity_once(self) -> None:
        job = self._job()
        running = RunningProcess(mock.Mock(), "scruffy-token")

        _ingest_gpu_identity(self.controller, job, running)
        self.assertNotIn("gpu_devices", job)

        self._write_identity()
        running.gpu_identity_checked_at = None
        _ingest_gpu_identity(self.controller, job, running)
        running.gpu_identity_checked_at = None
        _ingest_gpu_identity(self.controller, job, running)

        self.assertEqual(
            [("gpu-6", 2, UUID_A), ("gpu-6", 7, UUID_B)],
            [(item["node"], item["index"], item["uuid"]) for item in job["gpu_devices"]],
        )
        self.assertNotIn("gpu_identity_error", job)
        events = self._identified_events()
        self.assertEqual(1, len(events))
        self.assertEqual(UUID_A, events[0]["job"]["gpu_devices"][0]["uuid"])
        self.assertIn("job.gpu_identified", QUIET_EVENT_KINDS)

        compact = job_identity(job, include_project=False, include_elapsed=True)
        self.assertEqual(
            [
                {"node": "gpu-6", "index": 2, "uuid": UUID_A},
                {"node": "gpu-6", "index": 7, "uuid": UUID_B},
            ],
            compact["gpus"],
        )
        self.assertNotIn(
            "gpus", job_identity(job, include_project=False, include_elapsed=False)
        )
        self.assertEqual(job["gpu_devices"], job_view(job)["gpu_devices"])

    def test_record_from_another_launch_is_an_error_not_identity(self) -> None:
        job = self._job()
        self._write_identity(launch_token="scruffy-other")

        _ingest_gpu_identity(self.controller, job, RunningProcess(mock.Mock(), None))

        self.assertEqual([], job["gpu_devices"])
        self.assertIn("differs from the launch", job["gpu_identity_error"])

    def test_probe_error_is_kept_with_the_devices_found(self) -> None:
        job = self._job()
        self._write_identity(devices=[_device(UUID_A, 2, 0)], error="probe found 1 devices")

        _ingest_gpu_identity(self.controller, job, RunningProcess(mock.Mock(), None))

        self.assertEqual([UUID_A], [item["uuid"] for item in job["gpu_devices"]])
        self.assertEqual("gpu-6: probe found 1 devices", job["gpu_identity_error"])

    def test_old_jobs_are_left_alone_until_they_finish(self) -> None:
        started = (datetime.now(UTC) - timedelta(hours=1)).isoformat(timespec="milliseconds")
        job = self._job(started_at=started)
        self._write_identity()

        _ingest_gpu_identity(self.controller, job, RunningProcess(mock.Mock(), None))
        self.assertNotIn("gpu_devices", job)

        _finish_job(self.controller, "job-1", RunningProcess(mock.Mock(), None), 0)

        self.assertEqual(2, len(job["gpu_devices"]))
        terminal = [
            event
            for event in read_events(self.root)
            if event["kind"] == f"job.{job['state']}"
        ]
        self.assertEqual(UUID_A, terminal[-1]["job"]["gpu_devices"][0]["uuid"])
        self.assertEqual(job["gpu_devices"], _archived_job(job)["gpu_devices"])

    def test_cpu_only_and_local_jobs_have_no_identity(self) -> None:
        job = self._job()
        cpu = ResourceRequest(1, 0, 1, 1)
        job["request"] = cpu.to_dict()
        job["assignment"] = Assignment(
            "job-1", cpu, (NodeReservation("gpu-6", (), 1, 1),)
        ).to_dict()
        self._write_identity(devices=[])

        _ingest_gpu_identity(self.controller, job, RunningProcess(mock.Mock(), None))

        self.assertNotIn("gpu_devices", job)


if __name__ == "__main__":
    unittest.main()
