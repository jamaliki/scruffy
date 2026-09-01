from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scruffy.controller import _initialize_controller
from scruffy.lifecycle import start_job
from scruffy.models import Assignment, NodeInventory, NodeReservation, ResourceRequest
from scruffy.provenance import provenance_files
from scruffy.slurm import AllocationIncarnation
from scruffy.storage import read_immutable_json, utc_now, write_state
from scruffy.submissions import job_from_spec

INVENTORY = (NodeInventory("node", (0,), 2, 2),)


def _job(root: Path) -> dict[str, object]:
    return job_from_spec(
        {
            "v": 1,
            "job_id": "launch",
            "request_id": "request-launch",
            "name": "train",
            "submitted_at": utc_now(),
            "argv": ["true"],
            "cwd": str(root),
            "env": {},
            "resources": ResourceRequest(1, 1, 1, 1).to_dict(),
            "workflow_id": "flow",
            "task_id": "train",
            "needs": [],
            "wait_for": [],
        },
        1,
    )


def _write_queue(root: Path, incarnation: AllocationIncarnation, job: dict[str, object]) -> None:
    write_state(
        root,
        {
            "v": 1,
            "queue_id": "queue",
            "last_seq": 0,
            "journal_generation": 0,
            "journal_offset": 0,
            "allocation": {
                "id": incarnation.slurm_job_id,
                "state": "running",
                "incarnation": incarnation.to_dict(),
            },
            "nodes": {},
            "gpu_health": None,
            "jobs": {"launch": job},
            "next_queue_order": 1,
            "archived_jobs": 0,
            "archived_counts": {},
            "archived_project_counts": {},
            "draining": False,
            "drain_requested": False,
            "launches_paused": False,
            "updated_at": utc_now(),
        },
    )


def _assignment() -> Assignment:
    return Assignment(
        "launch",
        ResourceRequest(1, 1, 1, 1),
        (NodeReservation("node", (0,), 1, 1),),
    )


class LaunchReplayTests(unittest.TestCase):
    def test_same_allocation_id_new_incarnation_rejects_stale_launch_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            old = AllocationIncarnation("allocation", 0, INVENTORY)
            new = AllocationIncarnation("allocation", 1, INVENTORY)
            _write_queue(root, old, _job(root))
            controller = _initialize_controller(
                root=root,
                inventory=INVENTORY,
                launcher="slurm",
                allocation_id="allocation",
                slurm_job_id="allocation",
                allocation_incarnation=old,
                poll_interval=0.1,
                cancel_grace=0,
                gpu_health_mode="off",
            )
            try:
                original = __import__(
                    "scruffy.provenance", fromlist=["write_launch_record"]
                ).write_launch_record

                def crash_after_record(*args: object, **kwargs: object) -> object:
                    original(*args, **kwargs)
                    raise KeyboardInterrupt("crash after stale launch record")

                with (
                    mock.patch(
                        "scruffy.lifecycle.write_launch_record",
                        side_effect=crash_after_record,
                    ),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    start_job(controller, controller.state["jobs"]["launch"], _assignment())
            finally:
                controller.journal.close()

            restarted = _initialize_controller(
                root=root,
                inventory=INVENTORY,
                launcher="slurm",
                allocation_id="allocation",
                slurm_job_id="allocation",
                allocation_incarnation=new,
                poll_interval=0.1,
                cancel_grace=0,
                gpu_health_mode="off",
            )
            try:
                with mock.patch("scruffy.lifecycle.subprocess.Popen") as popen:
                    start_job(restarted, restarted.state["jobs"]["launch"], _assignment())
                popen.assert_not_called()
                self.assertEqual("failed", restarted.state["jobs"]["launch"]["state"])
                launch_file, _ = provenance_files(root, "launch")
                record, _ = read_immutable_json(launch_file)
                self.assertNotEqual(new.fingerprint_sha256, record["allocation_incarnation_sha256"])
            finally:
                restarted.journal.close()

    def test_crash_after_popen_restarts_by_reattaching_recorded_step(self) -> None:
        class Process:
            @property
            def pid(self) -> int:
                raise KeyboardInterrupt("crash after Popen")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            allocation = AllocationIncarnation("allocation", 0, INVENTORY)
            _write_queue(root, allocation, _job(root))
            controller = _initialize_controller(
                root=root,
                inventory=INVENTORY,
                launcher="slurm",
                allocation_id="allocation",
                slurm_job_id="allocation",
                allocation_incarnation=allocation,
                poll_interval=0.1,
                cancel_grace=0,
                gpu_health_mode="off",
            )
            try:
                with (
                    mock.patch("scruffy.lifecycle.subprocess.Popen", return_value=Process()),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    start_job(controller, controller.state["jobs"]["launch"], _assignment())
            finally:
                controller.journal.close()

            restarted = _initialize_controller(
                root=root,
                inventory=INVENTORY,
                launcher="slurm",
                allocation_id="allocation",
                slurm_job_id="allocation",
                allocation_incarnation=allocation,
                poll_interval=0.1,
                cancel_grace=0,
                gpu_health_mode="off",
            )
            try:
                with mock.patch("scruffy.lifecycle.subprocess.Popen") as popen:
                    self.assertIn("launch", restarted.running)
                    launch_file, _ = provenance_files(root, "launch")
                    launch_record, _ = read_immutable_json(launch_file)
                    self.assertTrue(launch_record["job"]["launch_token"].startswith("scruffy-"))
                    assignment = json.loads(
                        (root / "jobs" / "launch" / "assignment.json").read_text()
                    )
                    self.assertEqual(
                        launch_record["job"]["launch_token"], assignment["launch_token"]
                    )
                    self.assertEqual(
                        restarted.state["jobs"]["launch"]["launch_token"],
                        restarted.running["launch"].step_name,
                    )
                    popen.assert_not_called()
            finally:
                restarted.journal.close()


if __name__ == "__main__":
    unittest.main()
