from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scruffy.client import publish_event
from scruffy.controller import (
    _ingest_reports,
    _initialize_controller,
    _satisfy_from_current_artifacts,
)
from scruffy.lifecycle import start_job
from scruffy.models import Assignment, NodeInventory, NodeReservation, ResourceRequest
from scruffy.slurm import AllocationIncarnation
from scruffy.storage import (
    archive_terminal_job,
    find_archived_job,
    utc_now,
    write_state,
)
from scruffy.submissions import job_from_spec

INVENTORY = (NodeInventory("node", (0,), 2, 2),)
POLICY = {
    "max_attempts": 3,
    "retry_on": ["allocation_replaced", "allocation_incarnation_changed"],
    "evacuation": {"signal": "USR1", "grace_seconds": 600},
}


def _job(root: Path, job_id: str, task_id: str, workflow_id: str = "flow") -> dict[str, object]:
    return job_from_spec(
        {
            "v": 1,
            "job_id": job_id,
            "request_id": f"request-{job_id}",
            "name": task_id,
            "submitted_at": utc_now(),
            "argv": ["true"],
            "cwd": str(root),
            "env": {},
            "resources": ResourceRequest(1, 1, 1, 1).to_dict(),
            "workflow_id": workflow_id,
            "task_id": task_id,
            "needs": [],
            "wait_for": [],
        },
        1,
    )


def _write_queue(root: Path, allocation: AllocationIncarnation, jobs: dict[str, object]) -> None:
    write_state(
        root,
        {
            "v": 1,
            "queue_id": "queue",
            "last_seq": 0,
            "journal_generation": 0,
            "journal_offset": 0,
            "allocation": {
                "id": allocation.slurm_job_id,
                "state": "running",
                "incarnation": allocation.to_dict(),
            },
            "nodes": {},
            "gpu_health": None,
            "jobs": jobs,
            "next_queue_order": max(
                (int(job.get("queue_order", 0)) for job in jobs.values()), default=0
            ),
            "archived_jobs": 0,
            "archived_counts": {},
            "archived_project_counts": {},
            "draining": False,
            "drain_requested": False,
            "launches_paused": False,
            "updated_at": utc_now(),
        },
    )


class RecoveryCrashWindowTests(unittest.TestCase):
    def test_loss_result_written_before_job_lost_is_replay_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            old = _job(root, "old", "train")
            old.update(
                state="running",
                recovery=POLICY,
                launch_token="token",
                assignment=Assignment(
                    "old", ResourceRequest(1, 1, 1, 1), (NodeReservation("node", (0,), 1, 1),)
                ).to_dict(),
            )
            old_allocation = AllocationIncarnation("old", 0, INVENTORY)
            new_allocation = AllocationIncarnation("new", 0, INVENTORY)
            _write_queue(root, old_allocation, {"old": old})
            original = __import__("scruffy.controller", fromlist=["write_result_record"]).write_result_record

            def crash_after_result(*args: object, **kwargs: object) -> object:
                original(*args, **kwargs)
                raise KeyboardInterrupt("crash before job.lost")

            with (
                mock.patch(
                    "scruffy.controller.write_result_record",
                    side_effect=crash_after_result,
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                _initialize_controller(
                    root=root, inventory=INVENTORY, launcher="slurm", allocation_id="new",
                    slurm_job_id="new", allocation_incarnation=new_allocation,
                    poll_interval=0.1, cancel_grace=0, gpu_health_mode="off",
                )
            recovered = _initialize_controller(
                root=root, inventory=INVENTORY, launcher="slurm", allocation_id="new",
                slurm_job_id="new", allocation_incarnation=new_allocation,
                poll_interval=0.1, cancel_grace=0, gpu_health_mode="off",
            )
            try:
                successors = [
                    job for job in recovered.state["jobs"].values()
                    if job.get("predecessor_job_id") == "old"
                ]
                self.assertEqual(1, len(successors))
                self.assertEqual(2, len(recovered.state["jobs"]))
            finally:
                recovered.journal.close()

    def test_launch_record_crash_reuses_identity_without_duplicate_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            allocation = AllocationIncarnation("allocation", 0, INVENTORY)
            job = _job(root, "launch", "train")
            job["assignment"] = None
            _write_queue(root, allocation, {"launch": job})
            controller = _initialize_controller(
                root=root, inventory=INVENTORY, launcher="slurm", allocation_id="allocation",
                slurm_job_id="allocation", allocation_incarnation=allocation,
                poll_interval=0.1, cancel_grace=0, gpu_health_mode="off",
            )
            assignment = Assignment(
                "launch", ResourceRequest(1, 1, 1, 1), (NodeReservation("node", (0,), 1, 1),)
            )
            original = __import__("scruffy.lifecycle", fromlist=["write_launch_record"]).write_launch_record

            def crash_after_launch(*args: object, **kwargs: object) -> object:
                original(*args, **kwargs)
                raise KeyboardInterrupt("crash before job.starting")

            try:
                with (
                    mock.patch(
                        "scruffy.lifecycle.write_launch_record",
                        side_effect=crash_after_launch,
                    ),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    start_job(controller, job, assignment)
            finally:
                controller.journal.close()
            retry = _initialize_controller(
                root=root, inventory=INVENTORY, launcher="slurm", allocation_id="allocation",
                slurm_job_id="allocation", allocation_incarnation=allocation,
                poll_interval=0.1, cancel_grace=0, gpu_health_mode="off",
            )
            process = mock.Mock(pid=123)
            try:
                with mock.patch("scruffy.lifecycle.subprocess.Popen", return_value=process) as popen:
                    start_job(retry, retry.state["jobs"]["launch"], assignment)
                popen.assert_called_once()
                self.assertEqual("starting", retry.state["jobs"]["launch"]["state"])
            finally:
                retry.journal.close()

    def test_archived_artifact_report_updates_waiter_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            allocation = AllocationIncarnation("allocation", 0, INVENTORY)
            producer = _job(root, "producer", "prepare")
            producer.update(state="succeeded", finished_at=utc_now())
            archive_terminal_job(root, producer)
            consumer = _job(root, "consumer", "infer")
            consumer["state"] = "blocked"
            consumer["wait_for"] = [
                {"kind": "artifact", "task_id": "prepare", "artifact_id": "first"}
            ]
            _write_queue(root, allocation, {"consumer": consumer})
            controller = _initialize_controller(
                root=root, inventory=INVENTORY, launcher="local", allocation_id="allocation",
                slurm_job_id=None, poll_interval=0.1, cancel_grace=0, gpu_health_mode="off",
            )
            publication = {
                "v": 1,
                "artifact_id": "first",
                "path": str(root / "first"),
                "size_bytes": 1,
                "sha256": "a" * 64,
                "manifest_path": str(root / "first.ready.json"),
            }
            try:
                publish_event(
                    root,
                    job_id="producer",
                    event_id="delayed-publication",
                    kind="workload.artifact",
                    data={"artifact_type": "checkpoint", "publication": publication},
                )
                _ingest_reports(controller)
                current_consumer = controller.state["jobs"]["consumer"]
                self.assertEqual(
                    "producer", current_consumer["condition_satisfactions"][0]["producer_job_id"]
                )
                cold = find_archived_job(root, "producer")
                self.assertEqual(1, len(cold["artifact_condition_evidence"]))
            finally:
                controller.journal.close()
            restarted = _initialize_controller(
                root=root, inventory=INVENTORY, launcher="local", allocation_id="allocation",
                slurm_job_id=None, poll_interval=0.1, cancel_grace=0, gpu_health_mode="off",
            )
            try:
                self.assertEqual(
                    "producer",
                    restarted.state["jobs"]["consumer"]["condition_satisfactions"][0]["producer_job_id"],
                )
            finally:
                restarted.journal.close()

    def test_exact_condition_evidence_survives_more_than_eight_publications(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            allocation = AllocationIncarnation("allocation", 0, INVENTORY)
            producer = _job(root, "producer", "prepare")
            consumer = _job(root, "consumer", "infer")
            consumer["state"] = "blocked"
            consumer["wait_for"] = [
                {"kind": "artifact", "task_id": "prepare", "artifact_id": "artifact-0"}
            ]
            _write_queue(root, allocation, {"producer": producer, "consumer": consumer})
            controller = _initialize_controller(
                root=root, inventory=INVENTORY, launcher="local", allocation_id="allocation",
                slurm_job_id=None, poll_interval=0.1, cancel_grace=0, gpu_health_mode="off",
            )
            try:
                for index in range(9):
                    publication = {
                        "v": 1,
                        "artifact_id": f"artifact-{index}",
                        "path": str(root / f"artifact-{index}"),
                        "size_bytes": 1,
                        "sha256": f"{index + 1:064x}",
                        "manifest_path": str(root / f"artifact-{index}.ready.json"),
                    }
                    publish_event(
                        root,
                        job_id="producer",
                        event_id=f"publication-{index}",
                        kind="workload.artifact",
                        data={"artifact_type": "checkpoint", "publication": publication},
                    )
                    _ingest_reports(controller)
                current_consumer = controller.state["jobs"]["consumer"]
                current_producer = controller.state["jobs"]["producer"]
                self.assertEqual(1, len(current_consumer["condition_satisfactions"]))
                self.assertEqual("artifact-0", current_consumer["condition_satisfactions"][0]["artifact_id"])
                self.assertEqual(1, len(current_producer["artifact_condition_evidence"]))
                self.assertLessEqual(len(current_producer["artifact_evidence"]), 8)
                _satisfy_from_current_artifacts(current_consumer, [current_producer])
            finally:
                controller.journal.close()


if __name__ == "__main__":
    unittest.main()
