from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scruffy.controller import (
    _initial_job_event,
    _initialize_controller,
    _satisfy_from_current_artifacts,
    _stage_atomic_submission,
)
from scruffy.models import NodeInventory, ResourceRequest
from scruffy.provenance import write_request_record, write_result_record
from scruffy.slurm import AllocationIncarnation
from scruffy.state import apply_workload_event
from scruffy.storage import (
    StorageError,
    archive_terminal_job,
    read_events,
    utc_now,
    write_state,
)
from scruffy.submissions import job_from_spec, workflow_submission
from scruffy.workflows import WorkflowError

POLICY = {
    "max_attempts": 3,
    "retry_on": ["allocation_replaced", "allocation_incarnation_changed"],
    "evacuation": {"signal": "USR1", "grace_seconds": 600},
}
INVENTORY = (NodeInventory("node", (0,), 2, 2),)


def _spec(root: Path, *, job_id: str, task_id: str, workflow_id: str = "flow") -> dict[str, object]:
    return {
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
    }


def _job(root: Path, *, job_id: str, task_id: str, workflow_id: str = "flow") -> dict[str, object]:
    return job_from_spec(_spec(root, job_id=job_id, task_id=task_id, workflow_id=workflow_id), 1)


class RecoveryReviewTests(unittest.TestCase):
    def test_delayed_older_attempt_publication_releases_waiter(self) -> None:
        producer = _job(Path("/tmp"), job_id="attempt-1", task_id="prepare")
        producer.update(attempt=1, queue_order=1)
        newer = _job(Path("/tmp"), job_id="attempt-2", task_id="prepare")
        newer.update(attempt=2, queue_order=2)
        consumer = _job(Path("/tmp"), job_id="consumer", task_id="infer")
        consumer["wait_for"] = [{"kind": "artifact", "task_id": "prepare", "artifact_id": "ckpt"}]
        publication = {
            "v": 1,
            "artifact_id": "ckpt",
            "path": "/tmp/ckpt",
            "size_bytes": 1,
            "sha256": "a" * 64,
            "manifest_path": "/tmp/ckpt.ready.json",
        }
        apply_workload_event(
            producer,
            {
                "event_id": "older-publication",
                "occurred_at": "2026-09-01T10:00:00+00:00",
                "kind": "workload.artifact",
                "data": {"artifact_type": "checkpoint", "publication": publication},
            },
            recorded_at="2026-09-01T10:00:01+00:00",
        )
        _satisfy_from_current_artifacts(consumer, [producer, newer])
        self.assertEqual("attempt-1", consumer["condition_satisfactions"][0]["producer_job_id"])
        consumer["condition_satisfactions"] = []
        consumer["state"] = "queued"
        self.assertEqual(
            "job.queued",
            _initial_job_event(
                mock.Mock(inventory=INVENTORY),
                consumer,
                {item["id"]: item for item in [producer, newer, consumer]},
            ),
        )

    def test_archived_artifact_evidence_survives_late_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            producer = _job(root, job_id="archived-producer", task_id="prepare")
            producer.update(state="succeeded", finished_at=utc_now())
            publication = {
                "v": 1,
                "artifact_id": "ckpt",
                "path": str(root / "ckpt"),
                "size_bytes": 1,
                "sha256": "b" * 64,
                "manifest_path": str(root / "ckpt.ready.json"),
            }
            apply_workload_event(
                producer,
                {
                    "event_id": "archived-publication",
                    "occurred_at": "2026-09-01T10:00:00+00:00",
                    "kind": "workload.artifact",
                    "data": {"artifact_type": "checkpoint", "publication": publication},
                },
                recorded_at="2026-09-01T10:00:01+00:00",
            )
            archive_terminal_job(root, producer)
            from scruffy.storage import find_archived_job

            cold = find_archived_job(root, "archived-producer")
            self.assertEqual(1, len(cold["artifact_evidence"]))
            consumer = _job(root, job_id="consumer", task_id="infer")
            consumer["wait_for"] = [
                {"kind": "artifact", "task_id": "prepare", "artifact_id": "ckpt"}
            ]
            _satisfy_from_current_artifacts(consumer, [cold])
            self.assertEqual("archived-producer", consumer["condition_satisfactions"][0]["producer_job_id"])

    def test_atomic_history_uses_archived_attempts_and_succeeded_is_final(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            old = _job(root, job_id="old-job", task_id="train")
            old.update(state="failed", finished_at=utc_now(), attempt=4)
            archive_terminal_job(root, old)
            document = workflow_submission(
                request_id="new-request",
                workflow_id="flow",
                tasks=[
                    {
                        "task_id": "train",
                        "argv": ["true"],
                        "cwd": str(root),
                        "resources": ResourceRequest(1, 1, 1, 1).to_dict(),
                    }
                ],
            )
            controller = mock.Mock(root=root, inventory=INVENTORY)
            staged = _stage_atomic_submission(
                controller,
                document["submission_id"],
                document,
                4,
                {},
            )
            self.assertEqual(5, staged[0]["attempt"])
            old["state"] = "succeeded"
            archive_terminal_job(root, old)
            with self.assertRaises(WorkflowError):
                _stage_atomic_submission(controller, document["submission_id"], document, 5, {})

    def test_recovery_preserves_proven_gate_and_repairs_reverse_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            old = _job(root, job_id="old", task_id="train")
            old.update(
                state="running",
                attempt=1,
                recovery=POLICY,
                dependency_gate_passed=True,
                condition_satisfactions=[{"task_id": "prepare", "artifact_id": "ckpt"}],
                resolved_dependencies=[{"task_id": "prepare", "job_id": "prepare-1"}],
                resolved_conditions=[{"task_id": "prepare", "artifact_id": "ckpt"}],
                wait_for=[{"kind": "artifact", "task_id": "prepare", "artifact_id": "ckpt"}],
                assignment={
                    "job_id": "old",
                    "request": old["request"],
                    "reservations": [{"node": "node", "gpu_ids": [0], "cpus": 1, "memory_gb": 1}],
                },
                launch_token="token",
            )
            old_allocation = AllocationIncarnation("old-allocation", 0, INVENTORY)
            new_allocation = AllocationIncarnation("new-allocation", 0, INVENTORY)
            write_state(root, {
                "v": 1, "queue_id": "queue", "last_seq": 0, "journal_generation": 0,
                "journal_offset": 0, "allocation": {"id": "old-allocation", "state": "running", "incarnation": old_allocation.to_dict()},
                "nodes": {}, "gpu_health": None, "jobs": {"old": old}, "next_queue_order": 1,
                "archived_jobs": 0, "archived_counts": {}, "archived_project_counts": {},
                "draining": False, "drain_requested": False, "launches_paused": False, "updated_at": utc_now(),
            })
            controller = _initialize_controller(
                root=root, inventory=INVENTORY, launcher="slurm", allocation_id="new-allocation",
                slurm_job_id="new-allocation", allocation_incarnation=new_allocation,
                poll_interval=0.1, cancel_grace=0, gpu_health_mode="off",
            )
            try:
                predecessor = controller.state["jobs"]["old"]
                successor = next(
                    job for job in controller.state["jobs"].values() if job.get("attempt") == 2
                )
                self.assertEqual("queued", successor["state"])
                self.assertTrue(successor["dependency_gate_passed"])
                self.assertEqual(old["resolved_conditions"], successor["resolved_conditions"])
                self.assertEqual(successor["id"], predecessor["successor_job_id"])
                self.assertTrue(any(event["kind"] == "job.recovery_linked" for event in read_events(root)))
            finally:
                controller.journal.close()

    def test_provenance_replay_is_idempotent_and_conflicts_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            job = _job(root, job_id="provenance", task_id="train")
            record = write_request_record(root, job)
            before = record.read_bytes()
            write_request_record(root, job)
            self.assertEqual(before, record.read_bytes())
            job["argv"] = ["different"]
            with self.assertRaises(StorageError):
                write_request_record(root, job)
            job["argv"] = ["true"]
            job.update(state="succeeded", finished_at="2026-09-01T10:00:00+00:00")
            result = write_result_record(root, job)
            before = result.read_bytes()
            write_result_record(root, job)
            self.assertEqual(before, result.read_bytes())
            job["exit_code"] = 1
            with self.assertRaises(StorageError):
                write_result_record(root, job)

    def test_crash_after_job_lost_replays_one_deterministic_successor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            old = _job(root, job_id="crash-old", task_id="train")
            old.update(state="running", recovery=POLICY, launch_token="token", assignment={
                "job_id": "crash-old", "request": old["request"],
                "reservations": [{"node": "node", "gpu_ids": [0], "cpus": 1, "memory_gb": 1}],
            })
            old_allocation = AllocationIncarnation("old", 0, INVENTORY)
            new_allocation = AllocationIncarnation("new", 0, INVENTORY)
            write_state(root, {
                "v": 1, "queue_id": "queue", "last_seq": 0, "journal_generation": 0, "journal_offset": 0,
                "allocation": {"id": "old", "state": "running", "incarnation": old_allocation.to_dict()},
                "nodes": {}, "gpu_health": None, "jobs": {"crash-old": old}, "next_queue_order": 1,
                "archived_jobs": 0, "archived_counts": {}, "archived_project_counts": {},
                "draining": False, "drain_requested": False, "launches_paused": False, "updated_at": utc_now(),
            })
            with (
                mock.patch(
                    "scruffy.controller._recover_lost_workflow_jobs",
                    side_effect=RuntimeError("crash"),
                ),
                self.assertRaises(RuntimeError),
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
                successor_ids = [
                    job["id"] for job in recovered.state["jobs"].values()
                    if job.get("predecessor_job_id") == "crash-old"
                ]
                self.assertEqual(1, len(successor_ids))
                successor_id = successor_ids[0]
            finally:
                recovered.journal.close()
            replayed = _initialize_controller(
                root=root, inventory=INVENTORY, launcher="slurm", allocation_id="new",
                slurm_job_id="new", allocation_incarnation=new_allocation,
                poll_interval=0.1, cancel_grace=0, gpu_health_mode="off",
            )
            try:
                self.assertIn(successor_id, replayed.state["jobs"])
                self.assertEqual(2, len(replayed.state["jobs"]))
            finally:
                replayed.journal.close()


if __name__ == "__main__":
    unittest.main()
