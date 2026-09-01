from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scruffy.controller import _initialize_controller
from scruffy.models import NodeInventory, ResourceRequest
from scruffy.slurm import AllocationIncarnation
from scruffy.storage import (
    archive_terminal_job,
    create_job_id,
    recovery_request_id,
    utc_now,
    write_state,
)
from scruffy.submissions import job_from_spec
from scruffy.summary import job_view
from scruffy.workflows import select_task_attempts, validate_workflows

POLICY = {
    "max_attempts": 3,
    "retry_on": ["allocation_replaced", "allocation_incarnation_changed", "evacuated"],
    "evacuation": {"signal": "USR1", "grace_seconds": 600},
}


def _job(
    root: Path,
    *,
    job_id: str = "job-old",
    attempt: int = 1,
    policy: dict[str, object] | None = POLICY,
    wait_for: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    spec: dict[str, object] = {
        "v": 1,
        "job_id": job_id,
        "request_id": f"request-{job_id}",
        "name": "train",
        "submitted_at": utc_now(),
        "argv": ["true"],
        "cwd": str(root),
        "env": {},
        "resources": ResourceRequest(1, 1, 1, 1).to_dict(),
        "workflow_id": "workflow",
        "task_id": "train",
        "needs": [],
        "wait_for": [] if wait_for is None else wait_for,
    }
    if policy is not None:
        spec["recovery"] = policy
    job = job_from_spec(spec, 1)
    job["attempt"] = attempt
    job["state"] = "running"
    job["assignment"] = {
        "job_id": job_id,
        "request": job["request"],
        "reservations": [
            {"node": "node", "gpu_ids": [0], "cpus": 1, "memory_gb": 1}
        ],
    }
    job["launch_token"] = "launch-token"
    return job


def _state(root: Path, job: dict[str, object], allocation: AllocationIncarnation) -> None:
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
            "jobs": {str(job["id"]): job},
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


class RecoveryPolicyTests(unittest.TestCase):
    def test_policy_is_strict_and_evacuated_is_persistable(self) -> None:
        job = {"workflow_id": "flow", "task_id": "task", "recovery": POLICY}
        validate_workflows([job])
        for invalid in (
            {**POLICY, "extra": True},
            {**POLICY, "max_attempts": 11},
            {**POLICY, "retry_on": ["application_exit"]},
            {**POLICY, "evacuation": {"signal": "TERM", "grace_seconds": 1}},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_workflows(
                    [{"workflow_id": "flow", "task_id": "task", "recovery": invalid}]
                )

    def test_standalone_recovery_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires workflow_id"):
            validate_workflows([{"recovery": POLICY}])

    def test_internal_request_ids_are_stable_and_attempt_scoped(self) -> None:
        first = recovery_request_id("project", "workflow", "task", 2)
        self.assertEqual(first, recovery_request_id("project", "workflow", "task", 2))
        self.assertNotEqual(first, recovery_request_id("project", "workflow", "task", 3))
        self.assertNotEqual(
            create_job_id(first, project_id="project"),
            create_job_id(recovery_request_id("project", "workflow", "task", 3), project_id="project"),
        )


class RecoveryHandoverTests(unittest.TestCase):
    def test_replacement_replays_once_and_preserves_artifact_wait(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            inventory = (NodeInventory("node", (0,), 2, 2),)
            old = AllocationIncarnation("old", 0, inventory)
            new = AllocationIncarnation("new", 0, inventory)
            wait_for = [{"kind": "artifact", "task_id": "prepare", "artifact_id": "checkpoint"}]
            job = _job(root, wait_for=wait_for)
            _state(root, job, old)
            controller = _initialize_controller(
                root=root,
                inventory=inventory,
                launcher="slurm",
                allocation_id="new",
                slurm_job_id="new",
                allocation_incarnation=new,
                poll_interval=0.1,
                cancel_grace=0,
                gpu_health_mode="off",
            )
            controller.journal.close()
            successor = next(item for item in controller.state["jobs"].values() if item["attempt"] == 2)
            self.assertEqual("job-old", successor["predecessor_job_id"])
            self.assertEqual("allocation_replaced", successor["retry_reason"])
            self.assertEqual(wait_for, successor["wait_for"])
            successor_id = successor["id"]

            repeated = _initialize_controller(
                root=root,
                inventory=inventory,
                launcher="slurm",
                allocation_id="new",
                slurm_job_id="new",
                allocation_incarnation=new,
                poll_interval=0.1,
                cancel_grace=0,
                gpu_health_mode="off",
            )
            try:
                self.assertEqual(2, len(repeated.state["jobs"]))
                self.assertIn(successor_id, repeated.state["jobs"])
            finally:
                repeated.journal.close()

    def test_noneligible_and_exhausted_tasks_do_not_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            inventory = (NodeInventory("node", (0,), 2, 2),)
            old = AllocationIncarnation("old", 0, inventory)
            new = AllocationIncarnation("new", 0, inventory)
            noneligible = _job(
                root,
                job_id="job-noneligible",
                policy={**POLICY, "retry_on": ["evacuated"]},
            )
            _state(root, noneligible, old)
            controller = _initialize_controller(
                root=root, inventory=inventory, launcher="slurm", allocation_id="new",
                slurm_job_id="new", allocation_incarnation=new, poll_interval=0.1,
                cancel_grace=0, gpu_health_mode="off",
            )
            controller.journal.close()
            self.assertEqual(1, len(controller.state["jobs"]))

            exhausted_root = Path(temporary) / "exhausted-queue"
            exhausted = _job(exhausted_root, job_id="job-exhausted", attempt=3)
            _state(exhausted_root, exhausted, old)
            controller = _initialize_controller(
                root=exhausted_root, inventory=inventory, launcher="slurm", allocation_id="new",
                slurm_job_id="new", allocation_incarnation=new, poll_interval=0.1,
                cancel_grace=0, gpu_health_mode="off",
            )
            try:
                self.assertTrue(controller.state["jobs"]["job-exhausted"]["retry_exhausted"])
                self.assertEqual(1, len(controller.state["jobs"]))
            finally:
                controller.journal.close()

    def test_archive_and_summary_retain_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            job = _job(root, job_id="job-terminal")
            job.update(
                state="lost",
                assignment=None,
                finished_at=utc_now(),
                predecessor_job_id="job-predecessor",
                retry_reason="allocation_replaced",
                retry_exhausted=True,
            )
            archive_terminal_job(root, job)
            view = job_view(job)
            self.assertEqual("job-predecessor", view["predecessor_job_id"])
            self.assertTrue(view["retry_exhausted"])

    def test_latest_attempt_resolves_workflow_dependencies(self) -> None:
        predecessor = {"workflow_id": "flow", "task_id": "task", "state": "lost", "queue_order": 1}
        successor = {
            "workflow_id": "flow", "task_id": "task", "state": "queued", "queue_order": 2
        }
        selected = select_task_attempts([predecessor, successor])
        self.assertIs(successor, selected[("default", "flow", "task")])


if __name__ == "__main__":
    unittest.main()
