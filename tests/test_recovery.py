from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from scruffy._compat import UTC
from scruffy.controller import (
    _evacuation_target_terminal,
    _initialize_controller,
    _refresh_dependencies,
    _retry_pending,
)
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
    task_id: str = "train",
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
        "task_id": task_id,
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


def _ended(
    job: dict[str, object],
    state: str,
    reason: str,
    *,
    exit_code: int,
    finished_at: str | None = None,
) -> dict[str, object]:
    job.update(
        state=state,
        reason=reason,
        exit_code=exit_code,
        finished_at=finished_at or utc_now(),
        last_assignment=job["assignment"],
        assignment=None,
    )
    del job["launch_token"]
    return job


def _state(
    root: Path,
    job: dict[str, object],
    allocation: AllocationIncarnation,
    *others: dict[str, object],
) -> None:
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
            "jobs": {str(item["id"]): item for item in (job, *others)},
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

    def test_handover_retries_only_for_the_attempts_own_reason(self) -> None:
        inventory = (NodeInventory("node", (0,), 2, 2),)
        old = AllocationIncarnation("old", 0, inventory)
        for lost_reason, allocations in (
            (
                "allocation_replaced",
                (
                    AllocationIncarnation("new", 0, inventory),
                    AllocationIncarnation("newer", 0, inventory),
                ),
            ),
            (
                "allocation_incarnation_changed",
                (
                    AllocationIncarnation("old", 1, inventory),
                    AllocationIncarnation("old", 2, inventory),
                ),
            ),
        ):
            with self.subTest(lost_reason), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "queue"
                # A deterministic application failure listed nowhere in
                # retry_on, ended before the allocation was replaced.
                diverged = _ended(
                    _job(root, job_id="job-diverged", task_id="diverged"),
                    "failed",
                    "application_exit",
                    exit_code=1,
                )
                running = _job(root, job_id="job-running", task_id="train")
                _state(root, diverged, old, running)
                # Every later handover leaves the failure alone and retries the
                # job it actually lost exactly once.
                for allocation in allocations:
                    controller = _initialize_controller(
                        root=root, inventory=inventory, launcher="slurm",
                        allocation_id=allocation.slurm_job_id,
                        slurm_job_id=allocation.slurm_job_id,
                        allocation_incarnation=allocation, poll_interval=0.1,
                        cancel_grace=0, gpu_health_mode="off",
                    )
                    controller.journal.close()
                    jobs = controller.state["jobs"]
                    self.assertEqual(3, len(jobs))
                    failure = jobs["job-diverged"]
                    self.assertEqual(
                        ("failed", "application_exit"), (failure["state"], failure["reason"])
                    )
                    self.assertNotIn("successor_job_id", failure)
                    self.assertNotIn("retry_exhausted", failure)
                    self.assertEqual(lost_reason, jobs["job-running"]["reason"])
                    successors = [job for job in jobs.values() if "predecessor_job_id" in job]
                    self.assertEqual(1, len(successors))
                    self.assertEqual("job-running", successors[0]["predecessor_job_id"])
                    self.assertEqual(lost_reason, successors[0]["retry_reason"])
                    self.assertEqual(2, successors[0]["attempt"])
                    self.assertEqual("queued", successors[0]["state"])
                    self.assertEqual(successors[0]["id"], jobs["job-running"]["successor_job_id"])

    def test_evacuated_attempt_retries_once_for_its_own_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root, inventory=(NodeInventory("node", (0,), 2, 2),), launcher="local",
                allocation_id="local", slurm_job_id=None, poll_interval=0.1,
                cancel_grace=0, gpu_health_mode="off",
            )
            try:
                evacuated = _ended(
                    _job(root, job_id="job-evacuated", task_id="train"),
                    "failed",
                    "evacuated",
                    exit_code=75,
                )
                diverged = _ended(
                    _job(root, job_id="job-diverged", task_id="diverged"),
                    "failed",
                    "application_exit",
                    exit_code=1,
                )
                controller.state["jobs"].update(
                    {"job-evacuated": evacuated, "job-diverged": diverged}
                )
                evacuation = {"request_id": "evacuate", "state": "waiting", "targets": {}}
                outcomes = []
                for job_id in ("job-evacuated", "job-diverged", "job-evacuated"):
                    target = {"job_id": job_id, "outcome": "waiting"}
                    _evacuation_target_terminal(controller, evacuation, target)
                    outcomes.append(target)
                jobs = controller.state["jobs"]
                successors = [job for job in jobs.values() if "predecessor_job_id" in job]
                self.assertEqual(1, len(successors))
                successor = successors[0]
                self.assertEqual("job-evacuated", successor["predecessor_job_id"])
                self.assertEqual("evacuated", successor["retry_reason"])
                self.assertEqual(2, successor["attempt"])
                self.assertEqual(successor["id"], evacuated["successor_job_id"])
                retried = {
                    "job_id": "job-evacuated",
                    "outcome": "retry_queued",
                    "successor_job_id": successor["id"],
                }
                self.assertEqual([retried, retried], [outcomes[0], outcomes[2]])
                self.assertEqual(
                    {"job_id": "job-diverged", "outcome": "lost", "reason": "job_failed"},
                    outcomes[1],
                )
                self.assertNotIn("successor_job_id", diverged)
            finally:
                controller.journal.close()

    def test_waiters_on_an_application_failure_are_skipped_after_handover(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            inventory = (NodeInventory("node", (0,), 2, 2),)
            old = AllocationIncarnation("old", 0, inventory)
            new = AllocationIncarnation("new", 0, inventory)
            finished_at = (datetime.now(UTC) - timedelta(hours=2)).isoformat(
                timespec="milliseconds"
            )
            diverged = _ended(
                _job(root, job_id="job-diverged", task_id="train"),
                "failed",
                "application_exit",
                exit_code=1,
                finished_at=finished_at,
            )
            # This producer is still running when the allocation is replaced.
            preparing = _job(root, job_id="job-preparing", task_id="prepare")
            waiters = []
            for job_id, producer in (("job-sample", "train"), ("job-export", "prepare")):
                waiter = _job(
                    root,
                    job_id=job_id,
                    task_id=job_id,
                    wait_for=[{"kind": "artifact", "task_id": producer, "artifact_id": "ckpt"}],
                )
                waiter.update(state="blocked", assignment=None, dependency_gate_passed=False)
                del waiter["launch_token"]
                waiters.append(waiter)
            _state(root, diverged, old, preparing, *waiters)
            controller = _initialize_controller(
                root=root, inventory=inventory, launcher="slurm", allocation_id="new",
                slurm_job_id="new", allocation_incarnation=new, poll_interval=0.1,
                cancel_grace=0, gpu_health_mode="off",
            )
            try:
                jobs = controller.state["jobs"]
                self.assertFalse(_retry_pending(controller, jobs["job-diverged"]))
                self.assertNotIn("successor_job_id", jobs["job-diverged"])
                self.assertIn("successor_job_id", jobs["job-preparing"])
                _refresh_dependencies(controller)
                # The failure cannot retry, so its waiter is released; the
                # lost producer's retry keeps its own waiter blocked.
                self.assertEqual("skipped", jobs["job-sample"]["state"])
                self.assertEqual("condition_unsatisfied", jobs["job-sample"]["reason"])
                self.assertEqual("blocked", jobs["job-export"]["state"])
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
