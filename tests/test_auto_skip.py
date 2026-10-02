"""Blocked jobs become skipped once an awaited producer can no longer deliver."""

from __future__ import annotations

import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

import scruffy.controller as controller_module
from scruffy._compat import UTC
from scruffy.client import publish_event
from scruffy.controller import (
    ARTIFACT_SKIP_GRACE_SECONDS,
    _ingest_reports,
    _initialize_controller,
    _refresh_dependencies,
)
from scruffy.models import NodeInventory, ResourceRequest
from scruffy.runtime import Controller
from scruffy.storage import archive_terminal_job, read_events
from scruffy.workflows import resolve_blocked_jobs

REQUEST = ResourceRequest(1, 1, 1, 1)
INVENTORY = (NodeInventory("local", (0,), 2, 2),)
RECOVERY = {
    "max_attempts": 3,
    "retry_on": ["allocation_replaced"],
    "evacuation": {"signal": "USR1", "grace_seconds": 60},
}


def ago(seconds: float) -> str:
    return (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat(
        timespec="milliseconds"
    )


def producer(
    job_id: str,
    *,
    state: str = "failed",
    workflow_id: str = "flow",
    task_id: str = "train",
    finished_seconds_ago: float = 3600,
    order: int = 1,
    **extra: Any,
) -> dict[str, Any]:
    image = {
        "id": job_id,
        "name": job_id,
        "state": state,
        "submitted_at": ago(finished_seconds_ago + 60),
        "started_at": ago(finished_seconds_ago + 30),
        "finished_at": ago(finished_seconds_ago),
        "queue_order": order,
        "attempt": 1,
        "argv": ["true"],
        "cwd": "/tmp",
        "env": {},
        "request": REQUEST.to_dict(),
        "assignment": None,
        "workflow_id": workflow_id,
        "task_id": task_id,
        "needs": [],
        "wait_for": [],
        "dependency_gate_passed": True,
    }
    image.update(extra)
    return image


def consumer(
    job_id: str,
    *,
    workflow_id: str = "flow",
    task_id: str | None = None,
    wait_for: tuple[str, ...] = ("train",),
    needs: tuple[tuple[str, str], ...] = (),
    order: int = 2,
) -> dict[str, Any]:
    return {
        "id": job_id,
        "name": job_id,
        "state": "blocked",
        "submitted_at": ago(7200),
        "queue_order": order,
        "attempt": 1,
        "argv": ["true"],
        "cwd": "/tmp",
        "env": {},
        "request": REQUEST.to_dict(),
        "assignment": None,
        "workflow_id": workflow_id,
        "task_id": task_id or job_id,
        "needs": [{"task_id": task, "condition": condition} for task, condition in needs],
        "wait_for": [
            {"kind": "artifact", "task_id": task, "artifact_id": f"{task}/ckpt"}
            for task in wait_for
        ],
        "blockers": [],
        "dependency_gate_passed": False,
    }


class ResolverTests(unittest.TestCase):
    def test_terminal_producer_is_pending_until_declared_final(self) -> None:
        jobs = [producer("p"), consumer("c")]
        pending = resolve_blocked_jobs(jobs)[("default", "flow", "c")]
        self.assertEqual("blocked", pending["decision"])
        self.assertEqual("condition_pending", pending["blockers"][0]["reason"])

        final = resolve_blocked_jobs(jobs, final_producers={"p"})[("default", "flow", "c")]
        self.assertEqual("skipped", final["decision"])
        self.assertEqual("condition_unsatisfied", final["reason"])
        self.assertEqual("failed", final["blockers"][0]["state"])

    def test_skip_propagates_through_a_mixed_chain_in_one_pass(self) -> None:
        jobs = [
            producer("train"),
            consumer("eval", wait_for=("train",), order=2),
            consumer("report", wait_for=(), needs=(("eval", "succeeded"),), order=3),
            consumer("export", wait_for=("eval",), order=4),
            consumer("cleanup", wait_for=(), needs=(("export", "terminal"),), order=5),
        ]
        resolutions = resolve_blocked_jobs(jobs, final_producers={"train"})
        decisions = {key[2]: value["decision"] for key, value in resolutions.items()}
        reasons = {key[2]: value["reason"] for key, value in resolutions.items()}
        self.assertEqual(
            {
                "eval": "skipped",
                "report": "skipped",
                "export": "skipped",
                "cleanup": "ready",
            },
            decisions,
        )
        self.assertEqual("condition_unsatisfied", reasons["eval"])
        self.assertEqual("dependency_unsatisfied", reasons["report"])
        self.assertEqual("condition_unsatisfied", reasons["export"])

    def test_running_producer_never_becomes_final(self) -> None:
        jobs = [producer("p", state="running"), consumer("c")]
        result = resolve_blocked_jobs(jobs, final_producers={"p"})[("default", "flow", "c")]
        self.assertEqual("blocked", result["decision"])
        self.assertEqual("condition_pending", result["blockers"][0]["reason"])


class AutomaticSkipControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"

    def _controller(self, *images: dict[str, Any]) -> Controller:
        controller = _initialize_controller(
            root=self.root,
            inventory=INVENTORY,
            launcher="local",
            allocation_id="local-allocation",
            slurm_job_id=None,
            poll_interval=0.01,
            cancel_grace=0,
        )
        self.addCleanup(lambda: controller.journal.close())
        for image in images:
            controller.state["jobs"][image["id"]] = image
        return controller

    def _state(self, controller: Controller, job_id: str) -> str:
        return controller.state["jobs"][job_id]["state"]

    def test_recent_failure_waits_for_the_settling_time_then_skips(self) -> None:
        controller = self._controller(
            producer("p", finished_seconds_ago=1), consumer("c")
        )
        _refresh_dependencies(controller)
        self.assertEqual("blocked", self._state(controller, "c"))
        self.assertEqual(
            "condition_pending", controller.state["jobs"]["c"]["blockers"][0]["reason"]
        )
        recheck_at = controller.workflow_recheck_at[("default", "flow")]
        self.assertAlmostEqual(
            time.time() + ARTIFACT_SKIP_GRACE_SECONDS - 1, recheck_at, delta=5
        )

        # Unrelated ticks reuse the cached graph until the settling time ends.
        with mock.patch(
            "scruffy.controller.resolve_blocked_jobs", wraps=resolve_blocked_jobs
        ) as resolve:
            _refresh_dependencies(controller)
            self.assertEqual(0, resolve.call_count)
            with mock.patch("scruffy.controller.time.time", return_value=recheck_at + 1):
                _refresh_dependencies(controller)
            self.assertEqual(1, resolve.call_count)

        job = controller.state["jobs"]["c"]
        self.assertEqual("skipped", job["state"])
        self.assertEqual("condition_unsatisfied", job["reason"])
        self.assertEqual(
            {
                "kind": "artifact",
                "task_id": "train",
                "artifact_id": "train/ckpt",
                "state": "failed",
                "reason": "condition_unsatisfied",
            },
            job["blockers"][0],
        )
        self.assertNotIn(("default", "flow"), controller.workflow_recheck_at)
        result = (self.root / "provenance" / "c" / "result.json").read_text()
        self.assertIn("condition_unsatisfied", result)

    def test_archived_cancelled_producer_releases_stale_waiters(self) -> None:
        archive_terminal_job(self.root, producer("p", state="cancelled"))
        controller = self._controller(consumer("c-1"), consumer("c-2", order=3))
        _refresh_dependencies(controller)
        self.assertEqual("skipped", self._state(controller, "c-1"))
        self.assertEqual("skipped", self._state(controller, "c-2"))

    def test_pending_report_keeps_the_waiter_until_it_is_ingested(self) -> None:
        controller = self._controller(producer("p", state="succeeded"), consumer("c"))
        publish_event(
            self.root,
            job_id="p",
            kind="workload.artifact",
            event_id="ckpt",
            data={
                "artifact_type": "checkpoint",
                "publication": {
                    "v": 1,
                    "artifact_id": "train/ckpt",
                    "path": "/shared/ckpt.pt",
                    "size_bytes": 3,
                    "sha256": "a" * 64,
                    "manifest_path": "/shared/ckpt.pt.ready.json",
                },
            },
        )
        _refresh_dependencies(controller)
        self.assertEqual("blocked", self._state(controller, "c"))
        self.assertIn(("default", "flow"), controller.workflow_recheck_at)

        _ingest_reports(controller)
        _refresh_dependencies(controller)
        self.assertEqual("queued", self._state(controller, "c"))

    def test_lost_producer_with_a_pending_retry_keeps_waiters_blocked(self) -> None:
        lost = producer(
            "p", state="lost", reason="allocation_replaced", recovery=RECOVERY
        )
        controller = self._controller(lost, consumer("c"))
        _refresh_dependencies(controller)
        self.assertEqual("blocked", self._state(controller, "c"))

        # Once the retry exists the newest attempt is live, still pending.
        controller.state["jobs"]["p"]["successor_job_id"] = "p-2"
        controller.state["jobs"]["p-2"] = producer(
            "p-2", state="queued", order=3, attempt=2, finished_at=None
        )
        _refresh_dependencies(controller)
        self.assertEqual("blocked", self._state(controller, "c"))

        # When the retry also ends without the artifact, the waiter is skipped.
        controller.state["jobs"]["p-2"].update(
            {"state": "failed", "reason": "application_exit", "finished_at": ago(3600)}
        )
        _refresh_dependencies(controller)
        self.assertEqual("skipped", self._state(controller, "c"))

    def test_lost_producer_without_retry_after_replacement_releases_waiters(self) -> None:
        lost = producer("p", state="lost", reason="allocation_replaced")
        controller = self._controller(lost, consumer("c"))
        _refresh_dependencies(controller)
        self.assertEqual("skipped", self._state(controller, "c"))

    def test_unparseable_finish_time_keeps_waiting(self) -> None:
        broken = producer("p", finished_at="not-a-time", submitted_at="also-not")
        controller = self._controller(broken, consumer("c"))
        _refresh_dependencies(controller)
        self.assertEqual("blocked", self._state(controller, "c"))
        self.assertNotIn(("default", "flow"), controller.workflow_recheck_at)

    def test_missing_producer_still_waits(self) -> None:
        controller = self._controller(consumer("c"))
        _refresh_dependencies(controller)
        self.assertEqual("blocked", self._state(controller, "c"))
        self.assertEqual(
            "condition_source_missing",
            controller.state["jobs"]["c"]["blockers"][0]["reason"],
        )

    def test_skip_backlog_drains_over_bounded_ticks(self) -> None:
        images: list[dict[str, Any]] = []
        for index in range(7):
            workflow_id = f"flow-{index}"
            images.append(producer(f"p-{index}", workflow_id=workflow_id, order=index * 2))
            images.append(
                consumer(f"c-{index}", workflow_id=workflow_id, order=index * 2 + 1)
            )
        # One chain where the budget runs out inside the workflow.
        images.append(producer("chain-p", workflow_id="chain", order=100))
        images.append(consumer("chain-a", workflow_id="chain", wait_for=("train",), order=101))
        images.append(
            consumer(
                "chain-b",
                workflow_id="chain",
                wait_for=(),
                needs=(("chain-a", "succeeded"),),
                order=102,
            )
        )
        images.append(
            consumer(
                "chain-c",
                workflow_id="chain",
                wait_for=(),
                needs=(("chain-b", "terminal"),),
                order=103,
            )
        )
        controller = self._controller(*images)
        ticks = 0
        with mock.patch.object(controller_module, "MAX_DEPENDENCY_SKIPS_PER_TICK", 3):
            while any(
                job["state"] == "blocked" for job in controller.state["jobs"].values()
            ):
                before = sum(
                    job["state"] == "skipped" for job in controller.state["jobs"].values()
                )
                _refresh_dependencies(controller)
                after = sum(
                    job["state"] == "skipped" for job in controller.state["jobs"].values()
                )
                self.assertLessEqual(after - before, 3)
                # A dependant never runs ahead of an unapplied upstream skip.
                if self._state(controller, "chain-c") == "queued":
                    self.assertEqual("skipped", self._state(controller, "chain-b"))
                ticks += 1
                self.assertLess(ticks, 10)
        self.assertGreaterEqual(ticks, 3)
        self.assertEqual("queued", self._state(controller, "chain-c"))
        skipped = [
            event["job_id"] for event in read_events(self.root) if event["kind"] == "job.skipped"
        ]
        self.assertEqual(9, len(skipped))
        self.assertEqual(len(skipped), len(set(skipped)))


if __name__ == "__main__":
    unittest.main()
