"""The hot snapshot holds live work, not history, and old roots migrate."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

import scruffy.state as state_module
from scruffy._compat import UTC
from scruffy.client import observe, status, summary
from scruffy.controller import _apply_tick, _initialize_controller
from scruffy.models import NodeInventory, ResourceRequest
from scruffy.runtime import Controller
from scruffy.state import (
    MAX_TERMINAL_JOBS,
    compact_journal,
    emit,
    group_commit,
    load_recovered_state,
)
from scruffy.storage import (
    archive_terminal_job,
    job_directory,
    load_state,
    queue_id,
    read_events,
    write_state,
)

REQUEST = ResourceRequest(1, 1, 1, 1)
INVENTORY = (NodeInventory("local", (0,), 2, 2),)


def ago(seconds: float) -> str:
    return (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat(
        timespec="milliseconds"
    )


def image(
    job_id: str,
    state: str,
    order: int,
    *,
    workflow_id: str | None = None,
    task_id: str | None = None,
    project_id: str | None = None,
    started: bool = True,
    finished_seconds_ago: float = 7200,
    **extra: Any,
) -> dict[str, Any]:
    job: dict[str, Any] = {
        "id": job_id,
        "name": job_id,
        "state": state,
        "submitted_at": ago(finished_seconds_ago + 120),
        "queue_order": order,
        "argv": ["python", "-c", "print('x' * 64)", "--config", "/shared/config.yaml"],
        "cwd": "/shared/code/project",
        "env": {"PYTHONUNBUFFERED": "1", "WANDB_MODE": "offline"},
        "request": REQUEST.to_dict(),
        "assignment": None,
        "request_digest": "0" * 64,
    }
    if project_id is not None:
        job["project_id"] = project_id
    if started and state not in {"queued", "blocked"}:
        job["started_at"] = ago(finished_seconds_ago + 60)
    if state in {"succeeded", "failed", "cancelled", "lost", "skipped", "rejected"}:
        job["finished_at"] = ago(finished_seconds_ago)
    if workflow_id is not None:
        job.update(
            {
                "workflow_id": workflow_id,
                "task_id": task_id or job_id,
                "attempt": 1,
                "needs": [],
                "wait_for": [],
                "blockers": [],
                "dependency_gate_passed": state != "blocked",
            }
        )
    job.update(extra)
    return job


def legacy_state(root: Path, jobs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return a snapshot shaped like the production release writes it."""

    return {
        "v": 1,
        "queue_id": queue_id(root),
        "last_seq": 0,
        "journal_generation": 0,
        "journal_offset": 0,
        "allocation": None,
        "nodes": {},
        "gpu_health": None,
        "jobs": jobs,
        "report_acks": {},
        "report_ack_v": 1,
        "next_queue_order": max((job["queue_order"] for job in jobs.values()), default=0),
        "archived_jobs": 0,
        "archived_counts": {},
        "archived_project_counts": {},
        "draining": False,
        "drain_requested": False,
        "launches_paused": False,
        "evacuation": None,
        "evacuation_requests": {},
        "evacuation_history": {},
        "evacuation_cancel_requests": {},
        "updated_at": ago(0),
    }


class HotStateTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"

    def _controller(self) -> Controller:
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
        return controller

    def _terminal(self, controller: Controller, count: int, **extra: Any) -> list[str]:
        """Journal ``count`` finished jobs, oldest first, in one commit."""

        job_ids = []
        with group_commit(controller):
            for index in range(count):
                job_id = f"done-{index:04d}"
                job = image(
                    job_id,
                    "succeeded",
                    index + 1,
                    finished_seconds_ago=10_000 - index,
                    project_id="koochak",
                    **extra,
                )
                controller.state["jobs"][job_id] = job
                emit(controller, "job.succeeded", job=job)
                job_ids.append(job_id)
        return job_ids

    def test_archiving_is_journaled_without_rotating_or_resetting_cursors(self) -> None:
        controller = self._controller()
        job_ids = self._terminal(controller, 150)
        for job_id in job_ids:
            (job_directory(self.root, job_id) / "stderr.log").write_text(job_id)
        cursor = observe(self.root)["next_cursor"]

        with group_commit(controller):
            self.assertTrue(compact_journal(controller))

        hot = load_state(self.root)
        self.assertEqual(0, hot["journal_generation"])
        terminal = [job for job in hot["jobs"].values() if job["state"] == "succeeded"]
        self.assertEqual(MAX_TERMINAL_JOBS, len(terminal))
        self.assertEqual(set(job_ids[50:]), {job["id"] for job in terminal})
        self.assertEqual({"succeeded": 50}, hot["archived_counts"])
        self.assertEqual({"succeeded": 50}, hot["archived_project_counts"]["koochak"])
        self.assertEqual(job_ids[:50], hot["retained_log_jobs"])
        archived = status(self.root, job_ids[0])
        self.assertTrue(archived["archived"])
        self.assertEqual("succeeded", archived["state"])
        self.assertTrue((self.root / "jobs" / job_ids[0] / "stderr.log").exists())
        response = observe(self.root, after=cursor)
        self.assertFalse(response["reset"])
        self.assertEqual(["jobs.archived"], [event["kind"] for event in response["events"]])
        self.assertEqual(job_ids[:50], response["events"][0]["data"]["job_ids"])
        self.assertEqual(150, summary(self.root)["counts"]["succeeded"])

    def test_archive_record_replays_once_after_a_lost_snapshot(self) -> None:
        controller = self._controller()
        job_ids = self._terminal(controller, 140)
        saved = {
            name: (self.root / name).read_bytes() for name in ("state.json", "cursor.json")
        }
        with group_commit(controller):
            compact_journal(controller)
        controller.journal.close()
        for name, payload in saved.items():
            (self.root / name).write_bytes(payload)

        recovered = load_recovered_state(self.root)

        self.assertNotIn(job_ids[0], recovered["jobs"])
        self.assertIn(job_ids[-1], recovered["jobs"])
        self.assertEqual({"succeeded": 40}, recovered["archived_counts"])
        self.assertEqual(40, recovered["archived_jobs"])
        self.assertEqual(40, len(recovered["retained_log_jobs"]))

    def test_rebuild_from_a_checkpoint_replays_later_archiving(self) -> None:
        controller = self._controller()
        state_module.rotate_journal(controller)
        job_ids = self._terminal(controller, 130)
        with group_commit(controller):
            compact_journal(controller)
        controller.journal.close()
        (self.root / "state.json").unlink()

        recovered = load_recovered_state(self.root)

        self.assertEqual(MAX_TERMINAL_JOBS, len(recovered["jobs"]))
        self.assertNotIn(job_ids[0], recovered["jobs"])
        self.assertEqual({"succeeded": 30}, recovered["archived_counts"])

    def test_archiving_a_backlog_is_bounded_per_tick(self) -> None:
        controller = self._controller()
        self._terminal(controller, 1300)
        archived_per_tick = []
        for _ in range(4):
            with group_commit(controller):
                before = len(controller.state["jobs"])
                compact_journal(controller)
                archived_per_tick.append(before - len(controller.state["jobs"]))
        self.assertEqual([512, 512, 176, 0], archived_per_tick)

    def test_pinned_terminal_jobs_stay_hot(self) -> None:
        controller = self._controller()
        job_ids = self._terminal(controller, 200)
        controller.state["jobs"][job_ids[0]].update(
            {
                "state": "lost",
                "reason": "allocation_replaced",
                "workflow_id": "flow",
                "task_id": "train",
                "attempt": 1,
                "recovery": {
                    "max_attempts": 2,
                    "retry_on": ["allocation_replaced"],
                    "evacuation": {"signal": "USR1", "grace_seconds": 60},
                },
            }
        )
        controller.state["evacuation"] = {
            "request_id": "evacuate-1",
            "state": "waiting",
            "targets": {job_ids[1]: {"outcome": "waiting"}},
        }
        _apply_tick(controller)
        self.assertIn(job_ids[0], controller.state["jobs"])
        self.assertIn(job_ids[1], controller.state["jobs"])
        self.assertNotIn(job_ids[2], controller.state["jobs"])

    def test_log_retention_outlives_hot_state_but_stays_bounded(self) -> None:
        controller = self._controller()
        job_ids = self._terminal(controller, 8)
        controller.state["jobs"]["never-ran"] = image(
            "never-ran", "skipped", 0, started=False, finished_seconds_ago=20_000
        )
        for job_id in job_ids:
            (job_directory(self.root, job_id) / "stdout.log").write_text(job_id)
        removed: list[str] = []
        original = state_module.remove_job_directories

        def record(root: Path, released: list[str]) -> int:
            # Directories are released only after the retention list is durable.
            self.assertEqual(job_ids[3:6], load_state(self.root)["retained_log_jobs"])
            removed.extend(released)
            return original(root, released)

        with (
            mock.patch("scruffy.state.remove_job_directories", side_effect=record),
            group_commit(controller),
        ):
            compact_journal(
                controller, max_terminal_jobs=2, terminal_slack=0, max_retained_logs=3
            )
        # Jobs that never ran have no logs and never displace ones that did.
        self.assertEqual(job_ids[3:6], controller.state["retained_log_jobs"])
        self.assertEqual(job_ids[:3], removed)
        for job_id in job_ids[:3]:
            self.assertFalse((self.root / "jobs" / job_id).exists())
        for job_id in job_ids[3:]:
            self.assertTrue((self.root / "jobs" / job_id).exists())
        self.assertNotIn("never-ran", controller.state["jobs"])



class LegacyRootMigrationTests(unittest.TestCase):
    """Adopt a large production-shaped root and shrink it on the first ticks."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"

    def _legacy_jobs(self, stale_waiters: int, history: int) -> dict[str, dict[str, Any]]:
        jobs: dict[str, dict[str, Any]] = {}
        order = 0

        def add(job: dict[str, Any]) -> None:
            jobs[job["id"]] = job

        # Producers that ended without publishing; each workflow has 10 waiters.
        for index in range(stale_waiters // 10):
            workflow_id = f"campaign/{index:04d}"
            order += 1
            state = ("failed", "cancelled", "lost")[index % 3]
            add(
                image(
                    f"train-{index:04d}",
                    state,
                    order,
                    workflow_id=workflow_id,
                    task_id="train",
                    project_id="koochak",
                    reason="allocation_replaced" if state == "lost" else None,
                )
            )
            for consumer in range(10):
                order += 1
                waiter = image(
                    f"eval-{index:04d}-{consumer}",
                    "blocked",
                    order,
                    workflow_id=workflow_id,
                    task_id=f"eval-{consumer}",
                    project_id="koochak",
                    reason="waiting_for_dependencies",
                )
                waiter["wait_for"] = [
                    {
                        "kind": "artifact",
                        "task_id": "train",
                        "artifact_id": f"checkpoint/step{consumer:03d}.pt",
                    }
                ]
                add(waiter)
        for index in range(history):
            order += 1
            add(
                image(
                    f"history-{index:05d}",
                    ("succeeded", "cancelled", "failed")[index % 3],
                    order,
                    project_id="koochak",
                    finished_seconds_ago=100_000 - index,
                )
            )
        # Live work that must survive untouched.
        for index in range(5):
            order += 1
            add(image(f"queued-{index}", "queued", order, project_id="koochak"))
        order += 1
        missing = image(
            "waits-for-unsubmitted",
            "blocked",
            order,
            workflow_id="later",
            task_id="infer",
            project_id="koochak",
        )
        missing["needs"] = [{"task_id": "train", "condition": "succeeded"}]
        add(missing)
        return jobs

    def _migrate(self, stale_waiters: int, history: int) -> dict[str, Any]:
        jobs = self._legacy_jobs(stale_waiters, history)
        write_state(self.root, legacy_state(self.root, jobs))
        original_bytes = (self.root / "state.json").stat().st_size
        writes: list[int] = []
        original_write = state_module.write_state

        def record(root: Path, state: dict[str, Any]) -> None:
            writes.append(len(state["jobs"]))
            original_write(root, state)

        controller = _initialize_controller(
            root=self.root,
            inventory=INVENTORY,
            launcher="local",
            allocation_id="local-allocation",
            slurm_job_id=None,
            poll_interval=0.01,
            cancel_grace=0,
            start_paused=True,
        )
        self.addCleanup(lambda: controller.journal.close())
        ticks = 0
        with mock.patch("scruffy.state.write_state", side_effect=record):
            while True:
                before = len(writes)
                hot_before = len(controller.state["jobs"])
                _apply_tick(controller)
                ticks += 1
                self.assertLessEqual(len(writes) - before, 1)
                if hot_before == len(controller.state["jobs"]) and not any(
                    job["state"] == "blocked" and job["id"].startswith("eval-")
                    for job in controller.state["jobs"].values()
                ):
                    break
                self.assertLess(ticks, 50)
        return {
            "jobs": jobs,
            "ticks": ticks,
            "original_bytes": original_bytes,
            "final_bytes": (self.root / "state.json").stat().st_size,
            "controller": controller,
        }

    def test_large_legacy_root_is_migrated_on_load(self) -> None:
        result = self._migrate(stale_waiters=1500, history=400)
        jobs = result["jobs"]
        hot = load_state(self.root)

        self.assertEqual({}, hot["bulk_operations"])
        live = {job_id for job_id, job in hot["jobs"].items() if job["state"] not in {
            "succeeded", "failed", "cancelled", "lost", "skipped", "rejected"
        }}
        self.assertEqual(
            {f"queued-{index}" for index in range(5)} | {"waits-for-unsubmitted"}, live
        )
        self.assertLessEqual(len(hot["jobs"]) - len(live), MAX_TERMINAL_JOBS)
        self.assertLess(result["final_bytes"] * 10, result["original_bytes"])
        # 1,500 skips at 512 per tick, then the archive backlog drains.
        self.assertLessEqual(result["ticks"], 12)

        counts = summary(self.root)["counts"]
        self.assertEqual(len(jobs), sum(counts.values()))
        self.assertEqual(1500, counts["skipped"])
        skipped = status(self.root, "eval-0000-0")
        self.assertTrue(skipped["archived"])
        self.assertEqual("skipped", skipped["state"])
        self.assertEqual("condition_unsatisfied", skipped["reason"])
        self.assertEqual("blocked", status(self.root, "waits-for-unsubmitted")["state"])
        self.assertTrue(
            (self.root / "provenance" / "eval-0000-0" / "result.json").exists()
        )
        kinds = {event["kind"] for event in read_events(self.root)}
        self.assertIn("jobs.archived", kinds)
        self.assertIn("job.skipped", kinds)

    def test_archives_written_by_the_old_release_remain_readable(self) -> None:
        jobs = self._legacy_jobs(stale_waiters=20, history=10)
        old = jobs.pop("history-00000")
        archive_terminal_job(self.root, old)
        write_state(self.root, legacy_state(self.root, jobs))

        self.assertEqual(len(jobs), len(status(self.root)["jobs"]))
        archived = status(self.root, "history-00000")
        self.assertTrue(archived["archived"])
        recovered = load_recovered_state(self.root)
        self.assertEqual([], recovered["retained_log_jobs"])
        self.assertEqual({}, recovered["bulk_operations"])
        self.assertEqual(len(jobs), len(recovered["jobs"]))


if __name__ == "__main__":
    unittest.main()
