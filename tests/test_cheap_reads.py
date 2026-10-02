"""Reads stay bounded by live work, independent of queue history."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

from scruffy._compat import UTC
from scruffy.client import status, submit_job, summary
from scruffy.controller import _apply_tick, _initialize_controller
from scruffy.models import NodeInventory, ResourceRequest
from scruffy.storage import archive_terminal_job, queue_id, write_state
from scruffy.summary import QUEUE_VIEW_STATES, compact_job_page

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
    project_id: str | None = None,
    finished_seconds_ago: float = 7200,
) -> dict[str, Any]:
    job: dict[str, Any] = {
        "id": job_id,
        "name": job_id,
        "state": state,
        "submitted_at": ago(finished_seconds_ago + 120),
        "queue_order": order,
        "argv": ["python", "train.py", "--config", "/shared/config.yaml"],
        "cwd": "/shared/code/project",
        "env": {"PYTHONUNBUFFERED": "1"},
        "request": REQUEST.to_dict(),
        "assignment": None,
    }
    if project_id is not None:
        job["project_id"] = project_id
    if state not in {"queued", "blocked"}:
        job["started_at"] = ago(finished_seconds_ago + 60)
        job["finished_at"] = ago(finished_seconds_ago)
    if workflow_id is not None:
        job.update(
            {"workflow_id": workflow_id, "task_id": job_id, "needs": [], "wait_for": []}
        )
    return job


def legacy_state(root: Path, jobs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "v": 1,
        "queue_id": queue_id(root),
        "last_seq": 0,
        "journal_generation": 0,
        "journal_offset": 0,
        "allocation": None,
        "nodes": {},
        "jobs": jobs,
        "next_queue_order": max((job["queue_order"] for job in jobs.values()), default=0),
        "archived_jobs": 0,
        "archived_counts": {},
        "archived_project_counts": {},
        "draining": False,
    }


class CheapReadTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"

    def test_one_archived_job_is_read_without_the_snapshot(self) -> None:
        archive_terminal_job(self.root, image("old", "succeeded", 1, project_id="koochak"))
        write_state(
            self.root,
            legacy_state(self.root, {"live": image("live", "queued", 2)}),
        )
        with mock.patch(
            "scruffy.client.load_state", side_effect=AssertionError("decoded state.json")
        ):
            archived = status(self.root, "old")
            self.assertTrue(archived["archived"])
            self.assertEqual("succeeded", archived["state"])
            self.assertEqual("old", status(self.root, "old", project_id="koochak")["id"])
            with self.assertRaises(KeyError):
                status(self.root, "old", project_id="other")
        self.assertEqual("queued", status(self.root, "live")["state"])
        pending = submit_job(
            self.root,
            argv=["true"],
            name="pending",
            cwd=Path.cwd(),
            environment={},
            request=REQUEST,
            request_id="pending/1",
        )["job_id"]
        self.assertEqual("submitted", status(self.root, pending)["state"])
        with self.assertRaises(KeyError):
            status(self.root, "never-existed")

    def test_summaries_and_job_views_never_read_the_archive(self) -> None:
        for index in range(20):
            archive_terminal_job(
                self.root, image(f"old-{index}", "failed", index, workflow_id="flow")
            )
        write_state(
            self.root,
            legacy_state(
                self.root,
                {
                    "queued": image("queued", "queued", 30),
                    "blocked": image("blocked", "blocked", 31, workflow_id="flow"),
                },
            ),
        )
        untouchable = mock.Mock(side_effect=AssertionError("read the archive"))
        with (
            mock.patch("scruffy.client.find_archived_job", untouchable),
            mock.patch("scruffy.client.list_archived_workflow", untouchable),
            mock.patch("scruffy.storage.find_archived_job", untouchable),
            mock.patch("scruffy.storage.list_archived_workflow", untouchable),
        ):
            overview = summary(self.root)
            page = compact_job_page(
                status(self.root),
                states=QUEUE_VIEW_STATES,
                offset=0,
                limit=50,
                project_id=None,
                include_elapsed=False,
            )
        self.assertEqual(1, overview["counts"]["queued"])
        self.assertEqual(["queued"], [job["id"] for job in page["jobs"]])

    def test_snapshot_size_does_not_grow_with_history(self) -> None:
        sizes = {}
        for history in (500, 2500):
            root = self.root.parent / f"queue-{history}"
            jobs: dict[str, dict[str, Any]] = {
                f"live-{index}": image(f"live-{index}", "queued", index)
                for index in range(10)
            }
            for index in range(history):
                job_id = f"history-{index:05d}"
                jobs[job_id] = image(
                    job_id, "succeeded", 100 + index, finished_seconds_ago=10**6 - index
                )
            write_state(root, legacy_state(root, jobs))
            controller = _initialize_controller(
                root=root,
                inventory=INVENTORY,
                launcher="local",
                allocation_id="local-allocation",
                slurm_job_id=None,
                poll_interval=0.01,
                cancel_grace=0,
                start_paused=True,
            )
            self.addCleanup(lambda controller=controller: controller.journal.close())
            for _ in range(10):
                _apply_tick(controller)
            sizes[history] = (root / "state.json").stat().st_size
            self.assertEqual(len(jobs), sum(summary(root)["counts"].values()))
        self.assertLess(sizes[2500], sizes[500] * 1.5)
        self.assertLess(sizes[2500], 200_000)

    def test_summary_reports_controller_heartbeat_age(self) -> None:
        state = legacy_state(self.root, {})
        state["allocation"] = {"id": "1", "state": "running", "heartbeat_at": ago(3600)}
        write_state(self.root, state)
        age = summary(self.root)["allocation"]["heartbeat_age_seconds"]
        self.assertGreaterEqual(age, 3599)
        self.assertLess(age, 3700)


if __name__ == "__main__":
    unittest.main()
