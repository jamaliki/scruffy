"""One idempotent command cancels many jobs by explicit IDs or by filter."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import scruffy.state as state_module
from scruffy.bulk import cancel_preview, cancel_selector, selector_matches
from scruffy.cli import main
from scruffy.client import (
    cancel_jobs,
    preview_cancel_jobs,
    submit_job,
    summary,
    wait_for_command,
)
from scruffy.controller import (
    _discard_journaled_commands,
    _ingest_commands,
    _initialize_controller,
)
from scruffy.mcp_server import dispatch_tool
from scruffy.models import NodeInventory, ResourceRequest
from scruffy.runtime import Controller, RunningProcess
from scruffy.storage import (
    StorageError,
    archive_terminal_job,
    command_receipt,
    command_sources,
    load_state,
    read_events,
    utc_now,
)

REQUEST = ResourceRequest(1, 1, 1, 1)
INVENTORY = (NodeInventory("local", (0,), 2, 2),)


def job(
    job_id: str,
    *,
    state: str = "blocked",
    order: int = 1,
    project_id: str = "default",
    workflow_id: str | None = "flow",
    name: str | None = None,
    request_id: str | None = None,
    submitted_at: str | None = None,
) -> dict[str, Any]:
    image: dict[str, Any] = {
        "id": job_id,
        "name": name or job_id,
        "state": state,
        "submitted_at": submitted_at or utc_now(),
        "queue_order": order,
        "argv": ["true"],
        "cwd": "/tmp",
        "env": {},
        "request": REQUEST.to_dict(),
        "assignment": None,
        "request_id": request_id,
    }
    if project_id != "default":
        image["project_id"] = project_id
    if workflow_id is not None:
        image.update(
            {
                "workflow_id": workflow_id,
                "task_id": job_id,
                "needs": [{"task_id": "never-submitted", "condition": "succeeded"}],
                "blockers": [],
            }
        )
    if state in {"succeeded", "failed", "cancelled", "skipped"}:
        image["finished_at"] = utc_now()
    return image


class SelectorTests(unittest.TestCase):
    def test_selectors_are_canonical_and_reject_unsafe_shapes(self) -> None:
        self.assertEqual(
            {"job_ids": ["a", "b"]},
            cancel_selector({"job_ids": ["b", "a", "b"]}),
        )
        self.assertEqual(
            {"states": ["blocked", "queued"], "project_id": "p"},
            cancel_selector({"states": ["queued", "blocked"], "project_id": "p"}),
        )
        for invalid in (
            {},
            {"project_id": "p"},
            {"workflow_id_prefix": "x"},
            {"states": []},
            {"states": ["succeeded"]},
            {"states": ["blocked"], "colour": "red"},
            {"job_ids": []},
            {"job_ids": ["a/b"]},
            {"job_ids": "job-1"},
            {"states": ["blocked"], "submitted_before": "yesterday"},
            {"states": ["blocked"], "submitted_before": "2026-10-01T00:00:00"},
            {"states": ["blocked"], "name_prefix": ""},
        ):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                cancel_selector(invalid)
        with self.assertRaises(ValueError):
            cancel_selector({"job_ids": [f"job-{index}" for index in range(10_001)]})

    def test_filters_match_their_documented_fields(self) -> None:
        image = job(
            "job-1",
            project_id="koochak",
            workflow_id="campaign-7/seed-1",
            name="train-seed-1",
            request_id="agent/campaign-7/train",
            submitted_at="2026-09-30T00:00:00.000+00:00",
        )
        matching = [
            {"job_ids": ["job-1"]},
            {"states": ["blocked"]},
            {"states": ["blocked"], "project_id": "koochak"},
            {"states": ["blocked"], "workflow_id": "campaign-7/seed-1"},
            {"states": ["blocked"], "workflow_id_prefix": "campaign-7/"},
            {"states": ["blocked"], "request_id_prefix": "agent/campaign-7"},
            {"states": ["blocked"], "name_prefix": "train-"},
            {"states": ["blocked"], "submitted_before": "2026-10-01T00:00:00Z"},
        ]
        missing = [
            {"job_ids": ["job-2"]},
            {"states": ["queued"]},
            {"states": ["blocked"], "project_id": "default"},
            {"states": ["blocked"], "workflow_id": "campaign-7"},
            {"states": ["blocked"], "workflow_id_prefix": "campaign-8"},
            {"states": ["blocked"], "request_id_prefix": "other"},
            {"states": ["blocked"], "name_prefix": "infer-"},
            {"states": ["blocked"], "submitted_before": "2026-09-29T00:00:00Z"},
        ]
        for selector in matching:
            with self.subTest(selector=selector):
                self.assertTrue(selector_matches(image, cancel_selector(selector)))
        for selector in missing:
            with self.subTest(selector=selector):
                self.assertFalse(selector_matches(image, cancel_selector(selector)))

    def test_preview_counts_only_cancellable_matches(self) -> None:
        jobs = [
            job("a", state="blocked"),
            job("b", state="running"),
            job("c", state="succeeded"),
            job("d", state="submitted"),
        ]
        preview = cancel_preview(jobs, cancel_selector({"job_ids": ["a", "b", "c", "d", "e"]}))
        self.assertEqual(4, preview["matched"])
        self.assertEqual(2, preview["would_cancel"])
        self.assertEqual(1, preview["pending_admission"])
        self.assertEqual(1, preview["not_in_hot_state"])
        self.assertEqual(["a", "b"], preview["sample_job_ids"])


class BulkCancelControllerTests(unittest.TestCase):
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

    def _seed(self, controller: Controller, *images: dict[str, Any]) -> None:
        for image in images:
            controller.state["jobs"][image["id"]] = image
        state_module.commit_snapshot(controller)

    def _receipt_outcome(self, request_id: str) -> dict[str, Any]:
        receipt = command_receipt(self.root, request_id)
        self.assertIsNotNone(receipt)
        return receipt["outcome"]

    def test_filter_cancels_only_matching_jobs_with_one_snapshot_write(self) -> None:
        controller = self._controller()
        stale = [job(f"stale-{index}", order=index, project_id="koochak") for index in range(30)]
        self._seed(
            controller,
            *stale,
            job("queued-koochak", state="queued", order=40, project_id="koochak"),
            job("blocked-other", order=41, project_id="other"),
            job("done", state="succeeded", order=42, project_id="koochak"),
        )
        request = cancel_jobs(
            self.root, states=["blocked"], project_id="koochak", request_id="cleanup-1"
        )
        self.assertEqual("cancel_requested", request["state"])

        with mock.patch(
            "scruffy.state.write_state", wraps=state_module.write_state
        ) as write:
            _ingest_commands(controller)

        self.assertEqual(1, write.call_count)
        persisted = load_state(self.root)["jobs"]
        self.assertTrue(all(persisted[item["id"]]["state"] == "cancelled" for item in stale))
        self.assertTrue(
            all(persisted[item["id"]]["cancel_request_id"] == "cleanup-1" for item in stale)
        )
        self.assertEqual("queued", persisted["queued-koochak"]["state"])
        self.assertEqual("blocked", persisted["blocked-other"]["state"])
        outcome = self._receipt_outcome("cleanup-1")
        self.assertEqual("completed", outcome["state"])
        self.assertEqual(
            {"matched": 30, "cancelled": 30, "cancelling": 0, "ignored": 0, "unknown": 0},
            outcome["counts"],
        )
        self.assertEqual([], command_sources(self.root))
        events = read_events(self.root)
        completed = [event for event in events if event["kind"] == "jobs.cancel_completed"]
        self.assertEqual(1, len(completed))
        self.assertEqual("koochak", completed[0]["data"]["project_id"])
        per_job = [event for event in events if event["kind"] == "job.cancelled"]
        self.assertEqual(30, len(per_job))
        self.assertTrue(all("request_id" not in event["data"] for event in per_job))
        self.assertTrue(
            all(event["data"]["bulk_request_id"] == "cleanup-1" for event in per_job)
        )

    def test_retrying_the_same_request_is_idempotent(self) -> None:
        controller = self._controller()
        self._seed(controller, job("a"), job("b"))
        first = cancel_jobs(self.root, job_ids=["a", "b"], request_id="retry-me")
        second = cancel_jobs(self.root, job_ids=["b", "a"], request_id="retry-me")
        self.assertEqual(first, second)
        self.assertEqual(1, len(command_sources(self.root)))
        _ingest_commands(controller)
        third = cancel_jobs(self.root, job_ids=["a", "b"], request_id="retry-me")
        self.assertEqual(first, third)
        self.assertEqual([], command_sources(self.root))
        _ingest_commands(controller)
        self.assertEqual(
            1,
            sum(
                event["kind"] == "jobs.cancel_completed" for event in read_events(self.root)
            ),
        )
        with self.assertRaises(StorageError):
            cancel_jobs(self.root, job_ids=["a"], request_id="retry-me")

    def test_explicit_ids_report_cancelled_ignored_and_unknown(self) -> None:
        controller = self._controller()
        running = job("running", state="running", order=2, workflow_id=None)
        archived = job("archived", state="succeeded", order=3)
        archive_terminal_job(self.root, archived)
        self._seed(
            controller,
            job("blocked", order=1),
            running,
            job("finished", state="failed", order=4),
        )
        controller.running["running"] = RunningProcess(mock.Mock(), None)
        observed: list[str] = []

        def stop(_controller: Controller, _running: RunningProcess) -> None:
            observed.append(load_state(self.root)["jobs"]["running"]["state"])

        cancel_jobs(
            self.root,
            job_ids=["blocked", "running", "finished", "archived", "never-existed"],
            request_id="explicit",
        )
        with mock.patch("scruffy.controller.stop_launcher", side_effect=stop):
            _ingest_commands(controller)

        self.assertEqual(["cancelling"], observed)
        outcome = self._receipt_outcome("explicit")
        self.assertEqual(
            {"matched": 4, "cancelled": 1, "cancelling": 1, "ignored": 2, "unknown": 1},
            outcome["counts"],
        )
        self.assertEqual(["never-existed"], outcome["unknown_job_ids"])

    def test_large_operation_advances_over_ticks_and_survives_a_restart(self) -> None:
        controller = self._controller()
        images = [job(f"job-{index:02d}", order=index) for index in range(10)]
        self._seed(controller, *images)
        cancel_jobs(self.root, states=["blocked"], request_id="ticks")

        _ingest_commands(controller, bulk_limit=4)
        operation = load_state(self.root)["bulk_operations"]["ticks"]
        self.assertEqual(4, operation["position"])
        self.assertEqual(
            [
                {
                    "request_id": "ticks",
                    "kind": "cancel_jobs",
                    "started_at": operation["started_at"],
                    "processed": 4,
                    "total": 10,
                    "counts": operation["counts"],
                }
            ],
            summary(self.root)["bulk_operations"],
        )
        self.assertEqual(1, len(command_sources(self.root)))
        self.assertIsNone(command_receipt(self.root, "ticks"))

        # The second batch reaches the journal, but its snapshot is lost.
        with (
            mock.patch("scruffy.state.write_state", side_effect=OSError("EIO")),
            self.assertRaises(OSError),
        ):
            _ingest_commands(controller, bulk_limit=4)
        controller.journal.close()

        restarted = self._controller()
        self.assertEqual(4, restarted.state["bulk_operations"]["ticks"]["position"])
        _discard_journaled_commands(restarted)
        self.assertEqual(1, len(command_sources(self.root)))
        _ingest_commands(restarted, bulk_limit=4)
        _ingest_commands(restarted, bulk_limit=4)

        outcome = self._receipt_outcome("ticks")
        self.assertEqual(
            {"matched": 10, "cancelled": 10, "cancelling": 0, "ignored": 0, "unknown": 0},
            outcome["counts"],
        )
        self.assertEqual({}, load_state(self.root)["bulk_operations"])
        self.assertEqual([], command_sources(self.root))
        cancelled = [
            event["job_id"]
            for event in read_events(self.root)
            if event["kind"] == "job.cancelled"
        ]
        self.assertEqual(sorted(image["id"] for image in images), sorted(cancelled))

    def test_named_job_awaiting_admission_defers_the_whole_command(self) -> None:
        controller = self._controller()
        self._seed(controller, job("known"))
        pending = submit_job(
            self.root,
            argv=["true"],
            name="pending",
            cwd=Path.cwd(),
            environment={},
            request=REQUEST,
            request_id="pending/1",
        )["job_id"]
        cancel_jobs(self.root, job_ids=["known", pending], request_id="wait-for-admission")

        _ingest_commands(controller)

        self.assertEqual("blocked", controller.state["jobs"]["known"]["state"])
        self.assertEqual(1, len(command_sources(self.root)))
        self.assertNotIn("wait-for-admission", controller.state["bulk_operations"])

    def test_invalid_selector_is_rejected_with_a_receipt(self) -> None:
        controller = self._controller()
        from scruffy.storage import submit_command

        submit_command(
            self.root,
            {"kind": "cancel_jobs", "request_id": "bad", "selector": {"project_id": "p"}},
        )
        _ingest_commands(controller)
        outcome = self._receipt_outcome("bad")
        self.assertEqual("rejected", outcome["state"])
        self.assertIn("state", outcome["reason"])
        rejected = [
            event for event in read_events(self.root) if event["kind"] == "command.rejected"
        ]
        self.assertEqual("bad", rejected[-1]["data"]["request_id"])

    def test_restart_acknowledges_a_journaled_completion_with_its_outcome(self) -> None:
        controller = self._controller()
        self._seed(controller, job("a"), job("b"))
        cancel_jobs(self.root, job_ids=["a", "b"], request_id="journaled")
        with mock.patch("scruffy.controller.acknowledge_commands"):
            _ingest_commands(controller)
        self.assertEqual(1, len(command_sources(self.root)))
        controller.journal.close()

        restarted = self._controller()
        _discard_journaled_commands(restarted)

        self.assertEqual([], command_sources(self.root))
        self.assertEqual(2, self._receipt_outcome("journaled")["counts"]["cancelled"])

    def test_preview_and_wait_agree_with_the_applied_operation(self) -> None:
        controller = self._controller()
        self._seed(
            controller,
            job("w-1", project_id="koochak", workflow_id="sweep-1"),
            job("w-2", project_id="koochak", workflow_id="sweep-2"),
            job("other", project_id="koochak", workflow_id="keep"),
        )
        preview = preview_cancel_jobs(
            self.root, states=["blocked"], project_id="koochak", workflow_id_prefix="sweep-"
        )
        self.assertTrue(preview["dry_run"])
        self.assertEqual(2, preview["would_cancel"])
        self.assertEqual(["w-1", "w-2"], preview["sample_job_ids"])
        cancel_jobs(
            self.root,
            states=["blocked"],
            project_id="koochak",
            workflow_id_prefix="sweep-",
            request_id="sweep-cleanup",
        )
        with self.assertRaises(TimeoutError):
            wait_for_command(self.root, "sweep-cleanup", timeout=0)
        _ingest_commands(controller)
        receipt = wait_for_command(self.root, "sweep-cleanup", timeout=1)
        self.assertEqual(2, receipt["outcome"]["counts"]["cancelled"])
        self.assertEqual("blocked", controller.state["jobs"]["other"]["state"])


class BulkCancelInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"

    def _run(self, *argv: str) -> tuple[int, Any]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SCRUFFY_PROJECT", None)
            code = main(["--root", str(self.root), *argv])
        return code, json.loads(output.getvalue())

    def test_cli_previews_and_submits_one_command(self) -> None:
        ids_file = self.root.parent / "ids.txt"
        ids_file.write_text("job-b\n\njob-a\n")
        code, preview = self._run(
            "cancel-jobs", "--job-ids-file", str(ids_file), "job-c", "--dry-run"
        )
        self.assertEqual(0, code)
        self.assertEqual(["job-a", "job-b", "job-c"], preview["selector"]["job_ids"])
        self.assertEqual(3, preview["not_in_hot_state"])
        self.assertEqual([], command_sources(self.root))

        code, request = self._run(
            "cancel-jobs",
            "--state",
            "blocked",
            "--state",
            "queued",
            "--project",
            "koochak",
            "--workflow-prefix",
            "campaign-",
            "--request-id",
            "cli-cleanup",
        )
        self.assertEqual(0, code)
        self.assertEqual("cli-cleanup", request["request_id"])
        self.assertEqual(
            {
                "states": ["blocked", "queued"],
                "project_id": "koochak",
                "workflow_id_prefix": "campaign-",
            },
            request["selector"],
        )
        self.assertEqual(1, len(command_sources(self.root)))

    def test_cli_requires_a_state_for_filters(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            code = main(["--root", str(self.root), "cancel-jobs", "--project", "koochak"])
        self.assertEqual(2, code)
        self.assertEqual([], command_sources(self.root))

    def test_mcp_cancel_jobs_is_project_pinned(self) -> None:
        with self.assertRaises(ValueError):
            asyncio.run(dispatch_tool(self.root, "cancel_jobs", {"states": ["blocked"]}))
        result = asyncio.run(
            dispatch_tool(
                self.root,
                "cancel_jobs",
                {"states": ["blocked"], "request_id": "mcp-cleanup"},
                project_id="koochak",
            )
        )
        self.assertEqual(
            {"states": ["blocked"], "project_id": "koochak"}, result["selector"]
        )
        preview = asyncio.run(
            dispatch_tool(
                self.root,
                "cancel_jobs",
                {"job_ids": ["job-1"], "dry_run": True},
                project_id="koochak",
            )
        )
        self.assertEqual({"job_ids": ["job-1"], "project_id": "koochak"}, preview["selector"])
        timed_out = asyncio.run(
            dispatch_tool(
                self.root,
                "cancel_jobs",
                {"states": ["queued"], "request_id": "mcp-wait", "wait_seconds": 0.1},
                project_id="koochak",
            )
        )
        self.assertTrue(timed_out["timed_out"])


if __name__ == "__main__":
    unittest.main()
