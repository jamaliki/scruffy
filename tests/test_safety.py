from __future__ import annotations

import queue
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import scruffy.state as state_module
import scruffy.storage as storage_module
from scruffy.client import (
    cancel_job,
    observe,
    publish_event,
    resume_queue,
    status,
    submit_job,
)
from scruffy.controller import (
    _discard_journaled_commands,
    _discard_journaled_reports,
    _ingest_commands,
    _ingest_reports,
    _ingest_requests,
    _initialize_controller,
    _refresh_dependencies,
    _report_batch,
)
from scruffy.lifecycle import (
    begin_shutdown,
    drain_messages,
    poll_processes,
    schedule,
    start_job,
)
from scruffy.models import (
    Assignment,
    NodeInventory,
    NodeReservation,
    ResourceRequest,
)
from scruffy.runtime import Controller, OutputNotifier, RunningProcess
from scruffy.slurm import SlurmStep, SlurmStepResult
from scruffy.slurm_runtime import reconcile_slurm, refresh_slurm_snapshot
from scruffy.state import compact_journal, emit, load_recovered_state
from scruffy.storage import (
    TransientStorageError,
    append_event,
    archive_terminal_job,
    create_journal_generation,
    job_directory,
    list_commands,
    list_reports,
    load_state,
    open_journal,
    queue_id,
    read_events,
    submit_command,
    utc_now,
    write_state,
)
from scruffy.workflows import resolve_blocked_jobs

REQUEST = ResourceRequest(1, 1, 1, 1)


def assignment(job_id: str, node: str, gpu_id: int = 0) -> Assignment:
    return Assignment(
        job_id,
        REQUEST,
        (NodeReservation(node, (gpu_id,), 1, 1),),
    )


def job_image(job_id: str, node: str) -> dict[str, object]:
    return {
        "id": job_id,
        "name": job_id,
        "state": "running",
        "submitted_at": utc_now(),
        "queue_order": 1,
        "argv": ["true"],
        "cwd": "/tmp",
        "env": {},
        "request": REQUEST.to_dict(),
        "assignment": assignment(job_id, node).to_dict(),
    }


class RecoverySafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"

    def test_missing_snapshot_is_rebuilt_from_complete_job_events(self) -> None:
        job = job_image("job-replayed", "old-node")
        job["state"] = "succeeded"
        job["assignment"] = None
        with open_journal(self.root) as journal:
            append_event(
                journal,
                {
                    "seq": 1,
                    "kind": "job.succeeded",
                    "allocation_id": "old-allocation",
                    "job_id": job["id"],
                    "job": job,
                },
                sync=True,
            )

        state = load_recovered_state(self.root)

        self.assertEqual(1, state["last_seq"])
        self.assertEqual("old-allocation", state["allocation"]["id"])
        self.assertEqual("succeeded", state["jobs"]["job-replayed"]["state"])

    def test_same_slurm_allocation_reattaches_active_jobs(self) -> None:
        job = job_image("job-active", "gpu-3")
        job["launch_token"] = "scruffy-token"
        job["slurm_step_id"] = "240292.7"
        with open_journal(self.root) as journal:
            append_event(
                journal,
                {
                    "seq": 1,
                    "kind": "job.running",
                    "allocation_id": "240292",
                    "job_id": job["id"],
                    "job": job,
                },
                sync=True,
            )

        controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("gpu-3", (0,), 2, 2),),
            launcher="slurm",
            allocation_id="240292",
            slurm_job_id="240292",
            poll_interval=0.1,
            cancel_grace=30,
        )
        self.addCleanup(controller.journal.close)

        running = controller.running["job-active"]
        self.assertIsNone(running.process)
        self.assertEqual("scruffy-token", running.step_name)
        self.assertEqual("running", controller.state["jobs"]["job-active"]["state"])
        self.assertIsNotNone(controller.state["jobs"]["job-active"]["assignment"])
        self.assertTrue(controller.state["launches_paused"])

    def test_same_allocation_restart_requires_explicit_resume_to_launch(self) -> None:
        queued = job_image("job-queued", "gpu-3")
        queued.update({"state": "queued", "assignment": None})
        upstream = job_image("job-upstream", "gpu-3")
        upstream.update(
            {
                "state": "succeeded",
                "assignment": None,
                "workflow_id": "recovery-flow",
                "task_id": "upstream",
                "needs": [],
            }
        )
        blocked = job_image("job-blocked", "gpu-3")
        blocked.update(
            {
                "state": "blocked",
                "assignment": None,
                "queue_order": 2,
                "workflow_id": "recovery-flow",
                "task_id": "blocked",
                "needs": [{"task_id": "upstream", "condition": "succeeded"}],
                "blockers": [],
                "dependency_gate_passed": False,
            }
        )
        state = {
            "v": 1,
            "queue_id": queue_id(self.root),
            "last_seq": 0,
            "allocation": {"id": "240292"},
            "nodes": {},
            "jobs": {
                "job-queued": queued,
                "job-upstream": upstream,
                "job-blocked": blocked,
            },
            "draining": False,
        }
        write_state(self.root, state)
        controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("gpu-3", (0,), 2, 2),),
            launcher="slurm",
            allocation_id="240292",
            slurm_job_id="240292",
            poll_interval=0.1,
            cancel_grace=30,
        )
        self.addCleanup(controller.journal.close)

        _refresh_dependencies(controller)
        self.assertEqual("queued", controller.state["jobs"]["job-blocked"]["state"])
        with mock.patch("scruffy.lifecycle.start_job") as start:
            schedule(controller)
        start.assert_not_called()

        requested = resume_queue(self.root)
        _ingest_commands(controller)
        self.assertFalse(controller.state["launches_paused"])
        resumed = next(
            event
            for event in read_events(self.root)
            if event["kind"] == "allocation.launches_resumed"
        )
        self.assertEqual(requested["request_id"], resumed["data"]["request_id"])

        def mark_started(
            _controller: Controller, job: dict[str, object], _assignment: Assignment
        ) -> None:
            job["state"] = "starting"

        with mock.patch("scruffy.lifecycle.start_job", side_effect=mark_started) as start:
            schedule(controller)
        self.assertEqual(2, start.call_count)

    def test_same_slurm_allocation_refuses_job_without_launch_token(self) -> None:
        state = {
            "v": 1,
            "queue_id": queue_id(self.root),
            "last_seq": 0,
            "allocation": {"id": "240292"},
            "nodes": {},
            "jobs": {"job-active": job_image("job-active", "gpu-3")},
            "draining": False,
        }
        write_state(self.root, state)

        with self.assertRaisesRegex(RuntimeError, "no launch token"):
            _initialize_controller(
                root=self.root,
                inventory=(NodeInventory("gpu-3", (0,), 2, 2),),
                launcher="slurm",
                allocation_id="240292",
                slurm_job_id="240292",
                poll_interval=0.1,
                cancel_grace=30,
            )

    def test_local_restart_with_active_jobs_always_fails_closed(self) -> None:
        state = {
            "v": 1,
            "queue_id": queue_id(self.root),
            "last_seq": 0,
            "allocation": {"id": "old-local"},
            "nodes": {},
            "jobs": {"job-a": job_image("job-a", "local")},
            "draining": False,
        }
        write_state(self.root, state)

        with self.assertRaisesRegex(RuntimeError, "unsafe recovery"):
            _initialize_controller(
                root=self.root,
                inventory=(NodeInventory("local", (0,), 2, 2),),
                launcher="local",
                allocation_id="new-local",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
            )

    def test_replacement_slurm_allocation_clears_old_assignments(self) -> None:
        state = {
            "v": 1,
            "queue_id": queue_id(self.root),
            "last_seq": 0,
            "allocation": {"id": "old-allocation"},
            "nodes": {},
            "jobs": {
                "job-a": job_image("job-a", "old-a"),
                "job-b": job_image("job-b", "old-b"),
            },
            "draining": False,
            "updated_at": utc_now(),
        }
        write_state(self.root, state)

        controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("new-node", (0,), 2, 2),),
            launcher="slurm",
            allocation_id="new-allocation",
            slurm_job_id="new-allocation",
            poll_interval=0.01,
            cancel_grace=0,
        )
        self.addCleanup(lambda: controller.journal.close())

        self.assertEqual({"new-node"}, set(controller.state["nodes"]))
        for job in controller.state["jobs"].values():
            self.assertEqual("lost", job["state"])
            self.assertIsNone(job["assignment"])
            self.assertIsNotNone(job["last_assignment"])

    def test_compaction_bounds_hot_history_and_keeps_late_workflows(self) -> None:
        controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("local", (0,), 2, 2),),
            launcher="local",
            allocation_id="local-allocation",
            slurm_job_id=None,
            poll_interval=0.1,
            cancel_grace=0,
        )
        self.addCleanup(lambda: controller.journal.close())
        train = submit_job(
            self.root,
            argv=["true"],
            name="train",
            cwd=Path.cwd(),
            environment={},
            request=REQUEST,
            request_id="retention/train",
            project_id="retention-project",
            workflow_id="retention-flow",
            task_id="train",
        )
        recent = submit_job(
            self.root,
            argv=["true"],
            name="recent",
            cwd=Path.cwd(),
            environment={},
            request=REQUEST,
            request_id="retention/recent",
        )
        _ingest_requests(controller)
        for job_id, finished_at in (
            (train["job_id"], "2026-08-03T10:00:00+00:00"),
            (recent["job_id"], "2026-08-03T11:00:00+00:00"),
        ):
            job = controller.state["jobs"][job_id]
            job["state"] = "succeeded"
            job["finished_at"] = finished_at
            emit(controller, "job.succeeded", job=job)
            (job_directory(self.root, job_id) / "stdout.log").write_text("done")

        self.assertTrue(
            compact_journal(
                controller,
                max_bytes=1,
                max_terminal_jobs=1,
                terminal_slack=0,
            )
        )

        self.assertNotIn(train["job_id"], controller.state["jobs"])
        self.assertIn(recent["job_id"], controller.state["jobs"])
        self.assertEqual("succeeded", status(self.root, train["job_id"])["state"])
        self.assertFalse((self.root / "jobs" / train["job_id"]).exists())
        self.assertTrue((self.root / "jobs" / recent["job_id"]).exists())
        self.assertEqual(1, status(self.root)["archived_jobs"])
        self.assertEqual(
            {"succeeded": 1},
            controller.state["archived_project_counts"]["retention-project"],
        )

        infer = submit_job(
            self.root,
            argv=["true"],
            name="infer",
            cwd=Path.cwd(),
            environment={},
            request=REQUEST,
            request_id="retention/infer",
            project_id="retention-project",
            workflow_id="retention-flow",
            task_id="infer",
            needs=({"task_id": "train", "condition": "succeeded"},),
        )
        _ingest_requests(controller)
        self.assertEqual("queued", controller.state["jobs"][infer["job_id"]]["state"])

    def test_transient_request_read_is_retried_without_rejection(self) -> None:
        controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("local", (0,), 2, 2),),
            launcher="local",
            allocation_id="local-allocation",
            slurm_job_id=None,
            poll_interval=0.1,
            cancel_grace=0,
        )
        self.addCleanup(lambda: controller.journal.close())
        submitted = submit_job(
            self.root,
            argv=["true"],
            name="transient",
            cwd=Path.cwd(),
            environment={},
            request=REQUEST,
            request_id="transient-request",
        )
        job_id = str(submitted["job_id"])

        with mock.patch(
            "scruffy.storage.read_json",
            side_effect=TransientStorageError("ESTALE"),
        ):
            _ingest_requests(controller)

        self.assertNotIn(job_id, controller.state["jobs"])
        self.assertTrue((self.root / "requests" / job_id).exists())
        _ingest_requests(controller)
        self.assertEqual("queued", controller.state["jobs"][job_id]["state"])

    def test_compaction_archives_legacy_jobs_without_request_digests(self) -> None:
        controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("local", (0,), 2, 2),),
            launcher="local",
            allocation_id="local-allocation",
            slurm_job_id=None,
            poll_interval=0.1,
            cancel_grace=0,
        )
        self.addCleanup(lambda: controller.journal.close())
        for index in range(2):
            job_id = f"legacy-{index}"
            controller.state["jobs"][job_id] = {
                "id": job_id,
                "name": job_id,
                "state": "failed",
                "submitted_at": f"2026-08-03T10:0{index}:00+00:00",
                "finished_at": f"2026-08-03T10:0{index}:30+00:00",
                "queue_order": index,
            }

        self.assertTrue(
            compact_journal(
                controller,
                max_bytes=0,
                max_terminal_jobs=1,
                terminal_slack=0,
            )
        )

        self.assertEqual(1, len(controller.state["jobs"]))
        self.assertEqual("failed", status(self.root, "legacy-0")["state"])

    def test_checkpoint_recovery_replays_newer_workload_deltas(self) -> None:
        controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("local", (0,), 2, 2),),
            launcher="local",
            allocation_id="local-allocation",
            slurm_job_id=None,
            poll_interval=0.1,
            cancel_grace=0,
        )
        self.addCleanup(lambda: controller.journal.close())
        job = {
            "id": "job-delta",
            "name": "delta",
            "state": "succeeded",
            "queue_order": 1,
            "submitted_at": utc_now(),
            "finished_at": utc_now(),
        }
        controller.state["jobs"][job["id"]] = job
        emit(controller, "job.succeeded", job=job)
        self.assertTrue(
            compact_journal(
                controller,
                max_bytes=1,
                max_terminal_jobs=-1,
            )
        )
        publish_event(
            self.root,
            job_id=job["id"],
            event_id="delta-1",
            kind="workload.progress",
            data={"step": 9},
        )
        _ingest_reports(controller)
        (self.root / "state.json").unlink()
        controller.journal.close()

        recovered = load_recovered_state(self.root)

        self.assertEqual(9, recovered["jobs"][job["id"]]["workload"]["progress"]["step"])
        self.assertEqual(1, len(recovered["report_acks"]))

    def test_two_rotations_keep_one_fallback_and_recover_active_generation(self) -> None:
        controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("local", (0,), 2, 2),),
            launcher="local",
            allocation_id="local-allocation",
            slurm_job_id=None,
            poll_interval=0.1,
            cancel_grace=0,
        )
        self.addCleanup(lambda: controller.journal.close())
        submitted = submit_job(
            self.root,
            argv=["true"],
            name="rotate",
            cwd=Path.cwd(),
            environment={},
            request=REQUEST,
            request_id="rotation/job",
        )
        _ingest_requests(controller)
        job_id = submitted["job_id"]
        job = controller.state["jobs"][job_id]
        job["state"] = "succeeded"
        job["finished_at"] = utc_now()
        emit(controller, "job.succeeded", job=job)

        def report(step: int) -> None:
            publish_event(
                self.root,
                job_id=job_id,
                event_id=f"step-{step}",
                kind="workload.progress",
                data={"step": step},
            )
            _ingest_reports(controller)

        report(0)
        old_cursor = observe(self.root)["latest_cursor"]
        self.assertTrue(
            compact_journal(controller, max_bytes=1, max_terminal_jobs=-1)
        )
        report(1)
        create_journal_generation(
            self.root,
            2,
            {"queue_id": queue_id(self.root), "journal_generation": 2, "jobs": {}},
        )
        self.assertTrue(
            compact_journal(controller, max_bytes=1, max_terminal_jobs=-1)
        )
        report(2)

        journal_names = {source.name for source in (self.root / "journal").iterdir()}
        self.assertEqual(
            {
                "active.json",
                "checkpoint-000001.json",
                "checkpoint-000003.json",
                "events-000001.jsonl",
                "events-000003.jsonl",
            },
            journal_names,
        )
        self.assertFalse((self.root / "events.jsonl").exists())
        receipt_generations = {
            source.name
            for source in (self.root / "reports" / ".accepted").iterdir()
            if source.is_dir()
        }
        self.assertEqual({".g000001", ".g000003"}, receipt_generations)
        reset = observe(self.root, after=old_cursor)
        self.assertTrue(reset["reset"])
        self.assertEqual(3, reset["snapshot"]["journal_generation"])

        controller.journal.close()
        (self.root / "state.json").unlink()
        recovered = load_recovered_state(self.root)

        self.assertEqual(3, recovered["journal_generation"])
        self.assertEqual(
            2,
            recovered["jobs"][job_id]["workload"]["progress"]["step"],
        )

    def test_replacement_rejects_recovered_jobs_that_no_longer_fit(self) -> None:
        queued = job_image("job-too-large", "old-node")
        queued["state"] = "queued"
        queued["assignment"] = None
        queued["request"] = ResourceRequest(1, 2, 1, 1).to_dict()
        state = {
            "v": 1,
            "queue_id": queue_id(self.root),
            "last_seq": 0,
            "allocation": {"id": "old-allocation"},
            "nodes": {},
            "jobs": {"job-too-large": queued},
            "draining": False,
        }
        write_state(self.root, state)

        controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("new-node", (0,), 2, 2),),
            launcher="slurm",
            allocation_id="new-allocation",
            slurm_job_id="new-allocation",
            poll_interval=0.1,
            cancel_grace=30,
        )
        self.addCleanup(controller.journal.close)

        rejected = controller.state["jobs"]["job-too-large"]
        self.assertEqual("rejected", rejected["state"])
        self.assertEqual("request_cannot_fit", rejected["reason"])


class LaunchCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"
        self.controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("local", (0,), 2, 2),),
            launcher="local",
            allocation_id="local-allocation",
            slurm_job_id=None,
            poll_interval=0.01,
            cancel_grace=0,
        )
        self.addCleanup(self.controller.journal.close)

    def test_post_popen_reader_failure_holds_gpu_until_child_exit(self) -> None:
        job = {
            "id": "job-reader-failure",
            "name": "reader-failure",
            "state": "queued",
            "submitted_at": utc_now(),
            "queue_order": 1,
            "argv": ["sh", "-c", "sleep 10"],
            "cwd": str(self.root),
            "env": {},
            "request": REQUEST.to_dict(),
            "assignment": None,
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "signal": None,
            "reason": None,
            "error": None,
        }
        self.controller.state["jobs"][job["id"]] = job

        with mock.patch(
            "scruffy.runtime.threading.Thread.start",
            side_effect=RuntimeError("reader did not start"),
        ):
            start_job(self.controller, job, assignment(job["id"], "local"))

        self.assertIn(job["id"], self.controller.running)
        self.assertIsNotNone(job["assignment"])
        self.assertEqual("starting", job["state"])
        deadline = time.monotonic() + 5
        while self.controller.running and time.monotonic() < deadline:
            poll_processes(self.controller)
            time.sleep(0.01)

        self.assertFalse(self.controller.running)
        self.assertEqual("failed", job["state"])
        self.assertEqual("launch_failed", job["reason"])
        self.assertIsNone(job["assignment"])
        self.assertIsNotNone(job["last_assignment"])


class AsyncCommandRaceTests(unittest.TestCase):
    def test_dependency_refresh_reaches_a_fixed_point_in_one_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("local", (0,), 2, 2),),
                launcher="local",
                allocation_id="local-allocation",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
            )
            self.addCleanup(controller.journal.close)
            controller.state["jobs"] = {
                "grandchild": {
                    "id": "grandchild",
                    "state": "blocked",
                    "workflow_id": "flow",
                    "task_id": "grandchild",
                    "needs": [{"task_id": "child", "condition": "terminal"}],
                    "blockers": [],
                },
                "child": {
                    "id": "child",
                    "state": "blocked",
                    "workflow_id": "flow",
                    "task_id": "child",
                    "needs": [{"task_id": "root", "condition": "succeeded"}],
                    "blockers": [],
                },
                "root": {
                    "id": "root",
                    "state": "failed",
                    "workflow_id": "flow",
                    "task_id": "root",
                    "needs": [],
                },
            }

            with mock.patch(
                "scruffy.controller.resolve_blocked_jobs", wraps=resolve_blocked_jobs
            ) as resolve:
                _refresh_dependencies(controller)
                _refresh_dependencies(controller)

            self.assertEqual("skipped", controller.state["jobs"]["child"]["state"])
            self.assertEqual("queued", controller.state["jobs"]["grandchild"]["state"])
            self.assertEqual(1, resolve.call_count)
            self.assertEqual(
                ["child", "grandchild"],
                [
                    event["job_id"]
                    for event in read_events(root)
                    if event.get("kind") in {"job.skipped", "job.queued"}
                ],
            )

    def test_dependency_refresh_caches_clean_graphs_and_refreshes_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("local", (0,), 2, 2),),
                launcher="local",
                allocation_id="local-allocation",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
            )
            self.addCleanup(controller.journal.close)
            controller.state["jobs"] = {
                child_id: {
                    "id": child_id,
                    "state": "blocked",
                    "workflow_id": "flow",
                    "task_id": child_id,
                    "needs": [{"task_id": "root", "condition": "succeeded"}],
                    "blockers": [],
                }
                for child_id in ("child-a", "child-b")
            }

            with mock.patch(
                "scruffy.controller.resolve_blocked_jobs", wraps=resolve_blocked_jobs
            ) as resolve:
                _refresh_dependencies(controller)
                _refresh_dependencies(controller)
                self.assertEqual(1, resolve.call_count)
                self.assertTrue(
                    all(
                        job["blockers"][0]["reason"] == "dependency_missing"
                        for job in controller.state["jobs"].values()
                    )
                )

                controller.state["jobs"]["root"] = {
                    "id": "root",
                    "state": "running",
                    "workflow_id": "flow",
                    "task_id": "root",
                    "needs": [],
                }
                _refresh_dependencies(controller)
                self.assertEqual(2, resolve.call_count)
                self.assertTrue(
                    all(
                        job["blockers"][0]["state"] == "running"
                        for job_id, job in controller.state["jobs"].items()
                        if job_id.startswith("child-")
                    )
                )

                controller.state["jobs"]["root"]["state"] = "finishing"
                _refresh_dependencies(controller)
                self.assertEqual(3, resolve.call_count)
                self.assertTrue(
                    all(
                        job["blockers"][0]["state"] == "finishing"
                        for job_id, job in controller.state["jobs"].items()
                        if job_id.startswith("child-")
                    )
                )

                controller.state["jobs"]["root"]["state"] = "succeeded"
                _refresh_dependencies(controller)
                _refresh_dependencies(controller)

            self.assertEqual(4, resolve.call_count)
            self.assertEqual(
                {"queued"},
                {
                    job["state"]
                    for job_id, job in controller.state["jobs"].items()
                    if job_id.startswith("child-")
                },
            )

    def test_dependency_refresh_resolves_only_the_dirty_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("local", (0,), 2, 2),),
                launcher="local",
                allocation_id="local-allocation",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
            )
            self.addCleanup(controller.journal.close)
            controller.state["jobs"] = {}
            for workflow_id in ("flow-a", "flow-b"):
                controller.state["jobs"][f"{workflow_id}-root"] = {
                    "id": f"{workflow_id}-root",
                    "state": "running",
                    "workflow_id": workflow_id,
                    "task_id": "root",
                    "needs": [],
                }
                controller.state["jobs"][f"{workflow_id}-child"] = {
                    "id": f"{workflow_id}-child",
                    "state": "blocked",
                    "workflow_id": workflow_id,
                    "task_id": "child",
                    "needs": [{"task_id": "root", "condition": "succeeded"}],
                    "blockers": [],
                }

            with mock.patch(
                "scruffy.controller.resolve_blocked_jobs", wraps=resolve_blocked_jobs
            ) as resolve:
                _refresh_dependencies(controller)
                self.assertEqual(2, resolve.call_count)
                resolve.reset_mock()

                controller.state["jobs"]["flow-a-root"]["state"] = "finishing"
                _refresh_dependencies(controller)

            self.assertEqual(1, resolve.call_count)
            resolved_jobs = resolve.call_args.args[0]
            self.assertEqual(
                {"flow-a"},
                {job["workflow_id"] for job in resolved_jobs},
            )
            self.assertEqual(
                "running",
                controller.state["jobs"]["flow-b-child"]["blockers"][0]["state"],
            )

    def test_dependency_refresh_sanitizes_archived_invalid_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("local", (0,), 2, 2),),
                launcher="local",
                allocation_id="local-allocation",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
            )
            self.addCleanup(controller.journal.close)
            archive_terminal_job(
                root,
                {
                    "id": "bad",
                    "name": "bad",
                    "state": "rejected",
                    "request_digest": "a" * 64,
                    "workflow_id": "flow",
                    "task_id": "bad",
                    "needs": [{"task_id": "bad", "condition": "succeeded"}],
                    "workflow_invalid": True,
                },
            )
            controller.state["jobs"] = {
                "child": {
                    "id": "child",
                    "state": "blocked",
                    "workflow_id": "flow",
                    "task_id": "child",
                    "needs": [{"task_id": "root", "condition": "succeeded"}],
                    "blockers": [],
                }
            }

            _refresh_dependencies(controller)

            child = controller.state["jobs"]["child"]
            self.assertEqual("blocked", child["state"])
            self.assertEqual("dependency_missing", child["blockers"][0]["reason"])

    def test_transient_workflow_archive_read_retries_dependency_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("local", (0,), 2, 2),),
                launcher="local",
                allocation_id="local-allocation",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
            )
            self.addCleanup(controller.journal.close)
            archive_terminal_job(
                root,
                {
                    "id": "root",
                    "name": "root",
                    "state": "succeeded",
                    "queue_order": 1,
                    "request_digest": "a" * 64,
                    "workflow_id": "flow",
                    "task_id": "root",
                    "needs": [],
                },
            )
            controller.state["jobs"] = {
                "child": {
                    "id": "child",
                    "state": "blocked",
                    "workflow_id": "flow",
                    "task_id": "child",
                    "needs": [{"task_id": "root", "condition": "succeeded"}],
                    "blockers": [],
                    "dependency_gate_passed": False,
                }
            }

            with mock.patch(
                "scruffy.storage.read_json",
                side_effect=TransientStorageError("ESTALE"),
            ):
                _refresh_dependencies(controller)

            self.assertEqual("blocked", controller.state["jobs"]["child"]["state"])
            _refresh_dependencies(controller)
            self.assertEqual("queued", controller.state["jobs"]["child"]["state"])

    def test_batch_admission_never_snapshots_undecided_later_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("local", (0,), 2, 2),),
                launcher="local",
                allocation_id="local-allocation",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
            )
            self.addCleanup(controller.journal.close)
            for request_id in ("batch-a", "batch-b"):
                submit_job(
                    root,
                    argv=["true"],
                    name=request_id,
                    cwd=Path.cwd(),
                    environment={},
                    request=REQUEST,
                    request_id=request_id,
                )

            visible_jobs: list[set[str]] = []

            def record_emit(
                observed: Controller, _kind: str, **_kwargs: object
            ) -> dict[str, object]:
                visible_jobs.append(set(observed.state["jobs"]))
                return {}

            with mock.patch("scruffy.controller.emit", side_effect=record_emit):
                _ingest_requests(controller)

            self.assertEqual([1, 2], [len(job_ids) for job_ids in visible_jobs])

    def test_report_recovery_keys_idempotency_by_job_and_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("local", (0,), 2, 2),),
                launcher="local",
                allocation_id="local-allocation",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
            )
            self.addCleanup(controller.journal.close)
            for job_id in ("job-a", "job-b"):
                publish_event(
                    root,
                    job_id=job_id,
                    event_id="step-1",
                    kind="workload.progress",
                    data={"step": 1},
                )
            append_event(
                controller.journal,
                {
                    "seq": controller.state["last_seq"] + 1,
                    "kind": "workload.progress",
                    "job_id": "job-a",
                    "source_event_id": "step-1",
                },
                sync=True,
            )
            controller.state.pop("report_ack_v", None)

            _discard_journaled_reports(controller)

            pending = [document for _, document in list_reports(root)]
            self.assertEqual(["job-b"], [document["job_id"] for document in pending])

    def test_report_batch_round_robins_across_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("local", (0,), 2, 2),),
                launcher="local",
                allocation_id="local-allocation",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
            )
            self.addCleanup(controller.journal.close)
            for event_id in ("a-1", "a-2", "a-3"):
                publish_event(
                    root,
                    job_id="job-a",
                    event_id=event_id,
                    kind="workload.progress",
                    data={"event": event_id},
                )
            publish_event(
                root,
                job_id="job-b",
                event_id="b-1",
                kind="workload.progress",
                data={"event": "b-1"},
            )

            batch = _report_batch(controller, 2)

            self.assertEqual({"job-a", "job-b"}, {item[0].parent.name for item in batch})

    def test_transient_report_read_is_retried_without_rejection_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("local", (0,), 2, 2),),
                launcher="local",
                allocation_id="local-allocation",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
            )
            self.addCleanup(controller.journal.close)
            controller.state["jobs"]["job-report"] = {
                "id": "job-report",
                "name": "report",
                "state": "running",
            }
            report = {
                "job_id": "job-report",
                "event_id": "transient-progress",
                "kind": "workload.progress",
                "data": {"step": 1},
            }
            publish_event(root, **report)

            with mock.patch(
                "scruffy.storage.read_json",
                side_effect=TransientStorageError("ESTALE"),
            ):
                _ingest_reports(controller)

            self.assertEqual(1, len(list_reports(root)))
            self.assertTrue(publish_event(root, **report)["deduplicated"])
            _ingest_reports(controller)

            self.assertEqual([], list_reports(root))
            self.assertEqual(
                1,
                sum(
                    event.get("source_event_id") == "transient-progress"
                    for event in read_events(root)
                ),
            )

    def test_report_batch_has_one_journal_commit_and_one_state_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("local", (0,), 2, 2),),
                launcher="local",
                allocation_id="local-allocation",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
            )
            self.addCleanup(controller.journal.close)
            for index in range(128):
                job_id = f"job-{index:03d}"
                controller.state["jobs"][job_id] = {
                    "id": job_id,
                    "name": job_id,
                    "state": "succeeded",
                }
                publish_event(
                    root,
                    job_id=job_id,
                    event_id="progress-1",
                    kind="workload.progress",
                    data={"step": 1},
                )

            with (
                mock.patch(
                    "scruffy.state.sync_file", wraps=state_module.sync_file
                ) as sync,
                mock.patch(
                    "scruffy.state.write_state", wraps=storage_module.write_state
                ) as snapshot,
                mock.patch(
                    "scruffy.storage._fsync_directory",
                    wraps=storage_module._fsync_directory,
                ) as directory_sync,
            ):
                _ingest_reports(controller, limit=128)

            self.assertEqual(1, sync.call_count)
            self.assertEqual(1, snapshot.call_count)
            self.assertLessEqual(directory_sync.call_count, 6)
            workload_events = [
                event
                for event in read_events(root)
                if event.get("kind") == "workload.progress"
            ]
            self.assertEqual(128, len(workload_events))
            self.assertTrue(all("job" not in event for event in workload_events))

    def test_report_recovers_after_journal_sync_but_snapshot_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("local", (0,), 2, 2),),
                launcher="local",
                allocation_id="local-allocation",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
            )
            job = {
                "id": "job-recovery",
                "name": "recovery",
                "state": "succeeded",
                "queue_order": 1,
                "submitted_at": utc_now(),
                "finished_at": utc_now(),
            }
            controller.state["jobs"][job["id"]] = job
            emit(controller, "job.succeeded", job=job)
            publish_event(
                root,
                job_id=job["id"],
                event_id="step-1",
                kind="workload.progress",
                data={"step": 1},
            )

            with mock.patch("scruffy.state.write_state", side_effect=OSError("disk")):
                with self.assertRaises(OSError):
                    _ingest_reports(controller)
            recorded_at = controller.state["jobs"][job["id"]]["workload"][
                "last_recorded_at"
            ]
            controller.journal.close()

            restarted = _initialize_controller(
                root=root,
                inventory=(NodeInventory("local", (0,), 2, 2),),
                launcher="local",
                allocation_id="local-allocation-2",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
            )
            self.addCleanup(restarted.journal.close)
            self.assertEqual(
                1,
                restarted.state["jobs"][job["id"]]["workload"]["progress"]["step"],
            )
            self.assertEqual(
                recorded_at,
                restarted.state["jobs"][job["id"]]["workload"]["last_recorded_at"],
            )
            _discard_journaled_reports(restarted)

            self.assertEqual([], list_reports(root))
            workload_events = [
                event
                for event in read_events(root)
                if event.get("kind") == "workload.progress"
            ]
            self.assertEqual(1, len(workload_events))

    def test_report_recovers_after_snapshot_before_inbox_ack(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("local", (0,), 2, 2),),
                launcher="local",
                allocation_id="local-allocation",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
            )
            job = {
                "id": "job-ack-crash",
                "name": "ack-crash",
                "state": "succeeded",
                "queue_order": 1,
                "submitted_at": utc_now(),
                "finished_at": utc_now(),
            }
            controller.state["jobs"][job["id"]] = job
            emit(controller, "job.succeeded", job=job)
            publish_event(
                root,
                job_id=job["id"],
                event_id="step-1",
                kind="workload.progress",
                data={"step": 1},
            )

            with mock.patch(
                "scruffy.controller.accept_reports", side_effect=OSError("crash")
            ):
                with self.assertRaises(OSError):
                    _ingest_reports(controller)
            self.assertEqual(1, len(load_state(root)["report_acks"]))
            controller.journal.close()

            restarted = _initialize_controller(
                root=root,
                inventory=(NodeInventory("local", (0,), 2, 2),),
                launcher="local",
                allocation_id="local-allocation-2",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
            )
            self.addCleanup(restarted.journal.close)
            _discard_journaled_reports(restarted)

            self.assertEqual([], list_reports(root))
            self.assertEqual({}, restarted.state["report_acks"])
            self.assertEqual(
                1,
                sum(
                    event.get("kind") == "workload.progress"
                    for event in read_events(root)
                ),
            )

    def test_unprocessed_report_backlog_does_not_scan_the_journal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("local", (0,), 2, 2),),
                launcher="local",
                allocation_id="local-allocation",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
            )
            self.addCleanup(controller.journal.close)
            publish_event(
                root,
                job_id="job-backlog",
                event_id="pending",
                kind="workload.progress",
                data={"step": 1},
            )

            with mock.patch(
                "scruffy.controller.read_events",
                side_effect=AssertionError("journal scan"),
            ):
                _discard_journaled_reports(controller)

            self.assertEqual(1, len(list_reports(root)))
            self.assertEqual(1, load_state(root)["report_ack_v"])

    def test_command_recovery_discards_a_durably_journaled_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("local", (0,), 2, 2),),
                launcher="local",
                allocation_id="local-allocation",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
            )
            self.addCleanup(controller.journal.close)
            request_id = submit_command(
                root,
                {
                    "kind": "cancel",
                    "job_id": "job-done",
                    "request_id": "cancel-1",
                },
            )
            append_event(
                controller.journal,
                {
                    "seq": controller.state["last_seq"] + 1,
                    "kind": "job.cancel_ignored",
                    "data": {"request_id": request_id, "job_id": "job-done"},
                },
                sync=True,
            )

            _discard_journaled_commands(controller)

            self.assertEqual([], list_commands(root))

    def test_cancel_waits_for_a_durable_request_not_yet_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("local", (0,), 2, 2),),
                launcher="local",
                allocation_id="local-allocation",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
            )
            self.addCleanup(controller.journal.close)
            submitted = submit_job(
                root,
                argv=["true"],
                name="cancel-race",
                cwd=Path.cwd(),
                environment={},
                request=REQUEST,
                request_id="cancel-race",
            )
            cancellation = cancel_job(root, str(submitted["job_id"]))

            _ingest_commands(controller)

            self.assertEqual(1, len(list_commands(root)))
            self.assertFalse(
                any(event["kind"] == "command.rejected" for event in read_events(root))
            )

            _ingest_requests(controller)
            _ingest_commands(controller)

            self.assertEqual([], list_commands(root))
            job = controller.state["jobs"][submitted["job_id"]]
            self.assertEqual("cancelled", job["state"])
            cancelled = [
                event
                for event in read_events(root)
                if event["kind"] == "job.cancelled"
            ]
            self.assertEqual(cancellation["request_id"], cancelled[-1]["data"]["request_id"])

    def test_cancel_of_archived_job_is_ignored_not_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("local", (0,), 2, 2),),
                launcher="local",
                allocation_id="local-allocation",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
            )
            self.addCleanup(controller.journal.close)
            archive_terminal_job(
                root,
                {
                    "id": "job-archived-cancel",
                    "name": "archived",
                    "state": "succeeded",
                    "queue_order": 1,
                    "request_digest": "a" * 64,
                },
            )
            cancellation = cancel_job(root, "job-archived-cancel")

            _ingest_commands(controller)

            outcome = next(
                event
                for event in read_events(root)
                if event["kind"] == "job.cancel_ignored"
                and event["data"]["request_id"] == cancellation["request_id"]
            )
            self.assertEqual("job_is_succeeded", outcome["data"]["reason"])


class SlurmReleaseBarrierTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "queue"
        journal = open_journal(root)
        self.addCleanup(journal.close)
        messages: queue.SimpleQueue[dict[str, object]] = queue.SimpleQueue()
        self.controller = Controller(
            root=root,
            inventory=(NodeInventory("gpu-3", (0,), 2, 2),),
            launcher="slurm",
            allocation_id="240292",
            slurm_job_id="240292",
            poll_interval=0.1,
            cancel_grace=30,
            state={"queue_id": queue_id(root), "last_seq": 0, "jobs": {}, "nodes": {}},
            journal=journal,
            messages=messages,
            output=OutputNotifier(messages),
        )
        self.process = mock.Mock()
        self.process.poll.return_value = None

    def _running(self, token: str = "scruffy-token") -> RunningProcess:
        running = RunningProcess(self.process, token)
        running.exit_seen_at = 10
        return running

    def test_query_failure_and_stale_absence_never_release(self) -> None:
        job = {"launch_token": "scruffy-token", "slurm_step_id": "240292.7"}
        running = self._running()
        self.controller.slurm_snapshot_at = 9
        self.assertFalse(reconcile_slurm(self.controller, job, running))
        self.controller.slurm_snapshot_at = 11
        self.controller.slurm_query_error = "controller unavailable"
        self.assertFalse(reconcile_slurm(self.controller, job, running))

    def test_known_step_releases_only_after_fresh_absence(self) -> None:
        job = {"launch_token": "scruffy-token", "slurm_step_id": "240292.7"}
        running = self._running()
        self.controller.slurm_snapshot_at = 11
        self.controller.slurm_steps = (SlurmStep("240292.7", "scruffy-token", "gpu-3"),)
        self.assertFalse(reconcile_slurm(self.controller, job, running))
        self.controller.slurm_steps = ()
        self.controller.slurm_snapshot_at = 12
        self.assertTrue(reconcile_slurm(self.controller, job, running))

    def test_unidentified_step_requires_two_post_exit_absence_snapshots(self) -> None:
        job = {"launch_token": "scruffy-token"}
        running = self._running()
        self.controller.slurm_snapshot_at = 11
        self.assertFalse(reconcile_slurm(self.controller, job, running))
        self.controller.slurm_snapshot_at = 12
        self.assertTrue(reconcile_slurm(self.controller, job, running))

    def test_cancellation_targets_exact_step_and_keeps_assignment(self) -> None:
        reserved = assignment("job-a", "gpu-3").to_dict()
        job = {
            "id": "job-a",
            "state": "cancelling",
            "launch_token": "scruffy-token",
            "slurm_step_id": "240292.7",
            "assignment": reserved,
        }
        running = self._running()
        running.final_state = "cancelled"
        self.controller.slurm_snapshot_at = 11
        self.controller.slurm_steps = (SlurmStep("240292.7", "scruffy-token", "gpu-3"),)

        with mock.patch("scruffy.slurm_runtime.cancel_step") as cancel:
            self.assertFalse(reconcile_slurm(self.controller, job, running))

        cancel.assert_called_once_with("240292", "240292.7")
        self.assertEqual(reserved, job["assignment"])

    def test_reconciliation_queries_only_when_state_can_change(self) -> None:
        with mock.patch("scruffy.slurm_runtime.live_steps") as live:
            refresh_slurm_snapshot(self.controller, 10)
        live.assert_not_called()

        job = {
            "id": "job-a",
            "state": "running",
            "slurm_step_id": "240292.7",
            "assignment": None,
        }
        running = RunningProcess(self.process, "scruffy-token")
        self.controller.state["jobs"]["job-a"] = job
        self.controller.running["job-a"] = running
        with mock.patch("scruffy.slurm_runtime.live_steps") as live:
            refresh_slurm_snapshot(self.controller, 10)
        live.assert_not_called()

        job.pop("slurm_step_id")
        with mock.patch(
            "scruffy.slurm_runtime.live_steps", return_value=()
        ) as live:
            refresh_slurm_snapshot(self.controller, 10)
        live.assert_called_once_with("240292")
        self.assertEqual(10, self.controller.slurm_snapshot_at)

    def test_reattached_step_is_polled_until_it_disappears(self) -> None:
        job = {
            "id": "job-a",
            "state": "running",
            "slurm_step_id": "240292.7",
            "assignment": None,
        }
        self.controller.state["jobs"]["job-a"] = job
        self.controller.running["job-a"] = RunningProcess(None, "scruffy-token")

        with mock.patch(
            "scruffy.slurm_runtime.live_steps", return_value=()
        ) as live:
            refresh_slurm_snapshot(self.controller, 10)

        live.assert_called_once_with("240292")

    def test_graceful_slurm_stop_leaves_steps_for_the_next_controller(self) -> None:
        job = {"id": "job-a", "state": "running"}
        running = RunningProcess(self.process, "scruffy-token")
        self.controller.state.update(
            {
                "allocation": {"id": "240292", "state": "running"},
                "jobs": {"job-a": job},
                "draining": False,
            }
        )
        self.controller.running["job-a"] = running

        begin_shutdown(self.controller)

        self.assertIsNone(running.final_state)
        self.process.send_signal.assert_not_called()
        self.assertEqual("stopping", self.controller.state["allocation"]["state"])

    def test_reconciliation_error_is_visible_and_retries_are_bounded(self) -> None:
        job = {
            "id": "job-a",
            "state": "starting",
            "assignment": None,
            "launch_token": "scruffy-token",
        }
        self.controller.state["allocation"] = {"id": "240292"}
        self.controller.state["jobs"]["job-a"] = job
        self.controller.running["job-a"] = RunningProcess(
            self.process, "scruffy-token"
        )
        with mock.patch(
            "scruffy.slurm_runtime.live_steps",
            side_effect=RuntimeError("slurmctld unavailable"),
        ) as live:
            refresh_slurm_snapshot(self.controller, 10)
            refresh_slurm_snapshot(self.controller, 14)

        live.assert_called_once_with("240292")
        self.assertIn("slurmctld unavailable", self.controller.slurm_query_error or "")
        self.assertIn(
            "slurmctld unavailable",
            self.controller.state["allocation"]["reconciliation_error"],
        )

    def test_ambiguous_step_match_is_visible_on_the_job(self) -> None:
        job = {
            "id": "job-a",
            "state": "starting",
            "assignment": None,
            "launch_token": "scruffy-token",
        }
        self.controller.state["allocation"] = {"id": "240292"}
        self.controller.state["jobs"]["job-a"] = job
        self.controller.slurm_snapshot_at = 10
        self.controller.slurm_steps = (
            SlurmStep("240292.7", "scruffy-token", "gpu-3"),
            SlurmStep("240292.8", "scruffy-token", "gpu-3"),
        )

        self.assertFalse(reconcile_slurm(self.controller, job, self._running()))

        self.assertIn("multiple live steps", job["reconciliation_error"])

    def test_reattached_job_finishes_from_slurm_accounting(self) -> None:
        job = job_image("job-a", "gpu-3")
        job.update(
            {
                "launch_token": "scruffy-token",
                "slurm_step_id": "240292.7",
                "stdout": "jobs/job-a/stdout.log",
                "stderr": "jobs/job-a/stderr.log",
            }
        )
        self.controller.state.update(
            {
                "allocation": {"id": "240292", "state": "running"},
                "jobs": {"job-a": job},
            }
        )
        running = RunningProcess(None, "scruffy-token")
        running.closed_streams.update({"stdout", "stderr"})
        self.controller.running["job-a"] = running
        self.controller.slurm_snapshot_at = 10
        self.controller.slurm_steps = ()

        with (
            mock.patch("scruffy.lifecycle.refresh_slurm_snapshot"),
            mock.patch(
                "scruffy.lifecycle.completed_step",
                return_value=SlurmStepResult("COMPLETED", 0),
            ),
        ):
            poll_processes(self.controller)

        self.assertEqual("succeeded", job["state"])
        self.assertNotIn("job-a", self.controller.running)


class OutputCoalescingTests(unittest.TestCase):
    def test_many_chunks_create_one_pending_notification(self) -> None:
        messages: queue.SimpleQueue[dict[str, object]] = queue.SimpleQueue()
        notifier = OutputNotifier(messages)
        for offset in range(10_000):
            notifier.record("job-a", "stdout", offset, 1)

        self.assertEqual("output_ready", messages.get_nowait()["kind"])
        with self.assertRaises(queue.Empty):
            messages.get_nowait()
        self.assertEqual((0, 10_000), notifier.take("job-a", "stdout"))

    def test_large_ranges_are_split_into_bounded_events(self) -> None:
        messages: queue.SimpleQueue[dict[str, object]] = queue.SimpleQueue()
        notifier = OutputNotifier(messages)
        notifier.record("job-a", "stdout", 0, 70_000)

        self.assertEqual("output_ready", messages.get_nowait()["kind"])
        self.assertEqual((0, 65_536), notifier.take("job-a", "stdout"))
        self.assertEqual("output_ready", messages.get_nowait()["kind"])
        self.assertEqual((65_536, 4_464), notifier.take("job-a", "stdout"))
        with self.assertRaises(queue.Empty):
            messages.get_nowait()


class OutputProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"
        self.controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("local", (0,), 2, 2),),
            launcher="local",
            allocation_id="local-allocation",
            slurm_job_id=None,
            poll_interval=0.1,
            cancel_grace=0,
        )
        self.addCleanup(self.controller.journal.close)

    def test_message_drain_is_bounded_per_controller_tick(self) -> None:
        for index in range(300):
            self.controller.messages.put(
                {"kind": "output_error", "job_id": f"job-{index}", "error": "x"}
            )

        with mock.patch("scruffy.lifecycle.emit") as emit:
            drain_messages(self.controller)

        self.assertEqual(256, emit.call_count)
        self.assertEqual("job-256", self.controller.messages.get_nowait()["job_id"])

    def test_pending_output_is_a_terminal_event_barrier(self) -> None:
        job = job_image("job-output", "local")
        self.controller.state["jobs"]["job-output"] = job
        process = mock.Mock()
        process.poll.return_value = 0
        running = RunningProcess(process, None)
        running.closed_streams = {"stdout", "stderr"}
        self.controller.running["job-output"] = running
        self.controller.output.record("job-output", "stdout", 0, 70_000)

        poll_processes(self.controller)

        self.assertIn("job-output", self.controller.running)
        self.assertEqual("running", job["state"])
        drain_messages(self.controller)
        poll_processes(self.controller)

        self.assertNotIn("job-output", self.controller.running)
        self.assertEqual("succeeded", job["state"])
        relevant = [
            event
            for event in read_events(self.root)
            if event.get("job_id") == "job-output"
        ]
        output_sequences = [
            event["seq"] for event in relevant if event["kind"] == "job.output"
        ]
        terminal_sequence = next(
            event["seq"] for event in relevant if event["kind"] == "job.succeeded"
        )
        self.assertTrue(output_sequences)
        self.assertTrue(all(seq < terminal_sequence for seq in output_sequences))


if __name__ == "__main__":
    unittest.main()
