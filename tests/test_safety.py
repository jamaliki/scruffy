from __future__ import annotations

import hashlib
import json
import os
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
    _drain_for_deadline,
    _ingest_commands,
    _ingest_reports,
    _ingest_requests,
    _initialize_controller,
    _refresh_dependencies,
    _report_batch,
)
from scruffy.lifecycle import (
    _finish_job,
    _launch_arguments,
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
from scruffy.slurm import AllocationIncarnation, SlurmStep, SlurmStepResult
from scruffy.slurm_runtime import reconcile_slurm, refresh_slurm_snapshot
from scruffy.state import compact_journal, emit, load_recovered_state
from scruffy.storage import (
    TransientStorageError,
    append_event,
    archive_terminal_job,
    create_immutable_json,
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


def slurm_incarnation(
    job_id: str = "240292",
    *,
    restart_count: int = 0,
    node: str = "gpu-3",
) -> AllocationIncarnation:
    return AllocationIncarnation(
        slurm_job_id=job_id,
        restart_count=restart_count,
        inventory=(NodeInventory(node, (0,), 2, 2),),
    )


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

    def test_atomic_submission_replays_every_job_from_one_record(self) -> None:
        first = {**job_image("job-first", "old-node"), "state": "queued", "assignment": None}
        second = {
            **job_image("job-second", "old-node"),
            "state": "blocked",
            "assignment": None,
        }
        with open_journal(self.root) as journal:
            append_event(
                journal,
                {
                    "seq": 1,
                    "kind": "submission.admitted",
                    "allocation_id": "old-allocation",
                    "submission_id": "submission-one",
                    "jobs": [first, second],
                },
                sync=True,
            )

        recovered = load_recovered_state(self.root)

        self.assertEqual({"job-first", "job-second"}, set(recovered["jobs"]))
        self.assertEqual("queued", recovered["jobs"]["job-first"]["state"])
        self.assertEqual("blocked", recovered["jobs"]["job-second"]["state"])

    def test_durable_boundaries_replace_the_journal_handle(self) -> None:
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

        previous = controller.journal
        emit(controller, "notice", data={"message": "durable"})
        self.assertTrue(previous.closed)
        self.assertIsNot(previous, controller.journal)

        previous = controller.journal
        emit(
            controller,
            "notice",
            data={"message": "batched"},
            durable=False,
            snapshot=False,
        )
        self.assertIs(previous, controller.journal)
        state_module.commit_snapshot(controller)
        self.assertTrue(previous.closed)
        self.assertIsNot(previous, controller.journal)

    def test_same_slurm_allocation_reattaches_active_jobs(self) -> None:
        job = job_image("job-active", "gpu-3")
        job["launch_token"] = "scruffy-token"
        job["slurm_step_id"] = "240292.7"
        current_incarnation = slurm_incarnation()
        job["allocation_incarnation_sha256"] = (
            current_incarnation.fingerprint_sha256
        )
        write_state(
            self.root,
            {
                "v": 1,
                "queue_id": queue_id(self.root),
                "last_seq": 0,
                "allocation": {
                    "id": "240292",
                    "incarnation": current_incarnation.to_dict(),
                },
                "nodes": {},
                "jobs": {"job-active": job},
                "draining": False,
            },
        )

        controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("gpu-3", (0,), 2, 2),),
            launcher="slurm",
            allocation_id="240292",
            slurm_job_id="240292",
            allocation_incarnation=current_incarnation,
            poll_interval=0.1,
            cancel_grace=30,
        )
        self.addCleanup(lambda: controller.journal.close())

        running = controller.running["job-active"]
        self.assertIsNone(running.process)
        self.assertEqual("scruffy-token", running.step_name)
        self.assertEqual("running", controller.state["jobs"]["job-active"]["state"])
        self.assertIsNotNone(controller.state["jobs"]["job-active"]["assignment"])
        self.assertTrue(controller.state["launches_paused"])

    def test_journal_rebuild_retains_incarnation_for_strict_reattach(self) -> None:
        incarnation = slurm_incarnation()
        first = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("gpu-3", (0,), 2, 2),),
            launcher="slurm",
            allocation_id="240292",
            slurm_job_id="240292",
            allocation_incarnation=incarnation,
            poll_interval=0.1,
            cancel_grace=30,
        )
        job = job_image("job-journal", "gpu-3")
        job.update(
            {
                "launch_token": "scruffy-journal",
                "allocation_incarnation_sha256": incarnation.fingerprint_sha256,
            }
        )
        first.state["jobs"][job["id"]] = job
        emit(first, "job.running", job=job)
        first.journal.close()
        (self.root / "state.json").unlink()

        recovered = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("gpu-3", (0,), 2, 2),),
            launcher="slurm",
            allocation_id="240292",
            slurm_job_id="240292",
            allocation_incarnation=incarnation,
            poll_interval=0.1,
            cancel_grace=30,
        )
        self.addCleanup(lambda: recovered.journal.close())

        self.assertIn("job-journal", recovered.running)
        self.assertEqual(
            incarnation.to_dict(), recovered.state["allocation"]["incarnation"]
        )

    def test_same_job_id_with_new_restart_count_loses_old_steps(self) -> None:
        previous = slurm_incarnation(restart_count=0)
        current = slurm_incarnation(restart_count=1)
        job = job_image("job-stale", "gpu-3")
        job.update(
            {
                "launch_token": "scruffy-stale",
                "allocation_incarnation_sha256": previous.fingerprint_sha256,
            }
        )
        write_state(
            self.root,
            {
                "v": 1,
                "queue_id": queue_id(self.root),
                "last_seq": 0,
                "allocation": {"id": "240292", "incarnation": previous.to_dict()},
                "nodes": {},
                "jobs": {job["id"]: job},
                "draining": True,
                "drain_requested": True,
            },
        )

        controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("gpu-3", (0,), 2, 2),),
            launcher="slurm",
            allocation_id="240292",
            slurm_job_id="240292",
            allocation_incarnation=current,
            poll_interval=0.1,
            cancel_grace=30,
            start_paused=True,
        )
        self.addCleanup(lambda: controller.journal.close())

        recovered = controller.state["jobs"]["job-stale"]
        self.assertEqual("lost", recovered["state"])
        self.assertEqual("allocation_incarnation_changed", recovered["reason"])
        self.assertIsNone(recovered["assignment"])
        self.assertIsNotNone(recovered["last_assignment"])
        self.assertNotIn("job-stale", controller.running)
        self.assertFalse(controller.state["draining"])
        self.assertFalse(controller.state["drain_requested"])
        self.assertTrue(controller.state["launches_paused"])
        self.assertEqual(
            {
                "previous_allocation_id": "240292",
                "previous_incarnation_sha256": previous.fingerprint_sha256,
                "lost_jobs": 1,
                "queued_jobs": 0,
                "blocked_jobs": 0,
                "ineligible_jobs": 0,
            },
            controller.state["allocation"]["handover"],
        )
        self.assertEqual(
            current.fingerprint_sha256,
            controller.state["allocation"]["incarnation"]["fingerprint_sha256"],
        )

    def test_same_incarnation_preserves_explicit_drain(self) -> None:
        incarnation = slurm_incarnation()
        write_state(
            self.root,
            {
                "v": 1,
                "queue_id": queue_id(self.root),
                "last_seq": 0,
                "allocation": {
                    "id": "240292",
                    "incarnation": incarnation.to_dict(),
                },
                "nodes": {},
                "jobs": {},
                "draining": True,
                "drain_requested": True,
            },
        )

        controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("gpu-3", (0,), 2, 2),),
            launcher="slurm",
            allocation_id="240292",
            slurm_job_id="240292",
            allocation_incarnation=incarnation,
            poll_interval=0.1,
            cancel_grace=30,
        )
        self.addCleanup(lambda: controller.journal.close())

        self.assertTrue(controller.state["draining"])
        self.assertTrue(controller.state["drain_requested"])
        self.assertTrue(controller.state["launches_paused"])

    def test_same_restart_count_with_new_startup_inventory_loses_old_steps(self) -> None:
        previous = slurm_incarnation(node="gpu-2")
        current = slurm_incarnation(node="gpu-3")
        job = job_image("job-old-node", "gpu-2")
        job.update(
            {
                "launch_token": "scruffy-old-node",
                "allocation_incarnation_sha256": previous.fingerprint_sha256,
            }
        )
        write_state(
            self.root,
            {
                "v": 1,
                "queue_id": queue_id(self.root),
                "last_seq": 0,
                "allocation": {"id": "240292", "incarnation": previous.to_dict()},
                "nodes": {},
                "jobs": {job["id"]: job},
                "draining": False,
            },
        )

        controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("gpu-3", (0,), 2, 2),),
            launcher="slurm",
            allocation_id="240292",
            slurm_job_id="240292",
            allocation_incarnation=current,
            poll_interval=0.1,
            cancel_grace=30,
        )
        self.addCleanup(lambda: controller.journal.close())

        recovered = controller.state["jobs"]["job-old-node"]
        self.assertEqual("lost", recovered["state"])
        self.assertEqual("allocation_incarnation_changed", recovered["reason"])
        self.assertNotIn("job-old-node", controller.running)

    def test_legacy_active_job_is_not_upgraded_and_recovery_requires_resume(
        self,
    ) -> None:
        legacy = job_image("job-legacy-active", "gpu-3")
        legacy["launch_token"] = "scruffy-legacy"
        replacement = job_image("job-replacement", "gpu-3")
        replacement.update({"state": "queued", "assignment": None, "queue_order": 2})
        write_state(
            self.root,
            {
                "v": 1,
                "queue_id": queue_id(self.root),
                "last_seq": 0,
                "allocation": {"id": "240292"},
                "nodes": {},
                "jobs": {legacy["id"]: legacy, replacement["id"]: replacement},
                "draining": False,
            },
        )

        controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("gpu-3", (0,), 2, 2),),
            launcher="slurm",
            allocation_id="240292",
            slurm_job_id="240292",
            allocation_incarnation=slurm_incarnation(restart_count=1),
            poll_interval=0.1,
            cancel_grace=30,
        )
        self.addCleanup(lambda: controller.journal.close())

        recovered = controller.state["jobs"]["job-legacy-active"]
        self.assertEqual("lost", recovered["state"])
        self.assertEqual("allocation_incarnation_unavailable", recovered["reason"])
        self.assertNotIn("allocation_incarnation_sha256", recovered)
        self.assertTrue(controller.state["launches_paused"])
        with mock.patch("scruffy.lifecycle.start_job") as start:
            schedule(controller)
        start.assert_not_called()

        resume_queue(self.root)
        _ingest_commands(controller)

        def mark_started(
            _controller: Controller,
            queued: dict[str, object],
            _assignment: Assignment,
        ) -> None:
            queued["state"] = "starting"

        with mock.patch(
            "scruffy.lifecycle.start_job", side_effect=mark_started
        ) as start:
            schedule(controller)
        start.assert_called_once()
        self.assertEqual("job-replacement", start.call_args.args[1]["id"])

    def test_legacy_cpu_only_dependency_is_skipped_after_upstream_is_lost(
        self,
    ) -> None:
        cpu_request = ResourceRequest(1, 0, 1, 1)
        analyze = job_image("job-analyze", "gpu-3")
        analyze.update(
            {
                "request": cpu_request.to_dict(),
                "assignment": Assignment(
                    "job-analyze",
                    cpu_request,
                    (NodeReservation("gpu-3", (), 1, 1),),
                ).to_dict(),
                "launch_token": "scruffy-analyze",
                "workflow_id": "tmr-recovery",
                "task_id": "analyze",
                "needs": [],
            }
        )
        validate = {
            **job_image("job-validate", "gpu-3"),
            "state": "blocked",
            "request": cpu_request.to_dict(),
            "assignment": None,
            "queue_order": 2,
            "workflow_id": "tmr-recovery",
            "task_id": "validate",
            "needs": [{"task_id": "analyze", "condition": "succeeded"}],
            "blockers": [],
            "dependency_gate_passed": False,
            "reason": "waiting_for_dependencies",
        }
        write_state(
            self.root,
            {
                "v": 1,
                "queue_id": queue_id(self.root),
                "last_seq": 0,
                "allocation": {"id": "240292"},
                "nodes": {},
                "jobs": {
                    "job-analyze": analyze,
                    "job-validate": validate,
                },
                "draining": False,
            },
        )

        incarnation = slurm_incarnation(restart_count=1)
        controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("gpu-3", (0,), 2, 2),),
            launcher="slurm",
            allocation_id="240292",
            slurm_job_id="240292",
            allocation_incarnation=incarnation,
            poll_interval=0.1,
            cancel_grace=30,
        )
        self.addCleanup(lambda: controller.journal.close())

        self.assertTrue(controller.state["launches_paused"])
        self.assertEqual("lost", controller.state["jobs"]["job-analyze"]["state"])
        _refresh_dependencies(controller)
        recovered = controller.state["jobs"]["job-validate"]
        self.assertEqual("skipped", recovered["state"])
        self.assertEqual("dependency_unsatisfied", recovered["reason"])
        with mock.patch("scruffy.lifecycle.start_job") as start:
            schedule(controller)
        start.assert_not_called()

        controller.journal.close()
        restarted = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("gpu-3", (0,), 2, 2),),
            launcher="slurm",
            allocation_id="240292",
            slurm_job_id="240292",
            allocation_incarnation=incarnation,
            poll_interval=0.1,
            cancel_grace=30,
        )
        self.addCleanup(lambda: restarted.journal.close())
        self.assertTrue(restarted.state["launches_paused"])
        self.assertEqual("lost", restarted.state["jobs"]["job-analyze"]["state"])
        self.assertEqual("skipped", restarted.state["jobs"]["job-validate"]["state"])

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
            inventory=(NodeInventory("gpu-3", (0, 1), 2, 2),),
            launcher="slurm",
            allocation_id="240292",
            slurm_job_id="240292",
            allocation_incarnation=slurm_incarnation(),
            poll_interval=0.1,
            cancel_grace=30,
        )
        self.addCleanup(lambda: controller.journal.close())

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
            _controller: Controller, job: dict[str, object], assignment: Assignment
        ) -> None:
            job["state"] = "starting"
            job["assignment"] = assignment.to_dict()

        with mock.patch("scruffy.lifecycle.start_job", side_effect=mark_started) as start:
            schedule(controller)
            self.assertEqual(1, start.call_count)
            started = start.call_args.args[1]
            started["state"] = "running"
            schedule(controller)
        self.assertEqual(2, start.call_count)

    def test_same_slurm_allocation_refuses_job_without_launch_token(self) -> None:
        incarnation = slurm_incarnation()
        state = {
            "v": 1,
            "queue_id": queue_id(self.root),
            "last_seq": 0,
            "allocation": {"id": "240292", "incarnation": incarnation.to_dict()},
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
                allocation_incarnation=incarnation,
                poll_interval=0.1,
                cancel_grace=30,
            )

    def test_same_incarnation_refuses_active_job_without_exact_binding(self) -> None:
        incarnation = slurm_incarnation()
        job = job_image("job-active", "gpu-3")
        job["launch_token"] = "scruffy-token"
        write_state(
            self.root,
            {
                "v": 1,
                "queue_id": queue_id(self.root),
                "last_seq": 0,
                "allocation": {
                    "id": "240292",
                    "incarnation": incarnation.to_dict(),
                },
                "nodes": {},
                "jobs": {job["id"]: job},
                "draining": False,
            },
        )

        with self.assertRaisesRegex(RuntimeError, "incarnation differs"):
            _initialize_controller(
                root=self.root,
                inventory=(NodeInventory("gpu-3", (0,), 2, 2),),
                launcher="slurm",
                allocation_id="240292",
                slurm_job_id="240292",
                allocation_incarnation=incarnation,
                poll_interval=0.1,
                cancel_grace=30,
            )

    def test_malformed_persisted_incarnation_fails_closed(self) -> None:
        incarnation = slurm_incarnation().to_dict()
        incarnation["restart_count"] = 1
        write_state(
            self.root,
            {
                "v": 1,
                "queue_id": queue_id(self.root),
                "last_seq": 0,
                "allocation": {"id": "240292", "incarnation": incarnation},
                "nodes": {},
                "jobs": {},
                "draining": False,
            },
        )

        with self.assertRaisesRegex(RuntimeError, "invalid persisted"):
            _initialize_controller(
                root=self.root,
                inventory=(NodeInventory("gpu-3", (0,), 2, 2),),
                launcher="slurm",
                allocation_id="240292",
                slurm_job_id="240292",
                allocation_incarnation=slurm_incarnation(),
                poll_interval=0.1,
                cancel_grace=30,
            )

    def test_persisted_incarnation_must_match_persisted_allocation_id(self) -> None:
        incarnation = slurm_incarnation().to_dict()
        write_state(
            self.root,
            {
                "v": 1,
                "queue_id": queue_id(self.root),
                "last_seq": 0,
                "allocation": {"id": "other-job", "incarnation": incarnation},
                "nodes": {},
                "jobs": {},
                "draining": False,
            },
        )

        with self.assertRaisesRegex(RuntimeError, "ID differs"):
            _initialize_controller(
                root=self.root,
                inventory=(NodeInventory("gpu-3", (0,), 2, 2),),
                launcher="slurm",
                allocation_id="240292",
                slurm_job_id="240292",
                allocation_incarnation=slurm_incarnation(),
                poll_interval=0.1,
                cancel_grace=30,
            )

    def test_legacy_active_job_without_allocation_id_remains_paused(self) -> None:
        legacy = job_image("job-legacy-active", "gpu-3")
        legacy["launch_token"] = "scruffy-legacy"
        write_state(
            self.root,
            {
                "v": 1,
                "queue_id": queue_id(self.root),
                "last_seq": 0,
                "allocation": None,
                "nodes": {},
                "jobs": {legacy["id"]: legacy},
                "draining": False,
            },
        )

        controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("gpu-3", (0,), 2, 2),),
            launcher="slurm",
            allocation_id="240292",
            slurm_job_id="240292",
            allocation_incarnation=slurm_incarnation(restart_count=1),
            poll_interval=0.1,
            cancel_grace=30,
        )
        self.addCleanup(lambda: controller.journal.close())

        recovered = controller.state["jobs"][legacy["id"]]
        self.assertEqual("lost", recovered["state"])
        self.assertEqual("allocation_incarnation_unavailable", recovered["reason"])
        self.assertTrue(controller.state["launches_paused"])

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
            allocation_incarnation=slurm_incarnation(
                "new-allocation", node="new-node"
            ),
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

    def test_replacement_preserves_jobs_that_do_not_fit_and_starts_paused(self) -> None:
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
            allocation_incarnation=slurm_incarnation(
                "new-allocation", node="new-node"
            ),
            poll_interval=0.1,
            cancel_grace=30,
            start_paused=True,
        )
        self.addCleanup(lambda: controller.journal.close())

        preserved = controller.state["jobs"]["job-too-large"]
        self.assertEqual("queued", preserved["state"])
        self.assertTrue(controller.state["launches_paused"])
        self.assertEqual(
            {
                "previous_allocation_id": "old-allocation",
                "lost_jobs": 0,
                "queued_jobs": 1,
                "blocked_jobs": 0,
                "ineligible_jobs": 1,
            },
            controller.state["allocation"]["handover"],
        )
        with mock.patch("scruffy.lifecycle.start_job") as start:
            schedule(controller)
        start.assert_not_called()

        resume_queue(self.root)
        _ingest_commands(controller)
        self.assertFalse(controller.state["launches_paused"])
        with mock.patch("scruffy.lifecycle.start_job") as start:
            schedule(controller)
        start.assert_not_called()

    def test_deadline_window_drains_before_scheduling(self) -> None:
        controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("local", (0,), 2, 2),),
            launcher="local",
            allocation_id="local-allocation",
            slurm_job_id=None,
            poll_interval=0.1,
            cancel_grace=0,
            drain_before_end_seconds=600,
        )
        self.addCleanup(lambda: controller.journal.close())
        controller.state["allocation"]["automatic_drain_at"] = (
            "2000-01-01T00:00:00+00:00"
        )

        _drain_for_deadline(controller)
        _drain_for_deadline(controller)

        self.assertTrue(controller.state["draining"])
        self.assertTrue(controller.state["drain_requested"])
        events = [
            event
            for event in read_events(self.root)
            if event["kind"] == "allocation.draining"
        ]
        self.assertEqual(1, len(events))
        event = events[0]
        self.assertEqual("allocation_deadline", event["data"]["reason"])
        self.assertEqual(600, event["data"]["drain_before_end_seconds"])

    def test_deadline_window_is_derived_from_the_slurm_deadline(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"SLURM_JOB_END_TIME": "1893456000"},
            clear=False,
        ):
            controller = _initialize_controller(
                root=self.root,
                inventory=(NodeInventory("gpu-0", (0,), 2, 2),),
                launcher="slurm",
                allocation_id="new-allocation",
                slurm_job_id="new-allocation",
                allocation_incarnation=slurm_incarnation(
                    "new-allocation", node="gpu-0"
                ),
                poll_interval=0.1,
                cancel_grace=30,
                drain_before_end_seconds=900,
            )
        self.addCleanup(lambda: controller.journal.close())

        self.assertEqual(
            "2029-12-31T23:45:00+00:00",
            controller.state["allocation"]["automatic_drain_at"],
        )
        self.assertFalse(controller.state["draining"])

    def test_same_allocation_restart_preserves_handover_summary(self) -> None:
        incarnation = slurm_incarnation("new-allocation", node="gpu-0")
        handover = {
            "previous_allocation_id": "old-allocation",
            "lost_jobs": 2,
            "queued_jobs": 3,
            "blocked_jobs": 4,
            "ineligible_jobs": 1,
        }
        write_state(
            self.root,
            {
                "v": 1,
                "queue_id": queue_id(self.root),
                "last_seq": 0,
                "allocation": {
                    "id": "new-allocation",
                    "incarnation": incarnation.to_dict(),
                    "handover": handover,
                },
                "nodes": {},
                "jobs": {},
            },
        )

        controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("gpu-0", (0,), 2, 2),),
            launcher="slurm",
            allocation_id="new-allocation",
            slurm_job_id="new-allocation",
            allocation_incarnation=incarnation,
            poll_interval=0.1,
            cancel_grace=30,
        )
        self.addCleanup(lambda: controller.journal.close())

        self.assertEqual(handover, controller.state["allocation"]["handover"])


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
        self.addCleanup(lambda: self.controller.journal.close())

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


class SlurmLaunchTests(unittest.TestCase):
    def test_slurm_worker_document_owns_canonical_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = mock.Mock(
                root=root,
                inventory=(NodeInventory("gpu-3", (0,), 112, 1024),),
                launcher="slurm",
                allocation_id="240292",
                slurm_job_id="240292",
                allocation_incarnation=slurm_incarnation(),
                running={},
            )
            request = ResourceRequest(1, 1, 14, 128)
            assigned = Assignment(
                "job-1",
                request,
                (NodeReservation("gpu-3", (0,), 14, 128),),
            )
            job = {
                "id": "job-1",
                "project_id": "tests",
                "name": "worker-owned-logs",
                "state": "queued",
                "argv": ["true"],
                "cwd": str(root),
                "env": {},
            }

            def write_launch(*_args, **_kwargs):
                job["provenance"] = {"assignment_sha256": "test-sha"}
                return root / "provenance.json"

            process = mock.Mock(pid=123)
            with (
                mock.patch(
                    "scruffy.lifecycle.write_launch_record",
                    side_effect=write_launch,
                ),
                mock.patch("scruffy.lifecycle.emit"),
                mock.patch("scruffy.lifecycle.atomic_write_json") as write,
                mock.patch(
                    "scruffy.lifecycle._launch_arguments",
                    return_value=(["srun"], {}),
                ),
                mock.patch("scruffy.lifecycle.subprocess.Popen", return_value=process),
            ):
                start_job(controller, job, assigned)

        document = write.call_args.args[1]
        self.assertEqual(
            {
                "stdout": str((root / "jobs/job-1/stdout.log").resolve()),
                "stderr": str((root / "jobs/job-1/stderr.log").resolve()),
            },
            document["logs"],
        )

    def test_worker_step_gets_exact_admitted_resource_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = mock.Mock(
                inventory=(NodeInventory("gpu-3", (0,), 112, 1024),),
                launcher="slurm",
                slurm_job_id="240292",
                gpu_isolation="gpu",
            )
            request = ResourceRequest(1, 1, 14, 128)
            assigned = Assignment(
                "job-1",
                request,
                (NodeReservation("gpu-3", (0,), 14, 128),),
            )

            argv, _ = _launch_arguments(
                controller,
                {"launch_token": "scruffy-token"},
                assigned,
                root / "assignment.json",
                root / "stdout.log",
                root / "stderr.log",
            )

        self.assertIn("--gpus-per-task=1", argv)
        self.assertIn("--tres-bind=gres/gpu:mask:0x1", argv)
        self.assertNotIn("--gpu-bind=none", argv)
        self.assertIn("--cpus-per-task=14", argv)
        self.assertIn("--mem=128G", argv)
        self.assertNotIn("--overlap", argv)
        self.assertEqual(14, assigned.request.cpus_per_node)

    def test_start_job_passes_only_sanitized_environment_to_srun(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("gpu-3", (0,), 112, 1024),),
                launcher="slurm",
                allocation_id="240292",
                slurm_job_id="240292",
                allocation_incarnation=slurm_incarnation(),
                poll_interval=0.1,
                cancel_grace=30,
            )
            self.addCleanup(lambda: controller.journal.close())
            request = ResourceRequest(1, 1, 14, 128)
            job = {
                "id": "job-1",
                "name": "job-1",
                "state": "queued",
                "submitted_at": utc_now(),
                "queue_order": 1,
                "argv": ["true"],
                "cwd": "/tmp",
                "env": {},
                "request": request.to_dict(),
                "assignment": None,
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
                "signal": None,
                "reason": None,
                "error": None,
            }
            controller.state["jobs"][job["id"]] = job
            assigned = Assignment(
                job["id"], request, (NodeReservation("gpu-3", (0,), 14, 128),)
            )
            process = mock.Mock(pid=12345)

            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "PATH": "/usr/bin",
                        "SLURM_JOB_ID": "240292",
                        "SLURM_JOB_NODELIST": "gpu-[0,3]",
                        "SLURM_GRES": "gpu:h100:8",
                        "SLURM_HINT": "nomultithread",
                        "SLURM_EXPORT_ENV": "NONE",
                        "SRUN_EXPORT_ENV": "NONE",
                    },
                    clear=True,
                ),
                mock.patch(
                    "scruffy.lifecycle.new_step_name", return_value="scruffy-token"
                ),
                mock.patch(
                    "scruffy.lifecycle.subprocess.Popen", return_value=process
                ) as popen,
            ):
                start_job(controller, job, assigned)
            assignment_document = json.loads(
                (root / "jobs/job-1/assignment.json").read_text(encoding="utf-8")
            )

        argv = popen.call_args.args[0]
        environment = popen.call_args.kwargs["env"]
        self.assertIn("--export=ALL", argv)
        self.assertEqual(
            {
                "PATH": "/usr/bin",
                "SLURM_JOB_ID": "240292",
                "SLURM_JOB_NODELIST": "gpu-[0,3]",
            },
            environment,
        )
        self.assertEqual("starting", job["state"])
        self.assertIn(job["id"], controller.running)
        self.assertEqual(
            ["jobs/job-1/runtime-placement-0.json"], job["runtime_placement_files"]
        )
        self.assertEqual("slurm", assignment_document["launcher"])
        self.assertEqual("240292", assignment_document["slurm_job_id"])
        self.assertEqual(
            slurm_incarnation().fingerprint_sha256,
            assignment_document["allocation_incarnation_sha256"],
        )
        self.assertEqual(
            slurm_incarnation().fingerprint_sha256,
            job["allocation_incarnation_sha256"],
        )
        self.assertEqual(1, assignment_document["runtime_placement_contract"])
        self.assertEqual(1, job["runtime_placement_contract"])
        self.assertEqual(1, assignment_document["gpus_per_node"])
        self.assertEqual(
            "jobs/job-1/runtime-placement-0.json",
            assignment_document["assignment"][0]["runtime_placement"],
        )


class RuntimePlacementAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"
        self.controller = _initialize_controller(
            root=self.root,
            inventory=(NodeInventory("gpu-3", (0,), 2, 2),),
            launcher="slurm",
            allocation_id="240292",
            slurm_job_id="240292",
            allocation_incarnation=slurm_incarnation(),
            poll_interval=0.1,
            cancel_grace=30,
        )
        self.addCleanup(lambda: self.controller.journal.close())

    def _job(self) -> dict[str, object]:
        job = job_image("job-1", "gpu-3")
        job["slurm_step_id"] = "240292.7"
        job["runtime_placement_contract"] = 1
        job["runtime_placement_files"] = [
            "jobs/job-1/runtime-placement-0.json"
        ]
        self.controller.state["jobs"]["job-1"] = job
        return job

    def _placement(self, **updates: object) -> dict[str, object]:
        record = {
            "schema": 1,
            "job_id": "job-1",
            "node": "gpu-3",
            "requested_gpus": 1,
            "ledger_gpu_ids": [0],
            "slurm_job_id": "240292",
            "slurm_step_id": "7",
            "slurm_step_gpus": ["5"],
            "cuda_visible_devices": ["0"],
            "cuda_device_order": "PCI_BUS_ID",
        }
        record.update(updates)
        return record

    def test_terminal_result_binds_exact_runtime_placement_sha(self) -> None:
        job = self._job()
        source = self.root / "jobs/job-1/runtime-placement-0.json"
        digest = create_immutable_json(source, self._placement())

        _finish_job(self.controller, "job-1", RunningProcess(mock.Mock(), None), 0)

        self.assertEqual("succeeded", job["state"])
        self.assertEqual("authenticated", job["runtime_placement_status"])
        self.assertEqual(
            [
                {
                    "path": "jobs/job-1/runtime-placement-0.json",
                    "sha256": digest,
                    "node": "gpu-3",
                    "slurm_step_id": "240292.7",
                    "physical_gpu_ids": ["5"],
                    "visible_gpu_ids": ["0"],
                    "reserved_gpu_ids": [0],
                }
            ],
            job["runtime_placements"],
        )
        terminal = [
            event for event in read_events(self.root) if event["kind"] == "job.succeeded"
        ][-1]
        self.assertEqual(digest, terminal["job"]["runtime_placements"][0]["sha256"])
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)

    def test_cpu_only_terminal_binds_an_authenticated_empty_placement(self) -> None:
        request = ResourceRequest(1, 0, 1, 1)
        job = self._job()
        job["request"] = request.to_dict()
        job["assignment"] = Assignment(
            "job-1",
            request,
            (NodeReservation("gpu-3", (), 1, 1),),
        ).to_dict()
        source = self.root / "jobs/job-1/runtime-placement-0.json"
        digest = create_immutable_json(
            source,
            self._placement(
                requested_gpus=0,
                ledger_gpu_ids=[],
                slurm_step_gpus=[],
                cuda_visible_devices=[],
            ),
        )

        _finish_job(self.controller, "job-1", RunningProcess(mock.Mock(), None), 0)

        self.assertEqual("succeeded", job["state"])
        self.assertEqual("authenticated", job["runtime_placement_status"])
        self.assertEqual(
            [
                {
                    "path": "jobs/job-1/runtime-placement-0.json",
                    "sha256": digest,
                    "node": "gpu-3",
                    "slurm_step_id": "240292.7",
                    "physical_gpu_ids": [],
                    "visible_gpu_ids": [],
                    "reserved_gpu_ids": [],
                }
            ],
            job["runtime_placements"],
        )

    def test_success_fails_closed_on_mutable_or_substituted_placement(self) -> None:
        for record, mutate_mode in (
            (self._placement(), True),
            (self._placement(ledger_gpu_ids=[7]), False),
        ):
            with self.subTest(record=record, mutate_mode=mutate_mode):
                job = self._job()
                source = self.root / "jobs/job-1/runtime-placement-0.json"
                create_immutable_json(source, record)
                if mutate_mode:
                    source.chmod(0o644)

                _finish_job(
                    self.controller, "job-1", RunningProcess(mock.Mock(), None), 0
                )

                self.assertEqual("failed", job["state"])
                self.assertEqual("runtime_placement_invalid", job["reason"])
                self.assertEqual("invalid", job["runtime_placement_status"])
                self.assertIn("runtime_placement_error", job)
                if source.exists():
                    source.chmod(0o644)
                    source.unlink()

    def test_new_contract_success_fails_closed_when_authority_is_missing(self) -> None:
        job = self._job()

        _finish_job(self.controller, "job-1", RunningProcess(mock.Mock(), None), 0)

        self.assertEqual("failed", job["state"])
        self.assertEqual("runtime_placement_invalid", job["reason"])
        self.assertEqual("invalid", job["runtime_placement_status"])
        self.assertIn("runtime-placement-0.json", job["runtime_placement_error"])

    def test_legacy_reattached_results_are_preserved_but_not_authenticated(self) -> None:
        for returncode, expected_state in ((0, "succeeded"), (1, "failed")):
            with self.subTest(returncode=returncode):
                job = job_image("job-1", "gpu-3")
                job["slurm_step_id"] = "240292.7"
                self.controller.state["jobs"]["job-1"] = job

                _finish_job(
                    self.controller,
                    "job-1",
                    RunningProcess(mock.Mock(), None),
                    returncode,
                )

                self.assertEqual(expected_state, job["state"])
                self.assertEqual(
                    "process_exit" if returncode == 0 else "application_exit",
                    job["reason"],
                )
                self.assertEqual(
                    "legacy_unavailable", job["runtime_placement_status"]
                )
                self.assertNotIn("runtime_placements", job)
                self.assertNotIn("runtime_placement_error", job)


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
            self.addCleanup(lambda: controller.journal.close())
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
            self.addCleanup(lambda: controller.journal.close())
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
            self.addCleanup(lambda: controller.journal.close())
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
            self.addCleanup(lambda: controller.journal.close())
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
            self.addCleanup(lambda: controller.journal.close())
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
            self.addCleanup(lambda: controller.journal.close())
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
            self.addCleanup(lambda: controller.journal.close())
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
            self.addCleanup(lambda: controller.journal.close())
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
            self.addCleanup(lambda: controller.journal.close())
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
            self.addCleanup(lambda: controller.journal.close())
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
            self.addCleanup(lambda: restarted.journal.close())
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
            self.addCleanup(lambda: restarted.journal.close())
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
            self.addCleanup(lambda: controller.journal.close())
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
            self.addCleanup(lambda: controller.journal.close())
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
            self.addCleanup(lambda: controller.journal.close())
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
            self.addCleanup(lambda: controller.journal.close())
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
            allocation_incarnation=slurm_incarnation(),
            poll_interval=0.1,
            cancel_grace=30,
            drain_before_end_seconds=900,
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
                "runtime_placement_contract": 1,
                "runtime_placement_files": [
                    "jobs/job-a/runtime-placement-0.json"
                ],
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
        create_immutable_json(
            self.controller.root / "jobs/job-a/runtime-placement-0.json",
            {
                "schema": 1,
                "job_id": "job-a",
                "node": "gpu-3",
                "requested_gpus": 1,
                "ledger_gpu_ids": [0],
                "slurm_job_id": "240292",
                "slurm_step_id": "7",
                "slurm_step_gpus": ["5"],
                "cuda_visible_devices": ["0"],
                "cuda_device_order": None,
            },
        )

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

    def test_legacy_reattached_rc0_and_rc1_keep_process_results(self) -> None:
        for returncode, expected_state in ((0, "succeeded"), (1, "failed")):
            with self.subTest(returncode=returncode):
                job = job_image("job-legacy", "gpu-3")
                job.update(
                    {
                        "launch_token": "legacy-token",
                        "slurm_step_id": "240292.8",
                        "stdout": "jobs/job-legacy/stdout.log",
                        "stderr": "jobs/job-legacy/stderr.log",
                    }
                )
                self.controller.state["jobs"] = {"job-legacy": job}
                running = RunningProcess(None, "legacy-token")
                running.closed_streams.update({"stdout", "stderr"})
                self.controller.running = {"job-legacy": running}
                self.controller.slurm_snapshot_at += 1
                self.controller.slurm_steps = ()

                with (
                    mock.patch("scruffy.lifecycle.refresh_slurm_snapshot"),
                    mock.patch(
                        "scruffy.lifecycle.completed_step",
                        return_value=SlurmStepResult("COMPLETED", returncode),
                    ),
                ):
                    poll_processes(self.controller)

                self.assertEqual(expected_state, job["state"])
                self.assertEqual(
                    "legacy_unavailable", job["runtime_placement_status"]
                )
                self.assertNotIn("runtime_placements", job)
                self.assertNotIn("job-legacy", self.controller.running)


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
        self.addCleanup(lambda: self.controller.journal.close())

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
