"""One durable snapshot per controller poll iteration, with crash safety."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import scruffy.controller as controller_module
import scruffy.state as state_module
import scruffy.storage as storage_module
from scruffy.client import cancel_job, publish_event, submit_job
from scruffy.controller import (
    _discard_journaled_commands,
    _ingest_commands,
    _ingest_requests,
    _initialize_controller,
    _serve,
)
from scruffy.models import NodeInventory, ResourceRequest
from scruffy.runtime import Controller, RunningProcess
from scruffy.slurm import AllocationIncarnation
from scruffy.state import emit, group_commit
from scruffy.storage import (
    command_sources,
    journal_path,
    load_state,
    read_events,
    utc_now,
)

REQUEST = ResourceRequest(1, 1, 1, 1)
INVENTORY = (NodeInventory("local", (0,), 2, 2),)


def blocked_job(job_id: str, order: int) -> dict[str, Any]:
    return {
        "id": job_id,
        "name": job_id,
        "state": "blocked",
        "submitted_at": utc_now(),
        "queue_order": order,
        "argv": ["true"],
        "cwd": "/tmp",
        "env": {},
        "request": REQUEST.to_dict(),
        "assignment": None,
        "workflow_id": "flow",
        "task_id": job_id,
        "needs": [{"task_id": "never-submitted", "condition": "succeeded"}],
        "blockers": [],
    }


class GroupCommitTests(unittest.TestCase):
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

    def _with_blocked_jobs(self, controller: Controller, count: int) -> list[str]:
        job_ids = [f"job-{index:04d}" for index in range(count)]
        for index, job_id in enumerate(job_ids, start=1):
            controller.state["jobs"][job_id] = blocked_job(job_id, index)
        state_module.commit_snapshot(controller)
        return job_ids

    def test_n_cancel_commands_cost_one_snapshot_write(self) -> None:
        controller = self._controller()
        job_ids = self._with_blocked_jobs(controller, 50)
        for job_id in job_ids:
            cancel_job(self.root, job_id)

        with mock.patch(
            "scruffy.state.write_state", wraps=state_module.write_state
        ) as write:
            _ingest_commands(controller)

        self.assertEqual(1, write.call_count)
        self.assertEqual([], command_sources(self.root))
        persisted = load_state(self.root)
        self.assertEqual(
            {"cancelled"},
            {persisted["jobs"][job_id]["state"] for job_id in job_ids},
        )
        receipts = list((self.root / "commands" / ".accepted").glob("*.json"))
        self.assertEqual(len(job_ids), len(receipts))

    def test_poll_iteration_writes_one_snapshot_for_commands_and_transitions(self) -> None:
        controller = self._controller()
        job_ids = self._with_blocked_jobs(controller, 20)
        for job_id in job_ids:
            cancel_job(self.root, job_id)
        submitted = [
            submit_job(
                self.root,
                argv=["true"],
                name=f"new-{index}",
                cwd=Path.cwd(),
                environment={},
                request=REQUEST,
                request_id=f"batch/new-{index}",
                workflow_id="other",
                task_id=f"new-{index}",
                needs=({"task_id": "missing", "condition": "succeeded"},),
            )["job_id"]
            for index in range(10)
        ]
        for index in range(3):
            publish_event(
                self.root,
                job_id=job_ids[index],
                kind="workload.progress",
                event_id=f"progress-{index}",
                data={"step": index},
            )
        writes: list[str] = []
        original_write = storage_module.write_state

        def record_write(root: Path, state: dict[str, Any]) -> None:
            writes.append("write")
            original_write(root, state)

        class StopLoop(Exception):
            pass

        def stop_after_first_iteration(_seconds: float) -> None:
            raise StopLoop

        controller.last_heartbeat = time.monotonic()
        with (
            mock.patch("scruffy.state.write_state", side_effect=record_write),
            mock.patch(
                "scruffy.controller.time.sleep", side_effect=stop_after_first_iteration
            ),
            mock.patch("scruffy.controller.remove_cold_job_directories"),
            self.assertRaises(StopLoop),
        ):
            _serve(controller)

        self.assertEqual(["write"], writes)
        persisted = load_state(self.root)
        self.assertTrue(
            all(persisted["jobs"][job_id]["state"] == "cancelled" for job_id in job_ids)
        )
        self.assertTrue(
            all(persisted["jobs"][job_id]["state"] == "blocked" for job_id in submitted)
        )
        self.assertEqual([], command_sources(self.root))
        self.assertFalse(
            any((self.root / "requests" / job_id).exists() for job_id in submitted)
        )
        self.assertFalse(
            any((self.root / "reports" / job_id).exists() for job_id in job_ids)
        )
        self.assertEqual(
            2, persisted["jobs"][job_ids[2]]["workload"]["progress"]["step"]
        )

    def test_commands_are_acknowledged_only_after_their_commit(self) -> None:
        controller = self._controller()
        job_ids = self._with_blocked_jobs(controller, 5)
        for job_id in job_ids:
            cancel_job(self.root, job_id)
        order: list[str] = []
        original_write = storage_module.write_state
        original_receipt = storage_module.record_command_receipt

        def record_write(root: Path, state: dict[str, Any]) -> None:
            original_write(root, state)
            order.append("snapshot")

        def record_receipt(root: Path, command: dict[str, Any], **kwargs: Any) -> None:
            on_disk = load_state(self.root)["jobs"][command["job_id"]]["state"]
            order.append(f"receipt:{on_disk}")
            original_receipt(root, command, **kwargs)

        with (
            mock.patch("scruffy.state.write_state", side_effect=record_write),
            mock.patch(
                "scruffy.storage.record_command_receipt", side_effect=record_receipt
            ),
        ):
            _ingest_commands(controller)

        self.assertEqual(["snapshot"] + ["receipt:cancelled"] * 5, order)

    def test_failed_commit_keeps_commands_and_restart_applies_them_once(self) -> None:
        controller = self._controller()
        job_ids = self._with_blocked_jobs(controller, 8)
        for job_id in job_ids:
            cancel_job(self.root, job_id)

        with (
            mock.patch("scruffy.state.write_state", side_effect=OSError("EIO")),
            self.assertRaises(OSError),
        ):
            _ingest_commands(controller)

        # Nothing was acknowledged and the published snapshot is unchanged.
        self.assertEqual(len(job_ids), len(command_sources(self.root)))
        self.assertFalse((self.root / "commands" / ".accepted").exists())
        self.assertEqual(
            {"blocked"},
            {load_state(self.root)["jobs"][job_id]["state"] for job_id in job_ids},
        )
        controller.journal.close()

        # The journal was synced before the failed snapshot, so a restarted
        # controller replays every cancellation and acknowledges the commands
        # from their journaled outcomes without applying them twice.
        restarted = self._controller()
        self.assertEqual(
            {"cancelled"},
            {restarted.state["jobs"][job_id]["state"] for job_id in job_ids},
        )
        _discard_journaled_commands(restarted)
        _ingest_commands(restarted)
        self.assertEqual([], command_sources(self.root))
        cancelled = [
            event["job_id"]
            for event in read_events(self.root)
            if event.get("kind") == "job.cancelled"
        ]
        self.assertEqual(sorted(job_ids), sorted(cancelled))

    def test_unsynced_batch_lost_in_a_crash_is_retried_from_the_inbox(self) -> None:
        controller = self._controller()
        job_ids = self._with_blocked_jobs(controller, 4)
        for job_id in job_ids:
            cancel_job(self.root, job_id)
        committed_size = journal_path(self.root).stat().st_size

        with (
            mock.patch("scruffy.state.sync_file", side_effect=OSError("EIO")),
            self.assertRaises(OSError),
        ):
            _ingest_commands(controller)
        controller.journal.close()
        # Model a crash that loses the unsynced journal tail.
        with journal_path(self.root).open("r+b") as journal:
            journal.truncate(committed_size)

        restarted = self._controller()
        self.assertEqual(
            {"blocked"},
            {restarted.state["jobs"][job_id]["state"] for job_id in job_ids},
        )
        _discard_journaled_commands(restarted)
        self.assertEqual(len(job_ids), len(command_sources(self.root)))
        _ingest_commands(restarted)
        self.assertEqual([], command_sources(self.root))
        self.assertEqual(
            {"cancelled"},
            {load_state(self.root)["jobs"][job_id]["state"] for job_id in job_ids},
        )

    def test_command_batch_is_bounded_per_tick(self) -> None:
        controller = self._controller()
        job_ids = self._with_blocked_jobs(controller, 7)
        for job_id in job_ids:
            cancel_job(self.root, job_id)

        _ingest_commands(controller, limit=3)
        self.assertEqual(4, len(command_sources(self.root)))
        _ingest_commands(controller, limit=3)
        _ingest_commands(controller, limit=3)
        self.assertEqual([], command_sources(self.root))

    def test_cancelling_transition_is_durable_before_the_launcher_is_signalled(
        self,
    ) -> None:
        controller = self._controller()
        job = {
            key: value
            for key, value in blocked_job("job-running", 1).items()
            if key not in {"workflow_id", "task_id", "needs", "blockers"}
        }
        job["state"] = "running"
        controller.state["jobs"][job["id"]] = job
        controller.running[job["id"]] = RunningProcess(mock.Mock(), None)
        state_module.commit_snapshot(controller)
        cancel_job(self.root, job["id"])
        observed: list[str] = []

        def stop(_controller: Controller, _running: RunningProcess) -> None:
            observed.append(load_state(self.root)["jobs"][job["id"]]["state"])

        with mock.patch("scruffy.controller.stop_launcher", side_effect=stop):
            _ingest_commands(controller)

        self.assertEqual(["cancelling"], observed)

    def test_admissions_commit_once_before_request_directories_are_retired(
        self,
    ) -> None:
        controller = self._controller()
        submitted = [
            submit_job(
                self.root,
                argv=["true"],
                name=f"job-{index}",
                cwd=Path.cwd(),
                environment={},
                request=REQUEST,
                request_id=f"admit/{index}",
            )["job_id"]
            for index in range(6)
        ]
        order: list[str] = []
        original_write = storage_module.write_state
        original_accept = controller_module.accept_request

        def record_write(root: Path, state: dict[str, Any]) -> None:
            original_write(root, state)
            order.append("snapshot")

        def record_accept(root: Path, job_id: str, **kwargs: Any) -> bool:
            self.assertEqual("queued", load_state(self.root)["jobs"][job_id]["state"])
            order.append("accept")
            return original_accept(root, job_id, **kwargs)

        with (
            mock.patch("scruffy.state.write_state", side_effect=record_write),
            mock.patch("scruffy.controller.accept_request", side_effect=record_accept),
        ):
            _ingest_requests(controller)

        self.assertEqual(["snapshot"] + ["accept"] * len(submitted), order)

    def test_nested_group_commits_publish_once_at_the_outermost_exit(self) -> None:
        controller = self._controller()
        with mock.patch(
            "scruffy.state.write_state", wraps=state_module.write_state
        ) as write:
            with group_commit(controller):
                emit(controller, "notice", data={"n": 1})
                with group_commit(controller):
                    emit(controller, "notice", data={"n": 2})
                self.assertEqual(0, write.call_count)
            self.assertEqual(1, write.call_count)
        self.assertFalse(controller.commit_pending)

    def test_replacement_recovery_publishes_one_snapshot(self) -> None:
        inventory = (NodeInventory("node", (0, 1, 2, 3), 8, 8),)
        old = AllocationIncarnation("old", 0, inventory)
        new = AllocationIncarnation("new", 0, inventory)
        policy = {
            "max_attempts": 3,
            "retry_on": ["allocation_replaced"],
            "evacuation": {"signal": "USR1", "grace_seconds": 60},
        }
        jobs = {}
        for index in range(4):
            job_id = f"job-lost-{index}"
            jobs[job_id] = {
                **blocked_job(job_id, index + 1),
                "state": "running",
                "attempt": 1,
                "task_id": f"train-{index}",
                "recovery": policy,
                "launch_token": f"token-{index}",
                "assignment": {
                    "job_id": job_id,
                    "request": REQUEST.to_dict(),
                    "reservations": [
                        {"node": "node", "gpu_ids": [index], "cpus": 1, "memory_gb": 1}
                    ],
                },
                "needs": [],
            }
        storage_module.write_state(
            self.root,
            {
                "v": 1,
                "queue_id": "queue",
                "last_seq": 0,
                "allocation": {"id": "old", "incarnation": old.to_dict()},
                "nodes": {},
                "jobs": jobs,
                "next_queue_order": 4,
                "draining": False,
            },
        )
        with mock.patch(
            "scruffy.state.write_state", wraps=state_module.write_state
        ) as write:
            controller = _initialize_controller(
                root=self.root,
                inventory=inventory,
                launcher="slurm",
                allocation_id="new",
                slurm_job_id="new",
                allocation_incarnation=new,
                poll_interval=0.01,
                cancel_grace=0,
                start_paused=True,
            )
        self.addCleanup(lambda: controller.journal.close())

        self.assertEqual(1, write.call_count)
        persisted = load_state(self.root)["jobs"]
        self.assertEqual(
            {"lost"}, {persisted[job_id]["state"] for job_id in jobs}
        )
        successors = [job for job in persisted.values() if job.get("attempt") == 2]
        self.assertEqual(4, len(successors))
        self.assertTrue(load_state(self.root)["launches_paused"])

    def test_startup_removes_abandoned_snapshot_temporaries(self) -> None:
        controller = self._controller()
        controller.journal.close()
        stale = [
            self.root / ".state.json.0123456789abcdef0123456789abcdef.tmp",
            self.root / ".cursor.json.0123456789abcdef0123456789abcdef.tmp",
            self.root / "journal" / ".checkpoint-000001.json.abc.tmp",
        ]
        for source in stale:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("{")
        unrelated = self.root / "notes.tmp"
        unrelated.write_text("keep")

        restarted = self._controller()

        self.assertFalse(any(source.exists() for source in stale))
        self.assertTrue(unrelated.exists())
        started = [
            event
            for event in read_events(self.root)
            if event.get("kind") == "allocation.started"
        ]
        self.assertEqual(3, started[-1]["data"]["removed_stale_temporaries"])
        self.assertIsNotNone(restarted.state["allocation"])


if __name__ == "__main__":
    unittest.main()
