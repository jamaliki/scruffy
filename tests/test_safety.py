from __future__ import annotations

import queue
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from scruffy.client import cancel_job, submit_job
from scruffy.controller import _ingest_commands, _ingest_requests, _initialize_controller
from scruffy.lifecycle import drain_messages, poll_processes, start_job
from scruffy.models import (
    Assignment,
    NodeInventory,
    NodeReservation,
    ResourceRequest,
)
from scruffy.runtime import Controller, OutputNotifier, RunningProcess
from scruffy.slurm import SlurmStep
from scruffy.slurm_runtime import reconcile_slurm, refresh_slurm_snapshot
from scruffy.state import load_recovered_state
from scruffy.storage import (
    append_event,
    list_commands,
    open_journal,
    queue_id,
    read_events,
    utc_now,
    write_state,
)


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

    def test_journal_recovery_still_refuses_same_allocation_relaunch(self) -> None:
        job = job_image("job-active", "gpu-3")
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

        with self.assertRaisesRegex(RuntimeError, "unsafe recovery"):
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
        self.addCleanup(controller.journal.close)

        self.assertEqual({"new-node"}, set(controller.state["nodes"]))
        for job in controller.state["jobs"].values():
            self.assertEqual("lost", job["state"])
            self.assertIsNone(job["assignment"])
            self.assertIsNotNone(job["last_assignment"])

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
