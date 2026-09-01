from __future__ import annotations

import multiprocessing
import os
import signal
import sys
import tempfile
import threading
import time
import traceback
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from scruffy.client import (
    publish_event,
    request_evacuation,
    status,
    submit_job,
    wait_for_evacuation,
    wait_for_event_ack,
    wait_for_job,
)
from scruffy.controller import (
    _advance_evacuation,
    _begin_evacuation,
    _ingest_commands,
    _ingest_reports,
    _initialize_controller,
)
from scruffy.models import Assignment, NodeInventory, NodeReservation, ResourceRequest
from scruffy.runtime import RunningProcess
from scruffy.slurm import signal_step
from scruffy.storage import (
    StorageError,
    list_commands,
    read_events,
    record_command_receipt,
    remove_command,
    submit_command,
)

TIMEOUT = 15.0
REQUEST = ResourceRequest(nodes=1, gpus_per_node=0, cpus_per_node=1, memory_gb_per_node=1)
POLICY = {
    "max_attempts": 2,
    "retry_on": ["evacuated"],
    "evacuation": {"signal": "USR1", "grace_seconds": 3},
}
def _controller_worker(root: str, workspace: str) -> None:
    try:
        from scruffy.controller import run_controller

        run_controller(
            root=Path(root),
            inventory=(NodeInventory("local", (0,), 4, 8),),
            launcher="local",
            allocation_id="local-allocation",
            poll_interval=0.01,
            cancel_grace=0.1,
            gpu_health_mode="off",
        )
    except Exception:
        traceback.print_exc()
        raise
FAKE_WORKLOAD = r'''
import os
import signal
import time
from scruffy.client import publish_event

def evacuate(_signum, _frame):
    publish_event(
        __import__("pathlib").Path(os.environ["SCRUFFY_ROOT"]),
        job_id=os.environ["SCRUFFY_JOB_ID"],
        event_id="checkpoint-before-evacuated",
        kind="workload.artifact",
        data={"artifact_type": "checkpoint", "publication": {
            "v": 1,
            "artifact_id": "checkpoint/latest",
            "path": os.path.join(os.environ["SCRUFFY_ROOT"], "checkpoint.bin"),
            "size_bytes": 0,
            "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "manifest_path": os.path.join(os.environ["SCRUFFY_ROOT"], "checkpoint.ready.json"),
        }},
    )
    raise SystemExit(75)

if os.environ.get("SCRUFFY_ATTEMPT") != "2":
    signal.signal(signal.SIGUSR1, evacuate)
    while True:
        time.sleep(0.02)
else:
    print("restarted", flush=True)
'''
class EvacuationPhase2Tests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.root = self.workspace / "queue"
        self.process: multiprocessing.Process | None = None

    def tearDown(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if process.is_alive():
            os.kill(process.pid, signal.SIGTERM)
            process.join(TIMEOUT)
        if process.is_alive():
            process.kill()
            process.join()
        self.assertEqual(0, process.exitcode)
    def _start(self) -> None:
        context = multiprocessing.get_context("spawn")
        self.process = context.Process(
            target=_controller_worker, args=(str(self.root), str(self.workspace))
        )
        self.process.start()
        deadline = time.monotonic() + TIMEOUT
        while time.monotonic() < deadline:
            if not self.process.is_alive():
                self.fail("controller exited during startup")
            try:
                snapshot = status(self.root)
                allocation = snapshot.get("allocation") or {}
                if allocation.get("state") == "running":
                    return
            except (FileNotFoundError, KeyError):
                pass
            time.sleep(0.02)
        self.fail("controller did not start")
    def _submit(self, *, recovery: dict[str, Any] | None = POLICY, name: str = "workload") -> str:
        response = submit_job(
            self.root,
            argv=[sys.executable, "-c", FAKE_WORKLOAD],
            name=name,
            cwd=self.workspace,
            environment={},
            request=REQUEST,
            request_id=f"request-{name}",
            workflow_id="evacuation-workflow",
            task_id=name,
            recovery=recovery,
        )
        return str(response["job_id"])
    def _wait_state(self, job_id: str, expected: str) -> dict[str, Any]:
        deadline = time.monotonic() + TIMEOUT
        while time.monotonic() < deadline:
            job = status(self.root, job_id)
            if job.get("state") == expected:
                return job
            time.sleep(0.02)
        self.fail(f"timed out waiting for {job_id} to enter {expected}")
    def test_local_workload_checkpoint_exit75_and_exactly_one_restart(self) -> None:
        self._start()
        job_id = self._submit()
        running = self._wait_state(job_id, "running")
        self.assertEqual("running", running["state"])
        # Allow the worker to install its handler after the launcher reports
        # the process running; the controller's ownership decision remains
        # independent of application-level readiness.
        time.sleep(0.1)
        request = request_evacuation(
            self.root, job_id=job_id, request_id="evacuate-local", resume_after=True
        )
        operation = wait_for_evacuation(self.root, request["request_id"], timeout=TIMEOUT)
        self.assertEqual("complete", operation["state"])
        target = operation["targets"][job_id]
        self.assertEqual("retry_queued", target["outcome"])
        self.assertEqual(1, len([event for event in read_events(self.root) if event["kind"] == "job.recovery_linked"]))
        successor_id = target["successor_job_id"]
        successor = wait_for_job(self.root, successor_id, timeout=TIMEOUT)
        self.assertEqual("succeeded", successor["state"])
        original = status(self.root, job_id)
        self.assertEqual("failed", original["state"])
        self.assertEqual("evacuated", original["reason"])
        self.assertFalse(status(self.root)["draining"])
    def test_nonrestartable_target_is_partial_and_remains_drained(self) -> None:
        self._start()
        job_id = self._submit(recovery=None, name="nonrestartable")
        self.assertEqual("running", self._wait_state(job_id, "running")["state"])
        request_evacuation(self.root, job_id=job_id, request_id="evacuate-partial", resume_after=True)
        operation = wait_for_evacuation(self.root, "evacuate-partial", timeout=TIMEOUT)
        self.assertEqual("partial", operation["state"])
        self.assertEqual("not_restartable", operation["targets"][job_id]["outcome"])
        self.assertTrue(status(self.root)["draining"])
    def test_same_request_is_idempotent_and_conflicting_reuse_is_rejected(self) -> None:
        self._start()
        job_id = self._submit(name="idempotent")
        self.assertEqual("running", self._wait_state(job_id, "running")["state"])
        request_evacuation(self.root, job_id=job_id, request_id="evacuate-idempotent")
        wait_for_evacuation(self.root, "evacuate-idempotent", timeout=TIMEOUT)
        request_evacuation(self.root, job_id=job_id, request_id="evacuate-idempotent")
        with self.assertRaises(StorageError):
            request_evacuation(
                self.root,
                job_id=job_id,
                request_id="evacuate-idempotent",
                resume_after=True,
            )
    def test_scope_and_stale_identity_fail_closed_without_signalling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("node", (0,), 4, 8),),
                launcher="local",
                allocation_id="local",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
                gpu_health_mode="off",
            )
            try:
                job = {
                    "id": "stale",
                    "state": "running",
                    "project_id": "project-a",
                    "workflow_id": "flow",
                    "task_id": "task",
                    "recovery": POLICY,
                    "assignment": Assignment(
                        "stale",
                        ResourceRequest(1, 0, 1, 1),
                        (NodeReservation("node", (), 1, 1),),
                    ).to_dict(),
                    "launch_token": "token",
                }
                controller.state["jobs"]["stale"] = job
                controller.running["stale"] = RunningProcess(None, "token")
                _begin_evacuation(
                    controller,
                    {"request_id": "stale-request", "scope": {"project_id": "project-a"}},
                )
                _advance_evacuation(controller)
                target = controller.state["evacuation"]["targets"]["stale"]
                self.assertEqual("lost", target["outcome"])
                self.assertEqual("launch_identity_changed", target["reason"])
            finally:
                controller.journal.close()
    def test_project_workflow_scope_and_timeout_are_durable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("node", (0,), 4, 8),),
                launcher="local",
                allocation_id="local",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
                gpu_health_mode="off",
            )
            assignment = Assignment(
                "selected",
                ResourceRequest(1, 0, 1, 1),
                (NodeReservation("node", (), 1, 1),),
            ).to_dict()
            for job_id, project, workflow in (
                ("selected", "project-a", "flow-a"),
                ("other-workflow", "project-a", "flow-b"),
                ("other-project", "project-b", "flow-a"),
            ):
                controller.state["jobs"][job_id] = {
                    "id": job_id,
                    "state": "running",
                    "project_id": project,
                    "workflow_id": workflow,
                    "task_id": job_id,
                    "recovery": POLICY,
                    "assignment": {**assignment, "job_id": job_id},
                    "launch_token": job_id,
                }
                controller.running[job_id] = RunningProcess(mock.Mock(pid=100), job_id)
            try:
                _begin_evacuation(
                    controller,
                    {
                        "request_id": "project-workflow",
                        "scope": {"project_id": "project-a", "workflow_id": "flow-a"},
                    },
                )
                self.assertEqual({"selected"}, set(controller.state["evacuation"]["targets"]))
                controller.state["evacuation"]["targets"]["selected"]["deadline_at"] = "2000-01-01T00:00:00+00:00"
                with mock.patch("scruffy.controller.signal_process"):
                    _advance_evacuation(controller)
                target = controller.state["evacuation"]["targets"]["selected"]
                self.assertEqual("timed_out", target["outcome"])
                self.assertEqual("partial", controller.state["evacuation"]["state"])
                self.assertTrue(controller.state["draining"])
            finally:
                controller.journal.close()

    def test_armed_artifact_evacuation_is_exact_and_ack_signals_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("node", (0,), 4, 8),),
                launcher="local",
                allocation_id="local",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
                gpu_health_mode="off",
            )
            assignment = Assignment(
                "trainer",
                ResourceRequest(1, 0, 1, 1),
                (NodeReservation("node", (), 1, 1),),
            ).to_dict()
            job = {
                "id": "trainer",
                "state": "running",
                "project_id": "project-a",
                "workflow_id": "flow",
                "task_id": "trainer",
                "recovery": POLICY,
                "assignment": assignment,
                "launch_token": "trainer-token",
            }
            controller.state["jobs"][job["id"]] = job
            controller.running[job["id"]] = RunningProcess(mock.Mock(pid=100), "trainer-token")
            try:
                request_evacuation(
                    root,
                    project_id="project-a",
                    workflow_id="flow",
                    request_id="armed",
                    resume_after=True,
                    after_task="trainer",
                    after_artifact="checkpoint/2",
                )
                command = next(iter(list_commands(root)))[1]
                _begin_evacuation(controller, command)
                self.assertEqual("armed", controller.state["evacuation"]["state"])
                self.assertFalse(controller.state["draining"])

                publication = {
                    "v": 1,
                    "artifact_id": "checkpoint/2",
                    "path": "/tmp/checkpoint",
                    "size_bytes": 1,
                    "sha256": "a" * 64,
                    "manifest_path": "/tmp/checkpoint.ready.json",
                }
                with mock.patch("scruffy.controller.signal_process") as signal_process:
                    publish_event(
                        root,
                        job_id="trainer",
                        event_id="trigger-event",
                        kind="workload.artifact",
                        data={"publication": publication},
                        source={"launch_token": "trainer-token"},
                    )
                    _ingest_reports(controller)
                    self.assertEqual(1, signal_process.call_count)
                    self.assertEqual("waiting", controller.state["evacuation"]["state"])
                    acknowledged = wait_for_event_ack(
                        root,
                        job_id="trainer",
                        event_id="trigger-event",
                        timeout=0.1,
                    )
                    self.assertTrue(acknowledged[0])
            finally:
                controller.journal.close()
    def test_slurm_signal_accepts_only_numeric_worker_steps(self) -> None:
        with mock.patch("scruffy.slurm.subprocess.run") as run:
            signal_step("123", "123.7")
        run.assert_called_once_with(
            ["scancel", "--ctld", "--quiet", "--signal=USR1", "123.7"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        with self.assertRaises(ValueError):
            signal_step("123", "123.batch")

    def test_signal_receipt_prevents_resend_across_both_crash_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("node", (0,), 2, 2),),
                launcher="local",
                allocation_id="local",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
                gpu_health_mode="off",
            )
            assignment = Assignment(
                "crash-window",
                ResourceRequest(1, 0, 1, 1),
                (NodeReservation("node", (), 1, 1),),
            ).to_dict()
            job = {
                "id": "crash-window",
                "state": "running",
                "project_id": "default",
                "workflow_id": "flow",
                "task_id": "task",
                "recovery": POLICY,
                "assignment": assignment,
                "launch_token": "local-token",
            }
            controller.state["jobs"][job["id"]] = job
            controller.running[job["id"]] = RunningProcess(mock.Mock(pid=321), "local-token")
            try:
                _begin_evacuation(
                    controller,
                    {"request_id": "crash-request", "scope": {"job_id": job["id"]}},
                )
                sends: list[str] = []

                def crash_before_side_effect(*_args: object) -> None:
                    raise KeyboardInterrupt("crash before delivery")

                with (
                    mock.patch(
                        "scruffy.controller.signal_process",
                        side_effect=crash_before_side_effect,
                    ),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    _advance_evacuation(controller)
                self.assertEqual([], sends)
                with mock.patch(
                    "scruffy.controller.signal_process",
                    side_effect=lambda *_args: sends.append("delivered"),
                ):
                    controller.state["evacuation"]["targets"][job["id"]][
                        "deadline_at"
                    ] = "2000-01-01T00:00:00+00:00"
                    _advance_evacuation(controller)
                self.assertEqual([], sends)
                self.assertEqual(
                    "timed_out",
                    controller.state["evacuation"]["targets"][job["id"]]["outcome"],
                )
                self.assertEqual("partial", controller.state["evacuation"]["state"])

                evacuation = controller.state["evacuation"]
                evacuation["request_id"] = "crash-after-request"
                evacuation["targets"][job["id"]].update(
                    {
                        "request_id": "crash-after-request",
                        "outcome": "pending",
                        "deadline_at": "2099-01-01T00:00:00+00:00",
                    }
                )
                evacuation["state"] = "requested"
                controller.state["evacuation_requests"]["crash-after-request"] = {
                    "scope": {"job_id": job["id"]},
                    "resume_after": False,
                    "automatic": False,
                }

                def crash_after_side_effect(*_args: object) -> None:
                    sends.append("delivered")
                    raise KeyboardInterrupt("crash after delivery")

                with (
                    mock.patch(
                        "scruffy.controller.signal_process",
                        side_effect=crash_after_side_effect,
                    ),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    _advance_evacuation(controller)
                with mock.patch(
                    "scruffy.controller.signal_process",
                    side_effect=lambda *_args: sends.append("duplicate"),
                ):
                    _advance_evacuation(controller)
                self.assertEqual(["delivered"], sends)
            finally:
                controller.journal.close()

    def test_command_id_lock_deduplicates_race_and_rejected_ids_remain_burned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            identical = {"kind": "evacuate", "request_id": "raced", "scope": {}, "resume_after": False}
            barrier = threading.Barrier(2)
            results: list[str] = []

            def submit_identical() -> None:
                barrier.wait()
                results.append(submit_command(root, identical))

            threads = [threading.Thread(target=submit_identical) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(["raced", "raced"], sorted(results))
            conflicting = {**identical, "resume_after": True}
            with self.assertRaises(StorageError):
                submit_command(root, conflicting)

            conflict_barrier = threading.Barrier(2)
            conflict_results: list[str] = []

            def submit_conflicting(command: dict[str, Any]) -> None:
                conflict_barrier.wait()
                try:
                    submit_command(root, command)
                except StorageError:
                    conflict_results.append("conflict")
                else:
                    conflict_results.append("accepted")

            conflict_a = {
                "kind": "evacuate",
                "request_id": "conflict-race",
                "scope": {"job_id": "one"},
                "resume_after": False,
            }
            conflict_b = {**conflict_a, "scope": {"job_id": "two"}}
            threads = [
                threading.Thread(target=submit_conflicting, args=(command,))
                for command in (conflict_a, conflict_b)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(["accepted", "conflict"], sorted(conflict_results))

            source = root / "commands" / "raced.json"
            remove_command(source)
            record_command_receipt(root, identical)
            self.assertEqual("raced", submit_command(root, identical))
            with self.assertRaises(StorageError):
                submit_command(root, conflicting)

            controller = _initialize_controller(
                root=root,
                inventory=(NodeInventory("node", (0,), 2, 2),),
                launcher="local",
                allocation_id="local",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
                gpu_health_mode="off",
            )
            try:
                rejected = {
                    "kind": "not-a-command",
                    "request_id": "rejected-first",
                    "payload": "bad",
                }
                submit_command(root, rejected)
                _ingest_commands(controller)
                self.assertEqual("rejected-first", submit_command(root, rejected))
                with self.assertRaises(StorageError):
                    submit_command(root, {**rejected, "payload": "changed"})
            finally:
                controller.journal.close()
if __name__ == "__main__":
    unittest.main()
