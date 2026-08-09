from __future__ import annotations

import json
import multiprocessing
import os
import signal
import sys
import tempfile
import time
import traceback
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

from scruffy.client import (
    cancel_job,
    drain_queue,
    publish_event,
    status,
    submit_job,
    submit_workflow,
    wait_for_job,
)
from scruffy.controller import run_controller
from scruffy.models import NodeInventory, ResourceRequest
from scruffy.storage import (
    read_events,
    submission_identity_digest,
    submit_request,
    submit_submission,
    utc_now,
)
from scruffy.submissions import workflow_submission

TIMEOUT = 12.0
REQUEST = ResourceRequest(
    nodes=1,
    gpus_per_node=1,
    cpus_per_node=1,
    memory_gb_per_node=1,
)


class ControllerRetryTests(unittest.TestCase):
    def test_slurm_controller_retries_a_transient_startup_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = mock.Mock(running={})
            with (
                mock.patch(
                    "scruffy.controller._initialize_controller",
                    side_effect=[OSError("stale file handle"), controller],
                ) as initialize,
                mock.patch("scruffy.controller._serve") as serve,
                mock.patch("scruffy.controller.time.sleep") as sleep,
            ):
                run_controller(
                    root=root,
                    inventory=(NodeInventory("gpu-0", (0,), 1, 1),),
                    launcher="slurm",
                    allocation_id="123",
                    slurm_job_id="123",
                )

        self.assertEqual(2, initialize.call_count)
        serve.assert_called_once_with(controller)
        sleep.assert_called_once_with(5)
        controller.journal.close.assert_called_once_with()

    def test_slurm_controller_reloads_after_a_runtime_storage_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            first = mock.Mock(running={"job-1": mock.Mock()})
            recovered = mock.Mock(running={})
            with (
                mock.patch(
                    "scruffy.controller._initialize_controller",
                    side_effect=[first, recovered],
                ) as initialize,
                mock.patch(
                    "scruffy.controller._serve",
                    side_effect=[OSError("I/O error"), None],
                ) as serve,
                mock.patch("scruffy.controller.abandon_processes") as abandon,
                mock.patch("scruffy.controller.time.sleep") as sleep,
            ):
                run_controller(
                    root=root,
                    inventory=(NodeInventory("gpu-0", (0,), 1, 1),),
                    launcher="slurm",
                    allocation_id="123",
                    slurm_job_id="123",
                )

        self.assertEqual(2, initialize.call_count)
        self.assertEqual(2, serve.call_count)
        abandon.assert_called_once_with(first)
        first.journal.close.assert_called_once_with()
        recovered.journal.close.assert_called_once_with()
        sleep.assert_called_once_with(5)


def _controller_worker(
    root_dir: str,
    inventory_documents: list[dict[str, object]],
    cancel_grace: float,
) -> None:
    try:
        run_controller(
            root=Path(root_dir),
            inventory=tuple(
                NodeInventory.from_dict(document) for document in inventory_documents
            ),
            launcher="local",
            allocation_id="test-allocation",
            poll_interval=0.01,
            cancel_grace=cancel_grace,
        )
    except Exception:  # pragma: no cover - reported through the process exit
        traceback.print_exc()
        raise


class ControllerIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.root = self.workspace / "queue"
        self.controller: multiprocessing.Process | None = None

    def _start_controller(
        self, gpu_ids: tuple[int, ...], *, cancel_grace: float = 0.2
    ) -> None:
        inventory = [
            {
                "name": "local-node",
                "gpu_ids": list(gpu_ids),
                "cpus": 8,
                "memory_gb": 32,
            }
        ]
        context = multiprocessing.get_context("spawn")
        self.controller = context.Process(
            target=_controller_worker,
            args=(str(self.root), inventory, cancel_grace),
        )
        self.controller.start()
        self.addCleanup(self._stop_controller)
        self._wait_until(
            lambda: (
                snapshot
                if (snapshot := status(self.root)).get("allocation")
                and snapshot["allocation"]["state"] in {"running", "draining"}
                else None
            ),
            "controller startup",
        )

    def _stop_controller(self) -> None:
        process, self.controller = self.controller, None
        if process is None:
            return
        if process.is_alive():
            os.kill(process.pid, signal.SIGTERM)
            process.join(TIMEOUT)
        if process.is_alive():
            process.kill()
            process.join()
            self.fail("controller did not stop")
        self.assertEqual(0, process.exitcode, "controller exited unexpectedly")

    def _wait_until(self, predicate: Callable[[], Any], description: str) -> Any:
        deadline = time.monotonic() + TIMEOUT
        while time.monotonic() < deadline:
            process = self.controller
            if process is not None and not process.is_alive():
                self.fail(f"controller exited while waiting for {description}")
            try:
                result = predicate()
            except (FileNotFoundError, KeyError):
                result = None
            if result:
                return result
            time.sleep(0.02)
        self.fail(f"timed out waiting for {description}")

    def _wait_for_state(self, job_id: str, *states: str) -> dict[str, Any]:
        def matching_job() -> dict[str, Any] | None:
            try:
                job = status(self.root, job_id)
            except KeyError:
                return None
            return job if job["state"] in states else None

        return self._wait_until(matching_job, f"{job_id} to enter {states}")

    def _submit(
        self,
        name: str,
        code: str,
        *,
        environment: dict[str, str] | None = None,
        project_id: str = "default",
        workflow_id: str | None = None,
        task_id: str | None = None,
        needs: tuple[dict[str, str], ...] = (),
    ) -> str:
        response = submit_job(
            self.root,
            argv=[sys.executable, "-c", code],
            name=name,
            cwd=self.workspace,
            environment=environment or {},
            request=REQUEST,
            request_id=f"test/{name}",
            project_id=project_id,
            workflow_id=workflow_id,
            task_id=task_id,
            needs=needs,
        )
        self.assertEqual("submitted", response["state"])
        self.assertFalse(response["deduplicated"])
        return str(response["job_id"])

    def test_projects_isolate_workflow_identity_and_reach_workers(self) -> None:
        self._start_controller((0,))
        first = self._submit(
            "project-a-train",
            "import os; assert os.environ['SCRUFFY_PROJECT'] == 'project-a'",
            project_id="project-a",
            workflow_id="shared-flow",
            task_id="train",
        )
        second = self._submit(
            "project-b-train",
            "import os; assert os.environ['SCRUFFY_PROJECT'] == 'project-b'",
            project_id="project-b",
            workflow_id="shared-flow",
            task_id="train",
        )

        self.assertEqual("succeeded", wait_for_job(self.root, first, timeout=TIMEOUT)["state"])
        self.assertEqual("succeeded", wait_for_job(self.root, second, timeout=TIMEOUT)["state"])

    def test_atomic_workflow_submission_and_immutable_provenance(self) -> None:
        self._start_controller((0,))
        response = submit_workflow(
            self.root,
            request_id="test/atomic-workflow",
            workflow_id="atomic-workflow",
            project_id="project-a",
            tasks=[
                {
                    "task_id": "prepare",
                    "name": "atomic-prepare",
                    "argv": [
                        sys.executable,
                        "-c",
                        (
                            "import json, os; "
                            "p=os.environ['SCRUFFY_PROVENANCE_PATH']; "
                            "d=json.load(open(p)); "
                            "assert d['job']['task_id']=='prepare'; "
                            "assert os.environ['SCRUFFY_ATTEMPT']=='1'; "
                            "assert os.environ['SCRUFFY_ASSIGNMENT_SHA256']=="
                            "d['assignment_sha256']"
                        ),
                    ],
                    "cwd": str(self.workspace),
                    "resources": REQUEST.to_dict(),
                },
                {
                    "task_id": "finish",
                    "name": "atomic-finish",
                    "argv": [sys.executable, "-c", "print('finished')"],
                    "cwd": str(self.workspace),
                    "resources": REQUEST.to_dict(),
                    "needs": [{"task_id": "prepare", "condition": "succeeded"}],
                },
            ],
        )
        self.assertFalse(response["deduplicated"])
        jobs = {item["task_id"]: item["job_id"] for item in response["tasks"]}

        prepare = wait_for_job(self.root, jobs["prepare"], timeout=TIMEOUT)
        finish = wait_for_job(self.root, jobs["finish"], timeout=TIMEOUT)
        self.assertEqual("succeeded", prepare["state"])
        self.assertEqual("succeeded", finish["state"])
        self.assertEqual(jobs["prepare"], finish["resolved_dependencies"][0]["job_id"])
        self.assertIsNone(finish["assignment"])
        self.assertIsNotNone(finish["last_assignment"])
        request_record = self.root / finish["provenance"]["request"]
        launch = self.root / finish["provenance"]["launch"]
        result = self.root / finish["provenance"]["result"]
        self.assertEqual(0o444, request_record.stat().st_mode & 0o777)
        self.assertEqual(0o444, launch.stat().st_mode & 0o777)
        self.assertEqual(0o444, result.stat().st_mode & 0o777)
        request_document = json.loads(request_record.read_text(encoding="utf-8"))
        self.assertIn("environment_sha256", request_document)
        self.assertNotIn("env", request_document)
        events = read_events(self.root)
        admitted = [event for event in events if event["kind"] == "submission.admitted"]
        self.assertEqual(1, len(admitted))
        self.assertEqual(set(jobs.values()), {job["id"] for job in admitted[0]["jobs"]})

        duplicate = submit_workflow(
            self.root,
            request_id="test/atomic-workflow",
            workflow_id="atomic-workflow",
            project_id="project-a",
            tasks=[
                {
                    "task_id": "prepare",
                    "name": "atomic-prepare",
                    "argv": [
                        sys.executable,
                        "-c",
                        (
                            "import json, os; "
                            "p=os.environ['SCRUFFY_PROVENANCE_PATH']; "
                            "d=json.load(open(p)); "
                            "assert d['job']['task_id']=='prepare'; "
                            "assert os.environ['SCRUFFY_ATTEMPT']=='1'; "
                            "assert os.environ['SCRUFFY_ASSIGNMENT_SHA256']=="
                            "d['assignment_sha256']"
                        ),
                    ],
                    "cwd": str(self.workspace),
                    "resources": REQUEST.to_dict(),
                },
                {
                    "task_id": "finish",
                    "name": "atomic-finish",
                    "argv": [sys.executable, "-c", "print('finished')"],
                    "cwd": str(self.workspace),
                    "resources": REQUEST.to_dict(),
                    "needs": [{"task_id": "prepare", "condition": "succeeded"}],
                },
            ],
        )
        self.assertTrue(duplicate["deduplicated"])

    def test_cpu_only_job_and_wall_time_limit(self) -> None:
        self._start_controller((0,))
        cpu_request = ResourceRequest(1, 0, 1, 1)
        cpu = submit_job(
            self.root,
            argv=[
                sys.executable,
                "-c",
                "import os; assert os.environ['CUDA_VISIBLE_DEVICES']==''",
            ],
            name="cpu-only",
            cwd=self.workspace,
            environment={},
            request=cpu_request,
            request_id="test/cpu-only",
        )["job_id"]
        self.assertEqual("succeeded", wait_for_job(self.root, cpu, timeout=TIMEOUT)["state"])

        limited = submit_job(
            self.root,
            argv=[sys.executable, "-c", "import time; time.sleep(30)"],
            name="time-limited",
            cwd=self.workspace,
            environment={},
            request=ResourceRequest(1, 0, 1, 1, time_limit_seconds=1),
            request_id="test/time-limited",
        )["job_id"]
        terminal = wait_for_job(self.root, limited, timeout=TIMEOUT)
        self.assertEqual("failed", terminal["state"])
        self.assertEqual("timeout", terminal["reason"])

    def test_invalid_atomic_workflow_creates_no_partial_dag(self) -> None:
        self._start_controller((0,))
        document = workflow_submission(
            request_id="test/rejected-atomic",
            workflow_id="rejected-atomic",
            tasks=[
                {
                    "task_id": "good",
                    "argv": [sys.executable, "-c", "print('must not run')"],
                    "cwd": str(self.workspace),
                    "resources": REQUEST.to_dict(),
                },
                {
                    "task_id": "bad",
                    "argv": [sys.executable, "-c", "print('must not run')"],
                    "cwd": str(self.workspace),
                    "resources": REQUEST.to_dict(),
                    "needs": [{"task_id": "good", "condition": "succeeded"}],
                },
            ],
        )
        document["jobs"][1]["argv"] = []
        document["identity_sha256"] = submission_identity_digest(document)
        submit_submission(self.root, document)

        rejection = self._wait_until(
            lambda: next(
                (
                    event
                    for event in read_events(self.root)
                    if event.get("kind") == "submission.rejected"
                    and event.get("data", {}).get("submission_id")
                    == document["submission_id"]
                ),
                None,
            ),
            "atomic submission rejection",
        )
        self.assertIn("argv", rejection["data"]["reason"])
        snapshot = status(self.root)
        self.assertTrue(
            {job["job_id"] for job in document["jobs"]}.isdisjoint(snapshot["jobs"])
        )

    def test_dependencies_block_release_and_skip_without_waiting_to_submit(self) -> None:
        self._start_controller((0,))
        release = self.workspace / "release-dependency"
        child_id = self._submit(
            "workflow-child",
            "print('dependent ran')",
            workflow_id="workflow-success",
            task_id="infer",
            needs=({"task_id": "train", "condition": "succeeded"},),
        )
        missing = self._wait_for_state(child_id, "blocked")
        self.assertEqual("dependency_missing", missing["blockers"][0]["reason"])
        root_id = self._submit(
            "workflow-root",
            "import os, time; from pathlib import Path; "
            "release=Path(os.environ['RELEASE']); "
            "exec(\"while not release.exists():\\n time.sleep(0.01)\")",
            environment={"RELEASE": str(release)},
            workflow_id="workflow-success",
            task_id="train",
        )

        self._wait_for_state(root_id, "running")
        blocked = self._wait_for_state(child_id, "blocked")
        self.assertEqual("train", blocked["blockers"][0]["task_id"])
        release.write_text("go")
        self.assertEqual(
            "succeeded", wait_for_job(self.root, root_id, timeout=TIMEOUT)["state"]
        )
        self.assertEqual(
            "succeeded", wait_for_job(self.root, child_id, timeout=TIMEOUT)["state"]
        )

        failed_id = self._submit(
            "workflow-fails",
            "raise SystemExit(3)",
            workflow_id="workflow-failure",
            task_id="train",
        )
        skipped_id = self._submit(
            "workflow-skipped",
            "raise RuntimeError('must not run')",
            workflow_id="workflow-failure",
            task_id="infer",
            needs=({"task_id": "train", "condition": "succeeded"},),
        )
        cleanup_id = self._submit(
            "workflow-cleanup",
            "print('cleanup ran')",
            workflow_id="workflow-failure",
            task_id="cleanup",
            needs=({"task_id": "train", "condition": "terminal"},),
        )

        self.assertEqual(
            "failed", wait_for_job(self.root, failed_id, timeout=TIMEOUT)["state"]
        )
        skipped = wait_for_job(self.root, skipped_id, timeout=TIMEOUT)
        self.assertEqual(
            ("skipped", "dependency_unsatisfied"),
            (skipped["state"], skipped["reason"]),
        )
        self.assertEqual(
            "succeeded", wait_for_job(self.root, cleanup_id, timeout=TIMEOUT)["state"]
        )

    def test_failed_workflow_attempts_can_be_repaired_in_place(self) -> None:
        self._start_controller((0,))
        first_a = self._submit(
            "repair-a-1",
            "print('old a')",
            workflow_id="repair",
            task_id="a",
            needs=({"task_id": "b", "condition": "succeeded"},),
        )
        self._wait_for_state(first_a, "blocked")
        invalid_b = self._submit(
            "repair-b-invalid",
            "print('invalid b')",
            workflow_id="repair",
            task_id="b",
            needs=({"task_id": "a", "condition": "succeeded"},),
        )
        self._wait_for_state(invalid_b, "rejected")
        self._wait_for_state(first_a, "skipped")

        retry_b = self._submit(
            "repair-b-2",
            "print('new b')",
            workflow_id="repair",
            task_id="b",
        )
        self.assertEqual(
            "succeeded", wait_for_job(self.root, retry_b, timeout=TIMEOUT)["state"]
        )
        duplicate_b = self._submit(
            "repair-b-after-success",
            "print('must not rerun')",
            workflow_id="repair",
            task_id="b",
        )
        self._wait_for_state(duplicate_b, "rejected")
        retry_a = self._submit(
            "repair-a-2",
            "print('new a')",
            workflow_id="repair",
            task_id="a",
            needs=({"task_id": "b", "condition": "succeeded"},),
        )
        self.assertEqual(
            "succeeded", wait_for_job(self.root, retry_a, timeout=TIMEOUT)["state"]
        )

    def test_retry_cycle_check_ignores_edges_of_a_running_task(self) -> None:
        self._start_controller((0,))
        first_a = self._submit(
            "gate-a-1",
            "raise SystemExit(3)",
            workflow_id="gate-retry",
            task_id="a",
        )
        self.assertEqual(
            "failed", wait_for_job(self.root, first_a, timeout=TIMEOUT)["state"]
        )
        release = self.workspace / "release-gate-b"
        running_b = self._submit(
            "gate-b-1",
            "import os,time; "
            "release=os.environ['RELEASE']; "
            "exec(\"while not os.path.exists(release): time.sleep(0.02)\")",
            environment={"RELEASE": str(release)},
            workflow_id="gate-retry",
            task_id="b",
            needs=({"task_id": "a", "condition": "terminal"},),
        )
        self._wait_for_state(running_b, "running")

        retry_a = self._submit(
            "gate-a-2",
            "print('repaired')",
            workflow_id="gate-retry",
            task_id="a",
            needs=({"task_id": "b", "condition": "terminal"},),
        )
        self._wait_for_state(retry_a, "blocked")
        release.write_text("go", encoding="utf-8")

        self.assertEqual(
            "succeeded", wait_for_job(self.root, running_b, timeout=TIMEOUT)["state"]
        )
        self.assertEqual(
            "succeeded", wait_for_job(self.root, retry_a, timeout=TIMEOUT)["state"]
        )

    def test_lost_workflow_tasks_can_be_resubmitted_after_restart(self) -> None:
        self._start_controller((0,))
        first_a = self._submit(
            "lost-a-1",
            "import time; time.sleep(60)",
            workflow_id="lost-repair",
            task_id="a",
        )
        first_b = self._submit(
            "lost-b-1",
            "print('old b')",
            workflow_id="lost-repair",
            task_id="b",
            needs=({"task_id": "a", "condition": "succeeded"},),
        )
        self._wait_for_state(first_a, "running")
        self._wait_for_state(first_b, "blocked")
        self._stop_controller()
        self.assertEqual("lost", status(self.root, first_a)["state"])
        self.assertEqual("skipped", status(self.root, first_b)["state"])

        self._start_controller((0,))
        retry_a = self._submit(
            "lost-a-2",
            "print('new a')",
            workflow_id="lost-repair",
            task_id="a",
        )
        retry_b = self._submit(
            "lost-b-2",
            "print('new b')",
            workflow_id="lost-repair",
            task_id="b",
            needs=({"task_id": "a", "condition": "succeeded"},),
        )
        self.assertEqual(
            "succeeded", wait_for_job(self.root, retry_a, timeout=TIMEOUT)["state"]
        )
        self.assertEqual(
            "succeeded", wait_for_job(self.root, retry_b, timeout=TIMEOUT)["state"]
        )

    def test_invalid_request_inbox_entries_are_rejected_individually(self) -> None:
        request_root = self.root / "requests"
        for request_id, document in {
            "job-broken": "{broken",
            "job-list": "[]",
            "job-missing-id": '{"name":"missing"}',
            "job-directory-id": '{"job_id":"job-phantom","name":"mismatch"}',
        }.items():
            directory = request_root / request_id
            directory.mkdir(parents=True)
            (directory / "spec.json").write_text(document, encoding="utf-8")

        self._start_controller((0,))
        for request_id in (
            "job-broken",
            "job-list",
            "job-missing-id",
            "job-directory-id",
        ):
            self._wait_for_state(request_id, "rejected")
        snapshot = status(self.root)
        self.assertNotIn("job-phantom", snapshot["jobs"])
        self.assertFalse(
            any(job_id.startswith("invalid-") for job_id in snapshot["jobs"])
        )
        self.assertEqual([], list((self.root / "requests").glob("job-*")))
        valid = self._submit("after-corrupt", "print('still alive')")
        self.assertEqual(
            "succeeded", wait_for_job(self.root, valid, timeout=TIMEOUT)["state"]
        )

    def test_spec_invalid_upstream_keeps_workflow_identity(self) -> None:
        parent_id = "job-invalid-parent"
        submit_request(
            self.root,
            {
                "v": 1,
                "job_id": parent_id,
                "request_id": "invalid-parent",
                "name": "invalid-parent",
                "submitted_at": utc_now(),
                "argv": ["true"],
                "cwd": str(self.workspace),
                "env": {},
                "resources": [],
                "workflow_id": "invalid-upstream",
                "task_id": "parent",
                "needs": [],
            },
        )
        self._start_controller((0,))
        self._wait_for_state(parent_id, "rejected")
        child_id = self._submit(
            "invalid-upstream-child",
            "print('must not run')",
            workflow_id="invalid-upstream",
            task_id="child",
            needs=({"task_id": "parent", "condition": "succeeded"},),
        )

        child = self._wait_for_state(child_id, "skipped")

        self.assertEqual("rejected", child["blockers"][0]["state"])

    def test_workload_reports_are_projected_and_broadcast_once(self) -> None:
        self._start_controller((0,))
        release = self.workspace / "release-report"
        job_id = self._submit(
            "reporting",
            "import os, time; from pathlib import Path; "
            "release=Path(os.environ['RELEASE']); "
            "exec(\"while not release.exists():\\n time.sleep(0.01)\")",
            environment={"RELEASE": str(release)},
        )
        self._wait_for_state(job_id, "running")
        response = publish_event(
            self.root,
            job_id=job_id,
            event_id="koochak-progress-1",
            kind="workload.progress",
            data={
                "phase": "training",
                "completed": 12,
                "total": 20,
                "unit": "steps",
                "metrics": {"loss": 1.25},
            },
            source={"name": "koochak", "node": "local-node"},
        )
        self.assertEqual("spooled", response["state"])

        projected = self._wait_until(
            lambda: (
                job
                if (job := status(self.root, job_id)).get("workload", {})
                .get("progress", {})
                .get("completed")
                == 12
                else None
            ),
            "workload progress projection",
        )
        self.assertEqual("training", projected["workload"]["phase"])
        retry = publish_event(
            self.root,
            job_id=job_id,
            event_id="koochak-progress-1",
            kind="workload.progress",
            data={
                "phase": "training",
                "completed": 12,
                "total": 20,
                "unit": "steps",
                "metrics": {"loss": 1.25},
            },
            source={"name": "koochak", "node": "local-node"},
        )
        self.assertTrue(retry["deduplicated"])
        events = [
            event
            for event in read_events(self.root)
            if event.get("source_event_id") == "koochak-progress-1"
        ]
        self.assertEqual(1, len(events))
        self.assertEqual("workload.progress", events[0]["kind"])
        publish_event(
            self.root,
            job_id=job_id,
            event_id="empty-notice",
            kind="workload.notice",
            data={},
            source={},
        )
        empty = self._wait_until(
            lambda: next(
                (
                    event
                    for event in read_events(self.root)
                    if event.get("source_event_id") == "empty-notice"
                ),
                None,
            ),
            "empty workload envelope",
        )
        self.assertEqual({}, empty["data"])
        self.assertEqual({}, empty["source"])
        release.write_text("go")
        self.assertEqual(
            "succeeded", wait_for_job(self.root, job_id, timeout=TIMEOUT)["state"]
        )

    def test_async_success_and_failure_emit_output_before_terminal(self) -> None:
        self._start_controller((0, 1))
        succeeded_id = self._submit(
            "succeeds",
            "import sys; print('success-out', flush=True); "
            "print('success-err', file=sys.stderr, flush=True)",
        )
        failed_id = self._submit(
            "fails",
            "import sys; print('failure-out', flush=True); "
            "print('failure-err', file=sys.stderr, flush=True); raise SystemExit(7)",
        )

        succeeded = wait_for_job(self.root, succeeded_id, timeout=TIMEOUT)
        failed = wait_for_job(self.root, failed_id, timeout=TIMEOUT)

        self.assertEqual(("succeeded", 0), (succeeded["state"], succeeded["exit_code"]))
        self.assertEqual(("failed", 7), (failed["state"], failed["exit_code"]))
        self.assertIn("success-out", (self.root / succeeded["stdout"]).read_text())
        self.assertIn("failure-err", (self.root / failed["stderr"]).read_text())

        events = read_events(self.root)
        for job_id, terminal_kind in (
            (succeeded_id, "job.succeeded"),
            (failed_id, "job.failed"),
        ):
            job_events = [event for event in events if event.get("job_id") == job_id]
            output_events = [
                event for event in job_events if event["kind"] == "job.output"
            ]
            terminal = [event for event in job_events if event["kind"] == terminal_kind]
            self.assertEqual({"stdout", "stderr"}, {e["data"]["stream"] for e in output_events})
            self.assertEqual(1, len(terminal))
            self.assertTrue(all(e["seq"] < terminal[0]["seq"] for e in output_events))

    def test_queued_and_running_cancellation_release_resources(self) -> None:
        self._start_controller((0,), cancel_grace=0.5)
        ready_file = self.workspace / "holder-ready"
        holder_id = self._submit(
            "holder",
            "import os, signal; from pathlib import Path; "
            "signal.signal(signal.SIGTERM, lambda *_: None); "
            "Path(os.environ['READY_FILE']).write_text('ready'); "
            "exec('while True:\\n signal.pause()')",
            environment={"READY_FILE": str(ready_file)},
        )
        self._wait_for_state(holder_id, "running")
        self._wait_until(ready_file.exists, "holder signal handler")

        queued_id = self._submit("queued-cancel", "print('must not run')")
        self._wait_for_state(queued_id, "queued")
        queued_cancel = cancel_job(self.root, queued_id)
        queued = self._wait_for_state(queued_id, "cancelled")
        self.assertEqual("cancelled_before_start", queued["reason"])
        self.assertNotIn(
            "job.starting",
            {
                event["kind"]
                for event in read_events(self.root)
                if event.get("job_id") == queued_id
            },
        )

        running_cancel = cancel_job(self.root, holder_id)
        cancelling = self._wait_for_state(holder_id, "cancelling")
        self.assertIsNotNone(cancelling["assignment"])
        successor_id = self._submit("successor", "print('after cancel')")
        self._wait_for_state(successor_id, "queued")
        occupied = status(self.root)["nodes"]["local-node"]
        self.assertEqual([holder_id], list(occupied["assignments"]))
        self.assertEqual([], occupied["free"]["gpu_ids"])

        cancelled = wait_for_job(self.root, holder_id, timeout=TIMEOUT)
        successor = wait_for_job(self.root, successor_id, timeout=TIMEOUT)
        self.assertEqual("cancelled", cancelled["state"])
        self.assertEqual("succeeded", successor["state"])
        free = status(self.root)["nodes"]["local-node"]
        self.assertEqual({}, free["assignments"])
        self.assertEqual([0], free["free"]["gpu_ids"])
        events = read_events(self.root)
        self.assertTrue(
            any(
                event["kind"] == "job.cancelled"
                and event.get("data", {}).get("request_id")
                == queued_cancel["request_id"]
                for event in events
            )
        )
        self.assertTrue(
            any(
                event["kind"] == "job.cancelling"
                and event.get("data", {}).get("request_id")
                == running_cancel["request_id"]
                for event in events
            )
        )

    def test_two_running_jobs_receive_disjoint_gpu_ids(self) -> None:
        self._start_controller((0, 1))
        release_file = self.workspace / "release"
        ready_files = [self.workspace / "ready-a", self.workspace / "ready-b"]
        code = (
            "import os, time; from pathlib import Path; "
            "Path(os.environ['READY_FILE']).write_text(os.environ['CUDA_VISIBLE_DEVICES']); "
            "release = Path(os.environ['RELEASE_FILE']); "
            "exec(\"while not release.exists():\\n time.sleep(0.01)\")"
        )
        job_ids = [
            self._submit(
                f"parallel-{index}",
                code,
                environment={
                    "READY_FILE": str(ready_file),
                    "RELEASE_FILE": str(release_file),
                },
            )
            for index, ready_file in enumerate(ready_files)
        ]

        for job_id in job_ids:
            self._wait_for_state(job_id, "running")
        self._wait_until(
            lambda: all(ready_file.exists() for ready_file in ready_files),
            "both local workers",
        )
        jobs = status(self.root)["jobs"]
        gpu_ids = [
            jobs[job_id]["assignment"]["reservations"][0]["gpu_ids"][0]
            for job_id in job_ids
        ]
        self.assertEqual({0, 1}, set(gpu_ids))
        self.assertEqual(
            {str(gpu_id) for gpu_id in gpu_ids},
            {ready_file.read_text() for ready_file in ready_files},
        )
        node = status(self.root)["nodes"]["local-node"]
        self.assertEqual(set(job_ids), set(node["assignments"]))

        release_file.write_text("go")
        for job_id in job_ids:
            self.assertEqual(
                "succeeded", wait_for_job(self.root, job_id, timeout=TIMEOUT)["state"]
            )
        self.assertEqual([0, 1], status(self.root)["nodes"]["local-node"]["free"]["gpu_ids"])

    def test_cancel_commands_always_receive_an_observable_outcome(self) -> None:
        self._start_controller((0,))
        job_id = self._submit("already-done", "print('done')")
        self.assertEqual(
            "succeeded", wait_for_job(self.root, job_id, timeout=TIMEOUT)["state"]
        )

        terminal = cancel_job(self.root, job_id)
        unknown = cancel_job(self.root, "missing-job")

        def outcomes() -> tuple[dict[str, Any], dict[str, Any]] | None:
            events = read_events(self.root)
            ignored = next(
                (
                    event
                    for event in events
                    if event["kind"] == "job.cancel_ignored"
                    and event["data"]["request_id"] == terminal["request_id"]
                ),
                None,
            )
            rejected = next(
                (
                    event
                    for event in events
                    if event["kind"] == "command.rejected"
                    and event["data"]["request_id"] == unknown["request_id"]
                ),
                None,
            )
            return (ignored, rejected) if ignored and rejected else None

        ignored, rejected = self._wait_until(outcomes, "cancel acknowledgements")
        self.assertEqual("job_is_succeeded", ignored["data"]["reason"])
        self.assertEqual("unknown_job", rejected["data"]["reason"])
        self.assertEqual("succeeded", status(self.root, job_id)["state"])

    def test_drain_commands_are_correlated_and_idempotent(self) -> None:
        self._start_controller((0,))
        first = drain_queue(self.root)
        second = drain_queue(self.root)

        def outcomes() -> tuple[dict[str, Any], dict[str, Any]] | None:
            events = read_events(self.root)
            accepted = next(
                (
                    event
                    for event in events
                    if event["kind"] == "allocation.draining"
                ),
                None,
            )
            ignored = next(
                (
                    event
                    for event in events
                    if event["kind"] == "allocation.drain_ignored"
                ),
                None,
            )
            return (accepted, ignored) if accepted and ignored else None

        accepted, ignored = self._wait_until(outcomes, "drain acknowledgements")
        self.assertEqual(
            {first["request_id"], second["request_id"]},
            {accepted["data"]["request_id"], ignored["data"]["request_id"]},
        )
        self.assertEqual("already_draining", ignored["data"]["reason"])
        self.assertTrue(status(self.root)["draining"])

    def test_drain_survives_same_allocation_controller_restart(self) -> None:
        self._start_controller((0,))
        drain_queue(self.root)
        self._wait_until(
            lambda: status(self.root)["draining"], "controller to drain"
        )
        self._stop_controller()

        self._start_controller((0,))
        self.assertTrue(status(self.root)["draining"])
        queued = self._submit("drained-restart", "print('must stay queued')")
        time.sleep(0.1)
        self.assertEqual("queued", status(self.root, queued)["state"])


if __name__ == "__main__":
    unittest.main()
