from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scruffy.cli import main
from scruffy.client import submit_job
from scruffy.controller import _ingest_requests, _initialize_controller, _recovery_successor_spec
from scruffy.lifecycle import schedule
from scruffy.mcp_server import dispatch_tool
from scruffy.models import ModelError, NodeInventory, ResourceRequest
from scruffy.scheduler import queue_priority_key
from scruffy.slurm import AllocationIncarnation
from scruffy.storage import list_requests
from scruffy.submissions import job_from_spec, workflow_submission

INVENTORY = (NodeInventory("gpu-1", (0,), 8, 64),)
REQUEST = ResourceRequest(1, 1, 1, 1)


def _process() -> mock.MagicMock:
    process = mock.MagicMock()
    process.pid = 4321
    process.poll.return_value = None
    return process


def _spec(**fields: object) -> dict[str, object]:
    return {
        "v": 1,
        "job_id": "job-x",
        "request_id": "x",
        "name": "x",
        "submitted_at": "2026-11-25T00:00:00+00:00",
        "argv": ["true"],
        "cwd": "/tmp",
        "env": {},
        "resources": REQUEST.to_dict(),
        **fields,
    }


class PriorityOrderTests(unittest.TestCase):
    def test_priority_precedes_project_usage_and_fifo(self) -> None:
        usage = {"busy": 8}
        jobs = [
            {"id": "fifo-first", "project_id": "idle", "queue_order": 1},
            {"id": "busy-urgent", "project_id": "busy", "queue_order": 3, "priority": 10},
            {"id": "deferred", "project_id": "idle", "queue_order": 0, "priority": -1},
            {"id": "fifo-second", "project_id": "idle", "queue_order": 2},
        ]
        ordered = sorted(jobs, key=lambda job: queue_priority_key(job, usage))
        self.assertEqual(
            ["busy-urgent", "fifo-first", "fifo-second", "deferred"],
            [job["id"] for job in ordered],
        )

    def test_job_images_validate_and_omit_default_priority(self) -> None:
        self.assertNotIn("priority", job_from_spec(_spec(), 1))
        self.assertNotIn("priority", job_from_spec(_spec(priority=0), 1))
        self.assertEqual(7, job_from_spec(_spec(priority=7), 1)["priority"])
        for value in (True, 1.5, "1", 1001, -1001):
            with self.subTest(value=value), self.assertRaises(ModelError):
                job_from_spec(_spec(priority=value), 1)

    def test_workflow_tasks_carry_priority_only_when_set(self) -> None:
        task = {
            "argv": ["true"],
            "cwd": "/tmp",
            "resources": REQUEST.to_dict(),
        }
        document = workflow_submission(
            request_id="flow",
            workflow_id="flow",
            tasks=[{**task, "task_id": "plain"}, {**task, "task_id": "urgent", "priority": 3}],
        )
        plain, urgent = document["jobs"]
        self.assertNotIn("priority", plain)
        self.assertEqual(3, urgent["priority"])
        with self.assertRaisesRegex(ValueError, r"tasks\[0\]\.priority"):
            workflow_submission(
                request_id="flow",
                workflow_id="flow",
                tasks=[{**task, "task_id": "bad", "priority": 5000}],
            )

    def test_recovery_successor_keeps_priority(self) -> None:
        predecessor = {
            "id": "job-a",
            "name": "train",
            "finished_at": "2026-11-25T00:00:00+00:00",
            "argv": ["true"],
            "cwd": "/tmp",
            "env": {},
            "request": REQUEST.to_dict(),
            "workflow_id": "flow",
            "task_id": "train",
            "recovery": {"max_attempts": 2, "retry_on": ["allocation_replaced"]},
            "priority": 4,
        }
        spec = _recovery_successor_spec(predecessor, "job-b", "flow/train/2")
        self.assertEqual(4, spec["priority"])
        del predecessor["priority"]
        self.assertNotIn("priority", _recovery_successor_spec(predecessor, "job-b", "r"))


class PrioritySurfaceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"

    def test_default_priority_keeps_the_legacy_spec(self) -> None:
        result = submit_job(
            self.root,
            argv=["true"],
            name="x",
            cwd=self.root,
            environment={},
            request=REQUEST,
            request_id="x",
        )
        spec = dict(list_requests(self.root))[result["job_id"]]
        self.assertNotIn("priority", spec)
        with self.assertRaises(ModelError):
            submit_job(
                self.root,
                argv=["true"],
                name="y",
                cwd=self.root,
                environment={},
                request=REQUEST,
                request_id="y",
                priority=True,
            )

    def test_cli_priority_option(self) -> None:
        with (
            mock.patch("scruffy.cli.submit_job", return_value={"job_id": "job-1"}) as submit,
            mock.patch("scruffy.cli._json"),
        ):
            main(
                [
                    "--root",
                    str(self.root),
                    "submit",
                    "--gpus-per-node",
                    "1",
                    "--priority",
                    "-5",
                    "--",
                    "true",
                ]
            )
        self.assertEqual(-5, submit.call_args.kwargs["priority"])

    async def test_mcp_priority_parameter(self) -> None:
        result = await dispatch_tool(
            self.root,
            "submit_job",
            {
                "_project_id": "project-a",
                "request_id": "urgent",
                "name": "urgent",
                "argv": ["true"],
                "cwd": "/shared",
                "gpus_per_node": 1,
                "priority": 9,
            },
        )
        spec = dict(list_requests(self.root))[result["job_id"]]
        self.assertEqual(9, spec["priority"])


class PriorityControllerTests(unittest.TestCase):
    def test_higher_priority_job_from_a_busier_project_is_placed_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=INVENTORY,
                launcher="slurm",
                allocation_id="123",
                slurm_job_id="123",
                poll_interval=0.1,
                cancel_grace=0,
                allocation_incarnation=AllocationIncarnation("123", 0, INVENTORY),
                gpu_health_mode="off",
            )
            try:
                controller.state["launches_paused"] = False
                submitted = {}
                for request_id, project_id, priority in (
                    ("first", "idle", 0),
                    ("urgent", "busy", 2),
                ):
                    submitted[request_id] = submit_job(
                        root,
                        argv=["true"],
                        name=request_id,
                        cwd=root,
                        environment={},
                        request=REQUEST,
                        request_id=request_id,
                        project_id=project_id,
                        priority=priority,
                    )["job_id"]
                _ingest_requests(controller)
                with mock.patch("scruffy.lifecycle.subprocess.Popen", return_value=_process()):
                    schedule(controller)
                jobs = controller.state["jobs"]
                self.assertEqual("starting", jobs[submitted["urgent"]]["state"])
                self.assertEqual("queued", jobs[submitted["first"]]["state"])
                self.assertEqual(2, jobs[submitted["urgent"]]["priority"])
            finally:
                controller.journal.close()


if __name__ == "__main__":
    unittest.main()
