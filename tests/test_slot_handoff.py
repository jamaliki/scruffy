from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

from scruffy._compat import UTC
from scruffy.cli import main
from scruffy.client import submit_job, submit_workflow
from scruffy.controller import (
    _ingest_requests,
    _initialize_controller,
    _recovery_successor_spec,
    _refresh_dependencies,
)
from scruffy.lifecycle import _slot_handoff_choice, placement_targets, schedule
from scruffy.mcp_server import dispatch_tool
from scruffy.models import NodeInventory, ResourceRequest
from scruffy.slurm import AllocationIncarnation
from scruffy.state import emit
from scruffy.storage import list_requests, utc_now
from scruffy.submissions import job_from_spec, workflow_submission

INVENTORY = (NodeInventory("gpu-1", (0, 1), 8, 64),)
TASK = {"argv": ["true"], "cwd": "/tmp", "resources": ResourceRequest(1, 1, 1, 1).to_dict()}
NEEDS = [{"task_id": "train", "condition": "succeeded"}]


def _process() -> mock.MagicMock:
    process = mock.MagicMock()
    process.pid = 4321
    process.poll.return_value = None
    return process


class InheritSlotValidationTests(unittest.TestCase):
    def test_flag_requires_needs_and_is_omitted_when_false(self) -> None:
        document = workflow_submission(
            request_id="flow",
            workflow_id="flow",
            tasks=[
                {**TASK, "task_id": "train"},
                {**TASK, "task_id": "eval", "needs": NEEDS, "inherit_slot": True},
                {**TASK, "task_id": "plot", "needs": NEEDS, "inherit_slot": False},
            ],
        )
        train, evaluate, plot = document["jobs"]
        self.assertNotIn("inherit_slot", train)
        self.assertIs(True, evaluate["inherit_slot"])
        self.assertNotIn("inherit_slot", plot)
        self.assertIs(True, job_from_spec(evaluate, 1)["inherit_slot"])
        for task, message in (
            ({**TASK, "task_id": "solo", "inherit_slot": True}, "requires needs"),
            ({**TASK, "task_id": "solo", "needs": NEEDS, "inherit_slot": 1}, "boolean"),
        ):
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, message):
                workflow_submission(
                    request_id="bad",
                    workflow_id="bad",
                    tasks=[{**TASK, "task_id": "train"}, task],
                )

    def test_client_and_cli_pass_the_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = ResourceRequest(1, 1, 1, 1)
            with self.assertRaisesRegex(ValueError, "requires needs"):
                submit_job(
                    root, argv=["true"], name="x", cwd=root, environment={},
                    request=request, request_id="x", inherit_slot=True,
                )
            plain = submit_job(
                root, argv=["true"], name="plain", cwd=root, environment={},
                request=request, request_id="plain",
                workflow_id="flow", task_id="plain", needs=NEEDS,
            )
            chained = submit_job(
                root, argv=["true"], name="chained", cwd=root, environment={},
                request=request, request_id="chained",
                workflow_id="flow", task_id="chained", needs=NEEDS, inherit_slot=True,
            )
            specs = dict(list_requests(root))
            self.assertNotIn("inherit_slot", specs[plain["job_id"]])
            self.assertIs(True, specs[chained["job_id"]]["inherit_slot"])
            with (
                mock.patch("scruffy.cli.submit_job", return_value={}) as submit,
                mock.patch("scruffy.cli._json"),
            ):
                main([
                    "--root", str(root), "submit", "--gpus-per-node", "1",
                    "--workflow-id", "flow", "--task-id", "eval", "--needs", "train",
                    "--inherit-slot", "--", "true",
                ])
            self.assertIs(True, submit.call_args.kwargs["inherit_slot"])

    def test_recovery_successor_keeps_the_flag(self) -> None:
        predecessor = {
            "id": "job-a", "name": "eval", "argv": ["true"], "cwd": "/tmp",
            "request": TASK["resources"], "workflow_id": "flow", "task_id": "eval",
            "needs": NEEDS, "recovery": {"max_attempts": 2, "retry_on": ["node_failure"]},
            "inherit_slot": True, "finished_at": utc_now(),
        }
        spec = _recovery_successor_spec(predecessor, "job-b", "flow/eval/2")
        self.assertIs(True, spec["inherit_slot"])


class InheritSlotMcpTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_submit_passes_the_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = await dispatch_tool(
                root,
                "submit_job",
                {
                    "_project_id": "science",
                    "request_id": "eval",
                    "name": "eval",
                    "argv": ["true"],
                    "cwd": "/shared",
                    "gpus_per_node": 1,
                    "workflow_id": "flow",
                    "task_id": "eval",
                    "needs": NEEDS,
                    "inherit_slot": True,
                },
            )
            self.assertIs(True, dict(list_requests(root))[result["job_id"]]["inherit_slot"])


class SlotHandoffControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"
        self.controller = _initialize_controller(
            root=self.root,
            inventory=INVENTORY,
            launcher="slurm",
            allocation_id="123",
            slurm_job_id="123",
            poll_interval=0.1,
            cancel_grace=0,
            allocation_incarnation=AllocationIncarnation("123", 0, INVENTORY),
            gpu_health_mode="off",
        )
        self.addCleanup(lambda: self.controller.journal.close())
        self.controller.state["launches_paused"] = False
        self.jobs = self.controller.state["jobs"]

    def schedule(self) -> None:
        with mock.patch(
            "scruffy.lifecycle.subprocess.Popen", side_effect=lambda *_a, **_k: _process()
        ):
            schedule(self.controller)

    def job(self, project: str, task_id: str) -> dict[str, object]:
        return next(
            job
            for job in self.jobs.values()
            if job.get("project_id") == project and job.get("task_id") == task_id
        )

    def run_chain(self, *, inherit_slot: bool, gpus: int = 1) -> dict[str, object]:
        """Run a 1-GPU upstream, then free it while a priority job waits."""

        evaluate = {
            **TASK,
            "task_id": "eval",
            "needs": NEEDS,
            "resources": ResourceRequest(1, gpus, 1, 1).to_dict(),
        }
        if inherit_slot:
            evaluate["inherit_slot"] = True
        submit_workflow(
            self.root,
            request_id="chain",
            workflow_id="chain",
            project_id="science",
            tasks=[{**TASK, "task_id": "train"}, evaluate],
        )
        submit_job(
            self.root, argv=["true"], name="filler", cwd=self.root, environment={},
            request=ResourceRequest(1, 1, 1, 1), request_id="filler", project_id="other",
        )
        _ingest_requests(self.controller)
        self.schedule()
        train = self.job("science", "train")
        filler = next(job for job in self.jobs.values() if job.get("name") == "filler")
        self.assertEqual(("starting", "starting"), (train["state"], filler["state"]))
        submit_job(
            self.root, argv=["true"], name="urgent", cwd=self.root, environment={},
            request=ResourceRequest(1, 1, 1, 1), request_id="urgent", project_id="other",
            priority=1000,
        )
        _ingest_requests(self.controller)
        self.schedule()

        self.controller.running.pop(train["id"])
        train.update(
            {
                "state": "succeeded",
                "reason": "process_exit",
                "exit_code": 0,
                "finished_at": utc_now(),
                "last_assignment": train["assignment"],
                "assignment": None,
            }
        )
        emit(self.controller, "job.succeeded", job=train)
        _refresh_dependencies(self.controller)
        self.schedule()
        return next(job for job in self.jobs.values() if job.get("name") == "urgent")

    def test_dependent_takes_the_freed_gpu_before_priority_work(self) -> None:
        urgent = self.run_chain(inherit_slot=True)

        evaluate = self.job("science", "eval")
        self.assertEqual("starting", evaluate["state"])
        self.assertEqual("queued", urgent["state"])
        self.assertEqual(self.job("science", "train")["id"], evaluate["handoff"]["job_id"])
        self.assertEqual(["gpu-1"], evaluate["handoff"]["nodes"])

    def test_without_the_flag_fair_share_and_priority_decide(self) -> None:
        urgent = self.run_chain(inherit_slot=False)

        self.assertEqual("starting", urgent["state"])
        evaluate = self.job("science", "eval")
        self.assertEqual("queued", evaluate["state"])
        self.assertNotIn("handoff", evaluate)

    def test_a_dependent_that_does_not_fit_waits_its_turn(self) -> None:
        urgent = self.run_chain(inherit_slot=True, gpus=2)

        self.assertEqual("starting", urgent["state"])
        self.assertEqual("queued", self.job("science", "eval")["state"])

    def test_an_expired_handoff_is_ignored(self) -> None:
        submit_job(
            self.root, argv=["true"], name="late", cwd=self.root, environment={},
            request=ResourceRequest(1, 1, 1, 1), request_id="late", project_id="other",
        )
        _ingest_requests(self.controller)
        late = next(job for job in self.jobs.values() if job.get("name") == "late")
        handoff = {"job_id": "job-gone", "lease_id": None, "nodes": ["gpu-1"]}
        targets = placement_targets(self.controller)
        for offset, expected in ((-1, None), (30, late["id"])):
            late["handoff"] = {
                **handoff,
                "until": (datetime.now(UTC) + timedelta(seconds=offset)).isoformat(),
            }
            choice = _slot_handoff_choice(self.controller, [late], targets, (), 0.0)
            self.assertEqual(expected, None if choice is None else choice[0])


if __name__ == "__main__":
    unittest.main()
