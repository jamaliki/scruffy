from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scruffy.cli import main
from scruffy.client import submit_job, submit_workflow
from scruffy.controller import _ingest_requests, _initialize_controller
from scruffy.leases import PoolConfig, pool_accepts
from scruffy.lifecycle import schedule
from scruffy.mcp_server import dispatch_tool
from scruffy.models import ModelError, NodeInventory, QueuedJob, ResourceRequest
from scruffy.scheduler import choose_first_fitting_job, request_can_ever_fit
from scruffy.slurm import AllocationIncarnation
from scruffy.storage import (
    job_identity_digest,
    list_requests,
    read_events,
    submit_submission,
)
from scruffy.submissions import workflow_submission

INVENTORY = (
    NodeInventory("gpu-1", (0, 1), 8, 64),
    NodeInventory("gpu-6", (0, 1), 8, 64),
)


def _process() -> mock.MagicMock:
    process = mock.MagicMock()
    process.pid = 4321
    process.poll.return_value = None
    return process


class NodeConstraintModelTests(unittest.TestCase):
    def test_constraints_round_trip_sorted_and_omitted_when_unset(self) -> None:
        plain = ResourceRequest(1, 1, 1, 1)
        self.assertEqual(
            {"nodes": 1, "gpus_per_node": 1, "cpus_per_node": 1, "memory_gb_per_node": 1},
            plain.to_dict(),
        )
        constrained = ResourceRequest(
            1, 1, 1, 1, allowed_nodes=["gpu-6", "gpu-1"], exclude_nodes=("gpu-9",)
        )
        document = constrained.to_dict()
        self.assertEqual(["gpu-1", "gpu-6"], document["allowed_nodes"])
        self.assertEqual(["gpu-9"], document["exclude_nodes"])
        self.assertEqual(constrained, ResourceRequest.from_dict(document))

    def test_identity_digest_is_unchanged_without_constraints(self) -> None:
        spec = {
            "v": 1,
            "job_id": "job-x",
            "request_id": "x",
            "name": "x",
            "submitted_at": "2026-11-25T00:00:00+00:00",
            "argv": ["true"],
            "cwd": "/tmp",
            "env": {},
            "resources": {
                "nodes": 1,
                "gpus_per_node": 1,
                "cpus_per_node": 1,
                "memory_gb_per_node": 1,
            },
        }
        before = job_identity_digest(spec)
        spec["resources"] = ResourceRequest.from_dict(spec["resources"]).to_dict()
        self.assertEqual(before, job_identity_digest(spec))
        spec["resources"] = ResourceRequest(1, 1, 1, 1, exclude_nodes=["gpu-6"]).to_dict()
        self.assertNotEqual(before, job_identity_digest(spec))

    def test_invalid_constraints_are_rejected(self) -> None:
        for kwargs in (
            {"allowed_nodes": []},
            {"allowed_nodes": "gpu-1"},
            {"allowed_nodes": ["gpu 1"]},
            {"exclude_nodes": ["gpu-1", "GPU-1.cluster"]},
            {"allowed_nodes": ["gpu-1"], "exclude_nodes": ["gpu-1"]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ModelError):
                ResourceRequest(1, 1, 1, 1, **kwargs)
        with self.assertRaises(ModelError):
            ResourceRequest(2, 1, 1, 1, allowed_nodes=["gpu-1"])
        with self.assertRaises(ModelError):
            ResourceRequest.from_dict(
                {
                    "nodes": 1,
                    "gpus_per_node": 1,
                    "cpus_per_node": 1,
                    "memory_gb_per_node": 1,
                    "allowed_nodes": None,
                }
            )

    def test_short_and_case_insensitive_names_match(self) -> None:
        request = ResourceRequest(1, 1, 1, 1, allowed_nodes=["GPU-1"], exclude_nodes=["gpu-6"])
        self.assertTrue(request.allows_node("gpu-1.cluster.example"))
        self.assertFalse(request.allows_node("gpu-6"))
        self.assertFalse(request.allows_node("gpu-2"))


class NodeConstraintSchedulerTests(unittest.TestCase):
    def test_scheduler_honours_allow_and_exclude_lists(self) -> None:
        allowed = QueuedJob("allowed", ResourceRequest(1, 1, 1, 1, allowed_nodes=["gpu-6"]))
        _, assignment = choose_first_fitting_job(INVENTORY, [], [allowed])
        self.assertEqual("gpu-6", assignment.reservations[0].node)
        excluded = QueuedJob("excluded", ResourceRequest(1, 1, 1, 1, exclude_nodes=["gpu-1"]))
        _, assignment = choose_first_fitting_job(INVENTORY, [], [excluded])
        self.assertEqual("gpu-6", assignment.reservations[0].node)
        both = QueuedJob("both", ResourceRequest(2, 1, 1, 1, exclude_nodes=["gpu-1"]))
        self.assertIsNone(choose_first_fitting_job(INVENTORY, [], [both]))
        self.assertFalse(request_can_ever_fit(INVENTORY, both.request))
        self.assertTrue(request_can_ever_fit(INVENTORY, allowed.request))

    def test_fresh_elastic_leases_never_serve_an_allow_list(self) -> None:
        pool = PoolConfig(
            name="h100",
            nodes=1,
            gpus_per_node=8,
            cpus_per_node=96,
            memory_gb_per_node=900,
            time_limit_seconds=3600,
        )
        timed = {"time_limit_seconds": 60}
        self.assertTrue(pool_accepts(pool, ResourceRequest(1, 1, 1, 1, **timed), "default"))
        self.assertTrue(
            pool_accepts(
                pool, ResourceRequest(1, 1, 1, 1, exclude_nodes=["gpu-6"], **timed), "default"
            )
        )
        self.assertFalse(
            pool_accepts(
                pool, ResourceRequest(1, 1, 1, 1, allowed_nodes=["gpu-6"], **timed), "default"
            )
        )


class NodeConstraintSurfaceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"

    def test_cli_accepts_repeatable_comma_separated_node_lists(self) -> None:
        with (
            mock.patch("scruffy.cli.submit_job", return_value={"job_id": "job-1"}) as submit,
            mock.patch("scruffy.cli._json"),
        ):
            result = main(
                [
                    "--root",
                    str(self.root),
                    "submit",
                    "--gpus-per-node",
                    "1",
                    "--allowed-nodes",
                    "gpu-1,gpu-2",
                    "--allowed-nodes",
                    "gpu-3",
                    "--exclude-nodes",
                    "gpu-2",
                    "--",
                    "true",
                ]
            )
        self.assertEqual(0, result)
        request = submit.call_args.kwargs["request"]
        self.assertEqual(("gpu-1", "gpu-2", "gpu-3"), request.allowed_nodes)
        self.assertEqual(("gpu-2",), request.exclude_nodes)

    async def test_mcp_submission_records_node_lists(self) -> None:
        result = await dispatch_tool(
            self.root,
            "submit_job",
            {
                "_project_id": "project-a",
                "request_id": "pinned",
                "name": "pinned",
                "argv": ["true"],
                "cwd": "/shared",
                "gpus_per_node": 1,
                "allowed_nodes": ["gpu-6"],
                "exclude_nodes": None,
            },
        )
        spec = dict(list_requests(self.root))[result["job_id"]]
        self.assertEqual(["gpu-6"], spec["resources"]["allowed_nodes"])
        self.assertNotIn("exclude_nodes", spec["resources"])


class NodeConstraintControllerTests(unittest.TestCase):
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

    def submit(self, request_id: str, request: ResourceRequest) -> str:
        return submit_job(
            self.root,
            argv=["true"],
            name=request_id,
            cwd=self.root,
            environment={},
            request=request,
            request_id=request_id,
        )["job_id"]

    def test_controller_places_on_allowed_node_and_rejects_unknown_names(self) -> None:
        pinned = self.submit("pinned", ResourceRequest(1, 1, 1, 1, allowed_nodes=["gpu-6"]))
        avoid = self.submit("avoid", ResourceRequest(1, 2, 1, 1, exclude_nodes=["gpu-6"]))
        typo = self.submit("typo", ResourceRequest(1, 1, 1, 1, allowed_nodes=["gpu-66"]))
        excluded_typo = self.submit(
            "excluded-typo", ResourceRequest(1, 1, 1, 1, exclude_nodes=["gpu-66"])
        )
        _ingest_requests(self.controller)
        jobs = self.controller.state["jobs"]
        self.assertEqual("rejected", jobs[typo]["state"])
        self.assertEqual("unknown_nodes", jobs[typo]["reason"])
        self.assertIn("gpu-66", jobs[typo]["error"])
        self.assertEqual("unknown_nodes", jobs[excluded_typo]["reason"])

        with mock.patch("scruffy.lifecycle.subprocess.Popen", return_value=_process()):
            schedule(self.controller)
        self.assertEqual("gpu-6", jobs[pinned]["assignment"]["reservations"][0]["node"])
        self.assertEqual(["gpu-6"], jobs[pinned]["request"]["allowed_nodes"])
        self.assertEqual("gpu-1", jobs[avoid]["assignment"]["reservations"][0]["node"])

    def test_workflow_task_constraints_are_validated_atomically(self) -> None:
        task = {
            "argv": ["true"],
            "cwd": str(self.root),
            "resources": {
                "nodes": 1,
                "gpus_per_node": 1,
                "cpus_per_node": 1,
                "memory_gb_per_node": 1,
                "allowed_nodes": ["gpu-404"],
            },
        }
        tasks = [{**task, "task_id": "only"}]
        # The client preflight already refuses a task that cannot fit the
        # visible inventory; the controller revalidates a raw envelope.
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            submit_workflow(self.root, request_id="flow", workflow_id="flow", tasks=tasks)
        submit_submission(
            self.root,
            workflow_submission(request_id="flow", workflow_id="flow", tasks=tasks),
        )
        _ingest_requests(self.controller)
        self.assertEqual({}, self.controller.state["jobs"])
        rejection = next(
            event for event in read_events(self.root) if event["kind"] == "submission.rejected"
        )
        self.assertIn("outside this allocation", rejection["data"]["reason"])


if __name__ == "__main__":
    unittest.main()
