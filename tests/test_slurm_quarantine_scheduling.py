"""Regression: task binding is not a Slurm physical GPU reservation."""

from types import SimpleNamespace
import unittest
from unittest import mock

from scruffy.lifecycle import schedule
from scruffy.models import Assignment, NodeInventory, NodeReservation, ResourceRequest
from scruffy.state import refresh_nodes


class SlurmQuarantineSchedulingTests(unittest.TestCase):
    def controller(self, *, launcher="slurm", gpus=1, nodes=1, healthy_nodes=1):
        inventory = tuple(
            NodeInventory(name, (0, 1, 2, 3), 16, 64)
            for name in ("gpu-6", *(f"healthy-{i}" for i in range(healthy_nodes)))
        )
        request = ResourceRequest(nodes, gpus, 2, 8)
        active_request = ResourceRequest(1, 1, 2, 8)
        active = Assignment(
            "active", active_request, (NodeReservation("gpu-6", (2,), 2, 8),)
        )
        state = {
            "allocation": {"launcher": launcher},
            "draining": False,
            "gpu_health": {
                "mode": "observe", "isolation": "gpu",
                "nodes": {"gpu-6": {"devices": {
                    f"GPU-bad-{slot}": {
                        "uuid": f"GPU-bad-{slot}", "slot": slot,
                        "status": "quarantined", "quarantine_source": "operator",
                    } for slot in (0, 1)
                }}},
            },
            "jobs": {
                "queued": {"id": "queued", "state": "queued", "request": request.to_dict()},
                "active": {
                    "id": "active", "state": "running",
                    "request": active_request.to_dict(), "assignment": active.to_dict(),
                },
            },
        }
        return SimpleNamespace(
            launcher=launcher, inventory=inventory, state=state, stopping=False
        )

    def run_schedule(self, controller):
        def started(_controller, job, assignment):
            job.update(state="starting", assignment=assignment.to_dict())

        with mock.patch("scruffy.lifecycle.start_job", side_effect=started) as start:
            schedule(controller)
        refresh_nodes(controller.state, controller.inventory)
        self.assertEqual("running", controller.state["jobs"]["active"]["state"])
        return start

    def test_slurm_skips_partially_quarantined_best_fit_and_updates_public_capacity(self):
        controller = self.controller()
        start = self.run_schedule(controller)
        self.assertEqual("healthy-0", start.call_args.args[2].reservations[0].node)
        held = controller.state["nodes"]["gpu-6"]
        self.assertEqual([0, 1, 2, 3], held["unavailable_gpu_ids"])
        self.assertEqual([], held["free"]["gpu_ids"])
        self.assertIn("active", held["assignments"])

    def test_no_healthy_capacity_leaves_job_queued_without_launching(self):
        controller = self.controller(healthy_nodes=0)
        self.run_schedule(controller).assert_not_called()
        self.assertEqual("queued", controller.state["jobs"]["queued"]["state"])

    def test_multinode_job_uses_only_healthy_nodes(self):
        controller = self.controller(nodes=2, healthy_nodes=2)
        start = self.run_schedule(controller)
        self.assertEqual(
            {"healthy-0", "healthy-1"},
            {item.node for item in start.call_args.args[2].reservations},
        )

    def test_cpu_work_can_still_use_quarantined_node(self):
        controller = self.controller(gpus=0, healthy_nodes=0)
        self.run_schedule(controller).assert_called_once()

    def test_local_launcher_retains_partial_node_capacity(self):
        controller = self.controller(launcher="local", healthy_nodes=0)
        start = self.run_schedule(controller)
        self.assertEqual((3,), start.call_args.args[2].reservations[0].gpu_ids)
        self.assertEqual([0, 1], controller.state["nodes"]["gpu-6"]["unavailable_gpu_ids"])
