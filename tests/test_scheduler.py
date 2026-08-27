from __future__ import annotations

import unittest

from scruffy.models import (
    Assignment,
    NodeInventory,
    NodeReservation,
    QueuedJob,
    ResourceRequest,
)
from scruffy.scheduler import (
    InvariantError,
    assert_invariants,
    available_resources,
    choose_assignment,
    choose_first_fitting_job,
    project_gpu_usage,
    queue_priority_key,
    request_can_ever_fit,
)


def request(
    *, nodes: int = 1, gpus: int = 1, cpus: int = 14, memory_gb: int = 128
) -> ResourceRequest:
    return ResourceRequest(nodes, gpus, cpus, memory_gb)


class SchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inventory = (
            NodeInventory("gpu-a", tuple(range(8)), 112, 1024),
            NodeInventory("gpu-b", tuple(range(4)), 56, 512),
            NodeInventory("gpu-c", tuple(range(8)), 112, 1024),
        )

    def assignment_for(
        self,
        job_id: str,
        node: str,
        gpu_ids: tuple[int, ...],
        *,
        cpus: int = 14,
        memory_gb: int = 128,
    ) -> Assignment:
        resource_request = request(
            gpus=len(gpu_ids), cpus=cpus, memory_gb=memory_gb
        )
        return Assignment(
            job_id,
            resource_request,
            (NodeReservation(node, gpu_ids, cpus, memory_gb),),
        )

    def test_best_fit_packs_smallest_eligible_node(self) -> None:
        job = QueuedJob("small", request())

        assignment = choose_assignment(self.inventory, (), job)

        self.assertIsNotNone(assignment)
        assert assignment is not None
        self.assertEqual(assignment.reservations[0].node, "gpu-b")
        self.assertEqual(assignment.reservations[0].gpu_ids, (0,))

    def test_cpu_only_job_reserves_cpu_and_memory_without_a_gpu(self) -> None:
        job = QueuedJob("cpu-only", request(gpus=0, cpus=20, memory_gb=64))

        assignment = choose_assignment(self.inventory, (), job)

        self.assertIsNotNone(assignment)
        assert assignment is not None
        reservation = assignment.reservations[0]
        self.assertEqual((), reservation.gpu_ids)
        free = {
            node.name: node for node in available_resources(self.inventory, (assignment,))
        }
        self.assertEqual(self.inventory[1].gpu_ids, free["gpu-b"].gpu_ids)
        self.assertEqual(36, free["gpu-b"].cpus)
        self.assertEqual(448, free["gpu-b"].memory_gb)

    def test_cpu_only_and_gpu_jobs_share_only_within_cpu_memory_budgets(self) -> None:
        cpu_assignment = choose_assignment(
            self.inventory,
            (),
            QueuedJob("cpu-only", request(gpus=0, cpus=20, memory_gb=64)),
        )
        assert cpu_assignment is not None

        gpu_assignment = choose_assignment(
            self.inventory,
            (cpu_assignment,),
            QueuedJob("gpu", request(gpus=4, cpus=20, memory_gb=64)),
        )

        self.assertIsNotNone(gpu_assignment)
        assert gpu_assignment is not None
        self.assertEqual("gpu-b", gpu_assignment.reservations[0].node)
        self.assertEqual((0, 1, 2, 3), gpu_assignment.reservations[0].gpu_ids)

    def test_multi_node_placement_is_atomic(self) -> None:
        active = (
            self.assignment_for(
                "fills-b", "gpu-b", (0, 1, 2, 3), cpus=56, memory_gb=512
            ),
        )
        job = QueuedJob("two-node", request(nodes=2, gpus=4, cpus=56, memory_gb=512))

        assignment = choose_assignment(self.inventory, active, job)

        self.assertIsNotNone(assignment)
        assert assignment is not None
        self.assertEqual(
            {reservation.node for reservation in assignment.reservations},
            {"gpu-a", "gpu-c"},
        )
        self.assertTrue(all(len(item.gpu_ids) == 4 for item in assignment.reservations))

    def test_exact_multinode_placement_uses_one_common_gpu_slot_set(self) -> None:
        active = (
            self.assignment_for("fills-b", "gpu-b", (0, 1, 2, 3), cpus=56, memory_gb=512),
            self.assignment_for("uses-zero", "gpu-a", (0,)),
            self.assignment_for("uses-one", "gpu-c", (1,)),
        )
        job = QueuedJob("exact", request(nodes=2, gpus=2, cpus=14, memory_gb=128))

        assignment = choose_assignment(
            self.inventory,
            active,
            job,
            require_uniform_gpu_ids=True,
        )

        self.assertIsNotNone(assignment)
        assert assignment is not None
        self.assertEqual(
            [(2, 3), (2, 3)],
            [item.gpu_ids for item in assignment.reservations],
        )

    def test_multi_node_request_returns_none_without_partial_reservation(self) -> None:
        before: tuple[Assignment, ...] = ()
        job = QueuedJob("too-large", request(nodes=3, gpus=8, cpus=112, memory_gb=1024))

        assignment = choose_assignment(self.inventory, before, job)

        self.assertIsNone(assignment)
        self.assertEqual(before, ())

    def test_available_resources_can_represent_a_fully_reserved_node(self) -> None:
        full = self.assignment_for(
            "full", "gpu-b", (0, 1, 2, 3), cpus=56, memory_gb=512
        )

        free = {
            node.name: node
            for node in available_resources(self.inventory, (full,))
        }

        self.assertEqual(free["gpu-b"].gpu_ids, ())
        self.assertEqual(free["gpu-b"].cpus, 0)
        self.assertEqual(free["gpu-b"].memory_gb, 0)

    def test_unavailable_gpus_are_removed_without_invalidating_active_work(self) -> None:
        active = self.assignment_for("active", "gpu-a", (0,))

        free = {
            node.name: node
            for node in available_resources(
                self.inventory,
                (active,),
                {"gpu-a": {0, 2}, "unknown-node": {0}},
            )
        }

        self.assertEqual((1, 3, 4, 5, 6, 7), free["gpu-a"].gpu_ids)

    def test_placement_skips_unavailable_gpu_slots(self) -> None:
        job = QueuedJob("healthy-only", request(gpus=4))

        assignment = choose_assignment(
            self.inventory,
            (),
            job,
            {"gpu-b": {0}, "gpu-a": {0, 1, 2, 3, 4}, "gpu-c": {0}},
        )

        self.assertIsNotNone(assignment)
        assert assignment is not None
        reservation = assignment.reservations[0]
        self.assertEqual("gpu-c", reservation.node)
        self.assertEqual((1, 2, 3, 4), reservation.gpu_ids)

    def test_gpu_overlap_is_rejected(self) -> None:
        first = self.assignment_for("first", "gpu-a", (0,))
        second = self.assignment_for("second", "gpu-a", (0,))

        with self.assertRaisesRegex(InvariantError, "GPU overlap"):
            assert_invariants(self.inventory, (first, second))

    def test_cpu_and_memory_overcommit_are_rejected(self) -> None:
        cpu_a = self.assignment_for("cpu-a", "gpu-b", (0,), cpus=40, memory_gb=1)
        cpu_b = self.assignment_for("cpu-b", "gpu-b", (1,), cpus=40, memory_gb=1)
        mem_a = self.assignment_for("mem-a", "gpu-b", (0,), cpus=1, memory_gb=300)
        mem_b = self.assignment_for("mem-b", "gpu-b", (1,), cpus=1, memory_gb=300)

        with self.assertRaisesRegex(InvariantError, "CPU overcommit"):
            assert_invariants(self.inventory, (cpu_a, cpu_b))
        with self.assertRaisesRegex(InvariantError, "memory overcommit"):
            assert_invariants(self.inventory, (mem_a, mem_b))

    def test_unknown_node_and_gpu_are_rejected(self) -> None:
        unknown_node = self.assignment_for("node", "gpu-z", (0,))
        unknown_gpu = self.assignment_for("gpu", "gpu-b", (7,))

        with self.assertRaisesRegex(InvariantError, "unknown node"):
            assert_invariants(self.inventory, (unknown_node,))
        with self.assertRaisesRegex(InvariantError, "unknown GPUs"):
            assert_invariants(self.inventory, (unknown_gpu,))

    def test_non_rectangular_assignment_is_rejected(self) -> None:
        resource_request = request(nodes=1, gpus=2)
        malformed = Assignment(
            "job-1",
            resource_request,
            (NodeReservation("gpu-a", (0,), 14, 128),),
        )

        with self.assertRaisesRegex(InvariantError, "non-rectangular"):
            assert_invariants(self.inventory, (malformed,))

    def test_first_fit_backfills_around_job_that_cannot_fit(self) -> None:
        active = (
            self.assignment_for(
                "fills-b", "gpu-b", (0, 1, 2, 3), cpus=56, memory_gb=512
            ),
            self.assignment_for(
                "half-a", "gpu-a", (0, 1, 2, 3), cpus=56, memory_gb=512
            ),
            self.assignment_for(
                "half-c", "gpu-c", (0, 1, 2, 3), cpus=56, memory_gb=512
            ),
        )
        oldest = QueuedJob("oldest", request(nodes=2, gpus=5, cpus=1, memory_gb=1))
        newer = QueuedJob("newer", request())

        choice = choose_first_fitting_job(
            self.inventory, active, (oldest, newer)
        )

        self.assertIsNotNone(choice)
        assert choice is not None
        self.assertEqual(choice[0], newer)

    def test_queue_priority_favors_projects_using_fewer_gpus_then_fifo(self) -> None:
        active = {
            "id": "heavy-active",
            "project_id": "heavy",
            "state": "running",
            "assignment": self.assignment_for(
                "heavy-active", "gpu-a", (0, 1, 2, 3)
            ).to_dict(),
        }
        queued = [
            {"id": "heavy-old", "project_id": "heavy", "queue_order": 1},
            {"id": "light-new", "project_id": "light", "queue_order": 3},
            {"id": "light-old", "project_id": "light", "queue_order": 2},
        ]

        usage = project_gpu_usage([active, *queued])
        ordered = sorted(queued, key=lambda job: queue_priority_key(job, usage))

        self.assertEqual({"heavy": 4}, usage)
        self.assertEqual(
            ["light-old", "light-new", "heavy-old"],
            [job["id"] for job in ordered],
        )

    def test_duplicate_queued_jobs_are_rejected_before_planning(self) -> None:
        duplicate = QueuedJob("same", request())

        with self.assertRaisesRegex(InvariantError, "IDs must be unique"):
            choose_first_fitting_job(
                self.inventory, (), (duplicate, duplicate)
            )

    def test_request_can_ever_fit_checks_all_dimensions(self) -> None:
        self.assertTrue(request_can_ever_fit(self.inventory, request(nodes=2, gpus=8)))
        self.assertFalse(request_can_ever_fit(self.inventory, request(nodes=3, gpus=8)))
        self.assertFalse(request_can_ever_fit(self.inventory, request(cpus=113)))
        self.assertFalse(request_can_ever_fit(self.inventory, request(memory_gb=1025)))


if __name__ == "__main__":
    unittest.main()
