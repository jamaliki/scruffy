from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from scruffy.models import (
    Assignment,
    ModelError,
    NodeAvailability,
    NodeInventory,
    NodeReservation,
    QueuedJob,
    ResourceRequest,
    validate_inventory,
)


class ModelTests(unittest.TestCase):
    def test_node_inventory_is_canonical_and_frozen(self) -> None:
        node = NodeInventory("gpu-0", (3, 1, 2), 112, 2040)

        self.assertEqual(node.gpu_ids, (1, 2, 3))
        with self.assertRaises(FrozenInstanceError):
            node.cpus = 1  # type: ignore[misc]

    def test_node_availability_allows_exhausted_resources(self) -> None:
        free = NodeAvailability("gpu-0", (), 0, 0)

        self.assertEqual((free.gpu_ids, free.cpus, free.memory_gb), ((), 0, 0))

    def test_models_round_trip_through_json_compatible_dicts(self) -> None:
        request = ResourceRequest(2, 2, 28, 256)
        queued = QueuedJob("job-1", request)
        assignment = Assignment(
            job_id="job-1",
            request=request,
            reservations=(
                NodeReservation("gpu-0", (0, 1), 28, 256),
                NodeReservation("gpu-1", (2, 3), 28, 256),
            ),
        )

        self.assertEqual(ResourceRequest.from_dict(request.to_dict()), request)
        self.assertEqual((queued.job_id, queued.request), ("job-1", request))
        self.assertEqual(Assignment.from_dict(assignment.to_dict()), assignment)

    def test_inventory_round_trip_validates_unique_names(self) -> None:
        inventory = (
            NodeInventory("gpu-0", (0, 1), 28, 256),
            NodeInventory("gpu-1", (0, 1), 28, 256),
        )

        self.assertEqual(validate_inventory(inventory), inventory)
        with self.assertRaisesRegex(ModelError, "names must be unique"):
            validate_inventory((inventory[0], inventory[0]))

    def test_inventory_rejects_short_hostname_aliases(self) -> None:
        inventory = (
            NodeInventory("gpu-0", (0,), 1, 1),
            NodeInventory("gpu-0.cluster", (1,), 1, 1),
        )

        with self.assertRaisesRegex(ModelError, "short node names"):
            validate_inventory(inventory)

    def test_resource_quantities_reject_zero_bool_and_non_integer(self) -> None:
        invalid_values = (0, -1, True, 1.0, "1")

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ModelError):
                    ResourceRequest(value, 1, 1, 1)  # type: ignore[arg-type]

    def test_gpu_ids_must_be_unique_nonnegative_integers(self) -> None:
        invalid_gpu_ids = ((), (0, 0), (-1,), (True,), ("0",))

        for gpu_ids in invalid_gpu_ids:
            with self.subTest(gpu_ids=gpu_ids):
                with self.assertRaises(ModelError):
                    NodeInventory("gpu-0", gpu_ids, 1, 1)  # type: ignore[arg-type]

    def test_names_must_be_nonempty_and_trimmed(self) -> None:
        for name in ("", "   ", " gpu-0", "gpu-0 "):
            with self.subTest(name=name):
                with self.assertRaises(ModelError):
                    NodeInventory(name, (0,), 1, 1)

    def test_from_dict_rejects_missing_and_unknown_fields(self) -> None:
        valid = ResourceRequest(1, 1, 1, 1).to_dict()
        missing = {key: value for key, value in valid.items() if key != "nodes"}
        extra = {**valid, "priority": 1}

        with self.assertRaisesRegex(ModelError, "missing"):
            ResourceRequest.from_dict(missing)
        with self.assertRaisesRegex(ModelError, "unexpected"):
            ResourceRequest.from_dict(extra)

    def test_assignment_rejects_duplicate_nodes(self) -> None:
        request = ResourceRequest(2, 1, 1, 1)
        reservation = NodeReservation("gpu-0", (0,), 1, 1)

        with self.assertRaisesRegex(ModelError, "reserve a node twice"):
            Assignment("job-1", request, (reservation, reservation))


if __name__ == "__main__":
    unittest.main()
