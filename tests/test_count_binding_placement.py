from scruffy.models import Assignment, NodeInventory, NodeReservation, QueuedJob, ResourceRequest
from scruffy.scheduler import choose_first_fitting_job


def test_quarantined_hole_does_not_steal_best_fit_from_healthy_node():
    request = ResourceRequest(1, 1, 14, 240)
    inventory = (NodeInventory("gpu-5", tuple(range(8)), 112, 1992),
                 NodeInventory("gpu-9", tuple(range(8)), 112, 1992))
    occupied = [Assignment(f"job-{gpu}", request, (NodeReservation("gpu-9", (gpu,), 14, 240),))
                for gpu in (0, 1, 3, 4, 5)]
    _, assignment = choose_first_fitting_job(inventory, occupied, [QueuedJob("new", request)],
        {"gpu-9": (2, 7), "gpu-5": (7,)}, slurm_count_binding=True)
    assert assignment.reservations[0].node == "gpu-5"
    assert assignment.reservations[0].gpu_ids == (0,)


def test_lower_healthy_peer_and_cpu_work_remain_schedulable():
    inventory = (NodeInventory("gpu-9", tuple(range(8)), 112, 1992),)
    request = ResourceRequest(1, 1, 14, 240)
    _, assignment = choose_first_fitting_job(inventory, [], [QueuedJob("new", request)],
        {"gpu-9": (2, 7)}, slurm_count_binding=True)
    assert assignment.reservations[0].gpu_ids == (0,)
    cpu = QueuedJob("cpu", ResourceRequest(1, 0, 1, 4))
    assert choose_first_fitting_job(inventory, [], [cpu], {"gpu-9": (0,)}, slurm_count_binding=True)


def test_unreachable_healthy_slot_waits_instead_of_failing_worker_guard():
    inventory = (NodeInventory("gpu-9", tuple(range(8)), 112, 1992),)
    job = QueuedJob("new", ResourceRequest(1, 1, 14, 240))
    assert choose_first_fitting_job(inventory, [], [job], {"gpu-9": (0,)}, slurm_count_binding=True) is None
    assert choose_first_fitting_job(inventory, [], [job], {"gpu-9": (0,)}) is not None
