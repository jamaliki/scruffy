"""Pure resource placement for Scruffy.

There is intentionally no subprocess or filesystem code here.  The controller
serializes calls to these functions, persists the returned assignments, and
only then launches work.  That narrow boundary is what makes GPU overlap
impossible through Scruffy's queue API.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import (
    Assignment,
    NodeAvailability,
    NodeInventory,
    NodeReservation,
    QueuedJob,
    ResourceRequest,
    validate_inventory,
)


class InvariantError(RuntimeError):
    """Raised when resource state is inconsistent or overcommitted."""


def _inventory_by_name(
    inventory: Sequence[NodeInventory],
) -> dict[str, NodeInventory]:
    try:
        validated = validate_inventory(inventory)
    except ValueError as exc:
        raise InvariantError(str(exc)) from exc
    return {node.name: node for node in validated}


def _validate_assignment_shape(assignment: Assignment) -> None:
    request = assignment.request
    if len(assignment.reservations) != request.nodes:
        raise InvariantError(
            f"job {assignment.job_id!r} requested {request.nodes} nodes but has "
            f"{len(assignment.reservations)} reservations"
        )
    for reservation in assignment.reservations:
        actual = (len(reservation.gpu_ids), reservation.cpus, reservation.memory_gb)
        expected = (
            request.gpus_per_node,
            request.cpus_per_node,
            request.memory_gb_per_node,
        )
        if actual != expected:
            raise InvariantError(
                f"job {assignment.job_id!r} has a non-rectangular reservation "
                f"on {reservation.node!r}: expected {expected}, got {actual}"
            )


def assert_invariants(
    inventory: Sequence[NodeInventory],
    assignments: Sequence[Assignment],
) -> None:
    """Validate the complete allocation state.

    Active assignments must have unique job IDs, use known nodes and GPU IDs,
    and never overlap GPUs or exceed CPU or memory capacity.
    """

    nodes = _inventory_by_name(inventory)
    used_gpus = {name: set() for name in nodes}
    used_cpus = {name: 0 for name in nodes}
    used_memory = {name: 0 for name in nodes}
    job_ids: set[str] = set()

    for assignment in assignments:
        if not isinstance(assignment, Assignment):
            raise InvariantError("assignments must contain Assignment values")
        if assignment.job_id in job_ids:
            raise InvariantError(f"duplicate active job {assignment.job_id!r}")
        job_ids.add(assignment.job_id)
        _validate_assignment_shape(assignment)

        for reservation in assignment.reservations:
            if reservation.node not in nodes:
                raise InvariantError(f"unknown node {reservation.node!r}")
            node = nodes[reservation.node]
            gpu_ids = set(reservation.gpu_ids)
            unknown = gpu_ids - set(node.gpu_ids)
            overlap = gpu_ids & used_gpus[reservation.node]
            if unknown:
                raise InvariantError(
                    f"job {assignment.job_id!r} uses unknown GPUs "
                    f"{sorted(unknown)!r} on {reservation.node!r}"
                )
            if overlap:
                raise InvariantError(
                    f"GPU overlap on {reservation.node!r}: {sorted(overlap)!r}"
                )
            used_gpus[reservation.node].update(gpu_ids)
            used_cpus[reservation.node] += reservation.cpus
            used_memory[reservation.node] += reservation.memory_gb
            if used_cpus[reservation.node] > node.cpus:
                raise InvariantError(f"CPU overcommit on {reservation.node!r}")
            if used_memory[reservation.node] > node.memory_gb:
                raise InvariantError(f"memory overcommit on {reservation.node!r}")


def available_resources(
    inventory: Sequence[NodeInventory],
    assignments: Sequence[Assignment] = (),
) -> tuple[NodeAvailability, ...]:
    """Return the resources currently free on each inventory node."""

    assert_invariants(inventory, assignments)
    return _available_resources(inventory, assignments)


def _available_resources(
    inventory: Sequence[NodeInventory], assignments: Sequence[Assignment]
) -> tuple[NodeAvailability, ...]:
    """Compute availability after callers have validated the complete ledger."""

    free_gpu_ids = {node.name: set(node.gpu_ids) for node in inventory}
    free_cpus = {node.name: node.cpus for node in inventory}
    free_memory = {node.name: node.memory_gb for node in inventory}
    for assignment in assignments:
        for reservation in assignment.reservations:
            free_gpu_ids[reservation.node].difference_update(reservation.gpu_ids)
            free_cpus[reservation.node] -= reservation.cpus
            free_memory[reservation.node] -= reservation.memory_gb
    return tuple(
        NodeAvailability(
            name=node.name,
            gpu_ids=tuple(sorted(free_gpu_ids[node.name])),
            cpus=free_cpus[node.name],
            memory_gb=free_memory[node.name],
        )
        for node in inventory
    )


def request_can_ever_fit(
    inventory: Sequence[NodeInventory], request: ResourceRequest
) -> bool:
    """Return whether an empty allocation can satisfy a request atomically."""

    nodes = _inventory_by_name(inventory).values()
    eligible = sum(
        len(node.gpu_ids) >= request.gpus_per_node
        and node.cpus >= request.cpus_per_node
        and node.memory_gb >= request.memory_gb_per_node
        for node in nodes
    )
    return eligible >= request.nodes


def _fits(node: NodeAvailability, request: ResourceRequest) -> bool:
    return (
        len(node.gpu_ids) >= request.gpus_per_node
        and node.cpus >= request.cpus_per_node
        and node.memory_gb >= request.memory_gb_per_node
    )


def _best_fit_key(
    node: NodeAvailability, request: ResourceRequest
) -> tuple[int, int, int, str]:
    """Pack scarce GPU capacity first, with deterministic tie breaking."""

    return (
        len(node.gpu_ids) - request.gpus_per_node,
        node.cpus - request.cpus_per_node,
        node.memory_gb - request.memory_gb_per_node,
        node.name,
    )


def choose_assignment(
    inventory: Sequence[NodeInventory],
    assignments: Sequence[Assignment],
    job: QueuedJob,
) -> Assignment | None:
    """Choose an atomic best-fit assignment, or return ``None`` if it must wait."""

    if not isinstance(job, QueuedJob):
        raise InvariantError("job must be a QueuedJob")
    assert_invariants(inventory, assignments)
    if any(active.job_id == job.job_id for active in assignments):
        raise InvariantError(f"job {job.job_id!r} is already assigned")
    assignment = _candidate_assignment(_available_resources(inventory, assignments), job)
    if assignment is None:
        return None
    # Validate the proposed assignment against the whole state before exposing it.
    assert_invariants(inventory, (*assignments, assignment))
    return assignment


def _candidate_assignment(
    free_nodes: Sequence[NodeAvailability], job: QueuedJob
) -> Assignment | None:
    eligible = sorted(
        (node for node in free_nodes if _fits(node, job.request)),
        key=lambda node: _best_fit_key(node, job.request),
    )
    if len(eligible) < job.request.nodes:
        return None

    selected = eligible[: job.request.nodes]
    reservations = tuple(
        NodeReservation(
            node=node.name,
            gpu_ids=node.gpu_ids[: job.request.gpus_per_node],
            cpus=job.request.cpus_per_node,
            memory_gb=job.request.memory_gb_per_node,
        )
        for node in selected
    )
    assignment = Assignment(job.job_id, job.request, reservations)
    return assignment


def choose_oldest_fitting_job(
    inventory: Sequence[NodeInventory],
    assignments: Sequence[Assignment],
    queued_jobs: Sequence[QueuedJob],
) -> tuple[QueuedJob, Assignment] | None:
    """Return the oldest queued job that currently fits.

    The input is ordered oldest first.  Skipping a blocked job provides simple
    backfilling without making submission wait for unrelated work.
    """

    if not all(isinstance(job, QueuedJob) for job in queued_jobs):
        raise InvariantError("queued_jobs must contain QueuedJob values")
    job_ids = [job.job_id for job in queued_jobs]
    if len(set(job_ids)) != len(job_ids):
        raise InvariantError("queued job IDs must be unique")
    assert_invariants(inventory, assignments)
    active_ids = {assignment.job_id for assignment in assignments}
    duplicate = next((job_id for job_id in job_ids if job_id in active_ids), None)
    if duplicate is not None:
        raise InvariantError(f"job {duplicate!r} is already assigned")

    free_nodes = _available_resources(inventory, assignments)
    for job in queued_jobs:
        assignment = _candidate_assignment(free_nodes, job)
        if assignment is not None:
            assert_invariants(inventory, (*assignments, assignment))
            return job, assignment
    return None
