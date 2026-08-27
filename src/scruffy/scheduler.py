"""Pure resource placement for Scruffy.

There is intentionally no subprocess or filesystem code here.  The controller
serializes calls to these functions, persists the returned assignments, and
only then launches work.  That narrow boundary is what makes GPU overlap
impossible through Scruffy's queue API.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from itertools import combinations
from typing import Any

from .models import (
    ACTIVE_JOB_STATES,
    Assignment,
    ModelError,
    NodeAvailability,
    NodeInventory,
    NodeReservation,
    QueuedJob,
    ResourceRequest,
    job_project,
    validate_inventory,
)


class InvariantError(RuntimeError):
    """Raised when resource state is inconsistent or overcommitted."""


def project_gpu_usage(jobs: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Return GPUs currently assigned to each project.

    This is deliberately based only on live reservations. Queued demand does
    not make a project look busy, and completed work earns no lasting credit.
    """

    usage: Counter[str] = Counter()
    for job in jobs:
        if (
            job.get("state") not in ACTIVE_JOB_STATES
            or job.get("assignment") is None
        ):
            continue
        try:
            assignment = Assignment.from_dict(job["assignment"])
        except ModelError:
            # Read-only views should remain available if an old record has no
            # decodable reservation. The controller still validates its full
            # active ledger before it can place anything.
            continue
        usage[job_project(job)] += sum(
            len(reservation.gpu_ids) for reservation in assignment.reservations
        )
    return dict(usage)


def queue_priority_key(
    job: dict[str, Any], usage: dict[str, int]
) -> tuple[int, int, str]:
    """Order lower-usage projects first, then preserve FIFO within a tie."""

    queue_order = job.get("queue_order")
    return (
        usage.get(job_project(job), 0),
        queue_order if type(queue_order) is int else 2**63 - 1,
        str(job.get("id") or ""),
    )


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
    unavailable_gpu_ids: Mapping[str, Collection[int]] | None = None,
) -> tuple[NodeAvailability, ...]:
    """Return the resources currently free on each inventory node."""

    assert_invariants(inventory, assignments)
    return _available_resources(inventory, assignments, unavailable_gpu_ids)


def _available_resources(
    inventory: Sequence[NodeInventory],
    assignments: Sequence[Assignment],
    unavailable_gpu_ids: Mapping[str, Collection[int]] | None = None,
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
    if unavailable_gpu_ids:
        for node_name, gpu_ids in unavailable_gpu_ids.items():
            if node_name in free_gpu_ids:
                free_gpu_ids[node_name].difference_update(gpu_ids)
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
    unavailable_gpu_ids: Mapping[str, Collection[int]] | None = None,
    *,
    require_uniform_gpu_ids: bool = False,
) -> Assignment | None:
    """Choose an atomic best-fit assignment, or return ``None`` if it must wait."""

    if not isinstance(job, QueuedJob):
        raise InvariantError("job must be a QueuedJob")
    assert_invariants(inventory, assignments)
    if any(active.job_id == job.job_id for active in assignments):
        raise InvariantError(f"job {job.job_id!r} is already assigned")
    assignment = _candidate_assignment(
        _available_resources(inventory, assignments, unavailable_gpu_ids),
        job,
        require_uniform_gpu_ids=require_uniform_gpu_ids,
    )
    if assignment is None:
        return None
    # Validate the proposed assignment against the whole state before exposing it.
    assert_invariants(inventory, (*assignments, assignment))
    return assignment


def _candidate_assignment(
    free_nodes: Sequence[NodeAvailability],
    job: QueuedJob,
    *,
    require_uniform_gpu_ids: bool = False,
) -> Assignment | None:
    eligible = sorted(
        (node for node in free_nodes if _fits(node, job.request)),
        key=lambda node: _best_fit_key(node, job.request),
    )
    if len(eligible) < job.request.nodes:
        return None

    if require_uniform_gpu_ids and job.request.gpus_per_node:
        selected, common_gpu_ids = _uniform_gpu_candidate(eligible, job.request)
        if selected is None or common_gpu_ids is None:
            return None
    else:
        selected = eligible[: job.request.nodes]
        common_gpu_ids = None
    reservations = tuple(
        NodeReservation(
            node=node.name,
            gpu_ids=(
                common_gpu_ids
                if common_gpu_ids is not None
                else node.gpu_ids[: job.request.gpus_per_node]
            ),
            cpus=job.request.cpus_per_node,
            memory_gb=job.request.memory_gb_per_node,
        )
        for node in selected
    )
    assignment = Assignment(job.job_id, job.request, reservations)
    return assignment


def _uniform_gpu_candidate(
    eligible: Sequence[NodeAvailability], request: ResourceRequest
) -> tuple[list[NodeAvailability] | None, tuple[int, ...] | None]:
    """Find the best nodes sharing one exact GPU slot set.

    Slurm's task-to-GRES mask is node-local and applies the same mask to the
    lowest task on every node. A rectangular Scruffy job therefore needs one
    common logical slot set when exact per-GPU binding is enabled.
    """

    candidates: set[tuple[int, ...]] = set()
    for node in eligible:
        candidates.update(combinations(node.gpu_ids, request.gpus_per_node))

    best: tuple[
        tuple[tuple[int, int, int, str], ...], tuple[int, ...], list[NodeAvailability]
    ] | None = None
    for gpu_ids in sorted(candidates):
        selected = [
            node for node in eligible if set(gpu_ids).issubset(node.gpu_ids)
        ][: request.nodes]
        if len(selected) < request.nodes:
            continue
        key = (
            tuple(_best_fit_key(node, request) for node in selected),
            gpu_ids,
            selected,
        )
        if best is None or key[:2] < best[:2]:
            best = key
    if best is None:
        return None, None
    return best[2], best[1]


def choose_first_fitting_job(
    inventory: Sequence[NodeInventory],
    assignments: Sequence[Assignment],
    queued_jobs: Sequence[QueuedJob],
    unavailable_gpu_ids: Mapping[str, Collection[int]] | None = None,
    *,
    require_uniform_gpu_ids: bool = False,
) -> tuple[QueuedJob, Assignment] | None:
    """Return the first queued job that currently fits.

    The caller owns priority order. Skipping a job that cannot fit provides
    simple backfilling without making submission wait for unrelated work.
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

    free_nodes = _available_resources(
        inventory, assignments, unavailable_gpu_ids
    )
    for job in queued_jobs:
        assignment = _candidate_assignment(
            free_nodes,
            job,
            require_uniform_gpu_ids=require_uniform_gpu_ids,
        )
        if assignment is not None:
            assert_invariants(inventory, (*assignments, assignment))
            return job, assignment
    return None
