"""Queue-state transformations and durable event emission."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .models import Assignment, NodeInventory, ResourceRequest
from .runtime import Controller
from .scheduler import assert_invariants
from .storage import (
    append_event,
    journal_tail,
    load_state,
    queue_id,
    read_events,
    utc_now,
    write_state,
)


ACTIVE_STATES = {"starting", "running", "cancelling", "finishing"}
TERMINAL_STATES = {"succeeded", "failed", "cancelled", "lost", "rejected"}


def job_from_spec(spec: dict[str, Any], queue_order: int) -> dict[str, Any]:
    """Validate a client request and create its controller-owned job image."""

    request = ResourceRequest.from_dict(spec["resources"])
    argv = spec.get("argv")
    if not isinstance(argv, list) or not argv or not all(
        isinstance(item, str) and item for item in argv
    ):
        raise ValueError("argv must be a non-empty array of strings")
    environment = spec.get("env", {})
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ValueError("env must map strings to strings")
    cwd = Path(str(spec["cwd"]))
    if not cwd.is_absolute():
        raise ValueError("cwd must be absolute")
    return {
        "id": str(spec["job_id"]),
        "name": str(spec["name"]),
        "state": "queued",
        "submitted_at": str(spec["submitted_at"]),
        "queue_order": queue_order,
        "argv": argv,
        "cwd": str(cwd),
        "env": environment,
        "request": request.to_dict(),
        "assignment": None,
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
        "signal": None,
        "reason": None,
        "error": None,
    }


def active_assignments(state: dict[str, Any]) -> tuple[Assignment, ...]:
    """Decode every assignment which must still hold resources."""

    return tuple(
        Assignment.from_dict(job["assignment"])
        for job in state["jobs"].values()
        if job["state"] in ACTIVE_STATES and job.get("assignment") is not None
    )


def refresh_nodes(
    state: dict[str, Any], inventory: tuple[NodeInventory, ...]
) -> None:
    """Rebuild the public per-node ledger from active assignments."""

    assignments = active_assignments(state)
    assert_invariants(inventory, assignments)
    nodes = {
        item.name: {
            "capacity": item.to_dict(),
            "free": {
                "gpu_ids": list(item.gpu_ids),
                "cpus": item.cpus,
                "memory_gb": item.memory_gb,
            },
            "assignments": {},
        }
        for item in inventory
    }
    for assignment in assignments:
        for reservation in assignment.reservations:
            node = nodes[reservation.node]
            free = node["free"]
            free["gpu_ids"] = [
                gpu_id
                for gpu_id in free["gpu_ids"]
                if gpu_id not in reservation.gpu_ids
            ]
            free["cpus"] -= reservation.cpus
            free["memory_gb"] -= reservation.memory_gb
            node["assignments"][assignment.job_id] = reservation.to_dict()
    state["nodes"] = nodes
    state["updated_at"] = utc_now()


def emit(
    controller: Controller,
    kind: str,
    *,
    job: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    durable: bool = True,
    snapshot: bool = True,
) -> dict[str, Any]:
    """Append one ordered event, then atomically publish the new snapshot."""

    state = controller.state
    state["last_seq"] += 1
    event: dict[str, Any] = {
        "v": 1,
        "queue_id": state["queue_id"],
        "seq": state["last_seq"],
        "at": utc_now(),
        "kind": kind,
        "allocation_id": controller.allocation_id,
    }
    if job is not None:
        event["job_id"] = job["id"]
        # Complete images let a missing or stale snapshot be rebuilt.
        event["job"] = copy.deepcopy(job)
    if data:
        event["data"] = data
        if "job_id" in data and "job_id" not in event:
            event["job_id"] = data["job_id"]
    append_event(controller.journal, event, sync=durable)
    state["journal_offset"] = controller.journal.tell()
    if snapshot:
        refresh_nodes(state, controller.inventory)
        write_state(controller.root, state)
    return event


def load_recovered_state(root: Path) -> dict[str, Any]:
    """Load a snapshot and replay newer complete job images from the journal."""

    state = load_state(root)
    rebuilding = state is None
    if rebuilding:
        state = {
            "v": 1,
            "queue_id": queue_id(root),
            "last_seq": 0,
            "journal_offset": 0,
            "allocation": None,
            "nodes": {},
            "jobs": {},
            "draining": False,
            "updated_at": utc_now(),
        }
    for event in read_events(root, after=int(state.get("last_seq", 0))):
        allocation_id = event.get("allocation_id")
        if allocation_id and (
            rebuilding or event.get("kind") == "allocation.started"
        ):
            state["allocation"] = {"id": str(allocation_id), "state": "recovered"}
        job = event.get("job")
        if isinstance(job, dict) and "id" in job:
            state.setdefault("jobs", {})[str(job["id"])] = job
        state["last_seq"] = max(int(state.get("last_seq", 0)), int(event["seq"]))
    if rebuilding or int(state.get("last_seq", 0)) > 0:
        _, state["journal_offset"] = journal_tail(root)
    return state
