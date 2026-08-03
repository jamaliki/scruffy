"""Queue-state transformations and durable event emission."""

from __future__ import annotations

import copy
from datetime import datetime
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
TERMINAL_STATES = {
    "succeeded",
    "failed",
    "cancelled",
    "lost",
    "rejected",
    "skipped",
}


def _event_key(occurred_at: str, event_id: str) -> tuple[datetime, str]:
    return datetime.fromisoformat(occurred_at.replace("Z", "+00:00")), event_id


def _is_newer(
    event: dict[str, Any], workload: dict[str, Any], field: str
) -> bool:
    previous_at = workload.get(f"{field}_at")
    previous_id = workload.get(f"{field}_event_id")
    if not isinstance(previous_at, str) or not isinstance(previous_id, str):
        return True
    return _event_key(event["occurred_at"], event["event_id"]) >= _event_key(
        previous_at, previous_id
    )


def _remember_event(
    workload: dict[str, Any], field: str, event: dict[str, Any]
) -> None:
    workload[f"{field}_at"] = event["occurred_at"]
    workload[f"{field}_event_id"] = event["event_id"]


def apply_workload_event(
    job: dict[str, Any], event: dict[str, Any], *, recorded_at: str
) -> None:
    """Project one validated producer event onto a job's current workload view.

    Producer reports are deliberately unable to alter lifecycle or placement
    fields.  The complete event remains in the journal; this projection only
    keeps the bounded latest values agents need for a quick status check.
    """

    data = copy.deepcopy(event["data"])
    workload = job.setdefault(
        "workload",
        {
            "phase": None,
            "status": None,
            "phase_at": None,
            "phase_event_id": None,
            "progress": None,
            "progress_at": None,
            "progress_event_id": None,
            "last_update_at": None,
            "last_update_event_id": None,
            "last_recorded_at": None,
            "last_milestone": None,
            "latest_artifacts": [],
            "last_notice": None,
        },
    )
    kind = event["kind"]
    if kind == "workload.phase" and _is_newer(event, workload, "phase"):
        workload["phase"] = data.get("phase")
        workload["status"] = data.get("status")
        _remember_event(workload, "phase", event)
    elif kind == "workload.progress":
        if _is_newer(event, workload, "progress"):
            workload["progress"] = data
            _remember_event(workload, "progress", event)
        if _is_newer(event, workload, "phase"):
            if isinstance(data.get("phase"), str):
                workload["phase"] = data["phase"]
            workload["status"] = "active"
            _remember_event(workload, "phase", event)
    elif kind == "workload.milestone" and _is_newer(
        event, workload, "milestone"
    ):
        workload["last_milestone"] = {
            **data,
            "occurred_at": event["occurred_at"],
            "event_id": event["event_id"],
        }
        _remember_event(workload, "milestone", event)
    elif kind == "workload.artifact":
        artifacts = list(workload.get("latest_artifacts") or [])
        artifacts.append(
            {
                **data,
                "occurred_at": event["occurred_at"],
                "event_id": event["event_id"],
            }
        )
        artifacts.sort(
            key=lambda item: _event_key(item["occurred_at"], item["event_id"])
        )
        workload["latest_artifacts"] = artifacts[-8:]
    elif kind == "workload.notice" and _is_newer(event, workload, "notice"):
        workload["last_notice"] = {
            **data,
            "occurred_at": event["occurred_at"],
            "event_id": event["event_id"],
        }
        _remember_event(workload, "notice", event)
    if _is_newer(event, workload, "last_update"):
        workload["last_update_at"] = event["occurred_at"]
        workload["last_update_event_id"] = event["event_id"]
    workload["last_recorded_at"] = recorded_at


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
    job = {
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
    workflow_id = spec.get("workflow_id")
    task_id = spec.get("task_id")
    needs = spec.get("needs", [])
    if workflow_id is not None or task_id is not None or needs:
        if not isinstance(needs, list):
            raise ValueError("needs must be a JSON array")
        job.update(
            {
                "workflow_id": workflow_id,
                "task_id": task_id,
                "needs": copy.deepcopy(needs),
                "blockers": [],
            }
        )
    return job


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
    occurred_at: str | None = None,
    source_event_id: str | None = None,
    source: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Append one ordered event, then atomically publish the new snapshot."""

    state = controller.state
    state["last_seq"] += 1
    recorded_at = utc_now()
    event: dict[str, Any] = {
        "v": 1,
        "queue_id": state["queue_id"],
        "seq": state["last_seq"],
        "event_id": f"{state['queue_id']}:{state['last_seq']}",
        "at": recorded_at,
        "recorded_at": recorded_at,
        "kind": kind,
        "allocation_id": controller.allocation_id,
    }
    if occurred_at is not None:
        event["occurred_at"] = occurred_at
    if source_event_id is not None:
        event["source_event_id"] = source_event_id
    if source is not None:
        event["source"] = dict(source)
    if job is not None:
        event["job_id"] = job["id"]
        # Complete images let a missing or stale snapshot be rebuilt.
        event["job"] = copy.deepcopy(job)
    if data is not None:
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
