"""Non-blocking producer and observation functions used by the CLI."""

from __future__ import annotations

import copy
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .models import ResourceRequest
from .protocol import validate_event
from .storage import (
    create_job_id,
    find_request,
    journal_size,
    journal_tail,
    list_requests,
    load_state,
    queue_id,
    read_event_page,
    read_output,
    submit_command,
    submit_report,
    submit_request,
    utc_now,
)


TERMINAL_STATES = {
    "succeeded",
    "failed",
    "cancelled",
    "lost",
    "rejected",
    "skipped",
}


def _workflow_fields(
    workflow_id: str | None,
    task_id: str | None,
    needs: Sequence[Mapping[str, str]] | None,
) -> dict[str, Any]:
    """Validate one task's local metadata without requiring upstream jobs yet."""

    if (workflow_id is None) != (task_id is None):
        raise ValueError("workflow_id and task_id must be provided together")
    if workflow_id is not None:
        for value, label in ((workflow_id, "workflow_id"), (task_id, "task_id")):
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"{label} must be a non-empty trimmed string")

    if needs is None:
        dependencies: list[dict[str, str]] = []
    elif isinstance(needs, (str, bytes)) or not isinstance(needs, Sequence):
        raise ValueError("needs must be an array")
    else:
        dependencies = []
        seen: set[str] = set()
        for index, need in enumerate(needs):
            if not isinstance(need, Mapping) or set(need) != {"task_id", "condition"}:
                raise ValueError(
                    f"needs[{index}] must contain exactly task_id and condition"
                )
            dependency = need["task_id"]
            condition = need["condition"]
            if (
                not isinstance(dependency, str)
                or not dependency.strip()
                or dependency != dependency.strip()
            ):
                raise ValueError(f"needs[{index}].task_id must be a non-empty trimmed string")
            if not isinstance(condition, str) or condition not in {
                "succeeded",
                "terminal",
            }:
                raise ValueError(
                    f"needs[{index}].condition must be 'succeeded' or 'terminal'"
                )
            if dependency in seen:
                raise ValueError(f"duplicate dependency on task {dependency!r}")
            if dependency == task_id:
                raise ValueError("a task cannot depend on itself")
            seen.add(dependency)
            dependencies.append({"task_id": dependency, "condition": condition})

    if workflow_id is None:
        if dependencies:
            raise ValueError("needs requires workflow_id and task_id")
        return {}
    return {
        "workflow_id": workflow_id,
        "task_id": task_id,
        "needs": dependencies,
    }


def submit_job(
    root: Path,
    *,
    argv: list[str],
    name: str,
    cwd: Path,
    environment: dict[str, str],
    request: ResourceRequest,
    request_id: str | None,
    workflow_id: str | None = None,
    task_id: str | None = None,
    needs: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Durably enqueue and return without waiting for a controller or GPU."""

    if not argv or not all(isinstance(item, str) and item for item in argv):
        raise ValueError("command must contain at least one non-empty argument")
    if not name.strip():
        raise ValueError("name must not be empty")
    job_id = create_job_id(request_id)
    spec = {
        "v": 1,
        "job_id": job_id,
        "request_id": request_id,
        "name": name,
        "submitted_at": utc_now(),
        "argv": argv,
        "cwd": str(cwd.expanduser().resolve()),
        "env": dict(sorted(environment.items())),
        "resources": request.to_dict(),
        **_workflow_fields(workflow_id, task_id, needs),
    }
    job_id, deduplicated = submit_request(root, spec)
    return {"job_id": job_id, "state": "submitted", "deduplicated": deduplicated}


def publish_event(
    root: Path,
    *,
    job_id: str,
    kind: str,
    data: dict[str, Any],
    event_id: str | None = None,
    occurred_at: str | None = None,
    source: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Durably spool a non-authoritative event reported by a workload."""

    document = validate_event(
        {
            "v": 1,
            "event_id": (
                f"event-{uuid.uuid4().hex}" if event_id is None else event_id
            ),
            "job_id": job_id,
            "occurred_at": utc_now() if occurred_at is None else occurred_at,
            "kind": kind,
            "source": {} if source is None else source,
            "data": data,
        }
    )
    published_id, deduplicated = submit_report(root, document)
    return {
        "event_id": published_id,
        "job_id": document["job_id"],
        "state": "spooled",
        "deduplicated": deduplicated,
    }


def cancel_job(root: Path, job_id: str) -> dict[str, Any]:
    request_id = submit_command(
        root,
        {"kind": "cancel", "job_id": job_id, "submitted_at": utc_now()},
    )
    return {"job_id": job_id, "request_id": request_id, "state": "cancel_requested"}


def drain_queue(root: Path) -> dict[str, Any]:
    request_id = submit_command(root, {"kind": "drain", "submitted_at": utc_now()})
    return {"request_id": request_id, "state": "drain_requested"}


def _submitted_from_spec(job_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Build the small state view used before controller admission."""

    submitted = {
        "id": job_id,
        "name": spec["name"],
        "state": "submitted",
        "submitted_at": spec["submitted_at"],
        "request": copy.deepcopy(spec["resources"]),
        "assignment": None,
        "reason": None,
    }
    for key in ("workflow_id", "task_id", "needs"):
        if key in spec:
            submitted[key] = copy.deepcopy(spec[key])
    return submitted


def status(root: Path, job_id: str | None = None) -> dict[str, Any]:
    state = load_state(root)
    if state is None:
        state = {
            "v": 1,
            "queue_id": queue_id(root),
            "last_seq": 0,
            "journal_offset": 0,
            "allocation": None,
            "nodes": {},
            "jobs": {},
            "draining": False,
        }
    if job_id is None:
        return state
    job = state.get("jobs", {}).get(job_id)
    if job is not None:
        return job
    spec = find_request(root, job_id)
    if spec is not None:
        return _submitted_from_spec(job_id, spec)
    raise KeyError(f"unknown job {job_id}")


def _state_with_submitted(root: Path) -> dict[str, Any]:
    """Merge durable requests not yet reflected in the controller snapshot."""

    state = status(root)
    jobs = state["jobs"]
    for spec in list_requests(root, exclude=set(jobs)):
        job_id = str(spec["job_id"])
        jobs[job_id] = _submitted_from_spec(job_id, spec)
    return state


def _snapshot_cursor(root: Path) -> tuple[int, int]:
    snapshot = load_state(root)
    if snapshot is None:
        return 0, 0
    return int(snapshot.get("last_seq", 0)), int(snapshot.get("journal_offset", 0))


def parse_cursor(root: Path, cursor: str | int | None) -> tuple[int, int, bool]:
    """Return sequence, byte offset, and whether a foreign cursor was reset."""

    if cursor is None:
        sequence, offset = _snapshot_cursor(root)
        return sequence, offset, False
    if isinstance(cursor, int):
        return cursor, 0, False
    if ":" not in cursor:
        return int(cursor), 0, False
    parts = cursor.rsplit(":", 2)
    if len(parts) == 2:
        cursor_queue, sequence = parts
        offset = 0
    else:
        cursor_queue, sequence, offset = parts
    if cursor_queue != queue_id(root):
        sequence_value, offset_value = _snapshot_cursor(root)
        return sequence_value, offset_value, True
    return int(sequence), int(offset), False


def observe(
    root: Path,
    *,
    after: str | int | None = None,
    wait_seconds: float = 0,
    include_output: bool = False,
    limit: int = 1000,
) -> dict[str, Any]:
    """Return a snapshot and non-destructive events after an independent cursor."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    sequence, offset, reset = parse_cursor(root, after)
    page_limit = min(limit, 64) if include_output else limit
    deadline = time.monotonic() + max(wait_seconds, 0)
    observed_size = journal_size(root)
    page, offset, more = read_event_page(
        root, after=sequence, offset=offset, limit=page_limit
    )
    while True:
        if page or more or time.monotonic() >= deadline:
            break
        time.sleep(min(0.2, max(deadline - time.monotonic(), 0)))
        current_size = journal_size(root)
        if current_size == observed_size:
            continue
        observed_size = current_size
        page, offset, more = read_event_page(
            root, after=sequence, offset=offset, limit=page_limit
        )

    next_sequence = max((int(item["seq"]) for item in page), default=sequence)
    visible: list[dict[str, Any]] = []
    for original in page:
        event = copy.deepcopy(original)
        if event.get("kind") == "job.output" and include_output:
            data = event.get("data", {})
            data["text"] = read_output(
                root,
                str(data["log"]),
                int(data["offset"]),
                int(data["length"]),
            )
        visible.append(event)
    snapshot = status(root)
    identity = queue_id(root)
    cursor = f"{identity}:{next_sequence}:{offset}"
    latest_sequence, latest_offset = journal_tail(root)
    return {
        "snapshot": snapshot,
        "events": visible,
        "next_cursor": cursor,
        "latest_cursor": f"{identity}:{latest_sequence}:{latest_offset}",
        "more": more,
        "reset": reset,
    }


def wait_for_job(
    root: Path, job_id: str, *, timeout: float | None = None
) -> dict[str, Any]:
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        job = status(root, job_id)
        if job["state"] in TERMINAL_STATES:
            return job
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {job_id}")
        time.sleep(0.2)


def summary(root: Path, *, limit: int = 20) -> dict[str, Any]:
    """Return a bounded, action-oriented view of the whole allocation."""

    from .summary import build_summary

    return build_summary(_state_with_submitted(root), limit=limit)


def explain(root: Path, job_id: str) -> dict[str, Any]:
    """Explain one job's state and dependency chain."""

    from .summary import explain_job

    return explain_job(_state_with_submitted(root), job_id)
