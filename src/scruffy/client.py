"""Non-blocking producer and observation functions used by the CLI."""

from __future__ import annotations

import copy
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .models import ResourceRequest, TERMINAL_JOB_STATES
from .protocol import validate_event
from .storage import (
    create_job_id,
    find_archived_job,
    list_archived_workflow,
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
from .summary import build_summary, explain_job
from .workflows import validate_workflows


def _workflow_fields(
    workflow_id: str | None,
    task_id: str | None,
    needs: Sequence[Mapping[str, str]] | None,
) -> dict[str, Any]:
    """Validate one task's local metadata without requiring upstream jobs yet."""

    dependencies = () if needs is None else needs
    candidate: dict[str, object] = {"needs": dependencies}
    if workflow_id is not None:
        candidate["workflow_id"] = workflow_id
    if task_id is not None:
        candidate["task_id"] = task_id
    validate_workflows([candidate])
    if workflow_id is None:
        return {}
    return {
        "workflow_id": workflow_id,
        "task_id": task_id,
        "needs": [dict(need) for need in dependencies],
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
    """Durably enqueue a job and return immediately with its stable job ID."""

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
    """Spool an event; ``event_id`` deduplicates across retained generations."""

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
    """Request cancellation asynchronously and return its correlation ID."""

    request_id = submit_command(
        root,
        {"kind": "cancel", "job_id": job_id, "submitted_at": utc_now()},
    )
    return {"job_id": job_id, "request_id": request_id, "state": "cancel_requested"}


def drain_queue(root: Path) -> dict[str, Any]:
    """Disable launches for this allocation; running jobs continue."""

    request_id = submit_command(root, {"kind": "drain", "submitted_at": utc_now()})
    return {"request_id": request_id, "state": "drain_requested"}


def _submitted_from_spec(
    job_id: str, spec: dict[str, Any] | None
) -> dict[str, Any]:
    """Build the small state view used before controller admission."""

    document = spec or {}
    valid = spec is not None and spec.get("job_id") == job_id
    valid = valid and all(
        key in document for key in ("name", "submitted_at", "resources")
    )
    workflow: dict[str, Any] = {}
    workflow_id, task_id = document.get("workflow_id"), document.get("task_id")
    needs = document.get("needs", [])
    if workflow_id is not None or task_id is not None or needs:
        if (
            isinstance(workflow_id, str)
            and isinstance(task_id, str)
            and isinstance(needs, list)
            and all(isinstance(need, dict) for need in needs)
        ):
            workflow = {
                "workflow_id": workflow_id,
                "task_id": task_id,
                "needs": copy.deepcopy(needs),
            }
        else:
            valid = False
    submitted = {
        "id": job_id,
        "name": str(document.get("name", "invalid")),
        "state": "submitted",
        "submitted_at": document.get("submitted_at"),
        "request": copy.deepcopy(document.get("resources")),
        "assignment": None,
        "reason": None if valid else "invalid_request",
        "error": None if valid else "request awaits controller rejection",
        **workflow,
    }
    return submitted


def status(root: Path, job_id: str | None = None) -> dict[str, Any]:
    """Return one job or hot queue state, including unadmitted submissions."""

    state = load_state(root)
    if state is None:
        state = {
            "v": 1,
            "queue_id": queue_id(root),
            "last_seq": 0,
            "journal_generation": 0,
            "journal_offset": 0,
            "allocation": None,
            "nodes": {},
            "jobs": {},
            "report_acks": {},
            "report_ack_v": 1,
            "next_queue_order": 0,
            "archived_jobs": 0,
            "archived_counts": {},
            "draining": False,
            "drain_requested": False,
        }
    state.pop("report_acks", None)
    state.pop("report_ack_v", None)
    state.pop("drain_requested", None)
    if job_id is None:
        jobs = state["jobs"]
        for request_id, spec in list_requests(root, exclude=set(jobs)):
            jobs[request_id] = _submitted_from_spec(request_id, spec)
        return state
    job = state.get("jobs", {}).get(job_id)
    if job is not None:
        return job
    pending = dict(list_requests(root))
    if job_id in pending:
        return _submitted_from_spec(job_id, pending[job_id])
    archived = find_archived_job(root, job_id)
    if archived is not None:
        return archived
    raise KeyError(f"unknown job {job_id}")


def _snapshot_cursor(root: Path) -> tuple[int, int, int]:
    snapshot = load_state(root)
    if snapshot is None:
        return 0, 0, 0
    return (
        int(snapshot.get("journal_generation", 0)),
        int(snapshot.get("last_seq", 0)),
        int(snapshot.get("journal_offset", 0)),
    )


def parse_cursor(root: Path, cursor: str | int | None) -> tuple[int, int, int, bool]:
    """Return generation, sequence, offset, and whether the cursor reset."""

    current_generation, current_sequence, current_offset = _snapshot_cursor(root)
    if cursor is None:
        return current_generation, current_sequence, current_offset, False
    if isinstance(cursor, int):
        if current_generation:
            return current_generation, current_sequence, current_offset, True
        return current_generation, cursor, 0, False
    if ":" not in cursor:
        if current_generation:
            return current_generation, current_sequence, current_offset, True
        return current_generation, int(cursor), 0, False
    parts = cursor.split(":")
    if len(parts) == 2:
        cursor_queue, sequence = parts
        generation = 0
        offset = 0
    elif len(parts) == 3:  # v1 cursor, before journal generations.
        cursor_queue, sequence, offset = parts
        generation = 0
    elif len(parts) == 4:
        cursor_queue, generation, sequence, offset = parts
    else:
        raise ValueError("invalid cursor")
    if cursor_queue != queue_id(root) or int(generation) != current_generation:
        return current_generation, current_sequence, current_offset, True
    return current_generation, int(sequence), int(offset), False


def observe(
    root: Path,
    *,
    after: str | int | None = None,
    wait_seconds: float = 0,
    include_output: bool = False,
    limit: int = 1000,
) -> dict[str, Any]:
    """Return a queue snapshot and one non-consuming page after ``after``."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    generation, sequence, offset, reset = parse_cursor(root, after)
    page_limit = min(limit, 64) if include_output else limit
    deadline = time.monotonic() + max(wait_seconds, 0)
    committed = _snapshot_cursor(root)
    if committed[0] != generation:
        generation, sequence, offset = committed
        reset = True
    page, offset, more = read_event_page(
        root,
        after=sequence,
        offset=offset,
        limit=page_limit,
        end_offset=committed[2],
        generation=generation,
    )
    while True:
        if page or more or time.monotonic() >= deadline:
            break
        time.sleep(min(0.2, max(deadline - time.monotonic(), 0)))
        current = _snapshot_cursor(root)
        if current == committed:
            continue
        if current[0] != generation:
            generation, sequence, offset = current
            page, more, reset = [], False, True
            break
        committed = current
        page, offset, more = read_event_page(
            root,
            after=sequence,
            offset=offset,
            limit=page_limit,
            end_offset=committed[2],
            generation=generation,
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
    snapshot_generation = int(snapshot.get("journal_generation", 0))
    latest_sequence = int(snapshot.get("last_seq", 0))
    latest_offset = int(snapshot.get("journal_offset", 0))
    if snapshot_generation != generation:
        generation = snapshot_generation
        next_sequence = latest_sequence
        offset = latest_offset
        visible = []
        more = False
        reset = True
    cursor = f"{identity}:{generation}:{next_sequence}:{offset}"
    return {
        "snapshot": snapshot,
        "events": visible,
        "next_cursor": cursor,
        "latest_cursor": (
            f"{identity}:{snapshot_generation}:{latest_sequence}:{latest_offset}"
        ),
        "more": more,
        "reset": reset,
    }


def wait_for_job(
    root: Path, job_id: str, *, timeout: float | None = None
) -> dict[str, Any]:
    """Block until one job is terminal, or raise ``TimeoutError``."""

    now = time.monotonic()
    deadline = None if timeout is None else now + timeout
    missing_deadline = now + 1
    while True:
        try:
            job = status(root, job_id)
        except KeyError:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                raise TimeoutError(f"timed out waiting for {job_id}") from None
            if now >= missing_deadline:
                raise
            time.sleep(0.2)
            continue
        if job["state"] in TERMINAL_JOB_STATES:
            return job
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {job_id}")
        time.sleep(0.2)


def summary(root: Path, *, limit: int = 20) -> dict[str, Any]:
    """Return a bounded, action-oriented view of the whole allocation."""

    return build_summary(status(root), limit=limit)


def explain(root: Path, job_id: str) -> dict[str, Any]:
    """Explain one job's state and dependency chain."""

    state = status(root)
    job = state["jobs"].get(job_id)
    if job is None:
        job = status(root, job_id)
        state["jobs"][job_id] = job
    workflow_id = job.get("workflow_id")
    if isinstance(workflow_id, str):
        for archived in list_archived_workflow(root, workflow_id):
            state["jobs"].setdefault(archived["id"], archived)
    return explain_job(state, job_id)
