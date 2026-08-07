"""Small, pure views of queue state intended for humans and agents."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection
from datetime import datetime, timezone
from typing import Any

from .models import (
    ACTIVE_JOB_STATES,
    DEFAULT_PROJECT,
    TERMINAL_JOB_STATES,
    job_project,
    normalize_project_id,
)
from .workflows import select_task_attempts

ATTENTION_STATES = {"failed", "lost", "rejected", "skipped"}
QUEUE_VIEW_STATES = frozenset({"submitted", "queued"})
RUNNING_VIEW_STATES = ACTIVE_JOB_STATES
BLOCKED_VIEW_STATES = frozenset({"blocked"})


def state_cursor(state: dict[str, Any]) -> str:
    """Return the opaque cursor for one committed state snapshot."""

    return (
        f"{state.get('queue_id')}:{state.get('journal_generation', 0)}:"
        f"{state.get('last_seq', 0)}:{state.get('journal_offset', 0)}"
    )


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def job_view(job: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """Return the bounded job projection shared by summaries and observers."""

    current = now or datetime.now(timezone.utc)
    started = _parse_time(job.get("started_at"))
    finished = _parse_time(job.get("finished_at"))
    elapsed = max(0.0, ((finished or current) - started).total_seconds()) if started else None
    workload = job.get("workload") if isinstance(job.get("workload"), dict) else None
    updated = _parse_time(workload.get("last_update_at")) if workload else None
    progress_age = max(0.0, (current - updated).total_seconds()) if updated else None
    return {
        "id": job["id"],
        "project_id": job_project(job),
        "name": job.get("name"),
        "state": job.get("state"),
        "reason": job.get("reason"),
        "error": job.get("error"),
        "exit_code": job.get("exit_code"),
        "signal": job.get("signal"),
        "submitted_at": job.get("submitted_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "elapsed_seconds": elapsed,
        "request": job.get("request"),
        "workflow_id": job.get("workflow_id"),
        "task_id": job.get("task_id"),
        "needs": list(job.get("needs") or []),
        "blockers": list(job.get("blockers") or []),
        "assignment": job.get("assignment"),
        "workload": workload,
        "progress_age_seconds": progress_age,
        "stdout": job.get("stdout"),
        "stderr": job.get("stderr"),
    }


def job_identity(
    job: dict[str, Any], *, include_project: bool, include_elapsed: bool
) -> dict[str, Any]:
    """Return only the identity fields needed before inspecting a job."""

    view = job_view(job)
    result = {
        "id": view["id"],
        **({"project_id": view["project_id"]} if include_project else {}),
        "name": view["name"],
        "state": view["state"],
    }
    if include_elapsed:
        elapsed = view["elapsed_seconds"]
        result["elapsed_seconds"] = int(elapsed) if elapsed is not None else None
    return result


def compact_job_page(
    state: dict[str, Any],
    *,
    states: Collection[str] | None,
    offset: int,
    limit: int,
    project_id: str | None,
    include_elapsed: bool,
) -> dict[str, Any]:
    """Return one stable page of compact job identities."""

    selected_states = set(states) if states is not None else None
    jobs = [
        job
        for job in state.get("jobs", {}).values()
        if isinstance(job, dict)
        and (selected_states is None or job.get("state") in selected_states)
        and (project_id is None or job_project(job) == project_id)
    ]
    jobs.sort(
        key=lambda job: (
            _queue_order(job),
            str(job.get("submitted_at") or ""),
            str(job.get("id") or ""),
        )
    )
    page = jobs[offset : offset + limit]
    return {
        "v": 1,
        "queue_id": state.get("queue_id"),
        "project_id": project_id,
        "as_of_cursor": state_cursor(state),
        "total": len(jobs),
        "offset": offset,
        "more": offset + len(page) < len(jobs),
        "jobs": [
            job_identity(
                job,
                include_project=project_id is None,
                include_elapsed=include_elapsed,
            )
            for job in page
        ],
    }


def resource_totals(nodes: object) -> dict[str, int]:
    """Sum node capacity and availability without exposing assignments."""

    result = {
        "nodes": 0,
        "gpus_total": 0,
        "gpus_free": 0,
        "cpus_total": 0,
        "cpus_free": 0,
        "memory_gb_total": 0,
        "memory_gb_free": 0,
    }
    if not isinstance(nodes, dict):
        return result
    for node in nodes.values():
        if not isinstance(node, dict):
            continue
        capacity = node.get("capacity", {})
        free = node.get("free", {})
        result["nodes"] += 1
        result["gpus_total"] += len(capacity.get("gpu_ids", []))
        result["gpus_free"] += len(free.get("gpu_ids", []))
        for key in ("cpus", "memory_gb"):
            result[f"{key}_total"] += int(capacity.get(key, 0) or 0)
            result[f"{key}_free"] += int(free.get(key, 0) or 0)
    return result


def _node_sort_key(name: str) -> tuple[str, int, str]:
    prefix, separator, suffix = name.rpartition("-")
    if separator and suffix.isascii() and suffix.isdecimal():
        return prefix, int(suffix), name
    return name, -1, name


def resource_view(state: dict[str, Any]) -> dict[str, Any]:
    """Return aggregate and per-node physical resource availability."""

    nodes = state.get("nodes", {})
    rows = []
    if isinstance(nodes, dict):
        ordered_nodes = sorted(
            nodes.items(), key=lambda item: _node_sort_key(item[0])
        )
        for name, node in ordered_nodes:
            if not isinstance(node, dict):
                continue
            capacity = node.get("capacity", {})
            free = node.get("free", {})
            rows.append(
                {
                    "name": name,
                    "gpus_free": len(free.get("gpu_ids", [])),
                    "gpus_total": len(capacity.get("gpu_ids", [])),
                    "cpus_free": int(free.get("cpus", 0) or 0),
                    "cpus_total": int(capacity.get("cpus", 0) or 0),
                    "memory_gb_free": int(free.get("memory_gb", 0) or 0),
                    "memory_gb_total": int(capacity.get("memory_gb", 0) or 0),
                }
            )
    allocation = state.get("allocation")
    allocation = allocation if isinstance(allocation, dict) else {}
    return {
        "v": 1,
        "queue_id": state.get("queue_id"),
        "project_id": state.get("project_id"),
        "as_of_cursor": state_cursor(state),
        "allocation": {
            "id": allocation.get("id"),
            "state": allocation.get("state"),
        },
        "totals": resource_totals(nodes),
        "nodes": rows,
    }


def _queue_order(job: dict[str, Any]) -> int:
    value = job.get("queue_order")
    return value if type(value) is int else 0


def _recent_key(job: dict[str, Any]) -> str:
    workload = job.get("workload")
    workload_at = workload.get("last_update_at") if isinstance(workload, dict) else None
    return str(job.get("finished_at") or workload_at or job.get("submitted_at") or "")


def build_summary(
    state: dict[str, Any],
    *,
    now: datetime | None = None,
    limit: int = 20,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Return a bounded, action-oriented view without mutating queue state."""

    if limit <= 0:
        raise ValueError("summary limit must be positive")
    current = now or datetime.now(timezone.utc)
    selected_project = (
        normalize_project_id(project_id) if project_id is not None else None
    )
    jobs = [
        job
        for job in state.get("jobs", {}).values()
        if selected_project is None or job_project(job) == selected_project
    ]
    counts = Counter(str(job.get("state", "unknown")) for job in jobs)
    if selected_project is None:
        archived_counts = state.get("archived_counts", {})
    else:
        by_project = state.get("archived_project_counts")
        if isinstance(by_project, dict):
            archived_counts = by_project.get(selected_project, {})
        elif selected_project == DEFAULT_PROJECT:
            archived_counts = state.get("archived_counts", {})
        else:
            archived_counts = {}
    counts.update(
        {
            str(name): int(count)
            for name, count in archived_counts.items()
        }
    )
    submitted_jobs = sorted(
        (job for job in jobs if job.get("state") == "submitted"),
        key=lambda job: str(job.get("submitted_at") or ""),
    )
    active_jobs = sorted(
        (job for job in jobs if job.get("state") in ACTIVE_JOB_STATES),
        key=_queue_order,
    )
    queued_jobs = sorted(
        (job for job in jobs if job.get("state") == "queued"),
        key=_queue_order,
    )
    blocked_jobs = sorted(
        (job for job in jobs if job.get("state") == "blocked"),
        key=_queue_order,
    )
    attention_jobs = sorted(
        (
            job
            for job in jobs
            if job.get("state") in ATTENTION_STATES
            or (
                job.get("state") in ACTIVE_JOB_STATES
                and isinstance(job.get("error"), str)
                and bool(job["error"])
            )
        ),
        key=_recent_key,
        reverse=True,
    )
    submitted = [job_view(job, current) for job in submitted_jobs]
    active = [job_view(job, current) for job in active_jobs]
    queued = [job_view(job, current) for job in queued_jobs]
    blocked = [job_view(job, current) for job in blocked_jobs]
    attention = [job_view(job, current) for job in attention_jobs]
    recent = sorted(
        (job for job in jobs if job.get("state") in TERMINAL_JOB_STATES),
        key=lambda job: str(job.get("finished_at") or ""),
        reverse=True,
    )
    identity = state.get("queue_id")
    return {
        "v": 1,
        "queue_id": identity,
        "project_id": selected_project,
        "as_of_cursor": state_cursor(state),
        "allocation": state.get("allocation"),
        "updated_at": state.get("updated_at"),
        "draining": bool(state.get("draining", False)),
        "launches_paused": bool(state.get("launches_paused", False)),
        "counts": dict(sorted(counts.items())),
        "archived_jobs": sum(int(count) for count in archived_counts.values()),
        "nodes": state.get("nodes", {}),
        "submitted": submitted[:limit],
        "active": active[:limit],
        "queued": queued[:limit],
        "blocked": blocked[:limit],
        "requires_attention": attention[:limit],
        "recent_terminal": [job_view(job, current) for job in recent[:limit]],
        "truncated": {
            "submitted": len(submitted) > limit,
            "active": len(active) > limit,
            "queued": len(queued) > limit,
            "blocked": len(blocked) > limit,
            "requires_attention": len(attention) > limit,
            "recent_terminal": len(recent) > limit,
        },
    }


def explain_job(state: dict[str, Any], job_id: str) -> dict[str, Any]:
    """Explain one job and the current state of each declared dependency."""

    jobs = state.get("jobs", {})
    job = jobs.get(job_id)
    if job is None:
        raise KeyError(f"unknown job {job_id}")
    by_task = select_task_attempts(jobs.values())
    dependencies = []
    for need in job.get("needs") or []:
        if not isinstance(need, dict):
            continue
        upstream = by_task.get(
            (job_project(job), job.get("workflow_id"), need.get("task_id"))
        )
        dependencies.append(
            {
                **need,
                "job_id": upstream.get("id") if upstream else None,
                "state": upstream.get("state") if upstream else "missing",
                "reason": upstream.get("reason") if upstream else "missing_dependency",
            }
        )
    return {
        "v": 1,
        "job": job,
        "dependencies": dependencies,
        "blockers": list(job.get("blockers") or []),
        "explanation": job.get("reason") or job.get("state"),
    }
