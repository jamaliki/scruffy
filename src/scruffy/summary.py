"""Small, pure views of queue state intended for humans and agents."""

from __future__ import annotations

from collections import Counter
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
        "runtime_placement_contract": job.get("runtime_placement_contract"),
        "runtime_placement_files": list(job.get("runtime_placement_files") or []),
        "runtime_placements": list(job.get("runtime_placements") or []),
        "runtime_placement_error": job.get("runtime_placement_error"),
        "runtime_placement_status": job.get("runtime_placement_status"),
        "workload": workload,
        "progress_age_seconds": progress_age,
        "stdout": job.get("stdout"),
        "stderr": job.get("stderr"),
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
    cursor = (
        f"{identity}:{state.get('journal_generation', 0)}:"
        f"{state.get('last_seq', 0)}:{state.get('journal_offset', 0)}"
    )
    return {
        "v": 1,
        "queue_id": identity,
        "project_id": selected_project,
        "as_of_cursor": cursor,
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
