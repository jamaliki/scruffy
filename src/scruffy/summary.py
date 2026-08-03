"""Small, pure views of queue state intended for humans and agents."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .models import ACTIVE_JOB_STATES, TERMINAL_JOB_STATES
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


def _job_view(job: dict[str, Any], now: datetime) -> dict[str, Any]:
    workload = job.get("workload") if isinstance(job.get("workload"), dict) else None
    updated = _parse_time(workload.get("last_update_at")) if workload else None
    progress_age = max(0.0, (now - updated).total_seconds()) if updated else None
    return {
        "id": job["id"],
        "name": job.get("name"),
        "state": job.get("state"),
        "reason": job.get("reason"),
        "error": job.get("error"),
        "exit_code": job.get("exit_code"),
        "signal": job.get("signal"),
        "submitted_at": job.get("submitted_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
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


def _queue_order(job: dict[str, Any]) -> int:
    value = job.get("queue_order")
    return value if type(value) is int else 0


def _recent_key(job: dict[str, Any]) -> str:
    workload = job.get("workload")
    workload_at = workload.get("last_update_at") if isinstance(workload, dict) else None
    return str(job.get("finished_at") or workload_at or job.get("submitted_at") or "")


def build_summary(
    state: dict[str, Any], *, now: datetime | None = None, limit: int = 20
) -> dict[str, Any]:
    """Return a bounded, action-oriented view without mutating queue state."""

    if limit <= 0:
        raise ValueError("summary limit must be positive")
    current = now or datetime.now(timezone.utc)
    jobs = list(state.get("jobs", {}).values())
    counts = Counter(str(job.get("state", "unknown")) for job in jobs)
    counts.update(
        {
            str(name): int(count)
            for name, count in state.get("archived_counts", {}).items()
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
    submitted = [_job_view(job, current) for job in submitted_jobs]
    active = [_job_view(job, current) for job in active_jobs]
    queued = [_job_view(job, current) for job in queued_jobs]
    blocked = [_job_view(job, current) for job in blocked_jobs]
    attention = [_job_view(job, current) for job in attention_jobs]
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
        "as_of_cursor": cursor,
        "allocation": state.get("allocation"),
        "updated_at": state.get("updated_at"),
        "draining": bool(state.get("draining", False)),
        "counts": dict(sorted(counts.items())),
        "archived_jobs": int(state.get("archived_jobs", 0)),
        "nodes": state.get("nodes", {}),
        "submitted": submitted[:limit],
        "active": active[:limit],
        "queued": queued[:limit],
        "blocked": blocked[:limit],
        "requires_attention": attention[:limit],
        "recent_terminal": [_job_view(job, current) for job in recent[:limit]],
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
        upstream = by_task.get((job.get("workflow_id"), need.get("task_id")))
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
