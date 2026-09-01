"""Small, pure views of queue state intended for humans and agents."""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Collection
from datetime import datetime
from typing import Any

from ._compat import UTC
from .models import (
    ACTIVE_JOB_STATES,
    DEFAULT_PROJECT,
    TERMINAL_JOB_STATES,
    ResourceRequest,
    job_project,
    normalize_project_id,
)
from .scheduler import project_gpu_usage, queue_priority_key
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
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def job_view(job: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """Return the bounded job projection shared by summaries and observers."""

    current = now or datetime.now(UTC)
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
        "deadline_at": job.get("deadline_at"),
        "elapsed_seconds": elapsed,
        "request_id": job.get("request_id"),
        "request": job.get("request"),
        "workflow_id": job.get("workflow_id"),
        "task_id": job.get("task_id"),
        "attempt": job.get("attempt"),
        "recovery": copy.deepcopy(job.get("recovery")),
        "predecessor_job_id": job.get("predecessor_job_id"),
        "successor_job_id": job.get("successor_job_id"),
        "retry_reason": job.get("retry_reason"),
        "retry_exhausted": bool(job.get("retry_exhausted", False)),
        "retry_exhausted_reason": job.get("retry_exhausted_reason"),
        "needs": list(job.get("needs") or []),
        "wait_for": list(job.get("wait_for") or []),
        "condition_satisfactions": list(job.get("condition_satisfactions") or []),
        "blockers": list(job.get("blockers") or []),
        "assignment": job.get("assignment"),
        "placement": job.get("assignment") or job.get("last_assignment"),
        "provenance": job.get("provenance"),
        "resolved_dependencies": list(job.get("resolved_dependencies") or []),
        "resolved_conditions": list(job.get("resolved_conditions") or []),
        "allocation_incarnation_sha256": job.get(
            "allocation_incarnation_sha256"
        ),
        "evacuation": copy.deepcopy(job.get("evacuation")),
        "gpu_binding": job.get("gpu_binding"),
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
    filters: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return one stable page of compact job identities."""

    selected_states = set(states) if states is not None else None
    selected_filters = filters or {}
    for field in ("submitted_after", "submitted_before"):
        if field in selected_filters and _parse_time(selected_filters[field]) is None:
            raise ValueError(f"{field} must be an ISO 8601 timestamp")

    def matches(job: dict[str, Any]) -> bool:
        for field in ("workflow_id", "task_id", "request_id"):
            if field in selected_filters and job.get(field) != selected_filters[field]:
                return False
        prefix = selected_filters.get("name_prefix")
        if prefix is not None and not str(job.get("name") or "").startswith(prefix):
            return False
        submitted = _parse_time(job.get("submitted_at"))
        after = _parse_time(selected_filters.get("submitted_after"))
        before = _parse_time(selected_filters.get("submitted_before"))
        return not (
            (after is not None and (submitted is None or submitted < after))
            or (before is not None and (submitted is None or submitted > before))
        )

    jobs = [
        job
        for job in state.get("jobs", {}).values()
        if isinstance(job, dict)
        and (selected_states is None or job.get("state") in selected_states)
        and (project_id is None or job_project(job) == project_id)
        and matches(job)
    ]
    if selected_states == set(QUEUE_VIEW_STATES):
        usage = project_gpu_usage(state.get("jobs", {}).values())
        jobs.sort(key=lambda job: queue_priority_key(job, usage))
    elif selected_states == set(RUNNING_VIEW_STATES):
        jobs.sort(key=_running_recency_key, reverse=True)
    elif selected_states == set(BLOCKED_VIEW_STATES):
        jobs.sort(key=_accepted_recency_key, reverse=True)
    else:
        jobs.sort(key=lambda job: (_queue_order(job), str(job.get("id") or "")))
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


def scheduler_explanation(state: dict[str, Any]) -> dict[str, Any]:
    """Explain idle capacity using deterministic admission facts, not an ETA."""

    jobs = [job for job in state.get("jobs", {}).values() if isinstance(job, dict)]
    free_nodes = [
        node.get("free", {})
        for node in state.get("nodes", {}).values()
        if isinstance(node, dict)
    ]

    def fits(job: dict[str, Any]) -> bool:
        try:
            request = ResourceRequest.from_dict(job["request"])
        except (KeyError, ValueError):
            return False
        eligible = sum(
            len(node.get("gpu_ids", [])) >= request.gpus_per_node
            and int(node.get("cpus", 0) or 0) >= request.cpus_per_node
            and int(node.get("memory_gb", 0) or 0) >= request.memory_gb_per_node
            for node in free_nodes
        )
        return eligible >= request.nodes

    queued = [job for job in jobs if job.get("state") == "queued"]
    eligible = sum(fits(job) for job in queued)
    active_gpus: Counter[str] = Counter()
    for job in jobs:
        if job.get("state") not in ACTIVE_JOB_STATES:
            continue
        assignment = job.get("assignment")
        if not isinstance(assignment, dict):
            continue
        active_gpus[str(job["state"])] += sum(
            len(item.get("gpu_ids", []))
            for item in assignment.get("reservations", [])
            if isinstance(item, dict)
        )
    reason = None
    if state.get("draining"):
        reason = "allocation_draining"
    elif state.get("launches_paused"):
        reason = "launches_paused"
    elif not queued:
        reason = "no_queued_jobs"
    elif not eligible:
        reason = "no_resource_eligible_jobs"
    return {
        "submitted": sum(job.get("state") == "submitted" for job in jobs),
        "queued": len(queued),
        "resource_eligible": eligible,
        "dependency_blocked": sum(job.get("state") == "blocked" for job in jobs),
        "gpus_by_active_state": dict(sorted(active_gpus.items())),
        "idle_reason": reason,
    }


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
                    "unavailable_gpu_ids": list(node.get("unavailable_gpu_ids", [])),
                    "gpu_devices": list(node.get("gpu_devices", [])),
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
            "controller_release": allocation.get("controller_release", "unknown"),
        },
        "totals": resource_totals(nodes),
        "scheduler": scheduler_explanation(state),
        "gpu_health": state.get("gpu_health"),
        "nodes": rows,
    }


def gpu_view(state: dict[str, Any]) -> dict[str, Any]:
    """Return every managed GPU with identity, ownership, and scheduler state."""

    rows: list[dict[str, Any]] = []
    nodes = state.get("nodes")
    if isinstance(nodes, dict):
        for node_name, node in sorted(nodes.items(), key=lambda item: _node_sort_key(item[0])):
            if not isinstance(node, dict):
                continue
            free = set(node.get("free", {}).get("gpu_ids", []))
            unavailable = set(node.get("unavailable_gpu_ids", []))
            owners = {
                gpu_id: job_id
                for job_id, reservation in node.get("assignments", {}).items()
                if isinstance(reservation, dict)
                for gpu_id in reservation.get("gpu_ids", [])
            }
            devices = node.get("gpu_devices", [])
            sampled_devices = devices if isinstance(devices, list) else []
            by_slot = {
                device.get("slot"): device
                for device in sampled_devices
                if isinstance(device, dict) and type(device.get("slot")) is int
            }
            capacity = node.get("capacity", {})
            gpu_ids = capacity.get("gpu_ids", []) if isinstance(capacity, dict) else []
            for slot in gpu_ids if isinstance(gpu_ids, list) else []:
                if type(slot) is not int:
                    continue
                device = copy.deepcopy(by_slot.get(slot, {"slot": slot, "status": "unknown"}))
                owner = owners.get(slot)
                status = device.get("status")
                if owner:
                    scheduler_state = "assigned"
                elif slot in unavailable and status == "quarantined":
                    scheduler_state = "stopped"
                elif slot in unavailable and status == "healthy":
                    scheduler_state = "node_held"
                elif slot in unavailable:
                    scheduler_state = "health_unknown"
                elif status == "quarantined":
                    scheduler_state = "quarantined_observed"
                elif slot in free:
                    scheduler_state = "free"
                else:
                    scheduler_state = "unavailable"
                rows.append(
                    {
                        **device,
                        "node": node_name,
                        "assigned_job_id": owner,
                        "scheduler_state": scheduler_state,
                    }
                )
    return {
        "v": 1,
        "queue_id": state.get("queue_id"),
        "project_id": state.get("project_id"),
        "as_of_cursor": state_cursor(state),
        "policy": {
            "mode": state.get("gpu_health", {}).get("mode"),
            "isolation": state.get("gpu_health", {}).get("isolation"),
        }
        if isinstance(state.get("gpu_health"), dict)
        else {},
        "gpus": rows,
    }


def inspect_gpu(state: dict[str, Any], node: str, slot: int) -> dict[str, Any]:
    """Return one GPU's complete reportable identity and scheduler state."""

    view = gpu_view(state)
    for device in view["gpus"]:
        if device.get("node") == node and device.get("slot") == slot:
            health = state.get("gpu_health")
            health_node = health.get("nodes", {}).get(node, {}) if isinstance(health, dict) else {}
            return {
                **device,
                "policy": view["policy"],
                "cuda_probe": health_node.get("cuda_probe"),
                "last_received_at": health_node.get("last_received_at"),
                "monitor": health.get("monitor") if isinstance(health, dict) else None,
                "as_of_cursor": view["as_of_cursor"],
            }
    raise KeyError(f"unknown GPU slot {slot} on node {node!r}")


def _queue_order(job: dict[str, Any]) -> int:
    value = job.get("queue_order")
    return value if type(value) is int else 0


def _accepted_recency_key(job: dict[str, Any]) -> tuple[int, str]:
    return _queue_order(job), str(job.get("id") or "")


def _running_recency_key(job: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(job.get("started_at") or ""),
        _queue_order(job),
        str(job.get("id") or ""),
    )


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
    current = now or datetime.now(UTC)
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
        key=_running_recency_key,
        reverse=True,
    )
    usage = project_gpu_usage(jobs)
    queued_jobs = sorted(
        (job for job in jobs if job.get("state") == "queued"),
        key=lambda job: queue_priority_key(job, usage),
    )
    blocked_jobs = sorted(
        (job for job in jobs if job.get("state") == "blocked"),
        key=_accepted_recency_key,
        reverse=True,
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
    allocation = dict(state.get("allocation") or {})
    release = allocation.get("controller_release")
    allocation["controller_release"] = (
        release.strip()
        if isinstance(release, str) and release.strip()
        else "unknown"
    )
    allocation_deadline = _parse_time(allocation.get("deadline_at"))
    if allocation_deadline is not None:
        allocation["remaining_seconds"] = max(
            0, int((allocation_deadline - current).total_seconds())
        )
    return {
        "v": 1,
        "queue_id": identity,
        "project_id": selected_project,
        "as_of_cursor": state_cursor(state),
        "allocation": allocation,
        "evacuation": copy.deepcopy(state.get("evacuation")),
        "evacuation_history": copy.deepcopy(state.get("evacuation_history", {})),
        "updated_at": state.get("updated_at"),
        "draining": bool(state.get("draining", False)),
        "launches_paused": bool(state.get("launches_paused", False)),
        "gpu_health": copy.deepcopy(state.get("gpu_health")),
        "counts": dict(sorted(counts.items())),
        "scheduler": scheduler_explanation({**state, "jobs": {job["id"]: job for job in jobs}}),
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
    satisfactions = {
        (item.get("task_id"), item.get("artifact_id")): item
        for item in job.get("condition_satisfactions") or []
        if isinstance(item, dict)
    }
    conditions = []
    for condition in job.get("wait_for") or []:
        if not isinstance(condition, dict):
            continue
        identity = (condition.get("task_id"), condition.get("artifact_id"))
        upstream = by_task.get(
            (job_project(job), job.get("workflow_id"), condition.get("task_id"))
        )
        conditions.append(
            {
                **condition,
                "job_id": upstream.get("id") if upstream else None,
                "state": upstream.get("state") if upstream else "missing",
                "satisfied": identity in satisfactions,
                "evidence": copy.deepcopy(satisfactions.get(identity)),
            }
        )
    return {
        "v": 1,
        "job": job,
        "evacuation": copy.deepcopy(
            (state.get("evacuation") or {}).get("targets", {}).get(job_id)
            if isinstance(state.get("evacuation"), dict)
            else None
        ),
        "dependencies": dependencies,
        "conditions": conditions,
        "blockers": list(job.get("blockers") or []),
        "explanation": job.get("reason") or job.get("state"),
    }
