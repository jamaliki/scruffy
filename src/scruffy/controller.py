"""The single-writer event loop for one Scruffy queue."""

from __future__ import annotations

import copy
import queue
import signal
import subprocess
import sys
import time
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ._compat import UTC
from .health import (
    GPU_ISOLATION_MODES,
    HEALTH_MODES,
    HealthError,
    bind_health_incarnation,
    ensure_health_state,
    ingest_health_sample,
    reprobe_quarantine,
    set_quarantine,
)
from .lifecycle import (
    begin_shutdown,
    drain_messages,
    poll_processes,
    remaining_time_limit,
    request_cancellation,
    schedule,
)
from .models import (
    ACTIVE_JOB_STATES,
    DEFAULT_PROJECT,
    TERMINAL_JOB_STATES,
    Assignment,
    NodeInventory,
    ResourceRequest,
    job_project,
    normalize_project_id,
    validate_inventory,
)
from .protocol import artifact_publication, validate_event
from .provenance import write_request_record, write_result_record
from .runtime import (
    Controller,
    OutputNotifier,
    RunningProcess,
    abandon_processes,
    signal_process,
)
from .scheduler import InvariantError, assert_invariants, request_can_ever_fit
from .slurm import (
    AllocationIncarnation,
    SlurmStep,
    allocation_metadata,
    build_health_srun_argv,
    build_srun_environment,
    cancel_step,
)
from .state import (
    apply_workload_event,
    commit_snapshot,
    compact_journal,
    emit,
    emit_submission,
    load_recovered_state,
)
from .storage import (
    StorageError,
    TransientStorageError,
    UnsafeRecovery,
    accept_known_requests,
    accept_reports,
    accept_request,
    accept_submission,
    compact_report_receipts,
    controller_lock,
    create_job_id,
    ensure_layout,
    find_archived_job,
    job_identity_digest,
    list_archived_workflow,
    list_commands,
    list_reports,
    list_requests,
    list_submissions,
    open_journal,
    read_events,
    read_json,
    record_request_receipt,
    recovery_request_id,
    reject_request,
    remove_cold_job_directories,
    remove_command,
    report_identity_digest,
    report_streams,
    report_was_accepted,
    request_pending,
    request_receipt_digest,
    submission_identity_digest,
    utc_now,
)
from .submissions import job_from_spec
from .workflows import (
    AUTO_RECOVERY_REASONS,
    WorkflowError,
    artifact_conditions,
    resolve_blocked_jobs,
    resolve_dependencies,
    select_task_attempts,
    validate_workflows,
)

MAX_REPORTS_PER_TICK = 128
STORAGE_RETRY_SECONDS = 5
COMMAND_OUTCOME_KINDS = {
    "job.cancelled",
    "job.cancelling",
    "job.cancel_ignored",
    "command.rejected",
    "allocation.draining",
    "allocation.drain_ignored",
    "allocation.launches_resumed",
    "allocation.resume_ignored",
    "resource.gpu_health_changed",
}


def _initialize_controller(
    *,
    root: Path,
    inventory: tuple[NodeInventory, ...],
    launcher: str,
    allocation_id: str,
    slurm_job_id: str | None,
    poll_interval: float,
    cancel_grace: float,
    allocation_incarnation: AllocationIncarnation | None = None,
    start_paused: bool = False,
    drain_before_end_seconds: float = 900,
    gpu_health_mode: str = "observe",
    gpu_isolation: str = "gpu",
    gpu_health_interval: float = 10,
) -> Controller:
    state = load_recovered_state(root)
    health = ensure_health_state(state, mode=gpu_health_mode, isolation=gpu_isolation)
    bind_health_incarnation(
        health,
        (allocation_incarnation.fingerprint_sha256 if allocation_incarnation is not None else None),
    )
    active = [job for job in state.get("jobs", {}).values() if job["state"] in ACTIVE_JOB_STATES]
    previous = state.get("allocation") or {}
    if launcher == "slurm" and (
        allocation_incarnation is None or allocation_incarnation.slurm_job_id != slurm_job_id
    ):
        raise ValueError("Slurm allocation incarnation is missing or differs")
    previous_incarnation = None
    raw_previous_incarnation = previous.get("incarnation")
    if raw_previous_incarnation is not None:
        try:
            previous_incarnation = AllocationIncarnation.from_dict(raw_previous_incarnation)
        except (TypeError, ValueError) as exc:
            raise UnsafeRecovery(f"invalid persisted allocation incarnation: {exc}") from exc
        if previous.get("id") != previous_incarnation.slurm_job_id:
            raise UnsafeRecovery("persisted allocation ID differs from its incarnation")
    same_slurm_allocation = (
        launcher == "slurm"
        and previous.get("id") == allocation_id
        and previous_incarnation == allocation_incarnation
    )
    legacy_slurm_allocation = (
        launcher == "slurm"
        and raw_previous_incarnation is None
        and (
            previous.get("id") == allocation_id
            or (previous.get("id") is None and bool(state.get("jobs")))
        )
    )
    previous_allocation_id = previous.get("id")
    replacement = (
        launcher == "slurm"
        and isinstance(previous_allocation_id, str)
        and not same_slurm_allocation
    )
    if active and launcher == "local":
        raise UnsafeRecovery(
            "unresolved active jobs could still be running; refusing unsafe recovery"
        )
    if same_slurm_allocation:
        try:
            assignments = tuple(Assignment.from_dict(job["assignment"]) for job in active)
            assert_invariants(inventory, assignments)
        except (KeyError, ValueError, InvariantError) as exc:
            raise UnsafeRecovery(
                f"invalid active Slurm assignments; refusing recovery: {exc}"
            ) from exc
        if active and any(not job.get("launch_token") for job in active):
            raise UnsafeRecovery("active Slurm job has no launch token; refusing unsafe recovery")
        incarnation_sha256 = allocation_incarnation.fingerprint_sha256
        if active and any(
            job.get("allocation_incarnation_sha256") != incarnation_sha256 for job in active
        ):
            raise UnsafeRecovery(
                "active Slurm job allocation incarnation differs; refusing unsafe recovery"
            )

    journal = open_journal(root, int(state.get("journal_generation", 0)))
    messages: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
    controller = Controller(
        root=root,
        inventory=inventory,
        launcher=launcher,
        allocation_id=allocation_id,
        slurm_job_id=slurm_job_id,
        allocation_incarnation=allocation_incarnation,
        poll_interval=poll_interval,
        cancel_grace=cancel_grace,
        drain_before_end_seconds=drain_before_end_seconds,
        state=state,
        journal=journal,
        messages=messages,
        output=OutputNotifier(messages),
        gpu_health_mode=gpu_health_mode,
        gpu_isolation=gpu_isolation,
        gpu_health_interval=gpu_health_interval,
        health_step_name=(
            f"scruffy-health-{allocation_incarnation.fingerprint_sha256[:12]}"
            if launcher == "slurm"
            and gpu_health_mode != "off"
            and allocation_incarnation is not None
            else None
        ),
    )

    lost_reason = None
    if same_slurm_allocation:
        _reattach_slurm_jobs(controller, active)
    else:
        if legacy_slurm_allocation:
            active_lost_reason = "allocation_incarnation_unavailable"
        elif previous.get("id") == allocation_id:
            active_lost_reason = "allocation_incarnation_changed"
        else:
            active_lost_reason = "allocation_replaced"
        if active:
            lost_reason = active_lost_reason
        # A replaced or restarted Slurm incarnation cannot retain its old
        # steps. Legacy active records are not upgraded to a new incarnation;
        # launches remain paused below until an operator audits the old work.
        for job in active:
            job["state"] = "lost"
            job["finished_at"] = utc_now()
            job["reason"] = active_lost_reason
            _mark_retry_exhaustion(job, active_lost_reason)
            job["last_assignment"] = job.get("assignment")
            job["assignment"] = None
            write_result_record(root, job)
        for job in active:
            emit(controller, "job.lost", job=job, snapshot=False)
        if launcher == "slurm" and active_lost_reason in AUTO_RECOVERY_REASONS:
            _recover_lost_workflow_jobs(controller, active_lost_reason)

    # Queued and later workflow jobs have already crossed their dependency
    # gate. Persist the marker for snapshots created before the field existed.
    for job in state["jobs"].values():
        if isinstance(job.get("workflow_id"), str):
            job.setdefault("dependency_gate_passed", job.get("state") != "blocked")

    now = utc_now()
    metadata = allocation_metadata(allocation_id, launcher)
    if allocation_incarnation is not None:
        metadata["incarnation"] = allocation_incarnation.to_dict()
    metadata.update(
        {
            "state": "running",
            "started_at": (previous.get("started_at", now) if same_slurm_allocation else now),
            "controller_started_at": now,
            "heartbeat_at": now,
        }
    )
    if slurm_job_id:
        metadata["slurm_job_id"] = slurm_job_id
    if same_slurm_allocation and isinstance(previous.get("handover"), dict):
        metadata["handover"] = previous["handover"]
    deadline_at = metadata.get("deadline_at")
    if drain_before_end_seconds and isinstance(deadline_at, str):
        deadline = datetime.fromisoformat(deadline_at)
        metadata["automatic_drain_at"] = (
            (deadline - timedelta(seconds=drain_before_end_seconds))
            .astimezone(UTC)
            .isoformat(timespec="seconds")
        )
    drain_requested = bool(
        state.get(
            "drain_requested",
            state.get("draining") and previous.get("state") == "draining",
        )
    )
    # A drain belongs to one physical allocation incarnation. Slurm can reuse
    # the same job ID after requeue, but none of the old steps survive it.
    preserve_drain = drain_requested and (
        same_slurm_allocation or (launcher == "local" and previous.get("id") == allocation_id)
    )
    if preserve_drain:
        metadata["state"] = "draining"
    state["allocation"] = metadata
    state["draining"] = preserve_drain
    state["drain_requested"] = preserve_drain
    # Recovery owns existing steps but never admits additional work implicitly.
    # An operator must explicitly resume after checking the recovered snapshot.
    state["launches_paused"] = same_slurm_allocation or legacy_slurm_allocation or start_paused
    ineligible = [
        job["id"]
        for job in state["jobs"].values()
        if job["state"] in {"queued", "blocked"}
        and not request_can_ever_fit(inventory, ResourceRequest.from_dict(job["request"]))
    ]
    if replacement:
        metadata["handover"] = {
            "previous_allocation_id": previous_allocation_id,
            "lost_jobs": len(active),
            "queued_jobs": sum(job["state"] == "queued" for job in state["jobs"].values()),
            "blocked_jobs": sum(job["state"] == "blocked" for job in state["jobs"].values()),
            "ineligible_jobs": len(ineligible),
        }
        if previous_incarnation is not None:
            metadata["handover"]["previous_incarnation_sha256"] = (
                previous_incarnation.fingerprint_sha256
            )
    emit(
        controller,
        "allocation.resumed" if same_slurm_allocation else "allocation.started",
        data={
            "nodes": [item.to_dict() for item in inventory],
            "incarnation": (
                allocation_incarnation.to_dict() if allocation_incarnation is not None else None
            ),
            "reattached_jobs": ([job["id"] for job in active] if same_slurm_allocation else []),
            "lost_jobs": ([job["id"] for job in active] if lost_reason is not None else []),
            "lost_reason": lost_reason,
            **({"handover": metadata["handover"]} if replacement else {}),
        },
    )
    if state["launches_paused"]:
        emit(
            controller,
            "allocation.launches_paused",
            data={
                "reason": (
                    "controller_restart"
                    if same_slurm_allocation
                    else (
                        "legacy_incarnation_audit_required"
                        if legacy_slurm_allocation
                        else "operator_requested"
                    )
                )
            },
        )
    return controller


def _reattach_slurm_jobs(controller: Controller, jobs: list[dict[str, Any]]) -> None:
    """Restore ownership of persisted steps without their old local clients."""

    for job in jobs:
        job.pop("pid", None)
        running = RunningProcess(None, str(job["launch_token"]))
        running.closed_streams.update({"stdout", "stderr"})
        for stream_name in ("stdout", "stderr"):
            relative_name = job.get(stream_name, f"jobs/{job['id']}/{stream_name}.log")
            try:
                size = (controller.root / relative_name).stat().st_size
            except FileNotFoundError:
                size = 0
            running.output_offsets[stream_name] = size
        if job["state"] == "cancelling":
            running.final_state = "cancelled"
            running.final_reason = "cancelled"
        remaining = remaining_time_limit(job)
        if remaining is not None:
            running.time_limit_deadline = time.monotonic() + remaining
        controller.running[job["id"]] = running


def _rejected_job(
    spec: dict[str, Any], queue_order: int, job_id: str, exc: Exception
) -> dict[str, Any]:
    try:
        project_id = normalize_project_id(spec.get("project_id"))
    except ValueError:
        project_id = DEFAULT_PROJECT
    job = {
        "id": job_id or f"invalid-{queue_order}",
        "project_id": project_id,
        "name": str(spec.get("name", "invalid")),
        "state": "rejected",
        "submitted_at": str(spec.get("submitted_at", utc_now())),
        "queue_order": queue_order,
        "request_digest": job_identity_digest(spec),
        "request": spec.get("resources"),
        "assignment": None,
        "finished_at": utc_now(),
        "error": str(exc),
        "reason": "invalid_spec",
    }
    workflow_id, task_id = spec.get("workflow_id"), spec.get("task_id")
    if isinstance(workflow_id, str) and isinstance(task_id, str):
        job.update(
            {
                "workflow_id": workflow_id,
                "task_id": task_id,
                "needs": [],
                "wait_for": [],
                "blockers": [],
                "workflow_invalid": True,
            }
        )
    return job


def _mark_workflow_rejected(job: dict[str, Any], exc: Exception) -> None:
    job["workflow_invalid"] = True
    job["needs"] = []
    job["wait_for"] = []
    job["state"] = "rejected"
    job["finished_at"] = utc_now()
    job["reason"] = "invalid_workflow"
    job["error"] = str(exc)


def _recovery_policy(job: dict[str, Any]) -> dict[str, Any] | None:
    policy = job.get("recovery")
    return policy if isinstance(policy, dict) else None


def _mark_retry_exhaustion(job: dict[str, Any], reason: str) -> None:
    """Persist bounded retry exhaustion on the terminal predecessor."""

    policy = _recovery_policy(job)
    attempt = job.get("attempt") if type(job.get("attempt")) is int else 1
    max_attempts = policy.get("max_attempts") if policy else None
    retry_on = policy.get("retry_on") if policy else ()
    if (
        isinstance(max_attempts, int)
        and attempt >= max_attempts
        and isinstance(retry_on, list)
        and reason in retry_on
    ):
        job["retry_exhausted"] = True
        job["retry_exhausted_reason"] = reason
        job["retry_exhausted_at"] = job.get("finished_at")


def _recovery_successor_spec(
    predecessor: dict[str, Any], successor_id: str, request_id: str
) -> dict[str, Any]:
    """Copy only immutable task inputs into one deterministic retry spec."""

    spec = {
        "v": 1,
        "job_id": successor_id,
        "request_id": request_id,
        "name": predecessor.get("name", predecessor["task_id"]),
        # The predecessor loss event is durable before recovery admission, so
        # it provides a stable timestamp across a crash/replay window.
        "submitted_at": predecessor.get("finished_at")
        or predecessor.get("submitted_at")
        or utc_now(),
        "argv": copy.deepcopy(predecessor["argv"]),
        "cwd": predecessor["cwd"],
        "env": copy.deepcopy(predecessor.get("env", {})),
        "resources": copy.deepcopy(predecessor["request"]),
        "workflow_id": predecessor["workflow_id"],
        "task_id": predecessor["task_id"],
        "needs": copy.deepcopy(predecessor.get("needs", [])),
        "wait_for": copy.deepcopy(predecessor.get("wait_for", [])),
        "recovery": copy.deepcopy(predecessor["recovery"]),
    }
    project_id = job_project(predecessor)
    if project_id != DEFAULT_PROJECT:
        spec["project_id"] = project_id
    return spec


def _recovery_candidate(
    predecessor: dict[str, Any], reason: str
) -> tuple[int, str, str] | None:
    workflow_id = predecessor.get("workflow_id")
    task_id = predecessor.get("task_id")
    policy = _recovery_policy(predecessor)
    if (
        predecessor.get("state") != "lost"
        or not isinstance(workflow_id, str)
        or not isinstance(task_id, str)
        or policy is None
        or reason not in policy.get("retry_on", [])
        or not all(key in predecessor for key in ("argv", "cwd", "request"))
    ):
        return None
    attempt = predecessor.get("attempt") if type(predecessor.get("attempt")) is int else 1
    max_attempts = policy.get("max_attempts")
    if type(max_attempts) is not int or attempt >= max_attempts:
        return None
    successor_attempt = attempt + 1
    project_id = job_project(predecessor)
    request_id = recovery_request_id(project_id, workflow_id, task_id, successor_attempt)
    return successor_attempt, request_id, create_job_id(request_id, project_id=project_id)


def _link_recovery_predecessor(
    controller: Controller,
    predecessor: dict[str, Any],
    successor_id: str,
    reason: str,
) -> None:
    """Repair and durably publish a predecessor's reverse lineage pointer."""

    if predecessor.get("successor_job_id") == successor_id:
        return
    predecessor["successor_job_id"] = successor_id
    emit(
        controller,
        "job.recovery_linked",
        job=predecessor,
        data={"successor_job_id": successor_id, "retry_reason": reason},
    )


def _admit_recovery_successor(
    controller: Controller,
    predecessor: dict[str, Any],
    successor_id: str,
    request_id: str,
    attempt: int,
    queue_order: int,
    prospective: dict[str, dict[str, Any]],
    reason: str,
) -> None:
    spec = _recovery_successor_spec(predecessor, successor_id, request_id)
    successor = job_from_spec(spec, queue_order)
    successor.update(
        {
            "attempt": attempt,
            "predecessor_job_id": predecessor["id"],
            "retry_reason": reason,
        }
    )
    if predecessor.get("dependency_gate_passed") is True:
        for key in (
            "dependency_gate_passed",
            "condition_satisfactions",
            "resolved_dependencies",
            "resolved_conditions",
        ):
            if key in predecessor:
                successor[key] = copy.deepcopy(predecessor[key])
    prospective[successor_id] = successor
    event_kind = _initial_job_event(controller, successor, prospective)
    write_request_record(controller.root, successor)
    if successor.get("state") in TERMINAL_JOB_STATES:
        write_result_record(controller.root, successor)
    controller.state["jobs"][successor_id] = successor
    emit(
        controller,
        event_kind,
        job=successor,
        data={"predecessor_job_id": predecessor["id"], "retry_reason": reason},
    )
    _link_recovery_predecessor(controller, predecessor, successor_id, reason)
    record_request_receipt(controller.root, successor_id, job_identity_digest(spec))


def _recover_lost_workflow_jobs(controller: Controller, reason: str) -> None:
    """Admit at most one deterministic successor for each eligible lost task."""

    jobs = controller.state["jobs"]
    prospective = dict(jobs)
    next_order = max((int(job.get("queue_order", 0)) for job in jobs.values()), default=0)
    for predecessor in sorted(
        jobs.values(), key=lambda job: (int(job.get("queue_order", 0)), str(job["id"]))
    ):
        candidate = _recovery_candidate(predecessor, reason)
        if candidate is None:
            continue
        attempt, request_id, successor_id = candidate
        existing = jobs.get(successor_id)
        if existing is None:
            try:
                existing = find_archived_job(controller.root, successor_id)
            except (OSError, StorageError, TransientStorageError) as exc:
                _storage_notice(controller, "read_recovery_receipt", successor_id, exc)
                continue
        if existing is not None or predecessor.get("successor_job_id") == successor_id:
            _link_recovery_predecessor(controller, predecessor, successor_id, reason)
            continue
        _admit_recovery_successor(
            controller,
            predecessor,
            successor_id,
            request_id,
            attempt,
            next_order + 1,
            prospective,
            reason,
        )
        next_order += 1
    controller.state["next_queue_order"] = max(
        int(controller.state.get("next_queue_order", 0)), next_order
    )


def _resolution_workflow_jobs(
    jobs: Iterable[dict[str, Any]], project_id: str, workflow_id: str
) -> list[dict[str, Any]]:
    """Select one workflow, removing invalid edges from rejected task records."""

    selected = select_task_attempts(jobs)
    resolution_jobs = []
    for (candidate_project, candidate_workflow, _), candidate in selected.items():
        if candidate_project != project_id or candidate_workflow != workflow_id:
            continue
        if candidate.get("workflow_invalid"):
            candidate = {**candidate, "needs": [], "wait_for": []}
        resolution_jobs.append(candidate)
    return resolution_jobs


def _storage_notice(controller: Controller, operation: str, item: str, exc: Exception) -> None:
    """Publish one contained storage failure without stopping the controller."""

    emit(
        controller,
        "notice",
        data={
            "kind": "storage.item_skipped",
            "operation": operation,
            "item": item,
            "error": str(exc),
        },
    )


def _archived_workflow_jobs(
    controller: Controller, project_id: str, workflow_id: str
) -> list[dict[str, Any]] | None:
    """Load a workflow archive while surfacing only the bad entries."""

    try:
        return list_archived_workflow(
            controller.root,
            workflow_id,
            project_id=project_id,
            on_error=lambda source, exc: _storage_notice(
                controller, "read_workflow_archive", source.name, exc
            ),
        )
    except TransientStorageError as exc:
        _storage_notice(controller, "read_workflow_archive", f"{project_id}/{workflow_id}", exc)
        return None


def _stage_job(
    spec: dict[str, Any],
    queue_order: int,
    prospective: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build one non-public admission decision against earlier requests."""

    job_id = str(spec.get("job_id", ""))
    try:
        job = job_from_spec(spec, queue_order)
    except Exception as exc:
        return _rejected_job(spec, queue_order, job_id, exc)

    try:
        # Missing upstream tasks are valid during asynchronous submission; this
        # first pass checks only the task's own shape and self-dependencies.
        validate_workflows([job])
    except WorkflowError as exc:
        _mark_workflow_rejected(job, exc)
        return job

    workflow_id = job.get("workflow_id")
    task_id = job.get("task_id")
    project_id = job_project(job)
    duplicate = (
        select_task_attempts(prospective.values()).get((project_id, workflow_id, task_id))
        if isinstance(workflow_id, str) and isinstance(task_id, str)
        else None
    )
    duplicate_state = duplicate.get("state") if duplicate is not None else None
    if (
        duplicate is not None
        and not duplicate.get("workflow_invalid")
        and (duplicate_state not in TERMINAL_JOB_STATES or duplicate_state == "succeeded")
    ):
        _mark_workflow_rejected(
            job,
            WorkflowError(f"duplicate task_id {job.get('task_id')!r} in workflow {workflow_id!r}"),
        )
        return job

    if isinstance(workflow_id, str) and isinstance(task_id, str):
        prior_attempts = [
            (candidate["attempt"] if type(candidate.get("attempt")) is int else 1)
            for candidate in prospective.values()
            if job_project(candidate) == project_id
            and candidate.get("workflow_id") == workflow_id
            and candidate.get("task_id") == task_id
        ]
        job["attempt"] = (
            max(
                prior_attempts,
                default=0,
            )
            + 1
        )

    if workflow_id is not None:
        candidates = [
            candidate
            for candidate in _resolution_workflow_jobs(
                prospective.values(), project_id, workflow_id
            )
            if candidate.get("task_id") != task_id
        ]
        try:
            validate_workflows([*candidates, job])
        except WorkflowError as exc:
            # Reject the request which closes a cycle, not the whole workflow.
            _mark_workflow_rejected(job, exc)
    return job


def _resolved_dependency_ids(
    job: dict[str, Any], workflow_jobs: Iterable[dict[str, Any]]
) -> list[dict[str, str]]:
    """Bind logical dependencies to the concrete attempts used at launch."""

    selected = select_task_attempts(workflow_jobs)
    project_id = job_project(job)
    workflow_id = str(job.get("workflow_id") or "")
    result = []
    for need in job.get("needs") or []:
        dependency = selected.get((project_id, workflow_id, str(need.get("task_id") or "")))
        if dependency is not None:
            result.append(
                {
                    "task_id": str(need["task_id"]),
                    "job_id": str(dependency["id"]),
                    "condition": str(need["condition"]),
                }
            )
    return result


def _resolved_condition_evidence(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Return satisfied evidence in the immutable declaration order."""

    evidence = {
        (item.get("task_id"), item.get("artifact_id")): item
        for item in job.get("condition_satisfactions") or []
        if isinstance(item, dict)
    }
    return [
        copy.deepcopy(evidence[(task_id, artifact_id)])
        for task_id, artifact_id in artifact_conditions(job)
        if (task_id, artifact_id) in evidence
    ]


def _condition_evidence(
    *,
    producer: dict[str, Any],
    publication: dict[str, Any],
    producer_event_id: str,
    occurred_at: str,
    queue_event_id: str | None = None,
) -> dict[str, Any]:
    result = {
        "kind": "artifact",
        "task_id": producer["task_id"],
        "artifact_id": publication["artifact_id"],
        "producer_job_id": producer["id"],
        "producer_event_id": producer_event_id,
        "occurred_at": occurred_at,
        "publication": copy.deepcopy(publication),
    }
    if queue_event_id is not None:
        result["queue_event_id"] = queue_event_id
    return result


def _remember_condition(job: dict[str, Any], evidence: dict[str, Any]) -> bool:
    """Attach one immutable condition result, returning whether it was new."""

    identity = (evidence["task_id"], evidence["artifact_id"])
    satisfactions = job.setdefault("condition_satisfactions", [])
    if any(
        isinstance(item, dict) and (item.get("task_id"), item.get("artifact_id")) == identity
        for item in satisfactions
    ):
        return False
    satisfactions.append(copy.deepcopy(evidence))
    return True


def _artifact_attempts(
    workflow_jobs: Iterable[dict[str, Any]],
    project_id: str,
    workflow_id: str,
    task_id: str,
) -> list[dict[str, Any]]:
    """Return every valid logical producer attempt, newest evidence first."""

    attempts = [
        candidate
        for candidate in workflow_jobs
        if (
            not candidate.get("workflow_invalid")
            and job_project(candidate) == project_id
            and candidate.get("workflow_id") == workflow_id
            and candidate.get("task_id") == task_id
        )
    ]
    return sorted(
        attempts,
        key=lambda candidate: (
            int(candidate.get("attempt", 1))
            if type(candidate.get("attempt")) is int
            else 1,
            int(candidate.get("queue_order", 0))
            if type(candidate.get("queue_order")) is int
            else 0,
            str(candidate.get("id", "")),
        ),
        reverse=True,
    )


def _artifact_evidence(producer: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield durable and legacy artifact projections for one producer."""

    for item in reversed(list(producer.get("artifact_evidence") or [])):
        if not isinstance(item, dict):
            continue
        publication = item.get("publication")
        if not isinstance(publication, dict):
            continue
        try:
            publication = artifact_publication({"publication": publication})
        except ValueError:
            continue
        if publication is None:
            continue
        yield {
            "publication": publication,
            "producer_event_id": item.get("producer_event_id"),
            "occurred_at": item.get("occurred_at"),
        }
    workload = producer.get("workload")
    artifacts = workload.get("latest_artifacts") if isinstance(workload, dict) else None
    for item in reversed(artifacts if isinstance(artifacts, list) else []):
        if not isinstance(item, dict):
            continue
        try:
            publication = artifact_publication(item)
        except ValueError:
            continue
        if publication is not None:
            yield {
                "publication": publication,
                "producer_event_id": item.get("event_id"),
                "occurred_at": item.get("occurred_at"),
            }


def _satisfy_from_current_artifacts(
    job: dict[str, Any], workflow_jobs: Iterable[dict[str, Any]]
) -> None:
    """Resolve a newly admitted wait from any valid producer attempt."""

    workflow_jobs = list(workflow_jobs)
    project_id = job_project(job)
    workflow_id = str(job.get("workflow_id") or "")
    for task_id, artifact_id in artifact_conditions(job):
        for producer in _artifact_attempts(workflow_jobs, project_id, workflow_id, task_id):
            for item in _artifact_evidence(producer):
                publication = item["publication"]
                if publication.get("artifact_id") != artifact_id:
                    continue
                producer_event_id = item.get("producer_event_id")
                occurred_at = item.get("occurred_at")
                if not isinstance(producer_event_id, str) or not isinstance(occurred_at, str):
                    continue
                _remember_condition(
                    job,
                    _condition_evidence(
                        producer=producer,
                        publication=publication,
                        producer_event_id=producer_event_id,
                        occurred_at=occurred_at,
                    ),
                )
                break
            else:
                continue
            break


def _satisfy_artifact_waiters(
    controller: Controller,
    producer: dict[str, Any],
    event: dict[str, Any],
    queue_event_id: str,
) -> None:
    """Apply one typed publication only to explicitly waiting workflow jobs."""

    publication = artifact_publication(event["data"])
    workflow_id = producer.get("workflow_id")
    task_id = producer.get("task_id")
    if publication is None or not isinstance(workflow_id, str) or not isinstance(task_id, str):
        return
    project_id = job_project(producer)
    if (
        producer.get("workflow_invalid")
        or producer.get("workflow_id") != workflow_id
        or producer.get("task_id") != task_id
        or job_project(producer) != project_id
    ):
        return
    for job in controller.state["jobs"].values():
        if (
            job.get("state") != "blocked"
            or job.get("workflow_invalid")
            or job_project(job) != project_id
            or job.get("workflow_id") != workflow_id
            or (task_id, publication["artifact_id"]) not in artifact_conditions(job)
        ):
            continue
        evidence = _condition_evidence(
            producer=producer,
            publication=publication,
            producer_event_id=event["event_id"],
            occurred_at=event["occurred_at"],
            queue_event_id=queue_event_id,
        )
        if _remember_condition(job, evidence):
            emit(
                controller,
                "condition.satisfied",
                job=job,
                data=evidence,
                durable=False,
                snapshot=False,
            )


def _initial_job_event(
    controller: Controller,
    job: dict[str, Any],
    prospective: dict[str, dict[str, Any]],
) -> str:
    """Resolve one staged job's initial state without publishing it."""

    if job.get("state") == "rejected":
        return "job.rejected"
    request = ResourceRequest.from_dict(job["request"])
    if not request_can_ever_fit(controller.inventory, request):
        job["state"] = "rejected"
        job["finished_at"] = utc_now()
        job["reason"] = "request_cannot_fit"
        job["error"] = "request cannot fit this allocation inventory"
        return "job.rejected"
    if job.get("workflow_id") is None:
        return "job.queued"

    all_workflow_jobs = [
        candidate
        for candidate in prospective.values()
        if (
            job_project(candidate) == job_project(job)
            and candidate.get("workflow_id") == job["workflow_id"]
        )
    ]
    _satisfy_from_current_artifacts(job, all_workflow_jobs)
    workflow_jobs = _resolution_workflow_jobs(
        all_workflow_jobs, job_project(job), job["workflow_id"]
    )
    resolution = resolve_dependencies(job, workflow_jobs)
    job["blockers"] = resolution["blockers"]
    if resolution["decision"] == "ready":
        job["dependency_gate_passed"] = True
        job["resolved_dependencies"] = _resolved_dependency_ids(job, workflow_jobs)
        job["resolved_conditions"] = _resolved_condition_evidence(job)
        return "job.queued"
    elif resolution["decision"] == "blocked":
        job["dependency_gate_passed"] = False
        job["state"] = "blocked"
        job["reason"] = "waiting_for_dependencies"
        return "job.blocked"
    else:
        job["dependency_gate_passed"] = True
        job["state"] = "skipped"
        job["finished_at"] = utc_now()
        job["reason"] = "dependency_unsatisfied"
        return "job.skipped"


def _admit_job(
    controller: Controller,
    job: dict[str, Any],
    prospective: dict[str, dict[str, Any]],
) -> None:
    """Publish one staged job's first authoritative lifecycle state."""

    event_kind = _initial_job_event(controller, job, prospective)
    # Invalid legacy inbox items can contain only part of a job spec.  Write
    # request provenance only after the fields it promises have been validated.
    if all(key in job for key in ("argv", "cwd", "env", "request")):
        write_request_record(controller.root, job)
    if job.get("state") in TERMINAL_JOB_STATES:
        write_result_record(controller.root, job)
    controller.state["jobs"][job["id"]] = job
    emit(controller, event_kind, job=job)


def _stage_request(
    request_id: str,
    spec: dict[str, Any] | None,
    queue_order: int,
    prospective: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Stage one directory-keyed request and flag unusable identities."""

    if spec is None:
        return (
            _rejected_job({}, queue_order, request_id, ValueError("unreadable request spec")),
            True,
        )
    if spec.get("job_id") != request_id:
        return (
            _rejected_job(
                spec,
                queue_order,
                request_id,
                ValueError("job_id does not match request directory"),
            ),
            True,
        )
    return _stage_job(spec, queue_order, prospective), False


def _finish_staged_request(
    controller: Controller, job: dict[str, Any], malformed_identity: bool
) -> None:
    """Retire one admitted inbox entry without endangering the serve loop."""

    try:
        if malformed_identity:
            reject_request(controller.root, job["id"])
        else:
            accept_request(
                controller.root,
                job["id"],
                identity_digest=job["request_digest"],
            )
    except (OSError, StorageError) as exc:
        # Admission is already durable. Retain the request directory for
        # diagnosis or exact retry and let unrelated jobs continue.
        _storage_notice(controller, "finish_request", job["id"], exc)


def _atomic_specs(submission_id: str, document: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate an envelope's outer identity before staging any task."""

    if document.get("submission_id") != submission_id:
        raise ValueError("submission_id does not match request directory")
    if document.get("kind") != "workflow":
        raise ValueError("atomic submission kind must be 'workflow'")
    expected_digest = document.get("identity_sha256")
    if not isinstance(expected_digest, str) or expected_digest != submission_identity_digest(
        document
    ):
        raise ValueError("submission identity digest does not match its content")
    specs = document.get("jobs")
    if not isinstance(specs, list) or not specs or len(specs) > 256:
        raise ValueError("submission jobs must contain between 1 and 256 tasks")
    if not all(isinstance(spec, dict) for spec in specs):
        raise ValueError("submission jobs must contain JSON objects")
    project_id = normalize_project_id(document.get("project_id"))
    workflow_id = document.get("workflow_id")
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        raise ValueError("submission workflow_id must be a non-empty string")
    job_ids = [spec.get("job_id") for spec in specs]
    if not all(isinstance(job_id, str) and job_id for job_id in job_ids):
        raise ValueError("every submitted task must have a job_id")
    if len(set(job_ids)) != len(job_ids):
        raise ValueError("submission contains duplicate job IDs")
    for spec in specs:
        if normalize_project_id(spec.get("project_id")) != project_id:
            raise ValueError("every task must use the submission project")
        if spec.get("workflow_id") != workflow_id:
            raise ValueError("every task must use the submission workflow_id")
    return specs


def _stage_atomic_submission(
    controller: Controller,
    submission_id: str,
    document: dict[str, Any],
    next_order: int,
    prospective: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Stage a whole DAG or raise without changing authoritative state."""

    specs = _atomic_specs(submission_id, document)
    staged: list[dict[str, Any]] = []
    candidate_state = dict(prospective)
    project_id = normalize_project_id(document.get("project_id"))
    workflow_id = str(document["workflow_id"])
    archived = _archived_workflow_jobs(controller, project_id, workflow_id)
    if archived is None:
        raise TransientStorageError(
            f"workflow history unavailable for {project_id}/{workflow_id}"
        )
    candidate_state = {
        **{job["id"]: job for job in archived},
        **candidate_state,
    }
    for offset, spec in enumerate(specs, start=1):
        job_id = str(spec["job_id"])
        if (
            job_id in candidate_state
            or find_archived_job(controller.root, job_id)
            or request_receipt_digest(controller.root, job_id) is not None
        ):
            raise WorkflowError(f"job ID {job_id!r} was already admitted")
        job = _stage_job(spec, next_order + offset, candidate_state)
        if job.get("state") == "rejected":
            raise WorkflowError(str(job.get("error") or "invalid workflow task"))
        staged.append(job)
        candidate_state[job_id] = job

    workflow_jobs = _resolution_workflow_jobs(candidate_state.values(), project_id, workflow_id)
    validate_workflows(workflow_jobs)
    impossible = [
        str(job["task_id"])
        for job in staged
        if not request_can_ever_fit(controller.inventory, ResourceRequest.from_dict(job["request"]))
    ]
    if impossible:
        raise ValueError(f"tasks cannot fit this allocation: {impossible!r}")
    return staged


def _finish_atomic_submission(
    controller: Controller, submission_id: str, document: dict[str, Any]
) -> None:
    try:
        specs = document.get("jobs")
        if isinstance(specs, list):
            for spec in specs:
                if isinstance(spec, dict) and isinstance(spec.get("job_id"), str):
                    record_request_receipt(
                        controller.root,
                        spec["job_id"],
                        job_identity_digest(spec),
                    )
        accept_submission(
            controller.root,
            submission_id,
            identity_digest=submission_identity_digest(document),
        )
    except (OSError, StorageError) as exc:
        _storage_notice(controller, "finish_submission", submission_id, exc)


def _admit_atomic_submission(
    controller: Controller,
    submission_id: str,
    document: dict[str, Any],
    next_order: int,
    prospective: dict[str, dict[str, Any]],
) -> int:
    """Commit all initial task images in one record, or commit none of them."""

    try:
        jobs = _stage_atomic_submission(
            controller, submission_id, document, next_order, prospective
        )
    except TransientStorageError:
        return next_order
    except (KeyError, StorageError, TypeError, ValueError) as exc:
        emit(
            controller,
            "submission.rejected",
            data={
                "submission_id": submission_id,
                "project_id": document.get("project_id"),
                "workflow_id": document.get("workflow_id"),
                "reason": str(exc),
            },
        )
        _finish_atomic_submission(controller, submission_id, document)
        return next_order

    complete = {**prospective, **{job["id"]: job for job in jobs}}
    events = [_initial_job_event(controller, job, complete) for job in jobs]
    for job in jobs:
        write_request_record(controller.root, job)
        if job.get("state") in TERMINAL_JOB_STATES:
            write_result_record(controller.root, job)
    controller.state["jobs"].update({job["id"]: job for job in jobs})
    controller.state["next_queue_order"] = next_order + len(jobs)
    emit_submission(controller, submission_id, jobs)
    for job, event_kind in zip(jobs, events, strict=True):
        emit(
            controller,
            event_kind,
            job_id=job["id"],
            data={"submission_id": submission_id},
            durable=False,
            snapshot=False,
        )
    commit_snapshot(controller)
    _finish_atomic_submission(controller, submission_id, document)
    return next_order + len(jobs)


def _ingest_requests(controller: Controller) -> None:
    known = controller.state["jobs"]
    next_order = max(
        int(controller.state.get("next_queue_order", 0)),
        max(
            (int(job.get("queue_order", 0)) for job in known.values()),
            default=0,
        ),
    )
    # Directory timestamps are not a safe admission signal on every shared
    # filesystem. Listing names each poll is cheap; only unknown specs are read.
    submissions = list_submissions(controller.root)
    requests = list_requests(controller.root, exclude=known.keys())
    # Keep every new request outside public state until its admission event.
    # Otherwise an earlier emit could snapshot a later task before its
    # dependency decision, and a crash could make that task runnable.
    staged: list[tuple[dict[str, Any], bool]] = []
    prospective: dict[str, dict[str, Any]] = {}
    workflow_keys: set[tuple[str, str]] = set()
    for _, spec in requests:
        if spec is None or not isinstance(spec.get("workflow_id"), str):
            continue
        try:
            workflow_keys.add((job_project(spec), str(spec["workflow_id"])))
        except ValueError:
            continue
    deferred_workflows: set[tuple[str, str]] = set()
    for project_id, workflow_id in workflow_keys:
        archived = _archived_workflow_jobs(controller, project_id, workflow_id)
        if archived is None:
            deferred_workflows.add((project_id, workflow_id))
            continue
        prospective.update({job["id"]: job for job in archived})
    prospective.update(known)
    atomic_job_ids: set[str] = set()
    for submission_id, document, atomic in submissions:
        if not atomic:
            continue
        if document is None:
            # Decodable corruption is a permanent verdict; transient reads do
            # not appear in list_submissions and are retried next tick.
            try:
                reject_request(controller.root, submission_id)
                emit(
                    controller,
                    "submission.rejected",
                    data={"submission_id": submission_id, "reason": "invalid document"},
                )
            except (OSError, StorageError) as exc:
                _storage_notice(controller, "reject_submission", submission_id, exc)
            continue
        specs = document.get("jobs")
        job_ids = (
            {
                str(spec.get("job_id"))
                for spec in specs
                if isinstance(specs, list) and isinstance(spec, dict)
            }
            if isinstance(specs, list)
            else set()
        )
        atomic_job_ids.update(job_ids)
        if job_ids and job_ids <= set(known):
            _finish_atomic_submission(controller, submission_id, document)
            continue
        next_order = _admit_atomic_submission(
            controller, submission_id, document, next_order, prospective
        )
        prospective.update(known)

    for request_id, spec in requests:
        # Atomic specs were handled as one transaction above.
        if request_id in atomic_job_ids:
            continue
        if request_id in prospective:
            continue
        if spec is not None and isinstance(spec.get("workflow_id"), str):
            try:
                workflow_key = (job_project(spec), str(spec["workflow_id"]))
            except ValueError:
                workflow_key = None
            if workflow_key in deferred_workflows:
                continue
        next_order += 1
        job, malformed_identity = _stage_request(request_id, spec, next_order, prospective)
        staged.append((job, malformed_identity))
        prospective[request_id] = job

    controller.state["next_queue_order"] = next_order
    for job, malformed_identity in staged:
        _admit_job(controller, job, prospective)
        _finish_staged_request(controller, job, malformed_identity)


def _workflow_groups(
    jobs: dict[str, dict[str, Any]],
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]],
    dict[tuple[str, str], list[dict[str, Any]]],
    dict[tuple[str, str], tuple[tuple[str, object], ...]],
]:
    """Group jobs and cheap change signatures in one allocation-wide scan."""

    workflows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    blocked: dict[tuple[str, str], list[dict[str, Any]]] = {}
    signatures: dict[tuple[str, str], list[tuple[str, object]]] = {}
    for job in jobs.values():
        workflow_id = job.get("workflow_id")
        task_id = job.get("task_id")
        if not isinstance(workflow_id, str) or not isinstance(task_id, str):
            continue
        workflow_key = (job_project(job), workflow_id)
        condition_signature = tuple(
            sorted(
                (
                    str(item.get("task_id")),
                    str(item.get("artifact_id")),
                    str((item.get("publication") or {}).get("sha256")),
                )
                for item in job.get("condition_satisfactions") or []
                if isinstance(item, dict)
            )
        )
        signatures.setdefault(workflow_key, []).append(
            (job["id"], (job.get("state"), condition_signature))
        )
        workflows.setdefault(workflow_key, []).append(job)
        if job.get("state") == "blocked" and not job.get("workflow_invalid"):
            blocked.setdefault(workflow_key, []).append(job)
    return (
        workflows,
        blocked,
        {workflow_key: tuple(items) for workflow_key, items in signatures.items()},
    )


def _refresh_dependencies(controller: Controller) -> None:
    """Refresh dirty workflows, including terminal dependency cascades."""

    jobs = controller.state["jobs"]
    workflows, blocked_by_workflow, signatures = _workflow_groups(jobs)
    previous = controller.workflow_signatures
    dirty = (
        list(signatures)
        if previous is None
        else [
            workflow_key
            for workflow_key, signature in signatures.items()
            if previous.get(workflow_key) != signature
        ]
    )
    if not dirty:
        controller.workflow_signatures = signatures
        return

    retry_invalid: set[tuple[str, str]] = set()
    for workflow_key in dirty:
        project_id, workflow_id = workflow_key
        blocked_jobs = blocked_by_workflow.get(workflow_key, [])
        if not blocked_jobs:
            continue
        archived = _archived_workflow_jobs(controller, project_id, workflow_id)
        if archived is None:
            retry_invalid.add(workflow_key)
            continue
        try:
            resolution_jobs = _resolution_workflow_jobs(
                [
                    *archived,
                    *workflows[workflow_key],
                ],
                project_id,
                workflow_id,
            )
            resolutions = resolve_blocked_jobs(resolution_jobs)
        except WorkflowError as exc:
            # Repair invalid persisted graphs one task at a time. Leaving the
            # cache dirty retries the remaining graph on the next tick.
            job = blocked_jobs[0]
            _mark_workflow_rejected(job, exc)
            write_result_record(controller.root, job)
            emit(controller, "job.rejected", job=job)
            retry_invalid.add(workflow_key)
            continue

        jobs_by_key = {(project_id, workflow_id, job["task_id"]): job for job in blocked_jobs}
        # The batch resolver returns topological order, so predicted upstream
        # queued/skipped states become real before dependent events are emitted.
        for key, resolution in resolutions.items():
            job = jobs_by_key[key]
            blockers = resolution["blockers"]
            decision = resolution["decision"]
            if decision == "ready":
                job["dependency_gate_passed"] = True
                job["state"] = "queued"
                job["reason"] = None
                job["blockers"] = []
                job["resolved_dependencies"] = _resolved_dependency_ids(job, resolution_jobs)
                job["resolved_conditions"] = _resolved_condition_evidence(job)
                emit(controller, "job.queued", job=job)
            elif decision == "skipped":
                job["dependency_gate_passed"] = True
                job["state"] = "skipped"
                job["finished_at"] = utc_now()
                job["reason"] = resolution["reason"]
                job["blockers"] = blockers
                write_result_record(controller.root, job)
                emit(controller, "job.skipped", job=job)
            elif blockers != job.get("blockers"):
                job["blockers"] = blockers

    # Cache each workflow independently: an unrelated job transition must not
    # revalidate every blocked graph in the allocation.
    _, _, current = _workflow_groups(jobs)
    controller.workflow_signatures = {
        workflow_key: signature
        for workflow_key, signature in current.items()
        if workflow_key not in retry_invalid
    }


def _finish_missing_cancel(controller: Controller, command: dict[str, Any], job_id: str) -> bool:
    """Emit an archived/unknown outcome; return false when it should retry."""

    if request_pending(controller.root, job_id):
        return False
    try:
        archived = find_archived_job(controller.root, job_id)
    except TransientStorageError as exc:
        _storage_notice(controller, "read_archived_job", job_id, exc)
        return False
    except (OSError, StorageError) as exc:
        _storage_notice(controller, "read_archived_job", job_id, exc)
        archived = None

    data = {"request_id": command.get("request_id"), "job_id": job_id}
    if archived is None:
        emit(
            controller,
            "command.rejected",
            data={**data, "reason": "unknown_job"},
        )
    else:
        emit(
            controller,
            "job.cancel_ignored",
            data={**data, "reason": f"job_is_{archived.get('state', 'terminal')}"},
        )
    return True


def _ingest_commands(controller: Controller) -> None:
    for source, command in list_commands(controller.root):
        deferred = False
        kind = command.get("kind")
        if kind == "cancel":
            job_id = str(command.get("job_id"))
            job = controller.state["jobs"].get(job_id)
            if job is None:
                # Submit returns only after its request is durable. If command
                # ingestion won a polling race, retain the cancel for next tick.
                deferred = not _finish_missing_cancel(controller, command, job_id)
            elif not request_cancellation(controller, job, str(command.get("request_id") or "")):
                emit(
                    controller,
                    "job.cancel_ignored",
                    data={
                        "request_id": command.get("request_id"),
                        "job_id": job_id,
                        "reason": f"job_is_{job['state']}",
                    },
                )
        elif kind == "drain":
            data = {"request_id": command.get("request_id")}
            if controller.state["draining"]:
                emit(
                    controller,
                    "allocation.drain_ignored",
                    data={**data, "reason": "already_draining"},
                )
            else:
                controller.state["draining"] = True
                controller.state["drain_requested"] = True
                controller.state["allocation"]["state"] = "draining"
                emit(controller, "allocation.draining", data=data)
        elif kind == "resume":
            data = {"request_id": command.get("request_id")}
            was_draining = bool(controller.state["draining"])
            if not was_draining and not controller.state.get("launches_paused", False):
                emit(
                    controller,
                    "allocation.resume_ignored",
                    data={**data, "reason": "launches_not_paused"},
                )
            else:
                controller.state["draining"] = False
                controller.state["drain_requested"] = False
                controller.state["launches_paused"] = False
                controller.state["allocation"]["state"] = "running"
                emit(
                    controller,
                    "allocation.launches_resumed",
                    data={**data, "cleared_drain": was_draining},
                )
        elif kind in {"gpu.quarantine", "gpu.clear", "gpu.reprobe"}:
            request_id = command.get("request_id")
            try:
                node = str(command.get("node") or "")
                uuid = str(command.get("uuid") or "")
                at = utc_now()
                if kind == "gpu.reprobe":
                    transition = reprobe_quarantine(
                        controller.state["gpu_health"],
                        node=node,
                        uuid=uuid,
                        at=at,
                    )
                else:
                    transition = set_quarantine(
                        controller.state["gpu_health"],
                        node=node,
                        uuid=uuid,
                        quarantined=kind == "gpu.quarantine",
                        at=at,
                        reason=(
                            str(command["reason"])
                            if isinstance(command.get("reason"), str)
                            else None
                        ),
                    )
            except HealthError as exc:
                emit(
                    controller,
                    "command.rejected",
                    data={"request_id": request_id, "reason": str(exc)},
                )
            else:
                _emit_gpu_health(controller, [transition], request_id=request_id)
        else:
            emit(
                controller,
                "command.rejected",
                data={
                    "request_id": command.get("request_id"),
                    "reason": f"unknown_command:{kind}",
                },
            )
        if not deferred:
            remove_command(source)


def _discard_journaled_reports(controller: Controller) -> None:
    """Acknowledge pending files whose batch outcomes survived a crash.

    New report outcomes carry a tiny acknowledgement map in the committed
    snapshot. The journal scan below is only a compatibility path for an
    interrupted controller from before that map existed.
    """

    listed = list_reports(controller.root)
    pending = {_report_id(source): (source, document) for source, document in listed}
    acknowledged: dict[Path, str | None] = {}
    report_acks = controller.state.setdefault("report_acks", {})
    legacy_format = controller.state.get("report_ack_v") != 1
    changed_snapshot = bool(report_acks) or legacy_format
    for report_id, digest in list(report_acks.items()):
        item = pending.pop(report_id, None)
        if item is not None:
            source, _ = item
            acknowledged[source] = digest if isinstance(digest, str) else None
        report_acks.pop(report_id, None)
    if acknowledged:
        accept_reports(
            tuple(acknowledged.items()),
            generation=int(controller.state.get("journal_generation", 0)),
        )
    if not pending or not legacy_format:
        controller.state["report_ack_v"] = 1
        if changed_snapshot:
            commit_snapshot(controller)
        return
    legacy = {
        (str(document.get("job_id")), str(document.get("event_id"))): source
        for source, document in pending.values()
        if isinstance(document, dict) and document.get("job_id") and document.get("event_id")
    }
    acknowledged = {}
    generation = int(controller.state.get("journal_generation", 0))
    for event in read_events(controller.root, generation=generation):
        key = (event.get("job_id"), event.get("source_event_id"))
        source = legacy.pop(key, None)
        if source is not None:
            _, document = pending.pop(_report_id(source))
            try:
                digest = report_identity_digest(validate_event(document))
            except (TypeError, ValueError):
                digest = None
            acknowledged[source] = digest
        if not pending:
            break
    accept_reports(
        tuple(acknowledged.items()),
        generation=int(controller.state.get("journal_generation", 0)),
    )
    controller.state["report_ack_v"] = 1
    if changed_snapshot:
        commit_snapshot(controller)


def _discard_journaled_commands(controller: Controller) -> None:
    """Acknowledge commands whose durable outcome survived a controller crash."""

    pending = {
        str(command.get("request_id")): source
        for source, command in list_commands(controller.root)
        if command.get("request_id")
    }
    if not pending:
        return
    generation = int(controller.state.get("journal_generation", 0))
    for event in read_events(controller.root, generation=generation):
        if event.get("kind") not in COMMAND_OUTCOME_KINDS:
            continue
        data = event.get("data")
        request_id = data.get("request_id") if isinstance(data, dict) else None
        if not isinstance(request_id, str) or not request_id:
            continue
        source = pending.pop(request_id, None)
        if source is not None:
            remove_command(source)
        if not pending:
            return


def _report_id(source: Path) -> str:
    return f"{source.parent.name}/{source.name}"


def _reject_report(
    controller: Controller,
    source: Path,
    reason: str,
    *,
    digest: str | None = None,
) -> None:
    report_id = _report_id(source)
    emit(
        controller,
        "notice",
        data={
            "kind": "workload.report_rejected",
            "report": source.name,
            "report_id": report_id,
            "reason": reason,
        },
        report_id=report_id,
        report_digest=digest,
        durable=False,
        snapshot=False,
    )
    controller.state.setdefault("report_acks", {})[report_id] = digest


def _report_batch(controller: Controller, limit: int) -> list[tuple[Path, object | None]]:
    """Round-robin pending reports so one noisy job cannot starve another."""

    if limit <= 0:
        return []
    streams = report_streams(controller.root)
    job_ids = [job_id for job_id, _ in streams]
    if not streams:
        return []
    if controller.report_cursor in job_ids:
        pivot = job_ids.index(controller.report_cursor) + 1
        streams = streams[pivot:] + streams[:pivot]

    batch: list[tuple[Path, object | None]] = []
    exhausted: set[str] = set()
    try:
        while len(batch) < limit:
            added = False
            for job_id, reports in streams:
                if job_id in exhausted:
                    continue
                try:
                    item = next(reports)
                except StopIteration:
                    exhausted.add(job_id)
                    continue
                batch.append(item)
                added = True
                if len(batch) == limit:
                    break
            if not added:
                break
    finally:
        for _, reports in streams:
            close = getattr(reports, "close", None)
            if close is not None:
                close()
    if batch:
        controller.report_cursor = batch[-1][0].parent.name
    return batch


def _ingest_reports(controller: Controller, limit: int = MAX_REPORTS_PER_TICK) -> None:
    """Validate and commit one bounded report batch with one state rewrite."""

    acknowledged: list[tuple[Path, str | None]] = []
    new_report_ids: list[str] = []
    for source, document in _report_batch(controller, limit):
        try:
            retained, retained_digest = report_was_accepted(controller.root, source)
        except (OSError, StorageError) as exc:
            _storage_notice(controller, "read_report_receipt", _report_id(source), exc)
            continue
        if retained:
            acknowledged.append((source, retained_digest))
            continue
        if document is None:
            _reject_report(controller, source, "unreadable_report")
            acknowledged.append((source, None))
            new_report_ids.append(_report_id(source))
            continue
        try:
            event = validate_event(document)
        except (TypeError, ValueError) as exc:
            _reject_report(controller, source, str(exc))
            acknowledged.append((source, None))
            new_report_ids.append(_report_id(source))
            continue
        job_id = event["job_id"]
        digest = report_identity_digest(event)
        if source.parent.name != job_id:
            _reject_report(
                controller,
                source,
                "job_id does not match report directory",
                digest=digest,
            )
            acknowledged.append((source, digest))
            new_report_ids.append(_report_id(source))
            continue
        job = controller.state["jobs"].get(job_id)
        if job is None:
            # A worker cannot start before its job is known, but an external
            # publisher may win the controller's request-ingestion poll.
            if request_pending(controller.root, job_id):
                continue
            _reject_report(
                controller,
                source,
                f"unknown job {job_id}",
                digest=digest,
            )
            acknowledged.append((source, digest))
            new_report_ids.append(_report_id(source))
            continue
        journal_event = emit(
            controller,
            event["kind"],
            job_id=job_id,
            data=event["data"],
            occurred_at=event["occurred_at"],
            source_event_id=event["event_id"],
            source=event["source"],
            report_id=_report_id(source),
            report_digest=digest,
            durable=False,
            snapshot=False,
        )
        apply_workload_event(job, event, recorded_at=journal_event["recorded_at"])
        if event["kind"] == "workload.artifact":
            _satisfy_artifact_waiters(
                controller,
                job,
                event,
                journal_event["event_id"],
            )
        controller.state.setdefault("report_acks", {})[_report_id(source)] = digest
        acknowledged.append((source, digest))
        new_report_ids.append(_report_id(source))

    if not acknowledged:
        return
    if new_report_ids:
        # The inbox is acknowledged only after both the ordered events and
        # their cumulative workload projection are durable.
        commit_snapshot(controller)
    accept_reports(
        acknowledged,
        generation=int(controller.state.get("journal_generation", 0)),
    )
    report_acks = controller.state.setdefault("report_acks", {})
    for report_id in new_report_ids:
        report_acks.pop(report_id, None)


def _heartbeat(controller: Controller) -> None:
    now = time.monotonic()
    if now - controller.last_heartbeat < 5:
        return
    controller.last_heartbeat = now
    controller.state["allocation"]["heartbeat_at"] = utc_now()
    # Output events are intentionally group-committed by the heartbeat or the
    # next durable lifecycle event. Never publish their watermark first.
    commit_snapshot(controller)


def _emit_gpu_health(
    controller: Controller,
    transitions: list[dict[str, Any]],
    *,
    request_id: object | None = None,
) -> None:
    data: dict[str, Any] = {
        "transitions": copy.deepcopy(transitions),
        "gpu_health": copy.deepcopy(controller.state["gpu_health"]),
    }
    if request_id is not None:
        data["request_id"] = request_id
    emit(controller, "resource.gpu_health_changed", data=data)


def _set_health_monitor_status(
    controller: Controller, status: str, *, error: str | None = None
) -> bool:
    health = controller.state["gpu_health"]
    monitor = health.setdefault("monitor", {})
    expected_steps = dict(sorted(controller.health_step_ids.items()))
    if (
        monitor.get("status") == status
        and monitor.get("error") == error
        and monitor.get("slurm_step_ids") == (expected_steps or None)
    ):
        return False
    monitor.update({"status": status, "error": error, "changed_at": utc_now()})
    monitor.pop("slurm_step_id", None)
    if expected_steps:
        monitor["slurm_step_ids"] = expected_steps
    else:
        monitor.pop("slurm_step_ids", None)
    return True


def _ingest_gpu_health(controller: Controller) -> None:
    if controller.gpu_health_mode == "off":
        return
    health = controller.state["gpu_health"]
    errors = health.setdefault("sample_errors", {})
    transitions: list[dict[str, Any]] = []
    sample_updated = False
    durable_changed = False
    for inventory_node in controller.inventory:
        source = controller.root / "health" / "samples" / f"{inventory_node.name}.json"
        if not source.exists():
            continue
        try:
            document = read_json(source)
            if not isinstance(document, dict):
                raise HealthError("health sample must be an object")
            expected_incarnation = (
                controller.allocation_incarnation.fingerprint_sha256
                if controller.allocation_incarnation is not None
                else None
            )
            if (
                expected_incarnation is not None
                and document.get("allocation_incarnation_sha256") != expected_incarnation
            ):
                raise HealthError("health sample belongs to another allocation incarnation")
            before = health.get("nodes", {}).get(inventory_node.name, {}).get("last_sample_at")
            transitions.extend(ingest_health_sample(health, controller.inventory, document))
            after = health["nodes"][inventory_node.name].get("last_sample_at")
            if after != before:
                health["nodes"][inventory_node.name]["last_received_at"] = utc_now()
                sample_updated = True
        except (HealthError, OSError, StorageError) as exc:
            message = str(exc)
            if errors.get(inventory_node.name) != message:
                errors[inventory_node.name] = message
                durable_changed = True
            continue
        if inventory_node.name in errors:
            del errors[inventory_node.name]
            durable_changed = True
    monitor_errors = controller.health_monitor_errors
    monitor_status = "degraded" if errors or monitor_errors else "running"
    monitor_error = (
        "; ".join(f"{node}: {message}" for node, message in sorted(monitor_errors.items())) or None
    )
    if sample_updated and _set_health_monitor_status(
        controller, monitor_status, error=monitor_error
    ):
        durable_changed = True
    if transitions or durable_changed:
        _emit_gpu_health(controller, transitions)


def _health_step_name(controller: Controller, node: str) -> str:
    return f"{controller.health_step_name}-{node}"


def _health_monitor_matches(controller: Controller, node: str) -> list[SlurmStep]:
    known_id = controller.health_step_ids.get(node)
    matches = [
        step
        for step in controller.slurm_steps
        if step.name == _health_step_name(controller, node)
        or (known_id and step.step_id == known_id)
    ]
    return list({step.step_id: step for step in matches}.values())


def _health_monitor_error(controller: Controller, node: str, error: str) -> None:
    controller.health_processes.pop(node, None)
    controller.health_step_ids.pop(node, None)
    controller.health_monitor_errors[node] = error
    controller.health_retry_at[node] = time.monotonic() + 5
    detail = "; ".join(
        f"{name}: {message}" for name, message in sorted(controller.health_monitor_errors.items())
    )
    if _set_health_monitor_status(controller, "error", error=detail):
        _emit_gpu_health(controller, [])


def _attach_health_monitor(controller: Controller, node: str, step: SlurmStep) -> None:
    controller.health_step_ids[node] = step.step_id
    controller.health_monitor_errors.pop(node, None)
    current_status = controller.state["gpu_health"].get("monitor", {}).get("status", "starting")
    if current_status not in {"running", "degraded"}:
        current_status = "starting"
    error = (
        "; ".join(
            f"{name}: {message}"
            for name, message in sorted(controller.health_monitor_errors.items())
        )
        or None
    )
    if _set_health_monitor_status(controller, str(current_status), error=error):
        _emit_gpu_health(controller, [])


def _reconcile_health_process(controller: Controller, node: str) -> bool:
    """Return true when an existing client is still settling or was handled."""

    process = controller.health_processes.get(node)
    if process is None:
        return False
    if process.poll() is not None:
        _health_monitor_error(
            controller,
            node,
            f"health monitor srun exited with status {process.returncode}",
        )
        return True
    if controller.slurm_snapshot_at <= controller.health_launch_snapshot_at.get(node, 0):
        return True
    signal_process(process, signal.SIGTERM)
    _health_monitor_error(controller, node, "Slurm no longer reports the health monitor step")
    return True


def _launch_health_monitor(controller: Controller, node: str) -> None:
    health_root = controller.root / "health"
    health_root.mkdir(parents=True, exist_ok=True)
    try:
        controller.health_processes[node] = subprocess.Popen(
            build_health_srun_argv(
                slurm_job_id=controller.slurm_job_id or "",
                name=_health_step_name(controller, node),
                root=controller.root,
                node_names=[node],
                gpus_per_node=len(controller.inventory[0].gpu_ids),
                interval=controller.gpu_health_interval,
                allocation_incarnation_sha256=(
                    controller.allocation_incarnation.fingerprint_sha256
                    if controller.allocation_incarnation is not None
                    else ""
                ),
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=build_srun_environment(),
            start_new_session=True,
        )
        controller.health_launch_snapshot_at[node] = controller.slurm_snapshot_at
        controller.health_monitor_errors.pop(node, None)
    except (OSError, ValueError) as exc:
        _health_monitor_error(controller, node, str(exc))
        return
    if _set_health_monitor_status(controller, "starting"):
        _emit_gpu_health(controller, [])


def _maintain_health_monitor(controller: Controller) -> None:
    if controller.launcher != "slurm" or controller.gpu_health_mode == "off":
        return
    for inventory_node in controller.inventory:
        node = inventory_node.name
        matches = _health_monitor_matches(controller, node)
        if len(matches) > 1:
            _health_monitor_error(controller, node, "multiple health monitor steps are live")
            continue
        if matches:
            _attach_health_monitor(controller, node, matches[0])
            continue
        if _reconcile_health_process(controller, node):
            continue
        if (
            controller.slurm_snapshot_at == 0
            or controller.slurm_query_error
            or time.monotonic() < controller.health_retry_at.get(node, 0)
        ):
            continue
        _launch_health_monitor(controller, node)


def _stop_health_monitor(controller: Controller) -> None:
    if controller.launcher != "slurm" or controller.gpu_health_mode == "off":
        return
    for process in controller.health_processes.values():
        if process.poll() is None:
            signal_process(process, signal.SIGTERM)
    for step_id in controller.health_step_ids.values():
        try:
            cancel_step(controller.slurm_job_id or "", step_id)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            if _set_health_monitor_status(controller, "error", error=str(exc)):
                _emit_gpu_health(controller, [])
            return
    controller.health_processes.clear()
    controller.health_step_ids.clear()
    if _set_health_monitor_status(controller, "stopped"):
        _emit_gpu_health(controller, [])


def _drain_for_deadline(controller: Controller) -> None:
    """Stop new launches when the configured allocation shutdown window begins."""

    if controller.state.get("draining"):
        return
    allocation = controller.state.get("allocation")
    drain_at = allocation.get("automatic_drain_at") if isinstance(allocation, dict) else None
    if not isinstance(drain_at, str):
        return
    try:
        due = datetime.fromisoformat(drain_at)
    except ValueError:
        return
    if due.tzinfo is None:
        due = due.replace(tzinfo=UTC)
    if datetime.now(UTC) < due:
        return
    controller.state["draining"] = True
    controller.state["drain_requested"] = True
    allocation["state"] = "draining"
    emit(
        controller,
        "allocation.draining",
        data={
            "reason": "allocation_deadline",
            "drain_before_end_seconds": controller.drain_before_end_seconds,
        },
    )


def _serve(controller: Controller) -> None:
    def stop(_signum: int, _frame: Any) -> None:
        controller.stopping = True

    accept_known_requests(
        controller.root,
        controller.state["jobs"].keys(),
        on_error=lambda request_id, exc: _storage_notice(
            controller, "finish_known_request", request_id, exc
        ),
    )
    compact_report_receipts(controller.root)
    _discard_journaled_reports(controller)
    _discard_journaled_commands(controller)
    remove_cold_job_directories(controller.root, controller.state["jobs"].keys())
    previous_term = signal.signal(signal.SIGTERM, stop)
    previous_int = signal.signal(signal.SIGINT, stop)
    try:
        while True:
            _ingest_requests(controller)
            _ingest_commands(controller)
            _drain_for_deadline(controller)
            drain_messages(controller)
            poll_processes(controller)
            _maintain_health_monitor(controller)
            _ingest_gpu_health(controller)
            _ingest_reports(controller)
            _refresh_dependencies(controller)
            compact_journal(controller)
            if controller.stopping:
                begin_shutdown(controller)
                _stop_health_monitor(controller)
                if controller.launcher == "slurm" or not controller.running:
                    break
            else:
                schedule(controller)
            _heartbeat(controller)
            time.sleep(controller.poll_interval)
        drain_messages(controller)
        poll_processes(controller)
        if controller.launcher == "slurm":
            controller.state["allocation"]["state"] = "controller_stopped"
            controller.state["allocation"]["controller_stopped_at"] = utc_now()
            emit(controller, "allocation.controller_stopped")
        else:
            controller.state["allocation"]["state"] = "ended"
            controller.state["allocation"]["finished_at"] = utc_now()
            emit(controller, "allocation.ended")
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


def run_controller(
    *,
    root: Path,
    inventory: tuple[NodeInventory, ...],
    launcher: str,
    allocation_id: str,
    slurm_job_id: str | None = None,
    allocation_incarnation: AllocationIncarnation | None = None,
    poll_interval: float = 0.2,
    cancel_grace: float = 30,
    start_paused: bool = False,
    drain_before_end_seconds: float = 900,
    gpu_health_mode: str = "observe",
    gpu_isolation: str = "gpu",
    gpu_health_interval: float = 10,
) -> None:
    """Own a queue until interrupted, retrying transient storage failures."""

    if launcher not in {"local", "slurm"}:
        raise ValueError(f"unknown launcher {launcher!r}")
    inventory = validate_inventory(inventory)
    if gpu_health_mode not in HEALTH_MODES:
        raise ValueError(f"unknown GPU health mode {gpu_health_mode!r}")
    if gpu_isolation not in GPU_ISOLATION_MODES:
        raise ValueError(f"unknown GPU isolation mode {gpu_isolation!r}")
    if poll_interval <= 0 or cancel_grace < 0 or drain_before_end_seconds < 0:
        raise ValueError(
            "poll interval must be positive; grace and drain window must be non-negative"
        )
    if gpu_health_interval <= 0:
        raise ValueError("GPU health interval must be positive")
    if launcher == "local" and len(inventory) != 1:
        raise ValueError("the local launcher requires a one-node inventory")
    if launcher == "slurm" and (not slurm_job_id or allocation_id != slurm_job_id):
        raise ValueError("the Slurm allocation ID must equal its Slurm job ID")
    if launcher == "slurm" and (
        allocation_incarnation is None or allocation_incarnation.slurm_job_id != slurm_job_id
    ):
        raise ValueError("a matching Slurm allocation incarnation is required")
    if launcher == "local" and allocation_incarnation is not None:
        raise ValueError("local launchers cannot have a Slurm allocation incarnation")
    if (
        launcher == "slurm"
        and gpu_health_mode != "off"
        and len({len(node.gpu_ids) for node in inventory}) != 1
    ):
        raise ValueError("GPU health monitoring requires homogeneous GPU counts")

    root = ensure_layout(root)
    with controller_lock(root):
        while True:
            controller: Controller | None = None
            try:
                controller = _initialize_controller(
                    root=root,
                    inventory=inventory,
                    launcher=launcher,
                    allocation_id=allocation_id,
                    slurm_job_id=slurm_job_id,
                    allocation_incarnation=allocation_incarnation,
                    poll_interval=poll_interval,
                    cancel_grace=cancel_grace,
                    start_paused=start_paused,
                    drain_before_end_seconds=drain_before_end_seconds,
                    gpu_health_mode=gpu_health_mode,
                    gpu_isolation=gpu_isolation,
                    gpu_health_interval=gpu_health_interval,
                )
                _serve(controller)
                return
            except (OSError, TransientStorageError) as exc:
                if launcher != "slurm":
                    raise
                print(
                    "scruffy: shared storage unavailable; "
                    f"retrying in {STORAGE_RETRY_SECONDS}s: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                if controller is not None:
                    if controller.running:
                        abandon_processes(controller)
                    try:
                        controller.journal.close()
                    except OSError:
                        pass
            time.sleep(STORAGE_RETRY_SECONDS)
