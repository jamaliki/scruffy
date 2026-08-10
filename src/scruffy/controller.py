"""The single-writer event loop for one Scruffy queue."""

from __future__ import annotations

import queue
import signal
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .lifecycle import (
    begin_shutdown,
    drain_messages,
    poll_processes,
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
from .protocol import validate_event
from .runtime import Controller, OutputNotifier, RunningProcess, abandon_processes
from .scheduler import InvariantError, assert_invariants, request_can_ever_fit
from .slurm import AllocationIncarnation, allocation_metadata
from .state import (
    apply_workload_event,
    commit_snapshot,
    compact_journal,
    emit,
    job_from_spec,
    load_recovered_state,
)
from .storage import (
    StorageError,
    TransientStorageError,
    UnsafeRecovery,
    accept_known_requests,
    accept_reports,
    accept_request,
    compact_report_receipts,
    controller_lock,
    ensure_layout,
    find_archived_job,
    job_identity_digest,
    list_archived_workflow,
    list_commands,
    list_reports,
    list_requests,
    open_journal,
    read_events,
    reject_request,
    remove_cold_job_directories,
    remove_command,
    report_identity_digest,
    report_streams,
    report_was_accepted,
    request_pending,
    utc_now,
)
from .workflows import (
    WorkflowError,
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
) -> Controller:
    state = load_recovered_state(root)
    active = [
        job
        for job in state.get("jobs", {}).values()
        if job["state"] in ACTIVE_JOB_STATES
    ]
    previous = state.get("allocation") or {}
    if launcher == "slurm" and (
        allocation_incarnation is None
        or allocation_incarnation.slurm_job_id != slurm_job_id
    ):
        raise ValueError("Slurm allocation incarnation is missing or differs")
    previous_incarnation = None
    raw_previous_incarnation = previous.get("incarnation")
    if raw_previous_incarnation is not None:
        try:
            previous_incarnation = AllocationIncarnation.from_dict(
                raw_previous_incarnation
            )
        except (TypeError, ValueError) as exc:
            raise UnsafeRecovery(
                f"invalid persisted allocation incarnation: {exc}"
            ) from exc
        if previous.get("id") != previous_incarnation.slurm_job_id:
            raise UnsafeRecovery(
                "persisted allocation ID differs from its incarnation"
            )
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
    if active and launcher == "local":
        raise UnsafeRecovery(
            "unresolved active jobs could still be running; refusing unsafe recovery"
        )
    if same_slurm_allocation:
        try:
            assignments = tuple(
                Assignment.from_dict(job["assignment"]) for job in active
            )
            assert_invariants(inventory, assignments)
        except (KeyError, ValueError, InvariantError) as exc:
            raise UnsafeRecovery(
                f"invalid active Slurm assignments; refusing recovery: {exc}"
            ) from exc
        if active and any(not job.get("launch_token") for job in active):
            raise UnsafeRecovery(
                "active Slurm job has no launch token; refusing unsafe recovery"
            )
        incarnation_sha256 = allocation_incarnation.fingerprint_sha256
        if active and any(
            job.get("allocation_incarnation_sha256") != incarnation_sha256
            for job in active
        ):
            raise UnsafeRecovery(
                "active Slurm job allocation incarnation differs; "
                "refusing unsafe recovery"
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
        state=state,
        journal=journal,
        messages=messages,
        output=OutputNotifier(messages),
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
            job["last_assignment"] = job.get("assignment")
            job["assignment"] = None
        for job in active:
            emit(controller, "job.lost", job=job, snapshot=False)

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
            "started_at": (
                previous.get("started_at", now) if same_slurm_allocation else now
            ),
            "controller_started_at": now,
            "heartbeat_at": now,
        }
    )
    if slurm_job_id:
        metadata["slurm_job_id"] = slurm_job_id
    drain_requested = bool(
        state.get(
            "drain_requested",
            state.get("draining") and previous.get("state") == "draining",
        )
    )
    preserve_drain = drain_requested and previous.get("id") == allocation_id
    if preserve_drain:
        metadata["state"] = "draining"
    state["allocation"] = metadata
    state["draining"] = preserve_drain
    state["drain_requested"] = preserve_drain
    # Recovery owns existing steps but never admits additional work implicitly.
    # An operator must explicitly resume after checking the recovered snapshot.
    state["launches_paused"] = same_slurm_allocation or legacy_slurm_allocation
    emit(
        controller,
        "allocation.resumed" if same_slurm_allocation else "allocation.started",
        data={
            "nodes": [item.to_dict() for item in inventory],
            "incarnation": (
                allocation_incarnation.to_dict()
                if allocation_incarnation is not None
                else None
            ),
            "reattached_jobs": (
                [job["id"] for job in active] if same_slurm_allocation else []
            ),
            "lost_jobs": (
                [job["id"] for job in active] if lost_reason is not None else []
            ),
            "lost_reason": lost_reason,
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
                    else "legacy_incarnation_audit_required"
                )
            },
        )
    for job in state["jobs"].values():
        if job["state"] not in {"queued", "blocked"}:
            continue
        request = ResourceRequest.from_dict(job["request"])
        if request_can_ever_fit(inventory, request):
            continue
        job["state"] = "rejected"
        job["finished_at"] = utc_now()
        job["reason"] = "request_cannot_fit"
        job["error"] = "request cannot fit this allocation inventory"
        emit(controller, "job.rejected", job=job)
    return controller


def _reattach_slurm_jobs(
    controller: Controller, jobs: list[dict[str, Any]]
) -> None:
    """Restore ownership of persisted steps without their old local clients."""

    for job in jobs:
        job.pop("pid", None)
        running = RunningProcess(None, str(job["launch_token"]))
        running.closed_streams.update({"stdout", "stderr"})
        for stream_name in ("stdout", "stderr"):
            relative_name = job.get(
                stream_name, f"jobs/{job['id']}/{stream_name}.log"
            )
            try:
                size = (controller.root / relative_name).stat().st_size
            except FileNotFoundError:
                size = 0
            running.output_offsets[stream_name] = size
        if job["state"] == "cancelling":
            running.final_state = "cancelled"
            running.final_reason = "cancelled"
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
                "blockers": [],
                "workflow_invalid": True,
            }
        )
    return job


def _mark_workflow_rejected(job: dict[str, Any], exc: Exception) -> None:
    job["workflow_invalid"] = True
    job["needs"] = []
    job["state"] = "rejected"
    job["finished_at"] = utc_now()
    job["reason"] = "invalid_workflow"
    job["error"] = str(exc)


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
            candidate = {**candidate, "needs": []}
        resolution_jobs.append(candidate)
    return resolution_jobs


def _storage_notice(
    controller: Controller, operation: str, item: str, exc: Exception
) -> None:
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
        _storage_notice(
            controller, "read_workflow_archive", f"{project_id}/{workflow_id}", exc
        )
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
        select_task_attempts(prospective.values()).get(
            (project_id, workflow_id, task_id)
        )
        if isinstance(workflow_id, str) and isinstance(task_id, str)
        else None
    )
    duplicate_state = duplicate.get("state") if duplicate is not None else None
    if (
        duplicate is not None
        and not duplicate.get("workflow_invalid")
        and (
            duplicate_state not in TERMINAL_JOB_STATES
            or duplicate_state == "succeeded"
        )
    ):
        _mark_workflow_rejected(
            job,
            WorkflowError(
                f"duplicate task_id {job.get('task_id')!r} in workflow "
                f"{workflow_id!r}"
            ),
        )
        return job

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


def _admit_job(
    controller: Controller,
    job: dict[str, Any],
    prospective: dict[str, dict[str, Any]],
) -> None:
    """Publish one staged job's first authoritative lifecycle state."""

    controller.state["jobs"][job["id"]] = job
    if job.get("state") == "rejected":
        emit(controller, "job.rejected", job=job)
        return
    request = ResourceRequest.from_dict(job["request"])
    if not request_can_ever_fit(controller.inventory, request):
        job["state"] = "rejected"
        job["finished_at"] = utc_now()
        job["reason"] = "request_cannot_fit"
        job["error"] = "request cannot fit this allocation inventory"
        emit(controller, "job.rejected", job=job)
        return
    if job.get("workflow_id") is None:
        emit(controller, "job.queued", job=job)
        return

    workflow_jobs = _resolution_workflow_jobs(
        prospective.values(), job_project(job), job["workflow_id"]
    )
    resolution = resolve_dependencies(job, workflow_jobs)
    job["blockers"] = resolution["blockers"]
    if resolution["decision"] == "ready":
        job["dependency_gate_passed"] = True
        emit(controller, "job.queued", job=job)
    elif resolution["decision"] == "blocked":
        job["dependency_gate_passed"] = False
        job["state"] = "blocked"
        job["reason"] = "waiting_for_dependencies"
        emit(controller, "job.blocked", job=job)
    else:
        job["dependency_gate_passed"] = True
        job["state"] = "skipped"
        job["finished_at"] = utc_now()
        job["reason"] = "dependency_unsatisfied"
        emit(controller, "job.skipped", job=job)


def _stage_request(
    request_id: str,
    spec: dict[str, Any] | None,
    queue_order: int,
    prospective: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Stage one directory-keyed request and flag unusable identities."""

    if spec is None:
        return (
            _rejected_job(
                {}, queue_order, request_id, ValueError("unreadable request spec")
            ),
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
        prospective.update(
            {
                job["id"]: job
                for job in archived
            }
        )
    prospective.update(known)
    for request_id, spec in requests:
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
        job, malformed_identity = _stage_request(
            request_id, spec, next_order, prospective
        )
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
        signatures.setdefault(workflow_key, []).append((job["id"], job.get("state")))
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
            emit(controller, "job.rejected", job=job)
            retry_invalid.add(workflow_key)
            continue

        jobs_by_key = {
            (project_id, workflow_id, job["task_id"]): job for job in blocked_jobs
        }
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
                emit(controller, "job.queued", job=job)
            elif decision == "skipped":
                job["dependency_gate_passed"] = True
                job["state"] = "skipped"
                job["finished_at"] = utc_now()
                job["reason"] = "dependency_unsatisfied"
                job["blockers"] = blockers
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


def _finish_missing_cancel(
    controller: Controller, command: dict[str, Any], job_id: str
) -> bool:
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
            elif not request_cancellation(
                controller, job, str(command.get("request_id") or "")
            ):
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
            if controller.state["draining"]:
                emit(
                    controller,
                    "allocation.resume_ignored",
                    data={**data, "reason": "allocation_draining"},
                )
            elif not controller.state.get("launches_paused", False):
                emit(
                    controller,
                    "allocation.resume_ignored",
                    data={**data, "reason": "launches_not_paused"},
                )
            else:
                controller.state["launches_paused"] = False
                emit(controller, "allocation.launches_resumed", data=data)
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
        if isinstance(document, dict)
        and document.get("job_id")
        and document.get("event_id")
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


def _report_batch(
    controller: Controller, limit: int
) -> list[tuple[Path, object | None]]:
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


def _ingest_reports(
    controller: Controller, limit: int = MAX_REPORTS_PER_TICK
) -> None:
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
            drain_messages(controller)
            poll_processes(controller)
            _ingest_reports(controller)
            _refresh_dependencies(controller)
            compact_journal(controller)
            if controller.stopping:
                begin_shutdown(controller)
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
) -> None:
    """Own a queue until interrupted, retrying transient storage failures."""

    if launcher not in {"local", "slurm"}:
        raise ValueError(f"unknown launcher {launcher!r}")
    inventory = validate_inventory(inventory)
    if poll_interval <= 0 or cancel_grace < 0:
        raise ValueError("poll interval must be positive and cancel grace non-negative")
    if launcher == "local" and len(inventory) != 1:
        raise ValueError("the local launcher requires a one-node inventory")
    if launcher == "slurm" and (
        not slurm_job_id or allocation_id != slurm_job_id
    ):
        raise ValueError("the Slurm allocation ID must equal its Slurm job ID")
    if launcher == "slurm" and (
        allocation_incarnation is None
        or allocation_incarnation.slurm_job_id != slurm_job_id
    ):
        raise ValueError("a matching Slurm allocation incarnation is required")
    if launcher == "local" and allocation_incarnation is not None:
        raise ValueError("local launchers cannot have a Slurm allocation incarnation")

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
