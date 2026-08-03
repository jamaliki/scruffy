"""The single-writer event loop for one Scruffy queue."""

from __future__ import annotations

import queue
import signal
import time
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
    NodeInventory,
    ResourceRequest,
    validate_inventory,
)
from .protocol import validate_event
from .runtime import Controller, OutputNotifier, abandon_processes
from .scheduler import request_can_ever_fit
from .slurm import allocation_metadata
from .state import (
    apply_workload_event,
    emit,
    job_from_spec,
    load_recovered_state,
    refresh_nodes,
)
from .storage import (
    controller_lock,
    ensure_layout,
    find_request,
    list_commands,
    list_reports,
    list_requests,
    open_journal,
    read_events,
    report_streams,
    remove_command,
    remove_report,
    UnsafeRecovery,
    utc_now,
    write_state,
)
from .workflows import WorkflowError, resolve_dependencies, validate_workflows


MAX_REPORTS_PER_TICK = 128
COMMAND_OUTCOME_KINDS = {
    "job.cancelled",
    "job.cancelling",
    "job.cancel_ignored",
    "command.rejected",
    "allocation.draining",
    "allocation.drain_ignored",
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
) -> Controller:
    state = load_recovered_state(root)
    active = [
        job
        for job in state.get("jobs", {}).values()
        if job["state"] in ACTIVE_JOB_STATES
    ]
    previous = state.get("allocation") or {}
    if active and (launcher == "local" or previous.get("id") == allocation_id):
        raise UnsafeRecovery(
            "unresolved active jobs could still be running; refusing unsafe recovery"
        )

    journal = open_journal(root)
    messages: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
    controller = Controller(
        root=root,
        inventory=inventory,
        launcher=launcher,
        allocation_id=allocation_id,
        slurm_job_id=slurm_job_id,
        poll_interval=poll_interval,
        cancel_grace=cancel_grace,
        state=state,
        journal=journal,
        messages=messages,
        output=OutputNotifier(messages),
    )

    # Clear every old placement before the first snapshot is rebuilt against
    # the new inventory. This also handles replacement node-name changes.
    for job in active:
        job["state"] = "lost"
        job["finished_at"] = utc_now()
        job["reason"] = "allocation_replaced"
        job["last_assignment"] = job.get("assignment")
        job["assignment"] = None
    for job in active:
        # The old placements may name nodes outside the replacement inventory.
        # Journal each complete image, then publish one coherent snapshot with
        # allocation.started below. A crash is recovered by journal replay.
        emit(controller, "job.lost", job=job, snapshot=False)

    metadata = allocation_metadata(allocation_id, launcher)
    metadata.update(
        {"state": "running", "started_at": utc_now(), "heartbeat_at": utc_now()}
    )
    if slurm_job_id:
        metadata["slurm_job_id"] = slurm_job_id
    state["allocation"] = metadata
    state["draining"] = False
    emit(
        controller,
        "allocation.started",
        data={"nodes": [item.to_dict() for item in inventory]},
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


def _rejected_job(
    spec: dict[str, Any], queue_order: int, job_id: str, exc: Exception
) -> dict[str, Any]:
    return {
        "id": job_id or f"invalid-{queue_order}",
        "name": str(spec.get("name", "invalid")),
        "state": "rejected",
        "submitted_at": str(spec.get("submitted_at", utc_now())),
        "queue_order": queue_order,
        "request": spec.get("resources"),
        "assignment": None,
        "finished_at": utc_now(),
        "error": str(exc),
        "reason": "invalid_spec",
    }


def _mark_workflow_rejected(job: dict[str, Any], exc: Exception) -> None:
    job["workflow_invalid"] = True
    job["state"] = "rejected"
    job["finished_at"] = utc_now()
    job["reason"] = "invalid_workflow"
    job["error"] = str(exc)


def _resolution_workflow_jobs(
    jobs: dict[str, dict[str, Any]], workflow_id: str
) -> list[dict[str, Any]]:
    """Keep rejected task identities while removing their invalid edges."""

    selected: dict[str, dict[str, Any]] = {}
    for candidate in jobs.values():
        if candidate.get("workflow_id") != workflow_id:
            continue
        task_id = candidate.get("task_id")
        if not isinstance(task_id, str):
            continue
        if candidate.get("workflow_invalid"):
            selected.setdefault(task_id, {**candidate, "needs": []})
        else:
            selected[task_id] = candidate
    return list(selected.values())


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
    duplicate = next(
        (
            candidate
            for candidate in prospective.values()
            if workflow_id is not None
            and candidate.get("workflow_id") == workflow_id
            and candidate.get("task_id") == job.get("task_id")
        ),
        None,
    )
    if duplicate is not None:
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
            for candidate in prospective.values()
            if candidate.get("workflow_id") == workflow_id
            and not candidate.get("workflow_invalid")
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

    workflow_jobs = _resolution_workflow_jobs(prospective, job["workflow_id"])
    resolution = resolve_dependencies(job, workflow_jobs)
    job["blockers"] = resolution["blockers"]
    if resolution["decision"] == "ready":
        emit(controller, "job.queued", job=job)
    elif resolution["decision"] == "blocked":
        job["state"] = "blocked"
        job["reason"] = "waiting_for_dependencies"
        emit(controller, "job.blocked", job=job)
    else:
        job["state"] = "skipped"
        job["finished_at"] = utc_now()
        job["reason"] = "dependency_unsatisfied"
        emit(controller, "job.skipped", job=job)


def _ingest_requests(controller: Controller) -> None:
    known = controller.state["jobs"]
    next_order = max(
        (int(job.get("queue_order", 0)) for job in known.values()), default=0
    )
    # Directory timestamps are not a safe admission signal on every shared
    # filesystem. Listing names each poll is cheap; only unknown specs are read.
    specs = sorted(
        list_requests(controller.root, exclude=known.keys()),
        key=lambda spec: (str(spec.get("submitted_at", "")), str(spec.get("job_id", ""))),
    )
    # Keep every new request outside public state until its admission event.
    # Otherwise an earlier emit could snapshot a later task before its
    # dependency decision, and a crash could make that task runnable.
    staged: list[dict[str, Any]] = []
    prospective = dict(known)
    for spec in specs:
        job_id = str(spec.get("job_id", ""))
        if job_id in prospective:
            continue
        next_order += 1
        job = _stage_job(spec, next_order, prospective)
        staged.append(job)
        prospective[job_id] = job

    for job in staged:
        _admit_job(controller, job, prospective)


def _refresh_dependencies(controller: Controller) -> None:
    """Release or skip blocked work after upstream lifecycle transitions."""

    jobs = controller.state["jobs"]
    while True:
        transitioned = False
        for job in list(jobs.values()):
            if job.get("state") != "blocked" or job.get("workflow_invalid"):
                continue
            workflow_jobs = _resolution_workflow_jobs(jobs, job["workflow_id"])
            try:
                resolution = resolve_dependencies(job, workflow_jobs)
            except WorkflowError as exc:
                _mark_workflow_rejected(job, exc)
                emit(controller, "job.rejected", job=job)
                transitioned = True
                continue
            blockers = resolution["blockers"]
            decision = resolution["decision"]
            if decision == "ready":
                job["state"] = "queued"
                job["reason"] = None
                job["blockers"] = []
                emit(controller, "job.queued", job=job)
                transitioned = True
            elif decision == "skipped":
                job["state"] = "skipped"
                job["finished_at"] = utc_now()
                job["reason"] = "dependency_unsatisfied"
                job["blockers"] = blockers
                emit(controller, "job.skipped", job=job)
                transitioned = True
            elif blockers != job.get("blockers"):
                job["blockers"] = blockers
                emit(controller, "job.blocked", job=job)
        if not transitioned:
            return


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
                if find_request(controller.root, job_id) is not None:
                    deferred = True
                else:
                    emit(
                        controller,
                        "command.rejected",
                        data={
                            "request_id": command.get("request_id"),
                            "job_id": job_id,
                            "reason": "unknown_job",
                        },
                    )
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
                controller.state["allocation"]["state"] = "draining"
                emit(controller, "allocation.draining", data=data)
        if not deferred:
            remove_command(source)


def _discard_journaled_reports(controller: Controller) -> None:
    """Close the append/remove crash window without retaining a global ID set."""

    pending = {
        (str(document.get("job_id")), str(document.get("event_id"))): source
        for source, document in list_reports(controller.root)
        if isinstance(document, dict)
        and document.get("job_id")
        and document.get("event_id")
    }
    if not pending:
        return
    for event in read_events(controller.root):
        job_id = event.get("job_id")
        source_event_id = event.get("source_event_id")
        if not isinstance(job_id, str) or not isinstance(source_event_id, str):
            continue
        key = (job_id, source_event_id)
        source = pending.pop(key, None)
        if source is not None:
            remove_report(source)
        if not pending:
            return


def _discard_journaled_commands(controller: Controller) -> None:
    """Acknowledge commands whose durable outcome survived a controller crash."""

    pending = {
        str(command.get("request_id")): source
        for source, command in list_commands(controller.root)
        if command.get("request_id")
    }
    if not pending:
        return
    for event in read_events(controller.root):
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


def _reject_report(controller: Controller, source: Path, reason: str) -> None:
    emit(
        controller,
        "notice",
        data={
            "kind": "workload.report_rejected",
            "report": source.name,
            "reason": reason,
        },
    )
    remove_report(source)


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
    """Validate and sequence a bounded batch of non-authoritative job reports."""

    for source, document in _report_batch(controller, limit):
        if document is None:
            _reject_report(controller, source, "unreadable_report")
            continue
        try:
            event = validate_event(document)
        except (TypeError, ValueError) as exc:
            _reject_report(controller, source, str(exc))
            continue
        job_id = event["job_id"]
        if source.parent.name != job_id:
            _reject_report(controller, source, "job_id does not match report directory")
            continue
        job = controller.state["jobs"].get(job_id)
        if job is None:
            # A worker cannot start before its job is known, but an external
            # publisher may win the controller's request-ingestion poll.
            if find_request(controller.root, job_id) is not None:
                continue
            _reject_report(controller, source, f"unknown job {job_id}")
            continue
        recorded_at = utc_now()
        apply_workload_event(job, event, recorded_at=recorded_at)
        emit(
            controller,
            event["kind"],
            job=job,
            data=event["data"],
            occurred_at=event["occurred_at"],
            source_event_id=event["event_id"],
            source=event["source"],
        )
        remove_report(source)


def _heartbeat(controller: Controller) -> None:
    now = time.monotonic()
    if now - controller.last_heartbeat < 5:
        return
    controller.last_heartbeat = now
    controller.state["allocation"]["heartbeat_at"] = utc_now()
    refresh_nodes(controller.state, controller.inventory)
    write_state(controller.root, controller.state)


def _serve(controller: Controller) -> None:
    def stop(_signum: int, _frame: Any) -> None:
        controller.stopping = True

    _discard_journaled_reports(controller)
    _discard_journaled_commands(controller)
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
            if controller.stopping:
                begin_shutdown(controller)
                if not controller.running:
                    break
            else:
                schedule(controller)
            _heartbeat(controller)
            time.sleep(controller.poll_interval)
        drain_messages(controller)
        poll_processes(controller)
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
    poll_interval: float = 0.2,
    cancel_grace: float = 30,
) -> None:
    """Own a queue until interrupted, launching jobs as capacity becomes free."""

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

    root = ensure_layout(root)
    with controller_lock(root):
        controller = _initialize_controller(
            root=root,
            inventory=inventory,
            launcher=launcher,
            allocation_id=allocation_id,
            slurm_job_id=slurm_job_id,
            poll_interval=poll_interval,
            cancel_grace=cancel_grace,
        )
        try:
            _serve(controller)
        finally:
            if controller.running:
                abandon_processes(controller)
            controller.journal.close()
