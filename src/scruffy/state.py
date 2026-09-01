"""Queue-state transformations and durable event emission."""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

from .health import unavailable_gpu_ids
from .models import (
    ACTIVE_JOB_STATES,
    DEFAULT_PROJECT,
    TERMINAL_JOB_STATES,
    Assignment,
    NodeInventory,
    job_project,
)
from .protocol import EVENT_KINDS, artifact_publication
from .runtime import Controller
from .scheduler import available_resources
from .storage import (
    StorageError,
    activate_journal_generation,
    append_event,
    archive_terminal_job,
    create_journal_generation,
    latest_checkpoint,
    load_state,
    next_journal_generation,
    open_journal,
    prune_journal_generations,
    prune_report_receipts,
    queue_id,
    read_event_page,
    remove_cold_job_directories,
    sync_file,
    sync_report_inboxes,
    utc_now,
    write_state,
)

MAX_JOURNAL_BYTES = 64 * 1024 * 1024
MAX_TERMINAL_JOBS = 1000
TERMINAL_COMPACTION_SLACK = 100


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


def _remember_armed_trigger_evidence(
    state: dict[str, Any], job: dict[str, Any], event: dict[str, Any]
) -> None:
    """Retain an armed trigger publication while replaying its journal event."""

    evacuation = state.get("evacuation")
    if not isinstance(evacuation, dict) or evacuation.get("state") != "armed":
        return
    request_id = evacuation.get("request_id")
    request = state.get("evacuation_requests", {}).get(request_id)
    trigger = request.get("trigger") if isinstance(request, dict) else None
    publication = artifact_publication(event.get("data"))
    if (
        not isinstance(request_id, str)
        or not isinstance(request, dict)
        or not isinstance(trigger, dict)
        or publication is None
        or job_project(job) != request.get("scope", {}).get("project_id")
        or job.get("workflow_id") != request.get("scope", {}).get("workflow_id")
        or job.get("task_id") != trigger.get("task_id")
        or publication.get("artifact_id") != trigger.get("artifact_id")
    ):
        return
    expected = job.get("launch_token")
    source = event.get("source")
    if not isinstance(source, dict) or (
        isinstance(expected, str) and source.get("launch_token") != expected
    ):
        return
    request["trigger_evidence"] = {
        "producer_job_id": job.get("id"),
        "producer_event_id": event.get("event_id"),
        "publication": copy.deepcopy(publication),
    }


def apply_workload_event(
    job: dict[str, Any], event: dict[str, Any], *, recorded_at: str
) -> None:
    """Project one validated producer event onto a job's current workload view.

    Producer reports cannot directly alter lifecycle or placement fields. A
    strictly typed artifact publication may separately satisfy an explicitly
    declared workflow condition in the controller; this projection itself only
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
        artifacts = [
            item
            for item in list(workload.get("latest_artifacts") or [])
            if item.get("event_id") != event["event_id"]
        ]
        artifacts.append(
            {
                **data,
                "occurred_at": event["occurred_at"],
                "event_id": event["event_id"],
                "source": copy.deepcopy(event.get("source", {})),
            }
        )
        artifacts.sort(
            key=lambda item: _event_key(item["occurred_at"], item["event_id"])
        )
        workload["latest_artifacts"] = artifacts[-8:]
        try:
            publication = artifact_publication(data)
        except ValueError:
            publication = None
        if publication is not None:
            evidence = [
                item
                for item in list(job.get("artifact_evidence") or [])
                if item.get("producer_event_id") != event["event_id"]
            ]
            evidence.append(
                {
                    "publication": publication,
                    "producer_event_id": event["event_id"],
                    "occurred_at": event["occurred_at"],
                    "source": copy.deepcopy(event.get("source", {})),
                }
            )
            evidence.sort(
                key=lambda item: _event_key(
                    item["occurred_at"], item["producer_event_id"]
                )
            )
            job["artifact_evidence"] = evidence[-8:]
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


def active_assignments(state: dict[str, Any]) -> tuple[Assignment, ...]:
    """Decode every assignment which must still hold resources."""

    return tuple(
        Assignment.from_dict(job["assignment"])
        for job in state["jobs"].values()
        if job["state"] in ACTIVE_JOB_STATES and job.get("assignment") is not None
    )


def refresh_nodes(
    state: dict[str, Any], inventory: tuple[NodeInventory, ...]
) -> None:
    """Rebuild the public per-node ledger from active assignments."""

    assignments = active_assignments(state)
    health = state.get("gpu_health")
    health_view = health if isinstance(health, dict) else {}
    unavailable = unavailable_gpu_ids(health_view, inventory)
    free_by_node = {
        item.name: item
        for item in available_resources(inventory, assignments, unavailable)
    }
    raw_health_nodes = health_view.get("nodes")
    health_nodes = raw_health_nodes if isinstance(raw_health_nodes, dict) else {}
    nodes = {
        item.name: {
            "capacity": item.to_dict(),
            "free": {
                "gpu_ids": list(free_by_node[item.name].gpu_ids),
                "cpus": free_by_node[item.name].cpus,
                "memory_gb": free_by_node[item.name].memory_gb,
            },
            "assignments": {},
            "unavailable_gpu_ids": sorted(unavailable.get(item.name, ())),
            "gpu_devices": _gpu_device_view(
                item, health_nodes.get(item.name)
            ),
        }
        for item in inventory
    }
    for assignment in assignments:
        for reservation in assignment.reservations:
            node = nodes[reservation.node]
            node["assignments"][assignment.job_id] = reservation.to_dict()
    state["nodes"] = nodes
    state["updated_at"] = utc_now()


def _gpu_device_view(
    inventory: NodeInventory, node_health: object
) -> list[dict[str, Any]]:
    """Return one bounded public identity/status record per scheduler slot."""

    raw_devices = (
        node_health.get("devices") if isinstance(node_health, dict) else None
    )
    devices = raw_devices.values() if isinstance(raw_devices, dict) else ()
    by_slot = {
        device.get("slot"): device
        for device in devices
        if isinstance(device, dict) and type(device.get("slot")) is int
    }
    return [
        copy.deepcopy(
            by_slot.get(
                slot,
                {
                    "node": inventory.name,
                    "slot": slot,
                    "uuid": None,
                    "status": "unknown",
                    "last_sample_at": None,
                    "last_received_at": None,
                },
            )
        )
        for slot in inventory.gpu_ids
    ]


def emit(
    controller: Controller,
    kind: str,
    *,
    job: dict[str, Any] | None = None,
    job_id: str | None = None,
    data: dict[str, Any] | None = None,
    durable: bool = True,
    snapshot: bool = True,
    occurred_at: str | None = None,
    source_event_id: str | None = None,
    source: dict[str, str] | None = None,
    report_id: str | None = None,
    report_digest: str | None = None,
) -> dict[str, Any]:
    """Append one ordered event, then optionally publish the new snapshot.

    Lifecycle events carry a complete ``job`` image for recovery. High-rate
    output and workload events should pass only ``job_id`` and a small delta.
    """

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
    if report_id is not None:
        event["report_id"] = report_id
    if report_digest is not None:
        event["report_digest"] = report_digest
    if job is not None:
        if job_id is not None and job_id != job["id"]:
            raise ValueError("job and job_id refer to different jobs")
        job_id = str(job["id"])
    if job_id is not None:
        event["job_id"] = job_id
        authoritative = job or state.get("jobs", {}).get(job_id)
        if isinstance(authoritative, dict):
            event["project_id"] = job_project(authoritative)
    if job is not None:
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
    if durable:
        _reopen_journal(controller)
    return event


def emit_submission(
    controller: Controller,
    submission_id: str,
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Commit a complete admission set in one recoverable journal record."""

    state = controller.state
    state["last_seq"] += 1
    recorded_at = utc_now()
    projects = {job_project(job) for job in jobs}
    event: dict[str, Any] = {
        "v": 1,
        "queue_id": state["queue_id"],
        "seq": state["last_seq"],
        "event_id": f"{state['queue_id']}:{state['last_seq']}",
        "at": recorded_at,
        "recorded_at": recorded_at,
        "kind": "submission.admitted",
        "allocation_id": controller.allocation_id,
        "submission_id": submission_id,
        "jobs": copy.deepcopy(jobs),
    }
    if len(projects) == 1:
        event["project_id"] = projects.pop()
    append_event(controller.journal, event, sync=True)
    state["journal_offset"] = controller.journal.tell()
    _reopen_journal(controller)
    return event


def _reopen_journal(controller: Controller) -> None:
    """Replace a committed append handle before a network filesystem ages it."""

    replacement = open_journal(
        controller.root, int(controller.state.get("journal_generation", 0))
    )
    controller.journal.close()
    controller.journal = replacement


def commit_snapshot(controller: Controller) -> None:
    """Durably commit prior events, then publish one cumulative state image."""

    sync_file(controller.journal)
    refresh_nodes(controller.state, controller.inventory)
    write_state(controller.root, controller.state)
    _reopen_journal(controller)


def compact_journal(
    controller: Controller,
    *,
    max_bytes: int = MAX_JOURNAL_BYTES,
    max_terminal_jobs: int = MAX_TERMINAL_JOBS,
    terminal_slack: int = TERMINAL_COMPACTION_SLACK,
) -> bool:
    """Rotate history and move old terminal details out of the hot snapshot."""

    terminal = [
        job
        for job in controller.state["jobs"].values()
        if job.get("state") in TERMINAL_JOB_STATES
    ]
    oversized = max_bytes > 0 and controller.journal.tell() > max_bytes
    overfull = (
        max_terminal_jobs >= 0
        and len(terminal) > max_terminal_jobs + max(terminal_slack, 0)
    )
    if not oversized and not overfull:
        return False
    commit_snapshot(controller)
    terminal.sort(
        key=lambda job: (
            str(job.get("finished_at") or job.get("submitted_at") or ""),
            int(job.get("queue_order", 0)),
        ),
        reverse=True,
    )
    archived_counts = controller.state.setdefault("archived_counts", {})
    archived_project_counts = controller.state.setdefault(
        "archived_project_counts", {}
    )
    retain_count = len(terminal) if max_terminal_jobs < 0 else max_terminal_jobs
    remaining = len(terminal)
    for job in reversed(terminal):
        if remaining <= retain_count:
            break
        try:
            archive_terminal_job(controller.root, job)
        except (OSError, StorageError) as exc:
            emit(
                controller,
                "notice",
                data={
                    "kind": "storage.item_skipped",
                    "operation": "archive_terminal_job",
                    "item": str(job.get("id", "unknown")),
                    "error": str(exc),
                },
                snapshot=False,
            )
            continue
        state_name = str(job.get("state", "unknown"))
        archived_counts[state_name] = int(archived_counts.get(state_name, 0)) + 1
        project_counts = archived_project_counts.setdefault(job_project(job), {})
        project_counts[state_name] = int(project_counts.get(state_name, 0)) + 1
        del controller.state["jobs"][job["id"]]
        remaining -= 1
    controller.state["archived_jobs"] = sum(
        int(count) for count in archived_counts.values()
    )
    current = int(controller.state.get("journal_generation", 0))
    generation = next_journal_generation(controller.root, current)
    checkpoint = copy.deepcopy(controller.state)
    checkpoint["journal_generation"] = generation
    checkpoint["journal_offset"] = 0
    create_journal_generation(controller.root, generation, checkpoint)

    old_journal = controller.journal
    controller.state["journal_generation"] = generation
    controller.state["journal_offset"] = 0
    write_state(controller.root, controller.state)
    activate_journal_generation(controller.root, generation)
    controller.journal = open_journal(controller.root, generation)
    old_journal.close()
    # Keep one prior generation briefly so a reader which raced the atomic
    # state replacement can finish; older cursors reset from the new snapshot.
    retained_generations = {current, generation}
    prune_journal_generations(controller.root, retained_generations)
    sync_report_inboxes(controller.root)
    prune_report_receipts(controller.root, retained_generations)
    remove_cold_job_directories(controller.root, controller.state["jobs"].keys())
    return True


def load_recovered_state(root: Path) -> dict[str, Any]:
    """Load a snapshot and replay newer complete job images from the journal."""

    state = load_state(root)
    rebuilding = state is None
    if rebuilding:
        recovered = latest_checkpoint(root)
        if recovered is not None:
            _, state = recovered
            rebuilding = False
        else:
            state = {
                "v": 1,
                "queue_id": queue_id(root),
                "last_seq": 0,
                "journal_generation": 0,
                "journal_offset": 0,
                "allocation": None,
                "nodes": {},
                "gpu_health": None,
                "jobs": {},
                "report_acks": {},
                "report_ack_v": 1,
                "next_queue_order": 0,
                "archived_jobs": 0,
                "archived_counts": {},
                "archived_project_counts": {},
                "draining": False,
                "drain_requested": False,
                "launches_paused": False,
                "evacuation": None,
                "evacuation_requests": {},
                "evacuation_history": {},
                "evacuation_cancel_requests": {},
                "updated_at": utc_now(),
            }
    generation = int(state.get("journal_generation", 0))
    if not rebuilding and generation > 0:
        activate_journal_generation(root, generation)
    events, journal_offset, _ = read_event_page(
        root,
        after=int(state.get("last_seq", 0)),
        offset=0 if rebuilding else int(state.get("journal_offset", 0)),
        generation=generation,
    )
    for event in events:
        allocation_id = event.get("allocation_id")
        allocation_kind = event.get("kind") in {
            "allocation.started",
            "allocation.resumed",
        }
        if allocation_id and (state.get("allocation") is None or allocation_kind):
            recovered_allocation = {
                "id": str(allocation_id),
                "state": "recovered",
            }
            data = event.get("data")
            if allocation_kind and isinstance(data, dict):
                incarnation = data.get("incarnation")
                if isinstance(incarnation, dict):
                    recovered_allocation["incarnation"] = copy.deepcopy(incarnation)
                release = data.get("controller_release")
                recovered_allocation["controller_release"] = (
                    release.strip()
                    if isinstance(release, str) and release.strip()
                    else "unknown"
                )
            else:
                recovered_allocation["controller_release"] = "unknown"
            state["allocation"] = recovered_allocation
        admitted = event.get("jobs") if event.get("kind") == "submission.admitted" else None
        if isinstance(admitted, list):
            for candidate in admitted:
                if isinstance(candidate, dict) and "id" in candidate:
                    state.setdefault("jobs", {})[str(candidate["id"])] = candidate
        job = event.get("job")
        if isinstance(job, dict) and "id" in job:
            state.setdefault("jobs", {})[str(job["id"])] = job
        elif event.get("kind") in EVENT_KINDS:
            current = state.setdefault("jobs", {}).get(event.get("job_id"))
            if (
                isinstance(current, dict)
                and isinstance(event.get("source_event_id"), str)
                and isinstance(event.get("occurred_at"), str)
                and isinstance(event.get("data"), dict)
            ):
                apply_workload_event(
                    current,
                    {
                        "event_id": event["source_event_id"],
                        "occurred_at": event["occurred_at"],
                        "kind": event["kind"],
                        "source": event.get("source", {}),
                        "data": event["data"],
                    },
                    recorded_at=str(event.get("recorded_at") or event.get("at") or ""),
                )
                _remember_armed_trigger_evidence(
                    state,
                    current,
                    {
                        "event_id": event["source_event_id"],
                        "occurred_at": event["occurred_at"],
                        "kind": event["kind"],
                        "source": event.get("source", {}),
                        "data": event["data"],
                    },
                )
        if event.get("kind") == "resource.gpu_health_changed":
            data = event.get("data")
            recovered_health = data.get("gpu_health") if isinstance(data, dict) else None
            if isinstance(recovered_health, dict):
                state["gpu_health"] = copy.deepcopy(recovered_health)
        report_id = event.get("report_id")
        if isinstance(report_id, str):
            digest = event.get("report_digest")
            state.setdefault("report_acks", {})[report_id] = (
                digest if isinstance(digest, str) else None
            )
        if event.get("kind") == "allocation.started":
            state["draining"] = False
            state["drain_requested"] = False
            state["launches_paused"] = False
        elif event.get("kind") == "allocation.draining":
            state["draining"] = True
            state["drain_requested"] = True
            if isinstance(state.get("allocation"), dict):
                state["allocation"]["state"] = "draining"
        elif event.get("kind") == "allocation.launches_paused":
            state["launches_paused"] = True
        elif event.get("kind") == "allocation.launches_resumed":
            state["draining"] = False
            state["drain_requested"] = False
            state["launches_paused"] = False
            if isinstance(state.get("allocation"), dict):
                state["allocation"]["state"] = "running"
        if event.get("kind", "").startswith("evacuation."):
            data = event.get("data")
            if isinstance(data, dict):
                evacuation = data.get("evacuation")
                request_id = data.get("request_id")
                request = data.get("request")
                if isinstance(evacuation, dict):
                    evacuation_id = evacuation.get("request_id")
                    if isinstance(evacuation_id, str):
                        state.setdefault("evacuation_history", {})[
                            evacuation_id
                        ] = copy.deepcopy(evacuation)
                        # Cancellation events correlate by their separate
                        # command ID, while the operation remains keyed by
                        # its original evacuation ID.
                        if event.get("kind") in {
                            "evacuation.cancelled",
                            "evacuation.cancel_replayed",
                        }:
                            operation_id = data.get("evacuation_request_id")
                            if operation_id != evacuation_id:
                                operation_id = evacuation_id
                            current = state.get("evacuation")
                            if (
                                not isinstance(current, dict)
                                or current.get("request_id") == operation_id
                            ):
                                state["evacuation"] = copy.deepcopy(evacuation)
                            if isinstance(request, dict):
                                state.setdefault("evacuation_requests", {})[
                                    operation_id
                                ] = copy.deepcopy(request)
                            cancel_request_id = data.get("cancel_request_id", request_id)
                            cancel_request = data.get("cancel_request")
                            if (
                                isinstance(cancel_request_id, str)
                                and isinstance(cancel_request, dict)
                            ):
                                state.setdefault("evacuation_cancel_requests", {})[
                                    cancel_request_id
                                ] = copy.deepcopy(cancel_request)
                            if (
                                data.get("cleared_drain") is True
                                and (
                                    not isinstance(current, dict)
                                    or current.get("request_id") == operation_id
                                )
                            ):
                                state["draining"] = False
                                state["drain_requested"] = False
                                if isinstance(state.get("allocation"), dict):
                                    state["allocation"]["state"] = "running"
                        else:
                            state["evacuation"] = copy.deepcopy(evacuation)
                            if isinstance(request_id, str) and isinstance(request, dict):
                                state.setdefault("evacuation_requests", {})[request_id] = copy.deepcopy(
                                    request
                                )
                            if event.get("kind") == "evacuation.requested":
                                state["draining"] = True
                                state["drain_requested"] = True
                                if isinstance(state.get("allocation"), dict):
                                    state["allocation"]["state"] = "draining"
                            elif (
                                event.get("kind") == "evacuation.complete"
                                and evacuation.get("resume_after") is True
                            ):
                                state["draining"] = False
                                state["drain_requested"] = False
                                if isinstance(state.get("allocation"), dict):
                                    state["allocation"]["state"] = "running"
        state["last_seq"] = max(int(state.get("last_seq", 0)), int(event["seq"]))
    state["journal_offset"] = journal_offset
    state.setdefault("report_acks", {})
    state["next_queue_order"] = max(
        int(state.get("next_queue_order", 0)),
        max(
            (int(job.get("queue_order", 0)) for job in state["jobs"].values()),
            default=0,
        ),
    )
    state.setdefault("archived_jobs", 0)
    state.setdefault("archived_counts", {})
    if "archived_project_counts" not in state:
        # Every archive created before projects existed belongs to default.
        state["archived_project_counts"] = {
            DEFAULT_PROJECT: copy.deepcopy(state["archived_counts"])
        }
    state.setdefault("drain_requested", False)
    state.setdefault("launches_paused", False)
    state.setdefault("evacuation", None)
    state.setdefault("evacuation_requests", {})
    state.setdefault("evacuation_history", {})
    state.setdefault("evacuation_cancel_requests", {})
    return state
