"""Queue-state transformations and durable event emission."""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    ACTIVE_JOB_STATES,
    DEFAULT_PROJECT,
    TERMINAL_JOB_STATES,
    Assignment,
    NodeInventory,
    ResourceRequest,
    job_project,
    normalize_project_id,
)
from .protocol import EVENT_KINDS
from .runtime import Controller
from .scheduler import available_resources
from .storage import (
    StorageError,
    activate_journal_generation,
    append_event,
    archive_terminal_job,
    create_journal_generation,
    job_identity_digest,
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


def apply_workload_event(
    job: dict[str, Any], event: dict[str, Any], *, recorded_at: str
) -> None:
    """Project one validated producer event onto a job's current workload view.

    Producer reports are deliberately unable to alter lifecycle or placement
    fields.  The complete event remains in the journal; this projection only
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
            }
        )
        artifacts.sort(
            key=lambda item: _event_key(item["occurred_at"], item["event_id"])
        )
        workload["latest_artifacts"] = artifacts[-8:]
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


def job_from_spec(spec: dict[str, Any], queue_order: int) -> dict[str, Any]:
    """Validate a client request and create its controller-owned job image."""

    request = ResourceRequest.from_dict(spec["resources"])
    argv = spec.get("argv")
    if not isinstance(argv, list) or not argv or not all(
        isinstance(item, str) and item for item in argv
    ):
        raise ValueError("argv must be a non-empty array of strings")
    environment = spec.get("env", {})
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ValueError("env must map strings to strings")
    cwd = Path(str(spec["cwd"]))
    if not cwd.is_absolute():
        raise ValueError("cwd must be absolute")
    project_id = normalize_project_id(spec.get("project_id"))
    job = {
        "id": str(spec["job_id"]),
        "project_id": project_id,
        "name": str(spec["name"]),
        "state": "queued",
        "submitted_at": str(spec["submitted_at"]),
        "queue_order": queue_order,
        "request_digest": job_identity_digest(spec),
        "argv": argv,
        "cwd": str(cwd),
        "env": environment,
        "request": request.to_dict(),
        "assignment": None,
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
        "signal": None,
        "reason": None,
        "error": None,
    }
    workflow_id = spec.get("workflow_id")
    task_id = spec.get("task_id")
    needs = spec.get("needs", [])
    if workflow_id is not None or task_id is not None or needs:
        if not isinstance(needs, list):
            raise ValueError("needs must be a JSON array")
        job.update(
            {
                "workflow_id": workflow_id,
                "task_id": task_id,
                "needs": copy.deepcopy(needs),
                "blockers": [],
                "dependency_gate_passed": False,
            }
        )
    return job


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
    free_by_node = {
        item.name: item for item in available_resources(inventory, assignments)
    }
    nodes = {
        item.name: {
            "capacity": item.to_dict(),
            "free": {
                "gpu_ids": list(free_by_node[item.name].gpu_ids),
                "cpus": free_by_node[item.name].cpus,
                "memory_gb": free_by_node[item.name].memory_gb,
            },
            "assignments": {},
        }
        for item in inventory
    }
    for assignment in assignments:
        for reservation in assignment.reservations:
            node = nodes[reservation.node]
            node["assignments"][assignment.job_id] = reservation.to_dict()
    state["nodes"] = nodes
    state["updated_at"] = utc_now()


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
    return event


def commit_snapshot(controller: Controller) -> None:
    """Durably commit prior events, then publish one cumulative state image."""

    sync_file(controller.journal)
    refresh_nodes(controller.state, controller.inventory)
    write_state(controller.root, controller.state)


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
                "jobs": {},
                "report_acks": {},
                "report_ack_v": 1,
                "next_queue_order": 0,
                "archived_jobs": 0,
                "archived_counts": {},
                "archived_project_counts": {},
                "draining": False,
                "drain_requested": False,
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
        if allocation_id and (
            rebuilding or event.get("kind") == "allocation.started"
        ):
            state["allocation"] = {"id": str(allocation_id), "state": "recovered"}
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
                        "data": event["data"],
                    },
                    recorded_at=str(event.get("recorded_at") or event.get("at") or ""),
                )
        report_id = event.get("report_id")
        if isinstance(report_id, str):
            digest = event.get("report_digest")
            state.setdefault("report_acks", {})[report_id] = (
                digest if isinstance(digest, str) else None
            )
        if event.get("kind") == "allocation.started":
            state["draining"] = False
            state["drain_requested"] = False
        elif event.get("kind") == "allocation.draining":
            state["draining"] = True
            state["drain_requested"] = True
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
    return state
