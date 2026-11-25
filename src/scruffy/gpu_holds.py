"""Enforce operator GPU quarantines by holding the physical device.

Inside an allocation Slurm chooses which physical GPU a count-bound step
receives, and no step option names one device: ``--gpu-bind=map_gpu`` only
orders the devices a step was already given, GRES selection cannot name a
device per step, and the step's cgroup follows the GRES Slurm allocated. The
controller therefore enforces an operator quarantine with one-GPU sentinel
steps that check the UUID Slurm bound to them (the worker's identity probe).
A sentinel on the target holds it until the quarantine is cleared; one on
another device parks, so that the next sentinel is given a different device,
and is released once the target is held (or after ``PARK_SECONDS``). The
node's other GPUs stay schedulable; only its free GPUs are withheld, and only
while a search is placing sentinels.
"""

from __future__ import annotations

import copy
import sys
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from ._compat import UTC
from .health import HOLD_ENFORCEMENT, unavailable_gpu_ids
from .lifecycle import request_cancellation
from .models import (
    ACTIVE_JOB_STATES,
    TERMINAL_JOB_STATES,
    ResourceRequest,
    is_system_job,
)
from .provenance import write_request_record
from .runtime import Controller
from .scheduler import available_resources
from .state import active_assignments, emit
from .storage import create_job_id, job_identity_digest, record_request_receipt, utc_now
from .submissions import job_from_spec

SYSTEM_PROJECT = "scruffy-system"
GPU_HOLD_SYSTEM = "gpu_hold"
# A parked sentinel releases its device on its own after this long, so a
# controller that stops mid-search cannot strand healthy GPUs.
PARK_SECONDS = 600
# A sentinel not identified this long after submission counts as a failure.
SENTINEL_IDENTIFY_SECONDS = 300
MAX_HOLD_FAILURES = 3
MAX_SEARCH_LAUNCHES = 64
# A hold that could not account for every GPU on its node searches again
# after this long.
RETRY_SECONDS = 60
HOLD_STATES = frozenset({"searching", "holding", "waiting", "failed"})
_LIVE_STATES = ACTIVE_JOB_STATES | {"queued"}


def hold_key(node: str, uuid: str) -> str:
    return f"{node}/{uuid}"


def hold_enforced(controller: Controller, node: str) -> bool:
    """Return whether an operator quarantine on ``node`` is held by a sentinel."""

    health = controller.state.get("gpu_health")
    return (
        controller.launcher == "slurm"
        and isinstance(health, dict)
        and health.get("isolation") == "gpu"
        and any(item.name == node for item in controller.inventory)
    )


def _holds(controller: Controller) -> dict[str, dict[str, Any]]:
    return controller.state["gpu_health"].setdefault("holds", {})


def arm_gpu_hold(controller: Controller, node: str, uuid: str, *, at: str) -> None:
    """Start (or restart a failed) hold; an active hold is left unchanged."""

    holds = _holds(controller)
    key = hold_key(node, uuid)
    current = holds.get(key)
    if isinstance(current, dict) and current.get("state") != "failed":
        return
    health = controller.state["gpu_health"]
    generation = int(health.get("hold_generation", 0)) + 1
    health["hold_generation"] = generation
    if isinstance(current, dict):
        _cancel(controller, current, current.get("sentinels", {}))
    holds[key] = {
        "node": node,
        "uuid": uuid,
        "generation": generation,
        "state": "searching",
        "job_id": None,
        "blocked_by": [],
        "sentinels": {},
        "next_attempt": 1,
        "launches": 0,
        "failures": 0,
        "detail": None,
        "changed_at": at,
    }


def release_gpu_hold(controller: Controller, node: str, uuid: str) -> bool:
    """Drop a hold and cancel its sentinels; return whether one existed."""

    hold = _holds(controller).pop(hold_key(node, uuid), None)
    if not isinstance(hold, dict):
        return False
    _cancel(controller, hold, _sentinel_ids(controller, hold))
    return True


def _sentinel_ids(controller: Controller, hold: dict[str, Any]) -> set[str]:
    key = hold_key(hold["node"], hold["uuid"])
    return set(hold.get("sentinels", {})) | {
        job["id"]
        for job in controller.state["jobs"].values()
        if job.get("gpu_hold", {}).get("key") == key
        and job.get("state") in _LIVE_STATES
    }


def _cancel(
    controller: Controller, hold: dict[str, Any], job_ids: Iterable[str]
) -> None:
    for job_id in sorted(job_ids):
        hold.get("sentinels", {}).pop(job_id, None)
        job = controller.state["jobs"].get(job_id)
        if job is not None and job.get("state") in _LIVE_STATES - {"cancelling"}:
            request_cancellation(controller, job)


def _uuids(job: dict[str, Any]) -> set[str]:
    devices = job.get("gpu_devices")
    return {
        str(device.get("uuid")).lower()
        for device in (devices if isinstance(devices, list) else ())
        if isinstance(device, dict) and isinstance(device.get("uuid"), str)
    }


def _age_seconds(value: object) -> float:
    try:
        return (datetime.now(UTC) - datetime.fromisoformat(str(value))).total_seconds()
    except ValueError:
        return 0.0


def _on_node(job: dict[str, Any], node: str) -> bool:
    """Return whether a job holds GPUs on ``node`` in the primary allocation."""

    assignment = job.get("assignment")
    if not isinstance(assignment, dict) or assignment.get("lease_id") is not None:
        return False
    reservations = assignment.get("reservations")
    return any(
        isinstance(item, dict) and item.get("node") == node and item.get("gpu_ids")
        for item in (reservations or ())
    )


def _free_gpus(controller: Controller, node: str) -> int:
    health = controller.state["gpu_health"]
    for item in available_resources(
        controller.inventory,
        active_assignments(controller.state),
        unavailable_gpu_ids(health, controller.inventory),
    ):
        if item.name == node:
            return len(item.gpu_ids)
    return 0


def _launch_sentinel(controller: Controller, hold: dict[str, Any]) -> None:
    attempt = int(hold["next_attempt"])
    hold["next_attempt"] = attempt + 1
    hold["launches"] = int(hold["launches"]) + 1
    request_id = (
        f"gpu-hold/{hold['node']}/{hold['uuid']}/{hold['generation']}/{attempt}"
    )
    job_id = create_job_id(request_id, project_id=SYSTEM_PROJECT)
    spec = {
        "v": 1,
        "job_id": job_id,
        "request_id": request_id,
        "project_id": SYSTEM_PROJECT,
        "name": f"gpu-hold {hold['node']} {hold['uuid'][4:12]}",
        "submitted_at": utc_now(),
        "argv": [
            sys.executable,
            "-m",
            "scruffy.gpu_sentinel",
            "--target",
            hold["uuid"],
            "--park-seconds",
            str(PARK_SECONDS),
        ],
        "cwd": str(controller.root.resolve()),
        "env": {},
        "resources": ResourceRequest(
            nodes=1,
            gpus_per_node=1,
            cpus_per_node=1,
            memory_gb_per_node=1,
            allowed_nodes=(hold["node"],),
        ).to_dict(),
    }
    jobs = controller.state["jobs"]
    queue_order = 1 + max(
        int(controller.state.get("next_queue_order", 0)),
        max((int(job.get("queue_order", 0)) for job in jobs.values()), default=0),
    )
    job = job_from_spec(spec, queue_order)
    job.update(
        {
            "system": GPU_HOLD_SYSTEM,
            "gpu_hold": {
                "key": hold_key(hold["node"], hold["uuid"]),
                "generation": hold["generation"],
                "attempt": attempt,
            },
        }
    )
    controller.state["next_queue_order"] = queue_order
    write_request_record(controller.root, job)
    jobs[job_id] = job
    emit(controller, "job.queued", job=job)
    record_request_receipt(controller.root, job_id, job_identity_digest(spec))
    hold["sentinels"][job_id] = "pending"


def _set_state(
    hold: dict[str, Any],
    state: str,
    *,
    job_id: str | None = None,
    blocked_by: Iterable[str] = (),
    detail: str | None = None,
) -> None:
    hold.update(
        {
            "state": state,
            "job_id": job_id,
            "blocked_by": sorted(blocked_by),
            "detail": detail,
        }
    )


def _track_sentinels(controller: Controller, hold: dict[str, Any]) -> None:
    """Refresh each sentinel's status, adopting any a crash left untracked."""

    jobs = controller.state["jobs"]
    key = hold_key(hold["node"], hold["uuid"])
    sentinels = hold.setdefault("sentinels", {})
    for job in jobs.values():
        binding = job.get("gpu_hold")
        if (
            isinstance(binding, dict)
            and binding.get("key") == key
            and job.get("state") in _LIVE_STATES
            and job["id"] not in sentinels
        ):
            if binding.get("generation") == hold["generation"]:
                sentinels[job["id"]] = "pending"
            elif job.get("state") != "cancelling":
                request_cancellation(controller, job)
    target = hold["uuid"].lower()
    for job_id, status in sorted(sentinels.items()):
        job = jobs.get(job_id)
        if job is None or job.get("state") in TERMINAL_JOB_STATES:
            del sentinels[job_id]
            if status == "pending":
                hold["failures"] = int(hold["failures"]) + 1
                hold["detail"] = (
                    f"sentinel {job_id} ended unidentified"
                    f" ({(job or {}).get('reason') or 'unknown'})"
                )
            continue
        if job.get("state") == "cancelling":
            continue
        if "gpu_devices" not in job:
            if (
                isinstance(job.get("started_at"), str)
                and _age_seconds(job["started_at"]) > SENTINEL_IDENTIFY_SECONDS
            ):
                hold["failures"] = int(hold["failures"]) + 1
                hold["detail"] = f"sentinel {job_id} was not identified in time"
                _cancel(controller, hold, [job_id])
            else:
                sentinels[job_id] = "pending"
        elif target in _uuids(job):
            sentinels[job_id] = "holding"
        elif _uuids(job) and not job.get("gpu_identity_error"):
            sentinels[job_id] = "parked"
        else:
            hold["failures"] = int(hold["failures"]) + 1
            hold["detail"] = (
                f"sentinel {job_id} could not identify its GPU:"
                f" {job.get('gpu_identity_error') or 'no device'}"
            )
            _cancel(controller, hold, [job_id])


def _advance_hold(controller: Controller, hold: dict[str, Any], busy: set[str]) -> None:
    node = hold["node"]
    if hold["state"] == "failed":
        return
    if not any(item.name == node for item in controller.inventory):
        _cancel(controller, hold, list(hold.get("sentinels", {})))
        _set_state(hold, "waiting", detail="node is not in this allocation")
        return
    _track_sentinels(controller, hold)
    sentinels = hold["sentinels"]
    holders = sorted(job_id for job_id, status in sentinels.items() if status == "holding")
    if holders:
        _cancel(controller, hold, [job_id for job_id in sentinels if job_id != holders[0]])
        hold["launches"] = 0
        hold["failures"] = 0
        _set_state(hold, "holding", job_id=holders[0])
        return
    if int(hold["failures"]) >= MAX_HOLD_FAILURES:
        _cancel(controller, hold, list(sentinels))
        _set_state(hold, "failed", detail=hold.get("detail"))
        return
    jobs = controller.state["jobs"]
    target = hold["uuid"].lower()
    on_node = [
        job
        for job in jobs.values()
        if job.get("state") in ACTIVE_JOB_STATES and _on_node(job, node)
    ]
    others = [job for job in on_node if not is_system_job(job)]
    users = sorted(job["id"] for job in others if target in _uuids(job))
    if users:
        _cancel(controller, hold, list(sentinels))
        hold["launches"] = 0
        _set_state(hold, "waiting", blocked_by=users, detail="held by a running job")
        return
    if hold["state"] == "waiting":
        if hold.get("blocked_by"):
            # Search again only once a job that may hold the target has ended
            # or shown that it does not hold it.
            unresolved = {job["id"] for job in others if not _uuids(job)}
            if set(hold["blocked_by"]) <= unresolved:
                return
        elif _age_seconds(hold.get("changed_at")) < RETRY_SECONDS:
            return
    if any(status == "pending" for status in sentinels.values()):
        _set_state(hold, "searching", detail=hold.get("detail"))
        busy.add(node)
        return
    free = _free_gpus(controller, node)
    if free and node not in busy:
        if int(hold["launches"]) + free > MAX_SEARCH_LAUNCHES:
            _cancel(controller, hold, list(sentinels))
            _set_state(hold, "failed", detail="search launched too many sentinels")
            return
        for _ in range(free):
            _launch_sentinel(controller, hold)
        _set_state(hold, "searching")
        busy.add(node)
        return
    if node in busy:
        _set_state(hold, "searching", detail="another hold on this node is searching")
        return
    if any(
        is_system_job(job) and (job.get("state") == "cancelling" or target in _uuids(job))
        for job in on_node
    ):
        # A released sentinel's device is not free yet, or another hold's
        # sentinel is parked on the target until that hold settles.
        return
    # No GPU is free and every sentinel is parked: the target is not free.
    _cancel(controller, hold, list(sentinels))
    hold["launches"] = 0
    suspects = sorted(job["id"] for job in others if not _uuids(job))
    identified = set().union(*(_uuids(job) for job in on_node))
    capacity = next(len(item.gpu_ids) for item in controller.inventory if item.name == node)
    if suspects:
        _set_state(
            hold,
            "waiting",
            blocked_by=suspects,
            detail="the target may be under a job whose GPUs are not identified",
        )
    elif len(identified) < capacity:
        _set_state(
            hold,
            "waiting",
            detail=(
                f"{capacity - len(identified)} GPUs on the node are unavailable;"
                f" searching again every {RETRY_SECONDS} s"
            ),
        )
    else:
        _set_state(
            hold, "failed", detail="the target is not among this allocation's GPUs on the node"
        )


def advance_gpu_holds(controller: Controller) -> None:
    """Advance every hold one step and journal any change."""

    health = controller.state.get("gpu_health")
    holds = health.get("holds") if isinstance(health, dict) else None
    if not holds or controller.launcher != "slurm":
        return
    before = copy.deepcopy(holds)
    busy: set[str] = set()
    at = utc_now()
    transitions = []
    for key in sorted(holds):
        hold = holds[key]
        previous = hold.get("state")
        _advance_hold(controller, hold, busy)
        if hold != before.get(key):
            hold["changed_at"] = at
        if hold.get("state") != previous:
            transitions.append(
                {
                    "node": hold["node"],
                    "uuid": hold["uuid"],
                    "hold": hold["state"],
                    "from": previous,
                    "to": hold["state"],
                    "job_id": hold.get("job_id"),
                    "reasons": [hold["detail"]] if hold.get("detail") else [],
                    "at": at,
                }
            )
    if holds != before:
        emit(
            controller,
            "resource.gpu_health_changed",
            data={"transitions": transitions, "gpu_health": copy.deepcopy(health)},
        )


def is_hold_device(device: object) -> bool:
    return isinstance(device, dict) and device.get("enforcement") == HOLD_ENFORCEMENT


__all__ = [
    "GPU_HOLD_SYSTEM",
    "PARK_SECONDS",
    "SYSTEM_PROJECT",
    "advance_gpu_holds",
    "arm_gpu_hold",
    "hold_enforced",
    "release_gpu_hold",
]
