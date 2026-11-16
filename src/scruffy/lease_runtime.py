"""Controller-side lease lifecycle: attach, observe, autoscale, and release.

Every lease transition is journaled with its complete lease image before the
controller acts on Slurm, so a crash leaves a record that names what may exist:
a ``requested`` lease is found again by its job name, and a ``releasing``
lease is cancelled again. Only two consecutive observations of a lease's
absence end it, and only an ended lease lets its remaining jobs be resolved.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Any

from .lease_agent import heartbeat_path
from .leases import (
    LIVE_LEASE_STATES,
    LeaseDemand,
    LeaseSettings,
    PoolConfig,
    format_time,
    is_live,
    lease_gpus,
    new_lease_id,
    parse_time,
    plan_leases,
    prune_ended_leases,
)
from .models import ACTIVE_JOB_STATES, ResourceRequest, job_project, normalize_project_id
from .runtime import Controller, stop_launcher
from .scheduler import project_gpu_usage, queue_priority_key
from .slurm import (
    SlurmJobObservation,
    build_lease_sbatch_argv,
    cancel_lease,
    completed_step,
    discover_slurm_allocation,
    lease_job,
    probe_lease_start,
    submit_lease,
    user_jobs,
)
from .slurm_runtime import lease_steps
from .state import emit, ensure_committed, job_lease_id
from .storage import atomic_write_json, utc_now

HEARTBEAT_INTERVAL_SECONDS = 30
PLAN_INTERVAL_SECONDS = 5
ABSENCES_TO_END = 2
CANCEL_RETRY_SECONDS = 60
UNCONFIRMED_SUBMISSION_SECONDS = 60
_RUNNING_SLURM_STATES = frozenset({"RUNNING", "COMPLETING", "SUSPENDED", "STOPPED"})


# ---------------------------------------------------------------------------
# Heartbeat for lease agents


def leases_in_use(controller: Controller) -> bool:
    settings = controller.lease_settings
    return bool(settings is not None and settings.pools) or any(
        is_live(lease) for lease in controller.state.get("leases", {}).values()
    )


def write_lease_heartbeat(controller: Controller) -> None:
    """Tell lease agents the controller is alive; they exit once it is not."""

    now = time.monotonic()
    if (
        controller.launcher != "slurm"
        or now - controller.last_lease_heartbeat < HEARTBEAT_INTERVAL_SECONDS
        or not leases_in_use(controller)
    ):
        return
    controller.last_lease_heartbeat = now
    try:
        atomic_write_json(
            heartbeat_path(controller.root),
            {
                "v": 1,
                "at": time.time(),
                "allocation_id": controller.allocation_id,
                "pid": os.getpid(),
            },
        )
    except OSError:
        # Agents only act on a heartbeat older than minutes; one missed write
        # is harmless and the next tick retries.
        controller.last_lease_heartbeat = 0.0


# ---------------------------------------------------------------------------
# Records and events


def _leases(controller: Controller) -> dict[str, dict[str, Any]]:
    return controller.state.setdefault("leases", {})


def _emit_lease(
    controller: Controller,
    kind: str,
    lease: dict[str, Any],
    **data: Any,
) -> None:
    emit(controller, kind, data={**data, "lease": copy.deepcopy(lease)})


def _notice(controller: Controller, lease_id: str | None, error: str, **data: Any) -> None:
    emit(
        controller,
        "notice",
        data={"source": "leases", "lease_id": lease_id, "error": error, **data},
    )


def busy_lease_ids(controller: Controller) -> set[str]:
    """Return leases that still hold at least one active job."""

    result = set()
    for job in controller.state["jobs"].values():
        if job["state"] in ACTIVE_JOB_STATES:
            lease_id = job_lease_id(job)
            if lease_id is not None:
                result.add(lease_id)
    return result


def find_lease(controller: Controller, target: object) -> dict[str, Any] | None:
    """Find a lease by lease ID or by its Slurm job ID (live leases first)."""

    if not isinstance(target, str) or not target:
        return None
    leases = _leases(controller)
    if target in leases:
        return leases[target]
    matches = [lease for lease in leases.values() if lease.get("slurm_job_id") == target]
    matches.sort(key=lambda lease: (not is_live(lease), str(lease.get("requested_at") or "")))
    return matches[0] if matches else None


def _end_lease(controller: Controller, lease: dict[str, Any], reason: str) -> None:
    """Record a confirmed end and resolve every job still bound to it."""

    lease_id = str(lease["lease_id"])
    lease.update(
        {
            "state": "ended",
            "ended_at": utc_now(),
            "reason": lease.get("release_reason") or reason,
        }
    )
    controller.lease_absences.pop(lease_id, None)
    controller.lease_idle_since.pop(lease_id, None)
    view = lease_steps(controller, lease_id)
    view.ended = True
    view.query_error = None
    _emit_lease(controller, "lease.ended", lease)
    _resolve_lease_jobs(controller, lease_id)
    prune_ended_leases(_leases(controller))


def _resolve_lease_jobs(controller: Controller, lease_id: str) -> None:
    """Decide the fate of jobs whose lease allocation is gone.

    A job whose exit was already observed finishes normally. A step that Slurm
    accounting reports COMPLETED keeps its own exit code. Anything else was
    killed with its allocation and becomes ``lost`` with ``lease_ended``,
    which admits an automatic retry where its recovery policy allows.
    """

    for job_id, running in controller.running.items():
        job = controller.state["jobs"][job_id]
        if job_lease_id(job) != lease_id:
            continue
        if (
            running.exit_seen_at is not None
            or running.final_state is not None
            or isinstance(job.get("pending_returncode"), int)
        ):
            stop_launcher(controller, running)
            continue
        step_id = job.get("slurm_step_id")
        result = None
        if isinstance(step_id, str) and step_id:
            try:
                result = completed_step(step_id)
            except Exception as exc:  # noqa: BLE001 - accounting is advisory here
                job["reconciliation_error"] = str(exc)
        if result is not None and result.state.split(maxsplit=1)[0] == "COMPLETED":
            job["slurm_state"] = result.state
            job["pending_returncode"] = result.returncode
        else:
            running.final_state = "lost"
            running.final_reason = "lease_ended"
            if running.process is None:
                job["pending_returncode"] = 1
        stop_launcher(controller, running)


# ---------------------------------------------------------------------------
# Commands


def _positive_option(command: dict[str, Any], key: str) -> int | None:
    value = command.get(key)
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def lease_command_outcome(kind: object, data: object) -> dict[str, Any]:
    """Summarize a lease command's journaled outcome for its receipt."""

    values = data if isinstance(data, dict) else {}
    lease = values.get("lease")
    lease_id = values.get("lease_id") or (
        lease.get("lease_id") if isinstance(lease, dict) else None
    )
    state = {
        "lease.attached": "attached",
        "lease.attach_ignored": "already_attached",
        "lease.releasing": "releasing",
        "lease.release_ignored": "ignored",
        "command.rejected": "rejected",
    }.get(str(kind), "unknown")
    outcome: dict[str, Any] = {"state": state}
    if lease_id is not None:
        outcome["lease_id"] = lease_id
    if isinstance(lease, dict) and lease.get("slurm_job_id") is not None:
        outcome["slurm_job_id"] = lease["slurm_job_id"]
    if values.get("reason") is not None:
        outcome["reason"] = values["reason"]
    return outcome


def _command_outcome(controller: Controller, kind: str, data: dict[str, Any]) -> dict[str, Any]:
    emit(controller, kind, data=data)
    return lease_command_outcome(kind, data)


def attach_lease(controller: Controller, command: dict[str, Any]) -> dict[str, Any]:
    """Adopt an allocation the user already holds as an attached lease.

    Return the command outcome recorded in its receipt.
    """

    request_id = command.get("request_id")
    try:
        if controller.launcher != "slurm":
            raise ValueError("leases require the Slurm launcher")
        slurm_job_id = str(command.get("slurm_job_id") or "")
        if not slurm_job_id.isascii() or not slurm_job_id.isdecimal():
            raise ValueError("lease.attach needs a numeric slurm_job_id")
        if slurm_job_id == controller.slurm_job_id:
            raise ValueError("the controller's own allocation is already its primary")
        existing = next(
            (
                lease
                for lease in _leases(controller).values()
                if is_live(lease) and lease.get("slurm_job_id") == slurm_job_id
            ),
            None,
        )
        if existing is not None:
            return _command_outcome(
                controller,
                "lease.attach_ignored",
                {
                    "request_id": request_id,
                    "lease_id": existing["lease_id"],
                    "reason": "already_attached",
                },
            )
        projects = command.get("projects")
        if projects is not None:
            if isinstance(projects, (str, bytes)) or not isinstance(projects, list) or not projects:
                raise ValueError("projects must be a non-empty list")
            projects = [normalize_project_id(project) for project in projects]
        managed, incarnation = discover_slurm_allocation(
            slurm_job_id=slurm_job_id,
            gpus_per_node=_positive_option(command, "gpus_per_node"),
            cpus_per_node=_positive_option(command, "cpus_per_node"),
            memory_gb_per_node=_positive_option(command, "memory_gb_per_node"),
        )
        observed = lease_job(slurm_job_id)
        if observed.state not in _RUNNING_SLURM_STATES:
            raise ValueError(f"allocation {slurm_job_id} is {observed.state}, not running")
    except Exception as exc:  # noqa: BLE001 - every failure is the command's outcome
        return _command_outcome(
            controller, "command.rejected", {"request_id": request_id, "reason": str(exc)}
        )
    now = utc_now()
    lease: dict[str, Any] = {
        "lease_id": new_lease_id(),
        "kind": "attached",
        "pool": None,
        "state": "running",
        "slurm_job_id": slurm_job_id,
        "requested_at": now,
        "started_at": now,
        "end_at": format_time(observed.end_at) if observed.end_at else None,
        "inventory": [node.to_dict() for node in managed.values()],
        "incarnation": incarnation.to_dict(),
        "request_id": request_id,
    }
    if projects is not None:
        lease["projects"] = projects
    _leases(controller)[lease["lease_id"]] = lease
    controller.lease_idle_since[lease["lease_id"]] = time.time()
    _emit_lease(controller, "lease.attached", lease, request_id=request_id)
    return lease_command_outcome("lease.attached", {"lease": lease})


def release_lease(controller: Controller, command: dict[str, Any]) -> dict[str, Any]:
    """Stop placing work on a lease; give it back once its jobs finish.

    An elastic lease is cancelled in Slurm. An attached lease is only
    detached: Scruffy never cancels an allocation it did not create.
    """

    request_id = command.get("request_id")
    lease = find_lease(controller, command.get("lease_id"))
    if lease is None:
        return _command_outcome(
            controller, "command.rejected", {"request_id": request_id, "reason": "unknown_lease"}
        )
    if lease.get("state") in {"ended", "releasing"}:
        return _command_outcome(
            controller,
            "lease.release_ignored",
            {
                "request_id": request_id,
                "lease_id": lease["lease_id"],
                "reason": f"lease_is_{lease['state']}",
            },
        )
    _mark_releasing(controller, lease, "operator", request_id=request_id)
    return lease_command_outcome("lease.releasing", {"lease": lease})


def _mark_releasing(
    controller: Controller,
    lease: dict[str, Any],
    reason: str,
    *,
    request_id: object = None,
) -> None:
    lease["state"] = "releasing"
    lease["release_reason"] = reason
    lease["release_requested_at"] = utc_now()
    data: dict[str, Any] = {}
    if request_id is not None:
        data["request_id"] = request_id
    _emit_lease(controller, "lease.releasing", lease, **data)


# ---------------------------------------------------------------------------
# Observation


def observe_leases(controller: Controller) -> None:
    """Reconcile lease records with ``squeue`` at the configured interval."""

    if controller.launcher != "slurm" or not leases_in_use(controller):
        return
    settings = controller.lease_settings
    interval = settings.observe_interval_seconds if settings is not None else 15.0
    now = time.monotonic()
    if now - controller.last_lease_observation < interval:
        return
    controller.last_lease_observation = now
    try:
        observed = user_jobs()
    except Exception as exc:  # noqa: BLE001 - uncertainty keeps every lease
        error = str(exc)
        if controller.lease_observation_error != error:
            controller.lease_observation_error = error
            _notice(controller, None, error)
        return
    if controller.lease_observation_error is not None:
        controller.lease_observation_error = None
        _notice(controller, None, "", resolved=True)
    controller.user_gpus_in_use = sum(job.gpus for job in observed if not job.terminal)
    controller.lease_observed_at = time.time()
    by_id = {job.job_id: job for job in observed}
    by_name: dict[str, list[SlurmJobObservation]] = {}
    for job in observed:
        by_name.setdefault(job.name, []).append(job)
    for lease in [lease for lease in _leases(controller).values() if is_live(lease)]:
        _observe_lease(controller, lease, by_id, by_name)


def _observe_lease(
    controller: Controller,
    lease: dict[str, Any],
    by_id: dict[str, SlurmJobObservation],
    by_name: dict[str, list[SlurmJobObservation]],
) -> None:
    lease_id = str(lease["lease_id"])
    if not lease.get("slurm_job_id"):
        matches = [job for job in by_name.get(lease_id, []) if not job.terminal]
        if not matches:
            requested = parse_time(lease.get("requested_at")) or 0.0
            absences = controller.lease_absences.get(lease_id, 0) + 1
            controller.lease_absences[lease_id] = absences
            if (
                absences >= ABSENCES_TO_END
                and time.time() - requested >= UNCONFIRMED_SUBMISSION_SECONDS
            ):
                _end_lease(controller, lease, "submission_unconfirmed")
            return
        # The submission happened even though sbatch's answer was lost.
        lease["slurm_job_id"] = matches[0].job_id
        lease["submitted_at"] = lease.get("submitted_at") or utc_now()
        if lease["state"] == "requested":
            lease["state"] = "pending"
        _emit_lease(controller, "lease.submitted", lease, recovered=True)
    observed = by_id.get(str(lease["slurm_job_id"]))
    if observed is None or observed.terminal:
        absences = controller.lease_absences.get(lease_id, 0) + 1
        controller.lease_absences[lease_id] = absences
        if absences >= ABSENCES_TO_END:
            _end_lease(
                controller,
                lease,
                f"slurm_{observed.state.lower()}" if observed is not None else "allocation_gone",
            )
        return
    controller.lease_absences.pop(lease_id, None)
    if observed.state in _RUNNING_SLURM_STATES:
        if lease["state"] in {"requested", "pending"}:
            _start_lease(controller, lease, observed)
        elif lease["state"] in {"running", "releasing"}:
            _check_running_lease(controller, lease, observed)
        return
    # Pending (or configuring). A started lease never goes back to pending:
    # elastic leases never requeue, and a requeued attached lease lost its steps.
    if lease["state"] in {"running", "releasing"} and lease.get("inventory"):
        _end_lease(controller, lease, "requeued")
        return
    estimated = format_time(observed.start_at) if observed.start_at else None
    lease["estimated_start_at"] = estimated
    if lease["state"] == "requested" or lease.get("pending_reason") != observed.reason:
        if lease["state"] == "requested":
            lease["state"] = "pending"
        lease["pending_reason"] = observed.reason
        _emit_lease(controller, "lease.pending", lease)


def _start_lease(
    controller: Controller, lease: dict[str, Any], observed: SlurmJobObservation
) -> None:
    lease_id = str(lease["lease_id"])
    try:
        managed, incarnation = discover_slurm_allocation(slurm_job_id=str(lease["slurm_job_id"]))
    except Exception as exc:  # noqa: BLE001 - an unusable lease is given back
        lease["discovery_error"] = str(exc)
        _mark_releasing(controller, lease, "discovery_failed")
        return
    lease.update(
        {
            "state": "running",
            "started_at": utc_now(),
            "end_at": format_time(observed.end_at) if observed.end_at else None,
            "inventory": [node.to_dict() for node in managed.values()],
            "incarnation": incarnation.to_dict(),
        }
    )
    lease.pop("pending_reason", None)
    lease.pop("estimated_start_at", None)
    controller.lease_idle_since[lease_id] = time.time()
    view = lease_steps(controller, lease_id)
    view.slurm_job_id = str(lease["slurm_job_id"])
    _emit_lease(controller, "lease.running", lease)


def _check_running_lease(
    controller: Controller, lease: dict[str, Any], observed: SlurmJobObservation
) -> None:
    incarnation = lease.get("incarnation")
    restart_count = incarnation.get("restart_count") if isinstance(incarnation, dict) else None
    if (
        observed.restart_count is not None
        and restart_count is not None
        and observed.restart_count != restart_count
    ):
        _end_lease(controller, lease, "incarnation_changed")
        return
    end_at = format_time(observed.end_at) if observed.end_at else None
    if end_at != lease.get("end_at"):
        lease["end_at"] = end_at
        _emit_lease(controller, "lease.updated", lease)


# ---------------------------------------------------------------------------
# Release


def advance_releases(controller: Controller) -> None:
    """Give back releasing leases once no job is left on them."""

    busy = busy_lease_ids(controller)
    now = time.time()
    for lease in list(_leases(controller).values()):
        if lease.get("state") != "releasing" or lease["lease_id"] in busy:
            continue
        if lease.get("kind") == "attached":
            _end_lease(controller, lease, "detached")
            continue
        slurm_job_id = lease.get("slurm_job_id")
        if not isinstance(slurm_job_id, str) or not slurm_job_id:
            continue
        attempted = parse_time(lease.get("cancel_requested_at"))
        if attempted is not None and now - attempted < CANCEL_RETRY_SECONDS:
            continue
        # The releasing transition is durable before Slurm is asked.
        ensure_committed(controller)
        lease["cancel_requested_at"] = utc_now()
        try:
            cancel_lease(slurm_job_id)
        except Exception as exc:  # noqa: BLE001 - retried after CANCEL_RETRY_SECONDS
            _notice(controller, str(lease["lease_id"]), str(exc), action="cancel")


# ---------------------------------------------------------------------------
# Autoscaling


def _pool_status(controller: Controller) -> dict[str, dict[str, Any]]:
    return controller.state.setdefault("lease_pools", {})


def _backoff_pool(controller: Controller, pool: PoolConfig, reason: str) -> None:
    status = {
        "backoff_until": format_time(time.time() + pool.backoff_seconds),
        "reason": reason,
        "at": utc_now(),
    }
    _pool_status(controller)[pool.name] = status
    emit(controller, "lease.pool_backoff", data={"pool": pool.name, "pool_status": status})


def _lease_demand(controller: Controller) -> list[LeaseDemand]:
    jobs = controller.state["jobs"]
    usage = project_gpu_usage(jobs.values())
    queued = sorted(
        (job for job in jobs.values() if job["state"] == "queued"),
        key=lambda job: queue_priority_key(job, usage),
    )
    return [
        LeaseDemand(job["id"], job_project(job), ResourceRequest.from_dict(job["request"]))
        for job in queued
    ]


def _committed_gpus(controller: Controller, settings: LeaseSettings) -> int | None:
    """Return the user's GPU footprint, counting leases newer than ``squeue``."""

    if controller.user_gpus_in_use is None:
        return None
    observed_at = controller.lease_observed_at
    newer = sum(
        lease_gpus(lease, settings)
        for lease in _leases(controller).values()
        if lease.get("kind") == "elastic"
        and lease.get("state") in LIVE_LEASE_STATES
        and (parse_time(lease.get("requested_at")) or 0.0) > observed_at
    )
    return controller.user_gpus_in_use + newer


def plan_and_apply_leases(controller: Controller) -> None:
    """Request, withdraw, or release elastic leases from the configured pools."""

    settings = controller.lease_settings
    if controller.launcher != "slurm" or settings is None or not settings.pools:
        return
    now = time.monotonic()
    if now - controller.last_lease_plan < PLAN_INTERVAL_SECONDS:
        return
    controller.last_lease_plan = now
    wall = time.time()
    busy = busy_lease_ids(controller)
    for lease_id, lease in _leases(controller).items():
        if lease.get("state") != "running" or lease_id in busy:
            controller.lease_idle_since.pop(lease_id, None)
        else:
            controller.lease_idle_since.setdefault(lease_id, wall)
    allow_growth = (
        not controller.stopping
        and not controller.state.get("draining")
        and not controller.state.get("launches_paused")
        and controller.last_lease_observation > 0
        and controller.lease_observation_error is None
    )
    plan = plan_leases(
        settings,
        _leases(controller),
        _lease_demand(controller) if allow_growth else [],
        busy_lease_ids=busy,
        idle_since=controller.lease_idle_since,
        pool_status=_pool_status(controller),
        now=wall,
        allow_growth=allow_growth,
        committed_gpus=_committed_gpus(controller, settings),
    )
    for name, reason in plan.backoff:
        pool = settings.pool(name)
        if pool is not None:
            _backoff_pool(controller, pool, reason)
    for lease_id, reason in (*plan.cancel, *plan.release):
        lease = _leases(controller).get(lease_id)
        if lease is not None and lease.get("state") in {"requested", "pending", "running"}:
            _mark_releasing(controller, lease, reason)
    for name in plan.submit:
        pool = settings.pool(name)
        if pool is not None and not _submit_lease(controller, settings, pool):
            break


def _submit_lease(controller: Controller, settings: LeaseSettings, pool: PoolConfig) -> bool:
    """Journal a lease request, then submit it; return false to stop this round."""

    lease_id = new_lease_id()
    argv = build_lease_sbatch_argv(
        lease_id=lease_id,
        root=controller.root,
        nodes=pool.nodes,
        gpus_per_node=pool.gpus_per_node,
        cpus_per_node=pool.cpus_per_node,
        memory_gb_per_node=pool.memory_gb_per_node,
        time_limit_seconds=pool.time_limit_seconds,
        heartbeat_timeout_seconds=settings.heartbeat_timeout_seconds,
        partition=pool.partition,
        qos=pool.qos,
        account=pool.account,
        constraint=pool.constraint,
        gpu_type=pool.gpu_type,
        extra_args=pool.sbatch_args,
    )
    if pool.start_within_seconds is not None:
        try:
            estimated = probe_lease_start(argv)
        except Exception as exc:  # noqa: BLE001 - a failed probe backs the pool off
            _backoff_pool(controller, pool, f"probe_failed:{exc}")
            return False
        if estimated is not None and estimated - time.time() > pool.start_within_seconds:
            _backoff_pool(controller, pool, "start_too_late")
            return True
    lease: dict[str, Any] = {
        "lease_id": lease_id,
        "kind": "elastic",
        "pool": pool.name,
        "state": "requested",
        "slurm_job_id": None,
        "requested_at": utc_now(),
        "shape": pool.shape(),
    }
    _leases(controller)[lease_id] = lease
    _emit_lease(controller, "lease.requested", lease)
    # The request is durable before Slurm can create the allocation, so an
    # uncertain sbatch is found again by its job name after a crash.
    ensure_committed(controller)
    try:
        # Slurm cannot create the agent's output file in a missing directory.
        (controller.root / "leases" / "logs").mkdir(parents=True, exist_ok=True)
        slurm_job_id = submit_lease(argv)
    except Exception as exc:  # noqa: BLE001 - observation resolves the uncertainty
        lease["submit_error"] = str(exc)
        _backoff_pool(controller, pool, f"submit_failed:{exc}")
        return False
    lease.update({"slurm_job_id": slurm_job_id, "state": "pending", "submitted_at": utc_now()})
    lease_steps(controller, lease_id).slurm_job_id = slurm_job_id
    _emit_lease(controller, "lease.submitted", lease)
    return True
