"""Fail-closed reconciliation between local ``srun`` clients and Slurm steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .runtime import Controller, LeaseSteps, RunningProcess
from .slurm import SlurmStep, cancel_step, live_steps
from .state import emit, job_lease_id
from .storage import utc_now


@dataclass(frozen=True, slots=True)
class StepSnapshot:
    """The live-step view that proves one job's step present or absent."""

    slurm_job_id: str
    steps: tuple[SlurmStep, ...]
    snapshot_at: float
    query_error: str | None
    ended: bool = False


def job_slurm_job_id(controller: Controller, job: dict[str, Any]) -> str:
    """Return the Slurm allocation a job's step runs in."""

    lease_id = job_lease_id(job)
    if lease_id is None:
        return controller.slurm_job_id or ""
    lease = controller.state.get("leases", {}).get(lease_id)
    slurm_job_id = lease.get("slurm_job_id") if isinstance(lease, dict) else None
    if not isinstance(slurm_job_id, str) or not slurm_job_id:
        raise RuntimeError(f"lease {lease_id!r} has no Slurm allocation")
    return slurm_job_id


def lease_steps(controller: Controller, lease_id: str) -> LeaseSteps:
    """Return (creating) the live-step view of one lease."""

    view = controller.lease_steps.get(lease_id)
    if view is None:
        lease = controller.state.get("leases", {}).get(lease_id)
        slurm_job_id = lease.get("slurm_job_id") if isinstance(lease, dict) else None
        view = LeaseSteps(str(slurm_job_id or ""))
        controller.lease_steps[lease_id] = view
    return view


def step_snapshot(controller: Controller, job: dict[str, Any]) -> StepSnapshot:
    lease_id = job_lease_id(job)
    if lease_id is None:
        return StepSnapshot(
            controller.slurm_job_id or "",
            controller.slurm_steps,
            controller.slurm_snapshot_at,
            controller.slurm_query_error,
        )
    view = lease_steps(controller, lease_id)
    return StepSnapshot(
        view.slurm_job_id,
        view.steps,
        view.snapshot_at,
        view.query_error or (None if view.slurm_job_id else "lease has no Slurm job"),
        view.ended,
    )


def _allocation_error(controller: Controller, error: str | None) -> None:
    allocation = controller.state.get("allocation")
    if not isinstance(allocation, dict) or allocation.get("reconciliation_error") == error:
        return
    if error is None:
        allocation.pop("reconciliation_error", None)
    else:
        allocation["reconciliation_error"] = error
    allocation["reconciliation_checked_at"] = utc_now()
    emit(
        controller,
        "notice",
        data={"source": "slurm_reconciliation", "error": error},
    )


def _job_error(controller: Controller, job: dict[str, Any], error: str | None) -> None:
    if job.get("reconciliation_error") == error:
        return
    if error is None:
        job.pop("reconciliation_error", None)
    else:
        job["reconciliation_error"] = error
    emit(
        controller,
        "notice",
        data={"source": "slurm_reconciliation", "job_id": job["id"], "error": error},
    )


def _needs_reconciliation(job: dict[str, Any], running: RunningProcess) -> bool:
    return (
        not job.get("slurm_step_id")
        or running.final_state is not None
        or running.process is None
        or running.process.poll() is not None
    )


def refresh_slurm_snapshot(controller: Controller, now: float) -> None:
    """Refresh at most once a second to avoid a tight slurmctld RPC loop."""

    if controller.launcher != "slurm":
        return
    lease_jobs: dict[str, bool] = {}
    job_needs_reconciliation = False
    for job_id, running in controller.running.items():
        job = controller.state["jobs"][job_id]
        lease_id = job_lease_id(job)
        needed = _needs_reconciliation(job, running)
        if lease_id is None:
            job_needs_reconciliation = job_needs_reconciliation or needed
        else:
            lease_jobs[lease_id] = lease_jobs.get(lease_id, False) or needed
    for lease_id, needed in lease_jobs.items():
        if needed:
            _refresh_lease_steps(controller, lease_id, now)
    if controller.hub:
        return
    health_needs_reconciliation = controller.health_step_name is not None
    needs_reconciliation = job_needs_reconciliation or health_needs_reconciliation
    retry_interval = (
        5
        if controller.slurm_query_error
        else 1
        if job_needs_reconciliation or len(controller.health_step_ids) != len(controller.inventory)
        else 10
    )
    if (
        controller.launcher != "slurm"
        or not needs_reconciliation
        or now - controller.last_slurm_query < retry_interval
    ):
        return
    controller.last_slurm_query = now
    try:
        controller.slurm_steps = live_steps(controller.slurm_job_id or "")
    except Exception as exc:
        controller.slurm_query_error = str(exc)
        _allocation_error(controller, controller.slurm_query_error)
        return
    controller.slurm_query_error = None
    controller.slurm_snapshot_at = now
    _allocation_error(controller, None)


def _refresh_lease_steps(controller: Controller, lease_id: str, now: float) -> None:
    view = lease_steps(controller, lease_id)
    if view.ended:
        # The allocation is gone, so none of its steps can be live. Each
        # refresh is a new proof of absence for the two-snapshot rule.
        if now - view.last_query >= 1:
            view.last_query = now
            view.steps = ()
            view.query_error = None
            view.snapshot_at = now
        return
    if now - view.last_query < (5 if view.query_error else 1):
        return
    view.last_query = now
    if not view.slurm_job_id:
        view.query_error = "lease has no Slurm job"
        return
    try:
        view.steps = live_steps(view.slurm_job_id)
    except Exception as exc:
        # A lease that has just ended fails here until the lease monitor
        # confirms the end; uncertainty keeps every reservation.
        view.query_error = str(exc)
        return
    view.query_error = None
    view.snapshot_at = now


def _matching_step(
    controller: Controller, job: dict[str, Any], snapshot: StepSnapshot | None = None
):
    snapshot = snapshot or step_snapshot(controller, job)
    token = str(job.get("launch_token", ""))
    if not token:
        raise RuntimeError("Slurm job has no persisted launch token")
    known_id = job.get("slurm_step_id")
    matches = [
        step
        for step in snapshot.steps
        if step.name == token or (known_id and step.step_id == known_id)
    ]
    matches = list({step.step_id: step for step in matches}.values())
    if len(matches) > 1:
        raise RuntimeError(f"multiple live steps match {token!r}")
    if not matches:
        return None
    step = matches[0]
    prefix = f"{snapshot.slurm_job_id}."
    suffix = step.step_id.removeprefix(prefix)
    if not step.step_id.startswith(prefix) or not suffix.isascii() or not suffix.isdecimal():
        raise RuntimeError(f"invalid live step ID {step.step_id!r}")
    if not step.nodes:
        raise RuntimeError(f"step {step.step_id} has no node set")
    return step


def reconcile_slurm(
    controller: Controller,
    job: dict[str, Any],
    running: RunningProcess,
) -> bool:
    """Return true only when fresh Slurm state proves the step absent."""

    snapshot = step_snapshot(controller, job)
    if snapshot.query_error or snapshot.snapshot_at == 0:
        return False
    try:
        step = _matching_step(controller, job, snapshot)
    except Exception as exc:
        _job_error(controller, job, str(exc))
        return False
    if step is not None:
        running.absence_confirmations = 0
        if job.get("slurm_step_id") != step.step_id:
            job["slurm_step_id"] = step.step_id
            if job["state"] == "starting" and running.final_state is None:
                job["state"] = "running"
                emit(controller, "job.running", job=job)
            else:
                emit(controller, "job.step_attached", job=job)
        if running.final_state is not None and not running.step_cancelled:
            try:
                cancel_step(snapshot.slurm_job_id, step.step_id)
            except Exception as exc:
                _job_error(controller, job, str(exc))
                return False
            running.step_cancelled = True
        _job_error(controller, job, None)
        return False

    _job_error(controller, job, None)
    if running.process is not None and (
        running.exit_seen_at is None or snapshot.snapshot_at < running.exit_seen_at
    ):
        return False
    if running.last_absence_snapshot_at != snapshot.snapshot_at:
        running.last_absence_snapshot_at = snapshot.snapshot_at
        running.absence_confirmations += 1
    # An unidentified launch gets a second snapshot to close the create/exit race.
    required = 1 if job.get("slurm_step_id") else 2
    return running.absence_confirmations >= required
