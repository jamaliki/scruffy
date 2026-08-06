"""Fail-closed reconciliation between local ``srun`` clients and Slurm steps."""

from __future__ import annotations

from typing import Any

from .runtime import Controller, RunningProcess
from .slurm import cancel_step, live_steps
from .state import emit
from .storage import utc_now


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


def _job_error(
    controller: Controller, job: dict[str, Any], error: str | None
) -> None:
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


def refresh_slurm_snapshot(controller: Controller, now: float) -> None:
    """Refresh at most once a second to avoid a tight slurmctld RPC loop."""

    needs_reconciliation = any(
        not controller.state["jobs"][job_id].get("slurm_step_id")
        or running.final_state is not None
        or running.process is None
        or running.process.poll() is not None
        for job_id, running in controller.running.items()
    )
    retry_interval = 5 if controller.slurm_query_error else 1
    if (
        controller.launcher != "slurm"
        or not controller.running
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


def _matching_step(controller: Controller, job: dict[str, Any]):
    token = str(job.get("launch_token", ""))
    if not token:
        raise RuntimeError("Slurm job has no persisted launch token")
    known_id = job.get("slurm_step_id")
    matches = [
        step
        for step in controller.slurm_steps
        if step.name == token or (known_id and step.step_id == known_id)
    ]
    matches = list({step.step_id: step for step in matches}.values())
    if len(matches) > 1:
        raise RuntimeError(f"multiple live steps match {token!r}")
    if not matches:
        return None
    step = matches[0]
    prefix = f"{controller.slurm_job_id}."
    suffix = step.step_id.removeprefix(prefix)
    if (
        not step.step_id.startswith(prefix)
        or not suffix.isascii()
        or not suffix.isdecimal()
    ):
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

    if controller.slurm_query_error or controller.slurm_snapshot_at == 0:
        return False
    try:
        step = _matching_step(controller, job)
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
                cancel_step(controller.slurm_job_id or "", step.step_id)
            except Exception as exc:
                _job_error(controller, job, str(exc))
                return False
            running.step_cancelled = True
        _job_error(controller, job, None)
        return False

    _job_error(controller, job, None)
    if running.process is not None and (
        running.exit_seen_at is None
        or controller.slurm_snapshot_at < running.exit_seen_at
    ):
        return False
    if running.last_absence_snapshot_at != controller.slurm_snapshot_at:
        running.last_absence_snapshot_at = controller.slurm_snapshot_at
        running.absence_confirmations += 1
    # An unidentified launch gets a second snapshot to close the create/exit race.
    required = 1 if job.get("slurm_step_id") else 2
    return running.absence_confirmations >= required
