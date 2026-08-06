"""Job launch, cancellation, output, and resource-release lifecycle."""

from __future__ import annotations

import queue
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from .models import Assignment, QueuedJob, ResourceRequest, job_project
from .runtime import Controller, RunningProcess, signal_process, start_readers, stop_launcher
from .scheduler import choose_oldest_fitting_job
from .slurm import (
    build_local_argv,
    build_srun_argv,
    build_srun_environment,
    completed_step,
    new_step_name,
)
from .slurm_runtime import reconcile_slurm, refresh_slurm_snapshot
from .state import active_assignments, emit
from .storage import atomic_write_json, job_directory, utc_now

MAX_MESSAGES_PER_TICK = 256


def _fail_unlaunched(
    controller: Controller, job: dict[str, Any], assignment: Assignment, exc: Exception
) -> None:
    job.update(
        {
            "state": "failed",
            "last_assignment": assignment.to_dict(),
            "assignment": None,
            "finished_at": utc_now(),
            "reason": "launch_failed",
            "error": str(exc),
        }
    )
    emit(controller, "job.failed", job=job)


def _launch_arguments(
    controller: Controller,
    job: dict[str, Any],
    assignment: Assignment,
    assignment_file: Path,
    stdout_file: Path,
    stderr_file: Path,
) -> tuple[list[str], dict[str, str] | None]:
    if controller.launcher == "slurm":
        return (
            build_srun_argv(
                slurm_job_id=controller.slurm_job_id or "",
                name=job["launch_token"],
                assignment_file=assignment_file,
                stdout_file=stdout_file,
                stderr_file=stderr_file,
                node_names=[item.node for item in assignment.reservations],
                cpus_per_node=assignment.request.cpus_per_node,
                memory_gb_per_node=assignment.request.memory_gb_per_node,
            ),
            build_srun_environment(),
        )
    return build_local_argv(assignment_file, assignment.reservations[0].node)


def start_job(
    controller: Controller, job: dict[str, Any], assignment: Assignment
) -> None:
    """Persist a reservation before launching, and never clear a live process."""

    job["assignment"] = assignment.to_dict()
    job["state"] = "starting"
    job["started_at"] = utc_now()
    if controller.launcher == "slurm":
        job["launch_token"] = new_step_name()
    emit(controller, "job.starting", job=job)

    directory = job_directory(controller.root, job["id"])
    assignment_file = directory / "assignment.json"
    stdout_file = directory / "stdout.log"
    stderr_file = directory / "stderr.log"
    worker_document = {
        "root": str(controller.root),
        "job_id": job["id"],
        "project_id": job_project(job),
        "argv": job["argv"],
        "cwd": job["cwd"],
        "env": job["env"],
        "assignment": [item.to_dict() for item in assignment.reservations],
    }
    try:
        atomic_write_json(assignment_file, worker_document)
        argv, environment = _launch_arguments(
            controller,
            job,
            assignment,
            assignment_file,
            stdout_file,
            stderr_file,
        )
        process = subprocess.Popen(
            argv,
            stdout=(
                subprocess.DEVNULL
                if controller.launcher == "slurm"
                else subprocess.PIPE
            ),
            stderr=(
                subprocess.DEVNULL
                if controller.launcher == "slurm"
                else subprocess.PIPE
            ),
            env=environment,
            start_new_session=True,
        )
    except Exception as exc:
        _fail_unlaunched(controller, job, assignment, exc)
        return

    running = RunningProcess(process, job.get("launch_token"))
    controller.running[job["id"]] = running
    job.update(
        {
            "pid": process.pid,
            "stdout": f"jobs/{job['id']}/stdout.log",
            "stderr": f"jobs/{job['id']}/stderr.log",
        }
    )
    if controller.launcher == "slurm":
        running.closed_streams.update({"stdout", "stderr"})
        running.output_offsets = {"stdout": 0, "stderr": 0}
    else:
        try:
            start_readers(
                running,
                job_id=job["id"],
                stdout_file=stdout_file,
                stderr_file=stderr_file,
                messages=controller.messages,
                output=controller.output,
            )
        except Exception as exc:
            running.final_state = "failed"
            running.final_reason = "launch_failed"
            job["error"] = str(exc)
            stop_launcher(controller, running)
            return

    # Slurm jobs become running only once reconciliation observes their step.
    if controller.launcher == "local":
        job["state"] = "running"
        emit(controller, "job.running", job=job)


def schedule(controller: Controller) -> None:
    while (
        not controller.stopping
        and not controller.state["draining"]
        and not controller.state.get("launches_paused", False)
    ):
        queued = sorted(
            (
                QueuedJob(job["id"], ResourceRequest.from_dict(job["request"]))
                for job in controller.state["jobs"].values()
                if job["state"] == "queued"
            ),
            key=lambda item: controller.state["jobs"][item.job_id]["queue_order"],
        )
        choice = choose_oldest_fitting_job(
            controller.inventory, active_assignments(controller.state), queued
        )
        if choice is None:
            return
        queued_job, assignment = choice
        start_job(controller, controller.state["jobs"][queued_job.job_id], assignment)


def request_cancellation(
    controller: Controller, job: dict[str, Any], request_id: str | None = None
) -> bool:
    data = {"request_id": request_id} if request_id else None
    if job["state"] in {"queued", "blocked"}:
        job["state"] = "cancelled"
        job["finished_at"] = utc_now()
        job["reason"] = "cancelled_before_start"
        emit(controller, "job.cancelled", job=job, data=data)
        return True
    if job["state"] not in {"starting", "running", "finishing"}:
        return False
    job["state"] = "cancelling"
    job["reason"] = "cancel_requested"
    emit(controller, "job.cancelling", job=job, data=data)
    running = controller.running.get(job["id"])
    if running is not None:
        if running.final_state is None:
            running.final_state = "cancelled"
            running.final_reason = "cancelled"
        stop_launcher(controller, running)
    return True


def drain_messages(
    controller: Controller, limit: int = MAX_MESSAGES_PER_TICK
) -> None:
    """Handle a bounded batch so noisy output cannot starve the event loop."""

    for _ in range(limit):
        try:
            message = controller.messages.get_nowait()
        except queue.Empty:
            return
        running = controller.running.get(message["job_id"])
        if message["kind"] == "stream_closed":
            if running is not None:
                running.closed_streams.add(message["stream"])
            continue
        if message["kind"] == "output_ready":
            output_range = controller.output.take(message["job_id"], message["stream"])
            if output_range is None:
                continue
            offset, length = output_range
            emit(
                controller,
                "job.output",
                data={
                    "job_id": message["job_id"],
                    "stream": message["stream"],
                    "log": f"jobs/{message['job_id']}/{message['stream']}.log",
                    "offset": offset,
                    "length": length,
                },
                durable=False,
                snapshot=False,
            )
        elif message["kind"] == "output_error":
            emit(controller, "notice", data=message)


def _finish_job(
    controller: Controller, job_id: str, running: RunningProcess, returncode: int
) -> None:
    job = controller.state["jobs"][job_id]
    if running.final_state is not None:
        state = running.final_state
        reason = running.final_reason or state
    elif returncode == 0:
        state, reason = "succeeded", "process_exit"
    else:
        state, reason = "failed", "process_exit"
    job.update(
        {
            "state": state,
            "finished_at": utc_now(),
            "exit_code": returncode if returncode >= 0 else None,
            "signal": -returncode if returncode < 0 else None,
            "reason": reason,
            "last_assignment": job["assignment"],
            "assignment": None,
        }
    )
    job.pop("pid", None)
    job.pop("pending_returncode", None)
    for stream_name in ("stdout", "stderr"):
        relative_name = job.get(stream_name)
        if relative_name:
            source = controller.root / relative_name
            try:
                size = source.stat().st_size
            except FileNotFoundError:
                size = 0
            job[f"{stream_name}_bytes"] = size
    emit(controller, f"job.{state}", job=job)


def poll_processes(controller: Controller) -> None:
    now = time.monotonic()
    refresh_slurm_snapshot(controller, now)
    for job_id, running in list(controller.running.items()):
        if (
            controller.launcher == "local"
            and running.cancel_deadline is not None
            and now >= running.cancel_deadline
            and running.process is not None
            and running.process.poll() is None
        ):
            signal_process(running.process, signal.SIGKILL)
            running.cancel_deadline = None

        if controller.launcher == "slurm":
            _poll_slurm_output(controller, job_id, running)
        returncode = running.process.poll() if running.process is not None else None
        pending_returncode = controller.state["jobs"][job_id].get(
            "pending_returncode"
        )
        if returncode is None and isinstance(pending_returncode, int):
            returncode = pending_returncode
        job = controller.state["jobs"][job_id]
        if returncode is not None and running.exit_seen_at is None:
            running.exit_seen_at = time.monotonic()
            job["pending_returncode"] = returncode
            if controller.launcher == "slurm" and job["state"] != "cancelling":
                job["state"] = "finishing"
                emit(controller, "job.finishing", job=job)

        slurm_absent = True
        if controller.launcher == "slurm":
            slurm_absent = reconcile_slurm(controller, job, running)
            if slurm_absent and running.process is None and returncode is None:
                returncode = _recovered_returncode(controller, job, running)
                if returncode is None and not job.get("slurm_step_id"):
                    running.final_state = "lost"
                    running.final_reason = "controller_recovery_no_step"
                    returncode = 1
        if (
            returncode is None
            or running.closed_streams != {"stdout", "stderr"}
            or controller.output.has_pending(job_id)
            or not slurm_absent
        ):
            continue
        for reader in running.readers:
            reader.join(timeout=1)
        _finish_job(controller, job_id, running, returncode)
        del controller.running[job_id]


def _poll_slurm_output(
    controller: Controller, job_id: str, running: RunningProcess
) -> None:
    """Publish ranges appended by Slurm without reading log contents."""

    for stream_name in ("stdout", "stderr"):
        source = controller.root / "jobs" / job_id / f"{stream_name}.log"
        try:
            size = source.stat().st_size
        except FileNotFoundError:
            size = 0
        previous = min(running.output_offsets.get(stream_name, 0), size)
        running.output_offsets[stream_name] = size
        if size > previous:
            controller.output.record(job_id, stream_name, previous, size - previous)


def _recovered_returncode(
    controller: Controller, job: dict[str, Any], running: RunningProcess
) -> int | None:
    """Resolve an attached step through accounting after it leaves live state."""

    snapshot_at = controller.slurm_snapshot_at
    if running.last_accounting_snapshot_at == snapshot_at:
        return None
    running.last_accounting_snapshot_at = snapshot_at
    try:
        result = completed_step(str(job["slurm_step_id"]))
    except Exception as exc:
        error = str(exc)
        if job.get("reconciliation_error") != error:
            job["reconciliation_error"] = error
            emit(
                controller,
                "notice",
                data={
                    "source": "slurm_accounting",
                    "job_id": job["id"],
                    "error": error,
                },
            )
        return None
    if result is None:
        return None
    job.pop("reconciliation_error", None)
    job["slurm_state"] = result.state
    job["pending_returncode"] = result.returncode
    running.exit_seen_at = time.monotonic()
    return result.returncode


def begin_shutdown(controller: Controller) -> None:
    if controller.stop_announced:
        return
    controller.stop_announced = True
    controller.state["draining"] = True
    controller.state["allocation"]["state"] = "stopping"
    emit(controller, "allocation.stopping")
    if controller.launcher == "slurm":
        return
    for running in controller.running.values():
        if running.final_state is None:
            running.final_state = "lost"
            running.final_reason = "controller_stopped"
        stop_launcher(controller, running)
