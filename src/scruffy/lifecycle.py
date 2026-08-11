"""Job launch, cancellation, output, and resource-release lifecycle."""

from __future__ import annotations

import queue
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .models import (
    Assignment,
    NodeReservation,
    QueuedJob,
    ResourceRequest,
    job_project,
)
from .provenance import write_launch_record, write_result_record
from .runtime import Controller, RunningProcess, signal_process, start_readers, stop_launcher
from .scheduler import choose_first_fitting_job, project_gpu_usage, queue_priority_key
from .slurm import (
    build_local_argv,
    build_srun_argv,
    build_srun_environment,
    completed_step,
    new_step_name,
)
from .slurm_runtime import reconcile_slurm, refresh_slurm_snapshot
from .state import active_assignments, emit
from .storage import (
    StorageError,
    atomic_write_json,
    job_directory,
    read_immutable_json,
    utc_now,
)

MAX_MESSAGES_PER_TICK = 256
RUNTIME_PLACEMENT_CONTRACT = 1

_RUNTIME_PLACEMENT_KEYS = {
    "schema",
    "job_id",
    "node",
    "requested_gpus",
    "ledger_gpu_ids",
    "slurm_job_id",
    "slurm_step_id",
    "slurm_step_gpus",
    "cuda_visible_devices",
    "cuda_device_order",
}


def _string_list(value: object, label: str, expected: int) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} is not a list")
    result = list(value)
    if (
        len(result) != expected
        or any(
            not isinstance(item, str) or not item or item.strip() != item
            for item in result
        )
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{label} does not match the requested GPU count")
    return result


def _placement_entry(
    document: Mapping[str, Any],
    *,
    digest: str,
    relative: str,
    job_id: str,
    reservation: NodeReservation,
    requested: int,
    outer_job_id: str,
    live_step_id: str,
) -> dict[str, Any]:
    if set(document) != _RUNTIME_PLACEMENT_KEYS:
        raise ValueError("runtime placement record has invalid keys")
    step_gpus = _string_list(document["slurm_step_gpus"], "step GPUs", requested)
    visible = _string_list(document["cuda_visible_devices"], "visible GPUs", requested)
    record_step = document["slurm_step_id"]
    full_step_id = (
        record_step
        if isinstance(record_step, str) and record_step.startswith(f"{outer_job_id}.")
        else f"{outer_job_id}.{record_step}"
    )
    device_order = document["cuda_device_order"]
    ledger_ids = document["ledger_gpu_ids"]
    if (
        type(document["schema"]) is not int
        or document["schema"] != 1
        or type(document["requested_gpus"]) is not int
        or document["job_id"] != job_id
        or document["node"] != reservation.node
        or document["requested_gpus"] != requested
        or not isinstance(ledger_ids, list)
        or any(type(gpu_id) is not int for gpu_id in ledger_ids)
        or ledger_ids != list(reservation.gpu_ids)
        or document["slurm_job_id"] != outer_job_id
        or not isinstance(record_step, str)
        or not record_step
        or full_step_id != live_step_id
        or (
            device_order is not None
            and (not isinstance(device_order, str) or not device_order)
        )
    ):
        raise ValueError("runtime placement record differs from the launch")
    return {
        "path": relative,
        "sha256": digest,
        "node": reservation.node,
        "slurm_step_id": full_step_id,
        "physical_gpu_ids": step_gpus,
        "visible_gpu_ids": visible,
        "reserved_gpu_ids": list(reservation.gpu_ids),
    }


def _runtime_placement_authority(
    controller: Controller, job: dict[str, Any]
) -> list[dict[str, Any]]:
    assignment = Assignment.from_dict(job["assignment"])
    relative_files = job.get("runtime_placement_files")
    if isinstance(relative_files, (str, bytes)) or not isinstance(
        relative_files, Sequence
    ):
        raise ValueError("runtime placement file registry is missing")
    if len(relative_files) != len(assignment.reservations):
        raise ValueError("runtime placement file registry has the wrong size")
    outer_job_id = controller.slurm_job_id or ""
    live_step_id = job.get("slurm_step_id")
    if not outer_job_id or not isinstance(live_step_id, str):
        raise ValueError("runtime placement has no reconciled Slurm step")
    result = []
    for index, reservation in enumerate(assignment.reservations):
        relative = f"jobs/{job['id']}/runtime-placement-{index}.json"
        if relative_files[index] != relative:
            raise ValueError("runtime placement file registry path differs")
        document, digest = read_immutable_json(controller.root / relative)
        if not isinstance(document, Mapping):
            raise ValueError("runtime placement record is not an object")
        result.append(
            _placement_entry(
                document,
                digest=digest,
                relative=relative,
                job_id=job["id"],
                reservation=reservation,
                requested=assignment.request.gpus_per_node,
                outer_job_id=outer_job_id,
                live_step_id=live_step_id,
            )
        )
    return result


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
    write_result_record(controller.root, job)
    emit(controller, "job.failed", job=job)


def _job_deadline(started_at: str, seconds: int | None) -> str | None:
    """Return a restart-stable wall-clock deadline for an optional time limit."""

    if seconds is None:
        return None
    started = datetime.fromisoformat(started_at)
    return (started + timedelta(seconds=seconds)).astimezone(UTC).isoformat(
        timespec="milliseconds"
    )


def remaining_time_limit(job: dict[str, Any]) -> float | None:
    """Return remaining wall time, preserving limits across controller restarts."""

    deadline = job.get("deadline_at")
    if not isinstance(deadline, str):
        return None
    parsed = datetime.fromisoformat(deadline)
    return max(0.0, (parsed - datetime.now(UTC)).total_seconds())


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
                gpus_per_node=assignment.request.gpus_per_node,
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
    job["deadline_at"] = _job_deadline(
        job["started_at"], assignment.request.time_limit_seconds
    )
    if controller.launcher == "slurm":
        if controller.allocation_incarnation is None:
            raise RuntimeError("Slurm controller has no allocation incarnation")
        job["launch_token"] = new_step_name()
        job["allocation_incarnation_sha256"] = (
            controller.allocation_incarnation.fingerprint_sha256
        )
        job["runtime_placement_contract"] = RUNTIME_PLACEMENT_CONTRACT
        job["runtime_placement_files"] = [
            f"jobs/{job['id']}/runtime-placement-{index}.json"
            for index, _ in enumerate(assignment.reservations)
        ]

    directory = job_directory(controller.root, job["id"])
    assignment_file = directory / "assignment.json"
    stdout_file = directory / "stdout.log"
    stderr_file = directory / "stderr.log"
    try:
        launch_file = write_launch_record(
            controller.root, controller.allocation_id, job, assignment
        )
        worker_document = {
            "root": str(controller.root),
            "job_id": job["id"],
            "project_id": job_project(job),
            "workflow_id": job.get("workflow_id"),
            "task_id": job.get("task_id"),
            "attempt": job.get("attempt"),
            "launcher": controller.launcher,
            "slurm_job_id": controller.slurm_job_id,
            "allocation_incarnation_sha256": job.get(
                "allocation_incarnation_sha256"
            ),
            "runtime_placement_contract": job.get("runtime_placement_contract"),
            "provenance_path": str(launch_file),
            "assignment_sha256": job["provenance"]["assignment_sha256"],
            "argv": job["argv"],
            "cwd": job["cwd"],
            "env": job["env"],
            "assignment": [
                {
                    **item.to_dict(),
                    "runtime_placement": (
                        job["runtime_placement_files"][index]
                        if controller.launcher == "slurm"
                        else None
                    ),
                }
                for index, item in enumerate(assignment.reservations)
            ],
            "slurm_managed_gpus": controller.launcher == "slurm",
            "gpus_per_node": assignment.request.gpus_per_node,
            # A Slurm worker owns its log descriptors so output survives the
            # controller and its local srun client. Local jobs keep using the
            # controller's pipes and reader threads.
            "logs": (
                {"stdout": str(stdout_file), "stderr": str(stderr_file)}
                if controller.launcher == "slurm"
                else None
            ),
        }
        # The durable starting event owns the reservation before any process
        # can be launched. The immutable launch record is already available to
        # the worker by the time that transition is published.
        emit(controller, "job.starting", job=job)
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
    remaining = remaining_time_limit(job)
    if remaining is not None:
        running.time_limit_deadline = time.monotonic() + remaining
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
        jobs = controller.state["jobs"]
        usage = project_gpu_usage(jobs.values())
        queued_images = sorted(
            (job for job in jobs.values() if job["state"] == "queued"),
            key=lambda job: queue_priority_key(job, usage),
        )
        queued = [
            QueuedJob(job["id"], ResourceRequest.from_dict(job["request"]))
            for job in queued_images
        ]
        choice = choose_first_fitting_job(
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
        write_result_record(controller.root, job)
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
        state = "failed"
        slurm_parts = str(job.get("slurm_state") or "").split(maxsplit=1)
        slurm_state = slurm_parts[0] if slurm_parts else ""
        if slurm_state == "OUT_OF_MEMORY":
            reason = "oom_kill"
        elif slurm_state == "TIMEOUT":
            reason = "timeout"
        elif slurm_state in {"BOOT_FAIL", "NODE_FAIL", "REVOKED"}:
            reason = "infrastructure_failure"
        elif returncode < 0:
            reason = "signal"
        else:
            reason = "application_exit"
    runtime_placements = None
    placement_error = None
    placement_status = None
    if controller.launcher == "slurm":
        contract = job.get("runtime_placement_contract")
        if contract is None:
            placement_status = "legacy_unavailable"
        elif type(contract) is not int or contract != RUNTIME_PLACEMENT_CONTRACT:
            placement_status = "invalid"
            placement_error = "unsupported runtime placement contract"
            if state == "succeeded":
                state, reason = "failed", "runtime_placement_invalid"
        else:
            try:
                runtime_placements = _runtime_placement_authority(controller, job)
                placement_status = "authenticated"
            except (KeyError, OSError, StorageError, TypeError, ValueError) as exc:
                placement_status = "invalid"
                placement_error = str(exc)
                if state == "succeeded":
                    state, reason = "failed", "runtime_placement_invalid"
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
    if runtime_placements is not None:
        job["runtime_placements"] = runtime_placements
        job.pop("runtime_placement_error", None)
    elif placement_error is not None:
        job["runtime_placement_error"] = placement_error
    if placement_status is not None:
        job["runtime_placement_status"] = placement_status
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
    write_result_record(controller.root, job)
    emit(controller, f"job.{state}", job=job)


def poll_processes(controller: Controller) -> None:
    now = time.monotonic()
    refresh_slurm_snapshot(controller, now)
    for job_id, running in list(controller.running.items()):
        if (
            running.time_limit_deadline is not None
            and now >= running.time_limit_deadline
            and running.final_state is None
        ):
            running.final_state = "failed"
            running.final_reason = "timeout"
            stop_launcher(controller, running)
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
