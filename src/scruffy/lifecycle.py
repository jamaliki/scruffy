"""Job launch, cancellation, output, and resource-release lifecycle."""

from __future__ import annotations

import queue
import signal
import subprocess
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ._compat import UTC
from .health import (
    nodes_requiring_exact_gpu_binding,
    quarantined_nodes,
    searching_hold_nodes,
    unavailable_gpu_ids,
)
from .leases import lease_accepts, lease_inventory, lease_remaining_seconds
from .models import (
    Assignment,
    NodeInventory,
    NodeReservation,
    QueuedJob,
    ResourceRequest,
    is_system_job,
    job_project,
)
from .provenance import (
    provenance_files,
    read_result_record,
    write_launch_record,
    write_result_record,
)
from .runtime import Controller, RunningProcess, signal_process, start_readers, stop_launcher
from .scheduler import choose_first_fitting_job, project_gpu_usage, queue_priority_key
from .slurm import (
    AllocationIncarnation,
    build_lease_environment,
    build_local_argv,
    build_srun_argv,
    build_srun_environment,
    completed_step,
    expand_slurm_hostnames,
    new_step_name,
)
from .slurm_runtime import (
    job_slurm_job_id,
    reconcile_slurm,
    refresh_slurm_snapshot,
    step_snapshot,
)
from .state import active_assignments, emit, ensure_committed
from .storage import (
    StorageError,
    atomic_write_json,
    job_directory,
    read_immutable_json,
    read_json,
    utc_now,
)
from .workflows import AUTO_RECOVERY_REASONS, mark_retry_exhaustion

MAX_MESSAGES_PER_TICK = 256
RUNTIME_PLACEMENT_CONTRACT = 1
# Workers record physical GPU identity just before exec. The controller looks
# for those records at this interval during the window after a job starts; a
# worker from an older release writes none, and the job simply has none.
GPU_IDENTITY_POLL_SECONDS = 2.0
GPU_IDENTITY_WINDOW_SECONDS = 600.0
_GPU_DEVICE_KEYS = ("uuid", "name", "index", "cuda_index", "pci_bus_id")

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
    exact_gpu_binding: bool = False,
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
            exact_gpu_binding
            and set(step_gpus) != {str(gpu_id) for gpu_id in reservation.gpu_ids}
        )
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


def _job_allocation(controller: Controller, job: dict[str, Any]) -> str:
    try:
        return job_slurm_job_id(controller, job)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc


def _lease_record(controller: Controller, lease_id: str) -> dict[str, Any]:
    lease = controller.state.get("leases", {}).get(lease_id)
    if not isinstance(lease, dict):
        raise StorageError(f"unknown lease {lease_id!r}")
    return lease


def _launch_target(
    controller: Controller, assignment: Assignment
) -> tuple[str, str | None, str | None]:
    """Return the allocation ID, Slurm job ID, and incarnation for a launch."""

    if assignment.lease_id is None:
        incarnation = controller.allocation_incarnation
        return (
            controller.allocation_id,
            controller.slurm_job_id,
            incarnation.fingerprint_sha256 if incarnation is not None else None,
        )
    lease = _lease_record(controller, assignment.lease_id)
    slurm_job_id = lease.get("slurm_job_id")
    raw_incarnation = lease.get("incarnation")
    if not isinstance(slurm_job_id, str) or not slurm_job_id:
        raise StorageError(f"lease {assignment.lease_id!r} has no Slurm job")
    try:
        incarnation = AllocationIncarnation.from_dict(raw_incarnation)
    except (TypeError, ValueError) as exc:
        raise StorageError(f"lease {assignment.lease_id!r} has no incarnation") from exc
    if incarnation.slurm_job_id != slurm_job_id:
        raise StorageError(f"lease {assignment.lease_id!r} incarnation differs")
    return slurm_job_id, slurm_job_id, incarnation.fingerprint_sha256


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
    outer_job_id = _job_allocation(controller, job)
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
                exact_gpu_binding=job.get("gpu_binding") == "exact",
            )
        )
    return result


def _runtime_placement_candidate_step(
    controller: Controller, job: dict[str, Any]
) -> str:
    """Read a worker-authored step ID while retaining placement validation."""

    assignment = Assignment.from_dict(job["assignment"])
    relative_files = job.get("runtime_placement_files")
    if isinstance(relative_files, (str, bytes)) or not isinstance(
        relative_files, Sequence
    ):
        raise TypeError("runtime placement file registry is missing")
    if len(relative_files) != len(assignment.reservations):
        raise ValueError("runtime placement file registry has the wrong size")
    outer_job_id = _job_allocation(controller, job)
    if not outer_job_id:
        raise ValueError("runtime placement has no outer Slurm job")
    candidate = None
    for index, reservation in enumerate(assignment.reservations):
        relative = f"jobs/{job['id']}/runtime-placement-{index}.json"
        if relative_files[index] != relative:
            raise ValueError("runtime placement file registry path differs")
        document, digest = read_immutable_json(controller.root / relative)
        if not isinstance(document, Mapping):
            raise TypeError("runtime placement record is not an object")
        record_step = document.get("slurm_step_id")
        if not isinstance(record_step, str) or not record_step:
            raise ValueError("runtime placement record has no Slurm step")
        full_step_id = (
            record_step
            if record_step.startswith(f"{outer_job_id}.")
            else f"{outer_job_id}.{record_step}"
        )
        _placement_entry(
            document,
            digest=digest,
            relative=relative,
            job_id=job["id"],
            reservation=reservation,
            requested=assignment.request.gpus_per_node,
            outer_job_id=outer_job_id,
            live_step_id=full_step_id,
            exact_gpu_binding=job.get("gpu_binding") == "exact",
        )
        if candidate is not None and candidate != full_step_id:
            raise ValueError("runtime placement records disagree on the Slurm step")
        candidate = full_step_id
    if candidate is None:
        raise ValueError("runtime placement has no worker records")
    return candidate


def _recover_completed_placement_step(
    controller: Controller, job: dict[str, Any], returncode: int
) -> bool:
    """Recover a too-short step from placement plus completed Slurm authority.

    Return true while a valid candidate is still settling in Slurm accounting.
    Invalid or absent placement evidence falls through to the normal fail-closed
    terminal path.
    """

    if job.get("runtime_placement_contract") != RUNTIME_PLACEMENT_CONTRACT:
        return False
    try:
        candidate = _runtime_placement_candidate_step(controller, job)
    except (KeyError, OSError, StorageError, TypeError, ValueError):
        return False
    try:
        result = completed_step(candidate)
        if result is None:
            return True
        actual_nodes = expand_slurm_hostnames(result.nodes or "")
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        job["reconciliation_error"] = str(exc)
        return True
    assignment = Assignment.from_dict(job["assignment"])
    expected_nodes = tuple(sorted(item.node for item in assignment.reservations))
    if (
        result.name != job.get("launch_token")
        or tuple(sorted(actual_nodes)) != expected_nodes
        or result.returncode != returncode
    ):
        job["reconciliation_error"] = (
            "completed Slurm step differs from runtime placement"
        )
        return False
    job["slurm_step_id"] = candidate
    job["slurm_state"] = result.state
    job.pop("reconciliation_error", None)
    emit(controller, "job.step_attached", job=job)
    return False


def _fail_unlaunched(
    controller: Controller, job: dict[str, Any], assignment: Assignment, exc: Exception
) -> None:
    prior = _replayable_result(controller, job)
    if prior is not None:
        _apply_result_record(job, prior)
        emit(controller, f"job.{job['state']}", job=job)
        return
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


def _replayable_result(
    controller: Controller, job: dict[str, Any]
) -> Mapping[str, Any] | None:
    """Return a prior result only for a modern identity-bearing job."""

    prior = read_result_record(controller.root, str(job["id"]))
    if prior is None:
        return None
    if type(job.get("attempt")) is int:
        return prior
    launch_sha256 = (job.get("provenance") or {}).get("launch_sha256")
    if isinstance(launch_sha256, str) and prior.get("launch_sha256") == launch_sha256:
        return prior
    return None


def record_terminal_result(controller: Controller, job: dict[str, Any]) -> None:
    """Persist a terminal job's immutable result, adopting one left by a crash.

    The result record is written before its lifecycle event becomes durable.
    If a controller stopped in between, the job is resolved again on restart
    with a new timestamp; the existing record is then the authority for the
    terminal fields instead of a conflicting second record.
    """

    prior = _replayable_result(controller, job)
    if prior is not None:
        _apply_result_record(job, prior)
        return
    write_result_record(controller.root, job)


def _apply_result_record(job: dict[str, Any], record: Mapping[str, Any]) -> None:
    """Restore terminal fields from a result written before its lifecycle event."""

    job.update(
        {
            "state": record.get("state"),
            "reason": record.get("reason"),
            "error": record.get("error"),
            "exit_code": record.get("exit_code"),
            "signal": record.get("signal"),
            "finished_at": record.get("finished_at"),
            "last_assignment": record.get("assignment"),
            "assignment": None,
            "stdout_bytes": record.get("stdout_bytes"),
            "stderr_bytes": record.get("stderr_bytes"),
        }
    )


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


def _requires_exact_gpu_binding(
    controller: Controller, assignment: Assignment
) -> bool:
    if (
        assignment.request.gpus_per_node == 0
        or assignment.lease_id is not None
        or getattr(controller, "gpu_isolation", "node") != "gpu"
    ):
        # Leases have no health monitor, so nothing requires exact binding;
        # a quarantined lease node is withheld whole instead.
        return False
    state = getattr(controller, "state", {})
    health = state.get("gpu_health", {}) if isinstance(state, Mapping) else {}
    inventory = getattr(controller, "inventory", ())
    exact_nodes = nodes_requiring_exact_gpu_binding(health, inventory)
    return any(item.node in exact_nodes for item in assignment.reservations)


def _launch_arguments(
    controller: Controller,
    job: dict[str, Any],
    assignment: Assignment,
    assignment_file: Path,
    stdout_file: Path,
    stderr_file: Path,
) -> tuple[list[str], dict[str, str] | None]:
    if controller.launcher == "slurm":
        binding = job.get("gpu_binding")
        exact_gpu_binding = (
            binding == "exact"
            if binding is not None
            else _requires_exact_gpu_binding(controller, assignment)
        )
        return (
            build_srun_argv(
                slurm_job_id=_launch_target(controller, assignment)[1] or "",
                name=job["launch_token"],
                assignment_file=assignment_file,
                stdout_file=stdout_file,
                stderr_file=stderr_file,
                node_names=[item.node for item in assignment.reservations],
                gpus_per_node=assignment.request.gpus_per_node,
                cpus_per_node=assignment.request.cpus_per_node,
                memory_gb_per_node=assignment.request.memory_gb_per_node,
                gpu_ids_per_node=(
                    [item.gpu_ids for item in assignment.reservations]
                    if exact_gpu_binding
                    else None
                ),
            ),
            (
                build_srun_environment()
                if assignment.lease_id is None
                else build_lease_environment()
            ),
        )
    return build_local_argv(assignment_file, assignment.reservations[0].node)


def start_job(
    controller: Controller, job: dict[str, Any], assignment: Assignment
) -> None:
    """Persist a reservation before launching, and never clear a live process."""

    prior_result = _replayable_result(controller, job)
    if prior_result is not None:
        _apply_result_record(job, prior_result)
        emit(controller, f"job.{job['state']}", job=job)
        return
    launch_file, _ = provenance_files(controller.root, str(job["id"]))
    prior_launch: dict[str, Any] | None = None
    try:
        allocation_id, slurm_job_id, incarnation_sha256 = _launch_target(
            controller, assignment
        )
        if launch_file.exists():
            prior_launch, _ = read_immutable_json(launch_file)
            if not isinstance(prior_launch, dict):
                raise StorageError(f"invalid launch record {launch_file}")
            if prior_launch.get("assignment") != assignment.to_dict():
                raise StorageError(
                    f"launch assignment changed before admission {job['id']}"
                )
            prior_job = prior_launch.get("job")
            if not isinstance(prior_job, dict):
                raise StorageError(f"launch record has no job identity {job['id']}")
            for key in (
                "project_id",
                "id",
                "request_id",
                "name",
                "workflow_id",
                "task_id",
                "attempt",
            ):
                if prior_job.get(key) != job.get(key):
                    raise StorageError(f"launch identity changed before admission {job['id']}")
            if (
                job.get("launch_token") is not None
                and prior_job.get("launch_token") != job.get("launch_token")
            ):
                raise StorageError(f"launch token changed before admission {job['id']}")
            if prior_launch.get("allocation_id") != allocation_id:
                raise StorageError(f"launch allocation changed before admission {job['id']}")
            if controller.launcher == "slurm":
                if prior_launch.get("allocation_incarnation_sha256") != incarnation_sha256:
                    raise StorageError(
                        f"launch record belongs to a different allocation incarnation {job['id']}"
                    )
    except (OSError, StorageError, TypeError, ValueError) as exc:
        _fail_unlaunched(controller, job, assignment, exc)
        return
    job["assignment"] = assignment.to_dict()
    if assignment.lease_id is None:
        job.pop("lease_id", None)
    else:
        job["lease_id"] = assignment.lease_id
    job["state"] = "starting"
    prior_job = prior_launch.get("job") if prior_launch else None
    prior_started_at = prior_launch.get("started_at") if prior_launch else None
    prior_token = prior_job.get("launch_token") if isinstance(prior_job, dict) else None
    job["started_at"] = (
        job.get("started_at")
        or (prior_started_at if isinstance(prior_started_at, str) else None)
        or utc_now()
    )
    job["deadline_at"] = _job_deadline(
        job["started_at"], assignment.request.time_limit_seconds
    )
    if controller.launcher == "slurm":
        if incarnation_sha256 is None:
            raise RuntimeError("Slurm controller has no allocation incarnation")
        job["launch_token"] = job.get("launch_token") or prior_token or new_step_name()
        job["allocation_incarnation_sha256"] = incarnation_sha256
        job["runtime_placement_contract"] = RUNTIME_PLACEMENT_CONTRACT
        job["runtime_placement_files"] = [
            f"jobs/{job['id']}/runtime-placement-{index}.json"
            for index, _ in enumerate(assignment.reservations)
        ]
        job["gpu_binding"] = (
            "exact"
            if not is_system_job(job)
            and _requires_exact_gpu_binding(controller, assignment)
            else "count"
        )

    directory = job_directory(controller.root, job["id"])
    assignment_file = directory / "assignment.json"
    stdout_file = directory / "stdout.log"
    stderr_file = directory / "stderr.log"
    try:
        launch_file = write_launch_record(controller.root, allocation_id, job, assignment)
        worker_document = {
            "root": str(controller.root),
            "job_id": job["id"],
            "project_id": job_project(job),
            "workflow_id": job.get("workflow_id"),
            "task_id": job.get("task_id"),
            "attempt": job.get("attempt"),
            "launcher": controller.launcher,
            "slurm_job_id": slurm_job_id,
            "launch_token": job.get("launch_token"),
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
            "gpu_binding": job.get("gpu_binding", "count"),
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
        ensure_committed(controller)
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


@dataclass(frozen=True, slots=True)
class PlacementTarget:
    """One allocation the scheduler may place work on in this pass."""

    lease_id: str | None
    inventory: tuple[NodeInventory, ...]
    assignments: tuple[Assignment, ...]
    unavailable: Mapping[str, Collection[int]]
    exact_gpu_nodes: frozenset[str] = frozenset()
    lease: Mapping[str, Any] | None = None


def placement_targets(controller: Controller) -> list[PlacementTarget]:
    """Return the primary allocation, then running leases, in preference order.

    Leases with the least remaining time come first so that long leases stay
    free for long jobs; an untimed attached lease comes last.
    """

    health = controller.state.get("gpu_health") or {}
    targets = []
    if controller.inventory:
        targets.append(
            PlacementTarget(
                None,
                controller.inventory,
                active_assignments(controller.state),
                unavailable_gpu_ids(health, controller.inventory),
                (
                    nodes_requiring_exact_gpu_binding(health, controller.inventory)
                    if controller.launcher == "slurm"
                    else frozenset()
                ),
            )
        )
    if controller.launcher != "slurm":
        return targets
    now = time.time()
    quarantined = quarantined_nodes(health)
    leases = []
    for lease_id, lease in controller.state.get("leases", {}).items():
        if lease.get("state") != "running" or lease.get("draining"):
            continue
        view = controller.lease_steps.get(lease_id)
        if view is not None and view.ended:
            continue
        inventory = lease_inventory(lease)
        if not inventory:
            continue
        remaining = lease_remaining_seconds(lease, now)
        leases.append(
            (
                remaining is None,
                remaining or 0.0,
                lease_id,
                PlacementTarget(
                    lease_id,
                    inventory,
                    active_assignments(controller.state, lease_id),
                    {node.name: node.gpu_ids for node in inventory if node.name in quarantined},
                    lease=lease,
                ),
            )
        )
    targets.extend(item[3] for item in sorted(leases, key=lambda item: item[:3]))
    return targets


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
        if not queued_images:
            return
        queued = [
            QueuedJob(job["id"], ResourceRequest.from_dict(job["request"]))
            for job in queued_images
        ]
        priority = {job.job_id: index for index, job in enumerate(queued)}
        system = {image["id"] for image in queued_images if is_system_job(image)}
        # While a GPU hold searches for its device, that node's free GPUs are
        # kept for the hold's sentinels (see gpu_holds).
        searching = searching_hold_nodes(controller.state.get("gpu_health") or {})
        now = time.time()
        best: tuple[int, QueuedJob, Assignment] | None = None
        for target in placement_targets(controller):
            limit = len(queued) if best is None else best[0]
            if target.lease is None:
                candidates = queued[:limit]
            else:
                # System jobs hold devices of the primary allocation only.
                candidates = [
                    queued_job
                    for queued_job, image in zip(queued[:limit], queued_images[:limit])
                    if queued_job.job_id not in system
                    and lease_accepts(
                        target.lease,
                        queued_job.request,
                        job_project(image),
                        now,
                        settings=controller.lease_settings,
                        untimed_margin_seconds=controller.drain_before_end_seconds,
                    )
                ]
            if not candidates:
                continue
            passes: list[tuple[list[QueuedJob], Mapping[str, Collection[int]]]] = [
                (candidates, target.unavailable)
            ]
            if target.lease is None and searching:
                withheld = {
                    **target.unavailable,
                    **{
                        node.name: node.gpu_ids
                        for node in target.inventory
                        if node.name in searching
                    },
                }
                passes = [
                    ([job for job in candidates if job.job_id in system], target.unavailable),
                    ([job for job in candidates if job.job_id not in system], withheld),
                ]
            choice = None
            for group, unavailable in passes:
                if group:
                    choice = choose_first_fitting_job(
                        target.inventory,
                        target.assignments,
                        group,
                        unavailable,
                        exact_gpu_nodes=target.exact_gpu_nodes,
                        slurm_count_binding=controller.launcher == "slurm",
                    )
                if choice is not None:
                    break
            if choice is None:
                continue
            queued_job, assignment = choice
            if target.lease_id is not None:
                assignment = replace(assignment, lease_id=target.lease_id)
            index = priority[queued_job.job_id]
            if best is None or index < best[0]:
                best = (index, queued_job, assignment)
            if index == 0:
                break
        if best is None:
            return
        _, queued_job, assignment = best
        if assignment.lease_id is not None:
            controller.lease_idle_since.pop(assignment.lease_id, None)
        start_job(controller, controller.state["jobs"][queued_job.job_id], assignment)


def request_cancellation(
    controller: Controller,
    job: dict[str, Any],
    request_id: str | None = None,
    *,
    bulk: bool = False,
    deferred_stops: list[RunningProcess] | None = None,
) -> bool:
    """Cancel one job, returning false when its state cannot be cancelled.

    The job records ``cancel_request_id``. A bulk operation correlates its
    per-job events by ``bulk_request_id`` because only the operation's final
    summary is that command's outcome. When ``deferred_stops`` is given,
    launchers are collected for the caller to stop after one commit instead of
    being signalled one by one.
    """

    data = None
    if request_id:
        data = {"bulk_request_id" if bulk else "request_id": request_id}
    if job["state"] in {"queued", "blocked"}:
        prior = _replayable_result(controller, job)
        if prior is not None:
            _apply_result_record(job, prior)
            emit(controller, f"job.{job['state']}", job=job)
            return True
        job["state"] = "cancelled"
        job["finished_at"] = utc_now()
        job["reason"] = "cancelled_before_start"
        if request_id:
            job["cancel_request_id"] = request_id
        write_result_record(controller.root, job)
        emit(controller, "job.cancelled", job=job, data=data)
        return True
    if job["state"] not in {"starting", "running", "finishing"}:
        return False
    job["state"] = "cancelling"
    job["reason"] = "cancel_requested"
    if request_id:
        job["cancel_request_id"] = request_id
    emit(controller, "job.cancelling", job=job, data=data)
    running = controller.running.get(job["id"])
    if running is not None:
        if running.final_state is None:
            running.final_state = "cancelled"
            running.final_reason = "cancelled"
        if deferred_stops is not None:
            deferred_stops.append(running)
        else:
            # The cancelling transition must be durable before the launcher
            # can observe the request.
            ensure_committed(controller)
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
    evacuation = controller.state.get("evacuation")
    evacuation_target = (
        evacuation.get("targets", {}).get(job_id)
        if isinstance(evacuation, dict)
        else None
    )
    evacuated_exit = (
        returncode == 75
        and isinstance(evacuation_target, dict)
        and evacuation_target.get("outcome") in {"waiting", "checkpointed"}
    )
    if evacuated_exit:
        state, reason = "failed", "evacuated"
    elif running.final_state is not None:
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
        # A short job can end before the periodic check reads its identity.
        _ingest_gpu_identity(controller, job, running, final=True)
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
    prior = _replayable_result(controller, job)
    if prior is not None:
        _apply_result_record(job, prior)
        state = job["state"]
        reason = job.get("reason")
    else:
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
    if state == "lost" and isinstance(reason, str) and reason in AUTO_RECOVERY_REASONS:
        mark_retry_exhaustion(job, reason)
        controller.auto_recovery_pending = True
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
            _ingest_gpu_identity(controller, controller.state["jobs"][job_id], running)
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
            if (
                returncode is not None
                and not job.get("slurm_step_id")
                and _recover_completed_placement_step(controller, job, returncode)
            ):
                continue
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


def _gpu_identity_records(
    controller: Controller, job: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]] | None:
    """Return the devices and errors workers recorded, or None while one is missing."""

    assignment = Assignment.from_dict(job["assignment"])
    devices: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, reservation in enumerate(assignment.reservations):
        source = controller.root / "jobs" / str(job["id"]) / f"gpu-identity-{index}.json"
        if not source.exists():
            return None
        try:
            document = read_json(source)
        except StorageError as exc:
            errors.append(f"{reservation.node}: {exc}")
            continue
        if (
            not isinstance(document, dict)
            or document.get("schema") != 1
            or document.get("job_id") != job["id"]
            or document.get("node") != reservation.node
            or document.get("launch_token") != job.get("launch_token")
            or not isinstance(document.get("devices"), list)
        ):
            errors.append(f"{reservation.node}: identity record differs from the launch")
            continue
        if document.get("error"):
            errors.append(f"{reservation.node}: {document['error']}")
        for device in document["devices"]:
            if isinstance(device, dict) and isinstance(device.get("uuid"), str):
                devices.append(
                    {"node": reservation.node}
                    | {key: device.get(key) for key in _GPU_DEVICE_KEYS}
                )
    return devices, errors


def _ingest_gpu_identity(
    controller: Controller,
    job: dict[str, Any],
    running: RunningProcess,
    *,
    final: bool = False,
) -> None:
    """Record the physical GPUs a Slurm job's workers reported, once.

    Identity is reportable, not authority: Slurm chose the devices and the
    runtime placement record remains the launch contract. A job whose worker
    wrote no record, for example one launched by an older release, keeps no
    ``gpu_devices``.
    """

    if (
        controller.launcher != "slurm"
        or "gpu_devices" in job
        or job.get("runtime_placement_contract") != RUNTIME_PLACEMENT_CONTRACT
        or not isinstance(job.get("assignment"), dict)
        or not isinstance(job.get("started_at"), str)
    ):
        return
    try:
        assignment = Assignment.from_dict(job["assignment"])
        if assignment.request.gpus_per_node == 0:
            return
        age = (
            datetime.now(UTC) - datetime.fromisoformat(job["started_at"])
        ).total_seconds()
    except (KeyError, TypeError, ValueError):
        return
    now = time.monotonic()
    if not final:
        if age > GPU_IDENTITY_WINDOW_SECONDS or (
            running.gpu_identity_checked_at is not None
            and now - running.gpu_identity_checked_at < GPU_IDENTITY_POLL_SECONDS
        ):
            return
        running.gpu_identity_checked_at = now
    try:
        records = _gpu_identity_records(controller, job)
    except (KeyError, OSError, TypeError, ValueError):
        return
    if records is None:
        return
    devices, errors = records
    job["gpu_devices"] = devices
    if errors:
        job["gpu_identity_error"] = "; ".join(errors)
    if not final:
        emit(controller, "job.gpu_identified", job=job)


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

    snapshot_at = step_snapshot(controller, job).snapshot_at
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
