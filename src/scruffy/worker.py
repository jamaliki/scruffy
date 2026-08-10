"""Node-side worker that applies a controller assignment then execs the job."""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

from .models import job_project
from .storage import create_immutable_json

_SLURM_GPU_ENVIRONMENT = (
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_JOB_ID",
    "SLURM_JOBID",
    "SLURM_STEP_ID",
    "SLURM_STEPID",
)
_SCRUFFY_GPU_ENVIRONMENT = (
    "SCRUFFY_GPU_IDS",
    "SCRUFFY_PHYSICAL_GPU_IDS",
    "SCRUFFY_RESERVED_GPU_IDS",
    "SCRUFFY_RUNTIME_PLACEMENT",
    "SCRUFFY_RUNTIME_PLACEMENT_SHA256",
    "SCRUFFY_SLURM_JOB_ID",
    "SCRUFFY_SLURM_STEP_ID",
    "SCRUFFY_STEP_GPU_IDS",
)


def _comma_values(value: str, label: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(","))
    if not values or any(not item for item in values):
        raise ValueError(f"{label} must be a non-empty comma-separated list")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicate GPU identities")
    return values


def _runtime_placement_record(
    document: dict[str, Any],
    placement: dict[str, Any],
    *,
    expected: int,
    job_id: str,
    step_id: str,
    step_gpus: tuple[str, ...],
    visible: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema": 1,
        "job_id": str(document["job_id"]),
        "node": str(placement["node"]),
        "requested_gpus": expected,
        "ledger_gpu_ids": list(placement["gpu_ids"]),
        "slurm_job_id": job_id,
        "slurm_step_id": step_id,
        "slurm_step_gpus": list(step_gpus),
        "cuda_visible_devices": list(visible),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
    }


def _slurm_gpu_environment(
    document: dict[str, Any], placement: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    expected = document.get("gpus_per_node")
    if type(expected) is not int or expected <= 0:
        raise ValueError("gpus_per_node must be a positive integer")
    if len(placement["gpu_ids"]) != expected:
        raise ValueError("ledger GPU count differs from the requested Slurm step")

    inherited = os.environ
    visible_raw = inherited.get("CUDA_VISIBLE_DEVICES", "")
    step_raw = inherited.get("SLURM_STEP_GPUS", "")
    visible = _comma_values(visible_raw, "CUDA_VISIBLE_DEVICES")
    step_gpus = _comma_values(step_raw, "SLURM_STEP_GPUS")
    if len(visible) != expected or len(step_gpus) != expected:
        raise ValueError("Slurm GPU mapping count differs from gpus_per_node")

    step_id = inherited.get("SLURM_STEP_ID") or inherited.get("SLURM_STEPID")
    job_id = inherited.get("SLURM_JOB_ID") or inherited.get("SLURM_JOBID")
    if not step_id or not job_id:
        raise ValueError("Slurm worker is missing job or step identity")
    if document.get("slurm_job_id") != job_id:
        raise ValueError("Slurm worker allocation differs from its assignment")

    protected = {
        "CUDA_VISIBLE_DEVICES": visible_raw,
        "SCRUFFY_GPU_IDS": visible_raw,
        "SCRUFFY_PHYSICAL_GPU_IDS": step_raw,
        "SCRUFFY_STEP_GPU_IDS": step_raw,
        "SCRUFFY_RESERVED_GPU_IDS": ",".join(
            str(gpu_id) for gpu_id in placement["gpu_ids"]
        ),
        "SCRUFFY_SLURM_JOB_ID": job_id,
        "SCRUFFY_SLURM_STEP_ID": step_id,
    }
    device_order = inherited.get("CUDA_DEVICE_ORDER")
    if device_order is not None:
        protected["CUDA_DEVICE_ORDER"] = device_order
    record = _runtime_placement_record(
        document,
        placement,
        expected=expected,
        job_id=job_id,
        step_id=step_id,
        step_gpus=step_gpus,
        visible=visible,
    )
    return protected, record


def _publish_runtime_placement(
    root: str, placement: dict[str, Any], record: dict[str, Any]
) -> tuple[Path, str]:
    relative = placement.get("runtime_placement")
    if not isinstance(relative, str) or not relative:
        raise ValueError("Slurm assignment is missing runtime placement provenance")
    root_path = Path(root)
    target = root_path / relative
    if target.parent != root_path / "jobs" / record["job_id"]:
        raise ValueError("runtime placement provenance path is outside its job directory")
    return target, create_immutable_json(target, record)


def current_node() -> str:
    return (
        os.environ.get("SCRUFFY_NODE")
        or os.environ.get("SLURMD_NODENAME")
        or socket.gethostname().split(".", 1)[0]
    )


def find_node_assignment(document: dict[str, Any], node_name: str) -> dict[str, Any]:
    assignments = list(document["assignment"])
    exact = [item for item in assignments if str(item["node"]) == node_name]
    if exact:
        return exact[0]

    short_name = node_name.split(".", 1)[0]
    short_matches = [
        item
        for item in assignments
        if str(item["node"]).split(".", 1)[0] == short_name
    ]
    if len(short_matches) == 1:
        return short_matches[0]
    if short_matches:
        raise ValueError(f"ambiguous short node name {node_name!r}")
    raise ValueError(f"no assignment for node {node_name!r}")


def execute_assignment(source: Path) -> None:
    with source.open(encoding="utf-8") as handle:
        document = json.load(handle)

    node_name = current_node()
    placement = find_node_assignment(document, node_name)
    command = [str(argument) for argument in document["argv"]]
    if not command:
        raise ValueError("job command is empty")

    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in document["env"].items()})
    root = str(Path(document["root"]).expanduser().resolve())
    job_id = str(document["job_id"])
    # Queue identity is controller-owned. Apply these after submitted values so
    # a workload cannot redirect reports or impersonate another job.
    environment["SCRUFFY_ROOT"] = root
    environment["SCRUFFY_JOB_ID"] = job_id
    environment["SCRUFFY_PROJECT"] = job_project(document)
    environment["SCRUFFY_EVENT_DIR"] = str(Path(root) / "reports" / job_id)
    environment["SCRUFFY_NODE"] = str(placement["node"])
    for name in _SCRUFFY_GPU_ENVIRONMENT:
        environment.pop(name, None)
    if document.get("launcher") == "slurm":
        protected_gpu_environment, record = _slurm_gpu_environment(document, placement)
        for name in _SLURM_GPU_ENVIRONMENT:
            if name in os.environ:
                environment[name] = os.environ[name]
            else:
                environment.pop(name, None)
        environment.update(protected_gpu_environment)
        runtime_placement, placement_sha256 = _publish_runtime_placement(
            root, placement, record
        )
        environment["SCRUFFY_RUNTIME_PLACEMENT"] = str(runtime_placement)
        environment["SCRUFFY_RUNTIME_PLACEMENT_SHA256"] = placement_sha256
    else:
        # Local development has no Slurm step to allocate devices. Keep using
        # the scheduler reservation, applied after submitted environment values.
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        visible_devices = ",".join(
            str(gpu_id) for gpu_id in placement["gpu_ids"]
        )
        environment["CUDA_VISIBLE_DEVICES"] = visible_devices
        environment["SCRUFFY_GPU_IDS"] = visible_devices
        environment["SCRUFFY_RESERVED_GPU_IDS"] = visible_devices

    os.chdir(document["cwd"])
    os.execvpe(command[0], command, environment)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: python -m scruffy.worker ASSIGNMENT.json", file=sys.stderr)
        return 2
    try:
        execute_assignment(Path(arguments[0]))
    except Exception as exc:
        print(f"scruffy worker: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
