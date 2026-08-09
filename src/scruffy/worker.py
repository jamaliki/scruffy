"""Node-side worker that applies a controller assignment then execs the job."""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

from .models import job_project


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


def redirect_output(logs: dict[str, Any]) -> None:
    """Append stdout and stderr directly to controller-owned job logs.

    The descriptors are inherited by the workload after ``exec``. This keeps
    Slurm output independent of the controller-side ``srun`` client, which is
    necessarily lost during a controller restart.
    """

    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    descriptors: list[int] = []
    try:
        for stream in ("stdout", "stderr"):
            descriptors.append(os.open(os.fspath(logs[stream]), flags, 0o664))
        targets = (sys.stdout.fileno(), sys.stderr.fileno())
        for descriptor, target in zip(descriptors, targets):
            os.dup2(descriptor, target)
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


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
    environment["SCRUFFY_GPU_IDS"] = ",".join(
        str(gpu_id) for gpu_id in placement["gpu_ids"]
    )
    if document.get("provenance_path") is not None:
        environment["SCRUFFY_PROVENANCE_PATH"] = str(document["provenance_path"])
    if document.get("assignment_sha256") is not None:
        environment["SCRUFFY_ASSIGNMENT_SHA256"] = str(
            document["assignment_sha256"]
        )
    for field, variable in (
        ("workflow_id", "SCRUFFY_WORKFLOW_ID"),
        ("task_id", "SCRUFFY_TASK_ID"),
        ("attempt", "SCRUFFY_ATTEMPT"),
    ):
        value = document.get(field)
        if value is not None:
            environment[variable] = str(value)
    # Apply this last: jobs submitted through the API cannot choose another slot.
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(gpu_id) for gpu_id in placement["gpu_ids"]
    )

    logs = document.get("logs")
    if logs is not None:
        if not isinstance(logs, dict):
            raise ValueError("job logs must be a JSON object")
        redirect_output(logs)
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
