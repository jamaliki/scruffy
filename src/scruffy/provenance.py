"""Immutable execution records derived from controller-owned facts.

The launch record exists before workload code runs.  Terminal facts necessarily
arrive later, so they live in a second immutable result record rather than
mutating something that was advertised as immutable to the workload.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import Assignment, job_project
from .storage import (
    StorageError,
    atomic_write_json,
    create_immutable_json,
    read_immutable_json,
)


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def provenance_files(root: Path, job_id: str) -> tuple[Path, Path]:
    """Return stable launch and result paths outside the removable log tree."""

    directory = root / "provenance" / job_id
    return directory / "launch.json", directory / "result.json"


def _write_immutable_record(path: Path, record: dict[str, Any]) -> str:
    """Create a provenance record once, accepting only an identical replay."""

    try:
        stored, digest = read_immutable_json(path)
    except FileNotFoundError:
        try:
            return create_immutable_json(path, record)
        except FileExistsError:
            stored, digest = read_immutable_json(path)
    if stored != record:
        raise StorageError(f"conflicting immutable provenance record {path}")
    return digest


def write_request_record(root: Path, job: dict[str, Any]) -> Path:
    """Persist the immutable accepted specification before admission is visible."""

    directory = root / "provenance" / str(job["id"])
    request_file = directory / "request.json"
    record = {
        "v": 1,
        "job": {
            key: job[key]
            for key in (
                "id",
                "request_id",
                "project_id",
                "name",
                "workflow_id",
                "task_id",
                "attempt",
                "recovery",
                "predecessor_job_id",
                "retry_reason",
            )
            if key in job and job[key] is not None
        },
        "submitted_at": job.get("submitted_at"),
        "argv": list(job["argv"]),
        "cwd": job["cwd"],
        "environment_sha256": _digest(job.get("env", {})),
        "request": job["request"],
        "needs": list(job.get("needs") or []),
        "wait_for": list(job.get("wait_for") or []),
    }
    digest = _digest(record)
    _write_immutable_record(request_file, record)
    job["provenance"] = {
        **(job.get("provenance") or {}),
        "request": str(request_file.relative_to(root)),
        "request_sha256": digest,
    }
    return request_file


def launch_record(
    *,
    root: Path,
    allocation_id: str,
    job: Mapping[str, Any],
    assignment: Assignment,
) -> tuple[dict[str, Any], str]:
    """Build the complete pre-execution record and its stable content digest."""

    _, result_file = provenance_files(root, str(job["id"]))
    identity = {
        key: job[key]
        for key in (
            "id",
            "request_id",
            "name",
            "workflow_id",
            "task_id",
            "attempt",
        )
        if key in job and job[key] is not None
    }
    record = {
        "v": 1,
        "job": {"project_id": job_project(job), **identity},
        "request_sha256": (job.get("provenance") or {}).get("request_sha256"),
        "allocation_id": allocation_id,
        "submitted_at": job.get("submitted_at"),
        "started_at": job.get("started_at"),
        "deadline_at": job.get("deadline_at"),
        "argv": list(job["argv"]),
        "cwd": job["cwd"],
        "environment_sha256": _digest(job.get("env", {})),
        "request": assignment.request.to_dict(),
        "dependencies": list(job.get("resolved_dependencies") or []),
        "conditions": list(job.get("resolved_conditions") or []),
        "assignment_sha256": _digest(assignment.to_dict()),
        "assignment": assignment.to_dict(),
        "result_path": str(result_file),
    }
    digest = _digest(record)
    return record, digest


def write_launch_record(
    root: Path,
    allocation_id: str,
    job: dict[str, Any],
    assignment: Assignment,
) -> Path:
    """Persist a mode-0444 launch record and attach its reference to the job."""

    launch_file, result_file = provenance_files(root, str(job["id"]))
    record, digest = launch_record(
        root=root,
        allocation_id=allocation_id,
        job=job,
        assignment=assignment,
    )
    _write_immutable_record(launch_file, record)
    job["provenance"] = {
        **(job.get("provenance") or {}),
        "launch": str(launch_file.relative_to(root)),
        "launch_sha256": digest,
        "assignment_sha256": record["assignment_sha256"],
        "result": str(result_file.relative_to(root)),
    }
    return launch_file


def write_result_record(root: Path, job: Mapping[str, Any]) -> Path:
    """Persist the immutable terminal companion to a job's launch record."""

    _, result_file = provenance_files(root, str(job["id"]))
    record = {
        "v": 1,
        "job_id": job["id"],
        "project_id": job_project(job),
        "launch_sha256": (job.get("provenance") or {}).get("launch_sha256"),
        "state": job.get("state"),
        "reason": job.get("reason"),
        "error": job.get("error"),
        "exit_code": job.get("exit_code"),
        "signal": job.get("signal"),
        "finished_at": job.get("finished_at"),
        "assignment": job.get("last_assignment"),
        "stdout_bytes": job.get("stdout_bytes"),
        "stderr_bytes": job.get("stderr_bytes"),
    }
    # Records without an attempt or authenticated launch belong to legacy job
    # images, which may be reused by old callers. They have no immutable
    # execution identity to protect; modern admitted jobs take the strict path.
    if type(job.get("attempt")) is not int and record["launch_sha256"] is None:
        atomic_write_json(result_file, record, mode=0o444)
    else:
        _write_immutable_record(result_file, record)
    return result_file
