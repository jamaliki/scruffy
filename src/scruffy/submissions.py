"""Pure construction and validation of atomic workflow submissions."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .models import DEFAULT_PROJECT, NodeInventory, ResourceRequest, normalize_project_id
from .scheduler import request_can_ever_fit
from .storage import (
    create_job_id,
    create_submission_id,
    job_identity_digest,
    submission_identity_digest,
    utc_now,
)
from .workflows import validate_workflows

MAX_WORKFLOW_TASKS = 256
TASK_KEYS = frozenset(
    {
        "task_id",
        "request_id",
        "name",
        "argv",
        "cwd",
        "environment",
        "resources",
        "needs",
    }
)


def job_from_spec(spec: dict[str, Any], queue_order: int) -> dict[str, Any]:
    """Validate one immutable task spec and create its mutable lifecycle image."""

    request = ResourceRequest.from_dict(spec["resources"])
    argv = spec.get("argv")
    if not isinstance(argv, list) or not argv or not all(
        isinstance(item, str) and item for item in argv
    ):
        raise ValueError("argv must be a non-empty array of strings")
    environment = spec.get("env", {})
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ValueError("env must map strings to strings")
    cwd = Path(str(spec["cwd"]))
    if not cwd.is_absolute():
        raise ValueError("cwd must be absolute")
    project_id = normalize_project_id(spec.get("project_id"))
    job = {
        "id": str(spec["job_id"]),
        "request_id": spec.get("request_id"),
        "project_id": project_id,
        "name": str(spec["name"]),
        "state": "queued",
        "submitted_at": str(spec["submitted_at"]),
        "queue_order": queue_order,
        "request_digest": job_identity_digest(spec),
        "argv": argv,
        "cwd": str(cwd),
        "env": environment,
        "request": request.to_dict(),
        "assignment": None,
        "last_assignment": None,
        "attempt": 1,
        "started_at": None,
        "finished_at": None,
        "deadline_at": None,
        "exit_code": None,
        "signal": None,
        "reason": None,
        "error": None,
    }
    workflow_id = spec.get("workflow_id")
    task_id = spec.get("task_id")
    needs = spec.get("needs", [])
    if workflow_id is not None or task_id is not None or needs:
        if not isinstance(needs, list):
            raise ValueError("needs must be a JSON array")
        job.update(
            {
                "workflow_id": workflow_id,
                "task_id": task_id,
                "needs": copy.deepcopy(needs),
                "blockers": [],
                "dependency_gate_passed": False,
            }
        )
    return job


def _task_spec(
    task: Mapping[str, Any],
    *,
    index: int,
    workflow_request_id: str,
    workflow_id: str,
    project_id: str,
    submitted_at: str,
) -> dict[str, Any]:
    unexpected = set(task) - TASK_KEYS
    if unexpected:
        raise ValueError(f"tasks[{index}] has unexpected fields: {sorted(unexpected)!r}")
    task_id = task.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError(f"tasks[{index}].task_id must be a non-empty string")
    request_id = task.get("request_id", f"{workflow_request_id}/{task_id}")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError(f"tasks[{index}].request_id must be a non-empty string")
    name = task.get("name", task_id)
    argv = task.get("argv")
    cwd = task.get("cwd")
    environment = task.get("environment", {})
    resources = task.get("resources")
    needs = task.get("needs", [])
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"tasks[{index}].name must be a non-empty string")
    if not isinstance(argv, list) or not argv or not all(
        isinstance(item, str) and item for item in argv
    ):
        raise ValueError(f"tasks[{index}].argv must contain non-empty strings")
    if not isinstance(cwd, str) or not Path(cwd).is_absolute():
        raise ValueError(f"tasks[{index}].cwd must be an absolute path")
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ValueError(f"tasks[{index}].environment must map strings to strings")
    request = ResourceRequest.from_dict(resources)
    if not isinstance(needs, list) or not all(isinstance(item, dict) for item in needs):
        raise ValueError(f"tasks[{index}].needs must be a list of dependency objects")
    return {
        "v": 1,
        "job_id": create_job_id(request_id, project_id=project_id),
        "request_id": request_id,
        "name": name,
        "submitted_at": submitted_at,
        "argv": list(argv),
        "cwd": cwd,
        "env": dict(sorted(environment.items())),
        "resources": request.to_dict(),
        **({"project_id": project_id} if project_id != DEFAULT_PROJECT else {}),
        "workflow_id": workflow_id,
        "task_id": task_id,
        "needs": [dict(item) for item in needs],
    }


def workflow_submission(
    *,
    request_id: str,
    workflow_id: str,
    tasks: Sequence[Mapping[str, Any]],
    project_id: str = DEFAULT_PROJECT,
    submitted_at: str | None = None,
    inventory: Sequence[NodeInventory] | None = None,
) -> dict[str, Any]:
    """Build and fully preflight one explicit, all-or-nothing DAG envelope."""

    project_id = normalize_project_id(project_id)
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a non-empty string")
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        raise ValueError("workflow_id must be a non-empty string")
    if (
        isinstance(tasks, (str, bytes, Mapping))
        or not 1 <= len(tasks) <= MAX_WORKFLOW_TASKS
    ):
        raise ValueError(f"tasks must contain between 1 and {MAX_WORKFLOW_TASKS} jobs")
    accepted_at = submitted_at or utc_now()
    specs = [
        _task_spec(
            task,
            index=index,
            workflow_request_id=request_id,
            workflow_id=workflow_id,
            project_id=project_id,
            submitted_at=accepted_at,
        )
        for index, task in enumerate(tasks)
    ]
    task_ids = [str(spec["task_id"]) for spec in specs]
    request_ids = [str(spec["request_id"]) for spec in specs]
    job_ids = [str(spec["job_id"]) for spec in specs]
    for label, values in (
        ("task_id", task_ids),
        ("request_id", request_ids),
        ("job_id", job_ids),
    ):
        if len(set(values)) != len(values):
            raise ValueError(f"workflow contains duplicate {label} values")
    known_tasks = set(task_ids)
    missing = sorted(
        {
            str(need.get("task_id"))
            for spec in specs
            for need in spec["needs"]
            if need.get("task_id") not in known_tasks
        }
    )
    if missing:
        raise ValueError(f"workflow dependencies are missing tasks: {missing!r}")
    images = [job_from_spec(spec, index + 1) for index, spec in enumerate(specs)]
    validate_workflows(images)
    if inventory is not None:
        impossible = [
            str(spec["task_id"])
            for spec in specs
            if not request_can_ever_fit(
                inventory, ResourceRequest.from_dict(spec["resources"])
            )
        ]
        if impossible:
            raise ValueError(f"tasks cannot fit this allocation: {impossible!r}")
    document = {
        "v": 1,
        "kind": "workflow",
        "submission_id": create_submission_id(request_id, project_id=project_id),
        "request_id": request_id,
        "project_id": project_id,
        "workflow_id": workflow_id,
        "submitted_at": accepted_at,
        "jobs": specs,
    }
    document["identity_sha256"] = submission_identity_digest(document)
    return document


def submission_summary(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the bounded identity response shared by validation and submit."""

    jobs = document["jobs"]
    return {
        "submission_id": document["submission_id"],
        "project_id": document["project_id"],
        "workflow_id": document["workflow_id"],
        "identity_sha256": document["identity_sha256"],
        "tasks": [
            {
                "task_id": job["task_id"],
                "job_id": job["job_id"],
                "request_id": job["request_id"],
            }
            for job in jobs
        ],
    }
