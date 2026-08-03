"""Pure workflow validation and dependency resolution.

Workflow task identities are local to a workflow.  Dependencies therefore name
only a ``task_id`` and can never silently bind to a task in another workflow.
This module deliberately knows nothing about persistence, processes, or GPU
placement; callers remain responsible for applying the returned decisions.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence

from .models import TERMINAL_JOB_STATES


Job = Mapping[str, object]
TaskKey = tuple[str, str]
Need = tuple[str, str]

DEPENDENCY_CONDITIONS = frozenset({"succeeded", "terminal"})


class WorkflowError(ValueError):
    """Raised when workflow metadata is ambiguous or invalid."""


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise WorkflowError(f"{label} must not have leading or trailing whitespace")
    return value


def _identity(job: Job, label: str) -> TaskKey | None:
    has_workflow = "workflow_id" in job
    has_task = "task_id" in job
    if has_workflow != has_task:
        raise WorkflowError(
            f"{label} must provide workflow_id and task_id together"
        )
    if not has_workflow:
        return None
    return (
        _identifier(job["workflow_id"], f"{label}.workflow_id"),
        _identifier(job["task_id"], f"{label}.task_id"),
    )


def _dependencies(job: Job, label: str) -> tuple[Need, ...]:
    raw_needs = job.get("needs", ())
    if isinstance(raw_needs, (str, bytes)) or not isinstance(raw_needs, Sequence):
        raise WorkflowError(f"{label}.needs must be an array")

    dependencies: list[Need] = []
    seen: set[str] = set()
    for index, raw_need in enumerate(raw_needs):
        need_label = f"{label}.needs[{index}]"
        if not isinstance(raw_need, Mapping):
            raise WorkflowError(f"{need_label} must be an object")
        expected = {"task_id", "condition"}
        if set(raw_need) != expected:
            raise WorkflowError(
                f"{need_label} must contain exactly task_id and condition"
            )
        task_id = _identifier(raw_need["task_id"], f"{need_label}.task_id")
        condition = raw_need["condition"]
        if not isinstance(condition, str) or condition not in DEPENDENCY_CONDITIONS:
            choices = ", ".join(sorted(DEPENDENCY_CONDITIONS))
            raise WorkflowError(f"{need_label}.condition must be one of: {choices}")
        if task_id in seen:
            raise WorkflowError(f"{label} has duplicate dependency on {task_id!r}")
        seen.add(task_id)
        dependencies.append((task_id, condition))
    return tuple(dependencies)


def _materialize(jobs: Iterable[Job]) -> tuple[Job, ...]:
    if isinstance(jobs, (str, bytes, Mapping)):
        raise WorkflowError("jobs must be an iterable of job objects")
    try:
        result = tuple(jobs)
    except TypeError as error:
        raise WorkflowError("jobs must be an iterable of job objects") from error
    for index, job in enumerate(result):
        if not isinstance(job, Mapping):
            raise WorkflowError(f"jobs[{index}] must be an object")
    return result


def _validate_cycles(needs: Mapping[TaskKey, tuple[Need, ...]]) -> tuple[TaskKey, ...]:
    # Missing tasks are allowed while a workflow is being submitted piecemeal.
    # They cannot participate in a cycle until their task request exists.
    indegree = {
        key: sum((key[0], task_id) in needs for task_id, _ in dependencies)
        for key, dependencies in needs.items()
    }
    dependents: dict[TaskKey, list[TaskKey]] = {key: [] for key in needs}
    for key, dependencies in needs.items():
        workflow_id, _ = key
        for dependency_id, _ in dependencies:
            dependency_key = (workflow_id, dependency_id)
            if dependency_key in dependents:
                dependents[dependency_key].append(key)

    ready = deque(key for key, count in indegree.items() if count == 0)
    ordered: list[TaskKey] = []
    while ready:
        key = ready.popleft()
        ordered.append(key)
        for dependent in dependents[key]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)

    if len(ordered) != len(needs):
        cyclic = sorted(key for key, count in indegree.items() if count > 0)
        details = ", ".join(f"{workflow_id}/{task_id}" for workflow_id, task_id in cyclic)
        raise WorkflowError(f"dependency cycle involving: {details}")
    return tuple(ordered)


def _validate_self_dependencies(
    needs: Mapping[TaskKey, tuple[Need, ...]],
) -> None:
    for key, dependencies in needs.items():
        workflow_id, task_id = key
        for dependency_id, _ in dependencies:
            if dependency_id == task_id:
                raise WorkflowError(
                    f"task {workflow_id!r}/{task_id!r} cannot depend on itself"
                )


def _validated_graph(
    jobs: Iterable[Job],
) -> tuple[
    dict[TaskKey, Job],
    dict[TaskKey, tuple[Need, ...]],
    tuple[TaskKey, ...],
]:
    ordered = _materialize(jobs)
    by_key: dict[TaskKey, Job] = {}
    needs: dict[TaskKey, tuple[Need, ...]] = {}

    for index, job in enumerate(ordered):
        label = f"jobs[{index}]"
        key = _identity(job, label)
        dependencies = _dependencies(job, label)
        if key is None:
            if dependencies:
                raise WorkflowError(
                    f"{label} cannot declare needs without workflow_id and task_id"
                )
            continue
        if key in by_key:
            workflow_id, task_id = key
            raise WorkflowError(
                f"duplicate task_id {task_id!r} in workflow {workflow_id!r}"
            )
        by_key[key] = job
        needs[key] = dependencies

    _validate_self_dependencies(needs)
    order = _validate_cycles(needs)
    return by_key, needs, order


def validate_workflows(jobs: Iterable[Job]) -> None:
    """Validate workflow identities and dependency graphs.

    Jobs outside a workflow may omit all three workflow fields.  A task must
    provide both identifiers, and task IDs need only be unique inside their
    workflow. Missing upstream tasks are valid because workflows are submitted
    asynchronously; unresolved references remain blockers.
    """

    _validated_graph(jobs)


def _blockers(
    key: TaskKey,
    by_key: Mapping[TaskKey, Job],
    needs: Mapping[TaskKey, tuple[Need, ...]],
    states: Mapping[TaskKey, object] | None = None,
) -> list[dict[str, object]]:
    workflow_id, _ = key
    blockers: list[dict[str, object]] = []
    for task_id, condition in needs[key]:
        dependency_key = (workflow_id, task_id)
        dependency = by_key.get(dependency_key)
        if dependency is None:
            blockers.append(
                {
                    "task_id": task_id,
                    "condition": condition,
                    "state": "missing",
                    "reason": "dependency_missing",
                }
            )
            continue
        state = dependency.get("state") if states is None else states[dependency_key]
        is_terminal = isinstance(state, str) and state in TERMINAL_JOB_STATES
        if condition == "terminal" and is_terminal:
            continue
        if condition == "succeeded" and state == "succeeded":
            continue
        reason = (
            "dependency_unsatisfied"
            if condition == "succeeded" and is_terminal
            else "dependency_pending"
        )
        blockers.append(
            {
                "task_id": task_id,
                "condition": condition,
                "state": state,
                "reason": reason,
            }
        )
    return blockers


def _target_key(job: Job, by_key: Mapping[TaskKey, Job]) -> TaskKey | None:
    if not isinstance(job, Mapping):
        raise WorkflowError("job must be an object")
    key = _identity(job, "job")
    dependencies = _dependencies(job, "job")
    if key is None:
        if dependencies:
            raise WorkflowError(
                "job cannot declare needs without workflow_id and task_id"
            )
        return None
    if key not in by_key:
        raise WorkflowError(
            f"task {key[0]!r}/{key[1]!r} is not present in jobs"
        )
    return key


def _resolution(
    key: TaskKey | None,
    by_key: Mapping[TaskKey, Job],
    needs: Mapping[TaskKey, tuple[Need, ...]],
    states: Mapping[TaskKey, object] | None = None,
) -> dict[str, object]:
    blockers = [] if key is None else _blockers(key, by_key, needs, states)
    unsatisfied = any(
        blocker["reason"] == "dependency_unsatisfied" for blocker in blockers
    )
    if unsatisfied:
        decision = "skipped"
        reason: str | None = "dependency_unsatisfied"
    elif blockers:
        decision = "blocked"
        reason = None
    else:
        decision = "ready"
        reason = None
    return {"decision": decision, "reason": reason, "blockers": blockers}


def resolve_blocked_jobs(jobs: Iterable[Job]) -> dict[TaskKey, dict[str, object]]:
    """Resolve all blocked tasks to a fixed point from one graph build."""

    by_key, needs, order = _validated_graph(jobs)
    states = {key: job.get("state") for key, job in by_key.items()}
    resolutions: dict[TaskKey, dict[str, object]] = {}
    for key in order:
        if states[key] != "blocked":
            continue
        resolution = _resolution(key, by_key, needs, states)
        resolutions[key] = resolution
        if resolution["decision"] == "ready":
            states[key] = "queued"
        elif resolution["decision"] == "skipped":
            states[key] = "skipped"
    return resolutions


def resolve_dependencies(job: Job, jobs: Iterable[Job]) -> dict[str, object]:
    """Resolve one job to ``ready``, ``blocked``, or ``skipped``."""

    by_key, needs, _ = _validated_graph(jobs)
    return _resolution(_target_key(job, by_key), by_key, needs)
