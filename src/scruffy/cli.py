"""Command-line interface for Scruffy."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from . import __version__
from .client import (
    cancel_job,
    drain_queue,
    explain,
    observe,
    publish_event,
    resume_queue,
    status,
    submit_job,
    submit_workflow,
    summary,
    validate_workflow,
    wait_for_job,
)
from .controller import run_controller
from .dashboard import run_dashboard
from .models import (
    DEFAULT_PROJECT,
    TERMINAL_JOB_STATES,
    ResourceRequest,
    normalize_project_id,
)
from .protocol import EVENT_KINDS
from .slurm import (
    discover_slurm_allocation,
    discover_slurm_incarnation,
    load_inventory,
)
from .storage import StorageError, tail_window
from .summary import (
    BLOCKED_VIEW_STATES,
    QUEUE_VIEW_STATES,
    RUNNING_VIEW_STATES,
    compact_job_page,
    resource_view,
)


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _root(arguments: argparse.Namespace) -> Path:
    value = arguments.root or os.environ.get("SCRUFFY_ROOT")
    if not value:
        raise ValueError("set --root or SCRUFFY_ROOT")
    return Path(value)


def _project(
    arguments: argparse.Namespace, *, default: str | None = None
) -> str | None:
    """Resolve an optional project selector from the CLI or environment."""

    value = getattr(arguments, "project", None)
    if value is None:
        value = os.environ.get("SCRUFFY_PROJECT")
    if value is None:
        value = default
    return normalize_project_id(value) if value is not None else None


def _environment(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"environment override must be KEY=VALUE: {value!r}")
        key, setting = value.split("=", 1)
        if not key:
            raise ValueError("environment key must not be empty")
        result[key] = setting
    return result


def _command(arguments: argparse.Namespace) -> list[str]:
    command = list(arguments.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("submit requires a command after --")
    return command


def _needs(values: list[str]) -> list[dict[str, str]]:
    dependencies: list[dict[str, str]] = []
    for value in values:
        task_id, separator, condition = value.rpartition(":")
        if not separator:
            task_id, condition = value, "succeeded"
        elif condition not in {"succeeded", "terminal"}:
            raise ValueError(
                "dependency must be TASK_ID, TASK_ID:succeeded, or TASK_ID:terminal"
            )
        dependencies.append({"task_id": task_id, "condition": condition})
    return dependencies


def _json_object(value: str, label: str) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {exc.msg}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{label} must be a JSON object")
    return result


def _serve(arguments: argparse.Namespace) -> int:
    root = _root(arguments)
    if arguments.drain_before_end_seconds < 0:
        raise ValueError("--drain-before-end-seconds must be non-negative")
    slurm_job_id = arguments.slurm_job_id or os.environ.get("SLURM_JOB_ID")
    launcher = arguments.launcher
    if launcher == "auto":
        launcher = "slurm" if slurm_job_id else "local"
    if launcher == "slurm" and not slurm_job_id:
        raise ValueError("a Slurm job ID is required for the Slurm launcher")
    allocation_incarnation = None
    if arguments.inventory:
        inventory = tuple(load_inventory(Path(arguments.inventory)).values())
        if launcher == "slurm":
            allocation_incarnation = discover_slurm_incarnation(slurm_job_id or "")
    else:
        if launcher != "slurm":
            raise ValueError("--inventory is required outside a Slurm allocation")
        discovered, allocation_incarnation = discover_slurm_allocation(
            slurm_job_id=slurm_job_id,
            gpus_per_node=arguments.gpus_per_node,
            cpus_per_node=arguments.cpus_per_node,
            memory_gb_per_node=arguments.memory_gb_per_node,
        )
        inventory = tuple(discovered.values())
    if launcher == "local" and len(inventory) != 1:
        raise ValueError("the local launcher requires a one-node inventory")
    allocation_id = (
        arguments.allocation_id
        or slurm_job_id
        or f"local-{uuid.uuid4().hex[:12]}"
    )
    if launcher == "slurm" and allocation_id != slurm_job_id:
        raise ValueError("the Slurm allocation ID must equal its Slurm job ID")
    run_controller(
        root=root,
        inventory=inventory,
        launcher=launcher,
        allocation_id=allocation_id,
        slurm_job_id=slurm_job_id,
        allocation_incarnation=allocation_incarnation,
        poll_interval=arguments.poll_interval,
        cancel_grace=arguments.cancel_grace,
        start_paused=arguments.start_paused,
        drain_before_end_seconds=arguments.drain_before_end_seconds,
    )
    return 0


def _submit(arguments: argparse.Namespace) -> int:
    command = _command(arguments)
    gpus = arguments.gpus_per_node
    if gpus is None:
        raise ValueError("--gpus-per-node is required; use 0 for CPU-only work")
    request = ResourceRequest(
        nodes=arguments.nodes,
        gpus_per_node=gpus,
        cpus_per_node=(
            (14 * gpus if gpus else 1)
            if arguments.cpus_per_node is None
            else arguments.cpus_per_node
        ),
        memory_gb_per_node=(
            (128 * gpus if gpus else 4)
            if arguments.memory_gb_per_node is None
            else arguments.memory_gb_per_node
        ),
        time_limit_seconds=arguments.time_limit_seconds,
    )
    result = submit_job(
        _root(arguments),
        argv=command,
        name=arguments.name or Path(command[0]).name,
        cwd=Path(arguments.cwd or os.getcwd()),
        environment=_environment(arguments.env),
        request=request,
        request_id=arguments.request_id,
        project_id=_project(arguments, default=DEFAULT_PROJECT) or DEFAULT_PROJECT,
        workflow_id=arguments.workflow_id,
        task_id=arguments.task_id,
        needs=_needs(arguments.needs),
    )
    _json(result)
    return 0


def _workflow_file(arguments: argparse.Namespace) -> tuple[str, str, list[dict[str, Any]], str]:
    """Read the small JSON document shared by workflow validation and submit."""

    try:
        document = json.loads(Path(arguments.file).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"workflow file must contain valid JSON: {exc.msg}") from exc
    if not isinstance(document, dict):
        raise TypeError("workflow file must contain a JSON object")
    request_id = document.get("request_id")
    workflow_id = document.get("workflow_id")
    tasks = document.get("tasks")
    if not isinstance(request_id, str) or not isinstance(workflow_id, str):
        raise TypeError("workflow file requires string request_id and workflow_id")
    if not isinstance(tasks, list) or not all(isinstance(task, dict) for task in tasks):
        raise ValueError("workflow file tasks must be an array of objects")
    selected = _project(arguments, default=document.get("project_id") or DEFAULT_PROJECT)
    return request_id, workflow_id, tasks, selected or DEFAULT_PROJECT


def _validate_workflow(arguments: argparse.Namespace) -> int:
    request_id, workflow_id, tasks, project_id = _workflow_file(arguments)
    _json(
        validate_workflow(
            _root(arguments),
            request_id=request_id,
            workflow_id=workflow_id,
            tasks=tasks,
            project_id=project_id,
        )
    )
    return 0


def _submit_workflow(arguments: argparse.Namespace) -> int:
    request_id, workflow_id, tasks, project_id = _workflow_file(arguments)
    _json(
        submit_workflow(
            _root(arguments),
            request_id=request_id,
            workflow_id=workflow_id,
            tasks=tasks,
            project_id=project_id,
        )
    )
    return 0


def _status(arguments: argparse.Namespace) -> int:
    _json(
        status(
            _root(arguments),
            arguments.job_id,
            project_id=_project(arguments),
        )
    )
    return 0


def _summary(arguments: argparse.Namespace) -> int:
    _json(
        summary(
            _root(arguments), limit=arguments.limit, project_id=_project(arguments)
        )
    )
    return 0


def _job_list(arguments: argparse.Namespace) -> int:
    """Print one compact operational job view."""

    if arguments.offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if not 1 <= arguments.limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    project_id = _project(arguments)
    _json(
        compact_job_page(
            status(_root(arguments), project_id=project_id),
            states=arguments.job_states,
            offset=arguments.offset,
            limit=arguments.limit,
            project_id=project_id,
            include_elapsed=arguments.include_elapsed,
        )
    )
    return 0


def _resources(arguments: argparse.Namespace) -> int:
    """Print aggregate and per-node resource availability."""

    project_id = _project(arguments)
    _json(resource_view(status(_root(arguments), project_id=project_id)))
    return 0


def _explain(arguments: argparse.Namespace) -> int:
    _json(
        explain(
            _root(arguments), arguments.job_id, project_id=_project(arguments)
        )
    )
    return 0


def _report(arguments: argparse.Namespace) -> int:
    job_id = arguments.job_id or os.environ.get("SCRUFFY_JOB_ID")
    if not job_id:
        raise ValueError("set --job-id or SCRUFFY_JOB_ID")
    _json(
        publish_event(
            _root(arguments),
            job_id=job_id,
            kind=arguments.kind,
            data=_json_object(arguments.data_json, "--data-json"),
            event_id=arguments.event_id,
            occurred_at=arguments.occurred_at,
            source=_environment(arguments.source),
        )
    )
    return 0


def _observe(arguments: argparse.Namespace) -> int:
    root = _root(arguments)
    wait_seconds = arguments.wait
    if wait_seconds is None:
        wait_seconds = 30 if arguments.follow else 0
    if not arguments.follow:
        _json(
            observe(
                root,
                after=arguments.after,
                wait_seconds=wait_seconds,
                include_output=arguments.output,
                limit=arguments.limit,
                project_id=_project(arguments),
            )
        )
        return 0

    cursor = arguments.after
    first = True
    while True:
        response = observe(
            root,
            after=cursor,
            wait_seconds=wait_seconds,
            include_output=arguments.output,
            limit=arguments.limit,
            project_id=_project(arguments),
        )
        if first or response["reset"]:
            print(
                json.dumps({"kind": "snapshot", "data": response["snapshot"]}),
                flush=True,
            )
            first = False
        for event in response["events"]:
            print(json.dumps(event, sort_keys=True), flush=True)
        cursor = response["next_cursor"]


def _logs(arguments: argparse.Namespace) -> int:
    root = _root(arguments)
    job = status(root, arguments.job_id)
    streams = [arguments.stream] if arguments.stream else ["stdout", "stderr"]
    positions: dict[str, int] = {}
    sources: dict[str, Path] = {}
    for stream_name in streams:
        relative_name = job.get(stream_name, f"jobs/{arguments.job_id}/{stream_name}.log")
        source = root.expanduser().resolve() / relative_name
        sources[stream_name] = source
        data, positions[stream_name] = tail_window(source, arguments.tail)
        if data:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
    if not arguments.follow:
        return 0

    def flush_new_output() -> None:
        for stream_name, source in sources.items():
            try:
                with source.open("rb") as handle:
                    handle.seek(positions[stream_name])
                    data = handle.read()
            except FileNotFoundError:
                continue
            positions[stream_name] += len(data)
            if data:
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()

    while True:
        flush_new_output()
        job = status(root, arguments.job_id)
        if job["state"] in TERMINAL_JOB_STATES:
            flush_new_output()
            return 0
        time.sleep(0.2)


def _wait(arguments: argparse.Namespace) -> int:
    job = wait_for_job(_root(arguments), arguments.job_id, timeout=arguments.timeout)
    _json(job)
    if job["state"] == "succeeded":
        return 0
    if job["state"] == "failed" and isinstance(job.get("exit_code"), int):
        return max(1, min(int(job["exit_code"]), 125))
    return 1


def _cancel(arguments: argparse.Namespace) -> int:
    _json(cancel_job(_root(arguments), arguments.job_id))
    return 0


def _drain(arguments: argparse.Namespace) -> int:
    _json(drain_queue(_root(arguments)))
    return 0


def _resume(arguments: argparse.Namespace) -> int:
    _json(resume_queue(_root(arguments)))
    return 0


def _dashboard(arguments: argparse.Namespace) -> int:
    run_dashboard(
        str(_root(arguments)),
        port=arguments.port,
        connect_command=arguments.connect_command,
        remote_command=arguments.remote_command,
        open_browser=not arguments.no_open,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI and its concise operational help."""

    parser = argparse.ArgumentParser(
        prog="scruffy",
        description="A small GPU queue inside a multi-node Slurm allocation.",
    )
    parser.add_argument("--root", help="shared queue directory (or SCRUFFY_ROOT)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="subcommand", required=True)

    serve = commands.add_parser(
        "serve",
        help="run the allocation controller",
        description="Run the single controller for an allocation.",
    )
    serve.add_argument("--inventory", help="explicit JSON inventory file")
    serve.add_argument(
        "--launcher",
        choices=("auto", "local", "slurm"),
        default="auto",
        help="worker launcher (default: auto)",
    )
    serve.add_argument("--allocation-id", help="controller allocation identity")
    serve.add_argument("--slurm-job-id", help="outer Slurm allocation job ID")
    serve.add_argument(
        "--gpus-per-node",
        type=int,
        help="optional managed GPU cap per node (default: allocation)",
    )
    serve.add_argument(
        "--cpus-per-node",
        type=int,
        help="optional managed CPU cap per node (default: allocation)",
    )
    serve.add_argument(
        "--memory-gb-per-node",
        type=int,
        help="optional managed memory cap per node (default: allocation)",
    )
    serve.add_argument(
        "--poll-interval",
        type=float,
        default=0.2,
        help="controller poll seconds (default: 0.2)",
    )
    serve.add_argument(
        "--cancel-grace",
        type=float,
        default=30,
        help="local seconds before SIGKILL; Slurm uses scancel (default: 30)",
    )
    serve.add_argument(
        "--start-paused",
        action="store_true",
        help="recover state but require an explicit resume before launching jobs",
    )
    serve.add_argument(
        "--drain-before-end-seconds",
        type=float,
        default=900,
        help=(
            "stop new launches this many seconds before the allocation ends "
            "(default: 900; 0 disables)"
        ),
    )
    serve.set_defaults(handler=_serve)

    submit = commands.add_parser(
        "submit",
        help="enqueue without waiting for resources",
        description=(
            "Durably enqueue COMMAND and return immediately. Put COMMAND after --."
        ),
    )
    submit.add_argument("--name", help="display name; defaults to command name")
    submit.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="nodes requested atomically (default: 1)",
    )
    submit.add_argument(
        "--gpus-per-node",
        type=int,
        help="GPUs required on every node; use 0 explicitly for CPU-only work",
    )
    submit.add_argument(
        "--cpus-per-node", type=int, help="default: 14 times --gpus-per-node"
    )
    submit.add_argument(
        "--memory-gb-per-node",
        type=int,
        help="default: 128 times --gpus-per-node, or 4 for a CPU-only job",
    )
    submit.add_argument(
        "--time-limit-seconds",
        type=int,
        help="fail the job after this many execution seconds",
    )
    submit.add_argument("--cwd", help="worker directory; defaults to current directory")
    submit.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="worker override",
    )
    submit.add_argument("--project", help="project namespace (or SCRUFFY_PROJECT)")
    submit.add_argument("--request-id", help="idempotency key within the project")
    submit.add_argument("--workflow-id", help="workflow namespace for this task")
    submit.add_argument("--task-id", help="task name without ':'")
    submit.add_argument(
        "--needs",
        action="append",
        default=[],
        metavar="TASK[:CONDITION]",
        help="dependency condition: succeeded (default) or terminal",
    )
    submit.add_argument("command", nargs=argparse.REMAINDER, help="argv to execute")
    submit.set_defaults(handler=_submit)

    for command_name, help_text, handler in (
        (
            "validate-workflow",
            "preflight a complete workflow JSON document without submitting it",
            _validate_workflow,
        ),
        (
            "submit-workflow",
            "atomically enqueue a complete workflow JSON document",
            _submit_workflow,
        ),
    ):
        workflow_command = commands.add_parser(command_name, help=help_text)
        workflow_command.add_argument("file", help="workflow JSON document")
        workflow_command.add_argument(
            "--project",
            help="override project_id in the document (or SCRUFFY_PROJECT)",
        )
        workflow_command.set_defaults(handler=handler)

    show = commands.add_parser("status", help="show the queue or one job")
    show.add_argument("job_id", nargs="?", help="omit for the complete queue")
    show.add_argument("--project", help="show only this project")
    show.set_defaults(handler=_status)

    summary_parser = commands.add_parser(
        "summary",
        help="show a bounded allocation view for humans and agents",
    )
    summary_parser.add_argument(
        "--limit", type=int, default=20, help="jobs per section (default: 20)"
    )
    summary_parser.add_argument("--project", help="show only this project")
    summary_parser.set_defaults(handler=_summary)

    job_views = (
        (
            "queue",
            "list jobs waiting for admission or resources",
            QUEUE_VIEW_STATES,
            False,
        ),
        (
            "running",
            "list jobs currently holding resources",
            RUNNING_VIEW_STATES,
            True,
        ),
        (
            "blocked",
            "list jobs waiting on workflow dependencies",
            BLOCKED_VIEW_STATES,
            False,
        ),
    )
    for name, help_text, states, include_elapsed in job_views:
        job_list = commands.add_parser(name, help=help_text)
        job_list.add_argument("--project", help="show only this project")
        job_list.add_argument("--offset", type=int, default=0)
        job_list.add_argument("--limit", type=int, default=50)
        job_list.set_defaults(
            handler=_job_list,
            job_states=states,
            include_elapsed=include_elapsed,
        )

    resources = commands.add_parser(
        "resources", help="show per-node GPU, CPU, and memory availability"
    )
    resources.add_argument(
        "--project",
        help="label the project scope; availability remains allocation-wide",
    )
    resources.set_defaults(handler=_resources)

    explain_parser = commands.add_parser(
        "explain",
        help="explain one job and its dependency states",
    )
    explain_parser.add_argument("job_id", help="job to explain")
    explain_parser.add_argument("--project", help="require this job's project")
    explain_parser.set_defaults(handler=_explain)

    report = commands.add_parser("report", help="publish a semantic workload event")
    report.add_argument("kind", choices=sorted(EVENT_KINDS))
    report.add_argument("--job-id", help="defaults to SCRUFFY_JOB_ID")
    report.add_argument("--event-id", help="stable producer idempotency key")
    report.add_argument("--occurred-at", help="ISO 8601 producer timestamp")
    report.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="producer metadata",
    )
    report.add_argument(
        "--data-json",
        default="{}",
        metavar="OBJECT",
        help="bounded event payload (default: {})",
    )
    report.set_defaults(handler=_report)

    observe_parser = commands.add_parser(
        "observe",
        help="read or follow the shared allocation event stream",
        description=(
            "Return a snapshot and one event page. Without --after, start at the "
            "committed tail. --follow emits snapshot/event JSON lines continuously."
        ),
    )
    observe_parser.add_argument(
        "--after", help="opaque cursor returned by a prior call"
    )
    observe_parser.add_argument(
        "--wait",
        type=float,
        help="long-poll seconds; defaults to 0, or 30 with --follow",
    )
    observe_parser.add_argument(
        "--limit", type=int, default=1000, help="events per page (default: 1000)"
    )
    observe_parser.add_argument(
        "--output", action="store_true", help="expand job.output references into text"
    )
    observe_parser.add_argument(
        "--follow",
        action="store_true",
        help="emit snapshot/event JSON lines continuously",
    )
    observe_parser.add_argument("--project", help="emit only this project's jobs")
    observe_parser.set_defaults(handler=_observe)

    logs = commands.add_parser("logs", help="read a job's raw output")
    logs.add_argument("job_id", help="job whose logs to read")
    logs.add_argument(
        "--stream", choices=("stdout", "stderr"), help="select one stream"
    )
    logs.add_argument(
        "--tail", type=int, default=200, help="final lines per stream (default: 200)"
    )
    logs.add_argument(
        "--follow", action="store_true", help="follow until the job is terminal"
    )
    logs.set_defaults(handler=_logs)

    wait = commands.add_parser("wait", help="wait for one terminal job state")
    wait.add_argument("job_id", help="job to wait for")
    wait.add_argument("--timeout", type=float, help="maximum seconds to wait")
    wait.set_defaults(handler=_wait)

    cancel = commands.add_parser("cancel", help="asynchronously request cancellation")
    cancel.add_argument("job_id", help="job to cancel")
    cancel.set_defaults(handler=_cancel)

    drain = commands.add_parser(
        "drain",
        help="disable launches for the current allocation",
        description=(
            "Disable new launches until the allocation is replaced; running "
            "jobs continue and controller restarts preserve the drain."
        ),
    )
    drain.set_defaults(handler=_drain)

    resume = commands.add_parser(
        "resume",
        help="resume launches after controller recovery",
        description=(
            "Clear a controller-recovery launch pause. This does not override "
            "an allocation drain."
        ),
    )
    resume.set_defaults(handler=_resume)

    dashboard = commands.add_parser(
        "dashboard",
        help="open the read-only allocation dashboard",
        description="Serve a read-only Scruffy dashboard on 127.0.0.1.",
    )
    dashboard.add_argument(
        "--port", type=int, default=8765, help="local port (default: 8765)"
    )
    dashboard.add_argument(
        "--connect-command",
        help="invoke each read through this connector, such as tokyo-ssh",
    )
    dashboard.add_argument(
        "--remote-command",
        help="remote scruffy-mcp command used with --connect-command",
    )
    dashboard.add_argument(
        "--no-open", action="store_true", help="do not open a browser"
    )
    dashboard.set_defaults(handler=_dashboard)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Scruffy command-line client."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except KeyboardInterrupt:
        return 130
    except (
        KeyError,
        OSError,
        StorageError,
        subprocess.SubprocessError,
        TimeoutError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"scruffy: {exc}", file=sys.stderr)
        return 2
