"""Command-line interface for Scruffy."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from . import __version__
from .client import (
    TERMINAL_STATES,
    cancel_job,
    drain_queue,
    observe,
    status,
    submit_job,
    wait_for_job,
)
from .controller import run_controller
from .models import ResourceRequest
from .slurm import discover_slurm_inventory, load_inventory
from .storage import RequestConflict, StorageError, tail_bytes


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _root(arguments: argparse.Namespace) -> Path:
    value = arguments.root or os.environ.get("SCRUFFY_ROOT")
    if not value:
        raise ValueError("set --root or SCRUFFY_ROOT")
    return Path(value)


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


def _serve(arguments: argparse.Namespace) -> int:
    root = _root(arguments)
    if arguments.inventory:
        inventory = tuple(load_inventory(Path(arguments.inventory)).values())
    else:
        if arguments.gpus_per_node is None:
            raise ValueError(
                "--gpus-per-node is required when inventory is discovered"
            )
        inventory = tuple(
            discover_slurm_inventory(
                gpus_per_node=arguments.gpus_per_node,
                cpus_per_node=arguments.cpus_per_node,
                memory_gb_per_node=arguments.memory_gb_per_node,
            ).values()
        )
    launcher = arguments.launcher
    if launcher == "auto":
        launcher = "slurm" if os.environ.get("SLURM_JOB_ID") else "local"
    slurm_job_id = arguments.slurm_job_id or os.environ.get("SLURM_JOB_ID")
    if launcher == "slurm" and not slurm_job_id:
        raise ValueError("a Slurm job ID is required for the Slurm launcher")
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
        poll_interval=arguments.poll_interval,
        cancel_grace=arguments.cancel_grace,
    )
    return 0


def _submit(arguments: argparse.Namespace) -> int:
    command = _command(arguments)
    gpus = arguments.gpus_per_node
    request = ResourceRequest(
        nodes=arguments.nodes,
        gpus_per_node=gpus,
        cpus_per_node=(
            14 * gpus if arguments.cpus_per_node is None else arguments.cpus_per_node
        ),
        memory_gb_per_node=(
            128 * gpus
            if arguments.memory_gb_per_node is None
            else arguments.memory_gb_per_node
        ),
    )
    result = submit_job(
        _root(arguments),
        argv=command,
        name=arguments.name or Path(command[0]).name,
        cwd=Path(arguments.cwd or os.getcwd()),
        environment=_environment(arguments.env),
        request=request,
        request_id=arguments.request_id,
    )
    _json(result)
    return 0


def _status(arguments: argparse.Namespace) -> int:
    _json(status(_root(arguments), arguments.job_id))
    return 0


def _observe(arguments: argparse.Namespace) -> int:
    _json(
        observe(
            _root(arguments),
            after=arguments.after,
            wait_seconds=arguments.wait,
            include_output=arguments.output,
            limit=arguments.limit,
        )
    )
    return 0


def _watch(arguments: argparse.Namespace) -> int:
    root = _root(arguments)
    cursor = arguments.after
    first = True
    while True:
        response = observe(
            root,
            after=cursor,
            wait_seconds=arguments.wait,
            include_output=arguments.output,
            limit=arguments.limit,
        )
        if first:
            print(
                json.dumps({"kind": "snapshot", "data": response["snapshot"]}),
                flush=True,
            )
            first = False
        for event in response["events"]:
            print(json.dumps(event, sort_keys=True), flush=True)
        cursor = response["next_cursor"]
        if not arguments.follow:
            return 0


def _logs(arguments: argparse.Namespace) -> int:
    root = _root(arguments)
    job = status(root, arguments.job_id)
    streams = [arguments.stream] if arguments.stream else ["stdout", "stderr"]
    positions: dict[str, int] = {}
    for stream_name in streams:
        relative_name = job.get(stream_name, f"jobs/{arguments.job_id}/{stream_name}.log")
        source = root.expanduser().resolve() / relative_name
        data = tail_bytes(source, arguments.tail)
        if data:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        positions[stream_name] = source.stat().st_size if source.exists() else 0
    if not arguments.follow:
        return 0
    while True:
        for stream_name in streams:
            source = root.expanduser().resolve() / f"jobs/{arguments.job_id}/{stream_name}.log"
            if not source.exists():
                continue
            with source.open("rb") as handle:
                handle.seek(positions[stream_name])
                data = handle.read()
            positions[stream_name] += len(data)
            if data:
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
        job = status(root, arguments.job_id)
        if job["state"] in TERMINAL_STATES:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scruffy",
        description="A small GPU queue inside a multi-node Slurm allocation.",
    )
    parser.add_argument("--root", help="shared queue directory (or SCRUFFY_ROOT)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="subcommand", required=True)

    serve = commands.add_parser("serve", help="run the allocation controller")
    serve.add_argument("--inventory", help="explicit JSON inventory")
    serve.add_argument("--launcher", choices=("auto", "local", "slurm"), default="auto")
    serve.add_argument("--allocation-id")
    serve.add_argument("--slurm-job-id")
    serve.add_argument(
        "--gpus-per-node",
        type=int,
        help="full-node GPU count for automatic IDs 0..N-1",
    )
    serve.add_argument("--cpus-per-node", type=int, default=112)
    serve.add_argument("--memory-gb-per-node", type=int, default=1024)
    serve.add_argument("--poll-interval", type=float, default=0.2)
    serve.add_argument(
        "--cancel-grace",
        type=float,
        default=30,
        help="seconds before SIGKILL in local test mode (Slurm uses scancel)",
    )
    serve.set_defaults(handler=_serve)

    submit = commands.add_parser("submit", help="enqueue without waiting for resources")
    submit.add_argument("--name")
    submit.add_argument("--nodes", type=int, default=1)
    submit.add_argument("--gpus-per-node", type=int, default=1)
    submit.add_argument("--cpus-per-node", type=int)
    submit.add_argument("--memory-gb-per-node", type=int)
    submit.add_argument("--cwd")
    submit.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    submit.add_argument("--request-id", help="global idempotency key for this queue")
    submit.add_argument("command", nargs=argparse.REMAINDER)
    submit.set_defaults(handler=_submit)

    show = commands.add_parser("status", help="show the queue or one job")
    show.add_argument("job_id", nargs="?")
    show.set_defaults(handler=_status)

    observe_parser = commands.add_parser("observe", help="snapshot plus events after a cursor")
    observe_parser.add_argument("--after")
    observe_parser.add_argument("--wait", type=float, default=0)
    observe_parser.add_argument("--limit", type=int, default=1000)
    observe_parser.add_argument("--output", action="store_true")
    observe_parser.set_defaults(handler=_observe)

    watch = commands.add_parser("watch", help="stream the allocation event journal")
    watch.add_argument("--after")
    watch.add_argument("--wait", type=float, default=30)
    watch.add_argument("--limit", type=int, default=1000)
    watch.add_argument("--output", action="store_true")
    watch.add_argument("--follow", action="store_true")
    watch.set_defaults(handler=_watch)

    logs = commands.add_parser("logs", help="read a job's raw output")
    logs.add_argument("job_id")
    logs.add_argument("--stream", choices=("stdout", "stderr"))
    logs.add_argument("--tail", type=int, default=200)
    logs.add_argument("--follow", action="store_true")
    logs.set_defaults(handler=_logs)

    wait = commands.add_parser("wait", help="wait for one terminal job state")
    wait.add_argument("job_id")
    wait.add_argument("--timeout", type=float)
    wait.set_defaults(handler=_wait)

    cancel = commands.add_parser("cancel", help="asynchronously request cancellation")
    cancel.add_argument("job_id")
    cancel.set_defaults(handler=_cancel)

    drain = commands.add_parser("drain", help="stop launching queued jobs")
    drain.set_defaults(handler=_drain)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (KeyError, RequestConflict, StorageError, TimeoutError, ValueError) as exc:
        print(f"scruffy: {exc}", file=sys.stderr)
        return 2
