"""Placeholder batch step that holds an elastic lease for a Scruffy hub.

An elastic lease is an ordinary ``sbatch`` job whose only batch process is
this agent. Scruffy places its own work into the allocation as ``srun``
steps, so the agent itself does nothing except act as a dead-man's switch:
when the controller's lease heartbeat has been stale for longer than the
timeout *and* no worker step is live, it exits and Slurm ends the allocation.
It never exits while a worker step is running, and any uncertainty about the
live steps keeps the allocation.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from types import FrameType

HEARTBEAT_FILE = Path("leases") / "heartbeat.json"
DEFAULT_POLL_SECONDS = 30.0


def heartbeat_path(root: Path) -> Path:
    return root / HEARTBEAT_FILE


def read_heartbeat_age(root: Path, now: float | None = None) -> float | None:
    """Return seconds since the controller last wrote its lease heartbeat."""

    try:
        document = json.loads(heartbeat_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    at = document.get("at") if isinstance(document, dict) else None
    if isinstance(at, bool) or not isinstance(at, (int, float)):
        return None
    return (time.time() if now is None else now) - float(at)


def worker_steps_live(slurm_job_id: str) -> bool:
    """Return whether any numbered (worker) step is live in this allocation.

    The batch and extern steps do not count. Query errors count as live.
    """

    from .slurm import live_steps

    try:
        steps = live_steps(slurm_job_id)
    except Exception:  # noqa: BLE001 - uncertainty keeps the allocation
        return True
    prefix = f"{slurm_job_id}."
    return any(
        step.step_id.startswith(prefix) and step.step_id.removeprefix(prefix).isdecimal()
        for step in steps
    )


def should_exit(
    *,
    heartbeat_age: float | None,
    started_age: float,
    timeout: float,
    steps_live: bool,
) -> bool:
    """Exit only after the controller is silent past the timeout and idle."""

    age = started_age if heartbeat_age is None else heartbeat_age
    return age > timeout and not steps_live


def run_agent(
    root: Path,
    lease_id: str,
    *,
    heartbeat_timeout: float,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
) -> int:
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if not slurm_job_id:
        print("scruffy lease agent: SLURM_JOB_ID is not set", file=sys.stderr)
        return 2
    started = time.monotonic()
    stopping = False

    def stop(_signal: int, _frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(
        f"scruffy lease agent {lease_id} holding Slurm job {slurm_job_id}; "
        f"heartbeat timeout {heartbeat_timeout:g}s",
        flush=True,
    )
    while not stopping:
        time.sleep(poll_seconds)
        if stopping:
            break
        heartbeat_age = read_heartbeat_age(root)
        started_age = time.monotonic() - started
        silent = (started_age if heartbeat_age is None else heartbeat_age) > heartbeat_timeout
        if silent and should_exit(
            heartbeat_age=heartbeat_age,
            started_age=started_age,
            timeout=heartbeat_timeout,
            steps_live=worker_steps_live(slurm_job_id),
        ):
            print(
                f"scruffy lease agent {lease_id}: controller silent and no worker "
                "steps live; releasing the allocation",
                flush=True,
            )
            return 0
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scruffy-lease-agent",
        description="Hold an elastic Scruffy lease until its controller goes away.",
    )
    parser.add_argument("--root", required=True, help="shared queue directory")
    parser.add_argument("--lease-id", required=True, help="Scruffy lease ID")
    parser.add_argument(
        "--heartbeat-timeout",
        type=float,
        required=True,
        help="seconds of controller silence before an idle lease exits",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=DEFAULT_POLL_SECONDS,
        help=f"seconds between checks (default: {DEFAULT_POLL_SECONDS:g})",
    )
    arguments = parser.parse_args(argv)
    if arguments.heartbeat_timeout <= 0 or arguments.poll_seconds <= 0:
        parser.error("--heartbeat-timeout and --poll-seconds must be positive")
    return run_agent(
        Path(arguments.root),
        arguments.lease_id,
        heartbeat_timeout=arguments.heartbeat_timeout,
        poll_seconds=arguments.poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
