"""A GPU-hold sentinel: the workload of a controller-owned one-GPU step.

The worker exports ``SCRUFFY_GPU_UUIDS`` from its identity probe. On the
target device the sentinel holds it until the controller cancels the step;
on any other device it parks for a bounded time and exits, so that a
controller which stops mid-search cannot strand a healthy GPU. It never
touches the device itself.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from collections.abc import Sequence

NO_IDENTITY_EXIT = 3


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scruffy.gpu_sentinel")
    parser.add_argument("--target", required=True, help="physical GPU UUID to hold")
    parser.add_argument("--park-seconds", type=float, default=600.0)
    arguments = parser.parse_args(argv)

    bound = [
        item.strip().lower()
        for item in os.environ.get("SCRUFFY_GPU_UUIDS", "").split(",")
        if item.strip()
    ]
    if not bound:
        print("gpu sentinel: the worker reported no GPU identity", file=sys.stderr, flush=True)
        return NO_IDENTITY_EXIT
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, lambda _signum, _frame: stop.set())
    if arguments.target.strip().lower() in bound:
        print(f"gpu sentinel: holding {arguments.target}", flush=True)
        stop.wait()
    else:
        print(
            f"gpu sentinel: parked on {','.join(bound)} for at most "
            f"{arguments.park_seconds:g} s",
            flush=True,
        )
        stop.wait(max(0.0, arguments.park_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
