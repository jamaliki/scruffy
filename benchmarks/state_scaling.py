"""Measure snapshot write and read costs on a production-shaped queue root.

The synthetic root mirrors the queue measured on 2026-10-02: about 8,500 jobs,
of which 7,455 are blocked on artifacts whose producers ended without
publishing them, plus 1,064 terminal jobs. The script runs against whichever
``scruffy`` is importable, so the same measurements can be taken for an older
release by pointing ``PYTHONPATH`` at its ``src`` directory:

    PYTHONPATH=src python benchmarks/state_scaling.py
    PYTHONPATH=/path/to/old/src python benchmarks/state_scaling.py --commands 50

It reports state size, ``status``/``summary`` latency, the number of snapshot
writes and the time taken to apply N cancel commands in one controller poll,
and, when the controller supports it, the ticks needed to shrink the root.
Numbers depend heavily on the filesystem; a shared NFS home is far slower than
a local disk, but the write counts and byte sizes carry over directly.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

import scruffy.controller as controller_module
import scruffy.state as state_module
from scruffy._compat import UTC
from scruffy.client import cancel_job, status, summary
from scruffy.models import NodeInventory, ResourceRequest
from scruffy.storage import queue_id, write_state

REQUEST = ResourceRequest(1, 1, 14, 128)
INVENTORY = (NodeInventory("local", tuple(range(8)), 112, 1024),)


def _ago(seconds: float) -> str:
    return (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat(
        timespec="milliseconds"
    )


def _job(job_id: str, state: str, order: int, workflow_id: str, task_id: str) -> dict[str, Any]:
    """Return one job image with production-like argv, env, and provenance."""

    digest = f"{order:064x}"
    job: dict[str, Any] = {
        "id": job_id,
        "project_id": "koochak",
        "request_id": f"agent/campaign/{workflow_id}/{task_id}/attempt-1",
        "name": f"{workflow_id}-{task_id}",
        "state": state,
        "submitted_at": _ago(200_000 - order),
        "queue_order": order,
        "attempt": 1,
        "argv": [
            "/shared/envs/koochak-2026-09/bin/python",
            "-m",
            "koochak.train",
            "--config",
            f"/shared/runs/campaign/{workflow_id}/config.yaml",
            "--checkpoint-dir",
            f"/shared/checkpoints/campaign/{workflow_id}/{task_id}",
            "--seed",
            str(order),
        ],
        "cwd": "/shared/code/koochak",
        "env": {
            f"KOOCHAK_SETTING_{index:02d}": f"/shared/values/{workflow_id}/{index:02d}"
            for index in range(48)
        },
        "request": REQUEST.to_dict(),
        "assignment": None,
        "request_digest": digest,
        "workflow_id": workflow_id,
        "task_id": task_id,
        "needs": [],
        "wait_for": [],
        "blockers": [],
        "dependency_gate_passed": state != "blocked",
        "provenance": {
            "request": f"provenance/{job_id}/request.json",
            "request_sha256": digest,
        },
    }
    if state not in {"blocked", "queued"}:
        job.update(
            {
                "started_at": _ago(150_000 - order),
                "finished_at": _ago(100_000 - order),
                "reason": {"cancelled": "cancelled", "failed": "application_exit"}.get(
                    state, "process_exit"
                ),
                "last_assignment": {
                    "job_id": job_id,
                    "request": REQUEST.to_dict(),
                    "reservations": [
                        {"node": "local", "gpu_ids": [0], "cpus": 14, "memory_gb": 128}
                    ],
                },
                "stdout": f"jobs/{job_id}/stdout.log",
                "stderr": f"jobs/{job_id}/stderr.log",
            }
        )
    return job


def build_root(root: Path, blocked: int) -> dict[str, dict[str, Any]]:
    """Write a legacy snapshot with stale artifact waiters and history."""

    jobs: dict[str, dict[str, Any]] = {}
    order = 0
    producer_states = ["cancelled"] * 700 + ["failed"] * 21 + ["succeeded"] * 24
    workflows = len(producer_states)
    for index, state in enumerate(producer_states):
        order += 1
        workflow_id = f"sweep-{index:04d}"
        jobs[f"job-train-{index:04d}"] = _job(
            f"job-train-{index:04d}", state, order, workflow_id, "train"
        )
    for index in range(blocked):
        order += 1
        workflow = index % workflows
        workflow_id = f"sweep-{workflow:04d}"
        job_id = f"job-eval-{index:05d}"
        job = _job(job_id, "blocked", order, workflow_id, f"eval-{index:05d}")
        job["reason"] = "waiting_for_dependencies"
        job["wait_for"] = [
            {
                "kind": "artifact",
                "task_id": "train",
                "artifact_id": f"checkpoint/step{index:06d}.pt",
            }
        ]
        job["blockers"] = [
            {
                "kind": "artifact",
                "task_id": "train",
                "artifact_id": f"checkpoint/step{index:06d}.pt",
                "state": jobs[f"job-train-{workflow:04d}"]["state"],
                "reason": "condition_pending",
            }
        ]
        jobs[job_id] = job
    for index, state in enumerate(["cancelled"] * 16 + ["succeeded"] * 303):
        order += 1
        job_id = f"job-history-{index:04d}"
        jobs[job_id] = _job(job_id, state, order, f"history-{index:04d}", "run")
    write_state(
        root,
        {
            "v": 1,
            "queue_id": queue_id(root),
            "last_seq": 0,
            "journal_generation": 0,
            "journal_offset": 0,
            "allocation": None,
            "nodes": {},
            "gpu_health": None,
            "jobs": jobs,
            "report_acks": {},
            "report_ack_v": 1,
            "next_queue_order": order,
            "archived_jobs": 0,
            "archived_counts": {},
            "archived_project_counts": {},
            "draining": False,
            "drain_requested": False,
            "launches_paused": False,
            "evacuation": None,
            "evacuation_requests": {},
            "evacuation_history": {},
            "evacuation_cancel_requests": {},
            "updated_at": _ago(0),
        },
    )
    return jobs


def _median_seconds(operation: Callable[[], object], repeats: int = 3) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def measure_reads(root: Path, job_id: str) -> dict[str, Any]:
    return {
        "state_bytes": (root / "state.json").stat().st_size,
        "hot_jobs": len(status(root)["jobs"]),
        "status_all_s": round(_median_seconds(lambda: status(root)), 4),
        "summary_s": round(_median_seconds(lambda: summary(root)), 4),
        "status_one_terminal_job_s": round(_median_seconds(lambda: status(root, job_id)), 5),
    }


def _controller(root: Path) -> Any:
    return controller_module._initialize_controller(
        root=root,
        inventory=INVENTORY,
        launcher="local",
        allocation_id="benchmark",
        slurm_job_id=None,
        poll_interval=0.2,
        cancel_grace=0,
        start_paused=True,
    )


def run(workdir: Path, blocked: int, commands: int) -> dict[str, Any]:
    root = workdir / "queue"
    jobs = build_root(root, blocked)
    result: dict[str, Any] = {
        "scruffy": getattr(controller_module, "__file__", "?"),
        "jobs": len(jobs),
        "blocked": blocked,
    }
    sample_terminal = "job-history-0000"
    result["before"] = measure_reads(root, sample_terminal)

    writes: list[float] = []
    original_write = state_module.write_state

    def timed_write(target: Path, state: dict[str, Any]) -> None:
        started = time.perf_counter()
        original_write(target, state)
        writes.append(time.perf_counter() - started)

    controller = _controller(root)
    try:
        targets = [job_id for job_id in jobs if job_id.startswith("job-eval-")][:commands]
        for job_id in targets:
            cancel_job(root, job_id)
        with mock.patch("scruffy.state.write_state", side_effect=timed_write):
            started = time.perf_counter()
            controller_module._ingest_commands(controller)
            elapsed = time.perf_counter() - started
        result["apply_commands"] = {
            "commands": len(targets),
            "snapshot_writes": len(writes),
            "seconds": round(elapsed, 3),
            "seconds_per_command": round(elapsed / max(len(targets), 1), 4),
            "mean_snapshot_write_s": round(statistics.mean(writes), 4) if writes else None,
        }

        apply_tick = getattr(controller_module, "_apply_tick", None)
        if apply_tick is not None:
            ticks = []
            started = time.perf_counter()
            for _ in range(100):
                writes.clear()
                before = (
                    len(controller.state["jobs"]),
                    sum(job["state"] == "blocked" for job in controller.state["jobs"].values()),
                )
                tick_started = time.perf_counter()
                with mock.patch("scruffy.state.write_state", side_effect=timed_write):
                    apply_tick(controller)
                ticks.append(
                    {
                        "seconds": round(time.perf_counter() - tick_started, 3),
                        "snapshot_writes": len(writes),
                        "hot_jobs": len(controller.state["jobs"]),
                    }
                )
                after = (
                    len(controller.state["jobs"]),
                    sum(job["state"] == "blocked" for job in controller.state["jobs"].values()),
                )
                if after == before:
                    break
            result["migration"] = {
                "ticks": len(ticks),
                "seconds": round(time.perf_counter() - started, 3),
                "max_snapshot_writes_per_tick": max(tick["snapshot_writes"] for tick in ticks),
                "max_tick_seconds": max(tick["seconds"] for tick in ticks),
                "hot_jobs_after": ticks[-1]["hot_jobs"],
            }
    finally:
        controller.journal.close()
    result["after"] = measure_reads(root, sample_terminal)
    result["bulk_cancel"] = measure_bulk_cancel(workdir / "bulk-queue", blocked)
    return result


def measure_bulk_cancel(root: Path, blocked: int) -> dict[str, Any] | None:
    """Cancel every stale waiter with one command, when the release has one."""

    try:
        from scruffy.client import cancel_jobs
        from scruffy.storage import command_receipt
    except ImportError:
        return None
    build_root(root, blocked)
    writes: list[float] = []
    original_write = state_module.write_state

    def counted_write(target: Path, state: dict[str, Any]) -> None:
        writes.append(1.0)
        original_write(target, state)

    controller = _controller(root)
    try:
        cancel_jobs(root, states=["blocked"], request_id="benchmark-cleanup")
        ticks = 0
        started = time.perf_counter()
        with mock.patch("scruffy.state.write_state", side_effect=counted_write):
            while command_receipt(root, "benchmark-cleanup") is None:
                controller_module._ingest_commands(controller)
                ticks += 1
        elapsed = time.perf_counter() - started
    finally:
        controller.journal.close()
    receipt = command_receipt(root, "benchmark-cleanup") or {}
    return {
        "commands": 1,
        "ticks": ticks,
        "snapshot_writes": len(writes),
        "seconds": round(elapsed, 3),
        "counts": (receipt.get("outcome") or {}).get("counts"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--blocked", type=int, default=7455)
    parser.add_argument("--commands", type=int, default=500)
    parser.add_argument("--workdir", help="directory for the synthetic root")
    arguments = parser.parse_args()
    if arguments.workdir:
        workdir = Path(arguments.workdir)
        workdir.mkdir(parents=True, exist_ok=False)
        print(json.dumps(run(workdir, arguments.blocked, arguments.commands), indent=2))
        return
    with tempfile.TemporaryDirectory() as temporary:
        print(json.dumps(run(Path(temporary), arguments.blocked, arguments.commands), indent=2))


if __name__ == "__main__":
    main()
