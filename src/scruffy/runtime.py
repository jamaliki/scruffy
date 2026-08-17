"""In-memory process and output plumbing for the controller."""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from .models import NodeInventory
from .slurm import AllocationIncarnation, SlurmStep

MessageQueue = queue.SimpleQueue[dict[str, Any]]
MAX_OUTPUT_EVENT_BYTES = 65536


class OutputNotifier:
    """Coalesce log writes so noisy jobs cannot enqueue one object per chunk."""

    def __init__(self, messages: MessageQueue) -> None:
        self._messages = messages
        self._lock = threading.Lock()
        self._pending: dict[tuple[str, str], tuple[int, int]] = {}
        self._queued: set[tuple[str, str]] = set()

    def record(self, job_id: str, stream: str, offset: int, length: int) -> None:
        key = (job_id, stream)
        end = offset + length
        with self._lock:
            previous = self._pending.get(key)
            if previous is not None:
                offset, end = min(offset, previous[0]), max(end, previous[1])
            self._pending[key] = (offset, end)
            if key in self._queued:
                return
            self._queued.add(key)
            self._messages.put({"kind": "output_ready", "job_id": job_id, "stream": stream})

    def take(self, job_id: str, stream: str) -> tuple[int, int] | None:
        key = (job_id, stream)
        with self._lock:
            value = self._pending.get(key)
            if value is None:
                self._queued.discard(key)
                return None
            start, end = value
            chunk_end = min(end, start + MAX_OUTPUT_EVENT_BYTES)
            if chunk_end < end:
                self._pending[key] = (chunk_end, end)
                self._messages.put({"kind": "output_ready", "job_id": job_id, "stream": stream})
            else:
                self._pending.pop(key)
                self._queued.discard(key)
        return start, chunk_end - start

    def has_pending(self, job_id: str) -> bool:
        """Return whether a job still has log ranges awaiting journal events."""

        with self._lock:
            return any(key[0] == job_id for key in self._pending)


@dataclass(slots=True)
class RunningProcess:
    # Reattached Slurm steps have no handle to the old local ``srun`` client.
    process: subprocess.Popen[bytes] | None
    step_name: str | None
    readers: list[threading.Thread] = field(default_factory=list)
    closed_streams: set[str] = field(default_factory=set)
    output_offsets: dict[str, int] = field(default_factory=dict)
    cancel_deadline: float | None = None
    final_state: str | None = None
    final_reason: str | None = None
    client_signalled: bool = False
    step_cancelled: bool = False
    exit_seen_at: float | None = None
    absence_confirmations: int = 0
    last_absence_snapshot_at: float = 0.0
    last_accounting_snapshot_at: float = 0.0
    time_limit_deadline: float | None = None


@dataclass(slots=True)
class Controller:
    root: Path
    inventory: tuple[NodeInventory, ...]
    launcher: str
    allocation_id: str
    slurm_job_id: str | None
    poll_interval: float
    cancel_grace: float
    drain_before_end_seconds: float
    state: dict[str, Any]
    journal: TextIO
    messages: MessageQueue
    output: OutputNotifier
    allocation_incarnation: AllocationIncarnation | None = None
    running: dict[str, RunningProcess] = field(default_factory=dict)
    stopping: bool = False
    stop_announced: bool = False
    last_heartbeat: float = 0.0
    slurm_steps: tuple[SlurmStep, ...] = ()
    slurm_snapshot_at: float = 0.0
    last_slurm_query: float = 0.0
    slurm_query_error: str | None = None
    report_cursor: str | None = None
    workflow_signatures: dict[tuple[str, str], tuple[tuple[str, object], ...]] | None = None
    gpu_health_mode: str = "off"
    gpu_isolation: str = "node"
    gpu_health_interval: float = 10.0
    health_processes: dict[str, subprocess.Popen[bytes]] = field(default_factory=dict)
    health_step_name: str | None = None
    health_step_ids: dict[str, str] = field(default_factory=dict)
    health_launch_snapshot_at: dict[str, float] = field(default_factory=dict)
    health_retry_at: dict[str, float] = field(default_factory=dict)
    health_monitor_errors: dict[str, str] = field(default_factory=dict)
    health_ingest_errors: dict[str, str] = field(default_factory=dict)


def copy_stream(
    job_id: str,
    stream_name: str,
    source: BinaryIO,
    destination: Path,
    messages: MessageQueue,
    output: OutputNotifier,
) -> None:
    """Drain a launcher pipe into its canonical raw log."""

    try:
        with destination.open("ab") as target:
            while True:
                read_chunk = getattr(source, "read1", source.read)
                chunk = read_chunk(65536)
                if not chunk:
                    break
                offset = target.tell()
                target.write(chunk)
                target.flush()
                output.record(job_id, stream_name, offset, len(chunk))
    except Exception as exc:
        messages.put(
            {
                "kind": "output_error",
                "job_id": job_id,
                "stream": stream_name,
                "error": str(exc),
            }
        )
    finally:
        source.close()
        messages.put({"kind": "stream_closed", "job_id": job_id, "stream": stream_name})


def start_readers(
    running: RunningProcess,
    *,
    job_id: str,
    stdout_file: Path,
    stderr_file: Path,
    messages: MessageQueue,
    output: OutputNotifier,
) -> None:
    """Start both pipe drainers, closing any pipe whose thread cannot start."""

    if running.process is None:
        raise RuntimeError("local launcher process is missing")
    streams = (
        ("stdout", running.process.stdout, stdout_file),
        ("stderr", running.process.stderr, stderr_file),
    )
    for index, (stream_name, source, destination) in enumerate(streams):
        if source is None:
            raise RuntimeError(f"launcher {stream_name} pipe is missing")
        reader = threading.Thread(
            target=copy_stream,
            args=(job_id, stream_name, source, destination, messages, output),
            daemon=True,
        )
        try:
            reader.start()
        except Exception:
            for remaining_name, remaining_source, _ in streams[index:]:
                if remaining_source is not None:
                    remaining_source.close()
                running.closed_streams.add(remaining_name)
            raise
        running.readers.append(reader)


def signal_process(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    """Signal a local launcher process group without a shell."""

    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return
    except OSError:
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            pass


def stop_launcher(controller: Controller, running: RunningProcess) -> None:
    """Request launcher termination while retaining its resource assignment."""

    if running.client_signalled:
        return
    running.client_signalled = True
    if running.process is None or running.process.poll() is not None:
        return
    signal_process(running.process, signal.SIGTERM)
    if controller.launcher == "local":
        running.cancel_deadline = time.monotonic() + controller.cancel_grace


def abandon_processes(controller: Controller) -> None:
    """Best-effort cleanup after an unexpected controller exception.

    Persisted state keeps every assignment, so uncertainty remains fail-closed.
    """

    if controller.launcher == "slurm":
        # The replacement controller can recover these remote steps. Killing
        # them here would turn an ordinary controller crash into lost work.
        return

    for running in controller.running.values():
        stop_launcher(controller, running)
    deadline = time.monotonic() + controller.cancel_grace
    for running in controller.running.values():
        if running.process is None:
            continue
        try:
            running.process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            signal_process(running.process, signal.SIGKILL)
    for running in controller.running.values():
        if running.process is None:
            continue
        try:
            running.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
