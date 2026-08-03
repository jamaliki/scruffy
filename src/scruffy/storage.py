"""Durable, shared-filesystem storage primitives.

Clients only create immutable request, command, or workload-report files. The
controller is the single writer for state and events, which keeps coordination
understandable and avoids relying on a database on a network filesystem.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import AbstractSet, Any, BinaryIO, Iterator, TextIO

from .protocol import MAX_EVENT_BYTES, validate_event


LAYOUT_DIRECTORIES = ("requests", "commands", "jobs", "reports")


class StorageError(RuntimeError):
    """Raised when durable queue state is missing or inconsistent."""


class ConflictError(StorageError):
    """Raised when an idempotency key is reused with different content."""


class RequestConflict(ConflictError):
    """Raised when an idempotency key is reused for a different job."""


class ReportConflict(ConflictError):
    """Raised when an event ID is reused for a different workload report."""


class ControllerAlreadyRunning(StorageError):
    """Raised when a second controller tries to own the same queue."""


class UnsafeRecovery(StorageError):
    """Raised when restarting could duplicate work in the same allocation."""


def utc_now() -> str:
    """Return a lexically sortable UTC timestamp."""

    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def queue_id(root: Path) -> str:
    """Derive a stable, opaque ID without another mutable metadata file."""

    resolved = str(root.expanduser().resolve()).encode()
    return f"queue-{hashlib.sha256(resolved).hexdigest()[:12]}"


def ensure_layout(root: Path) -> Path:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for dirname in LAYOUT_DIRECTORIES:
        (root / dirname).mkdir(exist_ok=True)
    return root


def _fsync_directory(directory: Path) -> None:
    """Persist a directory entry where the platform supports directory fsync."""

    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_json(target: Path, value: Any) -> None:
    """Atomically replace a JSON file with a complete, flushed document."""

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(source: Path) -> Any:
    try:
        with source.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StorageError(f"cannot read {source}: {exc}") from exc


def canonical_job_identity(spec: dict[str, Any]) -> bytes:
    """Return the stable part of a job spec used for idempotency checks."""

    identity = {
        key: value
        for key, value in spec.items()
        if key not in {"job_id", "submitted_at"}
    }
    return json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def create_job_id(request_id: str | None = None) -> str:
    if request_id:
        digest = hashlib.sha256(request_id.encode()).hexdigest()[:20]
        return f"job-{digest}"
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    return f"job-{timestamp}-{uuid.uuid4().hex[:10]}"


def _existing_request(root: Path, job_id: str) -> dict[str, Any] | None:
    source = root / "requests" / job_id / "spec.json"
    if not source.exists():
        return None
    return read_json(source)


def submit_request(root: Path, spec: dict[str, Any]) -> tuple[str, bool]:
    """Durably enqueue a job without contacting or waiting for the controller.

    A complete temporary directory is renamed into place. Concurrent callers
    using the same job ID therefore see either the old request or the new one,
    never a partially written JSON document.
    """

    root = ensure_layout(root)
    job_id = str(spec["job_id"])
    destination = root / "requests" / job_id
    temporary = root / "requests" / f".{job_id}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        atomic_write_json(temporary / "spec.json", spec)
        _fsync_directory(temporary)
        try:
            os.rename(temporary, destination)
            _fsync_directory(destination.parent)
            return job_id, False
        except OSError as exc:
            if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise

        existing = _existing_request(root, job_id)
        if existing is None:
            raise StorageError(f"request {job_id} raced but is not readable")
        if canonical_job_identity(existing) != canonical_job_identity(spec):
            raise RequestConflict(
                f"request ID for {job_id} was already used for a different job"
            )
        return job_id, True
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def list_requests(
    root: Path, exclude: AbstractSet[str] = frozenset()
) -> list[dict[str, Any]]:
    request_root = ensure_layout(root) / "requests"
    requests: list[dict[str, Any]] = []
    for directory in sorted(request_root.iterdir()):
        if (
            directory.name in exclude
            or directory.name.startswith(".")
            or not directory.is_dir()
        ):
            continue
        spec_file = directory / "spec.json"
        if spec_file.exists():
            requests.append(read_json(spec_file))
    return requests


def find_request(root: Path, job_id: str) -> dict[str, Any] | None:
    return _existing_request(ensure_layout(root), job_id)


def _canonical_report_identity(report: dict[str, Any]) -> bytes:
    """Ignore retry time while detecting reuse of one producer event ID."""

    identity = {key: value for key, value in report.items() if key != "occurred_at"}
    return json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _report_directory(root: Path, job_id: str) -> Path:
    report_root = ensure_layout(root) / "reports"
    directory = report_root / job_id
    try:
        directory.mkdir()
    except FileExistsError:
        if not directory.is_dir():
            raise StorageError(f"report inbox {directory} is not a directory")
    else:
        _fsync_directory(report_root)
    return directory


def submit_report(root: Path, report: dict[str, Any]) -> tuple[str, bool]:
    """Durably spool one validated producer event, idempotently by event ID."""

    document = validate_event(report)
    event_id = document["event_id"]
    directory = _report_directory(root, document["job_id"])
    digest = hashlib.sha256(event_id.encode()).hexdigest()
    destination = directory / f"{digest}.json"
    accepted = directory.parent / ".accepted" / document["job_id"] / destination.name

    # A short per-job lock lets us use the existing write+replace primitive
    # without a check/replace race between concurrent publishers. Different
    # jobs never contend with one another.
    with (directory / ".inbox.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing_source = destination if destination.exists() else accepted
        if existing_source.exists():
            existing = read_json(existing_source)
            try:
                previous = validate_event(existing)
            except ValueError as exc:
                raise StorageError(
                    f"existing report for event {event_id!r} is invalid: {exc}"
                ) from exc
            if _canonical_report_identity(previous) != _canonical_report_identity(
                document
            ):
                raise ReportConflict(
                    f"event ID {event_id!r} was already used for a different report"
                )
            return event_id, True
        atomic_write_json(destination, document)
        return event_id, False


def list_reports(root: Path) -> list[tuple[Path, object | None]]:
    """List report files without allowing one corrupt file to abort a batch.

    ``None`` is an explicit unreadable/oversized sentinel. The controller can
    journal a rejection notice and remove that individual file while continuing
    with the rest of the inbox.
    """

    return [
        item
        for _, reports in report_streams(root)
        for item in reports
    ]


def _report_stream(directory: Path) -> Iterator[tuple[Path, object | None]]:
    """Lazily decode one job's inbox so controller work can stay bounded."""

    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.name.startswith(".") or not entry.name.endswith(".json"):
                continue
            source = Path(entry.path)
            try:
                if not entry.is_file():
                    continue
                document = (
                    read_json(source)
                    if entry.stat().st_size <= MAX_EVENT_BYTES
                    else None
                )
            except (OSError, StorageError):
                document = None
            yield source, document


def report_streams(
    root: Path,
) -> list[tuple[str, Iterator[tuple[Path, object | None]]]]:
    """Return one lazy report iterator per visible job inbox."""

    report_root = ensure_layout(root) / "reports"
    return [
        (directory.name, _report_stream(directory))
        for directory in sorted(report_root.iterdir())
        if not directory.name.startswith(".") and directory.is_dir()
    ]


def remove_report(source: Path) -> None:
    """Acknowledge a report while retaining its durable idempotency receipt."""

    directory = source.parent
    accepted_root = directory.parent / ".accepted"
    accepted_directory = accepted_root / directory.name
    with (directory / ".inbox.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not source.exists():
            return
        try:
            accepted_root.mkdir()
        except FileExistsError:
            pass
        else:
            _fsync_directory(accepted_root.parent)
        try:
            accepted_directory.mkdir()
        except FileExistsError:
            pass
        else:
            _fsync_directory(accepted_root)
        destination = accepted_directory / source.name
        if destination.exists():
            source.unlink()
        else:
            os.replace(source, destination)
            _fsync_directory(accepted_directory)
        _fsync_directory(directory)


def submit_command(root: Path, command: dict[str, Any]) -> str:
    """Create an immutable controller command and return its request ID."""

    root = ensure_layout(root)
    request_id = str(command.get("request_id") or uuid.uuid4().hex)
    document = {**command, "request_id": request_id}
    atomic_write_json(root / "commands" / f"{request_id}.json", document)
    return request_id


def list_commands(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    command_root = ensure_layout(root) / "commands"
    result: list[tuple[Path, dict[str, Any]]] = []
    for source in sorted(command_root.glob("*.json")):
        result.append((source, read_json(source)))
    return result


def remove_command(source: Path) -> None:
    source.unlink(missing_ok=True)
    _fsync_directory(source.parent)


def load_state(root: Path) -> dict[str, Any] | None:
    source = ensure_layout(root) / "state.json"
    return read_json(source) if source.exists() else None


def write_state(root: Path, state: dict[str, Any]) -> None:
    atomic_write_json(ensure_layout(root) / "state.json", state)


def open_journal(root: Path) -> TextIO:
    journal = ensure_layout(root) / "events.jsonl"
    if journal.exists() and journal.stat().st_size:
        with journal.open("rb+") as repair:
            repair.seek(-1, os.SEEK_END)
            if repair.read(1) != b"\n":
                # Preserve a torn record for diagnosis, but separate it from
                # the next complete event so subsequent records remain valid.
                repair.seek(0, os.SEEK_END)
                repair.write(b"\n")
                repair.flush()
                os.fsync(repair.fileno())
    return journal.open("a", encoding="utf-8", buffering=1)


def append_event(handle: TextIO, event: dict[str, Any], *, sync: bool) -> None:
    json.dump(event, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
    handle.flush()
    if sync:
        os.fsync(handle.fileno())


def read_event_page(
    root: Path,
    *,
    after: int = 0,
    offset: int = 0,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], int, bool]:
    """Read a bounded page and return its byte cursor and continuation flag."""

    if offset < 0 or (limit is not None and limit <= 0):
        raise ValueError("event offset must be non-negative and limit positive")
    journal = ensure_layout(root) / "events.jsonl"
    if not journal.exists():
        return [], offset, False
    if offset > journal.stat().st_size:
        offset = 0
    events: list[dict[str, Any]] = []
    next_offset = offset
    more = False
    with journal.open("rb") as handle:
        handle.seek(offset)
        while line := handle.readline():
            if not line.endswith(b"\n"):
                break
            try:
                event = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                next_offset = handle.tell()
                continue
            if int(event.get("seq", 0)) <= after:
                next_offset = handle.tell()
                continue
            if limit is not None and len(events) >= limit:
                more = True
                break
            events.append(event)
            next_offset = handle.tell()
    return events, next_offset, more


def read_events(root: Path, after: int = 0) -> list[dict[str, Any]]:
    """Read complete journal records, ignoring a torn final line after a crash."""

    return read_event_page(root, after=after)[0]


def journal_size(root: Path) -> int:
    source = ensure_layout(root) / "events.jsonl"
    return source.stat().st_size if source.exists() else 0


def journal_tail(root: Path) -> tuple[int, int]:
    """Find the last valid sequence by reading backward from the journal tail."""

    source = ensure_layout(root) / "events.jsonl"
    if not source.exists():
        return 0, 0
    end = source.stat().st_size
    position = end
    buffer = b""
    with source.open("rb") as handle:
        while position > 0:
            size = min(8192, position)
            position -= size
            handle.seek(position)
            buffer = handle.read(size) + buffer
            latest: tuple[int, int] | None = None
            line_end = position
            for line in buffer.splitlines(keepends=True):
                line_end += len(line)
                if not line.endswith(b"\n"):
                    continue
                try:
                    event = json.loads(line)
                    latest = int(event["seq"]), line_end
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
            if latest is not None:
                return latest
    # A cursor may only advance past complete records. In particular, do not
    # skip a torn first record that a concurrent writer may still complete.
    return 0, 0


def job_directory(root: Path, job_id: str) -> Path:
    directory = ensure_layout(root) / "jobs" / job_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_output(root: Path, relative_name: str, offset: int, length: int) -> str:
    source = ensure_layout(root) / relative_name
    try:
        with source.open("rb") as handle:
            handle.seek(offset)
            return handle.read(length).decode("utf-8", errors="replace")
    except OSError:
        return ""


def tail_bytes(source: Path, lines: int = 200) -> bytes:
    """Return the final lines without loading an unbounded log into memory."""

    if lines <= 0 or not source.exists():
        return b""
    block_size = 8192
    chunks: list[bytes] = []
    with source.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        newline_count = 0
        while position > 0 and newline_count <= lines:
            size = min(block_size, position)
            position -= size
            handle.seek(position)
            chunk = handle.read(size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")
    return b"".join(
        b"".join(reversed(chunks)).splitlines(keepends=True)[-lines:]
    )


@contextmanager
def controller_lock(root: Path) -> Iterator[BinaryIO]:
    """Hold the queue's advisory singleton lock for the controller lifetime."""

    lock_file = ensure_layout(root) / "controller.lock"
    handle = lock_file.open("a+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ControllerAlreadyRunning(
                f"a controller already owns {root}"
            ) from exc
        yield handle
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
