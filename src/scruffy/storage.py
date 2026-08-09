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
from collections.abc import Callable, Iterator, Sequence
from collections.abc import Set as AbstractSet
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from .models import DEFAULT_PROJECT, job_project, normalize_project_id
from .protocol import MAX_EVENT_BYTES, validate_event

LAYOUT_DIRECTORIES = ("requests", "commands", "jobs", "reports", "provenance")
LOCK_SHARDS = 64
MAX_TAIL_BYTES = 1024 * 1024
INVALID_REQUEST_DIGEST = "-"


class StorageError(RuntimeError):
    """Raised when durable queue state is missing or inconsistent."""


class TransientStorageError(StorageError):
    """Raised when an I/O failure says nothing about stored content validity."""


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

    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="milliseconds")


def queue_id(root: Path) -> str:
    """Return the persisted queue identity, deriving it before state exists.

    The state owns the identity once a queue has been created. This keeps
    observer cursors valid when the complete queue directory is moved.
    """

    resolved_root = root.expanduser().resolve()
    state_file = resolved_root / "state.json"
    if state_file.exists():
        state = read_json(state_file)
        identity = state.get("queue_id") if isinstance(state, dict) else None
        if not isinstance(identity, str) or not identity:
            raise StorageError(f"invalid queue identity in {state_file}")
        return identity

    resolved = str(resolved_root).encode()
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
    except OSError as exc:
        if exc.errno in {errno.EINVAL, errno.ENOTSUP, errno.EBADF}:
            return
        raise
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EBADF}:
            raise
    finally:
        os.close(descriptor)


def atomic_write_json(target: Path, value: Any, *, mode: int | None = None) -> None:
    """Atomically replace a JSON file with a complete, flushed document."""

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(source: Path) -> Any:
    try:
        with source.open(encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        raise TransientStorageError(f"cannot read {source}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StorageError(f"cannot read {source}: {exc}") from exc


def canonical_job_identity(spec: dict[str, Any]) -> bytes:
    """Return the stable part of a job spec used for idempotency checks."""

    identity = {
        key: value
        for key, value in spec.items()
        if key not in {"job_id", "submitted_at"}
    }
    # Adding projects must not invalidate receipts created before the field
    # existed. The explicit default and a missing legacy field are identical.
    if identity.get("project_id") == DEFAULT_PROJECT:
        identity.pop("project_id")
    return json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def job_identity_digest(spec: dict[str, Any]) -> str:
    """Return the stable fingerprint retained after request admission."""

    return hashlib.sha256(canonical_job_identity(spec)).hexdigest()


def canonical_submission_identity(document: dict[str, Any]) -> bytes:
    """Return the retry-stable identity of an atomic submission envelope."""

    identity = {
        key: value
        for key, value in document.items()
        if key not in {"submission_id", "submitted_at", "identity_sha256"}
    }
    jobs = identity.get("jobs")
    if isinstance(jobs, list):
        identity["jobs"] = [
            {
                key: value
                for key, value in job.items()
                if key not in {"job_id", "submitted_at"}
            }
            if isinstance(job, dict)
            else job
            for job in jobs
        ]
    if identity.get("project_id") == DEFAULT_PROJECT:
        identity.pop("project_id")
    return json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def submission_identity_digest(document: dict[str, Any]) -> str:
    """Return the durable idempotency digest for a submission envelope."""

    return hashlib.sha256(canonical_submission_identity(document)).hexdigest()


def create_job_id(
    request_id: str | None = None, *, project_id: str = DEFAULT_PROJECT
) -> str:
    """Create a stable ID, with idempotency keys scoped to one project."""

    project_id = normalize_project_id(project_id)
    if request_id:
        identity = request_id if project_id == DEFAULT_PROJECT else f"{project_id}\0{request_id}"
        digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
        return f"job-{digest}"
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    return f"job-{timestamp}-{uuid.uuid4().hex[:10]}"


def create_submission_id(request_id: str, *, project_id: str = DEFAULT_PROJECT) -> str:
    """Create the stable outer identity for an atomic multi-job submission."""

    project_id = normalize_project_id(project_id)
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("workflow request_id must be a non-empty string")
    identity = request_id if project_id == DEFAULT_PROJECT else f"{project_id}\0{request_id}"
    return f"submission-{hashlib.sha256(identity.encode()).hexdigest()[:20]}"


def _existing_request(root: Path, job_id: str) -> dict[str, Any] | None:
    source = root / "requests" / job_id / "spec.json"
    if not source.exists():
        return None
    document = read_json(source)
    if not isinstance(document, dict):
        raise StorageError(f"request {job_id!r} must contain a JSON object")
    return document


def _existing_submission(root: Path, submission_id: str) -> dict[str, Any] | None:
    source = root / "requests" / submission_id / "submission.json"
    if not source.exists():
        return None
    document = read_json(source)
    if not isinstance(document, dict):
        raise StorageError(f"submission {submission_id!r} must contain a JSON object")
    return document


def _key_lock(storage_root: Path, key: str) -> BinaryIO:
    """Open one of a fixed number of locks for a stable identity."""

    lock_root = storage_root / ".locks"
    _mkdir(lock_root)
    digest = hashlib.sha256(key.encode()).digest()
    shard = int.from_bytes(digest[:4], "big") % LOCK_SHARDS
    return (lock_root / f"{shard:02d}.lock").open("a+b")


def _request_receipt(request_root: Path, job_id: str) -> Path:
    digest = hashlib.sha256(job_id.encode()).hexdigest()
    return request_root / ".accepted" / digest[:2] / f"{digest}.json"


def _read_request_receipt(request_root: Path, job_id: str) -> dict[str, Any] | None:
    source = _request_receipt(request_root, job_id)
    if not source.exists():
        return None
    receipt = read_json(source)
    if not isinstance(receipt, dict) or receipt.get("job_id") != job_id:
        raise StorageError(f"invalid request receipt for {job_id}")
    return receipt


def submit_request(root: Path, spec: dict[str, Any]) -> tuple[str, bool]:
    """Durably enqueue a job without contacting or waiting for the controller.

    A complete temporary directory is renamed into place. Concurrent callers
    using the same job ID therefore see either the old request or the new one,
    never a partially written JSON document.
    """

    root = ensure_layout(root)
    job_id = str(spec["job_id"])
    request_root = root / "requests"
    destination = request_root / job_id
    identity_digest = job_identity_digest(spec)
    temporary = request_root / f".{job_id}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        atomic_write_json(temporary / "spec.json", spec)
        _fsync_directory(temporary)
        with _key_lock(request_root, job_id) as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            existing = _existing_request(root, job_id)
            if existing is not None:
                if canonical_job_identity(existing) != canonical_job_identity(spec):
                    raise RequestConflict(
                        f"request ID for {job_id} was already used for a different job"
                    )
                return job_id, True
            receipt = _read_request_receipt(request_root, job_id)
            if receipt is not None:
                if receipt.get("digest") == identity_digest:
                    return job_id, True
                raise RequestConflict(
                    f"request ID for {job_id} was already used for a different job"
                )
            os.rename(temporary, destination)
            _fsync_directory(destination.parent)
            return job_id, False
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def submit_submission(root: Path, document: dict[str, Any]) -> tuple[str, bool]:
    """Durably enqueue one all-or-nothing submission envelope.

    The envelope is one renamed directory, so every controller observes either
    all task specifications or none of them. The controller remains responsible
    for validating and admitting the complete set in one journal transaction.
    """

    root = ensure_layout(root)
    submission_id = str(document["submission_id"])
    request_root = root / "requests"
    destination = request_root / submission_id
    digest = submission_identity_digest(document)
    temporary = request_root / f".{submission_id}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        atomic_write_json(temporary / "submission.json", document)
        _fsync_directory(temporary)
        with _key_lock(request_root, submission_id) as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            existing = _existing_submission(root, submission_id)
            if existing is not None:
                if canonical_submission_identity(existing) != canonical_submission_identity(
                    document
                ):
                    raise RequestConflict(
                        f"submission ID {submission_id} was already used differently"
                    )
                return submission_id, True
            receipt = _read_request_receipt(request_root, submission_id)
            if receipt is not None:
                if receipt.get("digest") == digest:
                    return submission_id, True
                raise RequestConflict(
                    f"submission ID {submission_id} was already used differently"
                )
            os.rename(temporary, destination)
            _fsync_directory(request_root)
            return submission_id, False
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def list_submissions(
    root: Path,
) -> list[tuple[str, dict[str, Any] | None, bool]]:
    """List atomic envelopes and legacy single-job requests.

    The final flag is true for an atomic envelope and false for a legacy job.
    Transient reads are deferred; malformed durable content is returned as
    ``None`` so the controller can contain it without crashing the serve loop.
    """

    request_root = ensure_layout(root) / "requests"
    result: list[tuple[str, dict[str, Any] | None, bool]] = []
    for directory in sorted(request_root.iterdir()):
        if directory.name.startswith(".") or not directory.is_dir():
            continue
        submission_file = directory / "submission.json"
        source = submission_file if submission_file.exists() else directory / "spec.json"
        atomic = source == submission_file
        try:
            document = read_json(source)
        except TransientStorageError:
            continue
        except StorageError:
            document = None
        result.append(
            (
                directory.name,
                document if isinstance(document, dict) else None,
                atomic,
            )
        )
    return result


def list_requests(
    root: Path, exclude: AbstractSet[str] = frozenset()
) -> list[tuple[str, dict[str, Any] | None]]:
    """List requests, deferring transient reads and marking invalid documents."""

    requests: list[tuple[str, dict[str, Any] | None]] = []
    for submission_id, document, atomic in list_submissions(root):
        if not atomic:
            if submission_id not in exclude:
                requests.append((submission_id, document))
            continue
        jobs = document.get("jobs") if isinstance(document, dict) else None
        if not isinstance(jobs, list):
            continue
        for spec in jobs:
            job_id = spec.get("job_id") if isinstance(spec, dict) else None
            if isinstance(job_id, str) and job_id not in exclude:
                requests.append((job_id, spec))
    return requests


def find_request(root: Path, job_id: str) -> dict[str, Any] | None:
    return _existing_request(ensure_layout(root), job_id)


def request_pending(root: Path, job_id: str) -> bool:
    """Return whether the named request directory is still awaiting admission."""

    if (ensure_layout(root) / "requests" / job_id).is_dir():
        return True
    return any(candidate == job_id for candidate, _ in list_requests(root))


def accept_submission(
    root: Path, submission_id: str, *, identity_digest: str
) -> bool:
    """Replace an admitted atomic envelope with its compact outer receipt."""

    return _finish_request(root, submission_id, identity_digest)


def record_request_receipt(root: Path, job_id: str, identity_digest: str) -> None:
    """Persist a job idempotency receipt when admission came from a bundle."""

    request_root = ensure_layout(root) / "requests"
    with _key_lock(request_root, job_id) as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        receipt = _request_receipt(request_root, job_id)
        existing = _read_request_receipt(request_root, job_id)
        if existing is not None:
            if existing.get("digest") != identity_digest:
                raise StorageError(f"conflicting request receipt for {job_id}")
            return
        _mkdir(receipt.parent.parent)
        _mkdir(receipt.parent)
        atomic_write_json(
            receipt,
            {"v": 1, "job_id": job_id, "digest": identity_digest},
        )


def request_receipt_digest(root: Path, job_id: str) -> str | None:
    """Return a consumed job identity digest without exposing archive details."""

    receipt = _read_request_receipt(ensure_layout(root) / "requests", job_id)
    digest = receipt.get("digest") if receipt else None
    return digest if isinstance(digest, str) else None


def _finish_request(root: Path, job_id: str, digest: str | None) -> bool:
    request_root = ensure_layout(root) / "requests"
    with _key_lock(request_root, job_id) as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        source = request_root / job_id
        if not source.is_dir():
            return False
        receipt = _request_receipt(request_root, job_id)
        existing = _read_request_receipt(request_root, job_id)
        if existing is not None and existing.get("digest") != digest:
            raise StorageError(f"conflicting request receipt for {job_id}")
        if existing is None:
            _mkdir(receipt.parent.parent)
            _mkdir(receipt.parent)
            atomic_write_json(
                receipt,
                {"v": 1, "job_id": job_id, "digest": digest},
            )
        # An earlier attempt may have created but not durably published the
        # receipt. Re-sync its parent before deleting the only full request.
        _fsync_directory(receipt.parent)
        shutil.rmtree(source)
        _fsync_directory(request_root)
        return True


def accept_request(
    root: Path, job_id: str, *, identity_digest: str | None = None
) -> bool:
    """Replace one admitted request spec with a compact idempotency receipt."""

    if identity_digest is None:
        spec = _existing_request(ensure_layout(root), job_id)
        if spec is None:
            return False
        identity_digest = job_identity_digest(spec)
    return _finish_request(root, job_id, identity_digest)


def reject_request(root: Path, job_id: str) -> bool:
    """Remove an unreadable or identity-mismatched request and burn its ID."""

    return _finish_request(root, job_id, INVALID_REQUEST_DIGEST)


def accept_known_requests(
    root: Path,
    known_job_ids: AbstractSet[str],
    *,
    on_error: Callable[[str, Exception], None] | None = None,
) -> int:
    """Archive legacy request specs already represented in controller state."""

    accepted = 0
    for job_id, spec in list_requests(root):
        if job_id not in known_job_ids:
            continue
        try:
            accepted_request = (
                accept_request(root, job_id)
                if spec is not None and spec.get("job_id") == job_id
                else reject_request(root, job_id)
            )
        except (OSError, StorageError) as exc:
            if on_error is not None:
                on_error(job_id, exc)
            continue
        if accepted_request:
            accepted += 1
    return accepted


_ARCHIVED_JOB_FIELDS = (
    "id",
    "request_id",
    "name",
    "state",
    "submitted_at",
    "queue_order",
    "started_at",
    "finished_at",
    "exit_code",
    "signal",
    "reason",
    "error",
    "request",
    "last_assignment",
    "provenance",
    "deadline_at",
    "attempt",
    "resolved_dependencies",
    "workflow_id",
    "task_id",
    "needs",
    "workflow_invalid",
    "project_id",
)


def _archived_job(job: dict[str, Any]) -> dict[str, Any]:
    """Keep only terminal and workflow fields needed after hot retention."""

    return {
        **{key: job[key] for key in _ARCHIVED_JOB_FIELDS if key in job},
        "archived": True,
    }


def _workflow_archive(root: Path, workflow_id: str, project_id: str) -> Path:
    project_id = normalize_project_id(project_id)
    # Preserve the legacy path for the default project. Other projects use the
    # same compact index shape with a project-qualified identity.
    identity = workflow_id if project_id == DEFAULT_PROJECT else f"{project_id}\0{workflow_id}"
    digest = hashlib.sha256(identity.encode()).hexdigest()
    return ensure_layout(root) / "requests" / ".workflows" / digest[:2] / digest


def archive_terminal_job(root: Path, job: dict[str, Any]) -> None:
    """Durably move one terminal job's compact identity into the cold index."""

    job_id = str(job["id"])
    request_root = ensure_layout(root) / "requests"
    with _key_lock(request_root, job_id) as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        receipt_file = _request_receipt(request_root, job_id)
        receipt = _read_request_receipt(request_root, job_id)
        digest = (receipt or {}).get("digest") or job.get("request_digest")
        if not isinstance(digest, str):
            # Legacy jobs may predate retained request digests. Their ID was
            # nevertheless consumed, so fail closed rather than blocking all
            # future compaction or pretending an arbitrary retry is identical.
            digest = INVALID_REQUEST_DIGEST
        _mkdir(receipt_file.parent.parent)
        _mkdir(receipt_file.parent)
        archived = _archived_job(job)
        atomic_write_json(
            receipt_file,
            {"v": 1, "job_id": job_id, "digest": digest, "job": archived},
        )

        workflow_id = archived.get("workflow_id")
        task_id = archived.get("task_id")
        if isinstance(workflow_id, str) and isinstance(task_id, str):
            project_id = job_project(archived)
            workflow_directory = _workflow_archive(root, workflow_id, project_id)
            _mkdir(workflow_directory.parent.parent)
            _mkdir(workflow_directory.parent)
            _mkdir(workflow_directory)
            job_digest = hashlib.sha256(job_id.encode()).hexdigest()
            atomic_write_json(
                workflow_directory / f"{job_digest}.json",
                {
                    "v": 1,
                    "project_id": project_id,
                    "workflow_id": workflow_id,
                    "job": archived,
                },
            )


def find_archived_job(root: Path, job_id: str) -> dict[str, Any] | None:
    """Return an evicted terminal job's compact record, if retained."""

    receipt = _read_request_receipt(ensure_layout(root) / "requests", job_id)
    job = receipt.get("job") if receipt else None
    return job if isinstance(job, dict) else None


def list_archived_workflow(
    root: Path,
    workflow_id: str,
    *,
    project_id: str = DEFAULT_PROJECT,
    on_error: Callable[[Path, StorageError], None] | None = None,
) -> list[dict[str, Any]]:
    """Load one workflow, skipping corrupt entries and optionally reporting them."""

    project_id = normalize_project_id(project_id)
    directory = _workflow_archive(root, workflow_id, project_id)
    if not directory.exists():
        return []
    jobs: list[dict[str, Any]] = []
    for source in directory.glob("*.json"):
        try:
            document = read_json(source)
            try:
                document_project = normalize_project_id(
                    document.get("project_id") if isinstance(document, dict) else None
                )
            except ValueError as exc:
                raise StorageError(
                    f"invalid workflow archive entry {source}: {exc}"
                ) from exc
            if (
                not isinstance(document, dict)
                or document.get("workflow_id") != workflow_id
                or document_project != project_id
            ):
                raise StorageError(f"invalid workflow archive entry {source}")
            job = document.get("job")
            if not isinstance(job, dict):
                raise StorageError(f"workflow archive entry {source} has no job")
            if job_project(job) != project_id:
                raise StorageError(f"workflow archive entry {source} has wrong project")
        except TransientStorageError:
            raise
        except StorageError as exc:
            if on_error is not None:
                on_error(source, exc)
            continue
        jobs.append(job)
    return sorted(
        jobs,
        key=lambda job: (
            job.get("queue_order") if type(job.get("queue_order")) is int else -1
        ),
    )


def remove_cold_job_directories(root: Path, hot_job_ids: AbstractSet[str]) -> int:
    """Delete log directories for terminal jobs already moved out of hot state."""

    jobs_root = ensure_layout(root) / "jobs"
    removed = 0
    for directory in jobs_root.iterdir():
        if directory.is_dir() and directory.name not in hot_job_ids:
            shutil.rmtree(directory)
            removed += 1
    if removed:
        _fsync_directory(jobs_root)
    return removed


def _canonical_report_identity(report: dict[str, Any]) -> bytes:
    """Ignore retry time while detecting reuse of one producer event ID."""

    identity = {key: value for key, value in report.items() if key != "occurred_at"}
    return json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def report_identity_digest(report: dict[str, Any]) -> str:
    """Return the compact identity retained after a report is accepted."""

    return hashlib.sha256(_canonical_report_identity(report)).hexdigest()


def _mkdir(directory: Path) -> None:
    """Create and durably publish one directory, tolerating concurrent creators."""

    try:
        directory.mkdir()
    except FileExistsError:
        if not directory.is_dir():
            raise StorageError(f"{directory} is not a directory")
    else:
        _fsync_directory(directory.parent)


def _report_receipt_key(job_id: str, event_digest: str) -> str:
    return hashlib.sha256(f"{job_id}\0{event_digest}".encode()).hexdigest()


def _report_receipt_directory(report_root: Path, generation: int) -> Path:
    return report_root / ".accepted" / f".g{generation:06d}"


def _report_receipt_identity(
    report_root: Path, job_id: str, event_digest: str
) -> tuple[bool, str | None]:
    """Return a retained identity without scanning historical event keys."""

    accepted_root = report_root / ".accepted"
    if not accepted_root.exists():
        return False, None
    key = _report_receipt_key(job_id, event_digest)
    identities: set[str] = set()
    for directory in accepted_root.glob(".g[0-9]*"):
        receipt = directory / key
        if receipt.is_symlink():
            try:
                identities.add(os.readlink(receipt))
            except FileNotFoundError:
                continue
        elif os.path.lexists(receipt):
            raise StorageError(f"invalid report receipt {receipt}")
    if len(identities) > 1:
        raise StorageError(f"conflicting retained receipts for {job_id}/{event_digest}")
    if not identities:
        return False, None
    identity = identities.pop()
    return True, None if identity == "-" else identity


def report_was_accepted(root: Path, source: Path) -> tuple[bool, str | None]:
    """Check whether an inbox file is a stale copy of a committed report."""

    report_root = ensure_layout(root) / "reports"
    return _report_receipt_identity(report_root, source.parent.name, source.stem)


def submit_report(root: Path, report: dict[str, Any]) -> tuple[str, bool]:
    """Spool an event; its ID deduplicates across retained generations."""

    document = validate_event(report)
    event_id = document["event_id"]
    report_root = ensure_layout(root) / "reports"
    directory = report_root / document["job_id"]
    event_digest = hashlib.sha256(event_id.encode()).hexdigest()
    identity_digest = report_identity_digest(document)
    destination = directory / f"{event_digest}.json"
    accepted_directory = report_root / ".accepted" / document["job_id"]
    legacy_receipt = accepted_directory / destination.name

    # The lock lives outside the transient inbox, so the controller can remove
    # empty job directories without racing a concurrent publisher.
    with _key_lock(report_root, document["job_id"]) as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if destination.exists() or legacy_receipt.exists():
            existing_source = destination if destination.exists() else legacy_receipt
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
        retained, previous_digest = _report_receipt_identity(
            report_root, document["job_id"], event_digest
        )
        if retained:
            if previous_digest == identity_digest:
                return event_id, True
            raise ReportConflict(
                f"event ID {event_id!r} was already used for a different report"
            )
        _mkdir(directory)
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
            except (OSError, TransientStorageError):
                # Do not destroy or acknowledge a report because one read or
                # metadata lookup failed. A producer retry sees the same inbox.
                continue
            except StorageError:
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


def accept_reports(
    reports: Sequence[tuple[Path, str | None]], *, generation: int = 0
) -> None:
    """Retain one batch's identities with a single directory sync.

    Symlink targets hold identity digests without a file-content fsync. Inbox
    deletions need not be synced individually: a durable receipt makes a stale
    source harmless and lets a later controller remove it again.
    """

    by_root: dict[Path, list[tuple[Path, str | None]]] = {}
    for source, identity_digest in reports:
        by_root.setdefault(source.parent.parent, []).append((source, identity_digest))

    for report_root, items in by_root.items():
        accepted_root = report_root / ".accepted"
        receipt_directory = _report_receipt_directory(report_root, generation)
        _mkdir(accepted_root)
        _mkdir(receipt_directory)
        sources: list[Path] = []
        for source, identity_digest in items:
            job_id = source.parent.name
            with _key_lock(report_root, job_id) as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                if not source.exists():
                    continue
                receipt = receipt_directory / _report_receipt_key(job_id, source.stem)
                target = "-" if identity_digest is None else identity_digest
                if receipt.is_symlink():
                    if os.readlink(receipt) != target:
                        raise StorageError(
                            f"conflicting receipt for report {source.name!r}"
                        )
                elif os.path.lexists(receipt):
                    raise StorageError(f"invalid report receipt {receipt}")
                else:
                    os.symlink(target, receipt)
                sources.append(source)

        # This commits every receipt in the batch, including receipts found
        # after an interrupted earlier attempt, before any inbox source goes.
        if sources:
            _fsync_directory(receipt_directory)
        for source in sources:
            with _key_lock(report_root, source.parent.name) as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                source.unlink(missing_ok=True)
                legacy_lock = source.parent / ".inbox.lock"
                legacy_lock.unlink(missing_ok=True)
                try:
                    source.parent.rmdir()
                except OSError as exc:
                    if exc.errno not in {errno.ENOTEMPTY, errno.ENOENT}:
                        raise
        if sources:
            _fsync_directory(report_root)


def compact_report_receipts(root: Path) -> int:
    """Migrate legacy full receipts to the generation-scoped direct index."""

    report_root = ensure_layout(root) / "reports"
    marker = report_root / ".compact-receipts-v1.json"
    if marker.exists():
        return 0
    accepted_root = report_root / ".accepted"
    if not accepted_root.exists():
        atomic_write_json(marker, {"v": 1})
        return 0
    receipt_directory = _report_receipt_directory(report_root, 0)
    _mkdir(receipt_directory)
    migrated: list[tuple[Path, list[Path]]] = []
    for directory in sorted(accepted_root.iterdir()):
        if directory.name.startswith(".") or not directory.is_dir():
            continue
        entries: list[Path] = []
        with _key_lock(report_root, directory.name) as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            for source in directory.iterdir():
                if source.suffix == ".json" and source.is_file():
                    event_digest = source.stem
                    try:
                        document = validate_event(read_json(source))
                        identity_digest = report_identity_digest(document)
                    except (StorageError, TypeError, ValueError):
                        # Older controllers retained rejected input verbatim.
                        identity_digest = "-"
                else:
                    continue
                receipt = receipt_directory / _report_receipt_key(
                    directory.name, event_digest
                )
                if not receipt.is_symlink():
                    os.symlink(identity_digest, receipt)
                elif os.readlink(receipt) != identity_digest:
                    raise StorageError(f"conflicting legacy receipt {source}")
                entries.append(source)
        if entries:
            migrated.append((directory, entries))

    if migrated:
        _fsync_directory(receipt_directory)
    compacted = 0
    for directory, entries in migrated:
        with _key_lock(report_root, directory.name) as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            for source in entries:
                source.unlink(missing_ok=True)
                compacted += 1
            # Commit entry removal even if another file prevents the rmdir.
            _fsync_directory(directory)
            try:
                directory.rmdir()
            except OSError as exc:
                if exc.errno != errno.ENOTEMPTY:
                    raise
    if migrated:
        _fsync_directory(accepted_root)
    atomic_write_json(marker, {"v": 1})
    return compacted


def prune_report_receipts(root: Path, keep: AbstractSet[int]) -> None:
    """Expire telemetry idempotency together with old journal generations."""

    accepted_root = ensure_layout(root) / "reports" / ".accepted"
    if not accepted_root.exists():
        return
    changed = False
    for directory in accepted_root.glob(".g[0-9]*"):
        try:
            generation = int(directory.name[2:])
        except ValueError:
            continue
        if generation not in keep:
            shutil.rmtree(directory)
            changed = True
    if changed:
        _fsync_directory(accepted_root)


def sync_report_inboxes(root: Path) -> None:
    """Commit deferred inbox deletions before their receipts can expire."""

    report_root = ensure_layout(root) / "reports"
    for directory in report_root.iterdir():
        if not directory.name.startswith(".") and directory.is_dir():
            _fsync_directory(directory)
    _fsync_directory(report_root)


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


def journal_path(root: Path, generation: int = 0) -> Path:
    root = ensure_layout(root)
    if generation == 0:
        return root / "events.jsonl"
    return root / "journal" / f"events-{generation:06d}.jsonl"


def checkpoint_path(root: Path, generation: int) -> Path:
    return ensure_layout(root) / "journal" / f"checkpoint-{generation:06d}.json"


def latest_checkpoint(root: Path) -> tuple[int, dict[str, Any]] | None:
    journal_root = ensure_layout(root) / "journal"
    active = journal_root / "active.json"
    if not active.exists():
        return None
    marker = read_json(active)
    if not isinstance(marker, dict) or type(marker.get("generation")) is not int:
        raise StorageError(f"invalid journal activation marker {active}")
    generation = marker["generation"]
    source = checkpoint_path(root, generation)
    checkpoint = read_json(source) if source.exists() else None
    if not isinstance(checkpoint, dict) or not journal_path(root, generation).exists():
        raise StorageError(f"active journal generation {generation} is incomplete")
    return generation, checkpoint


def activate_journal_generation(root: Path, generation: int) -> None:
    """Publish which complete checkpoint may recover a missing state file."""

    if not checkpoint_path(root, generation).exists() or not journal_path(
        root, generation
    ).exists():
        raise StorageError(f"journal generation {generation} is incomplete")
    atomic_write_json(
        ensure_layout(root) / "journal" / "active.json",
        {"v": 1, "generation": generation},
    )


def create_journal_generation(
    root: Path, generation: int, checkpoint: dict[str, Any]
) -> None:
    """Durably stage a checkpoint and empty journal before state points to them."""

    journal_root = ensure_layout(root) / "journal"
    _mkdir(journal_root)
    event_file = journal_path(root, generation)
    checkpoint_file = checkpoint_path(root, generation)
    if event_file.exists() or checkpoint_file.exists():
        raise StorageError(f"journal generation {generation} already exists")
    atomic_write_json(checkpoint_file, checkpoint)
    temporary = event_file.with_name(f".{event_file.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, event_file)
        _fsync_directory(journal_root)
    finally:
        temporary.unlink(missing_ok=True)


def next_journal_generation(root: Path, current: int) -> int:
    """Choose a fresh generation even after an interrupted rotation."""

    journal_root = ensure_layout(root) / "journal"
    generations = [current]
    if journal_root.exists():
        for source in journal_root.iterdir():
            try:
                generations.append(int(source.stem.rsplit("-", 1)[1]))
            except (IndexError, ValueError):
                continue
    return max(generations) + 1


def prune_journal_generations(root: Path, keep: AbstractSet[int]) -> None:
    """Keep only the active journal and its reader-race fallback."""

    root = ensure_layout(root)
    root_changed = False
    journal_changed = False
    if 0 not in keep:
        legacy = root / "events.jsonl"
        if legacy.exists():
            legacy.unlink()
            root_changed = True
    journal_root = root / "journal"
    if not journal_root.exists():
        if root_changed:
            _fsync_directory(root)
        return
    for source in journal_root.iterdir():
        try:
            generation = int(source.stem.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if generation not in keep:
            source.unlink()
            journal_changed = True
    if root_changed:
        _fsync_directory(root)
    if journal_changed:
        _fsync_directory(journal_root)


def open_journal(root: Path, generation: int = 0) -> TextIO:
    journal = journal_path(root, generation)
    journal.parent.mkdir(parents=True, exist_ok=True)
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


def sync_file(handle: TextIO) -> None:
    """Flush and durably commit every prior write to an open file."""

    handle.flush()
    os.fsync(handle.fileno())


def read_event_page(
    root: Path,
    *,
    after: int = 0,
    offset: int = 0,
    limit: int | None = None,
    end_offset: int | None = None,
    generation: int = 0,
) -> tuple[list[dict[str, Any]], int, bool]:
    """Read a bounded page and return its byte cursor and continuation flag."""

    if (
        offset < 0
        or (limit is not None and limit <= 0)
        or (end_offset is not None and end_offset < 0)
    ):
        raise ValueError("event offset must be non-negative and limit positive")
    journal = journal_path(root, generation)
    if not journal.exists():
        return [], offset, False
    if offset > journal.stat().st_size:
        offset = 0
    events: list[dict[str, Any]] = []
    next_offset = offset
    more = False
    with journal.open("rb") as handle:
        if offset:
            handle.seek(offset - 1)
            if handle.read(1) != b"\n":
                offset = 0
                next_offset = 0
        handle.seek(offset)
        while end_offset is None or handle.tell() < end_offset:
            line = handle.readline()
            if not line or (end_offset is not None and handle.tell() > end_offset):
                break
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


def read_events(
    root: Path, after: int = 0, *, offset: int = 0, generation: int = 0
) -> list[dict[str, Any]]:
    """Read complete journal records, ignoring a torn final line after a crash."""

    return read_event_page(root, after=after, offset=offset, generation=generation)[0]


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


def tail_window(source: Path, lines: int = 200) -> tuple[bytes, int]:
    """Return bounded final lines and the atomically observed end offset."""

    if lines <= 0:
        try:
            return b"", source.stat().st_size
        except FileNotFoundError:
            return b"", 0
    try:
        with source.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            start = max(0, size - MAX_TAIL_BYTES)
            handle.seek(start)
            data = handle.read(size - start)
    except FileNotFoundError:
        return b"", 0
    return b"".join(data.splitlines(keepends=True)[-lines:]), size


def tail_bytes(source: Path, lines: int = 200) -> bytes:
    """Return final lines while reading at most one MiB from the log."""

    return tail_window(source, lines)[0]


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
