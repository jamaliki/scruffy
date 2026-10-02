"""Pure selectors for operations that act on many jobs with one command.

Clients validate and preview a selector with the same functions the controller
uses to apply it, so a dry run and the applied command agree on what matches.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from .models import job_project, normalize_project_id

CANCELLABLE_STATES = frozenset({"queued", "blocked", "starting", "running", "finishing"})
MAX_SELECTOR_JOB_IDS = 10_000
MAX_SELECTOR_JOB_ID_CHARS = 128
MAX_SELECTOR_TEXT = 256
# The selector is retained in the command file and its immutable receipt.
MAX_SELECTOR_BYTES = 1024 * 1024
SELECTOR_FIELDS = frozenset(
    {
        "job_ids",
        "states",
        "project_id",
        "workflow_id",
        "workflow_id_prefix",
        "request_id_prefix",
        "name_prefix",
        "submitted_before",
    }
)
FILTER_FIELDS = SELECTOR_FIELDS - {"job_ids", "states"}


def _text(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_SELECTOR_TEXT
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{label} must be 1-{MAX_SELECTOR_TEXT} printable characters")
    return value


def _timestamp(value: object, label: str) -> datetime:
    text = _text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a UTC offset")
    return parsed


def cancel_selector(value: object) -> dict[str, Any]:
    """Validate and canonicalize one bulk-cancel selector.

    A selector names explicit ``job_ids``, filters, or both (their
    intersection). Filters must name at least one cancellable state so a
    selector such as ``{"project_id": "p"}`` cannot cancel running work by
    accident. Explicit IDs may omit states and then match any cancellable
    state.
    """

    if not isinstance(value, Mapping):
        raise TypeError("selector must be an object")
    unknown = set(value) - SELECTOR_FIELDS
    if unknown:
        raise ValueError(f"selector has unknown fields: {sorted(unknown)!r}")
    selector: dict[str, Any] = {}
    if value.get("job_ids") is not None:
        raw_ids = value["job_ids"]
        if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Iterable):
            raise TypeError("job_ids must be an array of job IDs")
        job_ids = [_text(job_id, "job_ids entry") for job_id in raw_ids]
        if not job_ids or len(job_ids) > MAX_SELECTOR_JOB_IDS:
            raise ValueError(f"job_ids must contain 1-{MAX_SELECTOR_JOB_IDS} IDs")
        if any("/" in job_id or len(job_id) > MAX_SELECTOR_JOB_ID_CHARS for job_id in job_ids):
            raise ValueError(
                f"job_ids entries must be at most {MAX_SELECTOR_JOB_ID_CHARS} "
                "characters without '/'"
            )
        selector["job_ids"] = sorted(set(job_ids))
    if value.get("states") is not None:
        raw_states = value["states"]
        if isinstance(raw_states, (str, bytes)) or not isinstance(raw_states, Iterable):
            raise TypeError("states must be an array of job states")
        states = sorted(set(raw_states))
        if not states or any(state not in CANCELLABLE_STATES for state in states):
            choices = ", ".join(sorted(CANCELLABLE_STATES))
            raise ValueError(f"states must be a non-empty subset of: {choices}")
        selector["states"] = states
    for field in sorted(FILTER_FIELDS):
        if value.get(field) is None:
            continue
        if field == "project_id":
            selector[field] = normalize_project_id(value[field])
        elif field == "submitted_before":
            _timestamp(value[field], field)
            selector[field] = value[field]
        else:
            selector[field] = _text(value[field], field)
    if "job_ids" not in selector:
        if not any(field in selector for field in FILTER_FIELDS | {"states"}):
            raise ValueError("selector must name job_ids or at least one filter")
        if "states" not in selector:
            raise ValueError("a filter selector must name at least one state")
    if len(json.dumps(selector, separators=(",", ":")).encode()) > MAX_SELECTOR_BYTES:
        raise ValueError(f"selector must encode to at most {MAX_SELECTOR_BYTES} bytes")
    return selector


def selector_summary(selector: Mapping[str, Any]) -> dict[str, Any]:
    """Return a selector for events, replacing an explicit ID list by its size."""

    summary = {key: value for key, value in selector.items() if key != "job_ids"}
    if "job_ids" in selector:
        summary["job_id_count"] = len(selector["job_ids"])
    return summary


def selector_matches(job: Mapping[str, Any], selector: Mapping[str, Any]) -> bool:
    """Return whether one job image matches a canonical selector."""

    job_ids = selector.get("job_ids")
    if job_ids is not None and job.get("id") not in job_ids:
        return False
    states = selector.get("states")
    if states is not None and job.get("state") not in states:
        return False
    project_id = selector.get("project_id")
    if project_id is not None and job_project(job) != project_id:
        return False
    workflow_id = selector.get("workflow_id")
    if workflow_id is not None and job.get("workflow_id") != workflow_id:
        return False
    for field, prefix_field in (
        ("workflow_id", "workflow_id_prefix"),
        ("request_id", "request_id_prefix"),
        ("name", "name_prefix"),
    ):
        prefix = selector.get(prefix_field)
        if prefix is not None and not str(job.get(field) or "").startswith(prefix):
            return False
    before = selector.get("submitted_before")
    if before is not None:
        submitted = job.get("submitted_at")
        if not isinstance(submitted, str):
            return False
        try:
            submitted_at = _timestamp(submitted, "submitted_at")
        except ValueError:
            return False
        if submitted_at >= _timestamp(before, "submitted_before"):
            return False
    return True


def cancel_preview(
    jobs: Iterable[Mapping[str, Any]], selector: Mapping[str, Any], *, sample: int = 20
) -> dict[str, Any]:
    """Summarize what a selector matches in one snapshot, without changing it."""

    by_state: dict[str, int] = {}
    cancellable: list[str] = []
    pending: list[str] = []
    seen: set[str] = set()
    for job in jobs:
        if not selector_matches(job, selector):
            continue
        job_id = str(job.get("id"))
        seen.add(job_id)
        state = str(job.get("state"))
        by_state[state] = by_state.get(state, 0) + 1
        if state in CANCELLABLE_STATES:
            cancellable.append(job_id)
        elif state == "submitted":
            pending.append(job_id)
    missing = [job_id for job_id in selector.get("job_ids") or () if job_id not in seen]
    return {
        "matched": sum(by_state.values()),
        "by_state": dict(sorted(by_state.items())),
        "would_cancel": len(cancellable),
        "pending_admission": len(pending),
        "not_in_hot_state": len(missing),
        "sample_job_ids": sorted(cancellable)[:sample],
    }
