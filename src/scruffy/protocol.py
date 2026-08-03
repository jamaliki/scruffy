"""Validation for events reported by code running inside Scruffy jobs.

Workload reports are untrusted annotations: the controller may publish them,
but they can never change job lifecycle or resource ownership.  Keeping the
wire format here, independent of the controller, gives producers one small
stdlib-only contract to depend on.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from typing import Any


EVENT_KINDS = frozenset(
    {
        "workload.phase",
        "workload.progress",
        "workload.milestone",
        "workload.artifact",
        "workload.notice",
    }
)
EVENT_KEYS = frozenset(
    {"v", "event_id", "job_id", "occurred_at", "kind", "source", "data"}
)
MAX_EVENT_BYTES = 64 * 1024
MAX_EVENT_ID_CHARS = 256
MAX_JOB_ID_CHARS = 128
MAX_SOURCE_FIELDS = 32
MAX_SOURCE_KEY_CHARS = 64
MAX_SOURCE_VALUE_CHARS = 256
MAX_JSON_DEPTH = 32

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ProtocolError(ValueError):
    """Raised when a producer event does not match the public wire format."""


def _exact_keys(value: dict[object, object]) -> None:
    if not all(isinstance(key, str) for key in value):
        raise ProtocolError("workload event keys must be strings")
    keys = set(value)
    missing = EVENT_KEYS - keys
    extra = keys - EVENT_KEYS
    if not missing and not extra:
        return
    details = []
    if missing:
        details.append(f"missing {sorted(missing)!r}")
    if extra:
        details.append(f"unexpected {sorted(extra)!r}")
    raise ProtocolError(f"invalid workload event: {', '.join(details)}")


def _string(value: object, label: str, *, max_chars: int) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ProtocolError(f"{label} must not have leading or trailing whitespace")
    if len(value) > max_chars:
        raise ProtocolError(f"{label} exceeds {max_chars} characters")
    if any(ord(character) < 32 for character in value):
        raise ProtocolError(f"{label} must not contain control characters")
    return value


def _timestamp(value: object) -> str:
    timestamp = _string(value, "occurred_at", max_chars=64)
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolError("occurred_at must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProtocolError("occurred_at must include a UTC offset")
    return timestamp


def _source(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ProtocolError("source must be a JSON object")
    if len(value) > MAX_SOURCE_FIELDS:
        raise ProtocolError(f"source exceeds {MAX_SOURCE_FIELDS} fields")
    result: dict[str, str] = {}
    for key, item in value.items():
        source_key = _string(key, "source key", max_chars=MAX_SOURCE_KEY_CHARS)
        result[source_key] = _string(
            item,
            f"source[{source_key!r}]",
            max_chars=MAX_SOURCE_VALUE_CHARS,
        )
    return result


def _json_value(value: object, label: str, depth: int = 0) -> Any:
    """Copy a value while rejecting Python extensions to the JSON data model."""

    if depth > MAX_JSON_DEPTH:
        raise ProtocolError(f"{label} exceeds the maximum nesting depth")
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ProtocolError(f"{label} must not contain NaN or infinity")
        return value
    if isinstance(value, list):
        return [
            _json_value(item, f"{label}[{index}]", depth + 1)
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolError(f"{label} keys must be strings")
            result[key] = _json_value(item, f"{label}.{key}", depth + 1)
        return result
    raise ProtocolError(f"{label} contains a non-JSON value: {type(value).__name__}")


def validate_event(value: object) -> dict[str, Any]:
    """Validate and detach one workload event for durable publication.

    The returned dictionary contains only plain JSON values.  Its canonical
    on-disk representation, including the trailing newline, is capped at
    :data:`MAX_EVENT_BYTES` so one producer cannot flood the controller inbox
    with a single report.
    """

    if not isinstance(value, dict):
        raise ProtocolError("workload event must be a JSON object")
    _exact_keys(value)
    if type(value["v"]) is not int or value["v"] != 1:
        raise ProtocolError("workload event v must equal 1")

    event_id = _string(
        value["event_id"], "event_id", max_chars=MAX_EVENT_ID_CHARS
    )
    job_id = _string(value["job_id"], "job_id", max_chars=MAX_JOB_ID_CHARS)
    if not _JOB_ID_RE.fullmatch(job_id) or job_id in {".", ".."}:
        raise ProtocolError("job_id contains unsafe characters")
    kind = _string(value["kind"], "kind", max_chars=64)
    if kind not in EVENT_KINDS:
        raise ProtocolError(f"unsupported workload event kind {kind!r}")
    if not isinstance(value["data"], dict):
        raise ProtocolError("data must be a JSON object")

    event = {
        "v": 1,
        "event_id": event_id,
        "job_id": job_id,
        "occurred_at": _timestamp(value["occurred_at"]),
        "kind": kind,
        "source": _source(value["source"]),
        "data": _json_value(value["data"], "data"),
    }
    encoded = json.dumps(
        event,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if len(encoded) > MAX_EVENT_BYTES:
        raise ProtocolError(f"workload event exceeds {MAX_EVENT_BYTES} bytes")
    return event


__all__ = [
    "EVENT_KINDS",
    "MAX_EVENT_BYTES",
    "ProtocolError",
    "validate_event",
]
