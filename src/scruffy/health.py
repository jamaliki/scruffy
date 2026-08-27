"""GPU identity, health evaluation, and scheduling quarantine policy."""

from __future__ import annotations

import copy
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime
from typing import Any

from ._compat import UTC
from .models import NodeInventory

HEALTH_MODES = frozenset({"off", "observe", "enforce"})
GPU_ISOLATION_MODES = frozenset({"gpu", "node"})
DEFAULT_BAD_SAMPLES_TO_QUARANTINE = 3
DEFAULT_SAMPLE_STALE_SECONDS = 45


class HealthError(ValueError):
    """Raised when a health sample or command is invalid."""


def empty_health_state(*, mode: str, isolation: str) -> dict[str, Any]:
    """Return a versioned GPU health projection for a queue snapshot."""

    if mode not in HEALTH_MODES:
        raise HealthError(f"unknown GPU health mode {mode!r}")
    if isolation not in GPU_ISOLATION_MODES:
        raise HealthError(f"unknown GPU isolation mode {isolation!r}")
    return {"v": 1, "mode": mode, "isolation": isolation, "nodes": {}}


def ensure_health_state(state: dict[str, Any], *, mode: str, isolation: str) -> dict[str, Any]:
    """Upgrade the bounded health projection and apply current controller policy."""

    health = state.get("gpu_health")
    if not isinstance(health, dict) or health.get("v") != 1:
        health = empty_health_state(mode=mode, isolation=isolation)
        state["gpu_health"] = health
    health["mode"] = mode
    health["isolation"] = isolation
    health.setdefault("nodes", {})
    return health


def bind_health_incarnation(health: dict[str, Any], incarnation_sha256: str | None) -> None:
    """Invalidate healthy freshness when the physical allocation changes."""

    if health.get("allocation_incarnation_sha256") == incarnation_sha256:
        return
    for node_state in health.get("nodes", {}).values():
        if isinstance(node_state, dict):
            node_state.pop("last_received_at", None)
            node_state.pop("last_sample_at", None)
            node_state.pop("cuda_probe", None)
    health["sample_errors"] = {}
    health["monitor"] = {"status": "pending", "error": None}
    health["allocation_incarnation_sha256"] = incarnation_sha256


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HealthError(f"{label} must be a non-empty string")
    return value.strip()


def _integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise HealthError(f"{label} must be a non-negative integer")
    return value


def _parse_timestamp(value: object) -> str:
    timestamp = _nonempty(value, "recorded_at")
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise HealthError("recorded_at must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise HealthError("recorded_at must include a timezone")
    return parsed.astimezone(UTC).isoformat(timespec="milliseconds")


def _timestamp_value(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _identity(row: Mapping[str, object], *, node: str, slot: int) -> dict[str, Any]:
    return {
        "node": node,
        "slot": slot,
        "uuid": _nonempty(row.get("uuid"), "GPU uuid"),
        "nvidia_index": _integer(row.get("nvidia_index"), "nvidia_index"),
        "minor_number": row.get("minor_number"),
        "pci_bus_id": row.get("pci_bus_id"),
        "serial": row.get("serial"),
        "name": row.get("name"),
        "driver_version": row.get("driver_version"),
        "vbios_version": row.get("vbios_version"),
    }


def _reasons(row: Mapping[str, object], cuda_failed: bool) -> list[str]:
    reasons: list[str] = []
    if cuda_failed:
        reasons.append("cuda_probe_failed")
    if row.get("thermal_slowdown") is True:
        reasons.append("thermal_slowdown")
    ecc = row.get("uncorrectable_ecc_errors")
    if type(ecc) is int and ecc > 0:
        reasons.append("uncorrectable_ecc_errors")
    if row.get("query_ok") is False:
        reasons.append("nvidia_query_failed")
    return reasons


def _sample_metrics(row: Mapping[str, object]) -> dict[str, object]:
    names = (
        "temperature_c",
        "power_draw_w",
        "power_limit_w",
        "uncorrectable_ecc_errors",
        "thermal_slowdown",
        "software_thermal_slowdown",
        "hardware_thermal_slowdown",
    )
    return {name: row[name] for name in names if name in row}


def _new_device(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        **identity,
        "status": "unknown",
        "bad_samples": 0,
        "good_samples": 0,
        "quarantined_at": None,
        "quarantine_reason": None,
        "quarantine_source": None,
    }


def ingest_health_sample(
    health: dict[str, Any],
    inventory: Sequence[NodeInventory],
    document: Mapping[str, object],
    *,
    bad_samples_to_quarantine: int = DEFAULT_BAD_SAMPLES_TO_QUARANTINE,
) -> list[dict[str, Any]]:
    """Project one monitor sample and return durable status transitions.

    Device identity is keyed by physical node and NVIDIA UUID. Logical slots
    are refreshed on every sample and may change after hardware maintenance.
    Quarantine is sticky until an explicit operator clear.
    """

    if type(bad_samples_to_quarantine) is not int or bad_samples_to_quarantine < 1:
        raise HealthError("bad_samples_to_quarantine must be positive")
    node_name = _nonempty(document.get("node"), "node")
    nodes = {item.name: item for item in inventory}
    if node_name not in nodes:
        raise HealthError(f"health sample names unknown node {node_name!r}")
    recorded_at = _parse_timestamp(document.get("recorded_at"))
    raw_devices = document.get("gpus")
    if isinstance(raw_devices, (str, bytes)) or not isinstance(raw_devices, Sequence):
        raise HealthError("gpus must be a sequence")
    rows = []
    for raw in raw_devices:
        if not isinstance(raw, Mapping):
            raise HealthError("each GPU sample must be an object")
        rows.append(raw)
    rows.sort(key=lambda row: _integer(row.get("nvidia_index"), "nvidia_index"))
    slots = nodes[node_name].gpu_ids
    if len(rows) != len(slots):
        raise HealthError(f"node {node_name!r} reported {len(rows)} GPUs; expected {len(slots)}")
    uuids = [_nonempty(row.get("uuid"), "GPU uuid") for row in rows]
    if len(set(uuids)) != len(uuids):
        raise HealthError("GPU UUIDs must be unique within a node sample")

    cuda_probe = document.get("cuda_probe")
    if not isinstance(cuda_probe, Mapping) or type(cuda_probe.get("ok")) is not bool:
        raise HealthError("cuda_probe must contain a boolean ok field")
    cuda_ok = bool(cuda_probe["ok"])
    init_ok = cuda_probe.get("init_ok", cuda_ok)
    raw_cuda_devices = cuda_probe.get("devices")
    cuda_devices = raw_cuda_devices if isinstance(raw_cuda_devices, Sequence) else ()
    failed_cuda_indices = {
        item.get("nvidia_index")
        for item in cuda_devices
        if isinstance(item, Mapping) and item.get("ok") is False
    }
    failed_cuda_uuids = {
        str(item.get("uuid")).lower()
        for item in cuda_devices
        if isinstance(item, Mapping)
        and item.get("ok") is False
        and isinstance(item.get("uuid"), str)
    }
    cuda_device_count = cuda_probe.get("device_count")
    count_mismatch = type(cuda_device_count) is int and cuda_device_count != len(rows)
    global_cuda_failure = init_ok is False or count_mismatch or (not cuda_ok and not cuda_devices)
    node_state = health.setdefault("nodes", {}).setdefault(node_name, {"devices": {}})
    previous_sample_at = node_state.get("last_sample_at")
    if isinstance(previous_sample_at, str) and _timestamp_value(recorded_at) <= _timestamp_value(
        previous_sample_at
    ):
        return []
    previous_devices = node_state.setdefault("devices", {})
    next_devices: dict[str, dict[str, Any]] = {}
    transitions: list[dict[str, Any]] = []
    for slot, row in zip(slots, rows, strict=True):
        identity = _identity(row, node=node_name, slot=slot)
        uuid = identity["uuid"]
        previous = previous_devices.get(uuid)
        device = copy.deepcopy(previous) if isinstance(previous, dict) else _new_device(identity)
        device.update(identity)
        prior_status = str(device.get("status", "unknown"))
        reasons = _reasons(
            row,
            global_cuda_failure
            or identity["uuid"].lower() in failed_cuda_uuids
            or (not failed_cuda_uuids and identity["nvidia_index"] in failed_cuda_indices),
        )
        quarantined = device.get("quarantined_at") is not None
        if reasons:
            device["bad_samples"] = int(device.get("bad_samples", 0)) + 1
            device["good_samples"] = 0
            if quarantined or device["bad_samples"] >= bad_samples_to_quarantine:
                device["status"] = "quarantined"
                if not quarantined:
                    device["quarantined_at"] = recorded_at
                    device["quarantine_reason"] = reasons
                    device["quarantine_source"] = "automatic"
            else:
                device["status"] = "suspect"
        else:
            device["bad_samples"] = 0
            device["good_samples"] = int(device.get("good_samples", 0)) + 1
            device["status"] = "quarantined" if quarantined else "healthy"
        device["last_sample_at"] = recorded_at
        device["last_reasons"] = reasons
        device["metrics"] = _sample_metrics(row)
        if device["status"] != prior_status:
            transitions.append(
                {
                    "node": node_name,
                    "uuid": uuid,
                    "slot": slot,
                    "from": prior_status,
                    "to": device["status"],
                    "reasons": reasons,
                    "at": recorded_at,
                }
            )
        next_devices[uuid] = device
    node_state.update(
        {
            "last_sample_at": recorded_at,
            "cuda_probe": dict(cuda_probe),
            "devices": next_devices,
        }
    )
    return transitions


def set_quarantine(
    health: dict[str, Any],
    *,
    node: str,
    uuid: str,
    quarantined: bool,
    at: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """Apply one explicit operator quarantine or clear and return its transition."""

    if reason is not None and (
        not isinstance(reason, str) or not reason.strip() or len(reason) > 1024
    ):
        raise HealthError("quarantine reason must be 1-1024 characters")
    node_state = health.get("nodes", {}).get(node)
    device = node_state.get("devices", {}).get(uuid) if isinstance(node_state, dict) else None
    if not isinstance(device, dict):
        raise HealthError(f"unknown GPU {uuid!r} on node {node!r}")
    previous = str(device.get("status", "unknown"))
    if quarantined:
        device["status"] = "quarantined"
        device["quarantined_at"] = at
        device["quarantine_reason"] = [reason or "operator_quarantine"]
        device["quarantine_source"] = "operator"
    else:
        device["quarantined_at"] = None
        device["quarantine_reason"] = None
        device["quarantine_source"] = None
        device["bad_samples"] = 0
        device["status"] = "healthy" if not device.get("last_reasons") else "suspect"
    return {
        "node": node,
        "uuid": uuid,
        "slot": device.get("slot"),
        "from": previous,
        "to": device["status"],
        "reasons": device.get("quarantine_reason") or device.get("last_reasons") or [],
        "at": at,
    }


def _reprobe_evidence(
    device: Mapping[str, object],
    *,
    now: datetime,
    stale_seconds: float,
) -> tuple[str, int]:
    sample_at = device.get("last_sample_at")
    if not isinstance(sample_at, str):
        raise HealthError("GPU has no health sample to revalidate")
    if _sample_is_stale(sample_at, now, stale_seconds):
        raise HealthError("GPU health sample is stale")
    reasons = device.get("last_reasons")
    if isinstance(reasons, list) and reasons:
        raise HealthError(f"latest GPU health sample is not clean: {reasons!r}")
    if reasons is not None and not isinstance(reasons, list):
        raise HealthError("GPU health sample has invalid reasons")
    good_samples = device.get("good_samples")
    if type(good_samples) is not int or good_samples < 1:
        raise HealthError("GPU has no successful health sample to revalidate")
    return sample_at, good_samples


def reprobe_quarantine(
    health: dict[str, Any],
    *,
    node: str,
    uuid: str,
    at: str,
    now: datetime | None = None,
    stale_seconds: float = DEFAULT_SAMPLE_STALE_SECONDS,
) -> dict[str, Any]:
    """Release an automatic quarantine after a recent clean monitor sample.

    The node-local monitor already performs the CUDA and telemetry probe on a
    fixed interval. Re-probing therefore means accepting its latest evidence,
    rather than launching a second overlapping GPU probe from the controller.
    Explicit operator quarantines remain sticky and require ``set_quarantine``
    with ``quarantined=False``.
    """

    if stale_seconds <= 0:
        raise HealthError("stale_seconds must be positive")
    node_state = health.get("nodes", {}).get(node)
    device = node_state.get("devices", {}).get(uuid) if isinstance(node_state, dict) else None
    if not isinstance(device, dict):
        raise HealthError(f"unknown GPU {uuid!r} on node {node!r}")
    if device.get("status") != "quarantined":
        raise HealthError(f"GPU {uuid!r} on node {node!r} is not quarantined")
    if device.get("quarantine_source") != "automatic":
        raise HealthError("GPU quarantine is operator-owned; use gpu-clear to override it")
    current = now or _timestamp_value(_parse_timestamp(at))
    sample_at, good_samples = _reprobe_evidence(
        device,
        now=current,
        stale_seconds=stale_seconds,
    )

    previous = str(device.get("status"))
    device["status"] = "healthy"
    device["quarantined_at"] = None
    device["quarantine_reason"] = None
    device["quarantine_source"] = None
    device["bad_samples"] = 0
    device["last_reasons"] = []
    return {
        "node": node,
        "uuid": uuid,
        "slot": device.get("slot"),
        "from": previous,
        "to": device["status"],
        "reasons": [],
        "at": at,
        "evidence": {
            "sample_at": sample_at,
            "good_samples": good_samples,
        },
    }


def unavailable_gpu_ids(
    health: Mapping[str, object],
    inventory: Sequence[NodeInventory],
    *,
    now: datetime | None = None,
    stale_seconds: float = DEFAULT_SAMPLE_STALE_SECONDS,
) -> dict[str, Collection[int]]:
    """Return slots unavailable to new GPU work under the configured policy."""

    enforce_automatic = health.get("mode") == "enforce"
    current = now or datetime.now(UTC)
    health_nodes = health.get("nodes")
    known_nodes = health_nodes if isinstance(health_nodes, Mapping) else {}
    unavailable: dict[str, Collection[int]] = {}
    for inventory_node in inventory:
        node_state = known_nodes.get(inventory_node.name)
        if enforce_automatic and (
            not isinstance(node_state, Mapping)
            or _sample_is_stale(
                node_state.get("last_received_at") or node_state.get("last_sample_at"),
                current,
                stale_seconds,
            )
        ):
            unavailable[inventory_node.name] = inventory_node.gpu_ids
            continue
        if not isinstance(node_state, Mapping):
            continue
        raw_devices = node_state.get("devices")
        devices = raw_devices.values() if isinstance(raw_devices, Mapping) else ()
        quarantined = {
            device.get("uuid")
            for device in devices
            if isinstance(device, Mapping)
            and device.get("status") == "quarantined"
            and (enforce_automatic or device.get("quarantine_source") == "operator")
        }
        if not quarantined:
            continue
        if health.get("isolation") != "gpu":
            unavailable[inventory_node.name] = inventory_node.gpu_ids
            continue

        quarantined_slots: set[int] = set()
        unmappable = False
        for device in devices:
            if not isinstance(device, Mapping) or device.get("status") != "quarantined":
                continue
            slot = device.get("slot")
            if type(slot) is not int or slot not in inventory_node.gpu_ids:
                # A quarantine without a current logical-slot mapping cannot
                # be safely expressed as a per-GPU Slurm reservation.
                unmappable = True
                break
            quarantined_slots.add(slot)
        unavailable[inventory_node.name] = (
            inventory_node.gpu_ids
            if unmappable
            else tuple(sorted(quarantined_slots))
        )
    return unavailable


def nodes_requiring_exact_gpu_binding(
    health: Mapping[str, object],
    inventory: Sequence[NodeInventory],
    *,
    now: datetime | None = None,
    stale_seconds: float = DEFAULT_SAMPLE_STALE_SECONDS,
) -> frozenset[str]:
    """Return nodes where a GPU job must verify its physical Slurm mapping.

    Count-based Slurm allocation is safe on healthy nodes.  Exact binding is
    needed only when ``gpu`` isolation is actively excluding a subset of a
    node's GPUs for quarantine; a whole-node hold cannot be assigned and an
    unmappable quarantine is therefore intentionally excluded here.
    """

    if health.get("isolation") != "gpu":
        return frozenset()
    unavailable = unavailable_gpu_ids(
        health, inventory, now=now, stale_seconds=stale_seconds
    )
    capacities = {node.name: set(node.gpu_ids) for node in inventory}
    return frozenset(
        node_name
        for node_name, blocked in unavailable.items()
        if node_name in capacities
        and 0 < len(set(blocked)) < len(capacities[node_name])
    )


def _sample_is_stale(value: object, now: datetime, stale_seconds: float) -> bool:
    if not isinstance(value, str):
        return True
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return True
    if timestamp.tzinfo is None:
        return True
    return (now - timestamp.astimezone(UTC)).total_seconds() > stale_seconds


__all__ = [
    "DEFAULT_BAD_SAMPLES_TO_QUARANTINE",
    "DEFAULT_SAMPLE_STALE_SECONDS",
    "GPU_ISOLATION_MODES",
    "HEALTH_MODES",
    "HealthError",
    "bind_health_incarnation",
    "empty_health_state",
    "ensure_health_state",
    "ingest_health_sample",
    "nodes_requiring_exact_gpu_binding",
    "reprobe_quarantine",
    "set_quarantine",
    "unavailable_gpu_ids",
]
