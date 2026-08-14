"""Node-local NVIDIA telemetry and CUDA context probe for Scruffy."""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
import socket
import subprocess
import sys
import time
import uuid as uuid_module
from datetime import datetime
from pathlib import Path

from ._compat import UTC
from .storage import atomic_write_json

IDENTITY_FIELDS = (
    "index",
    "minor_number",
    "uuid",
    "pci.bus_id",
    "serial",
    "name",
    "driver_version",
    "vbios_version",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "ecc.errors.uncorrected.volatile.total",
)
IDENTITY_KEYS = (
    "nvidia_index",
    "minor_number",
    "uuid",
    "pci_bus_id",
    "serial",
    "name",
    "driver_version",
    "vbios_version",
    "temperature_c",
    "power_draw_w",
    "power_limit_w",
    "uncorrectable_ecc_errors",
)
THERMAL_FIELD_SETS = (
    (
        "clocks_event_reasons.sw_thermal_slowdown",
        "clocks_event_reasons.hw_thermal_slowdown",
    ),
    (
        "clocks_throttle_reasons.sw_thermal_slowdown",
        "clocks_throttle_reasons.hw_thermal_slowdown",
    ),
)


def _run_query(fields: tuple[str, ...]) -> list[list[str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return [row for row in csv.reader(result.stdout.splitlines()) if row]


def _value(value: str, *, numeric: type[int | float] | None = None) -> object:
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"n/a", "[n/a]", "not supported"}:
        return None
    if numeric is None:
        return cleaned
    try:
        return numeric(cleaned)
    except ValueError:
        return None


def _active(value: str) -> bool | None:
    cleaned = value.strip().lower()
    if cleaned in {"active", "yes", "true"}:
        return True
    if cleaned in {"not active", "no", "false"}:
        return False
    return None


def _managed_devices(devices: list[dict[str, object]]) -> list[dict[str, object]]:
    visible = os.environ.get("SLURM_STEP_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return devices
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    by_index = {str(device.get("nvidia_index")): device for device in devices}
    by_uuid = {str(device.get("uuid")).lower(): device for device in devices}
    selected = []
    for token in tokens:
        device = by_uuid.get(token.lower()) or by_index.get(token)
        if device is None:
            raise RuntimeError(f"Slurm GPU token {token!r} does not match nvidia-smi identity")
        if device not in selected:
            selected.append(device)
    if len(selected) != len(tokens):
        raise RuntimeError("Slurm GPU visibility contains duplicate devices")
    return selected


def query_nvidia_gpus() -> tuple[list[dict[str, object]], str | None]:
    """Return stable IDs and low-rate health metrics for every visible GPU."""

    raw = _run_query(IDENTITY_FIELDS)
    if any(len(row) != len(IDENTITY_KEYS) for row in raw):
        raise RuntimeError("nvidia-smi returned an unexpected identity row")
    numeric: dict[str, type[int | float]] = {
        "nvidia_index": int,
        "minor_number": int,
        "temperature_c": int,
        "power_draw_w": float,
        "power_limit_w": float,
        "uncorrectable_ecc_errors": int,
    }
    devices = [
        {
            key: _value(value, numeric=numeric.get(key))
            for key, value in zip(IDENTITY_KEYS, row, strict=True)
        }
        for row in raw
    ]
    thermal_error = None
    thermal_rows: list[list[str]] | None = None
    for fields in THERMAL_FIELD_SETS:
        try:
            thermal_rows = _run_query(fields)
            thermal_error = None
            break
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            thermal_error = str(exc)
    if thermal_rows is not None:
        if len(thermal_rows) != len(devices) or any(len(row) != 2 for row in thermal_rows):
            thermal_error = "nvidia-smi returned an unexpected thermal row set"
        else:
            for device, row in zip(devices, thermal_rows, strict=True):
                software, hardware = (_active(value) for value in row)
                device["software_thermal_slowdown"] = software
                device["hardware_thermal_slowdown"] = hardware
                device["thermal_slowdown"] = software is True or hardware is True
    return _managed_devices(devices), thermal_error


def _cuda_call(library: ctypes.CDLL, name: str, *arguments: object) -> int:
    function = getattr(library, name)
    function.restype = ctypes.c_int
    return int(function(*arguments))


def probe_cuda() -> dict[str, object]:
    """Initialize CUDA and create one context per visible device."""

    try:
        library = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        return {"ok": False, "init_ok": False, "error": str(exc), "devices": []}
    result = _cuda_call(library, "cuInit", ctypes.c_uint(0))
    if result != 0:
        return {
            "ok": False,
            "init_ok": False,
            "error": f"cuInit returned CUDA error {result}",
            "devices": [],
        }
    count = ctypes.c_int()
    result = _cuda_call(library, "cuDeviceGetCount", ctypes.byref(count))
    if result != 0:
        return {
            "ok": False,
            "init_ok": False,
            "error": f"cuDeviceGetCount returned CUDA error {result}",
            "devices": [],
        }
    devices: list[dict[str, object]] = []
    for index in range(count.value):
        device = ctypes.c_int()
        code = _cuda_call(library, "cuDeviceGet", ctypes.byref(device), ctypes.c_int(index))
        device_uuid = None
        if code == 0:
            uuid_bytes = (ctypes.c_ubyte * 16)()
            uuid_function = (
                "cuDeviceGetUuid_v2"
                if hasattr(library, "cuDeviceGetUuid_v2")
                else "cuDeviceGetUuid"
            )
            code = _cuda_call(library, uuid_function, ctypes.byref(uuid_bytes), device)
            if code == 0:
                device_uuid = f"GPU-{uuid_module.UUID(bytes=bytes(uuid_bytes))}"
        context = ctypes.c_void_p()
        if code == 0:
            code = _cuda_call(
                library,
                "cuCtxCreate_v2",
                ctypes.byref(context),
                ctypes.c_uint(0),
                device,
            )
        if code == 0:
            destroy_code = _cuda_call(library, "cuCtxDestroy_v2", context)
            code = destroy_code if destroy_code != 0 else code
        devices.append(
            {
                "nvidia_index": index,
                "uuid": device_uuid,
                "ok": code == 0,
                "error": None if code == 0 else f"CUDA driver error {code}",
            }
        )
    ok = all(device["ok"] for device in devices)
    return {
        "ok": ok,
        "init_ok": True,
        "device_count": count.value,
        "error": None if ok else "one or more CUDA contexts failed",
        "devices": devices,
    }


def collect_sample(node: str, allocation_incarnation_sha256: str) -> dict[str, object]:
    devices, thermal_error = query_nvidia_gpus()
    return {
        "v": 1,
        "node": node,
        "recorded_at": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "allocation_incarnation_sha256": allocation_incarnation_sha256,
        "cuda_probe": probe_cuda(),
        "thermal_query_error": thermal_error,
        "gpus": devices,
    }


def _node_name() -> str:
    return os.environ.get("SLURMD_NODENAME") or socket.gethostname().split(".", 1)[0]


def run(
    root: Path,
    *,
    interval: float,
    allocation_incarnation_sha256: str,
    once: bool = False,
) -> None:
    if interval <= 0:
        raise ValueError("interval must be positive")
    if len(allocation_incarnation_sha256) != 64:
        raise ValueError("allocation incarnation fingerprint must have 64 characters")
    node = _node_name()
    if not node or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in node
    ):
        raise ValueError(f"unsafe node name {node!r}")
    sample_root = root / "health" / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    target = sample_root / f"{node}.json"
    while True:
        started = time.monotonic()
        atomic_write_json(target, collect_sample(node, allocation_incarnation_sha256))
        if once:
            return
        time.sleep(max(0.0, interval - (time.monotonic() - started)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--allocation-incarnation-sha256", required=True)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        run(
            Path(arguments.root).expanduser().resolve(),
            interval=arguments.interval,
            allocation_incarnation_sha256=arguments.allocation_incarnation_sha256,
            once=arguments.once,
        )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"scruffy health monitor: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
