"""Identify the physical GPUs a worker step was bound to.

The worker runs this module as a short-lived subprocess with the step's own
environment, so the CUDA Driver API enumerates exactly the devices, in the
order, that the workload will see. It never creates a CUDA context, so it
allocates no device memory. The result is reportable identity only: a missing
or failed probe never fails the job.
"""

from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import uuid as uuid_module
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROBE_TIMEOUT_SECONDS = 30.0
_NAME_BYTES = 256
_PCI_BYTES = 32


def _driver_minor(pci_bus_id: str | None) -> int | None:
    """Return the ``/dev/nvidiaN`` minor the driver assigned to a PCI device."""

    if not pci_bus_id:
        return None
    domain, _, rest = pci_bus_id.lower().partition(":")
    # The driver directory uses a four-digit domain; CUDA may report eight.
    candidates = {pci_bus_id.lower(), f"{domain[-4:]}:{rest}"}
    for candidate in candidates:
        information = Path("/proc/driver/nvidia/gpus") / candidate / "information"
        try:
            text = information.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() == "device minor":
                try:
                    return int(value.strip())
                except ValueError:
                    return None
    return None


def probe_bound_devices() -> dict[str, Any]:
    """Return the UUID, name, PCI bus ID, and minor of every visible device."""

    try:
        library = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        return {"devices": [], "error": f"load_driver: {exc}"}

    def call(name: str, *arguments: object) -> int:
        function = getattr(library, name)
        function.restype = ctypes.c_int
        return int(function(*arguments))

    code = call("cuInit", ctypes.c_uint(0))
    if code != 0:
        return {"devices": [], "error": f"cuInit: CUDA error {code}"}
    count = ctypes.c_int()
    code = call("cuDeviceGetCount", ctypes.byref(count))
    if code != 0:
        return {"devices": [], "error": f"cuDeviceGetCount: CUDA error {code}"}
    uuid_function = (
        "cuDeviceGetUuid_v2" if hasattr(library, "cuDeviceGetUuid_v2") else "cuDeviceGetUuid"
    )
    devices = []
    for ordinal in range(count.value):
        device = ctypes.c_int()
        code = call("cuDeviceGet", ctypes.byref(device), ctypes.c_int(ordinal))
        if code != 0:
            return {"devices": devices, "error": f"cuDeviceGet: CUDA error {code}"}
        raw_uuid = (ctypes.c_ubyte * 16)()
        code = call(uuid_function, ctypes.byref(raw_uuid), device)
        if code != 0:
            return {"devices": devices, "error": f"{uuid_function}: CUDA error {code}"}
        name = ctypes.create_string_buffer(_NAME_BYTES)
        pci_bus_id = ctypes.create_string_buffer(_PCI_BYTES)
        name_ok = call("cuDeviceGetName", name, ctypes.c_int(_NAME_BYTES), device) == 0
        pci_ok = (
            call("cuDeviceGetPCIBusId", pci_bus_id, ctypes.c_int(_PCI_BYTES), device) == 0
        )
        bus_id = pci_bus_id.value.decode(errors="replace") if pci_ok else None
        devices.append(
            {
                "cuda_index": ordinal,
                "uuid": f"GPU-{uuid_module.UUID(bytes=bytes(raw_uuid))}",
                "name": name.value.decode(errors="replace") if name_ok else None,
                "pci_bus_id": bus_id,
                "index": _driver_minor(bus_id),
            }
        )
    return {"devices": devices, "error": None}


def run_probe(
    environment: Mapping[str, str], *, timeout: float = PROBE_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Run the probe in a child with the step environment; never raise."""

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "scruffy.gpu_identity"],
            env=dict(environment),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"devices": [], "error": f"probe: {exc}"}
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        detail = (completed.stderr or completed.stdout).strip()[-200:]
        return {
            "devices": [],
            "error": f"probe exited {completed.returncode}: {detail or 'no output'}",
        }
    if not isinstance(result, dict) or not isinstance(result.get("devices"), list):
        return {"devices": [], "error": "probe returned an invalid document"}
    return result


def main() -> int:
    try:
        result = probe_bound_devices()
    except Exception as exc:  # noqa: BLE001 - identity is best-effort
        result = {"devices": [], "error": f"probe: {exc}"}
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
