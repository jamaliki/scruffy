from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scruffy.health_worker import (
    _busy_gpu_indices,
    _minor_number,
    collect_sample,
    health_worker_release_sha256,
    probe_cuda,
    query_nvidia_gpus,
)
from scruffy.storage import atomic_write_json


class _FakeCudaFunction:
    def __init__(self, callback):
        self.callback = callback
        self.restype = None

    def __call__(self, *arguments):
        return self.callback(*arguments)


class _FakeCuda:
    def __init__(self, *, context_codes=(0, 0), count=2):
        self.calls = []
        self._context_codes = iter(context_codes)
        self.cuInit = _FakeCudaFunction(lambda *_: self._record("cuInit", 0))
        self.cuDeviceGetCount = _FakeCudaFunction(
            lambda pointer: self._device_count(pointer, count)
        )
        self.cuDeviceGet = _FakeCudaFunction(self._device_get)
        self.cuDeviceGetUuid_v2 = _FakeCudaFunction(
            lambda *_: self._record("cuDeviceGetUuid_v2", 0)
        )
        self.cuCtxCreate_v2 = _FakeCudaFunction(
            lambda *_: self._record("cuCtxCreate_v2", next(self._context_codes))
        )
        self.cuCtxDestroy_v2 = _FakeCudaFunction(
            lambda *_: self._record("cuCtxDestroy_v2", 0)
        )

    def _record(self, name, result):
        self.calls.append(name)
        return result

    def _device_count(self, pointer, count):
        self.calls.append("cuDeviceGetCount")
        pointer._obj.value = count
        return 0

    def _device_get(self, pointer, index):
        self.calls.append("cuDeviceGet")
        pointer._obj.value = index.value
        return 0


class HealthWorkerTests(unittest.TestCase):
    def test_minor_number_comes_from_the_linux_device(self) -> None:
        with patch("scruffy.health_worker.os.stat") as stat:
            stat.return_value.st_rdev = os.makedev(195, 7)
            self.assertEqual(7, _minor_number(2))
            stat.assert_called_once_with("/dev/nvidia2")

        with patch("scruffy.health_worker.os.stat", side_effect=FileNotFoundError):
            self.assertIsNone(_minor_number(2))

    def test_nvidia_query_preserves_real_ids_and_thermal_state(self) -> None:
        identity = [
            [
                "2",
                "GPU-aaaa",
                "00000000:17:00.0",
                "SERIAL-A",
                "NVIDIA H100 80GB HBM3",
                "570.00",
                "96.00",
                "61",
                "600.5",
                "700.0",
                "0",
            ]
        ]
        thermal = [["Active", "Not Active"]]
        with (
            patch("scruffy.health_worker._run_query", side_effect=[identity, thermal]),
            patch("scruffy.health_worker._minor_number", return_value=5),
        ):
            devices, error = query_nvidia_gpus()

        self.assertIsNone(error)
        self.assertEqual("GPU-aaaa", devices[0]["uuid"])
        self.assertEqual(2, devices[0]["nvidia_index"])
        self.assertEqual(5, devices[0]["minor_number"])
        self.assertEqual("00000000:17:00.0", devices[0]["pci_bus_id"])
        self.assertTrue(devices[0]["thermal_slowdown"])

    def test_unsupported_thermal_fields_do_not_discard_identity(self) -> None:
        identity = [
            ["0", "GPU-a", "bus", "serial", "H100", "570", "96", "45", "N/A", "700", "0"]
        ]
        with patch(
            "scruffy.health_worker._run_query",
            side_effect=[
                identity,
                RuntimeError("new field unavailable"),
                RuntimeError("old field unavailable"),
            ],
        ):
            devices, error = query_nvidia_gpus()

        self.assertEqual("GPU-a", devices[0]["uuid"])
        self.assertNotIn("thermal_slowdown", devices[0])
        self.assertIn("old field unavailable", error or "")

    def test_slurm_step_gpu_tokens_filter_unmanaged_physical_devices(self) -> None:
        identity = [
            [
                str(index),
                f"GPU-{index}",
                f"bus-{index}",
                f"serial-{index}",
                "H100",
                "570",
                "96",
                "45",
                "600",
                "700",
                "0",
            ]
            for index in range(4)
        ]
        thermal = [["Not Active", "Not Active"] for _ in identity]
        with (
            patch.dict("scruffy.health_worker.os.environ", {"SLURM_STEP_GPUS": "1,3"}, clear=True),
            patch("scruffy.health_worker._run_query", side_effect=[identity, thermal]),
        ):
            devices, _ = query_nvidia_gpus()

        self.assertEqual([1, 3], [device["nvidia_index"] for device in devices])

    def test_missing_cuda_driver_is_a_failed_probe(self) -> None:
        with patch("scruffy.health_worker.ctypes.CDLL", side_effect=OSError("missing")):
            result = probe_cuda()

        self.assertFalse(result["ok"])
        self.assertFalse(result["init_ok"])
        self.assertIn("missing", str(result["error"]))

    def test_assigned_gpu_skips_context_creation_but_probes_idle_gpu(self) -> None:
        library = _FakeCuda()
        with patch("scruffy.health_worker.ctypes.CDLL", return_value=library):
            result = probe_cuda(skip_indices={0})

        self.assertTrue(result["ok"])
        self.assertTrue(result["inconclusive"])
        self.assertEqual(1, library.calls.count("cuCtxCreate_v2"))
        self.assertEqual(
            {"skipped": True, "inconclusive": True},
            {
                key: result["devices"][0][key]
                for key in ("skipped", "inconclusive")
            },
        )

    def test_cuda_out_of_memory_is_inconclusive(self) -> None:
        library = _FakeCuda(context_codes=(2, 0))
        with patch("scruffy.health_worker.ctypes.CDLL", return_value=library):
            result = probe_cuda()

        self.assertTrue(result["ok"])
        self.assertTrue(result["inconclusive"])
        self.assertTrue(result["devices"][0]["inconclusive"])
        self.assertIn("CUDA_ERROR_OUT_OF_MEMORY (2)", str(result["devices"][0]["error"]))
        self.assertEqual(1, library.calls.count("cuCtxDestroy_v2"))

    def test_idle_cuda_context_error_remains_definite(self) -> None:
        library = _FakeCuda(context_codes=(999, 0))
        with patch("scruffy.health_worker.ctypes.CDLL", return_value=library):
            result = probe_cuda()

        self.assertFalse(result["ok"])
        self.assertFalse(result["inconclusive"])
        self.assertFalse(result["devices"][0]["ok"])
        self.assertIn("cuCtxCreate_v2: CUDA_ERROR_UNKNOWN (999)", str(result["devices"][0]["error"]))

    def test_uuid_lookup_failure_is_inconclusive(self) -> None:
        library = _FakeCuda()
        library.cuDeviceGetUuid_v2 = _FakeCudaFunction(
            lambda *_: library._record("cuDeviceGetUuid_v2", 999)
        )
        with patch("scruffy.health_worker.ctypes.CDLL", return_value=library):
            result = probe_cuda()

        self.assertTrue(result["ok"])
        self.assertTrue(result["inconclusive"])
        self.assertTrue(result["devices"][0]["inconclusive"])
        self.assertIsNone(result["devices"][0]["uuid"])

    def test_busy_snapshot_uses_active_reservations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_write_json(
                root / "state.json",
                {
                    "jobs": {
                        "active": {
                            "state": "running",
                            "assignment": {
                                "job_id": "active",
                                "reservations": [
                                    {"node": "gpu-a", "gpu_ids": [1], "cpus": 1, "memory_gb": 1}
                                ],
                            },
                        },
                        "done": {"state": "succeeded", "assignment": None},
                    }
                },
            )

            self.assertEqual({1}, _busy_gpu_indices(root, "gpu-a"))

    def test_collect_sample_passes_busy_snapshot_to_cuda_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_write_json(
                root / "state.json",
                {
                    "jobs": {
                        "active": {
                            "state": "running",
                            "assignment": {
                                "job_id": "active",
                                "reservations": [
                                    {"node": "gpu-a", "gpu_ids": [1], "cpus": 1, "memory_gb": 1}
                                ],
                            },
                        }
                    }
                },
            )
            with (
                patch("scruffy.health_worker.query_nvidia_gpus", return_value=([], None)),
                patch("scruffy.health_worker.probe_cuda", return_value={}) as probe,
            ):
                collect_sample("gpu-a", "a" * 64, root=root)

        probe.assert_called_once_with(skip_indices=frozenset({1}))

    def test_collect_sample_records_worker_and_reservation_provenance(self) -> None:
        with (
            patch("scruffy.health_worker.query_nvidia_gpus", return_value=([], None)),
            patch("scruffy.health_worker.probe_cuda", return_value={}),
        ):
            sample = collect_sample("gpu-a", "a" * 64, root=Path("/missing"))

        self.assertEqual(64, len(sample["health_worker_release_sha256"]))
        self.assertEqual("state.json", Path(sample["reservation_snapshot"]["path"]).name)
        self.assertFalse(sample["reservation_snapshot"]["available"])
        self.assertEqual(health_worker_release_sha256(), sample["health_worker_release_sha256"])


if __name__ == "__main__":
    unittest.main()
