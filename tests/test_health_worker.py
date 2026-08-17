from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from scruffy.health_worker import _minor_number, probe_cuda, query_nvidia_gpus


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


if __name__ == "__main__":
    unittest.main()
