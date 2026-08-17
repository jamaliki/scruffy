from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from scruffy.client import clear_gpu_quarantine
from scruffy.controller import (
    _ingest_commands,
    _ingest_gpu_health,
    _initialize_controller,
    _maintain_health_monitor,
)
from scruffy.models import NodeInventory
from scruffy.slurm import AllocationIncarnation, SlurmStep
from scruffy.state import load_recovered_state
from scruffy.storage import atomic_write_json, list_commands, read_events

UTC = timezone.utc


class GpuHealthControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"
        self.inventory = (NodeInventory("gpu-0", (0,), 4, 16),)
        self.at = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)

    def sample(self, offset: int, *, failed: bool) -> dict[str, object]:
        return {
            "v": 1,
            "node": "gpu-0",
            "recorded_at": (self.at + timedelta(seconds=offset)).isoformat(),
            "cuda_probe": {
                "ok": not failed,
                "init_ok": True,
                "devices": [
                    {
                        "nvidia_index": 0,
                        "ok": not failed,
                        "error": "CUDA error 999" if failed else None,
                    }
                ],
            },
            "gpus": [
                {
                    "uuid": "GPU-aaaa",
                    "nvidia_index": 0,
                    "minor_number": 0,
                    "pci_bus_id": "00000000:17:00.0",
                    "serial": "SERIAL-A",
                    "name": "NVIDIA H100 80GB HBM3",
                    "driver_version": "570.00",
                    "vbios_version": "96.00",
                    "thermal_slowdown": False,
                    "uncorrectable_ecc_errors": 0,
                }
            ],
        }

    def local_controller(self):
        controller = _initialize_controller(
            root=self.root,
            inventory=self.inventory,
            launcher="local",
            allocation_id="local-test",
            slurm_job_id=None,
            poll_interval=0.1,
            cancel_grace=0,
            gpu_health_mode="enforce",
            gpu_isolation="node",
        )
        self.addCleanup(lambda: controller.journal.close())
        return controller

    def write_sample(self, document: dict[str, object]) -> None:
        atomic_write_json(self.root / "health" / "samples" / "gpu-0.json", document)

    def test_automatic_quarantine_changes_public_capacity_and_replays(self) -> None:
        controller = self.local_controller()
        for offset in range(3):
            self.write_sample(self.sample(offset, failed=True))
            _ingest_gpu_health(controller)

        device = controller.state["nodes"]["gpu-0"]["gpu_devices"][0]
        self.assertEqual("quarantined", device["status"])
        self.assertEqual([0], controller.state["nodes"]["gpu-0"]["unavailable_gpu_ids"])
        self.assertEqual([], controller.state["nodes"]["gpu-0"]["free"]["gpu_ids"])
        self.assertTrue(
            any(event["kind"] == "resource.gpu_health_changed" for event in read_events(self.root))
        )

        controller.journal.close()
        (self.root / "state.json").unlink()
        recovered = load_recovered_state(self.root)
        self.assertEqual(
            "quarantined",
            recovered["gpu_health"]["nodes"]["gpu-0"]["devices"]["GPU-aaaa"]["status"],
        )

    def test_manual_clear_is_correlated_and_acknowledged(self) -> None:
        controller = self.local_controller()
        for offset in range(3):
            self.write_sample(self.sample(offset, failed=True))
            _ingest_gpu_health(controller)
        request = clear_gpu_quarantine(self.root, "gpu-0", "GPU-aaaa")

        _ingest_commands(controller)

        self.assertEqual([], list_commands(self.root))
        device = controller.state["nodes"]["gpu-0"]["gpu_devices"][0]
        self.assertEqual("suspect", device["status"])
        outcome = next(
            event
            for event in read_events(self.root)
            if event.get("data", {}).get("request_id") == request["request_id"]
        )
        self.assertEqual("resource.gpu_health_changed", outcome["kind"])

    def slurm_controller(self):
        incarnation = AllocationIncarnation("123", 0, self.inventory)
        controller = _initialize_controller(
            root=self.root,
            inventory=self.inventory,
            launcher="slurm",
            allocation_id="123",
            slurm_job_id="123",
            allocation_incarnation=incarnation,
            poll_interval=0.1,
            cancel_grace=0,
            gpu_health_mode="observe",
        )
        self.addCleanup(lambda: controller.journal.close())
        return controller

    @mock.patch("scruffy.controller.subprocess.Popen")
    def test_live_named_monitor_is_reattached_without_duplicate_srun(self, popen) -> None:
        controller = self.slurm_controller()
        controller.slurm_steps = (
            SlurmStep("123.7", f"{controller.health_step_name}-gpu-0", "gpu-0"),
        )

        _maintain_health_monitor(controller)

        self.assertEqual({"gpu-0": "123.7"}, controller.health_step_ids)
        popen.assert_not_called()

    @mock.patch("scruffy.controller.subprocess.Popen")
    def test_missing_monitor_launches_one_overlapping_step(self, popen) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        popen.return_value = process
        controller = self.slurm_controller()
        controller.slurm_snapshot_at = 1

        _maintain_health_monitor(controller)

        popen.assert_called_once()
        argv = popen.call_args.args[0]
        self.assertIn("--overlap", argv)
        self.assertIn("--gpus-per-task=1", argv)
        self.assertIn("--allocation-incarnation-sha256", argv)
        self.assertIn(controller.allocation_incarnation.fingerprint_sha256, argv)
        self.assertEqual("starting", controller.state["gpu_health"]["monitor"]["status"])

    @mock.patch("scruffy.controller.subprocess.Popen")
    def test_multinode_monitor_launches_one_step_per_node(self, popen) -> None:
        popen.return_value.poll.return_value = None
        self.inventory = (
            NodeInventory("gpu-0", (0,), 4, 16),
            NodeInventory("gpu-1", (0,), 4, 16),
        )
        controller = self.slurm_controller()
        controller.slurm_snapshot_at = 1

        _maintain_health_monitor(controller)

        self.assertEqual(2, popen.call_count)
        commands = [call.args[0] for call in popen.call_args_list]
        self.assertTrue(all("--nodes=1" in command for command in commands))
        self.assertEqual(
            {"--nodelist=gpu-0", "--nodelist=gpu-1"},
            {
                next(option for option in command if option.startswith("--nodelist="))
                for command in commands
            },
        )

    def test_slurm_controller_rejects_a_sample_from_an_old_incarnation(self) -> None:
        controller = self.slurm_controller()
        sample = self.sample(0, failed=False)
        sample["allocation_incarnation_sha256"] = "f" * 64
        self.write_sample(sample)

        _ingest_gpu_health(controller)

        error = controller.state["gpu_health"]["sample_errors"]["gpu-0"]
        self.assertIn("another allocation incarnation", error)
        self.assertNotIn("gpu-0", controller.state["gpu_health"]["nodes"])

    def test_unchanged_status_samples_do_not_grow_the_journal(self) -> None:
        controller = self.local_controller()
        self.write_sample(self.sample(0, failed=False))
        _ingest_gpu_health(controller)
        after_transition = len(read_events(self.root))

        self.write_sample(self.sample(1, failed=False))
        _ingest_gpu_health(controller)

        self.assertEqual(after_transition, len(read_events(self.root)))


if __name__ == "__main__":
    unittest.main()
