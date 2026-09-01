from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from scruffy.health import (
    HealthError,
    bind_health_incarnation,
    empty_health_state,
    ingest_health_sample,
    nodes_requiring_exact_gpu_binding,
    reprobe_quarantine,
    set_quarantine,
    unavailable_gpu_ids,
)
from scruffy.models import NodeInventory

UTC = timezone.utc


class HealthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inventory = (NodeInventory("gpu-a", (0, 1), 32, 256),)
        self.at = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)

    def sample(
        self,
        *,
        cuda_ok: bool = True,
        thermal: tuple[bool, bool] = (False, False),
        reverse: bool = False,
        at: datetime | None = None,
    ) -> dict[str, object]:
        rows = [
            {
                "uuid": "GPU-aaaa",
                "nvidia_index": 0,
                "minor_number": 0,
                "pci_bus_id": "00000000:17:00.0",
                "serial": "SERIAL-A",
                "name": "NVIDIA H100 80GB HBM3",
                "driver_version": "570.00",
                "vbios_version": "96.00",
                "temperature_c": 45,
                "thermal_slowdown": thermal[0],
                "uncorrectable_ecc_errors": 0,
            },
            {
                "uuid": "GPU-bbbb",
                "nvidia_index": 1,
                "minor_number": 1,
                "pci_bus_id": "00000000:65:00.0",
                "serial": "SERIAL-B",
                "name": "NVIDIA H100 80GB HBM3",
                "driver_version": "570.00",
                "vbios_version": "96.00",
                "temperature_c": 46,
                "thermal_slowdown": thermal[1],
                "uncorrectable_ecc_errors": 0,
            },
        ]
        if reverse:
            rows.reverse()
        return {
            "v": 1,
            "node": "gpu-a",
            "recorded_at": (at or self.at).isoformat(),
            "cuda_probe": {"ok": cuda_ok, "error": None},
            "gpus": rows,
        }

    def test_healthy_sample_records_stable_identity_and_real_ids(self) -> None:
        health = empty_health_state(mode="enforce", isolation="node")

        transitions = ingest_health_sample(health, self.inventory, self.sample())

        device = health["nodes"]["gpu-a"]["devices"]["GPU-aaaa"]
        self.assertEqual("healthy", device["status"])
        self.assertEqual(0, device["slot"])
        self.assertEqual("00000000:17:00.0", device["pci_bus_id"])
        self.assertEqual("SERIAL-A", device["serial"])
        self.assertEqual({"GPU-aaaa", "GPU-bbbb"}, {item["uuid"] for item in transitions})
        self.assertEqual({}, unavailable_gpu_ids(health, self.inventory, now=self.at))

    def test_rows_are_mapped_by_nvidia_index_not_query_order(self) -> None:
        health = empty_health_state(mode="observe", isolation="node")

        ingest_health_sample(health, self.inventory, self.sample(reverse=True))

        devices = health["nodes"]["gpu-a"]["devices"]
        self.assertEqual(0, devices["GPU-aaaa"]["slot"])
        self.assertEqual(1, devices["GPU-bbbb"]["slot"])

    def test_three_thermal_samples_quarantine_only_the_bad_device(self) -> None:
        health = empty_health_state(mode="enforce", isolation="node")

        for offset in range(3):
            ingest_health_sample(
                health,
                self.inventory,
                self.sample(thermal=(False, True), at=self.at + timedelta(seconds=offset)),
            )

        devices = health["nodes"]["gpu-a"]["devices"]
        self.assertEqual("healthy", devices["GPU-aaaa"]["status"])
        self.assertEqual("quarantined", devices["GPU-bbbb"]["status"])
        self.assertEqual(
            {"gpu-a": (0, 1)}, unavailable_gpu_ids(health, self.inventory, now=self.at)
        )

    def test_cuda_failure_quarantines_the_node_and_quarantine_is_sticky(self) -> None:
        health = empty_health_state(mode="enforce", isolation="node")
        for offset in range(3):
            ingest_health_sample(
                health,
                self.inventory,
                self.sample(cuda_ok=False, at=self.at + timedelta(seconds=offset)),
            )
        ingest_health_sample(
            health,
            self.inventory,
            self.sample(at=self.at + timedelta(seconds=3)),
        )

        devices = health["nodes"]["gpu-a"]["devices"]
        self.assertTrue(all(item["status"] == "quarantined" for item in devices.values()))
        self.assertEqual(
            {"gpu-a": (0, 1)}, unavailable_gpu_ids(health, self.inventory, now=self.at)
        )

    def test_cuda_device_count_mismatch_is_a_node_failure(self) -> None:
        health = empty_health_state(mode="enforce", isolation="node")
        for offset in range(3):
            sample = self.sample(at=self.at + timedelta(seconds=offset))
            sample["cuda_probe"] = {
                "ok": True,
                "init_ok": True,
                "device_count": 0,
                "devices": [],
            }
            ingest_health_sample(health, self.inventory, sample)

        devices = health["nodes"]["gpu-a"]["devices"]
        self.assertTrue(all(item["status"] == "quarantined" for item in devices.values()))

    def test_per_device_cuda_context_failure_quarantines_one_gpu(self) -> None:
        health = empty_health_state(mode="enforce", isolation="node")
        for offset in range(3):
            sample = self.sample(at=self.at + timedelta(seconds=offset))
            sample["cuda_probe"] = {
                "ok": False,
                "init_ok": True,
                "devices": [
                    {"nvidia_index": 0, "uuid": "GPU-aaaa", "ok": True, "error": None},
                    {"nvidia_index": 0, "uuid": "GPU-bbbb", "ok": False, "error": "CUDA error 999"},
                ],
            }
            ingest_health_sample(health, self.inventory, sample)

        devices = health["nodes"]["gpu-a"]["devices"]
        self.assertEqual("healthy", devices["GPU-aaaa"]["status"])
        self.assertEqual("quarantined", devices["GPU-bbbb"]["status"])

    def test_inconclusive_cuda_probe_does_not_accumulate_bad_samples(self) -> None:
        health = empty_health_state(mode="enforce", isolation="node")
        for offset in range(3):
            sample = self.sample(at=self.at + timedelta(seconds=offset))
            sample["cuda_probe"] = {
                "ok": True,
                "init_ok": True,
                "inconclusive": True,
                "devices": [
                    {
                        "nvidia_index": 0,
                        "uuid": "GPU-aaaa",
                        "ok": True,
                        "inconclusive": True,
                        "error": "cuCtxCreate_v2: CUDA_ERROR_OUT_OF_MEMORY (2)",
                    }
                ],
            }
            ingest_health_sample(health, self.inventory, sample)

        device = health["nodes"]["gpu-a"]["devices"]["GPU-aaaa"]
        self.assertEqual("unknown", device["status"])
        self.assertEqual(0, device["bad_samples"])
        self.assertEqual(0, device["good_samples"])

    def test_assigned_gpu_software_thermal_slowdown_is_observational(self) -> None:
        health = empty_health_state(mode="enforce", isolation="gpu")
        for offset in range(3):
            sample = self.sample(at=self.at + timedelta(seconds=offset))
            sample["gpus"][0]["software_thermal_slowdown"] = True  # type: ignore[index]
            sample["gpus"][0]["thermal_slowdown"] = True  # type: ignore[index]
            sample["cuda_probe"] = {  # type: ignore[assignment]
                "ok": True,
                "init_ok": True,
                "inconclusive": True,
                "devices": [
                    {
                        "nvidia_index": 0,
                        "uuid": "GPU-aaaa",
                        "ok": True,
                        "skipped": True,
                        "inconclusive": True,
                    }
                ],
            }
            ingest_health_sample(health, self.inventory, sample)

        device = health["nodes"]["gpu-a"]["devices"]["GPU-aaaa"]
        self.assertEqual("unknown", device["status"])
        self.assertEqual(0, device["bad_samples"])
        self.assertEqual(0, device["good_samples"])

    def test_bad_sample_streak_resets_after_a_long_gap(self) -> None:
        health = empty_health_state(mode="enforce", isolation="node")
        for at in (
            self.at,
            self.at + timedelta(minutes=10),
            self.at + timedelta(minutes=20),
        ):
            ingest_health_sample(health, self.inventory, self.sample(cuda_ok=False, at=at))

        device = health["nodes"]["gpu-a"]["devices"]["GPU-aaaa"]
        self.assertEqual("suspect", device["status"])
        self.assertEqual(1, device["bad_samples"])
        self.assertEqual(
            (self.at + timedelta(minutes=20)).isoformat(timespec="milliseconds"),
            device["bad_streak_started_at"],
        )

    def test_future_sample_is_rejected(self) -> None:
        health = empty_health_state(mode="observe", isolation="node")
        with self.assertRaisesRegex(HealthError, "future"):
            ingest_health_sample(
                health,
                self.inventory,
                self.sample(at=datetime.now(UTC) + timedelta(seconds=31)),
            )

    def test_operator_clear_releases_a_healthy_device(self) -> None:
        health = empty_health_state(mode="enforce", isolation="node")
        ingest_health_sample(health, self.inventory, self.sample())
        set_quarantine(
            health,
            node="gpu-a",
            uuid="GPU-aaaa",
            quarantined=True,
            at=self.at.isoformat(),
            reason="Xid 79",
        )

        transition = set_quarantine(
            health,
            node="gpu-a",
            uuid="GPU-aaaa",
            quarantined=False,
            at=(self.at + timedelta(seconds=1)).isoformat(),
        )

        self.assertEqual("healthy", transition["to"])
        self.assertEqual({}, unavailable_gpu_ids(health, self.inventory, now=self.at))

    def test_reprobe_releases_automatic_quarantine_after_clean_sample(self) -> None:
        health = empty_health_state(mode="enforce", isolation="node")
        for offset in range(3):
            ingest_health_sample(
                health,
                self.inventory,
                self.sample(cuda_ok=False, at=self.at + timedelta(seconds=offset)),
            )
        clean_at = self.at + timedelta(seconds=3)
        ingest_health_sample(health, self.inventory, self.sample(at=clean_at))

        transitions = [
            reprobe_quarantine(
                health,
                node="gpu-a",
                uuid=uuid,
                at=(clean_at + timedelta(seconds=1)).isoformat(),
                now=clean_at + timedelta(seconds=1),
            )
            for uuid in ("GPU-aaaa", "GPU-bbbb")
        ]

        self.assertTrue(all(item["from"] == "quarantined" for item in transitions))
        self.assertTrue(all(item["to"] == "healthy" for item in transitions))
        self.assertTrue(
            all(
                item["status"] == "healthy"
                for item in health["nodes"]["gpu-a"]["devices"].values()
            )
        )
        health["nodes"]["gpu-a"]["last_received_at"] = clean_at.isoformat()
        self.assertEqual({}, unavailable_gpu_ids(health, self.inventory, now=clean_at))

    def test_reprobe_rejects_stale_or_operator_owned_evidence(self) -> None:
        health = empty_health_state(mode="enforce", isolation="node")
        ingest_health_sample(health, self.inventory, self.sample())
        set_quarantine(
            health,
            node="gpu-a",
            uuid="GPU-aaaa",
            quarantined=True,
            at=self.at.isoformat(),
        )
        with self.assertRaisesRegex(HealthError, "operator-owned"):
            reprobe_quarantine(
                health,
                node="gpu-a",
                uuid="GPU-aaaa",
                at=(self.at + timedelta(seconds=1)).isoformat(),
                now=self.at + timedelta(seconds=1),
            )

        health = empty_health_state(mode="enforce", isolation="node")
        for offset in range(3):
            ingest_health_sample(
                health,
                self.inventory,
                self.sample(cuda_ok=False, at=self.at + timedelta(seconds=offset)),
            )
        clean_at = self.at + timedelta(seconds=3)
        ingest_health_sample(health, self.inventory, self.sample(at=clean_at))
        with self.assertRaisesRegex(HealthError, "stale"):
            reprobe_quarantine(
                health,
                node="gpu-a",
                uuid="GPU-aaaa",
                at=(clean_at + timedelta(seconds=46)).isoformat(),
                now=clean_at + timedelta(seconds=46),
            )

    def test_enforcement_fails_closed_when_samples_are_missing_or_stale(self) -> None:
        health = empty_health_state(mode="enforce", isolation="node")
        self.assertEqual(
            {"gpu-a": (0, 1)}, unavailable_gpu_ids(health, self.inventory, now=self.at)
        )
        ingest_health_sample(health, self.inventory, self.sample())
        future = self.at + timedelta(seconds=46)
        self.assertEqual({"gpu-a": (0, 1)}, unavailable_gpu_ids(health, self.inventory, now=future))

    def test_observe_mode_never_removes_scheduler_capacity(self) -> None:
        health = empty_health_state(mode="observe", isolation="node")
        for offset in range(3):
            ingest_health_sample(
                health,
                self.inventory,
                self.sample(cuda_ok=False, at=self.at + timedelta(seconds=offset)),
            )
        self.assertEqual({}, unavailable_gpu_ids(health, self.inventory, now=self.at))

    def test_operator_quarantine_withholds_capacity_in_observe_mode(self) -> None:
        health = empty_health_state(mode="observe", isolation="gpu")
        ingest_health_sample(health, self.inventory, self.sample())
        set_quarantine(
            health,
            node="gpu-a",
            uuid="GPU-aaaa",
            quarantined=True,
            at=self.at.isoformat(),
        )

        self.assertEqual(
            {"gpu-a": (0,)}, unavailable_gpu_ids(health, self.inventory, now=self.at)
        )
        self.assertEqual(
            {"gpu-a"}, nodes_requiring_exact_gpu_binding(health, self.inventory, now=self.at)
        )

    def test_healthy_gpu_isolation_nodes_do_not_require_exact_binding(self) -> None:
        health = empty_health_state(mode="observe", isolation="gpu")
        ingest_health_sample(health, self.inventory, self.sample())

        self.assertEqual(
            frozenset(),
            nodes_requiring_exact_gpu_binding(health, self.inventory, now=self.at),
        )

    def test_node_isolation_remains_a_conservative_fallback(self) -> None:
        health = empty_health_state(mode="observe", isolation="node")
        ingest_health_sample(health, self.inventory, self.sample())
        set_quarantine(
            health,
            node="gpu-a",
            uuid="GPU-aaaa",
            quarantined=True,
            at=self.at.isoformat(),
        )

        self.assertEqual(
            {"gpu-a": (0, 1)}, unavailable_gpu_ids(health, self.inventory, now=self.at)
        )
        self.assertEqual(
            frozenset(),
            nodes_requiring_exact_gpu_binding(health, self.inventory, now=self.at),
        )

    def test_gpu_isolation_falls_back_to_node_when_slot_is_not_mappable(self) -> None:
        health = empty_health_state(mode="observe", isolation="gpu")
        ingest_health_sample(health, self.inventory, self.sample())
        set_quarantine(
            health,
            node="gpu-a",
            uuid="GPU-aaaa",
            quarantined=True,
            at=self.at.isoformat(),
        )
        health["nodes"]["gpu-a"]["devices"]["GPU-aaaa"]["slot"] = "unknown"

        self.assertEqual(
            {"gpu-a": (0, 1)}, unavailable_gpu_ids(health, self.inventory, now=self.at)
        )

    def test_rejects_incomplete_or_wrong_node_samples(self) -> None:
        health = empty_health_state(mode="observe", isolation="node")
        incomplete = self.sample()
        incomplete["gpus"] = list(incomplete["gpus"])[0:1]  # type: ignore[arg-type]
        with self.assertRaisesRegex(HealthError, "expected 2"):
            ingest_health_sample(health, self.inventory, incomplete)
        wrong = self.sample()
        wrong["node"] = "gpu-z"
        with self.assertRaisesRegex(HealthError, "unknown node"):
            ingest_health_sample(health, self.inventory, wrong)

    def test_new_allocation_invalidates_freshness_but_keeps_quarantine(self) -> None:
        health = empty_health_state(mode="enforce", isolation="node")
        ingest_health_sample(health, self.inventory, self.sample())
        health["nodes"]["gpu-a"]["last_received_at"] = self.at.isoformat()
        set_quarantine(
            health,
            node="gpu-a",
            uuid="GPU-aaaa",
            quarantined=True,
            at=self.at.isoformat(),
        )

        bind_health_incarnation(health, "a" * 64)
        bind_health_incarnation(health, "b" * 64)

        node = health["nodes"]["gpu-a"]
        self.assertNotIn("last_received_at", node)
        self.assertNotIn("last_sample_at", node)
        self.assertNotIn("cuda_probe", node)
        self.assertEqual("quarantined", node["devices"]["GPU-aaaa"]["status"])


if __name__ == "__main__":
    unittest.main()
