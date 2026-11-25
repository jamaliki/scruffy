from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scruffy.client import cancel_job, clear_gpu_quarantine, quarantine_gpu, submit_job
from scruffy.controller import _ingest_commands, _ingest_requests, _initialize_controller
from scruffy.gpu_holds import SYSTEM_PROJECT, advance_gpu_holds
from scruffy.gpu_sentinel import NO_IDENTITY_EXIT
from scruffy.gpu_sentinel import main as sentinel_main
from scruffy.health import (
    HealthError,
    quarantined_nodes,
    searching_hold_nodes,
    set_quarantine,
    unavailable_gpu_ids,
)
from scruffy.lifecycle import _finish_job, schedule
from scruffy.models import NodeInventory, ResourceRequest
from scruffy.scheduler import project_gpu_usage
from scruffy.slurm import AllocationIncarnation
from scruffy.state import load_recovered_state
from scruffy.storage import read_events
from scruffy.summary import build_summary, gpu_view

TARGET = "GPU-5d9dfde5-b05d-eb77-1b88-bf46dab160a4"
OTHERS = [f"GPU-0000000{index}-1111-2222-3333-444455556666" for index in range(8)]


def _process() -> mock.MagicMock:
    process = mock.MagicMock()
    process.pid = 4321
    process.poll.return_value = None
    return process


class HoldPolicyTests(unittest.TestCase):
    def test_hold_quarantine_needs_no_monitor_and_withholds_no_slot(self) -> None:
        health = {"v": 1, "mode": "off", "isolation": "gpu", "nodes": {}}
        inventory = (NodeInventory("gpu-6", (0, 1, 2, 3), 8, 64),)
        with self.assertRaisesRegex(HealthError, "unknown GPU"):
            set_quarantine(health, node="gpu-6", uuid=TARGET, quarantined=True, at="t")
        with self.assertRaisesRegex(HealthError, "GPU uuid"):
            set_quarantine(
                health, node="gpu-6", uuid="GPU-1", quarantined=True, at="t",
                enforcement="hold",
            )

        transition = set_quarantine(
            health, node="gpu-6", uuid=TARGET.upper().replace("GPU-", "gpu-"),
            quarantined=True, at="t", enforcement="hold",
        )

        self.assertEqual(TARGET, transition["uuid"])
        device = health["nodes"]["gpu-6"]["devices"][TARGET]
        self.assertEqual(("quarantined", "hold", None), (
            device["status"], device["enforcement"], device["slot"]
        ))
        self.assertEqual({}, unavailable_gpu_ids(health, inventory))
        self.assertEqual(frozenset(), quarantined_nodes(health))

        set_quarantine(health, node="gpu-6", uuid=TARGET, quarantined=False, at="t")
        self.assertEqual({}, health["nodes"])

    def test_sentinel_holds_only_its_target(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch("sys.stderr"):
            self.assertEqual(NO_IDENTITY_EXIT, sentinel_main(["--target", TARGET]))
        for bound, expected in ((TARGET.upper(), ()), (OTHERS[0], (5.0,))):
            event = mock.MagicMock()
            with (
                mock.patch.dict("os.environ", {"SCRUFFY_GPU_UUIDS": bound}, clear=True),
                mock.patch("scruffy.gpu_sentinel.threading.Event", return_value=event),
                mock.patch("scruffy.gpu_sentinel.signal.signal"),
                mock.patch("sys.stdout"),
            ):
                code = sentinel_main(["--target", TARGET, "--park-seconds", "5"])
            self.assertEqual(0, code)
            event.wait.assert_called_once_with(*expected)


class HoldControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"
        # Cancelling a sentinel signals its srun client; these are mocks.
        patcher = mock.patch("scruffy.lifecycle.stop_launcher")
        patcher.start()
        self.addCleanup(patcher.stop)

    def controller(self, gpu_ids: tuple[int, ...]):
        inventory = (NodeInventory("gpu-6", gpu_ids, 16, 64),)
        controller = _initialize_controller(
            root=self.root,
            inventory=inventory,
            launcher="slurm",
            allocation_id="123",
            slurm_job_id="123",
            poll_interval=0.1,
            cancel_grace=0,
            allocation_incarnation=AllocationIncarnation("123", 0, inventory),
            gpu_health_mode="off",
        )
        self.addCleanup(lambda: controller.journal.close())
        controller.state["launches_paused"] = False
        self.jobs = controller.state["jobs"]
        return controller

    def submit(self, controller, name: str, *, priority: int = 0) -> str:
        job_id = submit_job(
            self.root,
            argv=["true"],
            name=name,
            cwd=self.root,
            environment={},
            request=ResourceRequest(1, 1, 1, 1),
            request_id=name,
            priority=priority,
        )["job_id"]
        _ingest_requests(controller)
        return job_id

    def schedule(self, controller) -> None:
        with mock.patch(
            "scruffy.lifecycle.subprocess.Popen", side_effect=lambda *_a, **_k: _process()
        ):
            schedule(controller)

    def identify(self, job_id: str, uuid: str, *, error: str | None = None) -> None:
        job = self.jobs[job_id]
        job["state"] = "running"
        job["gpu_devices"] = [] if error else [{"node": "gpu-6", "uuid": uuid, "index": 0}]
        if error:
            job["gpu_identity_error"] = error

    def finish(self, controller, job_id: str) -> None:
        running = controller.running.pop(job_id)
        if running.final_state is None:
            running.final_state, running.final_reason = "succeeded", "process_exit"
        _finish_job(controller, job_id, running, 0)

    def sentinels(self, *states: str) -> list[str]:
        return sorted(
            job_id
            for job_id, job in self.jobs.items()
            if job.get("system") == "gpu_hold" and (not states or job["state"] in states)
        )

    def hold(self, controller) -> dict[str, object]:
        return controller.state["gpu_health"]["holds"][f"gpu-6/{TARGET}"]

    def quarantine(self, controller) -> None:
        request = quarantine_gpu(self.root, "gpu-6", TARGET, reason="thermal throttling")
        _ingest_commands(controller)
        outcome = next(
            event
            for event in read_events(self.root)
            if event.get("data", {}).get("request_id") == request["request_id"]
        )
        self.assertEqual("resource.gpu_health_changed", outcome["kind"])

    def test_quarantine_with_monitor_off_holds_the_device_and_frees_the_rest(self) -> None:
        controller = self.controller((0, 1, 2, 3))
        user = self.submit(controller, "user-running")
        self.schedule(controller)
        self.identify(user, OTHERS[7])

        self.quarantine(controller)
        device = controller.state["gpu_health"]["nodes"]["gpu-6"]["devices"][TARGET]
        self.assertEqual("hold", device["enforcement"])
        advance_gpu_holds(controller)

        self.assertEqual("searching", self.hold(controller)["state"])
        self.assertEqual(frozenset({"gpu-6"}), searching_hold_nodes(controller.state["gpu_health"]))
        sentinels = self.sentinels("queued")
        self.assertEqual(3, len(sentinels))
        self.assertTrue(all(self.jobs[job]["project_id"] == SYSTEM_PROJECT for job in sentinels))

        # Sentinels are placed before even the highest-priority user work.
        waiting = self.submit(controller, "user-urgent", priority=1000)
        self.schedule(controller)
        self.assertEqual(sentinels, self.sentinels("starting"))
        self.assertEqual("queued", self.jobs[waiting]["state"])
        self.assertEqual({"default": 1}, project_gpu_usage(self.jobs.values()))

        held, *parked = sentinels
        self.identify(held, TARGET.upper())
        for job_id, uuid in zip(parked, OTHERS):
            self.identify(job_id, uuid)
        advance_gpu_holds(controller)

        hold = self.hold(controller)
        self.assertEqual(("holding", held), (hold["state"], hold["job_id"]))
        self.assertEqual(["cancelling", "cancelling"], [self.jobs[j]["state"] for j in parked])
        for job_id in parked:
            self.finish(controller, job_id)
        self.assertEqual(frozenset(), searching_hold_nodes(controller.state["gpu_health"]))
        node = controller.state["nodes"]["gpu-6"]
        self.assertEqual([], node["unavailable_gpu_ids"])
        self.assertEqual({held: {"uuid": TARGET, "state": "holding"}}, node["gpu_holds"])

        self.schedule(controller)
        self.assertEqual("starting", self.jobs[waiting]["state"])
        rows = gpu_view(controller.state)
        stopped = [row for row in rows["gpus"] if row["scheduler_state"] == "stopped"]
        self.assertEqual([(TARGET, held)], [(row["uuid"], row["assigned_job_id"]) for row in stopped])
        self.assertEqual("holding", rows["holds"][0]["state"])
        self.assertEqual("holding", build_summary(controller.state)["gpu_holds"][0]["state"])

        request = cancel_job(self.root, held)
        _ingest_commands(controller)
        ignored = next(
            event
            for event in read_events(self.root)
            if event.get("data", {}).get("request_id") == request["request_id"]
        )
        self.assertEqual(("job.cancel_ignored", "system_job"), (
            ignored["kind"], ignored["data"]["reason"]
        ))
        self.assertEqual("running", self.jobs[held]["state"])
        journaled = [
            event for event in read_events(self.root)
            if event["kind"] == "resource.gpu_health_changed"
        ][-1]
        self.assertEqual(
            "holding", journaled["data"]["gpu_health"]["holds"][f"gpu-6/{TARGET}"]["state"]
        )

        clear_gpu_quarantine(self.root, "gpu-6", TARGET)
        _ingest_commands(controller)
        self.assertEqual({}, controller.state["gpu_health"]["holds"])
        self.assertEqual("cancelling", self.jobs[held]["state"])
        self.assertNotIn("gpu-6", controller.state["gpu_health"]["nodes"])

        controller.journal.close()
        (self.root / "state.json").unlink()
        recovered = load_recovered_state(self.root)
        self.assertEqual({}, recovered["gpu_health"]["holds"])
        self.assertEqual("cancelling", recovered["jobs"][held]["state"])
        self.assertEqual("gpu_hold", recovered["jobs"][held]["system"])

    def test_target_under_a_running_job_waits_without_withholding(self) -> None:
        controller = self.controller((0, 1))
        user = self.submit(controller, "on-target")
        self.schedule(controller)
        self.identify(user, TARGET)

        self.quarantine(controller)
        advance_gpu_holds(controller)

        hold = self.hold(controller)
        self.assertEqual(("waiting", [user]), (hold["state"], hold["blocked_by"]))
        self.assertEqual([], self.sentinels())
        self.assertEqual(frozenset(), searching_hold_nodes(controller.state["gpu_health"]))

        self.finish(controller, user)
        advance_gpu_holds(controller)

        self.assertEqual("searching", self.hold(controller)["state"])
        self.assertEqual(2, len(self.sentinels("queued")))

    def test_unidentified_job_blocks_a_new_search_until_it_ends(self) -> None:
        controller = self.controller((0, 1))
        user = self.submit(controller, "unidentified")
        self.schedule(controller)
        self.jobs[user]["state"] = "running"

        self.quarantine(controller)
        advance_gpu_holds(controller)
        self.schedule(controller)
        (sentinel,) = self.sentinels("starting")
        self.identify(sentinel, OTHERS[0])
        advance_gpu_holds(controller)

        hold = self.hold(controller)
        self.assertEqual(("waiting", [user]), (hold["state"], hold["blocked_by"]))
        self.assertEqual("cancelling", self.jobs[sentinel]["state"])
        self.finish(controller, sentinel)
        advance_gpu_holds(controller)
        self.assertEqual([], self.sentinels("queued"))

        self.finish(controller, user)
        advance_gpu_holds(controller)
        self.assertEqual(2, len(self.sentinels("queued")))

    def test_repeated_identification_failures_stop_the_search(self) -> None:
        controller = self.controller((0,))
        self.quarantine(controller)
        for _ in range(3):
            advance_gpu_holds(controller)
            self.schedule(controller)
            (sentinel,) = self.sentinels("starting")
            self.identify(sentinel, "", error="load_driver: missing")
            advance_gpu_holds(controller)
            self.finish(controller, sentinel)

        advance_gpu_holds(controller)
        hold = self.hold(controller)
        self.assertEqual("failed", hold["state"])
        self.assertIn("could not identify", hold["detail"])
        self.assertEqual([], self.sentinels("queued", "starting", "running"))
        self.assertEqual(frozenset(), searching_hold_nodes(controller.state["gpu_health"]))

    def test_absent_target_fails_once_every_gpu_is_identified(self) -> None:
        controller = self.controller((0, 1))
        self.quarantine(controller)
        advance_gpu_holds(controller)
        self.schedule(controller)
        for job_id, uuid in zip(self.sentinels("starting"), OTHERS):
            self.identify(job_id, uuid)

        advance_gpu_holds(controller)

        hold = self.hold(controller)
        self.assertEqual("failed", hold["state"])
        self.assertIn("not among", hold["detail"])
        self.assertEqual(2, len(self.sentinels("cancelling")))

    def test_two_holds_on_one_node_hand_over_a_parked_sentinel(self) -> None:
        controller = self.controller((0, 1))
        self.quarantine(controller)
        other = f"gpu-6/{OTHERS[5]}"
        quarantine_gpu(self.root, "gpu-6", OTHERS[5])
        _ingest_commands(controller)
        advance_gpu_holds(controller)
        holds = controller.state["gpu_health"]["holds"]
        self.assertEqual("searching", holds[other]["state"])
        self.assertIn("another hold", holds[f"gpu-6/{TARGET}"]["detail"])
        self.schedule(controller)
        on_target, on_other = self.sentinels("starting")
        self.identify(on_target, TARGET)
        self.identify(on_other, OTHERS[5])

        advance_gpu_holds(controller)
        # The other hold keeps its device and releases the sentinel parked on
        # this hold's target; this hold waits for the release, then finds it.
        self.assertEqual(("holding", on_other), (holds[other]["state"], holds[other]["job_id"]))
        self.assertEqual("cancelling", self.jobs[on_target]["state"])
        self.assertNotEqual("failed", self.hold(controller)["state"])
        self.finish(controller, on_target)
        advance_gpu_holds(controller)
        self.schedule(controller)
        (retry,) = self.sentinels("starting")
        self.identify(retry, TARGET)
        advance_gpu_holds(controller)
        self.assertEqual(("holding", retry), (
            self.hold(controller)["state"], self.hold(controller)["job_id"]
        ))
        self.assertEqual({}, project_gpu_usage(self.jobs.values()))


if __name__ == "__main__":
    unittest.main()
