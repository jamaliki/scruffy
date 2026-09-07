from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scruffy.client import (
    publish_event,
    reconcile_event_ack,
)
from scruffy.controller import (
    _initialize_controller,
    _ingest_reports,
    _recover_lost_workflow_jobs,
    _refresh_dependencies,
)
from scruffy.lifecycle import (
    CHECKPOINT_ACK_TIMEOUT_REASON,
    _finish_job,
)
from scruffy.models import Assignment, NodeInventory, NodeReservation, ResourceRequest
from scruffy.state import commit_snapshot
from scruffy.storage import (
    ReportConflict,
    accept_reports,
    list_reports,
    report_acknowledged,
    report_identity_digest,
    read_events,
    utc_now,
)
from scruffy.submissions import job_from_spec
from scruffy.runtime import RunningProcess


INVENTORY = (NodeInventory("node", (0,), 4, 4),)
REQUEST = ResourceRequest(1, 1, 1, 1)
RECOVERY = {
    "max_attempts": 2,
    "retry_on": [CHECKPOINT_ACK_TIMEOUT_REASON],
    "evacuation": {"signal": "USR1", "grace_seconds": 60},
}


def _controller(root: Path):
    return _initialize_controller(
        root=root,
        inventory=INVENTORY,
        launcher="local",
        allocation_id="allocation",
        slurm_job_id=None,
        poll_interval=0.1,
        cancel_grace=0,
        gpu_health_mode="off",
    )


def _job(
    root: Path,
    job_id: str,
    task_id: str,
    *,
    state: str = "succeeded",
    queue_order: int = 1,
    recovery: dict[str, object] | None = None,
) -> dict[str, object]:
    spec: dict[str, object] = {
        "v": 1,
        "job_id": job_id,
        "request_id": f"request-{job_id}",
        "name": task_id,
        "submitted_at": utc_now(),
        "argv": ["python", "-c", "print('run')", "--resume", "auto"],
        "cwd": str(root),
        "env": {"CHECKPOINT_DIR": str(root / "checkpoints")},
        "resources": REQUEST.to_dict(),
        "workflow_id": "checkpoint-flow",
        "task_id": task_id,
        "needs": [],
        "wait_for": [],
    }
    if recovery is not None:
        spec["recovery"] = recovery
    job = job_from_spec(spec, queue_order)
    job["state"] = state
    job["queue_order"] = queue_order
    if state in {"running", "starting"}:
        job["assignment"] = Assignment(
            job_id,
            REQUEST,
            (NodeReservation("node", (0,), 1, 1),),
        ).to_dict()
        job["launch_token"] = "launch-token"
    return job


def _publication(root: Path, artifact_id: str = "checkpoint/step000000010.pt") -> dict[str, object]:
    return {
        "v": 1,
        "artifact_id": artifact_id,
        "path": str(root / "checkpoints" / "step000000010.pt"),
        "size_bytes": 1,
        "sha256": "a" * 64,
        "manifest_path": str(root / "checkpoints" / "step000000010.ready.json"),
    }


class CheckpointAcknowledgementTests(unittest.TestCase):
    def test_delayed_acceptance_reconciles_without_republishing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            data = {"step": 10, "loss": 1.0}
            with (
                mock.patch(
                    "scruffy.client.wait_for_event_ack",
                    side_effect=TimeoutError("receipt delayed"),
                ),
                mock.patch(
                    "scruffy.client.reconcile_event_ack",
                    side_effect=lambda *args, **kwargs: {
                        "state": "accepted",
                        "acknowledged": True,
                        "identity_sha256": kwargs["expected_identity_sha256"],
                        "evidence": "receipt",
                    },
                ) as reconcile,
                mock.patch(
                    "scruffy.client.submit_report",
                    wraps=__import__(
                        "scruffy.storage", fromlist=["submit_report"]
                    ).submit_report,
                ) as submit,
            ):
                response = publish_event(
                    root,
                    job_id="producer",
                    event_id="checkpoint-event-10",
                    kind="workload.progress",
                    data=data,
                    wait=True,
                    timeout=0,
                    reconciliation_timeout=1,
                )

            self.assertEqual("accepted", response["state"])
            submit.assert_called_once()
            reconcile.assert_called_once()
            self.assertEqual(
                report_identity_digest(submit.call_args.args[1]),
                reconcile.call_args.kwargs["expected_identity_sha256"],
            )

    def test_receipt_reconstruction_uses_durable_journal_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _controller(root)
            controller.state["jobs"]["producer"] = _job(root, "producer", "train")
            commit_snapshot(controller)
            event_id = "checkpoint-event-10"
            publish_event(
                root,
                job_id="producer",
                event_id=event_id,
                kind="workload.artifact",
                data={
                    "artifact_type": "checkpoint",
                    "publication": _publication(root),
                },
            )
            with mock.patch("scruffy.controller.accept_reports"):
                _ingest_reports(controller)
            controller.journal.close()

            evidence = reconcile_event_ack(
                root,
                job_id="producer",
                event_id=event_id,
                timeout=0,
            )

            self.assertFalse(report_acknowledged(root, "producer", event_id)[0])
            self.assertEqual("accepted", evidence["state"])
            self.assertEqual("journal", evidence["evidence"])
            self.assertTrue(evidence["durable_job"])

    def test_strict_artifact_receipt_precedes_large_telemetry_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _controller(root)
            controller.state["jobs"]["artifact-job"] = _job(
                root, "artifact-job", "train", queue_order=1
            )
            controller.state["jobs"]["telemetry-job"] = _job(
                root, "telemetry-job", "metrics", queue_order=2
            )
            commit_snapshot(controller)
            publish_event(
                root,
                job_id="artifact-job",
                event_id="checkpoint-event-10",
                kind="workload.artifact",
                data={
                    "artifact_type": "checkpoint",
                    "publication": _publication(root),
                },
            )
            for index in range(512):
                publish_event(
                    root,
                    job_id="telemetry-job",
                    event_id=f"telemetry-{index}",
                    kind="workload.progress",
                    data={"step": index},
                )

            ordering: list[str] = []
            from scruffy import controller as controller_module
            from scruffy import storage as storage_module

            real_accept = storage_module.accept_reports
            real_snapshot = controller_module.commit_snapshot

            def accept_with_order(*args, **kwargs):
                ordering.append("receipt")
                return real_accept(*args, **kwargs)

            def snapshot_with_order(*args, **kwargs):
                ordering.append("snapshot")
                return real_snapshot(*args, **kwargs)

            with (
                mock.patch("scruffy.controller.accept_reports", side_effect=accept_with_order),
                mock.patch(
                    "scruffy.controller.commit_snapshot",
                    side_effect=snapshot_with_order,
                ),
            ):
                _ingest_reports(controller, limit=128)

            self.assertEqual("receipt", ordering[0])
            self.assertIn("snapshot", ordering)
            self.assertTrue(report_acknowledged(root, "artifact-job", "checkpoint-event-10")[0])
            metrics = controller.state["report_observability"]
            self.assertTrue(metrics["backlog_saturated"])
            self.assertEqual(128, metrics["batch_size"])
            controller.journal.close()

    def test_restart_replays_artifact_evidence_and_releases_waiter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _controller(root)
            producer = _job(root, "producer", "prepare", queue_order=1)
            consumer = _job(root, "consumer", "infer", queue_order=2, state="blocked")
            consumer["wait_for"] = [
                {
                    "kind": "artifact",
                    "task_id": "prepare",
                    "artifact_id": "checkpoint/step000000010.pt",
                }
            ]
            consumer["blockers"] = [{"kind": "artifact", "task_id": "prepare"}]
            controller.state["jobs"].update(producer=producer, consumer=consumer)
            commit_snapshot(controller)
            publish_event(
                root,
                job_id="producer",
                event_id="checkpoint-event-10",
                kind="workload.artifact",
                data={
                    "artifact_type": "checkpoint",
                    "publication": _publication(root),
                },
            )
            _ingest_reports(controller)
            controller.journal.close()

            restarted = _controller(root)
            try:
                _refresh_dependencies(restarted)
                self.assertEqual("queued", restarted.state["jobs"]["consumer"]["state"])
                self.assertEqual(
                    "producer",
                    restarted.state["jobs"]["consumer"]["condition_satisfactions"][0][
                        "producer_job_id"
                    ],
                )
            finally:
                restarted.journal.close()

    def test_rejection_and_conflicting_identity_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            event_id = "conflicting-event"
            publish_event(
                root,
                job_id="producer",
                event_id=event_id,
                kind="workload.progress",
                data={"step": 1},
            )
            with self.assertRaises(ReportConflict):
                publish_event(
                    root,
                    job_id="producer",
                    event_id=event_id,
                    kind="workload.progress",
                    data={"step": 2},
                )
            source, _ = list_reports(root)[0]
            accept_reports([(source, None)])
            rejected = reconcile_event_ack(
                root,
                job_id="producer",
                event_id=event_id,
                timeout=0,
            )
            self.assertEqual("rejected", rejected["state"])

    def test_checkpoint_timeout_has_local_resume_and_capped_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _controller(root)
            job = _job(
                root,
                "trainer-attempt-1",
                "train",
                state="running",
                recovery=RECOVERY,
            )
            controller.state["jobs"][job["id"]] = job
            commit_snapshot(controller)
            _finish_job(controller, str(job["id"]), RunningProcess(mock.Mock(), None), 76)
            self.assertEqual(CHECKPOINT_ACK_TIMEOUT_REASON, job["reason"])
            self.assertEqual(76, job["exit_code"])

            _recover_lost_workflow_jobs(controller, CHECKPOINT_ACK_TIMEOUT_REASON)
            successor = next(
                item
                for item in controller.state["jobs"].values()
                if item.get("predecessor_job_id") == job["id"]
            )
            self.assertEqual(2, successor["attempt"])
            self.assertEqual(job["argv"], successor["argv"])
            self.assertIn("--resume", successor["argv"])
            self.assertEqual("auto", successor["argv"][-1])

            _finish_job(
                controller,
                str(successor["id"]),
                RunningProcess(mock.Mock(), None),
                76,
            )
            _recover_lost_workflow_jobs(controller, CHECKPOINT_ACK_TIMEOUT_REASON)
            self.assertTrue(successor["retry_exhausted"])
            self.assertEqual(
                0,
                sum(
                    item.get("predecessor_job_id") == successor["id"]
                    for item in controller.state["jobs"].values()
                ),
            )
            controller.journal.close()


if __name__ == "__main__":
    unittest.main()
