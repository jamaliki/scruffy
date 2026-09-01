from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from scruffy.controller import (
    _check_artifact_condition_conflict,
    _retain_exact_artifact_evidence,
    _stage_job,
)
from scruffy.models import ResourceRequest
from scruffy.storage import StorageError, utc_now
from scruffy.submissions import job_from_spec


def _publication(path: str) -> dict[str, object]:
    return {
        "v": 1,
        "artifact_id": "checkpoint",
        "path": f"/tmp/{path}",
        "size_bytes": 1,
        "sha256": "a" * 64,
        "manifest_path": f"/tmp/{path}.ready.json",
    }


class ArtifactIdentityTests(unittest.TestCase):
    def test_late_waiter_is_rejected_when_exact_evidence_was_not_declared(self) -> None:
        source = job_from_spec(
            {
                "v": 1,
                "job_id": "producer",
                "request_id": "request-producer",
                "name": "prepare",
                "submitted_at": utc_now(),
                "argv": ["true"],
                "cwd": "/tmp",
                "env": {},
                "resources": ResourceRequest(1, 1, 1, 1).to_dict(),
                "workflow_id": "flow",
                "task_id": "prepare",
                "needs": [],
                "wait_for": [],
            },
            1,
        )
        source.update(state="succeeded", finished_at=utc_now())
        consumer = _stage_job(
            {
                "v": 1,
                "job_id": "consumer",
                "request_id": "request-consumer",
                "name": "infer",
                "submitted_at": utc_now(),
                "argv": ["true"],
                "cwd": "/tmp",
                "env": {},
                "resources": ResourceRequest(1, 1, 1, 1).to_dict(),
                "workflow_id": "flow",
                "task_id": "infer",
                "needs": [],
                "wait_for": [
                    {"kind": "artifact", "task_id": "prepare", "artifact_id": "checkpoint"}
                ],
            },
            2,
            {"producer": source},
        )
        self.assertEqual("rejected", consumer["state"])
        self.assertIn("declared before publication", consumer["error"])

    def test_repeated_same_publication_is_deduplicated(self) -> None:
        producer = {"id": "producer", "workflow_id": "flow", "task_id": "prepare"}
        consumer = {
            "id": "consumer",
            "workflow_id": "flow",
            "task_id": "infer",
            "wait_for": [{"kind": "artifact", "task_id": "prepare", "artifact_id": "checkpoint"}],
        }
        controller = mock.Mock(state={"jobs": {"consumer": consumer}}, root=Path("/tmp"))
        event = {
            "event_id": "event-1",
            "occurred_at": "2026-09-01T10:00:00+00:00",
        }
        publication = _publication("checkpoint")
        _retain_exact_artifact_evidence(controller, producer, event, publication)
        _retain_exact_artifact_evidence(
            controller,
            producer,
            {**event, "event_id": "event-2"},
            publication,
        )
        self.assertEqual(1, len(producer["artifact_condition_evidence"]))

    def test_conflicting_publication_for_same_artifact_fails_closed(self) -> None:
        producer = {
            "artifact_condition_evidence": [
                {"publication": _publication("checkpoint")}
            ]
        }
        with self.assertRaises(StorageError):
            _check_artifact_condition_conflict(producer, _publication("other"))
        conflicting = _publication("other")
        conflicting["sha256"] = "b" * 64
        with self.assertRaises(StorageError):
            _check_artifact_condition_conflict(producer, conflicting)


if __name__ == "__main__":
    unittest.main()
