from __future__ import annotations

import unittest
from datetime import datetime, timezone

from scruffy.state import apply_workload_event
from scruffy.summary import build_summary, explain_job


class SummaryTests(unittest.TestCase):
    def test_late_reports_are_journaled_without_regressing_current_progress(self) -> None:
        job: dict[str, object] = {"id": "job-1", "state": "running"}
        newer = {
            "event_id": "event-2",
            "occurred_at": "2026-08-03T12:02:00+00:00",
            "kind": "workload.progress",
            "data": {"phase": "training", "step": 20},
        }
        older = {
            "event_id": "event-1",
            "occurred_at": "2026-08-03T12:01:00+00:00",
            "kind": "workload.progress",
            "data": {"phase": "training", "step": 10},
        }

        apply_workload_event(job, newer, recorded_at="2026-08-03T12:03:00+00:00")
        apply_workload_event(job, older, recorded_at="2026-08-03T12:04:00+00:00")

        workload = job["workload"]
        self.assertEqual(20, workload["progress"]["step"])
        self.assertEqual(newer["occurred_at"], workload["last_update_at"])
        self.assertEqual("2026-08-03T12:04:00+00:00", workload["last_recorded_at"])

    def test_summary_is_bounded_and_surfaces_progress_and_attention(self) -> None:
        state = {
            "queue_id": "queue-test",
            "last_seq": 9,
            "journal_offset": 123,
            "allocation": {"id": "allocation-1", "state": "running"},
            "nodes": {"node-a": {"free": {"gpu_ids": [1]}}},
            "draining": False,
            "jobs": {
                "running": {
                    "id": "running",
                    "name": "train",
                    "state": "running",
                    "workload": {
                        "phase": "training",
                        "last_update_at": "2026-08-03T12:00:00+00:00",
                        "progress": {"completed": 4, "total": 10, "unit": "steps"},
                    },
                },
                "failed": {
                    "id": "failed",
                    "name": "infer",
                    "state": "failed",
                    "reason": "process_exit",
                    "error": "rank 0 exited",
                    "exit_code": 17,
                    "signal": None,
                    "submitted_at": "2026-08-03T11:58:00+00:00",
                    "started_at": "2026-08-03T11:59:00+00:00",
                    "finished_at": "2026-08-03T12:01:00+00:00",
                    "request": {"gpus": 1},
                },
            },
        }

        result = build_summary(
            state,
            now=datetime(2026, 8, 3, 12, 2, tzinfo=timezone.utc),
            limit=1,
        )

        self.assertEqual({"failed": 1, "running": 1}, result["counts"])
        self.assertEqual("training", result["active"][0]["workload"]["phase"])
        self.assertEqual(120.0, result["active"][0]["progress_age_seconds"])
        self.assertEqual("failed", result["requires_attention"][0]["id"])
        self.assertEqual("rank 0 exited", result["requires_attention"][0]["error"])
        self.assertEqual(17, result["requires_attention"][0]["exit_code"])
        self.assertEqual(
            "2026-08-03T11:59:00+00:00",
            result["requires_attention"][0]["started_at"],
        )
        self.assertEqual({"gpus": 1}, result["requires_attention"][0]["request"])
        self.assertEqual("queue-test:0:9:123", result["as_of_cursor"])

    def test_explain_resolves_task_dependencies(self) -> None:
        state = {
            "jobs": {
                "upstream": {
                    "id": "upstream",
                    "state": "succeeded",
                    "workflow_id": "flow",
                    "task_id": "train",
                },
                "downstream": {
                    "id": "downstream",
                    "state": "blocked",
                    "reason": "waiting_for_dependencies",
                    "workflow_id": "flow",
                    "task_id": "infer",
                    "needs": [{"task_id": "train", "condition": "succeeded"}],
                    "blockers": [],
                },
                "duplicate": {
                    "id": "duplicate",
                    "state": "rejected",
                    "workflow_id": "flow",
                    "task_id": "train",
                    "workflow_invalid": True,
                    "reason": "invalid_workflow",
                },
            }
        }

        result = explain_job(state, "downstream")

        self.assertEqual("upstream", result["dependencies"][0]["job_id"])
        self.assertEqual("succeeded", result["dependencies"][0]["state"])


if __name__ == "__main__":
    unittest.main()
