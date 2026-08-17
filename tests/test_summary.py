from __future__ import annotations

import unittest
from datetime import datetime, timezone

from scruffy.state import apply_workload_event
from scruffy.summary import (
    BLOCKED_VIEW_STATES,
    QUEUE_VIEW_STATES,
    RUNNING_VIEW_STATES,
    build_summary,
    compact_job_page,
    explain_job,
    gpu_view,
    inspect_gpu,
    resource_view,
)


class SummaryTests(unittest.TestCase):
    @staticmethod
    def assignment(job_id: str, gpu_ids: list[int]) -> dict[str, object]:
        return {
            "job_id": job_id,
            "request": {
                "nodes": 1,
                "gpus_per_node": len(gpu_ids),
                "cpus_per_node": 1,
                "memory_gb_per_node": 1,
            },
            "reservations": [
                {
                    "node": "gpu-0",
                    "gpu_ids": gpu_ids,
                    "cpus": 1,
                    "memory_gb": 1,
                }
            ],
        }

    def test_operational_views_are_compact_and_state_specific(self) -> None:
        state = {
            "queue_id": "queue-test",
            "last_seq": 7,
            "journal_offset": 91,
            "jobs": {
                state: {
                    "id": state,
                    "project_id": "project-a",
                    "name": f"{state}-job",
                    "state": state,
                    "started_at": "2026-08-03T12:00:00+00:00",
                    "argv": ["secret"],
                }
                for state in (
                    "submitted",
                    "queued",
                    "blocked",
                    "starting",
                    "running",
                    "finishing",
                    "cancelling",
                )
            },
        }

        queue = compact_job_page(
            state,
            states=QUEUE_VIEW_STATES,
            offset=0,
            limit=50,
            project_id=None,
            include_elapsed=False,
        )
        running = compact_job_page(
            state,
            states=RUNNING_VIEW_STATES,
            offset=0,
            limit=50,
            project_id=None,
            include_elapsed=True,
        )
        blocked = compact_job_page(
            state,
            states=BLOCKED_VIEW_STATES,
            offset=0,
            limit=50,
            project_id=None,
            include_elapsed=False,
        )

        self.assertEqual(
            {"submitted", "queued"}, {job["state"] for job in queue["jobs"]}
        )
        self.assertEqual(
            {"starting", "running", "finishing", "cancelling"},
            {job["state"] for job in running["jobs"]},
        )
        self.assertEqual(["blocked"], [job["state"] for job in blocked["jobs"]])
        self.assertNotIn("elapsed_seconds", queue["jobs"][0])
        self.assertIn("elapsed_seconds", running["jobs"][0])
        self.assertNotIn(
            "argv", str({"queue": queue, "running": running, "blocked": blocked})
        )
        self.assertEqual("queue-test:0:7:91", queue["as_of_cursor"])

    def test_resource_view_keeps_capacity_but_hides_assignments(self) -> None:
        state = {
            "queue_id": "queue-test",
            "last_seq": 2,
            "journal_offset": 40,
            "allocation": {"id": "allocation-1", "state": "running"},
            "nodes": {
                "gpu-0": {
                    "capacity": {
                        "gpu_ids": list(range(8)),
                        "cpus": 112,
                        "memory_gb": 1992,
                    },
                    "free": {"gpu_ids": [6, 7], "cpus": 28, "memory_gb": 512},
                    "assignments": {"job-1": {"gpu_ids": list(range(6))}},
                },
                "gpu-13": {
                    "capacity": {"gpu_ids": list(range(8)), "cpus": 112},
                    "free": {"gpu_ids": [], "cpus": 0},
                },
                "gpu-5": {
                    "capacity": {"gpu_ids": list(range(8)), "cpus": 112},
                    "free": {"gpu_ids": list(range(8)), "cpus": 112},
                },
            },
        }

        result = resource_view(state)

        self.assertEqual(10, result["totals"]["gpus_free"])
        self.assertEqual(8, result["nodes"][0]["gpus_total"])
        self.assertEqual(512, result["nodes"][0]["memory_gb_free"])
        self.assertEqual(
            ["gpu-0", "gpu-5", "gpu-13"],
            [node["name"] for node in result["nodes"]],
        )
        self.assertNotIn("assignments", str(result))

    def test_gpu_views_keep_capacity_slots_without_health_samples(self) -> None:
        state = {
            "queue_id": "queue-test",
            "nodes": {
                "gpu-8": {
                    "capacity": {"gpu_ids": [0, 1]},
                    "free": {"gpu_ids": [1]},
                    "assignments": {"job-1": {"gpu_ids": [0]}},
                    "gpu_devices": [
                        {"slot": 0, "uuid": "GPU-known", "status": "healthy"}
                    ],
                }
            },
            "gpu_health": {"mode": "enforce", "isolation": "node"},
        }

        devices = gpu_view(state)["gpus"]
        unsampled = inspect_gpu(state, "gpu-8", 1)

        self.assertEqual([0, 1], [device["slot"] for device in devices])
        self.assertEqual("GPU-known", devices[0]["uuid"])
        self.assertEqual("unknown", unsampled["status"])
        self.assertEqual("free", unsampled["scheduler_state"])

    def test_operational_views_use_scheduler_and_recency_order(self) -> None:
        state = {
            "queue_id": "queue-test",
            "jobs": {
                "heavy-active": {
                    "id": "heavy-active",
                    "project_id": "heavy",
                    "name": "heavy active",
                    "state": "running",
                    "queue_order": 1,
                    "started_at": "2026-08-07T10:00:00+00:00",
                    "assignment": self.assignment("heavy-active", [0, 1, 2, 3]),
                },
                "light-active": {
                    "id": "light-active",
                    "project_id": "light",
                    "name": "light active",
                    "state": "running",
                    "queue_order": 2,
                    "started_at": "2026-08-07T11:00:00+00:00",
                    "assignment": self.assignment("light-active", [4]),
                },
                "heavy-queued": {
                    "id": "heavy-queued",
                    "project_id": "heavy",
                    "name": "heavy queued",
                    "state": "queued",
                    "queue_order": 3,
                },
                "light-queued": {
                    "id": "light-queued",
                    "project_id": "light",
                    "name": "light queued",
                    "state": "queued",
                    "queue_order": 4,
                },
                "blocked-old": {
                    "id": "blocked-old",
                    "name": "blocked old",
                    "state": "blocked",
                    "queue_order": 5,
                },
                "blocked-new": {
                    "id": "blocked-new",
                    "name": "blocked new",
                    "state": "blocked",
                    "queue_order": 6,
                },
            },
        }

        result = build_summary(state)
        running = compact_job_page(
            state,
            states=RUNNING_VIEW_STATES,
            offset=0,
            limit=20,
            project_id=None,
            include_elapsed=True,
        )
        queued = compact_job_page(
            state,
            states=QUEUE_VIEW_STATES,
            offset=0,
            limit=20,
            project_id=None,
            include_elapsed=False,
        )
        blocked = compact_job_page(
            state,
            states=BLOCKED_VIEW_STATES,
            offset=0,
            limit=20,
            project_id=None,
            include_elapsed=False,
        )

        self.assertEqual(
            ["light-active", "heavy-active"],
            [job["id"] for job in result["active"]],
        )
        self.assertEqual(
            ["light-queued", "heavy-queued"],
            [job["id"] for job in result["queued"]],
        )
        self.assertEqual(
            ["blocked-new", "blocked-old"],
            [job["id"] for job in result["blocked"]],
        )
        self.assertEqual(
            ["light-active", "heavy-active"],
            [job["id"] for job in running["jobs"]],
        )
        self.assertEqual(
            ["light-queued", "heavy-queued"],
            [job["id"] for job in queued["jobs"]],
        )
        self.assertEqual(
            ["blocked-new", "blocked-old"],
            [job["id"] for job in blocked["jobs"]],
        )

    def test_artifact_projection_replaces_a_replayed_event_id(self) -> None:
        job: dict[str, object] = {"id": "job-1", "state": "running"}
        event = {
            "event_id": "checkpoint-7",
            "occurred_at": "2026-08-03T12:02:00+00:00",
            "kind": "workload.artifact",
            "data": {"name": "latest.pt"},
        }

        apply_workload_event(job, event, recorded_at=event["occurred_at"])
        apply_workload_event(job, event, recorded_at=event["occurred_at"])

        self.assertEqual(1, len(job["workload"]["latest_artifacts"]))

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
        self.assertEqual(120.0, result["requires_attention"][0]["elapsed_seconds"])
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

    def test_project_summary_keeps_global_capacity_but_filters_jobs(self) -> None:
        state = {
            "queue_id": "queue-test",
            "last_seq": 3,
            "journal_offset": 99,
            "nodes": {"node-a": {"free": {"gpu_ids": [1]}}},
            "jobs": {
                "a": {"id": "a", "project_id": "project-a", "state": "running"},
                "b": {"id": "b", "project_id": "project-b", "state": "failed"},
            },
            "archived_counts": {"succeeded": 7},
            "archived_project_counts": {
                "project-a": {"succeeded": 2},
                "project-b": {"succeeded": 5},
            },
        }

        result = build_summary(state, project_id="project-a")

        self.assertEqual("project-a", result["project_id"])
        self.assertEqual({"running": 1, "succeeded": 2}, result["counts"])
        self.assertEqual(2, result["archived_jobs"])
        self.assertEqual(state["nodes"], result["nodes"])
        self.assertEqual(["a"], [job["id"] for job in result["active"]])


if __name__ == "__main__":
    unittest.main()
