from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from scruffy.dashboard import create_server, local_reader
from scruffy.storage import queue_id, write_state


def _state(root: Path) -> dict:
    identity = queue_id(root)
    job = {
        "id": "job-1",
        "project_id": "koochak",
        "name": "train-folding",
        "state": "running",
        "submitted_at": "2026-08-06T12:00:00+00:00",
        "started_at": "2026-08-06T12:01:00+00:00",
        "request": {
            "nodes": 1,
            "gpus_per_node": 1,
            "cpus_per_node": 14,
            "memory_gb_per_node": 128,
        },
        "assignment": {
            "job_id": "job-1",
            "reservations": [
                {"node": "gpu-0", "gpu_ids": [0], "cpus": 14, "memory_gb": 128}
            ],
        },
        "workflow_id": "workflow-1",
        "task_id": "train",
        "needs": [],
        "blockers": [],
        "workload": {
            "phase": "training",
            "progress": {"step": 12, "total_steps": 100},
            "last_update_at": "2026-08-06T12:02:00+00:00",
        },
        "stdout": "jobs/job-1/stdout.log",
        "stderr": "jobs/job-1/stderr.log",
        "argv": ["python", "train.py", "--token", "secret"],
        "cwd": "/secret/checkout",
        "environment": {"TOKEN": "secret"},
    }
    return {
        "v": 1,
        "queue_id": identity,
        "last_seq": 0,
        "journal_generation": 0,
        "journal_offset": 0,
        "allocation": {
            "id": "allocation-1",
            "state": "running",
            "heartbeat_at": "2026-08-06T12:02:00+00:00",
        },
        "nodes": {
            "gpu-0": {
                "capacity": {
                    "name": "gpu-0",
                    "gpu_ids": [0, 1],
                    "cpus": 28,
                    "memory_gb": 256,
                },
                "free": {"gpu_ids": [1], "cpus": 14, "memory_gb": 128},
                "assignments": {
                    "job-1": {
                        "node": "gpu-0",
                        "gpu_ids": [0],
                        "cpus": 14,
                        "memory_gb": 128,
                    }
                },
                "unavailable_gpu_ids": [1],
                "gpu_devices": [
                    {
                        "node": "gpu-0",
                        "slot": 0,
                        "uuid": "GPU-aaaa",
                        "nvidia_index": 0,
                        "minor_number": 0,
                        "pci_bus_id": "00000000:17:00.0",
                        "serial": "SERIAL-A",
                        "name": "NVIDIA H100 80GB HBM3",
                        "status": "healthy",
                    },
                    {
                        "node": "gpu-0",
                        "slot": 1,
                        "uuid": "GPU-bbbb",
                        "nvidia_index": 1,
                        "minor_number": 1,
                        "pci_bus_id": "00000000:65:00.0",
                        "serial": "SERIAL-B",
                        "name": "NVIDIA H100 80GB HBM3",
                        "status": "quarantined",
                        "last_reasons": ["thermal_slowdown"],
                    },
                ],
            }
        },
        "gpu_health": {
            "v": 1,
            "mode": "enforce",
            "isolation": "node",
            "monitor": {"status": "running"},
            "nodes": {
                "gpu-0": {
                    "cuda_probe": {"ok": True},
                    "devices": {},
                }
            },
        },
        "jobs": {"job-1": job},
        "archived_jobs": 0,
        "archived_counts": {},
        "archived_project_counts": {},
        "draining": False,
    }


class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"
        write_state(self.root, _state(self.root))
        logs = self.root / "jobs" / "job-1"
        logs.mkdir(parents=True)
        (logs / "stdout.log").write_bytes(b"first line\rprogress 50%\r\nlast line\n")
        (logs / "stderr.log").write_text("warning\n")
        self.server = create_server(str(self.root), port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def get(self, target: str) -> tuple[bytes, object]:
        response = urlopen(f"http://127.0.0.1:{self.server.server_port}{target}")
        return response.read(), response.headers

    def test_compact_reader_never_exposes_process_details(self) -> None:
        reader = local_reader(self.root)

        overview = reader("overview", {"limit": 20, "compact": False})
        explanation = reader("inspect_job", {"job_id": "job-1"})
        encoded = json.dumps({"overview": overview, "explanation": explanation})

        self.assertNotIn("argv", encoded)
        self.assertNotIn("environment", encoded)
        self.assertNotIn("secret", encoded)
        self.assertEqual("training", overview["active"][0]["workload"]["phase"])

    def test_workflow_reader_returns_only_compact_dependency_data(self) -> None:
        reader = local_reader(self.root)

        workflow = reader(
            "inspect_workflow",
            {"workflow_id": "workflow-1", "_project_id": "koochak"},
        )
        encoded = json.dumps(workflow)

        self.assertEqual("workflow-1", workflow["workflow_id"])
        self.assertEqual("train", workflow["tasks"][0]["task_id"])
        self.assertNotIn("argv", encoded)
        self.assertNotIn("environment", encoded)
        self.assertNotIn("secret", encoded)

    def test_assets_and_read_only_api_are_served_on_loopback(self) -> None:
        html, headers = self.get("/")
        css, _ = self.get("/assets/app.css")
        app, _ = self.get("/assets/app.js")
        model, _ = self.get("/assets/model.js")
        mascot, mascot_headers = self.get("/assets/scruffy-pixel.png")
        overview, _ = self.get("/api/overview")
        job, _ = self.get("/api/jobs/job-1")
        workflow, _ = self.get("/api/workflows/workflow-1?project=koochak")
        gpu, _ = self.get("/api/gpus/gpu-0/1")

        self.assertIn(b"SCRUFFY", html)
        self.assertIn(b"<h1>Resources</h1>", html)
        self.assertIn(b"<h2>Signals</h2>", html)
        self.assertIn(b"<h2>History</h2>", html)
        self.assertIn(b"<h2>Scheduler</h2>", html)
        self.assertIn(b"<h2>Workflows</h2>", html)
        self.assertIn(b"<h2>Projects</h2>", html)
        self.assertNotIn(b"GPU topology", html)
        self.assertNotIn(b"Project queues", html)
        self.assertIn(b'id="project-legend"', html)
        self.assertIn(b"scruffy-pixel.png", html)
        self.assertIn(b">Refresh</button>", html)
        self.assertIn(b"has not refreshed for five minutes", html)
        self.assertNotIn(b"Every GPU has one owner", html)
        self.assertNotIn(b"Sync now", html)
        self.assertIn(b"--background:#061315", css)
        self.assertIn(b"--accent:#ff6b42", css)
        self.assertIn(b'--sans:"IBM Plex Sans"', css)
        self.assertIn(b'--serif:"IBM Plex Serif"', css)
        self.assertIn(b".panel-header h1,.panel-header h2", css)
        self.assertIn(b"font-family:var(--sans)", css)
        self.assertIn(b".queue-columns .job-list { max-height:460px", css)
        self.assertIn(b"projectSummaries(snapshot)", app)
        self.assertIn(b"...(snapshot.queued || []), ...(snapshot.submitted || [])", app)
        self.assertIn(b'legendItem("All projects", null, "all")', app)
        self.assertIn(b'view.project === requested ? "all" : requested', app)
        self.assertIn(b'item.setAttribute("aria-pressed"', app)
        self.assertIn(b"export function projectSummaries", model)
        self.assertIn(b"export function workflowLayout", model)
        self.assertIn(b"export function focusedWorkflowTasks", model)
        self.assertIn(b"export function dependencyLinkedTasks", model)
        self.assertIn(b"export function workflowEdges", model)
        self.assertIn(b".workflow-edge.artifact", css)
        self.assertNotIn(b"GPU/n", model)
        self.assertIn(b'"node" : "nodes"', model)
        self.assertIn(b"setInterval(() => loadOverview(false), 5_000)", app)
        self.assertIn(b"Copy report", html)
        self.assertIn(b'id="log-dialog"', html)
        self.assertIn(b"Load earlier output", html)
        self.assertIn(b".log-text", css)
        self.assertIn(b"MAX_VISIBLE_LOG_BYTES", app)
        self.assertIn(b"/output/${log.stream}", app)
        self.assertIn(b'PageDown: text.scrollTop + page', app)
        self.assertIn(b"NVIDIA UUID", app)
        self.assertIn(b"function cudaSummary(device)", app)
        self.assertIn(b"No health warnings", app)
        self.assertNotIn(b'["CUDA probe", JSON.stringify', app)
        self.assertIn(b".gpu-summary", css)
        self.assertIn(b".gpu-stats", css)
        self.assertEqual("GPU-bbbb", json.loads(gpu)["uuid"])
        self.assertIn(b"...(snapshot?.queued || []), ...(snapshot?.submitted || [])", model)
        self.assertTrue(mascot.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual("image/png", mascot_headers["Content-Type"])
        self.assertEqual("DENY", headers["X-Frame-Options"])
        self.assertEqual("allocation-1", json.loads(overview)["allocation"]["id"])
        self.assertEqual("job-1", json.loads(job)["job"]["id"])
        self.assertEqual("workflow-1", json.loads(workflow)["workflow_id"])

    def test_job_output_is_bounded_paginated_and_human_readable(self) -> None:
        complete, _ = self.get("/api/jobs/job-1/output/stdout?offset=0&limit=1024")
        payload = json.loads(complete)

        self.assertEqual("first line\nprogress 50%\nlast line\n", payload["text"])
        self.assertEqual(0, payload["start"])
        self.assertEqual(payload["total_bytes"], payload["end"])
        self.assertFalse(payload["more_before"])
        self.assertFalse(payload["more_after"])
        self.assertTrue(payload["retained"])

        tail, _ = self.get("/api/jobs/job-1/output/stdout?limit=9")
        tail_payload = json.loads(tail)
        self.assertEqual(9, tail_payload["bytes"])
        self.assertTrue(tail_payload["more_before"])
        self.assertEqual(tail_payload["total_bytes"], tail_payload["end"])

        first, _ = self.get("/api/jobs/job-1/output/stdout?offset=0&limit=5")
        first_payload = json.loads(first)
        self.assertEqual("first", first_payload["text"])
        self.assertTrue(first_payload["more_after"])

    def test_missing_and_invalid_job_output_are_contained(self) -> None:
        (self.root / "jobs" / "job-1" / "stderr.log").unlink()
        missing, _ = self.get("/api/jobs/job-1/output/stderr")
        self.assertEqual(False, json.loads(missing)["retained"])

        for target in (
            "/api/jobs/job-1/output/combined",
            "/api/jobs/job-1/output/stdout?offset=-1",
            "/api/jobs/job-1/output/stdout?limit=999999",
        ):
            with self.subTest(target=target), self.assertRaises(HTTPError) as invalid:
                self.get(target)
            self.assertEqual(400, invalid.exception.code)

    def test_unknown_hosts_and_mutations_are_rejected(self) -> None:
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        connection.request("GET", "/", headers={"Host": "malicious.example"})
        self.assertEqual(421, connection.getresponse().status)
        connection.close()

        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        connection.request("POST", "/api/overview")
        self.assertEqual(405, connection.getresponse().status)
        connection.close()

        with self.assertRaises(HTTPError) as missing:
            self.get("/api/jobs/not-a-job")
        self.assertEqual(404, missing.exception.code)


if __name__ == "__main__":
    unittest.main()
