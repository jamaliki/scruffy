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
            }
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

    def test_assets_and_read_only_api_are_served_on_loopback(self) -> None:
        html, headers = self.get("/")
        css, _ = self.get("/assets/app.css")
        app, _ = self.get("/assets/app.js")
        model, _ = self.get("/assets/model.js")
        mascot, mascot_headers = self.get("/assets/scruffy-pixel.png")
        overview, _ = self.get("/api/overview")
        job, _ = self.get("/api/jobs/job-1")

        self.assertIn(b"SCRUFFY", html)
        self.assertIn(b"<h1>Resources</h1>", html)
        self.assertIn(b"<h2>Signals</h2>", html)
        self.assertIn(b"<h2>History</h2>", html)
        self.assertIn(b"<h2>Scheduler</h2>", html)
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
        self.assertIn(b'legendItem("All projects", null, "all")', app)
        self.assertIn(b'view.project === requested ? "all" : requested', app)
        self.assertIn(b'item.setAttribute("aria-pressed"', app)
        self.assertIn(b"export function projectSummaries", model)
        self.assertTrue(mascot.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual("image/png", mascot_headers["Content-Type"])
        self.assertEqual("DENY", headers["X-Frame-Options"])
        self.assertEqual("allocation-1", json.loads(overview)["allocation"]["id"])
        self.assertEqual("job-1", json.loads(job)["job"]["id"])

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
