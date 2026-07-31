from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scruffy.client import observe, status, submit_job
from scruffy.models import ResourceRequest
from scruffy.storage import (
    append_event,
    job_directory,
    open_journal,
    queue_id,
    read_events,
    write_state,
)


class ObserveTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"

    def test_readers_have_independent_cursors_and_output_expansion(self) -> None:
        job_id = "job-observe"
        output = b"prefix|expanded output|suffix"
        prefix = b"prefix|"
        payload = b"expanded output"
        directory = job_directory(self.root, job_id)
        (directory / "stdout.log").write_bytes(output)
        identity = queue_id(self.root)
        events = [
            {"v": 1, "queue_id": identity, "seq": 1, "kind": "job.queued", "job_id": job_id},
            {
                "v": 1,
                "queue_id": identity,
                "seq": 2,
                "kind": "job.output",
                "job_id": job_id,
                "data": {
                    "job_id": job_id,
                    "stream": "stdout",
                    "log": f"jobs/{job_id}/stdout.log",
                    "offset": len(prefix),
                    "length": len(payload),
                },
            },
            {"v": 1, "queue_id": identity, "seq": 3, "kind": "job.succeeded", "job_id": job_id},
        ]
        state = {
            "v": 1,
            "queue_id": identity,
            "last_seq": 3,
            "allocation": {"id": "allocation-1", "state": "running"},
            "nodes": {},
            "jobs": {job_id: {"id": job_id, "state": "succeeded"}},
            "draining": False,
        }
        write_state(self.root, state)
        with open_journal(self.root) as journal:
            for event in events:
                append_event(journal, event, sync=True)

        reader_a = observe(
            self.root, after=0, include_output=False, limit=2
        )
        reader_b = observe(
            self.root, after=0, include_output=True, limit=10
        )

        self.assertEqual([1, 2], [event["seq"] for event in reader_a["events"]])
        self.assertNotIn("text", reader_a["events"][1]["data"])
        self.assertTrue(reader_a["more"])
        self.assertTrue(reader_a["next_cursor"].startswith(f"{identity}:2:"))
        self.assertEqual([1, 2, 3], [event["seq"] for event in reader_b["events"]])
        expanded = next(
            event for event in reader_b["events"] if event["kind"] == "job.output"
        )
        self.assertEqual(payload.decode(), expanded["data"]["text"])
        self.assertTrue(reader_b["next_cursor"].startswith(f"{identity}:3:"))
        self.assertEqual(state, reader_a["snapshot"])

        resumed_a = observe(
            self.root,
            after=reader_a["next_cursor"],
            include_output=True,
        )
        self.assertEqual([3], [event["seq"] for event in resumed_a["events"]])
        replayed_b = observe(self.root, after=0, include_output=True)
        self.assertEqual(reader_b["events"], replayed_b["events"])
        self.assertEqual([], observe(self.root, after=None)["events"])
        raw_output = next(
            event for event in read_events(self.root) if event["kind"] == "job.output"
        )
        self.assertNotIn("text", raw_output["data"])

    def test_submit_and_status_agree_before_controller_admission(self) -> None:
        response = submit_job(
            self.root,
            argv=["true"],
            name="waiting",
            cwd=Path.cwd(),
            environment={},
            request=ResourceRequest(1, 1, 1, 1),
            request_id="agent-a/waiting",
        )

        self.assertEqual("submitted", response["state"])
        self.assertEqual("submitted", status(self.root, response["job_id"])["state"])

    def test_observe_closes_append_between_size_check_and_first_read(self) -> None:
        event = {"seq": 1, "kind": "job.queued"}
        with (
            mock.patch("scruffy.client.time.monotonic", return_value=0),
            mock.patch("scruffy.client.time.sleep"),
            mock.patch("scruffy.client.journal_size", side_effect=[0, 1]),
            mock.patch(
                "scruffy.client.read_event_page",
                side_effect=[([], 0, False), ([event], 10, False)],
            ) as read_page,
        ):
            response = observe(self.root, after=0, wait_seconds=1)

        self.assertEqual([event], response["events"])
        self.assertEqual(2, read_page.call_count)


if __name__ == "__main__":
    unittest.main()
