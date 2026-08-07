from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scruffy import ConflictError
from scruffy import publish_event as public_publish_event
from scruffy.client import (
    explain,
    observe,
    parse_cursor,
    publish_event,
    status,
    submit_job,
    summary,
    wait_for_job,
)
from scruffy.models import ResourceRequest
from scruffy.protocol import ProtocolError
from scruffy.storage import (
    append_event,
    job_directory,
    list_reports,
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

    def test_queue_identity_and_cursors_survive_a_root_move(self) -> None:
        original_root = self.root.with_name("original")
        original_identity = queue_id(original_root)
        derived_moved_identity = queue_id(self.root)
        self.assertNotEqual(original_identity, derived_moved_identity)

        state = {
            "v": 1,
            "queue_id": original_identity,
            "last_seq": 0,
            "journal_generation": 0,
            "journal_offset": 0,
            "allocation": None,
            "nodes": {},
            "jobs": {},
            "draining": False,
        }
        write_state(self.root, state)

        self.assertEqual(original_identity, queue_id(self.root))
        response = observe(self.root, after=f"{original_identity}:0:0:0")
        self.assertFalse(response["reset"])
        self.assertEqual(
            f"{original_identity}:0:0:0",
            response["next_cursor"],
        )
        self.assertEqual(original_identity, response["snapshot"]["queue_id"])

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
        with open_journal(self.root) as journal:
            for event in events:
                append_event(journal, event, sync=True)
            state["journal_generation"] = 0
            state["journal_offset"] = journal.tell()
        write_state(self.root, state)

        reader_a = observe(
            self.root, after=0, include_output=False, limit=2
        )
        reader_b = observe(
            self.root, after=0, include_output=True, limit=10
        )

        self.assertEqual([1, 2], [event["seq"] for event in reader_a["events"]])
        self.assertNotIn("text", reader_a["events"][1]["data"])
        self.assertTrue(reader_a["more"])
        self.assertTrue(reader_a["next_cursor"].startswith(f"{identity}:0:2:"))
        self.assertEqual([1, 2, 3], [event["seq"] for event in reader_b["events"]])
        expanded = next(
            event for event in reader_b["events"] if event["kind"] == "job.output"
        )
        self.assertEqual(payload.decode(), expanded["data"]["text"])
        self.assertTrue(reader_b["next_cursor"].startswith(f"{identity}:0:3:"))
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
        self.assertEqual(
            "submitted", status(self.root)["jobs"][response["job_id"]]["state"]
        )
        self.assertEqual(
            "submitted",
            observe(self.root)["snapshot"]["jobs"][response["job_id"]]["state"],
        )

    def test_projects_scope_idempotency_and_queue_views(self) -> None:
        submitted = [
            submit_job(
                self.root,
                argv=["true"],
                name=project_id,
                cwd=Path.cwd(),
                environment={},
                request=ResourceRequest(1, 1, 1, 1),
                request_id="shared-request",
                project_id=project_id,
            )
            for project_id in ("project-a", "project-b")
        ]

        self.assertNotEqual(submitted[0]["job_id"], submitted[1]["job_id"])
        scoped = status(self.root, project_id="project-a")
        self.assertEqual([submitted[0]["job_id"]], list(scoped["jobs"]))
        self.assertEqual(
            "project-a",
            summary(self.root, project_id="project-a")["project_id"],
        )

    def test_wait_tolerates_a_transient_unknown_job_view(self) -> None:
        with (
            mock.patch(
                "scruffy.client.status",
                side_effect=[KeyError("not visible yet"), {"state": "succeeded"}],
            ),
            mock.patch("scruffy.client.time.sleep"),
        ):
            job = wait_for_job(self.root, "job-delayed", timeout=2)

        self.assertEqual("succeeded", job["state"])

    def test_submit_persists_validated_workflow_metadata(self) -> None:
        response = submit_job(
            self.root,
            argv=["true"],
            name="infer",
            cwd=Path.cwd(),
            environment={},
            request=ResourceRequest(1, 1, 1, 1),
            request_id="workflow/infer",
            workflow_id="experiment-7",
            task_id="infer",
            needs=({"task_id": "train", "condition": "succeeded"},),
        )

        submitted = status(self.root, response["job_id"])
        self.assertEqual("experiment-7", submitted["workflow_id"])
        self.assertEqual("infer", submitted["task_id"])
        self.assertEqual(
            [{"task_id": "train", "condition": "succeeded"}],
            submitted["needs"],
        )
        with self.assertRaisesRegex(ValueError, "provide.*together"):
            submit_job(
                self.root,
                argv=["true"],
                name="bad",
                cwd=Path.cwd(),
                environment={},
                request=ResourceRequest(1, 1, 1, 1),
                request_id=None,
                workflow_id="experiment-7",
            )

    def test_summary_and_explain_include_requests_before_admission(self) -> None:
        child = submit_job(
            self.root,
            argv=["true"],
            name="infer",
            cwd=Path.cwd(),
            environment={},
            request=ResourceRequest(1, 1, 1, 1),
            request_id="workflow/infer",
            workflow_id="experiment-7",
            task_id="infer",
            needs=({"task_id": "train", "condition": "succeeded"},),
        )
        parent = submit_job(
            self.root,
            argv=["true"],
            name="train",
            cwd=Path.cwd(),
            environment={},
            request=ResourceRequest(1, 1, 1, 1),
            request_id="workflow/train",
            workflow_id="experiment-7",
            task_id="train",
        )

        allocation = summary(self.root)
        self.assertEqual({"submitted": 2}, allocation["counts"])
        self.assertEqual(
            {child["job_id"], parent["job_id"]},
            {job["id"] for job in allocation["submitted"]},
        )
        explanation = explain(self.root, child["job_id"])
        self.assertEqual(parent["job_id"], explanation["dependencies"][0]["job_id"])
        self.assertEqual("submitted", explanation["dependencies"][0]["state"])

    def test_corrupt_requests_do_not_break_client_views(self) -> None:
        child = submit_job(
            self.root,
            argv=["true"],
            name="child",
            cwd=Path.cwd(),
            environment={},
            request=ResourceRequest(1, 1, 1, 1),
            request_id="corrupt-view/child",
            workflow_id="corrupt-view",
            task_id="child",
            needs=({"task_id": "parent", "condition": "succeeded"},),
        )
        request_root = self.root / "requests"
        broken = request_root / "job-broken"
        broken.mkdir()
        (broken / "spec.json").write_text("[]", encoding="utf-8")
        mismatch = request_root / "job-parent-directory"
        mismatch.mkdir()
        (mismatch / "spec.json").write_text(
            '{"job_id":"job-phantom","name":"parent",'
            '"workflow_id":"corrupt-view","task_id":"parent","needs":[]}',
            encoding="utf-8",
        )

        allocation = summary(self.root)
        broken_view = status(self.root, "job-broken")
        explanation = explain(self.root, child["job_id"])

        self.assertEqual("submitted", broken_view["state"])
        self.assertEqual("invalid_request", broken_view["reason"])
        self.assertNotIn("job-phantom", status(self.root)["jobs"])
        self.assertEqual(3, allocation["counts"]["submitted"])
        self.assertEqual(
            "job-parent-directory", explanation["dependencies"][0]["job_id"]
        )

    def test_publish_event_is_public_durable_and_retryable(self) -> None:
        self.assertIs(public_publish_event, publish_event)
        response = publish_event(
            self.root,
            job_id="job-reporter",
            event_id="koochak/step-7",
            occurred_at="2026-08-03T12:00:00.000+00:00",
            kind="workload.progress",
            source={"name": "koochak", "node": "gpu-3"},
            data={"step": 7, "loss": 0.5},
        )

        self.assertEqual(
            {
                "event_id": "koochak/step-7",
                "job_id": "job-reporter",
                "state": "spooled",
                "deduplicated": False,
            },
            response,
        )
        pending = list_reports(self.root)
        self.assertEqual(1, len(pending))
        self.assertEqual({"name": "koochak", "node": "gpu-3"}, pending[0][1]["source"])

        retried = publish_event(
            self.root,
            job_id="job-reporter",
            event_id="koochak/step-7",
            kind="workload.progress",
            data={"step": 7, "loss": 0.5},
            source={"name": "koochak", "node": "gpu-3"},
        )
        self.assertTrue(retried["deduplicated"])
        self.assertEqual(1, len(list_reports(self.root)))
        with self.assertRaises(ConflictError):
            publish_event(
                self.root,
                job_id="job-reporter",
                event_id="koochak/step-7",
                kind="workload.progress",
                data={"step": 7, "loss": 0.4},
                source={"name": "koochak", "node": "gpu-3"},
            )

    def test_publish_event_defaults_source_and_validates_before_writing(self) -> None:
        response = publish_event(
            self.root,
            job_id="job-reporter",
            kind="workload.notice",
            data={"message": "warming up"},
        )
        self.assertTrue(str(response["event_id"]).startswith("event-"))
        self.assertEqual({}, list_reports(self.root)[0][1]["source"])

        with self.assertRaises(ProtocolError):
            publish_event(
                self.root,
                job_id="job-reporter",
                kind="job.succeeded",
                data={},
            )
        with self.assertRaises(ProtocolError):
            publish_event(
                self.root,
                job_id="job-reporter",
                event_id="",
                kind="workload.notice",
                data={},
            )
        self.assertEqual(1, len(list_reports(self.root)))

    def test_observe_closes_append_between_size_check_and_first_read(self) -> None:
        event = {"seq": 1, "kind": "job.queued"}
        identity = queue_id(self.root)
        with (
            mock.patch("scruffy.client.time.monotonic", return_value=0),
            mock.patch("scruffy.client.time.sleep"),
            mock.patch(
                "scruffy.client._snapshot_cursor",
                side_effect=[
                    (identity, 0, 0, 0),
                    (identity, 0, 1, 10),
                ],
            ),
            mock.patch(
                "scruffy.client.read_event_page",
                side_effect=[([], 0, False), ([event], 10, False)],
            ) as read_page,
        ):
            response = observe(self.root, after=0, wait_seconds=1)

        self.assertEqual([event], response["events"])
        self.assertEqual(2, read_page.call_count)

    def test_observe_never_crosses_the_committed_state_watermark(self) -> None:
        identity = queue_id(self.root)
        event = {"v": 1, "queue_id": identity, "seq": 1, "kind": "job.queued"}
        state = {
            "v": 1,
            "queue_id": identity,
            "last_seq": 0,
            "journal_generation": 0,
            "journal_offset": 0,
            "allocation": None,
            "nodes": {},
            "jobs": {},
            "draining": False,
        }
        with open_journal(self.root) as journal:
            append_event(journal, event, sync=True)
            committed_offset = journal.tell()
        write_state(self.root, state)

        self.assertEqual([], observe(self.root, after=0)["events"])
        state["last_seq"] = 1
        state["journal_offset"] = committed_offset
        write_state(self.root, state)
        self.assertEqual([event], observe(self.root, after=0)["events"])

    def test_generationless_cursor_resets_after_compaction(self) -> None:
        identity = queue_id(self.root)
        write_state(
            self.root,
            {
                "queue_id": identity,
                "journal_generation": 2,
                "last_seq": 40,
                "journal_offset": 123,
                "jobs": {},
                "nodes": {},
                "allocation": None,
                "draining": False,
            },
        )

        self.assertEqual((2, 40, 123, True), parse_cursor(self.root, 0))
        self.assertEqual((2, 40, 123, True), parse_cursor(self.root, "0"))
        self.assertEqual(
            (2, 40, 123, True),
            parse_cursor(self.root, f"{identity}:0:39:99"),
        )

    def test_malformed_cursor_is_rejected_before_generation_reset(self) -> None:
        identity = queue_id(self.root)
        for generation in (0, 2):
            write_state(
                self.root,
                {
                    "queue_id": identity,
                    "journal_generation": generation,
                    "last_seq": 40,
                    "journal_offset": 123,
                    "jobs": {},
                    "nodes": {},
                    "allocation": None,
                    "draining": False,
                },
            )
            for cursor in ("garbage", "wrong:not-a-number:39:99", "-1"):
                with self.subTest(generation=generation, cursor=cursor):
                    with self.assertRaisesRegex(ValueError, "invalid cursor"):
                        parse_cursor(self.root, cursor)


if __name__ == "__main__":
    unittest.main()
