from __future__ import annotations

import hashlib
import multiprocessing
import tempfile
import traceback
import unittest
from pathlib import Path
from typing import Any

from scruffy.storage import (
    ControllerAlreadyRunning,
    ReportConflict,
    RequestConflict,
    accept_request,
    accept_reports,
    activate_journal_generation,
    append_event,
    archive_terminal_job,
    compact_report_receipts,
    controller_lock,
    create_journal_generation,
    find_archived_job,
    find_request,
    job_identity_digest,
    latest_checkpoint,
    list_archived_workflow,
    list_reports,
    list_requests,
    open_journal,
    next_journal_generation,
    read_event_page,
    read_events,
    reject_request,
    report_identity_digest,
    report_was_accepted,
    prune_report_receipts,
    submit_report,
    submit_request,
    tail_bytes,
)


PROCESS_TIMEOUT = 15.0


def _job_spec(
    job_id: str,
    *,
    variant: str = "base",
    submitted_at: str = "2026-07-31T12:00:00.000Z",
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "submitted_at": submitted_at,
        "name": f"job-{variant}",
        "argv": ["python", "-c", f"print({variant!r})"],
        "resources": {
            "nodes": 1,
            "gpus_per_node": 1,
            "cpus_per_node": 2,
            "memory_gb_per_node": 4,
        },
    }


def _workload_report(
    event_id: str,
    *,
    job_id: str = "job-reporter",
    value: int = 1,
    occurred_at: str = "2026-08-03T12:00:00.000+00:00",
) -> dict[str, Any]:
    return {
        "v": 1,
        "event_id": event_id,
        "job_id": job_id,
        "occurred_at": occurred_at,
        "kind": "workload.progress",
        "source": {"name": "test-worker"},
        "data": {"step": value},
    }


def _storage_worker(
    root_dir: str,
    worker_id: int,
    operation: str,
    value: Any,
    ready_queue: Any,
    start_event: Any,
    result_queue: Any,
) -> None:
    """Spawn-safe worker that opens its own view of the queue."""

    ready_queue.put(worker_id)
    if not start_event.wait(PROCESS_TIMEOUT):
        result_queue.put((worker_id, "timeout", None, None))
        return
    try:
        if operation == "submit":
            job_id, duplicate = submit_request(Path(root_dir), value)
            result_queue.put((worker_id, "ok", job_id, duplicate))
        elif operation == "report":
            event_id, duplicate = submit_report(Path(root_dir), value)
            result_queue.put((worker_id, "ok", event_id, duplicate))
        elif operation == "read_events":
            result_queue.put(
                (worker_id, "ok", read_events(Path(root_dir), after=value), None)
            )
        else:  # pragma: no cover - test helper misuse
            raise ValueError(f"unknown operation {operation!r}")
    except RequestConflict as exc:
        result_queue.put((worker_id, "conflict", str(exc), None))
    except ReportConflict as exc:
        result_queue.put((worker_id, "conflict", str(exc), None))
    except Exception:
        result_queue.put((worker_id, "error", traceback.format_exc(), None))


def _lock_holder(root_dir: str, release_event: Any, result_queue: Any) -> None:
    try:
        with controller_lock(Path(root_dir)):
            result_queue.put("acquired")
            if not release_event.wait(PROCESS_TIMEOUT):
                result_queue.put("timeout")
                return
        result_queue.put("released")
    except Exception:
        result_queue.put(traceback.format_exc())


class StorageTestCase(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"

    def run_workers(
        self, operation: str, values: list[Any]
    ) -> list[tuple[int, str, Any, Any]]:
        context = multiprocessing.get_context("spawn")
        ready_queue = context.Queue()
        result_queue = context.Queue()
        start_event = context.Event()
        processes = [
            context.Process(
                target=_storage_worker,
                args=(
                    str(self.root),
                    worker_id,
                    operation,
                    value,
                    ready_queue,
                    start_event,
                    result_queue,
                ),
            )
            for worker_id, value in enumerate(values)
        ]
        try:
            for process in processes:
                process.start()
            ready = {ready_queue.get(timeout=PROCESS_TIMEOUT) for _ in processes}
            self.assertEqual(set(range(len(processes))), ready)
            start_event.set()
            results = [result_queue.get(timeout=PROCESS_TIMEOUT) for _ in processes]
            for process in processes:
                process.join(PROCESS_TIMEOUT)
                self.assertFalse(process.is_alive(), "storage worker hung")
                self.assertEqual(0, process.exitcode)
        finally:
            start_event.set()
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join()
            ready_queue.close()
            result_queue.close()
        return sorted(results)


class SubmitRequestTests(StorageTestCase):
    def test_concurrent_retries_create_one_request(self) -> None:
        specs = [
            _job_spec(
                "job-shared",
                submitted_at=f"2026-07-31T12:00:{index:02d}.000Z",
            )
            for index in range(12)
        ]

        results = self.run_workers("submit", specs)

        self.assertEqual({"ok"}, {status for _, status, _, _ in results})
        self.assertEqual({"job-shared"}, {job_id for _, _, job_id, _ in results})
        self.assertEqual(1, sum(not duplicate for *_, duplicate in results))
        requests = list_requests(self.root)
        self.assertEqual(1, len(requests))
        _, request = requests[0]
        assert request is not None
        self.assertIn(
            request["submitted_at"], {spec["submitted_at"] for spec in specs}
        )

    def test_concurrent_conflicting_retries_have_one_winner(self) -> None:
        specs = [
            _job_spec("job-conflict", variant="alpha" if index < 6 else "beta")
            for index in range(12)
        ]

        results = self.run_workers("submit", specs)

        stored = find_request(self.root, "job-conflict")
        self.assertIsNotNone(stored)
        assert stored is not None
        winner = str(stored["name"]).removeprefix("job-")
        for worker_id, status, _, _ in results:
            submitted = "alpha" if worker_id < 6 else "beta"
            self.assertEqual("ok" if submitted == winner else "conflict", status)
        successful = [result for result in results if result[1] == "ok"]
        self.assertEqual(6, len(successful))
        self.assertEqual(1, sum(not result[3] for result in successful))
        self.assertEqual(1, len(list_requests(self.root)))

    def test_concurrent_distinct_requests_are_all_preserved(self) -> None:
        specs = [
            _job_spec(f"job-{index:02d}", variant=f"variant-{index:02d}")
            for index in range(12)
        ]

        results = self.run_workers("submit", specs)

        self.assertEqual({"ok"}, {status for _, status, _, _ in results})
        self.assertTrue(all(not duplicate for *_, duplicate in results))
        expected_ids = {spec["job_id"] for spec in specs}
        self.assertEqual(expected_ids, {job_id for _, _, job_id, _ in results})
        stored = {
            request_id: request
            for request_id, request in list_requests(self.root)
            if request is not None
        }
        self.assertEqual(expected_ids, set(stored))
        for spec in specs:
            self.assertEqual(spec, stored[spec["job_id"]])

    def test_corrupt_requests_are_returned_as_individual_sentinels(self) -> None:
        valid = _job_spec("job-valid")
        submit_request(self.root, valid)
        corrupt = self.root / "requests" / "job-corrupt"
        corrupt.mkdir()
        (corrupt / "spec.json").write_text("{broken", encoding="utf-8")
        non_object = self.root / "requests" / "job-list"
        non_object.mkdir()
        (non_object / "spec.json").write_text("[]", encoding="utf-8")

        requests = dict(list_requests(self.root))

        self.assertEqual(valid, requests["job-valid"])
        self.assertIsNone(requests["job-corrupt"])
        self.assertIsNone(requests["job-list"])

        self.assertTrue(reject_request(self.root, "job-corrupt"))
        with self.assertRaises(RequestConflict):
            submit_request(self.root, _job_spec("job-corrupt"))

    def test_request_listing_does_not_trust_client_timestamps(self) -> None:
        submit_request(
            self.root,
            _job_spec("job-z", submitted_at="1900-01-01T00:00:00.000Z"),
        )
        submit_request(
            self.root,
            _job_spec("job-a", submitted_at="2100-01-01T00:00:00.000Z"),
        )

        self.assertEqual(
            ["job-a", "job-z"],
            [request_id for request_id, _ in list_requests(self.root)],
        )

    def test_admitted_request_keeps_exact_compact_idempotency(self) -> None:
        spec = _job_spec("job-admitted")
        submit_request(self.root, spec)

        self.assertTrue(accept_request(self.root, spec["job_id"]))

        self.assertEqual([], list_requests(self.root))
        self.assertEqual((spec["job_id"], True), submit_request(self.root, spec))
        with self.assertRaises(RequestConflict):
            submit_request(self.root, _job_spec("job-admitted", variant="changed"))
        lock_files = list((self.root / "requests" / ".locks").glob("*.lock"))
        self.assertLessEqual(len(lock_files), 64)

    def test_terminal_archive_preserves_request_and_workflow_lookup(self) -> None:
        spec = _job_spec("job-archived")
        submit_request(self.root, spec)
        accept_request(self.root, spec["job_id"])
        job = {
            "id": spec["job_id"],
            "name": "train",
            "state": "succeeded",
            "submitted_at": spec["submitted_at"],
            "finished_at": "2026-08-03T12:00:00+00:00",
            "queue_order": 7,
            "request_digest": job_identity_digest(spec),
            "workflow_id": "flow-1",
            "task_id": "train",
            "needs": [],
        }

        archive_terminal_job(self.root, job)

        self.assertEqual("succeeded", find_archived_job(self.root, job["id"])["state"])
        self.assertEqual(
            [job["id"]],
            [item["id"] for item in list_archived_workflow(self.root, "flow-1")],
        )
        self.assertEqual((job["id"], True), submit_request(self.root, spec))


class SubmitReportTests(StorageTestCase):
    def test_concurrent_retries_create_one_durable_report(self) -> None:
        reports = [
            _workload_report(
                "shared-event",
                occurred_at=f"2026-08-03T12:00:{index:02d}.000+00:00",
            )
            for index in range(12)
        ]

        results = self.run_workers("report", reports)

        self.assertEqual({"ok"}, {status for _, status, _, _ in results})
        self.assertEqual({"shared-event"}, {event_id for _, _, event_id, _ in results})
        self.assertEqual(1, sum(not duplicate for *_, duplicate in results))
        pending = list_reports(self.root)
        self.assertEqual(1, len(pending))
        self.assertIsInstance(pending[0][1], dict)

    def test_consumed_retry_uses_hidden_receipt_and_conflicts_stay_visible(self) -> None:
        report = _workload_report("stable/event-id")
        event_id, duplicate = submit_report(self.root, report)
        self.assertEqual("stable/event-id", event_id)
        self.assertFalse(duplicate)
        source, document = list_reports(self.root)[0]
        self.assertEqual(report, document)
        expected_name = hashlib.sha256(event_id.encode()).hexdigest() + ".json"
        self.assertEqual(expected_name, source.name)

        accept_reports(
            ((source, report_identity_digest(report)),),
            generation=0,
        )

        self.assertEqual([], list_reports(self.root))
        retained, digest = report_was_accepted(self.root, source)
        self.assertTrue(retained)
        self.assertEqual(report_identity_digest(report), digest)
        receipts = list((self.root / "reports" / ".accepted" / ".g000000").iterdir())
        self.assertEqual(1, len(receipts))
        self.assertTrue(receipts[0].is_symlink())
        retried = {**report, "occurred_at": "2026-08-03T12:05:00.000+00:00"}
        self.assertEqual((event_id, True), submit_report(self.root, retried))
        self.assertEqual([], list_reports(self.root))
        with self.assertRaises(ReportConflict):
            submit_report(self.root, {**report, "data": {"step": 2}})

    def test_report_receipts_expire_with_journal_generations(self) -> None:
        old = _workload_report("old-event")
        current = _workload_report("current-event")
        submit_report(self.root, old)
        old_source, _ = list_reports(self.root)[0]
        accept_reports(
            ((old_source, report_identity_digest(old)),),
            generation=0,
        )
        submit_report(self.root, current)
        current_source = list_reports(self.root)[0][0]
        accept_reports(
            ((current_source, report_identity_digest(current)),),
            generation=1,
        )

        prune_report_receipts(self.root, keep={1})

        self.assertEqual(("current-event", True), submit_report(self.root, current))
        self.assertEqual(("old-event", False), submit_report(self.root, old))

    def test_legacy_receipts_migrate_valid_and_rejected_identities(self) -> None:
        valid = _workload_report("legacy-valid")
        submit_report(self.root, valid)
        valid_source = list_reports(self.root)[0][0]
        valid_legacy = self.root / "reports" / ".accepted" / valid["job_id"]
        valid_legacy.mkdir(parents=True)
        valid_source.rename(valid_legacy / valid_source.name)

        event_digest = hashlib.sha256(b"bad-event").hexdigest()
        rejected_legacy = self.root / "reports" / ".accepted" / "job-bad"
        rejected_legacy.mkdir(parents=True)
        (rejected_legacy / f"{event_digest}.json").write_text("not json")
        (rejected_legacy / "keep.txt").write_text("unrecognized survivor")

        self.assertEqual(2, compact_report_receipts(self.root))

        pending_source = self.root / "reports" / "job-bad" / f"{event_digest}.json"
        self.assertEqual((True, None), report_was_accepted(self.root, pending_source))
        self.assertEqual(("legacy-valid", True), submit_report(self.root, valid))
        self.assertFalse(valid_legacy.exists())
        self.assertEqual(["keep.txt"], [source.name for source in rejected_legacy.iterdir()])

    def test_one_corrupt_or_oversized_report_does_not_hide_valid_reports(self) -> None:
        submit_report(self.root, _workload_report("valid-event"))
        bad_directory = self.root / "reports" / "job-bad"
        bad_directory.mkdir()
        (bad_directory / "bad.json").write_text("{broken", encoding="utf-8")
        (bad_directory / "huge.json").write_bytes(b"x" * (64 * 1024 + 1))

        reports = list_reports(self.root)

        self.assertEqual(3, len(reports))
        self.assertEqual(2, sum(document is None for _, document in reports))
        valid = [document for _, document in reports if isinstance(document, dict)]
        self.assertEqual(["valid-event"], [document["event_id"] for document in valid])


class ControllerLockTests(StorageTestCase):
    def test_lock_excludes_another_process_and_is_released(self) -> None:
        context = multiprocessing.get_context("spawn")
        release_event = context.Event()
        result_queue = context.Queue()
        process = context.Process(
            target=_lock_holder,
            args=(str(self.root), release_event, result_queue),
        )
        try:
            process.start()
            self.assertEqual("acquired", result_queue.get(timeout=PROCESS_TIMEOUT))
            with self.assertRaises(ControllerAlreadyRunning):
                with controller_lock(self.root):
                    self.fail("a second controller acquired the lock")
            release_event.set()
            self.assertEqual("released", result_queue.get(timeout=PROCESS_TIMEOUT))
            process.join(PROCESS_TIMEOUT)
            self.assertFalse(process.is_alive(), "lock holder hung")
            self.assertEqual(0, process.exitcode)
            with controller_lock(self.root):
                pass
        finally:
            release_event.set()
            if process.is_alive():
                process.terminate()
            process.join()
            result_queue.close()


class EventJournalTests(StorageTestCase):
    EVENTS = [
        {"seq": 1, "kind": "allocation.state", "state": "running"},
        {"seq": 2, "kind": "job.state", "job_id": "job-1", "state": "running"},
        {"seq": 3, "kind": "job.state", "job_id": "job-1", "state": "succeeded"},
    ]

    def test_independent_readers_do_not_consume_events(self) -> None:
        with open_journal(self.root) as journal:
            for event in self.EVENTS:
                append_event(journal, event, sync=True)

        results = self.run_workers("read_events", [0, 0])

        self.assertEqual({"ok"}, {status for _, status, _, _ in results})
        for _, _, events, _ in results:
            self.assertEqual(self.EVENTS, events)
        self.assertEqual(self.EVENTS, read_events(self.root))
        self.assertEqual(self.EVENTS[1:], read_events(self.root, after=1))

    def test_incomplete_trailing_event_is_ignored(self) -> None:
        complete = self.EVENTS[:2]
        with open_journal(self.root) as journal:
            for event in complete:
                append_event(journal, event, sync=True)
        with (self.root / "events.jsonl").open("ab") as journal:
            journal.write(b'{"kind":"job.state","seq":3,"state":"fail')

        self.assertEqual(complete, read_events(self.root))
        self.assertEqual([], read_events(self.root, after=2))

    def test_torn_first_record_does_not_advance_byte_cursor(self) -> None:
        journal = self.root / "events.jsonl"
        journal.parent.mkdir(parents=True)
        journal.write_bytes(b'{"seq":1,"kind":"' + b"x" * 9000 + b'"')

        events, offset, more = read_event_page(self.root, offset=0, limit=1)
        self.assertEqual([], events)
        self.assertEqual(0, offset)
        self.assertFalse(more)

        with journal.open("ab") as handle:
            handle.write(b"}\n")
        events, next_offset, more = read_event_page(self.root, offset=0, limit=1)
        self.assertEqual([1], [event["seq"] for event in events])
        self.assertEqual(journal.stat().st_size, next_offset)
        self.assertFalse(more)

    def test_event_pages_resume_at_byte_offsets(self) -> None:
        with open_journal(self.root) as journal:
            for event in self.EVENTS:
                append_event(journal, event, sync=True)

        first, offset, more = read_event_page(self.root, limit=2)
        second, final_offset, final_more = read_event_page(
            self.root, after=2, offset=offset, limit=2
        )

        self.assertEqual([1, 2], [event["seq"] for event in first])
        self.assertTrue(more)
        self.assertEqual([3], [event["seq"] for event in second])
        self.assertFalse(final_more)
        self.assertEqual((self.root / "events.jsonl").stat().st_size, final_offset)

    def test_mid_record_cursor_restarts_from_a_safe_boundary(self) -> None:
        with open_journal(self.root) as journal:
            for event in self.EVENTS:
                append_event(journal, event, sync=True)
        with (self.root / "events.jsonl").open("rb") as journal:
            middle_of_second = len(journal.readline()) + 5

        events, _, _ = read_event_page(
            self.root, after=1, offset=middle_of_second
        )

        self.assertEqual([2, 3], [event["seq"] for event in events])

    def test_orphan_rotation_never_supersedes_the_active_checkpoint(self) -> None:
        active = {"queue_id": "queue-test", "journal_generation": 1, "jobs": {}}
        orphan = {"queue_id": "queue-test", "journal_generation": 2, "jobs": {}}
        create_journal_generation(self.root, 1, active)
        activate_journal_generation(self.root, 1)
        create_journal_generation(self.root, 2, orphan)

        self.assertEqual((1, active), latest_checkpoint(self.root))
        self.assertEqual(3, next_journal_generation(self.root, 1))


class TailBytesTests(StorageTestCase):
    def test_missing_limits_and_line_endings(self) -> None:
        source = self.root / "worker.log"
        self.assertEqual(b"", tail_bytes(source))
        source.parent.mkdir(parents=True)
        source.write_bytes(b"one\ntwo\nthree\n")
        self.assertEqual(b"", tail_bytes(source, lines=0))
        self.assertEqual(b"", tail_bytes(source, lines=-1))
        self.assertEqual(b"three\n", tail_bytes(source, lines=1))
        self.assertEqual(b"two\nthree\n", tail_bytes(source, lines=2))
        self.assertEqual(b"one\ntwo\nthree\n", tail_bytes(source, lines=20))
        source.write_bytes(b"one\ntwo\nthree")
        self.assertEqual(b"two\nthree", tail_bytes(source, lines=2))

    def test_reads_across_internal_blocks(self) -> None:
        source = self.root / "large.log"
        source.parent.mkdir(parents=True)
        records = [f"{index:04d}:{'x' * 120}\n".encode() for index in range(160)]
        source.write_bytes(b"".join(records))
        self.assertGreater(source.stat().st_size, 2 * 8192)
        self.assertEqual(b"".join(records[-9:]), tail_bytes(source, lines=9))

    def test_newline_free_log_is_bounded(self) -> None:
        source = self.root / "large-progress.log"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"x" * (2 * 1024 * 1024))

        result = tail_bytes(source, lines=200)

        self.assertEqual(1024 * 1024, len(result))
        self.assertEqual(b"x" * (1024 * 1024), result)


if __name__ == "__main__":
    unittest.main()
