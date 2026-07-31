from __future__ import annotations

import multiprocessing
import tempfile
import traceback
import unittest
from pathlib import Path
from typing import Any

from scruffy.storage import (
    ControllerAlreadyRunning,
    RequestConflict,
    append_event,
    controller_lock,
    find_request,
    journal_tail,
    last_event_sequence,
    list_requests,
    open_journal,
    read_event_page,
    read_events,
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
        elif operation == "read_events":
            result_queue.put(
                (worker_id, "ok", read_events(Path(root_dir), after=value), None)
            )
        else:  # pragma: no cover - test helper misuse
            raise ValueError(f"unknown operation {operation!r}")
    except RequestConflict as exc:
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
        self.assertIn(
            requests[0]["submitted_at"], {spec["submitted_at"] for spec in specs}
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
        stored = {request["job_id"]: request for request in list_requests(self.root)}
        self.assertEqual(expected_ids, set(stored))
        for spec in specs:
            self.assertEqual(spec, stored[spec["job_id"]])


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
        self.assertEqual(3, last_event_sequence(self.root))

    def test_incomplete_trailing_event_is_ignored(self) -> None:
        complete = self.EVENTS[:2]
        with open_journal(self.root) as journal:
            for event in complete:
                append_event(journal, event, sync=True)
        with (self.root / "events.jsonl").open("ab") as journal:
            journal.write(b'{"kind":"job.state","seq":3,"state":"fail')

        self.assertEqual(complete, read_events(self.root))
        self.assertEqual([], read_events(self.root, after=2))
        self.assertEqual(2, last_event_sequence(self.root))

    def test_torn_first_record_does_not_advance_byte_cursor(self) -> None:
        journal = self.root / "events.jsonl"
        journal.parent.mkdir(parents=True)
        journal.write_bytes(b'{"seq":1,"kind":"' + b"x" * 9000 + b'"')

        self.assertEqual((0, 0), journal_tail(self.root))

        with journal.open("ab") as handle:
            handle.write(b"}\n")
        sequence, offset = journal_tail(self.root)
        events, next_offset, more = read_event_page(self.root, offset=0, limit=1)
        self.assertEqual(1, sequence)
        self.assertEqual(journal.stat().st_size, offset)
        self.assertEqual([1], [event["seq"] for event in events])
        self.assertEqual(offset, next_offset)
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
        self.assertEqual((3, final_offset), journal_tail(self.root))


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


if __name__ == "__main__":
    unittest.main()
