from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from typing import Any

from scruffy.mcp_hub import UpdateBroker, parse_cursor
from scruffy.storage import append_event, open_journal, queue_id, write_state

try:
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
except ModuleNotFoundError:
    ClientSession = httpx = streamable_http_client = None


def _cursor(sequence: int) -> str:
    return f"queue-test:0:{sequence}:{sequence * 100}"


def _read_url(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=0.2) as response:
        return response.read()


def _page(*events: dict[str, Any], reset: bool = False) -> dict[str, Any]:
    sequence = max((int(event["_seq"]) for event in events), default=0)
    return {
        "events": list(events),
        "next_cursor": _cursor(sequence),
        "latest_cursor": _cursor(sequence),
        "more": False,
        "reset": reset,
    }


class FakeRemote:
    def __init__(self) -> None:
        self.responses: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.polls = 0
        self.active_polls = 0
        self.max_active_polls = 0

    async def __call__(self, tool: str, params: dict[str, Any]) -> dict[str, Any]:
        if tool == "overview":
            return {
                "queue_id": "queue-test",
                "project_id": params.get("_project_id"),
                "as_of_cursor": _cursor(0),
                "counts": {},
                "resources": {},
            }
        if tool != "_poll_updates":
            raise AssertionError(tool)
        self.polls += 1
        self.active_polls += 1
        self.max_active_polls = max(self.max_active_polls, self.active_polls)
        try:
            return await self.responses.get()
        finally:
            self.active_polls -= 1


async def _ready(broker: UpdateBroker) -> None:
    for _ in range(100):
        if broker.current_cursor is not None:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("broker did not initialize")


class CursorTests(unittest.TestCase):
    def test_parse_cursor_uses_ascii_numeric_components(self) -> None:
        self.assertEqual(7, parse_cursor("queue-a:2:7:91").sequence)
        with self.assertRaises(ValueError):
            parse_cursor("queue-a:2:٧:91")
        with self.assertRaises(ValueError):
            parse_cursor("broken")


class BrokerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.remote = FakeRemote()
        self.broker = UpdateBroker(self.remote, poll_seconds=1)
        await self.broker.start()
        await _ready(self.broker)

    async def asyncTearDown(self) -> None:
        await self.broker.close()

    async def test_independent_waiters_share_one_upstream_observer(self) -> None:
        first = asyncio.create_task(self.broker.wait(after=_cursor(0), timeout_seconds=2))
        second = asyncio.create_task(self.broker.wait(after=_cursor(0), timeout_seconds=2))
        await self.remote.responses.put(
            _page(
                {
                    "kind": "job.running",
                    "job_id": "job-1",
                    "project_id": "project-a",
                    "name": "train",
                    "_seq": 1,
                    "_workflow_id": "workflow-a",
                }
            )
        )

        left, right = await asyncio.gather(first, second)

        self.assertEqual(left, right)
        self.assertEqual(["job.running"], [event["kind"] for event in left["events"]])
        self.assertNotIn("_seq", left["events"][0])
        self.assertEqual(1, self.remote.max_active_polls)

    async def test_suppressed_pages_advance_before_a_matching_event(self) -> None:
        waiting = asyncio.create_task(
            self.broker.wait(
                after=_cursor(0),
                timeout_seconds=2,
                project_id="project-a",
                workflow_id="workflow-a",
            )
        )
        await self.remote.responses.put(
            _page(
                {
                    "kind": "workload.progress",
                    "job_id": "job-1",
                    "project_id": "project-a",
                    "_seq": 1,
                    "_workflow_id": "workflow-a",
                }
            )
        )
        await self.remote.responses.put(
            {
                **_page(
                    {
                        "kind": "job.succeeded",
                        "job_id": "job-1",
                        "project_id": "project-a",
                        "name": "train",
                        "_seq": 2,
                        "_workflow_id": "workflow-a",
                    }
                ),
                "next_cursor": _cursor(2),
                "latest_cursor": _cursor(2),
            }
        )

        result = await waiting

        self.assertEqual(_cursor(2), result["next_cursor"])
        self.assertEqual(["job.succeeded"], [event["kind"] for event in result["events"]])
        self.assertNotIn("project_id", result["events"][0])

    async def test_exact_kind_filter_can_receive_quiet_events(self) -> None:
        waiting = asyncio.create_task(
            self.broker.wait(
                after=_cursor(0),
                timeout_seconds=2,
                event_kinds=["workload.progress"],
            )
        )
        await self.remote.responses.put(
            _page(
                {
                    "kind": "workload.progress",
                    "job_id": "job-1",
                    "project_id": "project-a",
                    "_seq": 1,
                }
            )
        )

        result = await waiting

        self.assertEqual("workload.progress", result["events"][0]["kind"])

    async def test_cancelling_one_wait_does_not_cancel_the_observer(self) -> None:
        cancelled = asyncio.create_task(self.broker.wait(after=_cursor(0), timeout_seconds=60))
        await asyncio.sleep(0)
        cancelled.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled

        waiting = asyncio.create_task(self.broker.wait(after=_cursor(0), timeout_seconds=2))
        await self.remote.responses.put(
            _page(
                {
                    "kind": "job.succeeded",
                    "job_id": "job-1",
                    "project_id": "project-a",
                    "_seq": 1,
                }
            )
        )

        result = await waiting
        self.assertEqual("job.succeeded", result["events"][0]["kind"])

    async def test_old_or_foreign_cursor_resets_to_project_overview(self) -> None:
        result = await self.broker.wait(
            after="another-queue:0:1:100",
            timeout_seconds=0,
            project_id="project-a",
        )

        self.assertTrue(result["reset"])
        self.assertEqual("queue_replaced", result["reset_reason"])
        self.assertEqual("project-a", result["overview"]["project_id"])
        self.assertEqual(result["next_cursor"], result["overview"]["as_of_cursor"])

    async def test_trimmed_cursor_reports_bounded_buffer_expiry(self) -> None:
        self.broker.max_events = 1
        for sequence in (1, 2):
            await self.remote.responses.put(
                _page(
                    {
                        "kind": "job.running",
                        "job_id": f"job-{sequence}",
                        "_seq": sequence,
                    }
                )
            )
            for _ in range(100):
                if self.broker.current_cursor == _cursor(sequence):
                    break
                await asyncio.sleep(0.01)

        result = await self.broker.wait(after=_cursor(0), timeout_seconds=0)

        self.assertTrue(result["reset"])
        self.assertEqual("hub_buffer_expired", result["reset_reason"])

    async def test_timeout_keeps_the_advanced_private_cursor(self) -> None:
        await self.remote.responses.put(
            _page(
                {
                    "kind": "workload.progress",
                    "job_id": "job-1",
                    "project_id": "project-a",
                    "_seq": 1,
                }
            )
        )
        for _ in range(100):
            if self.broker.current_cursor == _cursor(1):
                break
            await asyncio.sleep(0.01)

        result = await self.broker.wait(after=_cursor(0), timeout_seconds=0)

        self.assertTrue(result["timed_out"])
        self.assertEqual(_cursor(1), result["next_cursor"])

    async def test_cursor_ahead_of_observer_does_not_claim_more_backlog(self) -> None:
        result = await self.broker.wait(after=_cursor(1), timeout_seconds=0)

        self.assertTrue(result["timed_out"])
        self.assertFalse(result["more"])
        self.assertEqual(result["next_cursor"], result["latest_cursor"])


@unittest.skipIf(ClientSession is None, "MCP extra is not installed")
class HttpHubTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "queue"
        identity = queue_id(self.root)
        write_state(
            self.root,
            {
                "v": 1,
                "queue_id": identity,
                "last_seq": 0,
                "journal_generation": 0,
                "journal_offset": 0,
                "allocation": {"id": "allocation-1", "state": "running"},
                "nodes": {},
                "jobs": {},
                "draining": False,
                "archived_jobs": 0,
                "archived_counts": {},
            },
        )
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            self.port = listener.getsockname()[1]
        source_root = Path(__file__).resolve().parents[1] / "src"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(source_root), environment.get("PYTHONPATH")))
        )
        self.log = open(  # noqa: ASYNC230, SIM115 - retained through tearDown
            Path(self.temporary.name) / "hub.log", "w+"
        )
        self.process = subprocess.Popen(  # noqa: ASYNC220 - managed in tearDown
            [
                sys.executable,
                "-m",
                "scruffy.mcp_server",
                "--root",
                str(self.root),
                "--transport",
                "streamable-http",
                "--port",
                str(self.port),
            ],
            env=environment,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        health = f"http://127.0.0.1:{self.port}/health"
        for _ in range(100):
            try:
                if b'"observer":"connected"' in await asyncio.to_thread(_read_url, health):
                    break
            except OSError:
                if self.process.poll() is not None:
                    self.log.seek(0)
                    self.fail(self.log.read())
                await asyncio.sleep(0.05)
        else:
            self.fail("HTTP hub did not become healthy")

    async def asyncTearDown(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.log.close()
        self.temporary.cleanup()

    async def test_streamable_http_header_scope_and_shared_waits(self) -> None:
        url = f"http://127.0.0.1:{self.port}/mcp"
        headers = {"x-scruffy-project": "project-a"}
        async with (
            httpx.AsyncClient(headers=headers) as client_a,
            httpx.AsyncClient(headers=headers) as client_b,
            streamable_http_client(url, http_client=client_a) as (reader_a, writer_a, _),
            streamable_http_client(url, http_client=client_b) as (reader_b, writer_b, _),
            ClientSession(reader_a, writer_a) as session_a,
            ClientSession(reader_b, writer_b) as session_b,
        ):
            await session_a.initialize()
            await session_b.initialize()
            tools = await session_a.list_tools()
            self.assertIn("submit_job", {tool.name for tool in tools.tools})
            overview_a = (await session_a.call_tool("overview", {})).structuredContent
            overview_b = (await session_b.call_tool("overview", {})).structuredContent
            self.assertEqual("project-a", overview_a["project_id"])
            wait_a = asyncio.create_task(
                session_a.call_tool(
                    "wait_for_updates",
                    {"after": overview_a["as_of_cursor"], "timeout_seconds": 2},
                )
            )
            wait_b = asyncio.create_task(
                session_b.call_tool(
                    "wait_for_updates",
                    {"after": overview_b["as_of_cursor"], "timeout_seconds": 2},
                )
            )
            await asyncio.sleep(0.1)
            event = {
                "v": 1,
                "queue_id": queue_id(self.root),
                "seq": 1,
                "kind": "job.running",
                "job_id": "job-1",
                "project_id": "project-a",
                "job": {
                    "id": "job-1",
                    "project_id": "project-a",
                    "name": "train",
                    "state": "running",
                },
            }
            with open_journal(self.root) as journal:
                append_event(journal, event, sync=True)
                state = {
                    "v": 1,
                    "queue_id": queue_id(self.root),
                    "last_seq": 1,
                    "journal_generation": 0,
                    "journal_offset": journal.tell(),
                    "allocation": {"id": "allocation-1", "state": "running"},
                    "nodes": {},
                    "jobs": {"job-1": event["job"]},
                    "draining": False,
                    "archived_jobs": 0,
                    "archived_counts": {},
                }
            write_state(self.root, state)
            results = [result.structuredContent for result in await asyncio.gather(wait_a, wait_b)]

        self.assertEqual(
            [["job.running"], ["job.running"]],
            [[event["kind"] for event in result["events"]] for result in results],
        )
        self.assertEqual(results[0]["next_cursor"], results[1]["next_cursor"])


if __name__ == "__main__":
    unittest.main()
