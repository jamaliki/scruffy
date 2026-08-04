from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scruffy.mcp_server import (
    compact_event,
    compact_explanation,
    event_matches,
    wait_for_updates,
)
from scruffy.storage import (
    append_event,
    load_state,
    open_journal,
    queue_id,
    write_state,
)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ModuleNotFoundError:
    ClientSession = StdioServerParameters = stdio_client = None


def _base_state(root: Path) -> dict:
    return {
        "v": 1,
        "queue_id": queue_id(root),
        "last_seq": 0,
        "journal_generation": 0,
        "journal_offset": 0,
        "allocation": {"id": "allocation-1", "state": "running"},
        "nodes": {},
        "jobs": {},
        "draining": False,
        "archived_jobs": 0,
        "archived_counts": {},
    }


def _commit(
    root: Path,
    events: list[dict],
    *,
    jobs: dict[str, dict] | None = None,
) -> dict:
    state = load_state(root) or _base_state(root)
    with open_journal(root) as journal:
        for event in events:
            append_event(journal, event, sync=True)
        state["journal_offset"] = journal.tell()
    if events:
        state["last_seq"] = max(state["last_seq"], max(event["seq"] for event in events))
    if jobs is not None:
        state["jobs"] = jobs
    write_state(root, state)
    return state


def _job(job_id: str, state: str = "running") -> dict:
    return {
        "id": job_id,
        "name": "training",
        "state": state,
        "workflow_id": "workflow-1",
        "task_id": "train",
        "assignment": {"node-1": [0]},
        "workload": {"phase": "training"},
        "argv": ["python", "train.py"],
        "cwd": "/secret/checkout",
        "environment": {"TOKEN": "secret"},
    }


class ProjectionTests(unittest.TestCase):
    def test_compact_views_exclude_process_details(self) -> None:
        job = _job("job-1")
        event = {
            "v": 1,
            "seq": 1,
            "event_id": "queue:1",
            "kind": "job.running",
            "job_id": "job-1",
            "job": job,
        }

        compact = compact_event(event, {"jobs": {"job-1": job}})
        explanation = compact_explanation(
            {"v": 1, "job": job, "dependencies": [], "explanation": "running"}
        )

        for view in (compact["job"], explanation["job"]):
            self.assertEqual("running", view["state"])
            self.assertEqual({"phase": "training"}, view["workload"])
            self.assertNotIn("argv", view)
            self.assertNotIn("cwd", view)
            self.assertNotIn("environment", view)

    def test_default_and_exact_kind_filters(self) -> None:
        snapshot = {"jobs": {}}
        self.assertFalse(event_matches({"kind": "job.output"}, snapshot))
        self.assertFalse(event_matches({"kind": "workload.progress"}, snapshot))
        self.assertTrue(event_matches({"kind": "workload.milestone"}, snapshot))
        self.assertTrue(
            event_matches(
                {"kind": "workload.progress"},
                snapshot,
                event_kinds={"workload.progress"},
            )
        )
        self.assertFalse(
            event_matches(
                {"kind": "job.failed"},
                snapshot,
                event_kinds={"workload.progress"},
            )
        )

    def test_scopes_intersect_but_global_events_always_match(self) -> None:
        job = _job("job-1")
        snapshot = {"jobs": {"job-1": job}}
        event = {"kind": "job.running", "job_id": "job-1", "job": job}

        self.assertTrue(
            event_matches(
                event,
                snapshot,
                workflow_id="workflow-1",
                job_ids={"job-1"},
            )
        )
        self.assertFalse(event_matches(event, snapshot, workflow_id="other", job_ids={"job-1"}))
        self.assertFalse(event_matches(event, snapshot, job_ids={"job-2"}))
        self.assertTrue(
            event_matches(
                {"kind": "allocation.ended"},
                snapshot,
                workflow_id="other",
                job_ids={"job-2"},
            )
        )


class WaitTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"
        self.identity = queue_id(self.root)

    def event(self, sequence: int, kind: str, job_id: str = "job-1") -> dict:
        return {
            "v": 1,
            "queue_id": self.identity,
            "seq": sequence,
            "event_id": f"{self.identity}:{sequence}",
            "kind": kind,
            "job_id": job_id,
        }

    async def test_noise_is_suppressed_and_cursor_is_advanced(self) -> None:
        job = _job("job-1")
        _commit(
            self.root,
            [
                self.event(1, "workload.progress"),
                self.event(2, "job.output"),
            ],
            jobs={"job-1": job},
        )

        quiet = await wait_for_updates(self.root, after="0", timeout_seconds=0)

        self.assertTrue(quiet["timed_out"])
        self.assertEqual([], quiet["events"])
        self.assertIn(":0:2:", quiet["next_cursor"])

        milestone = self.event(3, "workload.milestone")
        milestone["data"] = {"name": "checkpoint"}
        _commit(self.root, [milestone], jobs={"job-1": job})
        update = await wait_for_updates(self.root, after=quiet["next_cursor"], timeout_seconds=0)

        self.assertFalse(update["timed_out"])
        self.assertEqual(["workload.milestone"], [item["kind"] for item in update["events"]])

    async def test_filtered_backlog_is_drained_page_by_page(self) -> None:
        events = [self.event(seq, "workload.progress") for seq in range(1, 66)]
        events.append(self.event(66, "job.succeeded"))
        _commit(self.root, events, jobs={"job-1": _job("job-1", "succeeded")})

        result = await wait_for_updates(self.root, after="0", timeout_seconds=1)

        self.assertEqual([66], [event["seq"] for event in result["events"]])
        self.assertFalse(result["more"])

    async def test_readers_keep_independent_cursors(self) -> None:
        _commit(
            self.root,
            [self.event(1, "job.running")],
            jobs={"job-1": _job("job-1")},
        )

        first = await wait_for_updates(self.root, after="0", timeout_seconds=0)
        second = await wait_for_updates(self.root, after="0", timeout_seconds=0)

        self.assertEqual([1], [event["seq"] for event in first["events"]])
        self.assertEqual(first["events"], second["events"])
        self.assertEqual(first["next_cursor"], second["next_cursor"])

    async def test_stale_cursor_returns_authoritative_overview(self) -> None:
        _commit(self.root, [], jobs={"job-1": _job("job-1")})

        result = await wait_for_updates(
            self.root,
            after="another-queue:0:0:0",
            timeout_seconds=0,
        )

        self.assertTrue(result["reset"])
        self.assertFalse(result["timed_out"])
        self.assertEqual(result["next_cursor"], result["overview"]["as_of_cursor"])
        self.assertEqual("running", result["overview"]["active"][0]["state"])

    async def test_wait_is_cancellable_without_a_polling_thread(self) -> None:
        entered = asyncio.Event()
        calls = 0

        def observe_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            entered.set()
            return {
                "snapshot": {"jobs": {}},
                "events": [],
                "next_cursor": "queue:0:0:0",
                "latest_cursor": "queue:0:0:0",
                "more": False,
                "reset": False,
            }

        with mock.patch("scruffy.mcp_server.observe", side_effect=observe_once):
            task = asyncio.create_task(
                wait_for_updates(
                    self.root,
                    after="queue:0:0:0",
                    timeout_seconds=3600,
                    poll_seconds=3600,
                )
            )
            await entered.wait()
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0)

        self.assertEqual(1, calls)

    async def test_invalid_wait_arguments_fail_before_observation(self) -> None:
        with mock.patch("scruffy.mcp_server.observe") as observe_mock:
            with self.assertRaisesRegex(ValueError, "between 0"):
                await wait_for_updates(self.root, timeout_seconds=3601)
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                await wait_for_updates(self.root, event_kinds=[])
        observe_mock.assert_not_called()


@unittest.skipUnless(ClientSession is not None, "MCP extra is not installed")
class McpProtocolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"
        self.identity = queue_id(self.root)
        self.job_id = "job-mcp"
        _commit(self.root, [], jobs={self.job_id: _job(self.job_id)})

    @staticmethod
    def structured(result) -> dict:
        if result.structuredContent is not None:
            return result.structuredContent
        text = next(item.text for item in result.content if item.type == "text")
        return json.loads(text)

    async def test_stdio_tools_and_blocking_wake(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(source_root), environment.get("PYTHONPATH")))
        )
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "scruffy.mcp_server", "--root", str(self.root)],
            env=environment,
        )

        async with (
            stdio_client(parameters) as (reader, writer),
            ClientSession(reader, writer) as session,
        ):
            await session.initialize()
            tools = await session.list_tools()
            self.assertEqual(
                {"overview", "inspect_job", "wait_for_updates"},
                {tool.name for tool in tools.tools},
            )
            overview = self.structured(await session.call_tool("overview", {}))
            inspected = self.structured(
                await session.call_tool("inspect_job", {"job_id": self.job_id})
            )
            self.assertNotIn("environment", inspected["job"])
            invalid = await session.call_tool("wait_for_updates", {"timeout_seconds": 3601})
            self.assertTrue(invalid.isError)
            self.assertFalse(
                (await session.call_tool("overview", {})).isError,
                "one tool error must not terminate the MCP server",
            )

            pending = asyncio.create_task(
                session.call_tool(
                    "wait_for_updates",
                    {
                        "after": overview["as_of_cursor"],
                        "timeout_seconds": 2,
                        "job_ids": [self.job_id],
                    },
                )
            )
            await asyncio.sleep(0.1)
            finished = _job(self.job_id, "succeeded")
            finished["finished_at"] = "2026-08-03T12:00:00+00:00"
            event = {
                "v": 1,
                "queue_id": self.identity,
                "seq": 1,
                "event_id": f"{self.identity}:1",
                "kind": "job.succeeded",
                "job_id": self.job_id,
                "job": finished,
            }
            _commit(self.root, [event], jobs={self.job_id: finished})
            result = self.structured(await pending)

        self.assertEqual(["job.succeeded"], [event["kind"] for event in result["events"]])
        self.assertFalse(result["timed_out"])
        self.assertNotIn("argv", result["events"][0]["job"])

    async def test_local_gateway_survives_remote_failure_and_cancellation(self) -> None:
        connector = self.root.parent / "connector.py"
        counter = self.root.parent / "connector-count"
        connector.write_text(
            """\
import os
import shlex
import stat
from pathlib import Path
import sys

if not stat.S_ISCHR(os.fstat(0).st_mode):
    print("connector inherited MCP stdin", file=sys.stderr)
    raise SystemExit(64)
counter = Path(os.environ["SCRUFFY_TEST_CONNECTOR_COUNT"])
count = int(counter.read_text() if counter.exists() else "0") + 1
counter.write_text(str(count))
if count == 1:
    print("simulated SSH disconnect", file=sys.stderr)
    raise SystemExit(255)
command = shlex.split(sys.argv[1])
os.execv(command[0], command)
"""
        )
        source_root = Path(__file__).resolve().parents[1] / "src"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(source_root), environment.get("PYTHONPATH")))
        )
        environment["SCRUFFY_TEST_CONNECTOR_COUNT"] = str(counter)
        connect_command = " ".join((shlex.quote(sys.executable), shlex.quote(str(connector))))
        remote_command = f"{shlex.quote(sys.executable)} -m scruffy.mcp_server"
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "scruffy.mcp_server",
                "--root",
                str(self.root),
                "--connect-command",
                connect_command,
                "--remote-command",
                remote_command,
            ],
            env=environment,
        )

        async with (
            stdio_client(parameters) as (reader, writer),
            ClientSession(reader, writer) as session,
        ):
            await session.initialize()
            failed = await session.call_tool("overview", {})
            failure_text = "".join(item.text for item in failed.content if item.type == "text")
            self.assertTrue(failed.isError)
            self.assertIn("retry this tool call", failure_text)

            recovered = self.structured(await session.call_tool("overview", {}))
            self.assertEqual(self.identity, recovered["queue_id"])

            pending = asyncio.create_task(
                session.call_tool(
                    "wait_for_updates",
                    {
                        "after": recovered["as_of_cursor"],
                        "timeout_seconds": 600,
                        "event_kinds": ["diagnostic.never"],
                    },
                )
            )
            await asyncio.sleep(0.2)
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending

            after_cancel = self.structured(await session.call_tool("overview", {}))
            self.assertEqual(self.identity, after_cancel["queue_id"])

        self.assertGreaterEqual(int(counter.read_text()), 4)


if __name__ == "__main__":
    unittest.main()
