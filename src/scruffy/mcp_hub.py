"""One shared remote observer for the loopback Scruffy MCP server."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Collection
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from .mcp_gateway import RemoteCall
from .mcp_server import (
    DEFAULT_TIMEOUT_SECONDS,
    _validate_wait,
    event_matches,
)
from .models import normalize_project_id

POLL_SECONDS = 30.0
RETRY_SECONDS = 1.0
MAX_BUFFERED_EVENTS = 4096


@dataclass(frozen=True)
class Cursor:
    queue_id: str
    generation: int
    sequence: int
    offset: int


@dataclass(frozen=True)
class EventPage:
    after: str
    next_cursor: str
    events: tuple[dict[str, Any], ...]


def parse_cursor(value: str) -> Cursor:
    """Parse the current opaque Scruffy cursor shape for local comparisons."""

    if not isinstance(value, str):
        raise TypeError("cursor must be a string")
    parts = value.rsplit(":", 3)
    if len(parts) != 4 or not parts[0]:
        raise ValueError("invalid cursor")
    numbers = parts[1:]
    if any(not item.isascii() or not item.isdecimal() for item in numbers):
        raise ValueError("invalid cursor")
    return Cursor(parts[0], *(int(item) for item in numbers))


def _position(cursor: Cursor) -> tuple[int, int, int]:
    return cursor.generation, cursor.sequence, cursor.offset


def _public_event(event: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if not key.startswith("_")}


def _has_more(cursor: str, latest: str) -> bool:
    """Return whether latest is ahead in the same journal generation."""

    current, end = parse_cursor(cursor), parse_cursor(latest)
    return (
        current.queue_id == end.queue_id
        and current.generation == end.generation
        and _position(current) < _position(end)
    )


class UpdateBroker:
    """Fan one queue observer out to independent, non-consuming agent cursors."""

    def __init__(
        self,
        remote: RemoteCall,
        *,
        poll_seconds: float = POLL_SECONDS,
        max_events: int = MAX_BUFFERED_EVENTS,
    ) -> None:
        self.remote = remote
        self.poll_seconds = poll_seconds
        self.max_events = max_events
        self.pages: deque[EventPage] = deque()
        self.floor_cursor: str | None = None
        self.current_cursor: str | None = None
        self.latest_cursor: str | None = None
        self.last_error: str | None = None
        self._event_count = 0
        self._condition = asyncio.Condition()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the sole upstream observer without delaying HTTP startup."""

        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="scruffy-mcp-observer")
            await asyncio.sleep(0)  # Let lifespan start the observer before yielding.

    async def close(self) -> None:
        """Cancel the observer and its current connector process."""

        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def health(self) -> dict[str, Any]:
        """Return bounded process and upstream status for launch supervision."""

        task_state = "stopped"
        if self._task is not None:
            task_state = "failed" if self._task.done() else "running"
        return {
            "status": "ok",
            "observer": "connected" if self.current_cursor else "starting",
            "task": task_state,
            "cursor": self.current_cursor,
            "buffered_events": self._event_count,
            "last_error": self.last_error,
        }

    async def _run(self) -> None:
        while True:
            try:
                if self.current_cursor is None:
                    overview = await self.remote("overview", {})
                    cursor = overview["as_of_cursor"]
                    parse_cursor(cursor)
                    async with self._condition:
                        self.floor_cursor = self.current_cursor = self.latest_cursor = cursor
                        self.last_error = None
                        self._condition.notify_all()

                after = self.current_cursor
                response = await self.remote(
                    "_poll_updates",
                    {"after": after, "timeout_seconds": self.poll_seconds},
                )
                async with self._condition:
                    self._accept(after, response)
                    self.last_error = None
                    self._condition.notify_all()
            except asyncio.CancelledError:
                raise
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                async with self._condition:
                    self.last_error = str(exc)
                    self._condition.notify_all()
                await asyncio.sleep(RETRY_SECONDS)

    def _accept(self, after: str, response: dict[str, Any]) -> None:
        next_cursor = response["next_cursor"]
        latest_cursor = response["latest_cursor"]
        parse_cursor(next_cursor)
        parse_cursor(latest_cursor)
        if response.get("reset"):
            self.pages.clear()
            self._event_count = 0
            self.floor_cursor = next_cursor
        elif next_cursor != after:
            events = tuple(response.get("events") or ())
            self.pages.append(EventPage(after, next_cursor, events))
            self._event_count += len(events)
            self._trim()
        self.current_cursor = next_cursor
        self.latest_cursor = latest_cursor

    def _trim(self) -> None:
        while len(self.pages) > 1 and self._event_count > self.max_events:
            page = self.pages.popleft()
            self._event_count -= len(page.events)
            self.floor_cursor = page.next_cursor

    def _scan(
        self,
        after: str,
        *,
        workflow_id: str | None,
        job_ids: Collection[str] | None,
        event_kinds: Collection[str] | None,
        project_id: str | None,
    ) -> tuple[dict[str, Any] | None, str, bool]:
        """Scan buffered pages, returning result, advanced cursor, and reset."""

        assert self.floor_cursor is not None
        assert self.current_cursor is not None
        assert self.latest_cursor is not None
        try:
            requested = parse_cursor(after)
            floor = parse_cursor(self.floor_cursor)
            current = parse_cursor(self.current_cursor)
        except ValueError:
            return None, self.current_cursor, True
        if (
            requested.queue_id != current.queue_id
            or requested.generation != current.generation
            or _position(requested) < _position(floor)
        ):
            return None, self.current_cursor, True
        cursor = after
        for page in self.pages:
            page_end = parse_cursor(page.next_cursor)
            if _position(page_end) <= _position(requested):
                continue
            matches = [
                _public_event(event)
                for event in page.events
                if int(event.get("_seq", -1)) > requested.sequence
                and event_matches(
                    event,
                    {},
                    workflow_id=workflow_id,
                    job_ids=job_ids,
                    event_kinds=event_kinds,
                    project_id=project_id,
                )
            ]
            cursor = page.next_cursor
            if matches:
                return (
                    {
                        "events": matches,
                        "next_cursor": cursor,
                        "latest_cursor": self.latest_cursor,
                        "more": _has_more(cursor, self.latest_cursor),
                        "reset": False,
                        "timed_out": False,
                    },
                    cursor,
                    False,
                )
            requested = page_end
        return None, cursor, False

    async def wait(
        self,
        *,
        after: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        workflow_id: str | None = None,
        job_ids: Collection[str] | None = None,
        event_kinds: Collection[str] | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Wait for a relevant buffered event without opening another SSH wait."""

        timeout, selected_jobs, selected_kinds = _validate_wait(
            timeout_seconds, workflow_id, job_ids, event_kinds
        )
        selected_project = normalize_project_id(project_id) if project_id else None
        deadline = asyncio.get_running_loop().time() + timeout
        cursor = after
        if cursor is None:
            overview_params = {"_project_id": selected_project} if selected_project else {}
            overview = await self.remote("overview", overview_params)
            cursor = overview["as_of_cursor"]
        while True:
            reset = False
            async with self._condition:
                if self.current_cursor is not None:
                    result, cursor, reset = self._scan(
                        cursor,
                        workflow_id=workflow_id,
                        job_ids=selected_jobs,
                        event_kinds=selected_kinds,
                        project_id=selected_project,
                    )
                    if result is not None:
                        return result
                remaining = deadline - asyncio.get_running_loop().time()
                if reset or remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(self._condition.wait(), remaining)
                except TimeoutError:
                    break

        if reset:
            overview_params = {"_project_id": selected_project} if selected_project else {}
            overview = await self.remote("overview", overview_params)
            return {
                "events": [],
                "next_cursor": overview["as_of_cursor"],
                "latest_cursor": overview["as_of_cursor"],
                "more": False,
                "reset": True,
                "timed_out": False,
                "overview": overview,
            }
        latest = self.latest_cursor or cursor
        if cursor is None or latest is None:
            detail = f": {self.last_error}" if self.last_error else ""
            raise RuntimeError(f"Scruffy observer has not started{detail}")
        current_position, latest_position = parse_cursor(cursor), parse_cursor(latest)
        if (
            current_position.queue_id == latest_position.queue_id
            and current_position.generation == latest_position.generation
            and _position(current_position) > _position(latest_position)
        ):
            latest = cursor
        return {
            "events": [],
            "next_cursor": cursor,
            "latest_cursor": latest,
            "more": _has_more(cursor, latest),
            "reset": False,
            "timed_out": True,
        }


class HubCaller:
    """Route waits through the broker and short operations to the queue."""

    def __init__(self, remote: RemoteCall, broker: UpdateBroker) -> None:
        self.remote = remote
        self.broker = broker

    async def __call__(self, tool: str, params: dict[str, Any]) -> dict[str, Any]:
        if tool != "wait_for_updates":
            return await self.remote(tool, params)
        values = dict(params)
        project_id = values.pop("_project_id", None)
        return await self.broker.wait(project_id=project_id, **values)


def hub_app(server: Any, broker: UpdateBroker) -> Any:
    """Build the HTTP app with the broker bound to its process lifetime."""

    app = server.streamable_http_app()
    session_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application: Any):
        async with session_lifespan(application) as state:
            await broker.start()
            try:
                yield state
            finally:
                await broker.close()

    app.router.lifespan_context = lifespan
    return app


def run_hub(server: Any, broker: UpdateBroker) -> None:
    """Run the supervised loopback HTTP endpoint until interrupted."""

    import uvicorn

    uvicorn.run(
        hub_app(server, broker),
        host=server.settings.host,
        port=server.settings.port,
        log_level=server.settings.log_level.lower(),
    )


__all__ = ["HubCaller", "UpdateBroker", "hub_app", "parse_cursor", "run_hub"]
