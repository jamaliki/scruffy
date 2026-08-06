"""Restartable remote calls for the local Scruffy MCP gateway."""

from __future__ import annotations

import asyncio
import json
import math
import os
import shlex
import signal
import uuid
from collections.abc import Awaitable, Callable, Sequence
from typing import Any


class GatewayError(RuntimeError):
    """A remote call failed without closing the local MCP transport."""


RemoteCall = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]

CALL_TIMEOUT_SECONDS = 120.0
MAX_WAIT_SECONDS = 60.0 * 60
WAIT_TIMEOUT_GRACE_SECONDS = 5.0


def _call_timeout(tool: str, params: dict[str, Any]) -> float:
    """Bound one connector process, including a wedged remote wait."""

    requested = params.get("timeout_seconds")
    if (
        tool == "wait_for_updates"
        and not isinstance(requested, bool)
        and isinstance(requested, (int, float))
        and math.isfinite(requested)
        and 0 <= requested <= MAX_WAIT_SECONDS
    ):
        return float(requested) + WAIT_TIMEOUT_GRACE_SECONDS
    return CALL_TIMEOUT_SECONDS


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    """Stop a cancelled connector and every process it launched."""

    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=2)
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


def _decode_response(
    stdout: bytes,
    stderr: bytes,
    returncode: int,
    request_id: str,
) -> dict[str, Any]:
    """Validate one response from the remote one-shot Scruffy process."""

    if returncode:
        detail = stderr.decode(errors="replace").strip()[-2000:]
        suffix = f": {detail}" if detail else ""
        raise GatewayError(
            f"Scruffy connector failed [{request_id}] with exit {returncode}{suffix}; "
            "retry this tool call"
        )
    try:
        envelope = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GatewayError(
            f"Scruffy connector returned an invalid response [{request_id}]; retry this tool call"
        ) from exc
    if not isinstance(envelope, dict) or envelope.get("request_id") != request_id:
        raise GatewayError(f"Scruffy connector response did not match request [{request_id}]")
    if envelope.get("ok") is True and isinstance(envelope.get("result"), dict):
        return envelope["result"]
    error = envelope.get("error")
    if envelope.get("ok") is False and isinstance(error, dict):
        error_type = str(error.get("type", "error"))
        message = str(error.get("message", "remote Scruffy call failed"))
        raise GatewayError(f"Remote Scruffy {error_type} [{request_id}]: {message}")
    raise GatewayError(f"Scruffy connector returned an invalid envelope [{request_id}]")


async def call_remote(
    connect_command: Sequence[str],
    remote_command: Sequence[str],
    root: str,
    tool: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Run one read-only tool call in a fresh remote process."""

    request_id = uuid.uuid4().hex[:12]
    encoded = json.dumps(params, separators=(",", ":"), allow_nan=False)
    remote_argv = [
        *remote_command,
        "--root",
        root,
        "--rpc-tool",
        tool,
        "--rpc-params",
        encoded,
        "--rpc-id",
        request_id,
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *connect_command,
            shlex.join(remote_argv),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise GatewayError(
            f"Scruffy connector could not start [{request_id}]: {exc}; retry this tool call"
        ) from exc
    timeout = _call_timeout(tool, params)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError as exc:
        await _stop_process(process)
        raise GatewayError(
            f"Remote Scruffy {tool} exceeded its {timeout:g}s connector deadline "
            f"[{request_id}]; retry this tool call"
        ) from exc
    except asyncio.CancelledError:
        await _stop_process(process)
        raise
    return _decode_response(stdout, stderr, process.returncode, request_id)


def remote_caller(
    connect_command: Sequence[str], remote_command: Sequence[str], root: str
) -> RemoteCall:
    """Bind remote process details once for FastMCP tool functions."""

    async def call(tool: str, params: dict[str, Any]) -> dict[str, Any]:
        return await call_remote(connect_command, remote_command, root, tool, params)

    return call


__all__ = ["GatewayError", "RemoteCall", "call_remote", "remote_caller"]
