"""Small read-only web dashboard for a Scruffy allocation."""

from __future__ import annotations

import asyncio
import json
import shlex
import webbrowser
from collections.abc import Callable, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .mcp_gateway import remote_caller
from .mcp_server import dispatch_tool

QueueReader = Callable[[str, dict[str, Any]], dict[str, Any]]
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/assets/app.css": ("app.css", "text/css; charset=utf-8"),
    "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/assets/model.js": ("model.js", "text/javascript; charset=utf-8"),
}
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; style-src 'self'; "
        "script-src 'self'; connect-src 'self'; font-src 'self'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


def local_reader(root: Path) -> QueueReader:
    """Return a compact tool reader for a locally visible queue root."""

    def read(tool: str, params: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(dispatch_tool(root, tool, params))

    return read


def gateway_reader(
    root: str,
    connect_command: Sequence[str],
    remote_command: Sequence[str],
) -> QueueReader:
    """Return a synchronous reader whose individual calls use fresh connectors."""

    call = remote_caller(connect_command, remote_command, root)

    def read(tool: str, params: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(call(tool, params))

    return read


def _handler(reader: QueueReader) -> type[BaseHTTPRequestHandler]:
    assets = files("scruffy").joinpath("dashboard_assets")

    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "ScruffyDashboard/1"

        def _headers(self, status: HTTPStatus, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            for name, value in SECURITY_HEADERS.items():
                self.send_header(name, value)
            self.end_headers()

        def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
            self._headers(status, "application/json; charset=utf-8")
            self.wfile.write(body)

        def _error(self, status: HTTPStatus, message: str) -> None:
            self._json({"error": message}, status)

        def _valid_host(self) -> bool:
            hostname = (self.headers.get("Host") or "").partition(":")[0]
            return hostname in {"127.0.0.1", "localhost"}

        def do_GET(self) -> None:
            if not self._valid_host():
                self._error(HTTPStatus.MISDIRECTED_REQUEST, "dashboard is loopback-only")
                return
            request = urlsplit(self.path)
            try:
                if request.path == "/api/overview":
                    self._json(reader("overview", {"limit": 100, "compact": False}))
                    return
                prefix = "/api/jobs/"
                if request.path.startswith(prefix):
                    job_id = unquote(request.path[len(prefix) :])
                    if not job_id or "/" in job_id:
                        self._error(HTTPStatus.BAD_REQUEST, "invalid job id")
                        return
                    self._json(reader("inspect_job", {"job_id": job_id}))
                    return
                asset = ASSETS.get(request.path)
                if asset is None:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
                    return
                filename, content_type = asset
                body = assets.joinpath(filename).read_bytes()
                self._headers(HTTPStatus.OK, content_type)
                self.wfile.write(body)
            except KeyError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc))
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))

        def do_POST(self) -> None:
            self._error(HTTPStatus.METHOD_NOT_ALLOWED, "dashboard is read-only")

        def log_message(self, format: str, *args: object) -> None:
            return  # Keep the operational console quiet.

    return DashboardHandler


def create_server(
    root: str,
    *,
    port: int = 8765,
    connect_command: Sequence[str] | None = None,
    remote_command: Sequence[str] | None = None,
) -> ThreadingHTTPServer:
    """Create a loopback-only dashboard server without starting it."""

    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("dashboard port must be between 0 and 65535")
    gateway_options = (connect_command, remote_command)
    if any(option is not None for option in gateway_options) and not all(gateway_options):
        raise ValueError("dashboard connector commands must be used together")
    reader = (
        gateway_reader(root, connect_command or (), remote_command or ())
        if all(gateway_options)
        else local_reader(Path(root).expanduser().resolve())
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(reader))
    server.daemon_threads = True
    return server


def run_dashboard(
    root: str,
    *,
    port: int = 8765,
    connect_command: str | None = None,
    remote_command: str | None = None,
    open_browser: bool = True,
) -> None:
    """Serve the dashboard until interrupted."""

    connectors = (connect_command, remote_command)
    if any(connectors) and not all(connectors):
        raise ValueError("--connect-command and --remote-command must be used together")
    server = create_server(
        root,
        port=port,
        connect_command=shlex.split(connect_command) if connect_command else None,
        remote_command=shlex.split(remote_command) if remote_command else None,
    )
    address = f"http://127.0.0.1:{server.server_port}/"
    print(f"Scruffy dashboard: {address}", flush=True)
    if open_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    finally:
        server.server_close()


__all__ = ["create_server", "gateway_reader", "local_reader", "run_dashboard"]
