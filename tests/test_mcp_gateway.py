from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from scruffy import mcp_gateway


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    def test_wait_deadline_tracks_the_requested_timeout(self) -> None:
        self.assertEqual(
            305,
            mcp_gateway._call_timeout("wait_for_updates", {"timeout_seconds": 300}),
        )
        self.assertEqual(
            mcp_gateway.CALL_TIMEOUT_SECONDS,
            mcp_gateway._call_timeout("overview", {}),
        )

    async def test_wedged_remote_wait_is_stopped(self) -> None:
        async def hang() -> tuple[bytes, bytes]:
            await asyncio.sleep(60)
            return b"", b""

        process = mock.Mock(pid=123, returncode=None)
        process.communicate = mock.AsyncMock(side_effect=hang)

        with (
            mock.patch.object(
                mcp_gateway.asyncio,
                "create_subprocess_exec",
                new=mock.AsyncMock(return_value=process),
            ),
            mock.patch.object(
                mcp_gateway, "_call_timeout", return_value=0.01
            ),
            mock.patch.object(
                mcp_gateway, "_stop_process", new=mock.AsyncMock()
            ) as stop_process,
            self.assertRaisesRegex(mcp_gateway.GatewayError, "connector deadline"),
        ):
            await mcp_gateway.call_remote(
                ["connector"],
                ["scruffy-mcp"],
                "/queue",
                "wait_for_updates",
                {"timeout_seconds": 300},
            )

        stop_process.assert_awaited_once_with(process)


if __name__ == "__main__":
    unittest.main()
