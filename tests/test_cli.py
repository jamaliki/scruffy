from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scruffy.cli import main
from scruffy.models import NodeInventory, ResourceRequest
from scruffy.storage import UnsafeRecovery


class SubmitCliTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.root = self.workspace / "queue"

    def test_submit_strips_separator_and_passes_argv_without_a_shell(self) -> None:
        response = {"job_id": "job-1", "state": "submitted"}
        with (
            mock.patch("scruffy.cli.submit_job", return_value=response) as submit,
            mock.patch("scruffy.cli._json") as print_json,
        ):
            result = main(
                [
                    "--root",
                    str(self.root),
                    "submit",
                    "--name",
                    "training",
                    "--nodes",
                    "2",
                    "--gpus-per-node",
                    "3",
                    "--env",
                    "MODE=fast",
                    "--request-id",
                    "agent-a/run-1",
                    "--",
                    "python",
                    "-c",
                    "print('hello')",
                ]
            )

        self.assertEqual(0, result)
        print_json.assert_called_once_with(response)
        submit.assert_called_once_with(
            self.root,
            argv=["python", "-c", "print('hello')"],
            name="training",
            cwd=Path.cwd(),
            environment={"MODE": "fast"},
            request=ResourceRequest(
                nodes=2,
                gpus_per_node=3,
                cpus_per_node=42,
                memory_gb_per_node=384,
            ),
            request_id="agent-a/run-1",
        )

    def test_submit_requires_a_command(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch("scruffy.cli.submit_job") as submit,
            contextlib.redirect_stderr(stderr),
        ):
            result = main(["--root", str(self.root), "submit"])

        self.assertEqual(2, result)
        self.assertIn("submit requires a command", stderr.getvalue())
        submit.assert_not_called()

    def test_cpu_only_submit_keeps_positive_cpu_and_memory_defaults(self) -> None:
        with mock.patch("scruffy.cli.submit_job", return_value={}) as submit:
            result = main(
                [
                    "--root",
                    str(self.root),
                    "submit",
                    "--gpus-per-node",
                    "0",
                    "--",
                    "true",
                ]
            )

        self.assertEqual(0, result)
        request = submit.call_args.kwargs["request"]
        self.assertEqual(ResourceRequest(1, 0, 14, 128), request)

    def test_submit_rejects_malformed_environment_override(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch("scruffy.cli.submit_job") as submit,
            contextlib.redirect_stderr(stderr),
        ):
            result = main(
                [
                    "--root",
                    str(self.root),
                    "submit",
                    "--env",
                    "MISSING_SEPARATOR",
                    "--",
                    "true",
                ]
            )

        self.assertEqual(2, result)
        self.assertIn("KEY=VALUE", stderr.getvalue())
        submit.assert_not_called()

    def test_explicit_zero_resource_budget_is_not_replaced_by_a_default(self) -> None:
        for option in ("--cpus-per-node", "--memory-gb-per-node"):
            with self.subTest(option=option):
                stderr = io.StringIO()
                with (
                    mock.patch("scruffy.cli.submit_job") as submit,
                    contextlib.redirect_stderr(stderr),
                ):
                    result = main(
                        ["--root", str(self.root), "submit", option, "0", "--", "true"]
                    )

                self.assertEqual(2, result)
                self.assertIn("positive integer", stderr.getvalue())
                submit.assert_not_called()


class ServeCliTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.root = self.workspace / "queue"
        self.inventory = {"gpu-3": NodeInventory("gpu-3", (0, 1), 16, 128)}

    def test_auto_launcher_uses_local_outside_slurm(self) -> None:
        inventory_file = self.workspace / "inventory.json"
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("scruffy.cli.load_inventory", return_value=self.inventory),
            mock.patch("scruffy.cli.run_controller") as run_controller,
        ):
            result = main(
                [
                    "--root",
                    str(self.root),
                    "serve",
                    "--inventory",
                    str(inventory_file),
                    "--allocation-id",
                    "local-test",
                    "--poll-interval",
                    "0.1",
                    "--cancel-grace",
                    "2.5",
                ]
            )

        self.assertEqual(0, result)
        run_controller.assert_called_once_with(
            root=self.root,
            inventory=tuple(self.inventory.values()),
            launcher="local",
            allocation_id="local-test",
            slurm_job_id=None,
            poll_interval=0.1,
            cancel_grace=2.5,
        )

    def test_automatic_inventory_requires_an_explicit_gpu_count(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("scruffy.cli.run_controller") as run_controller,
            contextlib.redirect_stderr(stderr),
        ):
            result = main(["--root", str(self.root), "serve"])

        self.assertEqual(2, result)
        self.assertIn("--gpus-per-node is required", stderr.getvalue())
        run_controller.assert_not_called()

    def test_slurm_launcher_requires_a_job_id_before_starting(self) -> None:
        stderr = io.StringIO()
        inventory_file = self.workspace / "inventory.json"
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("scruffy.cli.load_inventory", return_value=self.inventory),
            mock.patch("scruffy.cli.run_controller") as run_controller,
            contextlib.redirect_stderr(stderr),
        ):
            result = main(
                [
                    "--root",
                    str(self.root),
                    "serve",
                    "--inventory",
                    str(inventory_file),
                    "--launcher",
                    "slurm",
                    "--allocation-id",
                    "test-allocation",
                ]
            )

        self.assertEqual(2, result)
        self.assertIn("Slurm job ID", stderr.getvalue())
        run_controller.assert_not_called()

    def test_expected_recovery_refusal_is_reported_without_a_traceback(self) -> None:
        stderr = io.StringIO()
        inventory_file = self.workspace / "inventory.json"
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("scruffy.cli.load_inventory", return_value=self.inventory),
            mock.patch(
                "scruffy.cli.run_controller",
                side_effect=UnsafeRecovery("unsafe active jobs"),
            ),
            contextlib.redirect_stderr(stderr),
        ):
            result = main(
                [
                    "--root",
                    str(self.root),
                    "serve",
                    "--inventory",
                    str(inventory_file),
                    "--allocation-id",
                    "local-test",
                ]
            )

        self.assertEqual(2, result)
        self.assertEqual("scruffy: unsafe active jobs\n", stderr.getvalue())


class WatchCliTests(unittest.TestCase):
    def test_watch_flushes_the_initial_snapshot(self) -> None:
        response = {
            "snapshot": {"allocation": None, "jobs": {}},
            "events": [],
            "next_cursor": "queue-test:0:0",
        }
        with (
            mock.patch("scruffy.cli.observe", return_value=response),
            mock.patch("builtins.print") as print_output,
        ):
            result = main(["--root", "/tmp/scruffy-test", "watch"])

        self.assertEqual(0, result)
        print_output.assert_called_once_with(
            '{"kind": "snapshot", "data": {"allocation": null, "jobs": {}}}',
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
