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
            project_id="default",
            workflow_id=None,
            task_id=None,
            needs=[],
        )

    def test_submit_parses_workflow_dependencies(self) -> None:
        response = {"job_id": "job-infer", "state": "submitted"}
        with (
            mock.patch("scruffy.cli.submit_job", return_value=response) as submit,
            mock.patch("scruffy.cli._json"),
        ):
            result = main(
                [
                    "--root",
                    str(self.root),
                    "submit",
                    "--workflow-id",
                    "run-7",
                    "--task-id",
                    "infer",
                    "--needs",
                    "train",
                    "--needs",
                    "evaluate:terminal",
                    "--",
                    "true",
                ]
            )

        self.assertEqual(0, result)
        self.assertEqual("run-7", submit.call_args.kwargs["workflow_id"])
        self.assertEqual("infer", submit.call_args.kwargs["task_id"])
        self.assertEqual(
            [
                {"task_id": "train", "condition": "succeeded"},
                {"task_id": "evaluate", "condition": "terminal"},
            ],
            submit.call_args.kwargs["needs"],
        )

    def test_submit_uses_explicit_project_namespace(self) -> None:
        with (
            mock.patch(
                "scruffy.cli.submit_job",
                return_value={"job_id": "job-1", "state": "submitted"},
            ) as submit,
            mock.patch("scruffy.cli._json"),
        ):
            result = main(
                [
                    "--root",
                    str(self.root),
                    "submit",
                    "--project",
                    "koochak",
                    "--request-id",
                    "train-1",
                    "--",
                    "true",
                ]
            )

        self.assertEqual(0, result)
        self.assertEqual("koochak", submit.call_args.kwargs["project_id"])

    def test_submit_rejects_ambiguous_colons_in_task_ids(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = main(
                [
                    "--root",
                    str(self.root),
                    "submit",
                    "--workflow-id",
                    "run-7",
                    "--task-id",
                    "eval:terminal",
                    "--",
                    "true",
                ]
            )

        self.assertEqual(2, result)
        self.assertIn("must not contain ':'", stderr.getvalue())

    def test_report_uses_worker_identity_and_parses_json(self) -> None:
        response = {"event_id": "event-1", "state": "spooled"}
        with (
            mock.patch.dict(
                os.environ,
                {"SCRUFFY_JOB_ID": "job-1", "SCRUFFY_ROOT": str(self.root)},
                clear=True,
            ),
            mock.patch("scruffy.cli.publish_event", return_value=response) as publish,
            mock.patch("scruffy.cli._json") as print_json,
        ):
            result = main(
                [
                    "report",
                    "workload.progress",
                    "--event-id",
                    "progress-4",
                    "--source",
                    "name=trainer",
                    "--data-json",
                    '{"step":4,"loss":0.25}',
                ]
            )

        self.assertEqual(0, result)
        print_json.assert_called_once_with(response)
        publish.assert_called_once_with(
            self.root,
            job_id="job-1",
            kind="workload.progress",
            data={"step": 4, "loss": 0.25},
            event_id="progress-4",
            occurred_at=None,
            source={"name": "trainer"},
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


class OperationalViewsCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("/tmp/scruffy-test")
        self.state = {
            "queue_id": "queue-test",
            "last_seq": 4,
            "journal_offset": 80,
            "allocation": {"id": "allocation-1", "state": "running"},
            "nodes": {
                "gpu-0": {
                    "capacity": {"gpu_ids": [0, 1], "cpus": 28, "memory_gb": 256},
                    "free": {"gpu_ids": [1], "cpus": 14, "memory_gb": 128},
                    "assignments": {"running": {"gpu_ids": [0]}},
                }
            },
            "jobs": {
                "queued": {"id": "queued", "name": "queued", "state": "queued"},
                "blocked": {"id": "blocked", "name": "blocked", "state": "blocked"},
                "running": {
                    "id": "running",
                    "name": "running",
                    "state": "running",
                },
            },
        }

    def test_job_view_commands_return_only_the_selected_lane(self) -> None:
        expected = {
            "queue": ("queued", False),
            "running": ("running", True),
            "blocked": ("blocked", False),
        }
        for command, (job_id, has_elapsed) in expected.items():
            with self.subTest(command=command):
                with (
                    mock.patch("scruffy.cli.status", return_value=self.state),
                    mock.patch("scruffy.cli._json") as print_json,
                ):
                    result = main(["--root", str(self.root), command])

                self.assertEqual(0, result)
                payload = print_json.call_args.args[0]
                self.assertEqual([job_id], [job["id"] for job in payload["jobs"]])
                self.assertEqual(has_elapsed, "elapsed_seconds" in payload["jobs"][0])

    def test_resources_command_hides_assignment_details(self) -> None:
        with (
            mock.patch("scruffy.cli.status", return_value=self.state),
            mock.patch("scruffy.cli._json") as print_json,
        ):
            result = main(["--root", str(self.root), "resources"])

        self.assertEqual(0, result)
        payload = print_json.call_args.args[0]
        self.assertEqual(1, payload["totals"]["gpus_free"])
        self.assertEqual(2, payload["nodes"][0]["gpus_total"])
        self.assertNotIn("assignments", str(payload))


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

    def test_missing_inventory_is_reported_without_a_traceback(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch("scruffy.cli.load_inventory", side_effect=FileNotFoundError("gone")),
            contextlib.redirect_stderr(stderr),
        ):
            result = main(
                [
                    "--root",
                    str(self.root),
                    "serve",
                    "--inventory",
                    str(self.workspace / "missing.json"),
                ]
            )

        self.assertEqual(2, result)
        self.assertEqual("scruffy: gone\n", stderr.getvalue())

    def test_automatic_inventory_requires_a_slurm_allocation(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("scruffy.cli.run_controller") as run_controller,
            contextlib.redirect_stderr(stderr),
        ):
            result = main(["--root", str(self.root), "serve"])

        self.assertEqual(2, result)
        self.assertIn("--inventory is required outside", stderr.getvalue())
        run_controller.assert_not_called()

    def test_automatic_inventory_uses_the_slurm_allocation(self) -> None:
        with (
            mock.patch.dict(os.environ, {"SLURM_JOB_ID": "263105"}, clear=True),
            mock.patch(
                "scruffy.cli.discover_slurm_inventory", return_value=self.inventory
            ) as discover,
            mock.patch("scruffy.cli.run_controller") as run_controller,
        ):
            result = main(["--root", str(self.root), "serve"])

        self.assertEqual(0, result)
        discover.assert_called_once_with(
            slurm_job_id="263105",
            gpus_per_node=None,
            cpus_per_node=None,
            memory_gb_per_node=None,
        )
        run_controller.assert_called_once_with(
            root=self.root,
            inventory=tuple(self.inventory.values()),
            launcher="slurm",
            allocation_id="263105",
            slurm_job_id="263105",
            poll_interval=0.2,
            cancel_grace=30,
        )

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


class LogsCliTests(unittest.TestCase):
    def test_follow_reads_once_more_after_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            output_file = root / "jobs" / "job-1" / "stdout.log"
            output_file.parent.mkdir(parents=True)
            output_file.write_bytes(b"")
            calls = 0

            def job_status(_root: Path, _job_id: str) -> dict[str, object]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return {"id": "job-1", "state": "running"}
                output_file.write_bytes(b"final bytes\n")
                return {"id": "job-1", "state": "succeeded"}

            output = mock.Mock(buffer=io.BytesIO())
            output.fileno.return_value = 1
            with (
                mock.patch("scruffy.cli.status", side_effect=job_status),
                mock.patch("scruffy.cli.sys.stdout", output),
            ):
                result = main(
                    [
                        "--root",
                        str(root),
                        "logs",
                        "job-1",
                        "--stream",
                        "stdout",
                        "--tail",
                        "0",
                        "--follow",
                    ]
                )

        self.assertEqual(0, result)
        self.assertEqual(b"final bytes\n", output.buffer.getvalue())


class ObserveFollowCliTests(unittest.TestCase):
    def test_follow_flushes_the_initial_snapshot(self) -> None:
        response = {
            "snapshot": {"allocation": None, "jobs": {}},
            "events": [],
            "next_cursor": "queue-test:0:0",
            "reset": False,
        }
        with (
            mock.patch("scruffy.cli.observe", return_value=response),
            mock.patch("builtins.print", side_effect=KeyboardInterrupt) as print_output,
        ):
            result = main(["--root", "/tmp/scruffy-test", "observe", "--follow"])

        self.assertEqual(130, result)
        print_output.assert_called_once_with(
            '{"kind": "snapshot", "data": {"allocation": null, "jobs": {}}}',
            flush=True,
        )

    def test_follow_prints_a_replacement_snapshot_after_cursor_reset(self) -> None:
        initial = {
            "snapshot": {"jobs": {"old": {}}},
            "events": [],
            "next_cursor": "queue-test:0:1:10",
            "reset": False,
        }
        replacement = {
            "snapshot": {"jobs": {"new": {}}},
            "events": [],
            "next_cursor": "queue-test:1:2:0",
            "reset": True,
        }
        with (
            mock.patch(
                "scruffy.cli.observe",
                side_effect=[initial, replacement, KeyboardInterrupt],
            ),
            mock.patch("builtins.print") as print_output,
        ):
            result = main(["--root", "/tmp/scruffy-test", "observe", "--follow"])

        self.assertEqual(130, result)
        self.assertEqual(
            [
                mock.call(
                    '{"kind": "snapshot", "data": {"jobs": {"old": {}}}}',
                    flush=True,
                ),
                mock.call(
                    '{"kind": "snapshot", "data": {"jobs": {"new": {}}}}',
                    flush=True,
                ),
            ],
            print_output.call_args_list,
        )


class DashboardCliTests(unittest.TestCase):
    def test_dashboard_forwards_remote_mode(self) -> None:
        with mock.patch("scruffy.cli.run_dashboard") as run:
            result = main(
                [
                    "--root",
                    "/shared/queue",
                    "dashboard",
                    "--port",
                    "9000",
                    "--connect-command",
                    "tokyo-ssh",
                    "--remote-command",
                    "/shared/env/bin/scruffy-mcp",
                    "--no-open",
                ]
            )

        self.assertEqual(0, result)
        run.assert_called_once_with(
            "/shared/queue",
            port=9000,
            connect_command="tokyo-ssh",
            remote_command="/shared/env/bin/scruffy-mcp",
            open_browser=False,
        )


if __name__ == "__main__":
    unittest.main()
