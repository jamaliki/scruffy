"""Commands name jobs by job ID, request ID, workflow task or job name."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any
from unittest import mock

from scruffy.cli import main
from scruffy.client import resolve_job_reference, submit_job
from scruffy.models import ResourceRequest
from scruffy.storage import archive_terminal_job, create_job_id, queue_id, write_state

REQUEST = ResourceRequest(1, 1, 1, 1)


def image(
    job_id: str,
    state: str,
    order: int,
    *,
    project_id: str,
    name: str | None = None,
    request_id: str | None = None,
    workflow_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    job: dict[str, Any] = {
        "id": job_id,
        "name": name or job_id,
        "project_id": project_id,
        "state": state,
        "submitted_at": "2026-11-25T00:00:00.000+00:00",
        "queue_order": order,
        "argv": ["true"],
        "cwd": "/shared",
        "request": REQUEST.to_dict(),
        "assignment": None,
    }
    if request_id is not None:
        job["request_id"] = request_id
    if state not in {"queued", "blocked"}:
        job["started_at"] = "2026-11-25T00:01:00.000+00:00"
    if state in {"succeeded", "failed", "cancelled"}:
        job["finished_at"] = "2026-11-25T00:02:00.000+00:00"
    if workflow_id is not None:
        job.update(
            {"workflow_id": workflow_id, "task_id": task_id, "needs": [], "wait_for": []}
        )
    return job


class JobReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "queue"
        self.first = create_job_id("flow/train", project_id="science")
        self.solo = create_job_id("solo-req", project_id="science")
        archive_terminal_job(
            self.root,
            image(
                self.first, "failed", 1, project_id="science", name="train",
                request_id="flow/train", workflow_id="flow", task_id="train",
            ),
        )
        archive_terminal_job(
            self.root,
            image(
                "job-prep", "succeeded", 2, project_id="science",
                workflow_id="flow", task_id="prep",
            ),
        )
        jobs = {
            "job-retry": image(
                "job-retry", "running", 5, project_id="science", name="train",
                request_id="scruffy-recovery-x", workflow_id="flow", task_id="train",
            ),
            self.solo: image(
                self.solo, "queued", 6, project_id="science", name="solo",
                request_id="solo-req",
            ),
            "job-nested": image(
                "job-nested", "queued", 7, project_id="science",
                workflow_id="group/flow", task_id="eval",
            ),
            "job-dup-a": image("job-dup-a", "queued", 8, project_id="alpha", name="dup"),
            "job-dup-b": image("job-dup-b", "queued", 9, project_id="beta", name="dup"),
            "job-w-a": image(
                "job-w-a", "queued", 10, project_id="alpha", workflow_id="w", task_id="t"
            ),
            "job-w-b": image(
                "job-w-b", "queued", 11, project_id="beta", workflow_id="w", task_id="t"
            ),
        }
        write_state(
            self.root,
            {
                "v": 1,
                "queue_id": queue_id(self.root),
                "last_seq": 0,
                "journal_generation": 0,
                "journal_offset": 0,
                "allocation": None,
                "nodes": {},
                "jobs": jobs,
                "next_queue_order": 12,
                "archived_jobs": 2,
                "archived_counts": {},
                "archived_project_counts": {},
                "draining": False,
            },
        )
        self.pending = submit_job(
            self.root, argv=["true"], name="fresh", cwd=self.root, environment={},
            request=REQUEST, request_id="fresh-req", project_id="other",
        )["job_id"]

    def resolve(self, reference: str, project_id: str | None = None) -> str:
        return resolve_job_reference(self.root, reference, project_id=project_id)

    def test_each_kind_of_reference_names_one_job(self) -> None:
        for reference, project, expected in (
            ("job-retry", None, "job-retry"),
            (self.first, None, self.first),
            (self.pending, None, self.pending),
            # A workflow task or its first request ID follows the retry.
            ("flow/train", None, "job-retry"),
            ("flow/train", "science", "job-retry"),
            ("flow/prep", None, "job-prep"),
            ("group/flow/eval", None, "job-nested"),
            ("solo-req", None, self.solo),
            ("fresh-req", "other", self.pending),
            ("solo", None, self.solo),
            ("fresh", None, self.pending),
            ("dup", "alpha", "job-dup-a"),
            ("w/t", "beta", "job-w-b"),
            # No match is returned unchanged for the command to report.
            ("flow/train", "other", "flow/train"),
            ("job-never", None, "job-never"),
        ):
            with self.subTest(reference=reference, project=project):
                self.assertEqual(expected, self.resolve(reference, project))

    def test_two_matches_are_refused_with_their_ids(self) -> None:
        for reference, first, second in (
            ("dup", "job-dup-a", "job-dup-b"),
            ("w/t", "job-w-a", "job-w-b"),
        ):
            with self.subTest(reference=reference), self.assertRaises(ValueError) as caught:
                self.resolve(reference)
            self.assertIn(first, str(caught.exception))
            self.assertIn(second, str(caught.exception))
        with self.assertRaisesRegex(ValueError, "trimmed"):
            self.resolve(" solo")

    def test_commands_resolve_references(self) -> None:
        root = str(self.root)
        with mock.patch("scruffy.cli._json") as printed:
            self.assertEqual(0, main(["--root", root, "status", "solo"]))
        self.assertEqual(self.solo, printed.call_args.args[0]["id"])
        with mock.patch("scruffy.cli._json") as printed:
            self.assertEqual(
                0, main(["--root", root, "status", "--project", "science", "flow/train"])
            )
        self.assertEqual("job-retry", printed.call_args.args[0]["id"])
        with (
            mock.patch("scruffy.cli.cancel_job", return_value={}) as cancel,
            mock.patch("scruffy.cli._json"),
        ):
            self.assertEqual(0, main(["--root", root, "cancel", "flow/train"]))
        self.assertEqual("job-retry", cancel.call_args.args[1])
        with (
            mock.patch(
                "scruffy.cli.wait_for_job", return_value={"state": "succeeded"}
            ) as wait,
            mock.patch("scruffy.cli._json"),
        ):
            self.assertEqual(0, main(["--root", root, "wait", "solo-req"]))
        self.assertEqual(self.solo, wait.call_args.args[1])
        with mock.patch("scruffy.cli.tail_window", return_value=(b"", 0)) as tail:
            self.assertEqual(
                0, main(["--root", root, "logs", "--project", "alpha", "dup"])
            )
        self.assertIn("job-dup-a", str(tail.call_args.args[0]))
        errors = io.StringIO()
        with redirect_stderr(errors):
            self.assertEqual(2, main(["--root", root, "cancel", "dup"]))
        self.assertIn("matches 2 jobs", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
