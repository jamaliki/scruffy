from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scruffy.controller import _initialize_controller
from scruffy.lifecycle import _finish_job
from scruffy.models import Assignment, NodeInventory, NodeReservation, ResourceRequest
from scruffy.runtime import RunningProcess
from scruffy.slurm import AllocationIncarnation
from scruffy.storage import utc_now, write_state
from scruffy.submissions import job_from_spec

INVENTORY = (NodeInventory("node", (0,), 2, 2),)


def _job(root: Path) -> dict[str, object]:
    return job_from_spec(
        {
            "v": 1,
            "job_id": "terminal",
            "request_id": "request-terminal",
            "name": "terminal",
            "submitted_at": utc_now(),
            "argv": ["true"],
            "cwd": str(root),
            "env": {},
            "resources": ResourceRequest(1, 1, 1, 1).to_dict(),
            "workflow_id": "flow",
            "task_id": "task",
            "needs": [],
            "wait_for": [],
        },
        1,
    )


class TerminalReplayTests(unittest.TestCase):
    def test_result_before_terminal_event_replays_without_retry(self) -> None:
        for returncode, expected in ((0, "succeeded"), (1, "failed")):
            with self.subTest(returncode=returncode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "queue"
                old = AllocationIncarnation("old", 0, INVENTORY)
                new = AllocationIncarnation("new", 0, INVENTORY)
                job = _job(root)
                job["assignment"] = Assignment(
                    "terminal", ResourceRequest(1, 1, 1, 1), (NodeReservation("node", (0,), 1, 1),)
                ).to_dict()
                job["state"] = "running"
                job["launch_token"] = "terminal-launch-token"
                job["allocation_incarnation_sha256"] = old.fingerprint_sha256
                write_state(
                    root,
                    {
                        "v": 1,
                        "queue_id": "queue",
                        "last_seq": 0,
                        "journal_generation": 0,
                        "journal_offset": 0,
                        "allocation": {"id": "old", "state": "running", "incarnation": old.to_dict()},
                        "nodes": {},
                        "gpu_health": None,
                        "jobs": {"terminal": job},
                        "next_queue_order": 1,
                        "archived_jobs": 0,
                        "archived_counts": {},
                        "archived_project_counts": {},
                        "draining": False,
                        "drain_requested": False,
                        "launches_paused": False,
                        "updated_at": utc_now(),
                    },
                )
                controller = _initialize_controller(
                    root=root, inventory=INVENTORY, launcher="slurm", allocation_id="old",
                    slurm_job_id="old", allocation_incarnation=old,
                    poll_interval=0.1, cancel_grace=0, gpu_health_mode="off",
                )
                original = __import__("scruffy.lifecycle", fromlist=["write_result_record"]).write_result_record

                def crash_after_result(
                    *args: object,
                    _original=original,
                    **kwargs: object,
                ) -> object:
                    _original(*args, **kwargs)
                    raise KeyboardInterrupt("crash before terminal event")

                try:
                    with (
                        mock.patch(
                            "scruffy.lifecycle.write_result_record",
                            side_effect=crash_after_result,
                        ),
                        self.assertRaises(KeyboardInterrupt),
                    ):
                        _finish_job(
                            controller,
                            "terminal",
                            RunningProcess(mock.Mock(), None),
                            returncode,
                        )
                finally:
                    controller.journal.close()
                recovered = _initialize_controller(
                    root=root, inventory=INVENTORY, launcher="slurm", allocation_id="new",
                    slurm_job_id="new", allocation_incarnation=new,
                    poll_interval=0.1, cancel_grace=0, gpu_health_mode="off",
                )
                try:
                    self.assertEqual(expected, recovered.state["jobs"]["terminal"]["state"])
                    self.assertEqual(1, len(recovered.state["jobs"]))
                finally:
                    recovered.journal.close()


if __name__ == "__main__":
    unittest.main()
