from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scruffy.client import status, summary
from scruffy.controller import _initialize_controller
from scruffy.models import NodeInventory
from scruffy.storage import read_events

INVENTORY = (NodeInventory("node", (0,), 2, 2),)


class ReleaseAttestationTests(unittest.TestCase):
    def test_release_is_authoritative_across_restart_and_journal_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            first = _initialize_controller(
                root=root,
                inventory=INVENTORY,
                launcher="local",
                allocation_id="local",
                slurm_job_id=None,
                controller_release="release-one",
                poll_interval=0.1,
                cancel_grace=0,
                gpu_health_mode="off",
            )
            first.journal.close()
            root.joinpath("state.json").unlink()

            second = _initialize_controller(
                root=root,
                inventory=INVENTORY,
                launcher="local",
                allocation_id="local",
                slurm_job_id=None,
                controller_release="release-two",
                poll_interval=0.1,
                cancel_grace=0,
                gpu_health_mode="off",
            )
            try:
                self.assertEqual("release-two", status(root)["allocation"]["controller_release"])
                self.assertEqual("release-two", summary(root)["allocation"]["controller_release"])
                lifecycle = [
                    event
                    for event in read_events(root)
                    if event["kind"] == "allocation.started"
                ]
                self.assertEqual(
                    ["release-one", "release-two"],
                    [event["data"]["controller_release"] for event in lifecycle],
                )
            finally:
                second.journal.close()

    def test_missing_release_is_unknown_and_never_a_fabricated_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "queue"
            controller = _initialize_controller(
                root=root,
                inventory=INVENTORY,
                launcher="local",
                allocation_id="local",
                slurm_job_id=None,
                poll_interval=0.1,
                cancel_grace=0,
                gpu_health_mode="off",
            )
            try:
                self.assertEqual("unknown", status(root)["allocation"]["controller_release"])
            finally:
                controller.journal.close()


if __name__ == "__main__":
    unittest.main()
