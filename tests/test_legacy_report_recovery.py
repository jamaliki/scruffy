"""Live upgrades must not discard reports from pre-capability Slurm workers."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scruffy.controller import _reattach_slurm_jobs, _report_capability_valid
from scruffy.storage import _archived_job


class LegacyReportRecoveryTests(unittest.TestCase):
    def recover(self, **extra):
        job = {
            "id": "old-trainer", "state": "running", "launch_token": "token",
            "runtime_placement_contract": 1, **extra,
        }
        with tempfile.TemporaryDirectory() as directory:
            controller = SimpleNamespace(root=Path(directory), running={})
            _reattach_slurm_jobs(controller, [job])
            self.assertIn(job["id"], controller.running)
        return job

    def test_pre_binding_worker_keeps_legacy_trust_after_recovery_and_archive(self):
        job = self.recover()
        for image in (job, _archived_job(job)):
            self.assertTrue(_report_capability_valid(image, {"source": {}}))
            self.assertTrue(_report_capability_valid(image, {
                "source": {"launch_token": "token"},
            }))
            self.assertFalse(_report_capability_valid(image, {
                "source": {"launch_token": "wrong"},
            }))

    def test_modern_recovered_workers_still_require_their_capability(self):
        for binding in ("count", "exact"):
            job = self.recover(gpu_binding=binding)
            for image in (job, _archived_job(job)):
                self.assertNotIn("legacy_report_source", image)
                self.assertFalse(_report_capability_valid(image, {"source": {}}))
                self.assertTrue(_report_capability_valid(image, {
                    "source": {"launch_token": "token"},
                }))

    def test_unmigrated_or_unknown_workers_do_not_get_legacy_trust(self):
        for job in (
            {"launch_token": "token"},
            self.recover(runtime_placement_contract=None),
        ):
            self.assertFalse(_report_capability_valid(job, {"source": {}}))
