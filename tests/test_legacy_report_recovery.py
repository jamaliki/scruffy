"""Compatibility is opt-in by campaign, never inferred from a missing token."""

import unittest

from scruffy.controller import _report_capability_valid
from scruffy.storage import _archived_job


class LegacyReportRecoveryTests(unittest.TestCase):
    def test_only_allowlisted_project_can_publish_without_token(self):
        job = {"project_id": "legacy-campaign", "launch_token": "token"}
        for image in (job, _archived_job(job)):
            self.assertTrue(_report_capability_valid(
                image, {"source": {}}, ("legacy-campaign",),
            ))
            self.assertFalse(_report_capability_valid(image, {"source": {}}))
            self.assertFalse(_report_capability_valid(
                image, {"source": {}}, ("another-campaign",),
            ))

    def test_incorrect_token_is_rejected_even_in_legacy_project(self):
        job = {"project_id": "legacy-campaign", "launch_token": "token"}
        for allowlist in ((), ("legacy-campaign",)):
            self.assertFalse(_report_capability_valid(
                job, {"source": {"launch_token": "wrong"}}, allowlist,
            ))
            self.assertTrue(_report_capability_valid(
                job, {"source": {"launch_token": "token"}}, allowlist,
            ))
