from __future__ import annotations

import unittest

from scruffy.protocol import EVENT_KINDS, MAX_EVENT_BYTES, ProtocolError, validate_event


def event(**changes: object) -> dict[str, object]:
    document: dict[str, object] = {
        "v": 1,
        "event_id": "trainer/step-12",
        "job_id": "job-abc123",
        "occurred_at": "2026-08-03T10:11:12.123+00:00",
        "kind": "workload.progress",
        "source": {"name": "koochak", "node": "gpu-3"},
        "data": {"step": 12, "loss": 0.25, "tags": ["train", None]},
    }
    document.update(changes)
    return document


class ProtocolTests(unittest.TestCase):
    def test_every_public_kind_has_one_strict_json_envelope(self) -> None:
        for kind in EVENT_KINDS:
            with self.subTest(kind=kind):
                original = event(kind=kind)
                validated = validate_event(original)
                self.assertEqual(original, validated)
                self.assertIsNot(original, validated)
                self.assertIsNot(original["data"], validated["data"])

    def test_envelope_rejects_missing_extra_and_invalid_version_keys(self) -> None:
        invalid = [
            {key: value for key, value in event().items() if key != "data"},
            {**event(), "state": "succeeded"},
            {**event(), "v": True},
            {**event(), 7: "not-a-string-key"},
        ]
        for document in invalid:
            with self.subTest(document=document):
                with self.assertRaises(ProtocolError):
                    validate_event(document)

    def test_ids_timestamp_kind_and_source_are_constrained(self) -> None:
        invalid = [
            event(event_id=" event-1"),
            event(job_id="../other-job"),
            event(occurred_at="2026-08-03T10:11:12"),
            event(kind="job.succeeded"),
            event(source="koochak"),
            event(source={"name": 7}),
            event(source={"x" * 65: "value"}),
        ]
        for document in invalid:
            with self.subTest(document=document):
                with self.assertRaises(ProtocolError):
                    validate_event(document)

    def test_data_accepts_json_only(self) -> None:
        invalid_values = [float("nan"), float("inf"), (1, 2), {1: "value"}]
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ProtocolError):
                    validate_event(event(data={"value": value}))

    def test_event_size_includes_utf8_safe_on_disk_encoding(self) -> None:
        validated = validate_event(event(data={"text": "small"}))
        self.assertEqual("small", validated["data"]["text"])
        with self.assertRaisesRegex(ProtocolError, str(MAX_EVENT_BYTES)):
            validate_event(event(data={"text": "x" * MAX_EVENT_BYTES}))


if __name__ == "__main__":
    unittest.main()
