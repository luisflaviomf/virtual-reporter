from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ifmt_models.evaluation_checkpoint import (
    CheckpointCorruptionError,
    CheckpointMismatchError,
    CheckpointStore,
    DuplicateCheckpointError,
    RunFingerprint,
)


def fingerprint(runtime_digest: str = "runtime-a") -> RunFingerprint:
    return RunFingerprint(
        seed=42,
        case_count=300,
        sampling_version="stratified-v1",
        masking_version="balanced-mask-v1",
        prompt_version="prompts-v1",
        model="gemma4:e4b",
        runtime_digest=runtime_digest,
    )


class EvaluationCheckpointTests(unittest.TestCase):
    def test_saves_and_reloads_completed_cases_in_case_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = CheckpointStore.create_or_open(root, fingerprint())
            store.save_case(
                {"case": {"case_id": "case_0002"}, "outputs": [1], "scores": [2]}
            )
            store.save_case(
                {"case": {"case_id": "case_0001"}, "outputs": [3], "scores": [4]}
            )

            reopened = CheckpointStore.create_or_open(root, fingerprint())

            self.assertEqual(
                [item["case"]["case_id"] for item in reopened.load_records()],
                ["case_0001", "case_0002"],
            )
            self.assertEqual(
                reopened.completed_case_ids(), {"case_0001", "case_0002"}
            )
            self.assertFalse(list((root / "checkpoint" / "cases").glob("*.tmp")))

    def test_rejects_incompatible_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            CheckpointStore.create_or_open(root, fingerprint("runtime-a"))

            with self.assertRaisesRegex(
                CheckpointMismatchError, "checkpoint fingerprint .* does not match"
            ):
                CheckpointStore.create_or_open(root, fingerprint("runtime-b"))

    def test_duplicate_identical_record_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = CheckpointStore.create_or_open(Path(temp), fingerprint())
            record = {
                "case": {"case_id": "case_0001"},
                "outputs": [{"configuration": "fixed_form_baseline"}],
                "scores": [{}],
            }

            store.save_case(record)
            store.save_case(record)

            self.assertEqual(len(store.load_records()), 1)

    def test_rejects_divergent_duplicate_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = CheckpointStore.create_or_open(Path(temp), fingerprint())
            store.save_case(
                {"case": {"case_id": "case_0001"}, "outputs": [1], "scores": [2]}
            )

            with self.assertRaisesRegex(DuplicateCheckpointError, "case_0001"):
                store.save_case(
                    {
                        "case": {"case_id": "case_0001"},
                        "outputs": ["changed"],
                        "scores": [2],
                    }
                )

    def test_reports_corrupt_case_file_instead_of_skipping_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = CheckpointStore.create_or_open(root, fingerprint())
            corrupt = root / "checkpoint" / "cases" / "case_0001.json"
            corrupt.write_text('{"case":', encoding="utf-8")

            with self.assertRaisesRegex(
                CheckpointCorruptionError, "case_0001.json"
            ):
                store.load_records()


if __name__ == "__main__":
    unittest.main()
