from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from ifmt_models.baseline_comparison import _extract_fields_from_pitch
from ifmt_models.evaluation_checkpoint import RunFingerprint
from ifmt_models.expanded_evaluation import (
    ExpandedEvaluationConfig,
    execute_case,
    finalize_expanded_run,
    run_cases_with_resume,
    validate_run_paths,
)
from ifmt_models.llm_adapter import LLMConfig, LLMGeneration


VALID_LLM_JSON = json.dumps(
    {
        "confirmed_information": {"what": "Projeto X"},
        "missing_information": ["when"],
        "suggested_questions": [
            {"field": "when", "question": "Qual e a data confirmada?"}
        ],
        "factual_claims": ["Projeto X"],
    }
)

MINIMAL_CASE = {
    "case_id": "case_0001",
    "article_id": "article-1",
    "original_title": "Projeto X",
    "original_url": "",
    "original_unit": "",
    "original_date": "2026-01-02",
    "original_predicted_label": "event",
    "macro_group": "event",
    "requested_mask_depth": 1,
    "mask_depth": 1,
    "incomplete_pitch": (
        "Fato informado: Projeto X. "
        "Ha informacoes ainda pendentes de confirmacao."
    ),
    "removed_fields": ["when"],
    "available_fields_in_pitch": {"what": "Projeto X"},
    "ground_truth": {
        "what": "Projeto X",
        "who": "",
        "when": "2026-01-02",
        "where": "",
        "why": "",
        "how": "",
        "source": "",
        "unit": "",
    },
}


class SequencedAdapter:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.prompts: list[str] = []
        self.config = LLMConfig(
            "ollama", "gemma4:e4b", True, "http://127.0.0.1:11435"
        )

    @property
    def is_available(self) -> bool:
        return True

    def generate_with_metrics(self, prompt: str) -> LLMGeneration:
        self.prompts.append(prompt)
        response = self.responses[self.calls]
        self.calls += 1
        return LLMGeneration(
            response,
            {"provider": "ollama", "eval_count": self.calls},
        )


def run_fingerprint() -> RunFingerprint:
    return RunFingerprint(
        42,
        2,
        "stratified-v1",
        "balanced-mask-v1",
        "baseline-prompts-v1",
        "gemma4:e4b",
        "runtime-test",
    )


def completed_record(case_id: str) -> dict:
    case = dict(MINIMAL_CASE)
    case["case_id"] = case_id
    configurations = [
        "fixed_form_baseline",
        "direct_llm_baseline",
        "virtual_reporter",
        "virtual_reporter_llm_rag",
    ]
    outputs = [
        {
            "case_id": case_id,
            "configuration": configuration,
            "status": "ok",
            "parse_error": False,
            "input_pitch": case["incomplete_pitch"],
            "generated_questions": [],
            "missing_fields_detected": ["when"],
            "retrieved_documents": [],
            "retrieved_article_ids": [],
            "briefing_text": "Projeto X",
            "claims": ["Projeto X"],
            "evidence_texts": [case["incomplete_pitch"]],
            "generation_metrics": {},
        }
        for configuration in configurations
    ]
    scores = [
        {
            "case_id": case_id,
            "configuration": configuration,
            "completeness": 0.5,
            "factual_precision": 1.0,
            "hallucination_rate": 0.0,
            "question_relevance": 0.0,
            "question_coverage": 0.0,
            "question_redundancy_rate": 0.0,
            "missing_fields_detected_ratio": 1.0,
            "claims_total": 1,
            "claims_supported": 1,
            "questions_total": 0,
            "removed_fields": "when",
            "missing_fields_detected": "when",
        }
        for configuration in configurations
    ]
    return {"case": case, "outputs": outputs, "scores": scores, "timings": {}}


class ExpandedEvaluationResumeTests(unittest.TestCase):
    def test_resume_skips_completed_cases_without_duplicate_calls(self) -> None:
        cases = [{"case_id": "case_0001"}, {"case_id": "case_0002"}]
        calls: list[str] = []

        def executor(case: dict) -> dict:
            calls.append(case["case_id"])
            return completed_record(case["case_id"])

        with tempfile.TemporaryDirectory() as temp:
            config = ExpandedEvaluationConfig(
                output_dir=Path(temp), case_count=2, seed=42, resume=True
            )
            run_cases_with_resume(cases, config, executor, run_fingerprint())
            run_cases_with_resume(cases, config, executor, run_fingerprint())

        self.assertEqual(calls, ["case_0001", "case_0002"])

    def test_case_workers_two_executes_two_cases_concurrently(self) -> None:
        cases = [{"case_id": "case_0001"}, {"case_id": "case_0002"}]
        barrier = threading.Barrier(2, timeout=1.0)

        def executor(case: dict) -> dict:
            barrier.wait()
            return completed_record(case["case_id"])

        with tempfile.TemporaryDirectory() as temp:
            config = ExpandedEvaluationConfig(
                output_dir=Path(temp), case_count=2, seed=42, resume=True
            )
            with mock.patch.dict(
                os.environ, {"VIRTUAL_REPORTER_CASE_WORKERS": "2"}, clear=False
            ):
                records = run_cases_with_resume(
                    cases, config, executor, run_fingerprint()
                )

        self.assertEqual(len(records), 2)

    def test_case_cooldown_is_applied_after_checkpoint(self) -> None:
        cases = [{"case_id": "case_0001"}]

        with tempfile.TemporaryDirectory() as temp:
            config = ExpandedEvaluationConfig(
                output_dir=Path(temp), case_count=1, seed=42, resume=True
            )
            with mock.patch.dict(
                os.environ,
                {
                    "VIRTUAL_REPORTER_CASE_WORKERS": "1",
                    "VIRTUAL_REPORTER_CASE_COOLDOWN_SECONDS": "2.5",
                },
                clear=False,
            ), mock.patch("ifmt_models.expanded_evaluation.time.sleep") as sleep:
                run_cases_with_resume(
                    cases,
                    config,
                    lambda case: completed_record(case["case_id"]),
                    run_fingerprint(),
                )

        sleep.assert_called_once_with(2.5)

    def test_labeled_pitch_extractor_reads_only_confirmed_values(self) -> None:
        pitch = (
            "Fato informado: Projeto X. Data confirmada: 2026-01-02. "
            "Ha informacoes ainda pendentes de confirmacao."
        )

        fields = _extract_fields_from_pitch(pitch)

        self.assertEqual(fields, {"what": "Projeto X", "when": "2026-01-02"})

    def test_execute_case_retries_invalid_llm_json_before_checkpoint(self) -> None:
        adapter = SequencedAdapter(
            ["not-json", VALID_LLM_JSON, "not-json", VALID_LLM_JSON]
        )

        record = execute_case(
            MINIMAL_CASE,
            {"docs": [], "idf": {}},
            adapter,
            parse_retries=1,
        )

        self.assertEqual(len(record["outputs"]), 4)
        self.assertFalse(any(output["parse_error"] for output in record["outputs"]))
        self.assertEqual(adapter.calls, 4)
        llm_outputs = [
            output
            for output in record["outputs"]
            if output["configuration"]
            in {"direct_llm_baseline", "virtual_reporter_llm_rag"}
        ]
        self.assertTrue(
            all(output["generation_metrics"]["provider"] == "ollama" for output in llm_outputs)
        )
        self.assertNotIn("RETRY_COMPACT_JSON", adapter.prompts[0])
        self.assertIn("RETRY_COMPACT_JSON", adapter.prompts[1])
        self.assertNotIn("RETRY_COMPACT_JSON", adapter.prompts[2])
        self.assertIn("RETRY_COMPACT_JSON", adapter.prompts[3])

    def test_execute_case_applies_cooldown_between_llm_stages(self) -> None:
        adapter = SequencedAdapter([VALID_LLM_JSON, VALID_LLM_JSON])

        with mock.patch.dict(
            os.environ,
            {"VIRTUAL_REPORTER_LLM_COOLDOWN_SECONDS": "4"},
            clear=False,
        ), mock.patch("ifmt_models.expanded_evaluation.time.sleep") as sleep:
            execute_case(
                MINIMAL_CASE,
                {"docs": [], "idf": {}},
                adapter,
                parse_retries=0,
            )

        sleep.assert_called_once_with(4.0)

    def test_refuses_to_use_pilot_as_expanded_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            pilot = Path(temp) / "pilot"
            pilot.mkdir()
            config = ExpandedEvaluationConfig(output_dir=pilot, pilot_dir=pilot)

            with self.assertRaisesRegex(ValueError, "must differ from pilot_dir"):
                validate_run_paths(config)


class ExpandedEvaluationArtifactTests(unittest.TestCase):
    def test_finalization_is_idempotent_and_deterministically_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "expanded"
            pilot = Path(temp) / "pilot"
            pilot.mkdir()
            config = ExpandedEvaluationConfig(
                output_dir=output,
                pilot_dir=pilot,
                case_count=2,
                seed=42,
            )
            records = [completed_record("case_0002"), completed_record("case_0001")]

            finalize_expanded_run(config, records)
            first = self._hashes(output)
            finalize_expanded_run(config, records)
            second = self._hashes(output)

            self.assertEqual(first, second)
            case_lines = (output / "test_cases.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(json.loads(case_lines[0])["case_id"], "case_0001")
            self.assertEqual(json.loads(case_lines[1])["case_id"], "case_0002")
            self.assertTrue((output / "sampling_summary.json").exists())
            self.assertTrue((output / "baseline_comparison_article.csv").exists())

    def test_metadata_records_non_secret_runtime_and_gpu_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "expanded"
            pilot = Path(temp) / "pilot"
            pilot.mkdir()
            config = ExpandedEvaluationConfig(
                output_dir=output,
                pilot_dir=pilot,
                case_count=1,
                seed=42,
            )
            runtime = {
                "OLLAMA_NUM_PARALLEL": "2",
                "VIRTUAL_REPORTER_GPU_NAME": "NVIDIA RTX Test",
                "VIRTUAL_REPORTER_GPU_TOTAL_MIB": "8188",
            }
            with mock.patch.dict(os.environ, runtime, clear=False):
                finalize_expanded_run(config, [completed_record("case_0001")])

            metadata = json.loads(
                (output / "evaluation_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["runtime_options"]["OLLAMA_NUM_PARALLEL"], "2")
            self.assertEqual(
                metadata["runtime_options"]["VIRTUAL_REPORTER_GPU_NAME"],
                "NVIDIA RTX Test",
            )
            self.assertEqual(
                metadata["runtime_options"]["VIRTUAL_REPORTER_GPU_TOTAL_MIB"],
                "8188",
            )

    @staticmethod
    def _hashes(root: Path) -> dict[str, str]:
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }


if __name__ == "__main__":
    unittest.main()
