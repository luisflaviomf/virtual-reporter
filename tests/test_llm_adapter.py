from __future__ import annotations

import os
import unittest
import urllib.error
from unittest.mock import patch

from ifmt_models.llm_adapter import (
    LLMAdapter,
    LLMConfig,
    LLMGeneration,
    _ollama_payload,
    _parse_ollama_response,
)


class OllamaPayloadTests(unittest.TestCase):
    def test_serializes_gpu_options_without_false_mtp(self) -> None:
        env = {
            "VIRTUAL_REPORTER_OLLAMA_NUM_GPU": "999",
            "VIRTUAL_REPORTER_OLLAMA_NUM_CTX": "4096",
            "VIRTUAL_REPORTER_OLLAMA_NUM_THREAD": "14",
            "VIRTUAL_REPORTER_OLLAMA_NUM_PREDICT": "512",
            "VIRTUAL_REPORTER_OLLAMA_FORMAT": "json",
            "VIRTUAL_REPORTER_OLLAMA_THINK": "false",
        }
        config = LLMConfig(
            "ollama", "gemma4:e4b", True, "http://127.0.0.1:11435"
        )

        with patch.dict(os.environ, env, clear=False):
            payload = _ollama_payload("prompt", config)

        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertEqual(payload["options"]["num_gpu"], 999)
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertEqual(payload["options"]["num_thread"], 14)
        self.assertEqual(payload["options"]["num_predict"], 512)
        self.assertEqual(payload["format"], "json")
        self.assertFalse(payload["think"])
        self.assertEqual(payload["keep_alive"], -1)
        self.assertNotIn("draft_num_predict", payload["options"])

    def test_rejects_negative_retry_configuration(self) -> None:
        config = LLMConfig(
            "ollama", "gemma4:e4b", True, "http://127.0.0.1:11435"
        )
        with patch.dict(
            os.environ, {"VIRTUAL_REPORTER_LLM_RETRIES": "-1"}, clear=False
        ):
            with self.assertRaisesRegex(
                RuntimeError, "VIRTUAL_REPORTER_LLM_RETRIES must be non-negative"
            ):
                LLMAdapter(config).generate_with_metrics("prompt")


class OllamaGenerationTests(unittest.TestCase):
    def test_parses_native_duration_and_token_metrics(self) -> None:
        response = {
            "model": "gemma4:e4b",
            "message": {"role": "assistant", "content": '{"ok":true}'},
            "done": True,
            "done_reason": "stop",
            "total_duration": 10_000_000,
            "load_duration": 1_000_000,
            "prompt_eval_count": 120,
            "prompt_eval_duration": 2_000_000,
            "eval_count": 40,
            "eval_duration": 7_000_000,
        }

        result = _parse_ollama_response(response)

        self.assertEqual(result.text, '{"ok":true}')
        self.assertEqual(result.metrics["provider"], "ollama")
        self.assertEqual(result.metrics["model"], "gemma4:e4b")
        self.assertEqual(result.metrics["prompt_eval_count"], 120)
        self.assertEqual(result.metrics["eval_count"], 40)
        self.assertEqual(result.metrics["total_duration_ns"], 10_000_000)
        self.assertEqual(result.metrics["done_reason"], "stop")

    def test_retries_transport_failures_and_preserves_string_api(self) -> None:
        config = LLMConfig(
            "ollama", "gemma4:e4b", True, "http://127.0.0.1:11435"
        )
        success = LLMGeneration("final text", {"eval_count": 3})
        failures = [
            urllib.error.URLError("first"),
            urllib.error.URLError("second"),
            success,
        ]

        with patch.dict(
            os.environ, {"VIRTUAL_REPORTER_LLM_RETRIES": "2"}, clear=False
        ), patch(
            "ifmt_models.llm_adapter._call_ollama_chat_api",
            side_effect=failures,
        ) as request, patch("ifmt_models.llm_adapter.time.sleep") as sleep:
            result = LLMAdapter(config).generate_with_metrics("prompt")

        self.assertEqual(result, success)
        self.assertEqual(request.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

        with patch.dict(
            os.environ, {"VIRTUAL_REPORTER_LLM_RETRIES": "0"}, clear=False
        ), patch(
            "ifmt_models.llm_adapter._call_ollama_chat_api",
            return_value=success,
        ):
            self.assertEqual(LLMAdapter(config).generate("prompt"), "final text")


if __name__ == "__main__":
    unittest.main()
