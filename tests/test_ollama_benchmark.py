from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ifmt_models.llm_adapter import LLMGeneration
from ifmt_models.ollama_benchmark import (
    aggregate_benchmark,
    load_gpu_samples,
    select_parallelism,
)


class BenchmarkSelectionTests(unittest.TestCase):
    def test_selects_two_only_for_safe_fifteen_percent_gain(self) -> None:
        results = {
            "1": {
                "requests_per_minute": 1.0,
                "peak_vram_mib": 7600,
                "failures": 0,
                "parse_failures": 0,
            },
            "2": {
                "requests_per_minute": 1.16,
                "peak_vram_mib": 7750,
                "failures": 0,
                "parse_failures": 0,
            },
        }

        self.assertEqual(select_parallelism(results, max_vram_mib=7800), 2)

    def test_falls_back_to_one_on_unsafe_or_small_gain(self) -> None:
        unsafe_candidates = (
            {
                "requests_per_minute": 1.3,
                "peak_vram_mib": 7900,
                "failures": 0,
                "parse_failures": 0,
            },
            {
                "requests_per_minute": 1.3,
                "peak_vram_mib": 7700,
                "failures": 1,
                "parse_failures": 0,
            },
            {
                "requests_per_minute": 1.3,
                "peak_vram_mib": 7700,
                "failures": 0,
                "parse_failures": 1,
            },
            {
                "requests_per_minute": 1.14,
                "peak_vram_mib": 7700,
                "failures": 0,
                "parse_failures": 0,
            },
        )

        for candidate in unsafe_candidates:
            with self.subTest(candidate=candidate):
                results = {
                    "1": {"requests_per_minute": 1.0},
                    "2": candidate,
                }
                self.assertEqual(select_parallelism(results, 7800), 1)


class BenchmarkAggregationTests(unittest.TestCase):
    def test_aggregates_native_metrics_and_gpu_samples(self) -> None:
        generations = [
            LLMGeneration(
                "{}",
                {
                    "prompt_eval_count": 100,
                    "eval_count": 30,
                    "total_duration_ns": 10_000_000_000,
                },
            ),
            LLMGeneration(
                "{}",
                {
                    "prompt_eval_count": 200,
                    "eval_count": 50,
                    "total_duration_ns": 20_000_000_000,
                },
            ),
        ]
        gpu_samples = [
            {"memory_used_mib": 7400.0, "gpu_utilization_percent": 88.0, "temperature_c": 70.0},
            {"memory_used_mib": 7750.0, "gpu_utilization_percent": 97.0, "temperature_c": 74.0},
        ]

        result = aggregate_benchmark(
            generations=generations,
            wall_seconds=60.0,
            gpu_samples=gpu_samples,
            failures=[],
            parse_failures=0,
            parallelism=2,
        )

        self.assertEqual(result["parallelism"], 2)
        self.assertEqual(result["requests_per_minute"], 2.0)
        self.assertEqual(result["prompt_tokens"], 300)
        self.assertEqual(result["generated_tokens"], 80)
        self.assertEqual(result["peak_vram_mib"], 7750.0)
        self.assertEqual(result["peak_gpu_utilization_percent"], 97.0)
        self.assertEqual(result["max_temperature_c"], 74.0)
        self.assertEqual(result["failures"], 0)

    def test_loads_nvidia_csv_with_units(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gpu.csv"
            path.write_text(
                "timestamp, utilization.gpu [%], memory.used [MiB], memory.total [MiB], temperature.gpu, power.draw [W]\n"
                "2026/08/17 10:00:00.000, 95 %, 7650 MiB, 8188 MiB, 73, 81.25 W\n"
                "2026/08/17 10:00:02.000, 97 %, 7750 MiB, 8188 MiB, 74, 82.00 W\n",
                encoding="utf-8",
            )

            samples = load_gpu_samples(path)

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["memory_used_mib"], 7650.0)
        self.assertEqual(samples[1]["gpu_utilization_percent"], 97.0)
        self.assertEqual(samples[1]["power_w"], 82.0)


if __name__ == "__main__":
    unittest.main()
