from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .baseline_comparison import (
    _adaptive_questions,
    _detect_missing_fields,
    _direct_llm_prompt,
    _extract_fields_from_pitch,
    _parse_direct_llm_response,
    _predict_type,
    _questionable_missing_fields,
    _retrieve,
    _virtual_reporter_llm_rag_prompt,
)
from .expanded_evaluation import ExpandedEvaluationConfig, prepare_expanded_evaluation
from .llm_adapter import LLMAdapter, LLMConfig, LLMGeneration


def select_parallelism(
    results: dict[str, dict[str, Any]], max_vram_mib: int = 7800
) -> int:
    one = results["1"]
    two = results["2"]
    safe = (
        int(two.get("failures", 0)) == 0
        and int(two.get("parse_failures", 0)) == 0
        and float(two.get("peak_vram_mib", 0)) <= max_vram_mib
    )
    baseline_rate = max(float(one.get("requests_per_minute", 0)), 1e-9)
    gain = float(two.get("requests_per_minute", 0)) / baseline_rate
    return 2 if safe and gain >= 1.15 else 1


def aggregate_benchmark(
    generations: list[LLMGeneration],
    wall_seconds: float,
    gpu_samples: list[dict[str, float]],
    failures: list[str],
    parse_failures: int,
    parallelism: int,
) -> dict[str, Any]:
    successful = len(generations)
    request_rate = successful * 60 / wall_seconds if wall_seconds > 0 else 0.0
    native_seconds = [
        float(generation.metrics.get("total_duration_ns", 0)) / 1_000_000_000
        for generation in generations
        if generation.metrics.get("total_duration_ns")
    ]
    return {
        "parallelism": parallelism,
        "successful_requests": successful,
        "failures": len(failures),
        "failure_messages": failures,
        "parse_failures": parse_failures,
        "wall_seconds": round(wall_seconds, 4),
        "requests_per_minute": round(request_rate, 4),
        "prompt_tokens": sum(
            int(generation.metrics.get("prompt_eval_count", 0))
            for generation in generations
        ),
        "generated_tokens": sum(
            int(generation.metrics.get("eval_count", 0))
            for generation in generations
        ),
        "mean_native_request_seconds": round(
            sum(native_seconds) / len(native_seconds), 4
        )
        if native_seconds
        else 0.0,
        "p95_native_request_seconds": round(_percentile(native_seconds, 0.95), 4),
        "peak_vram_mib": _maximum(gpu_samples, "memory_used_mib"),
        "peak_gpu_utilization_percent": _maximum(
            gpu_samples, "gpu_utilization_percent"
        ),
        "max_temperature_c": _maximum(gpu_samples, "temperature_c"),
        "max_power_w": _maximum(gpu_samples, "power_w"),
    }


def load_gpu_samples(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    samples: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 6:
                continue
            samples.append(
                {
                    "gpu_utilization_percent": _number(row[1]),
                    "memory_used_mib": _number(row[2]),
                    "memory_total_mib": _number(row[3]),
                    "temperature_c": _number(row[4]),
                    "power_w": _number(row[5]),
                }
            )
    return samples


def run_benchmark(
    prompts: list[str],
    adapter: LLMAdapter,
    parallelism: int,
    gpu_samples_path: Path,
) -> dict[str, Any]:
    if parallelism not in {1, 2}:
        raise ValueError("parallelism must be 1 or 2")
    generations: list[LLMGeneration] = []
    failures: list[str] = []
    parse_failures = 0
    request_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    def generate(index: int, prompt: str) -> tuple[int, LLMGeneration, bool]:
        generation = adapter.generate_with_metrics(prompt)
        parsed = _parse_direct_llm_response(generation.text)
        return index, generation, bool(parsed["parse_error"])

    if parallelism == 1:
        futures_results: list[tuple[int, LLMGeneration, bool]] = []
        for index, prompt in enumerate(prompts):
            try:
                futures_results.append(generate(index, prompt))
            except Exception as exc:
                failures.append(f"request {index}: {type(exc).__name__}: {exc}")
    else:
        futures_results = []
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            future_map = {
                pool.submit(generate, index, prompt): index
                for index, prompt in enumerate(prompts)
            }
            for future in as_completed(future_map):
                index = future_map[future]
                try:
                    futures_results.append(future.result())
                except Exception as exc:
                    failures.append(f"request {index}: {type(exc).__name__}: {exc}")

    for index, generation, parse_error in sorted(futures_results):
        generations.append(generation)
        parse_failures += int(parse_error)
        request_rows.append(
            {
                "request_index": index,
                "response_sha256": hashlib.sha256(
                    generation.text.encode("utf-8")
                ).hexdigest(),
                "parse_error": parse_error,
                "metrics": generation.metrics,
            }
        )
    wall_seconds = time.perf_counter() - started
    result = aggregate_benchmark(
        generations,
        wall_seconds,
        load_gpu_samples(gpu_samples_path),
        failures,
        parse_failures,
        parallelism,
    )
    result["requests"] = request_rows
    return result


def build_representative_prompts(pilot_dir: Path) -> list[str]:
    prepared = prepare_expanded_evaluation(
        ExpandedEvaluationConfig(
            output_dir=Path("data/results/benchmark_unused"),
            pilot_dir=pilot_dir,
            case_count=300,
            seed=42,
        )
    )
    selected_cases: list[dict[str, Any]] = []
    for target_group in ("edital_selection", "event"):
        selected_cases.append(
            next(
                case
                for case in prepared.cases
                if case.get("macro_group") == target_group
            )
        )

    prompts: list[str] = []
    for case in selected_cases:
        pitch = str(case["incomplete_pitch"])
        prompts.append(_direct_llm_prompt(pitch))
        predicted_type = _predict_type(pitch)
        confirmed = _extract_fields_from_pitch(pitch)
        missing = _detect_missing_fields(pitch, confirmed)
        question_fields = _questionable_missing_fields(predicted_type, missing)
        questions = _adaptive_questions(predicted_type, question_fields)
        retrieved = _retrieve(prepared.retrieval_index, pitch, top_k=5)
        prompts.append(
            _virtual_reporter_llm_rag_prompt(
                pitch,
                predicted_type,
                confirmed,
                missing,
                questions,
                retrieved,
            )
        )
    return prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark do Ollama para a avaliacao expandida.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--parallelism", type=int, choices=(1, 2), required=True)
    run_parser.add_argument("--base-url", default="http://127.0.0.1:11435")
    run_parser.add_argument("--gpu-log", type=Path, required=True)
    run_parser.add_argument("--pilot-dir", type=Path, default=Path("data/results/evaluation"))
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--one", type=Path, required=True)
    select_parser.add_argument("--two", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.add_argument("--max-vram-mib", type=int, default=7800)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        config = LLMConfig("ollama", "gemma4:e4b", True, args.base_url)
        result = run_benchmark(
            build_representative_prompts(args.pilot_dir),
            LLMAdapter(config),
            args.parallelism,
            args.gpu_log,
        )
    else:
        results = {
            "1": json.loads(args.one.read_text(encoding="utf-8")),
            "2": json.loads(args.two.read_text(encoding="utf-8")),
        }
        selected = select_parallelism(results, args.max_vram_mib)
        result = {"selected_parallelism": selected, "candidates": results}
    _atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _number(value: str) -> float:
    cleaned = "".join(
        character
        for character in value.strip()
        if character.isdigit() or character in {"-", "."}
    )
    return float(cleaned) if cleaned not in {"", "-", "."} else 0.0


def _maximum(rows: list[dict[str, float]], key: str) -> float:
    return max((float(row.get(key, 0)) for row in rows), default=0.0)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[position]


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode(
        "utf-8"
    ) + b"\n"
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
