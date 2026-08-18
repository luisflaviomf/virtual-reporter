from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .baseline_comparison import (
    _aggregate_scores,
    _article_id,
    _build_retrieval_index,
    _fixed_split,
    _latex_escape,
    _latex_table,
    _load_usable_articles,
    _predict_type,
    _run_direct_llm,
    _run_fixed_form,
    _run_virtual_reporter,
    _run_virtual_reporter_llm_rag,
    _score_output,
    _select_example_case,
    _write_artifacts,
    _write_csv,
)
from .config import PATHS
from .evaluation_cases import (
    MASKING_VERSION,
    SAMPLING_VERSION,
    make_masked_case,
    sampling_summary,
    select_stratified_articles,
)
from .evaluation_checkpoint import CheckpointStore, RunFingerprint
from .io_utils import normalize_space, read_jsonl
from .llm_adapter import LLMAdapter


PROMPT_VERSION = "baseline-prompts-v1"
CONFIGURATION_ORDER = (
    "fixed_form_baseline",
    "direct_llm_baseline",
    "virtual_reporter",
    "virtual_reporter_llm_rag",
)
RUNTIME_ENV_NAMES = (
    "CUDA_VISIBLE_DEVICES",
    "VIRTUAL_REPORTER_LLM_PROVIDER",
    "VIRTUAL_REPORTER_LLM_MODEL",
    "VIRTUAL_REPORTER_LLM_BASE_URL",
    "VIRTUAL_REPORTER_OLLAMA_NUM_GPU",
    "VIRTUAL_REPORTER_OLLAMA_NUM_CTX",
    "VIRTUAL_REPORTER_OLLAMA_NUM_THREAD",
    "VIRTUAL_REPORTER_OLLAMA_NUM_PREDICT",
    "VIRTUAL_REPORTER_OLLAMA_FORMAT",
    "VIRTUAL_REPORTER_OLLAMA_THINK",
    "VIRTUAL_REPORTER_LLM_RETRIES",
    "VIRTUAL_REPORTER_PARSE_RETRIES",
    "VIRTUAL_REPORTER_CASE_WORKERS",
    "OLLAMA_FLASH_ATTENTION",
    "OLLAMA_KV_CACHE_TYPE",
    "OLLAMA_MAX_LOADED_MODELS",
    "OLLAMA_KEEP_ALIVE",
    "OLLAMA_GPU_OVERHEAD",
    "OLLAMA_NUM_PARALLEL",
    "VIRTUAL_REPORTER_GPU_NAME",
    "VIRTUAL_REPORTER_GPU_TOTAL_MIB",
    "VIRTUAL_REPORTER_GPU_DRIVER",
)
OPERATIONAL_ENV_NAMES = (
    "VIRTUAL_REPORTER_CASE_COOLDOWN_SECONDS",
    "VIRTUAL_REPORTER_LLM_COOLDOWN_SECONDS",
)


@dataclass(frozen=True)
class ExpandedEvaluationConfig:
    output_dir: Path
    pilot_dir: Path = Path("data/results/evaluation")
    case_count: int = 300
    seed: int = 42
    resume: bool = True
    prompt_version: str = PROMPT_VERSION
    model: str = "gemma4:e4b"
    runtime_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedExpandedEvaluation:
    cases: list[dict[str, Any]]
    retrieval_index: dict[str, Any]
    train_records: int
    test_records_available: int
    pilot_article_ids: set[str]


def validate_run_paths(config: ExpandedEvaluationConfig) -> None:
    output = config.output_dir.resolve()
    pilot = config.pilot_dir.resolve()
    if output == pilot:
        raise ValueError("output_dir must differ from pilot_dir")
    if output.exists() and any(output.iterdir()) and not (output / "checkpoint").exists():
        raise ValueError(
            f"output_dir {output} is non-empty and has no compatible checkpoint"
        )


def build_run_fingerprint(config: ExpandedEvaluationConfig) -> RunFingerprint:
    runtime = {
        name: os.environ.get(name, "")
        for name in RUNTIME_ENV_NAMES
    }
    runtime.update(config.runtime_metadata)
    runtime_digest = hashlib.sha256(_canonical_json(runtime)).hexdigest()
    return RunFingerprint(
        seed=config.seed,
        case_count=config.case_count,
        sampling_version=SAMPLING_VERSION,
        masking_version=MASKING_VERSION,
        prompt_version=config.prompt_version,
        model=config.model,
        runtime_digest=runtime_digest,
    )


def run_cases_with_resume(
    cases: list[dict[str, Any]],
    config: ExpandedEvaluationConfig,
    executor: Callable[[dict[str, Any]], dict[str, Any]],
    fingerprint: RunFingerprint,
) -> list[dict[str, Any]]:
    store = CheckpointStore.create_or_open(config.output_dir, fingerprint)
    completed = store.completed_case_ids()
    if completed and not config.resume:
        raise RuntimeError("checkpoint contains completed cases but resume is disabled")
    pending = [
        (position, case)
        for position, case in enumerate(cases, start=1)
        if str(case["case_id"]) not in completed
    ]
    total = len(cases)
    workers = _case_workers()
    if workers == 1:
        for position, case in pending:
            record = _execute_positioned_case(position, total, case, executor)
            store.save_case(record)
            print(
                f"[expanded_evaluation] case {position}/{total} "
                f"{case['case_id']} checkpointed",
                flush=True,
            )
            _apply_case_cooldown()
        return store.load_records()

    pool = ThreadPoolExecutor(max_workers=workers)
    futures: dict[Future[dict[str, Any]], tuple[int, dict[str, Any]]] = {}
    next_index = 0
    try:
        while next_index < min(workers, len(pending)):
            position, case = pending[next_index]
            futures[pool.submit(_execute_positioned_case, position, total, case, executor)] = (
                position,
                case,
            )
            next_index += 1
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                position, case = futures.pop(future)
                record = future.result()
                store.save_case(record)
                print(
                    f"[expanded_evaluation] case {position}/{total} "
                    f"{case['case_id']} checkpointed",
                    flush=True,
                )
                _apply_case_cooldown()
                if next_index < len(pending):
                    next_position, next_case = pending[next_index]
                    futures[
                        pool.submit(
                            _execute_positioned_case,
                            next_position,
                            total,
                            next_case,
                            executor,
                        )
                    ] = (next_position, next_case)
                    next_index += 1
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    return store.load_records()


def _execute_positioned_case(
    position: int,
    total: int,
    case: dict[str, Any],
    executor: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    print(
        f"[expanded_evaluation] case {position}/{total} {case_id} started",
        flush=True,
    )
    record = executor(case)
    require_complete_record(record)
    return record


def _case_workers() -> int:
    raw = os.environ.get("VIRTUAL_REPORTER_CASE_WORKERS", "1").strip()
    try:
        workers = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"VIRTUAL_REPORTER_CASE_WORKERS must be 1 or 2, got {raw!r}"
        ) from exc
    if workers not in {1, 2}:
        raise RuntimeError(
            f"VIRTUAL_REPORTER_CASE_WORKERS must be 1 or 2, got {workers}"
        )
    return workers


def _apply_case_cooldown() -> None:
    _apply_operational_delay("VIRTUAL_REPORTER_CASE_COOLDOWN_SECONDS")


def _apply_operational_delay(name: str) -> None:
    raw = os.environ.get(name, "0").strip()
    try:
        seconds = float(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"{name} must be a non-negative number, got {raw!r}"
        ) from exc
    if seconds < 0:
        raise RuntimeError(
            f"{name} must be non-negative, got {seconds}"
        )
    if seconds:
        time.sleep(seconds)


def execute_case(
    case: dict[str, Any],
    retrieval_index: dict[str, Any],
    adapter: Any,
    parse_retries: int = 2,
) -> dict[str, Any]:
    if parse_retries < 0:
        raise ValueError("parse_retries must be non-negative")

    timings: dict[str, float] = {}
    started = time.perf_counter()
    fixed_output = _run_fixed_form(case)
    timings["fixed_form_baseline_seconds"] = round(time.perf_counter() - started, 6)

    direct_output, direct_seconds = _run_until_valid_json(
        lambda attempt: _run_direct_llm(
            case, adapter, compact_retry=attempt > 0
        ),
        "direct_llm_baseline",
        parse_retries,
    )
    timings["direct_llm_baseline_seconds"] = round(direct_seconds, 6)
    _apply_operational_delay("VIRTUAL_REPORTER_LLM_COOLDOWN_SECONDS")

    started = time.perf_counter()
    virtual_output = _run_virtual_reporter(case, retrieval_index)
    timings["virtual_reporter_seconds"] = round(time.perf_counter() - started, 6)

    rag_output, rag_seconds = _run_until_valid_json(
        lambda attempt: _run_virtual_reporter_llm_rag(
            case,
            retrieval_index,
            adapter,
            compact_retry=attempt > 0,
        ),
        "virtual_reporter_llm_rag",
        parse_retries,
    )
    timings["virtual_reporter_llm_rag_seconds"] = round(rag_seconds, 6)

    outputs = [fixed_output, direct_output, virtual_output, rag_output]
    scores = [_score_output(case, output) for output in outputs]
    record = {
        "case": case,
        "outputs": outputs,
        "scores": scores,
        "timings": timings,
    }
    require_complete_record(record)
    return record


def _run_until_valid_json(
    operation: Callable[[int], dict[str, Any]],
    configuration: str,
    parse_retries: int,
) -> tuple[dict[str, Any], float]:
    total_seconds = 0.0
    for attempt in range(parse_retries + 1):
        started = time.perf_counter()
        output = operation(attempt)
        total_seconds += time.perf_counter() - started
        if output.get("status") != "ok":
            raise RuntimeError(
                f"{configuration} returned status={output.get('status')}: "
                f"{output.get('status_reason', '')}"
            )
        if not output.get("parse_error"):
            output["parse_attempts"] = attempt + 1
            return output, total_seconds
    raise RuntimeError(
        f"{configuration} returned invalid JSON after {parse_retries + 1} attempts"
    )


def require_complete_record(record: dict[str, Any]) -> None:
    case = record.get("case")
    outputs = record.get("outputs")
    scores = record.get("scores")
    if not isinstance(case, dict) or not case.get("case_id"):
        raise RuntimeError("case record is missing case_id")
    if not isinstance(outputs, list) or len(outputs) != 4:
        raise RuntimeError("case record must contain exactly four outputs")
    if not isinstance(scores, list) or len(scores) != 4:
        raise RuntimeError("case record must contain exactly four score rows")

    case_id = str(case["case_id"])
    configurations = [str(output.get("configuration", "")) for output in outputs]
    if set(configurations) != set(CONFIGURATION_ORDER):
        raise RuntimeError("case outputs do not contain the four required configurations")
    for output in outputs:
        if output.get("case_id") != case_id:
            raise RuntimeError("output case_id does not match checkpoint case")
        if output.get("status") != "ok":
            raise RuntimeError(f"output {output.get('configuration')} is not ok")
        if output.get("parse_error"):
            raise RuntimeError(f"output {output.get('configuration')} has a parse error")
    score_configurations = {str(score.get("configuration", "")) for score in scores}
    if score_configurations != set(CONFIGURATION_ORDER):
        raise RuntimeError("score rows do not contain the four required configurations")


def prepare_expanded_evaluation(
    config: ExpandedEvaluationConfig,
) -> PreparedExpandedEvaluation:
    if config.case_count <= 0 or config.case_count % 6:
        raise ValueError("case_count must be a positive multiple of 6")
    pilot_cases = read_jsonl(config.pilot_dir / "test_cases.jsonl")
    pilot_ids = {str(case.get("article_id", "")) for case in pilot_cases}
    articles = _load_usable_articles()
    train_articles, test_articles = _fixed_split(articles, seed=config.seed)
    selected = select_stratified_articles(
        test_articles,
        pilot_article_ids=pilot_ids,
        per_group=config.case_count // 6,
        seed=config.seed,
        labeler=lambda article: _predict_type(
            normalize_space(f"{article.get('title', '')}\n{article.get('lead', '')}")
        ),
    )
    cases: list[dict[str, Any]] = []
    group_positions: Counter[str] = Counter()
    for number, item in enumerate(selected, start=1):
        position = group_positions[item.macro_group]
        group_positions[item.macro_group] += 1
        requested_depth = (2, 3, 4, 5)[position % 4]
        cases.append(
            make_masked_case(
                item.article,
                number,
                item.macro_group,
                item.original_label,
                config.seed,
                requested_depth,
            )
        )

    selected_ids = {str(case["article_id"]) for case in cases}
    if len(cases) != config.case_count or len(selected_ids) != config.case_count:
        raise RuntimeError("expanded sample is not unique or has the wrong size")
    if selected_ids & pilot_ids:
        raise RuntimeError("expanded sample contains pilot article IDs")
    retrieval_articles = [
        article for article in train_articles if _article_id(article) not in selected_ids
    ]
    retrieval_index = _build_retrieval_index(retrieval_articles)
    if selected_ids & {str(doc["article_id"]) for doc in retrieval_index["docs"]}:
        raise RuntimeError("selected articles leaked into the retrieval index")
    return PreparedExpandedEvaluation(
        cases=cases,
        retrieval_index=retrieval_index,
        train_records=len(retrieval_articles),
        test_records_available=len(test_articles),
        pilot_article_ids=pilot_ids,
    )


def run_expanded_evaluation(
    config: ExpandedEvaluationConfig,
    adapter: LLMAdapter | None = None,
    case_executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    validate_run_paths(config)
    prepared = prepare_expanded_evaluation(config)
    adapter = adapter or LLMAdapter()
    if not adapter.is_available:
        raise RuntimeError(adapter.config.reason or "real LLM execution is unavailable")
    parse_retries = int(os.environ.get("VIRTUAL_REPORTER_PARSE_RETRIES", "2"))
    executor = case_executor or (
        lambda case: execute_case(
            case,
            prepared.retrieval_index,
            adapter,
            parse_retries=parse_retries,
        )
    )
    fingerprint = build_run_fingerprint(config)
    records = run_cases_with_resume(prepared.cases, config, executor, fingerprint)
    result = finalize_expanded_run(
        config,
        records,
        extra_metadata={
            "train_records": prepared.train_records,
            "test_records_available": prepared.test_records_available,
            "fingerprint": fingerprint.as_dict(),
            "fingerprint_digest": fingerprint.digest(),
        },
    )
    return result


def finalize_expanded_run(
    config: ExpandedEvaluationConfig,
    records: list[dict[str, Any]],
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ordered_records = sorted(records, key=lambda item: str(item["case"]["case_id"]))
    cases = [record["case"] for record in ordered_records]
    output_order = {name: index for index, name in enumerate(CONFIGURATION_ORDER)}
    outputs = sorted(
        [output for record in ordered_records for output in record["outputs"]],
        key=lambda item: (str(item["case_id"]), output_order[str(item["configuration"])]),
    )
    scores = sorted(
        [score for record in ordered_records for score in record["scores"]],
        key=lambda item: (str(item["case_id"]), output_order[str(item["configuration"])]),
    )
    comparison = _aggregate_scores(scores)
    _write_artifacts(config.output_dir, cases, outputs, scores, comparison)

    article_comparison = [dict(row) for row in comparison]
    for row in article_comparison:
        if row.get("Configuration") == "Virtual Reporter":
            row["Configuration"] = "Structured workflow simulation"
    _write_csv(config.output_dir / "baseline_comparison_article.csv", article_comparison)
    (config.output_dir / "baseline_comparison_article.tex").write_text(
        _latex_table(article_comparison), encoding="utf-8"
    )

    sample_summary = sampling_summary(cases)
    _atomic_write_json(config.output_dir / "sampling_summary.json", sample_summary)
    _write_sampling_tables(config.output_dir, cases, sample_summary)

    self_leaks = _retrieval_self_leaks(cases, outputs)
    parse_errors = sum(bool(output.get("parse_error")) for output in outputs)
    status_counts = dict(sorted(Counter(str(output.get("status", "")) for output in outputs).items()))
    metadata = {
        "case_count": len(cases),
        "seed": config.seed,
        "model": config.model,
        "sampling_version": SAMPLING_VERSION,
        "masking_version": MASKING_VERSION,
        "prompt_version": config.prompt_version,
        "mtp_enabled": False,
        "mtp_reason": "The Windows/CUDA gemma4:e4b checkpoint has no compatible draft layer.",
        "runtime_options": {
            name: os.environ.get(name, "")
            for name in RUNTIME_ENV_NAMES + OPERATIONAL_ENV_NAMES
        },
        "runtime_metadata": config.runtime_metadata,
        "dataset_sha256": _sha256_file(PATHS.canonical_jsonl),
        "source_sha256": _source_hashes(),
        "status_counts": status_counts,
        "parse_errors": parse_errors,
        "retrieval_original_article_leaks": len(self_leaks),
        "uses_ground_truth_during_generation": False,
        "ground_truth_usage": "scoring_only",
        "sampling_summary": sample_summary,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    _atomic_write_json(config.output_dir / "evaluation_metadata.json", metadata)

    result = {
        "cases": len(cases),
        "seed": config.seed,
        "example_case_id": _select_example_case(cases)["case_id"] if cases else "",
        "metrics": comparison,
        "outputs": {
            "run_dir": str(config.output_dir),
            "test_cases": str(config.output_dir / "test_cases.jsonl"),
            "baseline_outputs": str(config.output_dir / "baseline_outputs.jsonl"),
            "metrics_by_case": str(config.output_dir / "metrics_by_case.csv"),
            "evaluation_metadata": str(config.output_dir / "evaluation_metadata.json"),
        },
    }
    _atomic_write_json(config.output_dir / "baseline_comparison_summary.json", result)
    return result


def audit_expanded_run(output_dir: Path, pilot_dir: Path) -> dict[str, Any]:
    cases = read_jsonl(output_dir / "test_cases.jsonl")
    outputs = read_jsonl(output_dir / "baseline_outputs.jsonl")
    pilot_cases = read_jsonl(pilot_dir / "test_cases.jsonl")
    with (output_dir / "metrics_by_case.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        metrics = list(csv.DictReader(handle))

    case_ids = [str(case.get("case_id", "")) for case in cases]
    article_ids = [str(case.get("article_id", "")) for case in cases]
    pilot_ids = {str(case.get("article_id", "")) for case in pilot_cases}
    case_article = {str(case["case_id"]): str(case["article_id"]) for case in cases}
    output_counts = Counter(str(output.get("configuration", "")) for output in outputs)
    group_counts = Counter(str(case.get("macro_group", "")) for case in cases)
    self_leaks = [
        f"{output.get('case_id')}:{output.get('configuration')}"
        for output in outputs
        if case_article.get(str(output.get("case_id", "")))
        in set(str(value) for value in output.get("retrieved_article_ids", []))
    ]
    masked_input_leaks: list[str] = []
    cases_by_id = {str(case["case_id"]): case for case in cases}
    for output in outputs:
        case = cases_by_id.get(str(output.get("case_id", "")))
        if not case:
            continue
        pitch = str(output.get("input_pitch", "")).casefold()
        for field_name in case.get("removed_fields", []):
            value = normalize_space(case.get("ground_truth", {}).get(field_name, ""))
            if value and value.casefold() in pitch:
                masked_input_leaks.append(
                    f"{case['case_id']}:{output.get('configuration')}:{field_name}"
                )
    direct_retrieval = [
        str(output.get("case_id"))
        for output in outputs
        if output.get("configuration") == "direct_llm_baseline"
        and output.get("retrieved_article_ids")
    ]
    parse_errors = [
        f"{output.get('case_id')}:{output.get('configuration')}"
        for output in outputs
        if output.get("parse_error")
    ]
    non_ok = [
        f"{output.get('case_id')}:{output.get('configuration')}"
        for output in outputs
        if output.get("status") != "ok"
    ]
    forbidden_keys = {"ground_truth", "original_title", "expected_fields", "expected_missing_fields"}
    forbidden_output_keys = [
        f"{output.get('case_id')}:{key}"
        for output in outputs
        for key in forbidden_keys
        if key in output
    ]
    checks = {
        "case_count_300": len(cases) == 300,
        "unique_case_ids_300": len(set(case_ids)) == 300,
        "unique_article_ids_300": len(set(article_ids)) == 300,
        "pilot_overlap_zero": not (set(article_ids) & pilot_ids),
        "six_groups_of_50": len(group_counts) == 6 and set(group_counts.values()) == {50},
        "output_count_1200": len(outputs) == 1200,
        "metric_count_1200": len(metrics) == 1200,
        "each_configuration_300": set(output_counts) == set(CONFIGURATION_ORDER)
        and set(output_counts.values()) == {300},
        "parse_errors_zero": not parse_errors,
        "non_ok_zero": not non_ok,
        "retrieval_self_leaks_zero": not self_leaks,
        "masked_input_leaks_zero": not masked_input_leaks,
        "direct_retrieval_zero": not direct_retrieval,
        "forbidden_output_keys_zero": not forbidden_output_keys,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "group_counts": dict(sorted(group_counts.items())),
        "configuration_counts": dict(sorted(output_counts.items())),
        "parse_errors": parse_errors,
        "non_ok": non_ok,
        "retrieval_self_leaks": self_leaks,
        "masked_input_leaks": masked_input_leaks,
        "direct_retrieval": direct_retrieval,
        "forbidden_output_keys": forbidden_output_keys,
    }


def _write_sampling_tables(
    output_dir: Path,
    cases: list[dict[str, Any]],
    sample_summary: dict[str, Any],
) -> None:
    sampling_rows = [
        {"Dimension": "macro_group", "Value": key, "Cases": value}
        for key, value in sample_summary["macro_groups"].items()
    ] + [
        {"Dimension": "mask_depth", "Value": key, "Cases": value}
        for key, value in sample_summary["mask_depths"].items()
    ]
    _write_csv(output_dir / "sampling_summary.csv", sampling_rows)
    removed_rows = [
        {
            "Removed field": field_name,
            "Number of cases": count,
            "Percentage": round(100 * count / len(cases), 1) if cases else 0.0,
        }
        for field_name, count in sample_summary["removed_fields"].items()
    ]
    _write_csv(output_dir / "removed_fields_summary.csv", removed_rows)
    lines = [
        "\\begin{tabular}{lrr}",
        "\\hline",
        r"Removed field & Number of cases & Percentage \\ ",
        "\\hline",
    ]
    for row in removed_rows:
        lines.append(
            f"{_latex_escape(row['Removed field'])} & {row['Number of cases']} & {row['Percentage']}"
            + r" \\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    (output_dir / "removed_fields_summary.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _retrieval_self_leaks(
    cases: list[dict[str, Any]], outputs: list[dict[str, Any]]
) -> list[str]:
    case_articles = {str(case["case_id"]): str(case["article_id"]) for case in cases}
    return [
        f"{output.get('case_id')}:{output.get('configuration')}"
        for output in outputs
        if case_articles.get(str(output.get("case_id", "")))
        in {str(value) for value in output.get("retrieved_article_ids", [])}
    ]


def _source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__),
        Path(__file__).with_name("baseline_comparison.py"),
        Path(__file__).with_name("evaluation_cases.py"),
        Path(__file__).with_name("llm_adapter.py"),
    )
    return {str(path.as_posix()): _sha256_file(path) for path in paths}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
