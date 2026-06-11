from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .config import PATHS, ensure_dirs
from .io_utils import read_jsonl, write_json, write_table
from .text_utils import tokenize


REQUIRED_STRUCTURE = [
    "suggested_title",
    "confirmed_information",
    "missing_information",
    "suggested_questions",
    "similar_news",
    "notes_for_journalist",
]

FIELDS = ["who", "what", "when", "where", "why", "how", "source", "unit"]


def evaluate_briefings(
    quick: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    ensure_dirs()
    pitches = {row["case_id"]: row for row in read_jsonl(PATHS.results_dir / "test_pitches.jsonl")}
    outputs = read_jsonl(PATHS.results_dir / "briefing_outputs.jsonl")

    case_rows = []
    for output in outputs:
        pitch = pitches[output["case_id"]]
        case_rows.append(_score_output(pitch, output))

    case_df = pd.DataFrame(case_rows)
    write_table(PATHS.tables_dir / "briefing_case_level_scores.csv", case_df)

    metrics_df = (
        case_df.groupby("configuration")
        .agg(
            cases=("case_id", "count"),
            completeness=("completeness", "mean"),
            question_relevance=("question_relevance", "mean"),
            original_in_top_1=("original_in_top_1", "mean"),
            original_in_top_5=("original_in_top_5", "mean"),
            original_in_top_10=("original_in_top_10", "mean"),
            briefing_structure_score=("briefing_structure_score", "mean"),
            factual_precision_approx=("factual_precision_approx", "mean"),
            hallucination_rate_approx=("hallucination_rate_approx", "mean"),
        )
        .reset_index()
    )
    for col in metrics_df.columns:
        if col != "configuration" and col != "cases":
            metrics_df[col] = metrics_df[col].round(4)
    write_table(PATHS.tables_dir / "briefing_metrics_by_configuration.csv", metrics_df)
    write_table(PATHS.tables_dir / "paper_main_results_table.csv", metrics_df)
    _plot_briefing_metrics(metrics_df)

    result = {
        "cases": len(pitches),
        "outputs": len(outputs),
        "metrics_by_configuration": metrics_df.to_dict(orient="records"),
        "note": "Factual precision and hallucination are automatic approximations based on lexical support.",
    }
    write_json(PATHS.results_dir / "briefing_evaluation_results.json", result)
    return result


def _score_output(pitch: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    confirmed = output.get("confirmed_fields") or {}
    questions = output.get("generated_questions") or []
    missing = set(pitch.get("missing_fields_intended") or [])
    retrieved_ids = [str(item) for item in output.get("retrieved_article_ids") or []]
    original_id = str(pitch.get("original_article_id", ""))

    expected_values = {field: pitch.get(f"expected_{field}") for field in FIELDS}
    available = {field: value for field, value in expected_values.items() if value not in (None, "")}
    correctly_filled = sum(_field_supported(field, confirmed.get(field), expected) for field, expected in available.items())
    completeness = correctly_filled / len(available) if available else 0.0

    relevant_questions = 0
    for question in questions:
        field = question.get("field", "")
        if field in missing or (field == "type_specific" and output.get("predicted_article_type") not in {"", "other"}):
            relevant_questions += 1
    question_relevance = relevant_questions / len(questions) if questions else 0.0

    structure_score = sum(section in output.get("briefing_text", "") for section in REQUIRED_STRUCTURE) / len(REQUIRED_STRUCTURE)
    factual_precision = _factual_precision_approx(pitch, output)

    return {
        "case_id": output["case_id"],
        "configuration": output["configuration"],
        "completeness": round(completeness, 4),
        "question_relevance": round(question_relevance, 4),
        "original_in_top_1": int(bool(original_id and original_id in retrieved_ids[:1])),
        "original_in_top_5": int(bool(original_id and original_id in retrieved_ids[:5])),
        "original_in_top_10": int(bool(original_id and original_id in retrieved_ids[:10])),
        "briefing_structure_score": round(structure_score, 4),
        "factual_precision_approx": round(factual_precision, 4),
        "hallucination_rate_approx": round(1 - factual_precision, 4),
        "generated_questions_count": len(questions),
        "confirmed_fields_count": len([v for v in confirmed.values() if v]),
    }


def _field_supported(field: str, confirmed_value: Any, expected_value: Any) -> int:
    if confirmed_value in (None, ""):
        return 0
    if field == "what":
        return 1
    confirmed_tokens = set(tokenize(str(confirmed_value)))
    expected_tokens = set(tokenize(str(expected_value)))
    if not expected_tokens:
        return 0
    return int(bool(confirmed_tokens & expected_tokens) or str(expected_value) in str(confirmed_value))


def _factual_precision_approx(pitch: dict[str, Any], output: dict[str, Any]) -> float:
    briefing = output.get("briefing_text", "")
    claims = []
    for line in briefing.splitlines():
        line = line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered in {"suggested_questions:"} or lowered.startswith("- "):
            continue
        if lowered.startswith("missing_information:") or lowered.startswith("notes_for_journalist:"):
            continue
        claims.append(line)
    if not claims:
        return 0.0
    support_text = " ".join(
        str(value)
        for value in [
            pitch.get("input_pitch", ""),
            pitch.get("original_title", ""),
            pitch.get("expected_who", ""),
            pitch.get("expected_what", ""),
            pitch.get("expected_when", ""),
            pitch.get("expected_where", ""),
            pitch.get("expected_why", ""),
            pitch.get("expected_source", ""),
            pitch.get("expected_unit", ""),
            " ".join(output.get("similar_news_titles") or []),
        ]
        if value
    )
    support_tokens = set(tokenize(support_text))
    supported = 0
    for claim in claims:
        claim_tokens = {token for token in tokenize(claim) if len(token) >= 4}
        if not claim_tokens:
            supported += 1
        elif len(claim_tokens & support_tokens) >= min(3, len(claim_tokens)):
            supported += 1
    return supported / len(claims)


def _plot_briefing_metrics(metrics_df: pd.DataFrame) -> None:
    path = PATHS.figures_dir / "briefing_metrics_comparison.png"
    plot_df = metrics_df.set_index("configuration")[
        ["completeness", "question_relevance", "briefing_structure_score", "factual_precision_approx"]
    ]
    plot_df.plot(kind="bar", figsize=(10, 5), color=["#476f95", "#7a9b68", "#c59a44", "#8a6f9b"])
    plt.title("Comparacao das configuracoes do Virtual Reporter")
    plt.xlabel("Configuracao")
    plt.ylabel("Score medio")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    try:
        plt.savefig(path.with_suffix(".pdf"))
    except Exception:
        pass
    plt.close()
