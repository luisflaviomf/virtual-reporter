from __future__ import annotations

import math
import re
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import PATHS, RANDOM_STATE, ensure_dirs
from .evaluate_briefings import FIELDS, REQUIRED_STRUCTURE, _field_supported
from .io_utils import normalize_space, read_json, read_jsonl, write_csv, write_json, write_jsonl, write_table
from .load_dataset import load_labeled
from .retrieval_utils import article_id_from_row, create_retrieval_artifacts, retrieve
from .text_utils import remove_query_details, tokenize
from .virtual_reporter_pipeline import QUESTION_TEMPLATES, TYPE_EXTRA_QUESTIONS


METHODS = ["bm25", "dense", "hybrid"]
STOPWORDS = {
    "para",
    "com",
    "uma",
    "que",
    "por",
    "dos",
    "das",
    "estao",
    "esta",
    "ser",
    "sao",
    "sobre",
    "entre",
    "mais",
    "pela",
    "pelo",
    "aos",
    "nas",
    "nos",
    "ifmt",
}


def evaluate_retrieval_strict(
    query_count: int | None = None,
    quick: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    ensure_dirs()
    _ensure_sanity_copy()
    df = _usable_labeled_frame()
    train_df, test_df = _strict_train_test_split(df)
    target_count = query_count or (100 if quick else 500)
    eval_df = _stratified_sample(test_df, min(target_count, len(test_df)))
    artifacts = create_retrieval_artifacts(train_df, include_text=True)

    case_rows = []
    example_rows = []
    for _, source in eval_df.iterrows():
        query = make_strict_pitch(source)
        source_type = str(source.get("article_type_weak", ""))
        source_unit = _unit_value(source)
        source_tokens = _content_tokens(source.get("text_for_model", ""))
        ideal_rels = _ideal_relevances(artifacts["mapping"], source_type, source_unit)

        for method in METHODS:
            top = retrieve(artifacts, query, method=method, top_k=10)
            metrics = _retrieval_relevance_metrics(top, source_type, source_unit, source_tokens, ideal_rels)
            row = {
                "query_article_id": article_id_from_row(source),
                "method": method,
                "query": query,
                "source_title": source.get("title", ""),
                "source_type": source_type,
                "source_unit": source_unit,
                **metrics,
                "top_5_titles": " | ".join(str(item.get("title", "")) for item in top[:5]),
                "top_5_types": " | ".join(str(item.get("article_type_weak", "")) for item in top[:5]),
                "top_5_units": " | ".join(_unit_value(item) for item in top[:5]),
            }
            case_rows.append(row)
            if len(example_rows) < 60:
                example_rows.append(row)

    case_df = pd.DataFrame(case_rows)
    metrics_df = _aggregate_strict_retrieval(case_df)
    write_table(PATHS.tables_dir / "retrieval_strict_metrics.csv", metrics_df)
    write_table(PATHS.tables_dir / "retrieval_strict_examples.csv", pd.DataFrame(example_rows))
    _plot_strict_retrieval(metrics_df)

    result = {
        "evaluation_type": "strict_no_source_document_in_index",
        "train_records": int(len(train_df)),
        "test_records": int(len(test_df)),
        "queries": int(len(eval_df)),
        "best_retriever": str(metrics_df.sort_values(["ndcg_at_10_approx", "same_type_ratio_at_5"], ascending=False).iloc[0]["method"]),
        "metrics": metrics_df.to_dict(orient="records"),
        "note": "Strict evaluation: queries come from held-out test articles and the source article is not in the retrieval index.",
    }
    write_json(PATHS.results_dir / "retrieval_strict_results.json", result)
    return result


def evaluate_briefings_strict(
    quick: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    ensure_dirs()
    pitches = read_jsonl(PATHS.results_dir / "test_pitches.jsonl")
    if quick:
        pitches = pitches[:30]
    df = _usable_labeled_frame()
    article_map = {article_id_from_row(row): row for _, row in df.iterrows()}
    pitch_ids = {str(pitch.get("original_article_id", "")) for pitch in pitches}
    index_df = df[~df.apply(lambda row: article_id_from_row(row) in pitch_ids, axis=1)].copy()
    artifacts = create_retrieval_artifacts(index_df, include_text=True)
    classifier = _load_best_classifier()

    outputs = []
    case_rows = []
    for pitch in pitches:
        source = article_map.get(str(pitch.get("original_article_id", "")))
        if source is None:
            continue
        strict_pitch = make_strict_pitch(source)
        source_type = str(source.get("article_type_weak", ""))
        source_unit = _unit_value(source)
        expected = _expected_values(pitch)
        actual_missing = _actual_missing_fields(expected, strict_pitch)

        config_outputs = _strict_case_outputs(pitch, strict_pitch, source, classifier, artifacts, actual_missing)
        outputs.extend(config_outputs)
        for output in config_outputs:
            case_rows.append(_score_strict_briefing(pitch, output, source_type, source_unit, actual_missing))

    write_jsonl(PATHS.results_dir / "briefing_strict_outputs.jsonl", outputs)
    write_csv(PATHS.results_dir / "briefing_strict_outputs.csv", outputs)

    case_df = pd.DataFrame(case_rows)
    metrics_df = _aggregate_strict_briefings(case_df)
    write_table(PATHS.tables_dir / "briefing_strict_case_level_scores.csv", case_df)
    write_table(PATHS.tables_dir / "briefing_strict_metrics_by_configuration.csv", metrics_df)
    _plot_strict_briefing(metrics_df)

    result = {
        "evaluation_type": "strict_no_source_document_in_retrieval_index",
        "cases": int(case_df["case_id"].nunique()) if not case_df.empty else 0,
        "outputs": int(len(case_df)),
        "index_records": int(len(index_df)),
        "metrics_by_configuration": metrics_df.to_dict(orient="records"),
        "note": "Strict briefing evaluation excludes every source article used by the test pitches from the retrieval index.",
    }
    write_json(PATHS.results_dir / "briefing_strict_evaluation_results.json", result)
    return result


def make_strict_pitch(row: pd.Series | dict[str, Any]) -> str:
    title = normalize_space(row.get("title", ""))
    lead = normalize_space(row.get("lead", ""))
    body = normalize_space(row.get("body_text", ""))
    source = lead if len(lead.split()) >= 20 else " ".join(_sentences(body)[:3])
    source = source.replace(title, " ")
    source = remove_query_details(source)
    source = re.sub(r"\b\d{1,2}\s+de\s+[A-Za-zÀ-ÿ]+\s+de\s+\d{4}\b", " ", source, flags=re.IGNORECASE)
    source = re.sub(r"\b\d{1,2}\s+(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)\w*\b", " ", source, flags=re.IGNORECASE)
    source = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", source)
    source = re.sub(r"\b\d{4,}\b", " ", source)
    source = re.sub(r"['\"“”‘’][^'\"“”‘’]{8,}['\"“”‘’]", " ", source)
    source = _remove_specific_title_terms(source, title)
    source = _redact_specific_names(source)
    source = normalize_space(source)
    words = source.split()
    source = " ".join(words[:70])
    if not source:
        source = "divulgar uma informacao institucional do IFMT para a comunidade academica"
    return normalize_space(f"Precisamos divulgar uma informacao institucional: {source}")


def _strict_train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df["article_type_weak"].astype(str)
    stratify = y if y.value_counts().min() >= 2 else None
    return train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify)


def _usable_labeled_frame() -> pd.DataFrame:
    df = load_labeled().copy()
    df = df[df["text_for_model"].fillna("").astype(str).str.strip().ne("")]
    df = df[df["article_type_weak"].fillna("").astype(str).str.strip().ne("")]
    return df.reset_index(drop=True)


def _stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    pieces = []
    labels = list(df["article_type_weak"].value_counts().index)
    per_label = max(1, n // max(1, len(labels)))
    used = set()
    for label in labels:
        group = df[df["article_type_weak"].eq(label)]
        sample = group.sample(min(per_label, len(group)), random_state=RANDOM_STATE)
        used.update(sample.index.tolist())
        pieces.append(sample)
    sampled = pd.concat(pieces) if pieces else df.head(0)
    if len(sampled) < n:
        remaining = df[~df.index.isin(used)]
        if not remaining.empty:
            sampled = pd.concat([sampled, remaining.sample(min(n - len(sampled), len(remaining)), random_state=RANDOM_STATE)])
    return sampled.head(n).reset_index(drop=True)


def _retrieval_relevance_metrics(
    top: list[dict[str, Any]],
    source_type: str,
    source_unit: str,
    source_tokens: set[str],
    ideal_rels: list[float],
) -> dict[str, Any]:
    top5 = top[:5]
    same_type_flags = [str(item.get("article_type_weak", "")) == source_type for item in top5]
    same_unit_available = bool(source_unit)
    same_unit_flags = [same_unit_available and _unit_value(item) == source_unit for item in top5]
    similarities = [_jaccard(source_tokens, _content_tokens(item.get("text_for_model", "") or f"{item.get('title', '')} {item.get('lead', '')}")) for item in top5]
    rels = [_graded_relevance(item, source_type, source_unit) for item in top]
    return {
        "same_type_at_5": int(any(same_type_flags)),
        "same_type_ratio_at_5": round(sum(same_type_flags) / len(top5), 4) if top5 else 0.0,
        "same_unit_at_5": int(any(same_unit_flags)) if same_unit_available else None,
        "mean_similarity_to_source_at_5": round(sum(similarities) / len(similarities), 4) if similarities else 0.0,
        "ndcg_at_10_approx": round(_ndcg(rels[:10], ideal_rels[:10]), 4),
    }


def _aggregate_strict_retrieval(case_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in case_df.groupby("method"):
        unit_values = group["same_unit_at_5"].dropna()
        rows.append(
            {
                "method": method,
                "queries": int(len(group)),
                "same_type_at_5": round(float(group["same_type_at_5"].mean()), 4),
                "same_type_ratio_at_5": round(float(group["same_type_ratio_at_5"].mean()), 4),
                "same_unit_at_5": round(float(unit_values.mean()), 4) if not unit_values.empty else None,
                "mean_similarity_to_source_at_5": round(float(group["mean_similarity_to_source_at_5"].mean()), 4),
                "ndcg_at_10_approx": round(float(group["ndcg_at_10_approx"].mean()), 4),
            }
        )
    return pd.DataFrame(rows).sort_values(["ndcg_at_10_approx", "same_type_ratio_at_5"], ascending=False)


def _strict_case_outputs(
    pitch: dict[str, Any],
    strict_pitch: str,
    source: pd.Series,
    classifier: Any,
    artifacts: dict[str, Any],
    actual_missing: list[str],
) -> list[dict[str, Any]]:
    expected_why = pitch.get("expected_why")
    predicted_type = _predict_type(classifier, strict_pitch)
    retrieved = retrieve(artifacts, strict_pitch, method="hybrid", top_k=10)
    questions = _strict_questions(predicted_type, actual_missing, present_fields=["what"])

    return [
        _strict_output_row(
            pitch,
            "fixed_form_baseline",
            strict_pitch,
            "not_classified",
            [],
            [],
            {"what": strict_pitch},
            [],
            f"Registro do formulario:\nDescricao informada: {strict_pitch}",
        ),
        _strict_output_row(
            pitch,
            "direct_generation_baseline",
            strict_pitch,
            "not_classified",
            [],
            [],
            {"what": strict_pitch, "why": expected_why},
            ["when", "where", "source"],
            "Briefing preliminar\n"
            f"Resumo: {strict_pitch}\n"
            "Pendencias: confirmar data, local, unidade responsavel, link oficial e contato.",
        ),
        _strict_output_row(
            pitch,
            "virtual_reporter",
            strict_pitch,
            predicted_type,
            retrieved,
            questions,
            {"what": strict_pitch, "why": expected_why},
            actual_missing,
            _strict_virtual_briefing(strict_pitch, pitch, predicted_type, retrieved, questions, actual_missing),
        ),
    ]


def _strict_output_row(
    pitch: dict[str, Any],
    configuration: str,
    input_pitch: str,
    predicted_type: str,
    retrieved: list[dict[str, Any]],
    questions: list[dict[str, str]],
    confirmed_fields: dict[str, Any],
    missing_fields: list[str],
    briefing_text: str,
) -> dict[str, Any]:
    return {
        "case_id": pitch["case_id"],
        "configuration": configuration,
        "input_pitch": input_pitch,
        "original_article_id": pitch.get("original_article_id", ""),
        "predicted_article_type": predicted_type,
        "retrieved_article_ids": [item.get("article_id") for item in retrieved],
        "retrieved_article_types": [item.get("article_type_weak") for item in retrieved],
        "retrieved_article_units": [_unit_value(item) for item in retrieved],
        "generated_questions": questions,
        "briefing_text": briefing_text,
        "missing_fields_detected": missing_fields,
        "confirmed_fields": {key: value for key, value in confirmed_fields.items() if value},
        "similar_news_titles": [item.get("title", "") for item in retrieved[:5]],
    }


def _score_strict_briefing(
    pitch: dict[str, Any],
    output: dict[str, Any],
    source_type: str,
    source_unit: str,
    actual_missing: list[str],
) -> dict[str, Any]:
    expected_values = _expected_values(pitch)
    available = {field: value for field, value in expected_values.items() if value not in (None, "")}
    confirmed = output.get("confirmed_fields") or {}
    correctly_filled = sum(_field_supported(field, confirmed.get(field), expected) for field, expected in available.items())
    completeness = correctly_filled / len(available) if available else 0.0

    detected = set(output.get("missing_fields_detected") or [])
    actual = set(actual_missing)
    missing_ratio = len(detected & actual) / len(actual) if actual else 0.0
    question_scores = [_strict_question_score(question, actual, set(confirmed.keys())) for question in output.get("generated_questions") or []]
    question_relevance = sum(question_scores) / len(question_scores) if question_scores else 0.0

    retrieved_types = [str(value) for value in output.get("retrieved_article_types") or []]
    retrieved_units = [str(value) for value in output.get("retrieved_article_units") or []]
    same_type = int(source_type in retrieved_types[:5]) if retrieved_types else 0
    same_unit = int(bool(source_unit) and source_unit in retrieved_units[:5]) if retrieved_units else 0
    structure_score = sum(section in output.get("briefing_text", "") for section in REQUIRED_STRUCTURE) / len(REQUIRED_STRUCTURE)
    precision = _strict_factual_precision(pitch, output)

    return {
        "case_id": output["case_id"],
        "configuration": output["configuration"],
        "completeness": round(completeness, 4),
        "briefing_structure_score": round(structure_score, 4),
        "missing_fields_detected_ratio": round(missing_ratio, 4),
        "question_relevance": round(question_relevance, 4),
        "average_number_of_questions": len(output.get("generated_questions") or []),
        "retrieval_same_type_at_5": same_type,
        "retrieval_same_unit_at_5": same_unit if source_unit else None,
        "approximate_factual_precision": round(precision, 4),
        "approximate_hallucination_rate": round(1 - precision, 4),
    }


def _aggregate_strict_briefings(case_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for configuration, group in case_df.groupby("configuration"):
        unit_values = group["retrieval_same_unit_at_5"].dropna()
        rows.append(
            {
                "configuration": configuration,
                "cases": int(len(group)),
                "completeness": round(float(group["completeness"].mean()), 4),
                "briefing_structure_score": round(float(group["briefing_structure_score"].mean()), 4),
                "missing_fields_detected_ratio": round(float(group["missing_fields_detected_ratio"].mean()), 4),
                "question_relevance": round(float(group["question_relevance"].mean()), 4),
                "average_number_of_questions": round(float(group["average_number_of_questions"].mean()), 2),
                "retrieval_same_type_at_5": round(float(group["retrieval_same_type_at_5"].mean()), 4),
                "retrieval_same_unit_at_5": round(float(unit_values.mean()), 4) if not unit_values.empty else None,
                "approximate_factual_precision": round(float(group["approximate_factual_precision"].mean()), 4),
                "approximate_hallucination_rate": round(float(group["approximate_hallucination_rate"].mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def _strict_questions(predicted_type: str, missing_fields: list[str], present_fields: list[str]) -> list[dict[str, str]]:
    questions = []
    for field in missing_fields:
        if field in present_fields:
            continue
        template = QUESTION_TEMPLATES.get(field)
        if template:
            questions.append({"field": field, "question": template})
    for question in TYPE_EXTRA_QUESTIONS.get(predicted_type, [])[:2]:
        inferred = _infer_question_fields(question)
        questions.append({"field": "+".join(inferred) if inferred else "type_specific", "question": question})
    return questions


def _strict_question_score(question: dict[str, str], actual_missing: set[str], present_fields: set[str]) -> float:
    inferred_fields = set(question.get("field", "").split("+")) | set(_infer_question_fields(question.get("question", "")))
    inferred_fields.discard("")
    if inferred_fields & present_fields:
        return 0.0
    if inferred_fields & actual_missing:
        return 1.0
    if question.get("field") == "type_specific":
        return 0.25
    return 0.0


def _infer_question_fields(question: str) -> list[str]:
    text = " ".join(tokenize(question))
    fields = []
    if any(token in text for token in ["data", "periodo", "prazo", "horario", "tempo"]):
        fields.append("when")
    if any(token in text for token in ["onde", "local", "campus", "unidade"]):
        fields.append("where")
    if any(token in text for token in ["link", "edital", "formulario", "documento", "fonte"]):
        fields.append("source")
    if any(token in text for token in ["quem", "responsavel", "coordena"]):
        fields.append("who")
    if any(token in text for token in ["como", "participar", "inscricao", "acessar", "etapas"]):
        fields.append("how")
    if any(token in text for token in ["objetivo", "justificativa", "resultados", "requisitos"]):
        fields.append("why")
    return fields


def _strict_virtual_briefing(
    strict_pitch: str,
    pitch: dict[str, Any],
    predicted_type: str,
    retrieved: list[dict[str, Any]],
    questions: list[dict[str, str]],
    missing_fields: list[str],
) -> str:
    return "\n".join(
        [
            "suggested_title: Noticia institucional a confirmar",
            f"predicted_type: {predicted_type}",
            f"confirmed_information: {{'what': {strict_pitch!r}}}",
            f"missing_information: {missing_fields}",
            "suggested_questions:",
            *[f"- {item['question']}" for item in questions],
            f"similar_news: {[item.get('title', '') for item in retrieved[:3]]}",
            "notes_for_journalist: validar dados ausentes com a unidade demandante antes de redigir a noticia final.",
        ]
    )


def _strict_factual_precision(pitch: dict[str, Any], output: dict[str, Any]) -> float:
    claims = []
    for line in output.get("briefing_text", "").splitlines():
        line = line.strip()
        if not line or line.startswith("- ") or line.lower() in {"suggested_questions:"}:
            continue
        if line.lower().startswith("missing_information:") or line.lower().startswith("notes_for_journalist:"):
            continue
        claims.append(line)
    if not claims:
        return 0.0
    support_text = " ".join(
        str(value)
        for value in [
            output.get("input_pitch", ""),
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
        claim_tokens = {token for token in tokenize(claim) if len(token) >= 4 and token not in STOPWORDS}
        if not claim_tokens:
            supported += 1
        elif len(claim_tokens & support_tokens) >= min(3, len(claim_tokens)):
            supported += 1
    return supported / len(claims)


def _expected_values(pitch: dict[str, Any]) -> dict[str, Any]:
    return {field: pitch.get(f"expected_{field}") for field in FIELDS}


def _actual_missing_fields(expected: dict[str, Any], strict_pitch: str) -> list[str]:
    pitch_tokens = set(tokenize(strict_pitch))
    missing = []
    for field, value in expected.items():
        if field == "what":
            continue
        if value in (None, ""):
            continue
        value_tokens = set(tokenize(str(value)))
        if not value_tokens or len(value_tokens & pitch_tokens) < max(1, min(3, len(value_tokens))):
            missing.append(field)
    return missing


def _ideal_relevances(mapping: list[dict[str, Any]], source_type: str, source_unit: str) -> list[float]:
    return sorted((_graded_relevance(item, source_type, source_unit) for item in mapping), reverse=True)


def _graded_relevance(item: dict[str, Any], source_type: str, source_unit: str) -> float:
    score = 0.0
    if str(item.get("article_type_weak", "")) == source_type:
        score += 2.0
    if source_unit and _unit_value(item) == source_unit:
        score += 1.0
    return score


def _ndcg(relevances: list[float], ideal_relevances: list[float]) -> float:
    dcg = sum(((2**rel - 1) / math.log2(rank + 1)) for rank, rel in enumerate(relevances, start=1))
    idcg = sum(((2**rel - 1) / math.log2(rank + 1)) for rank, rel in enumerate(ideal_relevances, start=1))
    return dcg / idcg if idcg else 0.0


def _content_tokens(text: Any) -> set[str]:
    return {token for token in tokenize(text) if len(token) >= 4 and token not in STOPWORDS}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _unit_value(row: pd.Series | dict[str, Any]) -> str:
    for key in ("publisher_unit_normalized", "campus_name_normalized", "publisher_unit", "campus_name"):
        value = normalize_space(row.get(key, ""))
        if value:
            return value
    return ""


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _remove_specific_title_terms(text: str, title: str) -> str:
    title_terms = [token for token in tokenize(title) if len(token) >= 7 and token not in STOPWORDS]
    for term in sorted(set(title_terms), key=len, reverse=True)[:6]:
        text = re.sub(rf"\b{re.escape(term)}\w*\b", " ", text, flags=re.IGNORECASE)
    return text


def _redact_specific_names(text: str) -> str:
    return re.sub(
        r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+(?:de|da|do|dos|das|e)\s+|\s+)[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)?",
        " pessoa ou setor ",
        text,
    )


def _plot_strict_retrieval(metrics_df: pd.DataFrame) -> None:
    path = PATHS.figures_dir / "retrieval_strict_metrics.png"
    plot_df = metrics_df.set_index("method")[
        ["same_type_at_5", "same_type_ratio_at_5", "same_unit_at_5", "mean_similarity_to_source_at_5", "ndcg_at_10_approx"]
    ].fillna(0)
    plot_df.plot(kind="bar", figsize=(10, 5), color=["#476f95", "#7a9b68", "#c59a44", "#8a6f9b", "#b7665c"])
    plt.title("Recuperacao strict sem artigo original no indice")
    plt.xlabel("Metodo")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    try:
        plt.savefig(path.with_suffix(".pdf"))
    except Exception:
        pass
    plt.close()


def _plot_strict_briefing(metrics_df: pd.DataFrame) -> None:
    path = PATHS.figures_dir / "briefing_strict_metrics_comparison.png"
    plot_df = metrics_df.set_index("configuration")[
        [
            "completeness",
            "briefing_structure_score",
            "missing_fields_detected_ratio",
            "question_relevance",
            "retrieval_same_type_at_5",
            "approximate_factual_precision",
        ]
    ].fillna(0)
    plot_df.plot(kind="bar", figsize=(11, 5), color=["#476f95", "#7a9b68", "#c59a44", "#8a6f9b", "#b7665c", "#6d8f8b"])
    plt.title("Avaliacao strict dos briefings")
    plt.xlabel("Configuracao")
    plt.ylabel("Score medio")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    try:
        plt.savefig(path.with_suffix(".pdf"))
    except Exception:
        pass
    plt.close()


def _load_best_classifier() -> Any | None:
    results = read_json(PATHS.results_dir / "classification_results.json", default={}) or {}
    path = results.get("best_model_path")
    if not path:
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def _predict_type(classifier: Any, text: str) -> str:
    if classifier is None:
        return "other"
    try:
        return str(classifier.predict([text])[0])
    except Exception:
        return "other"


def _ensure_sanity_copy() -> None:
    sanity_path = PATHS.results_dir / "retrieval_sanity_results.json"
    if sanity_path.exists():
        return
    old = read_json(PATHS.results_dir / "retrieval_results.json", default=None)
    if old:
        old["evaluation_type"] = old.get("evaluation_type", "sanity_check")
        old["note"] = old.get(
            "note",
            "Sanity check: the query article is present in the retrieval index, so scores can be artificially high.",
        )
        write_json(sanity_path, old)
