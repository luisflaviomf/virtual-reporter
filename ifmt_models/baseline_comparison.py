from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PATHS, RANDOM_STATE, ensure_dirs
from .io_utils import normalize_space, read_jsonl, write_json, write_jsonl
from .llm_adapter import LLMAdapter
from .text_utils import tokenize


EVALUATION_DIR = PATHS.results_dir / "evaluation"
FIELDS = ["what", "who", "when", "where", "why", "how", "source", "unit"]
QUESTION_TEMPLATES = {
    "who": "Quem e o responsavel direto pela informacao ou atividade?",
    "when": "Qual e a data ou periodo exato da atividade, inscricao ou publicacao?",
    "where": "Onde a atividade acontece ou qual local deve ser informado?",
    "why": "Qual e o objetivo ou justificativa da divulgacao?",
    "how": "Como o publico deve participar, se inscrever ou acessar o servico?",
    "source": "Qual link oficial, edital, formulario, documento ou contato deve ser usado como fonte?",
    "unit": "Qual campus, reitoria ou unidade institucional deve assinar a informacao?",
}
TYPE_EXTRA_QUESTIONS = {
    "edital_selection": [
        "Qual e o prazo de inscricao e onde esta publicado o edital?",
        "Quais sao os requisitos, vagas e etapas do processo seletivo?",
    ],
    "event": [
        "Qual e a programacao, horario e local do evento?",
        "O evento exige inscricao previa ou tem limite de vagas?",
    ],
    "service_announcement": [
        "Quem sera afetado pelo comunicado e por quanto tempo?",
        "Existe canal de atendimento para duvidas da comunidade?",
    ],
    "research": ["Quem coordena a pesquisa e quais resultados devem ser destacados?"],
    "extension": ["Qual publico externo sera atendido pela acao de extensao?"],
}
STOPWORDS = {
    "para",
    "com",
    "uma",
    "que",
    "por",
    "dos",
    "das",
    "esta",
    "estao",
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
    "informacao",
    "institucional",
    "divulgacao",
    "demanda",
    "recebida",
}


def run_baseline_comparison(
    case_count: int = 30,
    seed: int = RANDOM_STATE,
    quick: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    ensure_dirs()
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    case_count = 10 if quick else case_count

    articles = _load_usable_articles()
    train_articles, test_articles = _fixed_split(articles, seed=seed)
    selected_test = test_articles[: min(case_count, len(test_articles))]
    test_ids = {_article_id(article) for article in selected_test}
    train_articles = [article for article in train_articles if _article_id(article) not in test_ids]
    retrieval_index = _build_retrieval_index(train_articles)

    cases = [_make_case(article, index + 1) for index, article in enumerate(selected_test)]
    adapter = LLMAdapter()
    outputs: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []

    total_cases = len(cases)
    for index, case in enumerate(cases, start=1):
        print(
            f"[baseline_comparison] case {index}/{total_cases} {case['case_id']} started",
            file=sys.stderr,
            flush=True,
        )
        fixed_output = _run_fixed_form(case)
        direct_started = time.perf_counter()
        direct_output = _run_direct_llm(case, adapter)
        direct_elapsed = time.perf_counter() - direct_started
        print(
            "[baseline_comparison] "
            f"case {index}/{total_cases} {case['case_id']} direct_llm "
            f"status={direct_output.get('status')} parse_error={direct_output.get('parse_error')} "
            f"seconds={direct_elapsed:.1f}",
            file=sys.stderr,
            flush=True,
        )
        virtual_output = _run_virtual_reporter(case, retrieval_index)
        vr_llm_started = time.perf_counter()
        virtual_llm_output = _run_virtual_reporter_llm_rag(case, retrieval_index, adapter)
        vr_llm_elapsed = time.perf_counter() - vr_llm_started
        print(
            "[baseline_comparison] "
            f"case {index}/{total_cases} {case['case_id']} virtual_reporter_llm_rag "
            f"status={virtual_llm_output.get('status')} parse_error={virtual_llm_output.get('parse_error')} "
            f"seconds={vr_llm_elapsed:.1f}",
            file=sys.stderr,
            flush=True,
        )
        case_outputs = [fixed_output, direct_output, virtual_output, virtual_llm_output]
        outputs.extend(case_outputs)
        for output in case_outputs:
            if output.get("status") == "ok":
                scores.append(_score_output(case, output))
        print(
            f"[baseline_comparison] case {index}/{total_cases} {case['case_id']} finished",
            file=sys.stderr,
            flush=True,
        )

    comparison = _aggregate_scores(scores)
    _write_artifacts(EVALUATION_DIR, cases, outputs, scores, comparison)
    metadata = {
        "case_count": len(cases),
        "seed": seed,
        "llm_provider": adapter.config.provider,
        "llm_model": adapter.config.model,
        "llm_base_url": adapter.config.base_url,
        "ollama_options": {
            "num_gpu": os.environ.get("VIRTUAL_REPORTER_OLLAMA_NUM_GPU", ""),
            "num_ctx": os.environ.get("VIRTUAL_REPORTER_OLLAMA_NUM_CTX", ""),
            "num_thread": os.environ.get("VIRTUAL_REPORTER_OLLAMA_NUM_THREAD", ""),
            "num_predict": os.environ.get("VIRTUAL_REPORTER_OLLAMA_NUM_PREDICT", ""),
            "format": os.environ.get("VIRTUAL_REPORTER_OLLAMA_FORMAT", ""),
            "think": os.environ.get("VIRTUAL_REPORTER_OLLAMA_THINK", ""),
        },
        "direct_llm_executed": any(
            output.get("configuration") == "direct_llm_baseline" and output.get("status") == "ok"
            for output in outputs
        ),
        "virtual_reporter_llm_rag_executed": any(
            output.get("configuration") == "virtual_reporter_llm_rag" and output.get("status") == "ok"
            for output in outputs
        ),
        "run_datetime_utc": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "Ground truth is stored in test_cases.jsonl for scoring only. It is not passed to Direct LLM "
            "generation, Virtual Reporter generation, retrieval queries, prompts, or fixed-form generation."
        ),
    }
    write_json(EVALUATION_DIR / "evaluation_metadata.json", metadata)

    result = {
        "cases": len(cases),
        "seed": seed,
        "train_records": len(train_articles),
        "test_records_available": len(test_articles),
        "direct_llm_status": "available" if adapter.is_available else "requires_llm",
        "direct_llm_reason": "" if adapter.is_available else adapter.config.reason,
        "example_case_id": _select_example_case(cases)["case_id"] if cases else "",
        "outputs": {
            "test_cases": str(EVALUATION_DIR / "test_cases.jsonl"),
            "baseline_outputs": str(EVALUATION_DIR / "baseline_outputs.jsonl"),
            "baseline_comparison_csv": str(EVALUATION_DIR / "baseline_comparison.csv"),
            "baseline_comparison_tex": str(EVALUATION_DIR / "baseline_comparison.tex"),
            "baseline_comparison_available_methods_csv": str(
                EVALUATION_DIR / "baseline_comparison_available_methods.csv"
            ),
            "baseline_comparison_available_methods_tex": str(
                EVALUATION_DIR / "baseline_comparison_available_methods.tex"
            ),
            "metrics_by_case": str(EVALUATION_DIR / "metrics_by_case.csv"),
            "example_case_md": str(EVALUATION_DIR / "example_case.md"),
            "example_case_tex": str(EVALUATION_DIR / "example_case.tex"),
            "evaluation_metadata": str(EVALUATION_DIR / "evaluation_metadata.json"),
        },
        "metrics": comparison,
    }
    write_json(EVALUATION_DIR / "baseline_comparison_summary.json", result)
    return result


def _load_usable_articles() -> list[dict[str, Any]]:
    rows = read_jsonl(PATHS.canonical_jsonl)
    usable = []
    for row in rows:
        if str(row.get("status", "collected")) != "collected":
            continue
        if not normalize_space(row.get("title", "")):
            continue
        if not normalize_space(row.get("body_text", "")):
            continue
        try:
            word_count = int(row.get("word_count") or 0)
        except (TypeError, ValueError):
            word_count = 0
        if word_count < 80:
            continue
        if not _unit_value(row):
            continue
        usable.append(row)
    return usable


def _fixed_split(articles: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(articles)
    random.Random(seed).shuffle(shuffled)
    split_at = int(len(shuffled) * 0.8)
    return shuffled[:split_at], shuffled[split_at:]


def _make_case(article: dict[str, Any], number: int) -> dict[str, Any]:
    title = normalize_space(article.get("title", ""))
    body = normalize_space(article.get("body_text", ""))
    lead = normalize_space(article.get("lead", "")) or _first_sentence(body)
    unit = _unit_value(article)
    source = normalize_space(article.get("original_url", "") or article.get("canonical_url", ""))
    when = normalize_space(article.get("publication_date", ""))
    who = normalize_space(article.get("author_name", "")) or unit
    how = _extract_how(body)

    ground_truth = {
        "what": title,
        "who": who,
        "when": when,
        "where": unit,
        "why": _first_sentence(lead or body),
        "how": how,
        "source": source,
        "unit": unit,
    }

    redaction_values = [when, unit, source, who]
    pitch_title = _redact_values(title, redaction_values)
    pitch_context = _redact_values(_first_sentence(lead or body), redaction_values)
    pitch = _build_incomplete_pitch(pitch_title, pitch_context)
    available_fields = {"what": pitch_title}
    if pitch_context:
        available_fields["why"] = pitch_context
    removed_fields = [
        field
        for field in FIELDS
        if field not in available_fields and normalize_space(ground_truth.get(field, ""))
    ]

    return {
        "case_id": f"case_{number:04d}",
        "article_id": _article_id(article),
        "original_title": title,
        "original_url": source,
        "original_unit": unit,
        "original_date": when,
        "incomplete_pitch": pitch,
        "removed_fields": removed_fields,
        "available_fields_in_pitch": available_fields,
        "ground_truth": ground_truth,
    }


def _build_incomplete_pitch(title: str, context: str) -> str:
    parts = [
        "Demanda recebida para divulgacao institucional.",
        f"Fato informado: {title or 'noticia institucional a apurar'}.",
    ]
    if context and context != title:
        parts.append(f"Contexto informado: {context}.")
    parts.append(
        "Dados ainda nao confirmados: data, local/unidade responsavel, fonte oficial/link e contato."
    )
    return normalize_space(" ".join(parts))


def _run_fixed_form(case: dict[str, Any]) -> dict[str, Any]:
    pitch = case["incomplete_pitch"]
    confirmed = _extract_fields_from_pitch(pitch)
    missing = _detect_missing_fields(pitch, confirmed)
    questions = [_question(field) for field in missing if field in QUESTION_TEMPLATES]
    claims = [value for value in confirmed.values() if value]
    briefing = "\n".join(
        [
            "configuration: fixed_form_baseline",
            "confirmed_information:",
            *[f"- {field}: {value}" for field, value in confirmed.items()],
            f"missing_information: {missing}",
            "suggested_questions:",
            *[f"- {item['question']}" for item in questions],
            "notes_for_journalist: preencher apenas campos confirmados pela fonte demandante.",
        ]
    )
    return _output_row(
        case=case,
        configuration="fixed_form_baseline",
        status="ok",
        predicted_type="not_classified",
        confirmed_fields=confirmed,
        missing_fields=missing,
        questions=questions,
        retrieved=[],
        briefing_text=briefing,
        claims=claims,
        evidence_texts=[pitch],
    )


def _run_direct_llm(
    case: dict[str, Any],
    adapter: LLMAdapter,
    compact_retry: bool = False,
) -> dict[str, Any]:
    pitch = case["incomplete_pitch"]
    if not adapter.is_available:
        return _output_row(
            case=case,
            configuration="direct_llm_baseline",
            status="requires_llm",
            predicted_type="not_classified",
            confirmed_fields={},
            missing_fields=[],
            questions=[],
            retrieved=[],
            briefing_text="",
            claims=[],
            evidence_texts=[pitch],
            status_reason=adapter.config.reason,
        )

    prompt = _direct_llm_prompt(pitch, compact_retry=compact_retry)
    generation = adapter.generate_with_metrics(prompt)
    response = generation.text
    parsed = _parse_direct_llm_response(response)
    confirmed = parsed["confirmed_fields"]
    missing = parsed["missing_fields"]
    questions = parsed["questions"]
    briefing = parsed["briefing_text"]
    claims = parsed["claims"]
    return _output_row(
        case=case,
        configuration="direct_llm_baseline",
        status="ok",
        predicted_type="not_classified",
        confirmed_fields={str(k): normalize_space(v) for k, v in confirmed.items() if normalize_space(v)},
        missing_fields=[str(field) for field in missing if str(field) in FIELDS],
        questions=questions,
        retrieved=[],
        briefing_text=briefing,
        claims=claims,
        evidence_texts=[pitch],
        raw_response=response,
        parse_error=parsed["parse_error"],
        status_reason=parsed["parse_error_reason"],
        generation_metrics=generation.metrics,
    )


def _run_virtual_reporter(case: dict[str, Any], retrieval_index: dict[str, Any]) -> dict[str, Any]:
    pitch = case["incomplete_pitch"]
    predicted_type = _predict_type(pitch)
    confirmed = _extract_fields_from_pitch(pitch)
    missing = _detect_missing_fields(pitch, confirmed)
    question_missing = _questionable_missing_fields(predicted_type, missing)
    questions = _adaptive_questions(predicted_type, question_missing)
    retrieved = _retrieve(retrieval_index, pitch, top_k=5)
    retrieved_titles = [item.get("title", "") for item in retrieved[:3] if item.get("title")]
    claims = [value for value in confirmed.values() if value] + retrieved_titles
    evidence_texts = [pitch]
    evidence_texts.extend(
        normalize_space(f"{item.get('title', '')}\n{item.get('lead', '')}\n{item.get('body_preview', '')}")
        for item in retrieved
    )
    briefing = "\n".join(
        [
            "suggested_title: Noticia institucional a apurar",
            f"predicted_type: {predicted_type}",
            "confirmed_information:",
            *[f"- {field}: {value}" for field, value in confirmed.items()],
            f"missing_information: {missing}",
            "suggested_questions:",
            *[f"- {item['question']}" for item in questions],
            "similar_news:",
            *[f"- {title}" for title in retrieved_titles],
            "notes_for_journalist: nao transformar exemplos recuperados em fatos do caso atual sem confirmacao.",
        ]
    )
    return _output_row(
        case=case,
        configuration="virtual_reporter",
        status="ok",
        predicted_type=predicted_type,
        confirmed_fields=confirmed,
        missing_fields=missing,
        questions=questions,
        retrieved=retrieved,
        briefing_text=briefing,
        claims=claims,
        evidence_texts=evidence_texts,
    )


def _run_virtual_reporter_llm_rag(
    case: dict[str, Any],
    retrieval_index: dict[str, Any],
    adapter: LLMAdapter,
    compact_retry: bool = False,
) -> dict[str, Any]:
    pitch = case["incomplete_pitch"]
    predicted_type = _predict_type(pitch)
    base_confirmed = _extract_fields_from_pitch(pitch)
    base_missing = _detect_missing_fields(pitch, base_confirmed)
    question_missing = _questionable_missing_fields(predicted_type, base_missing)
    base_questions = _adaptive_questions(predicted_type, question_missing)
    retrieved = _retrieve(retrieval_index, pitch, top_k=5)
    evidence_texts = _evidence_texts(pitch, retrieved)

    if not adapter.is_available:
        return _output_row(
            case=case,
            configuration="virtual_reporter_llm_rag",
            status="requires_llm",
            predicted_type=predicted_type,
            confirmed_fields=base_confirmed,
            missing_fields=base_missing,
            questions=base_questions,
            retrieved=retrieved,
            briefing_text="",
            claims=[],
            evidence_texts=evidence_texts,
            status_reason=adapter.config.reason,
        )

    prompt = _virtual_reporter_llm_rag_prompt(
        pitch=pitch,
        predicted_type=predicted_type,
        confirmed=base_confirmed,
        missing=base_missing,
        questions=base_questions,
        retrieved=retrieved,
        compact_retry=compact_retry,
    )
    generation = adapter.generate_with_metrics(prompt)
    response = generation.text
    parsed = _parse_direct_llm_response(response)

    confirmed = dict(base_confirmed)
    for field, value in parsed["confirmed_fields"].items():
        if field in FIELDS and field not in confirmed and normalize_space(value):
            confirmed[field] = value
    missing = _merge_fields(base_missing, parsed["missing_fields"])
    questions = _merge_questions(parsed["questions"], base_questions, missing)
    claims = parsed["claims"] or [value for value in confirmed.values() if value]
    briefing = parsed["briefing_text"] if not parsed["parse_error"] else _fallback_virtual_llm_briefing(
        predicted_type=predicted_type,
        confirmed=confirmed,
        missing=missing,
        questions=questions,
        retrieved=retrieved,
    )

    return _output_row(
        case=case,
        configuration="virtual_reporter_llm_rag",
        status="ok",
        predicted_type=predicted_type,
        confirmed_fields=confirmed,
        missing_fields=missing,
        questions=questions,
        retrieved=retrieved,
        briefing_text=briefing,
        claims=claims,
        evidence_texts=evidence_texts,
        raw_response=response,
        parse_error=parsed["parse_error"],
        status_reason=parsed["parse_error_reason"],
        generation_metrics=generation.metrics,
    )


def _output_row(
    case: dict[str, Any],
    configuration: str,
    status: str,
    predicted_type: str,
    confirmed_fields: dict[str, Any],
    missing_fields: list[str],
    questions: list[dict[str, str]],
    retrieved: list[dict[str, Any]],
    briefing_text: str,
    claims: list[str],
    evidence_texts: list[str],
    status_reason: str = "",
    raw_response: str = "",
    parse_error: bool = False,
    generation_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "configuration": configuration,
        "status": status,
        "status_reason": status_reason,
        "input_pitch": case["incomplete_pitch"],
        "predicted_article_type": predicted_type,
        "confirmed_fields": {key: value for key, value in confirmed_fields.items() if normalize_space(value)},
        "missing_fields_detected": missing_fields,
        "generated_questions": questions,
        "retrieved_documents": retrieved,
        "retrieved_article_ids": [item.get("article_id", "") for item in retrieved],
        "briefing_text": briefing_text,
        "claims": [normalize_space(claim) for claim in claims if normalize_space(claim)],
        "evidence_texts": [normalize_space(text) for text in evidence_texts if normalize_space(text)],
        "raw_response": raw_response,
        "parse_error": parse_error,
        "generation_metrics": generation_metrics or {},
    }


def _direct_llm_prompt(pitch: str, compact_retry: bool = False) -> str:
    prompt = (
        "Voce apoia uma equipe de comunicacao institucional. Receba apenas o pitch incompleto abaixo. "
        "Nao use ground truth, noticia original completa, documentos recuperados, classificador, label prevista "
        "ou campos esperados. Nao invente fatos ausentes.\n\n"
        "Responda somente JSON valido, sem Markdown, sem comentarios e sem texto fora do JSON. "
        "Use string vazia para campos nao confirmados. Liste campos ausentes em missing_information usando "
        "somente os IDs de campo, sem descricoes adicionais. Seja conciso: use no maximo 3 perguntas "
        "e 3 factual_claims.\n\n"
        "Schema obrigatorio:\n"
        "{\n"
        '  "confirmed_information": {\n'
        '    "what": "",\n'
        '    "who": "",\n'
        '    "when": "",\n'
        '    "where": "",\n'
        '    "why": "",\n'
        '    "how": "",\n'
        '    "source": "",\n'
        '    "unit": ""\n'
        "  },\n"
        '  "missing_information": [],\n'
        '  "suggested_questions": [],\n'
        '  "factual_claims": []\n'
        "}\n\n"
        "Campos validos para missing_information: what, who, when, where, why, how, source, unit.\n\n"
        f"Pitch incompleto:\n{pitch}"
    )
    return prompt + _compact_retry_instruction(compact_retry)


def _virtual_reporter_llm_rag_prompt(
    pitch: str,
    predicted_type: str,
    confirmed: dict[str, str],
    missing: list[str],
    questions: list[dict[str, str]],
    retrieved: list[dict[str, Any]],
    compact_retry: bool = False,
) -> str:
    retrieved_examples = [
        {
            "rank": item.get("rank"),
            "title": item.get("title", ""),
            "lead": normalize_space(item.get("lead", ""))[:260],
            "publication_date": item.get("publication_date", ""),
            "unit": item.get("unit", ""),
            "article_type": item.get("article_type", ""),
        }
        for item in retrieved[:2]
    ]
    pipeline_context = {
        "predicted_article_type": predicted_type,
        "confirmed_from_pitch": confirmed,
        "missing_fields_detected_by_pipeline": missing,
        "initial_adaptive_questions": questions,
        "retrieved_news_examples": retrieved_examples,
    }
    prompt = (
        "Voce e o modulo LLM do Virtual Reporter. Use apenas o pitch incompleto e os sinais do pipeline abaixo. "
        "Nao use ground truth, noticia original completa, expected fields, original_title ou qualquer dado fora desta entrada.\n\n"
        "Os exemplos recuperados vieram de um indice sem a noticia original do caso. Eles podem ajudar no estilo, "
        "no contexto institucional e em perguntas, mas nao devem ser copiados como fatos do caso atual se o pitch "
        "nao confirmar esses fatos. Diferencie fatos confirmados do caso atual e informacoes ainda pendentes.\n\n"
        "Mantenha campos ausentes detectados pelo pipeline, a menos que o proprio pitch confirme claramente o campo. "
        "Nao invente datas, links, contatos, unidades ou locais ausentes. Responda somente JSON valido, sem Markdown "
        "e sem texto fora do JSON. Seja conciso: use no maximo 3 perguntas e 3 factual_claims.\n\n"
        "Schema obrigatorio:\n"
        "{\n"
        '  "confirmed_information": {\n'
        '    "what": "",\n'
        '    "who": "",\n'
        '    "when": "",\n'
        '    "where": "",\n'
        '    "why": "",\n'
        '    "how": "",\n'
        '    "source": "",\n'
        '    "unit": ""\n'
        "  },\n"
        '  "missing_information": [],\n'
        '  "suggested_questions": [],\n'
        '  "factual_claims": []\n'
        "}\n\n"
        "Campos validos para missing_information: what, who, when, where, why, how, source, unit.\n\n"
        f"Pitch incompleto:\n{pitch}\n\n"
        "Sinais do pipeline Virtual Reporter:\n"
        f"{json.dumps(pipeline_context, ensure_ascii=False)}"
    )
    return prompt + _compact_retry_instruction(compact_retry)


def _compact_retry_instruction(enabled: bool) -> str:
    if not enabled:
        return ""
    return (
        "\n\nRETRY_COMPACT_JSON: A resposta anterior nao formou JSON valido dentro do "
        "limite. Gere o objeto completo em no maximo 350 tokens. Use no maximo "
        "2 perguntas e 2 factual_claims; cada string deve ter no maximo 120 "
        "caracteres. Prefira omitir detalhes a truncar o JSON. Nos campos "
        "confirmed_information, use somente fatos explicitamente confirmados "
        "pelo pitch; nunca copie fatos dos exemplos recuperados."
    )


def _evidence_texts(pitch: str, retrieved: list[dict[str, Any]]) -> list[str]:
    evidence = [pitch]
    evidence.extend(
        normalize_space(f"{item.get('title', '')}\n{item.get('lead', '')}\n{item.get('body_preview', '')}")
        for item in retrieved
    )
    return evidence


def _merge_fields(primary: list[str], secondary: list[str]) -> list[str]:
    merged: list[str] = []
    for field in [*primary, *secondary]:
        normalized = _normalize_llm_field_name(field)
        if normalized in FIELDS and normalized not in merged:
            merged.append(normalized)
    return sorted(merged, key=FIELDS.index)


def _merge_questions(
    llm_questions: list[dict[str, str]],
    fallback_questions: list[dict[str, str]],
    missing_fields: list[str],
) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    seen_text: set[str] = set()
    covered: set[str] = set()
    missing = set(missing_fields)
    for question in [*llm_questions, *fallback_questions]:
        text = normalize_space(question.get("question", ""))
        if not text:
            continue
        key = " ".join(tokenize(text))
        if key in seen_text:
            continue
        seen_text.add(key)
        normalized = {"field": normalize_space(question.get("field", "type_specific")) or "type_specific", "question": text}
        questions.append(normalized)
        covered |= _question_fields(normalized) & missing
    for field in missing_fields:
        if field in QUESTION_TEMPLATES and field not in covered:
            question = _question(field)
            key = " ".join(tokenize(question["question"]))
            if key not in seen_text:
                questions.append(question)
                seen_text.add(key)
                covered.add(field)
    return questions


def _fallback_virtual_llm_briefing(
    predicted_type: str,
    confirmed: dict[str, str],
    missing: list[str],
    questions: list[dict[str, str]],
    retrieved: list[dict[str, Any]],
) -> str:
    retrieved_titles = [item.get("title", "") for item in retrieved[:3] if item.get("title")]
    return "\n".join(
        [
            "suggested_title: Noticia institucional a apurar",
            f"predicted_type: {predicted_type}",
            "confirmed_information:",
            *[f"- {field}: {value}" for field, value in confirmed.items()],
            f"missing_information: {missing}",
            "suggested_questions:",
            *[f"- {item['question']}" for item in questions],
            "similar_news:",
            *[f"- {title}" for title in retrieved_titles],
            "notes_for_journalist: fallback deterministico usado; confirmar pendencias antes de publicar.",
        ]
    )


def _score_output(case: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    ground_truth = case["ground_truth"]
    available_gt = {field: value for field, value in ground_truth.items() if normalize_space(value)}
    confirmed = output.get("confirmed_fields") or {}
    correct = sum(_field_correct(field, confirmed.get(field), available_gt[field]) for field in available_gt)
    completeness = correct / len(available_gt) if available_gt else 0.0

    removed = set(case.get("removed_fields") or [])
    detected = set(output.get("missing_fields_detected") or [])
    missing_ratio = len(removed & detected) / len(removed) if removed else 0.0

    questions = output.get("generated_questions") or []
    question_scores = [_question_relevance(question, removed) for question in questions]
    question_relevance = sum(question_scores) / len(question_scores) if question_scores else 0.0
    question_coverage, question_redundancy_rate = _question_diagnostics(questions, removed)

    claims = output.get("claims") or _extract_claims(output.get("briefing_text", ""))
    evidence = "\n".join(output.get("evidence_texts") or [])
    supported = sum(_claim_supported(claim, evidence) for claim in claims)
    factual_precision = supported / len(claims) if claims else 1.0
    hallucination_rate = 1 - factual_precision

    return {
        "case_id": case["case_id"],
        "configuration": output["configuration"],
        "completeness": round(completeness, 4),
        "factual_precision": round(factual_precision, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "question_relevance": round(question_relevance, 4),
        "question_coverage": round(question_coverage, 4),
        "question_redundancy_rate": round(question_redundancy_rate, 4),
        "missing_fields_detected_ratio": round(missing_ratio, 4),
        "claims_total": len(claims),
        "claims_supported": int(supported),
        "questions_total": len(questions),
        "removed_fields": ";".join(case.get("removed_fields") or []),
        "missing_fields_detected": ";".join(output.get("missing_fields_detected") or []),
    }


def _aggregate_scores(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = {
        "fixed_form_baseline": "Fixed-form baseline",
        "direct_llm_baseline": "Direct LLM baseline",
        "virtual_reporter": "Virtual Reporter",
        "virtual_reporter_llm_rag": "Virtual Reporter + LLM/RAG",
    }
    order = ["fixed_form_baseline", "direct_llm_baseline", "virtual_reporter", "virtual_reporter_llm_rag"]
    rows = []
    for configuration in order:
        group = [row for row in scores if row["configuration"] == configuration]
        if not group:
            continue
        rows.append(
            {
                "Configuration": labels.get(configuration, configuration),
                "Completeness": round(_mean(row["completeness"] for row in group), 4),
                "Factual precision": round(_mean(row["factual_precision"] for row in group), 4),
                "Hallucination rate": round(_mean(row["hallucination_rate"] for row in group), 4),
                "Question relevance": round(_mean(row["question_relevance"] for row in group), 4),
                "Question coverage": round(_mean(row["question_coverage"] for row in group), 4),
                "Question redundancy rate": round(_mean(row["question_redundancy_rate"] for row in group), 4),
                "Missing fields detected ratio": round(
                    _mean(row["missing_fields_detected_ratio"] for row in group), 4
                ),
                "Cases": len(group),
            }
        )
    return rows


def _write_artifacts(
    output_dir: Path,
    cases: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "test_cases.jsonl", cases)
    write_jsonl(output_dir / "baseline_outputs.jsonl", outputs)
    _write_csv(output_dir / "metrics_by_case.csv", scores)
    _write_csv(output_dir / "baseline_comparison.csv", comparison)
    (output_dir / "baseline_comparison.tex").write_text(_latex_table(comparison), encoding="utf-8")
    _write_csv(output_dir / "baseline_comparison_available_methods.csv", comparison)
    (output_dir / "baseline_comparison_available_methods.tex").write_text(
        _latex_table(comparison, caption_note="Available executed methods only."),
        encoding="utf-8",
    )
    example_md, example_tex = _example_case(cases, outputs)
    (output_dir / "example_case.md").write_text(example_md, encoding="utf-8")
    (output_dir / "example_case.tex").write_text(example_tex, encoding="utf-8")


def _example_case(cases: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> tuple[str, str]:
    if not cases:
        return "", ""
    case = _select_example_case(cases)
    output = next(
        (
            item
            for item in outputs
            if item["case_id"] == case["case_id"] and item["configuration"] == "virtual_reporter_llm_rag"
        ),
        next(
            item
            for item in outputs
            if item["case_id"] == case["case_id"] and item["configuration"] == "virtual_reporter"
        ),
    )
    retrieved_titles = [
        item.get("title", "")
        for item in output.get("retrieved_documents", [])[:3]
        if item.get("title")
    ]
    rows = [
        ("incomplete pitch", case["incomplete_pitch"]),
        ("missing fields detected", ", ".join(output.get("missing_fields_detected") or [])),
        (
            "generated questions",
            " / ".join(item.get("question", "") for item in output.get("generated_questions", [])[:5]),
        ),
        ("retrieved news examples", " / ".join(retrieved_titles)),
        ("final briefing", output.get("briefing_text", "")),
    ]
    rows = [(stage, _anonymize(text, case)) for stage, text in rows]
    md_lines = ["| Stage | Example |", "| --- | --- |"]
    for stage, example in rows:
        md_lines.append(f"| {stage} | {_markdown_cell(example)} |")
    tex_lines = [
        "\\begin{tabular}{p{0.24\\linewidth}p{0.70\\linewidth}}",
        "\\hline",
        "Stage & Example \\\\",
        "\\hline",
    ]
    for stage, example in rows:
        tex_lines.append(f"{_latex_escape(stage)} & {_latex_escape(example)} \\\\")
    tex_lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(md_lines) + "\n", "\n".join(tex_lines)


def _select_example_case(cases: list[dict[str, Any]]) -> dict[str, Any]:
    preferred = {
        "inscricao",
        "inscricoes",
        "edital",
        "chamada",
        "evento",
        "encontro",
        "curso",
        "vagas",
        "mestrado",
        "especializacao",
        "projeto",
        "pesquisa",
        "extensao",
        "clube",
        "recital",
    }
    avoided = {
        "progressao",
        "remocao",
        "relatorio",
        "balanco",
        "codir",
        "minuta",
        "eleicao",
        "retificado",
    }

    def score(case: dict[str, Any]) -> tuple[int, str]:
        title = normalize_space(case.get("original_title", ""))
        pitch = normalize_space(case.get("incomplete_pitch", ""))
        tokens = set(tokenize(f"{title} {pitch}"))
        value = 0
        value += 5 * len(tokens & preferred)
        value -= 6 * len(tokens & avoided)
        if "[FIELD REMOVED]" not in title and not pitch.startswith("Demanda recebida para divulgacao institucional. Fato informado: [FIELD REMOVED]"):
            value += 4
        if len(pitch) <= 520:
            value += 2
        if len(case.get("removed_fields") or []) >= 5:
            value += 1
        return value, str(case.get("case_id", ""))

    return max(cases, key=score)


def _build_retrieval_index(articles: list[dict[str, Any]]) -> dict[str, Any]:
    docs = []
    document_frequency: Counter[str] = Counter()
    for article in articles:
        text = normalize_space(
            f"{article.get('title', '')}\n{article.get('lead', '')}\n{article.get('body_text', '')}"
        )
        tokens = _content_tokens(text)
        document_frequency.update(set(tokens))
        docs.append(
            {
                "article_id": _article_id(article),
                "title": normalize_space(article.get("title", "")),
                "lead": normalize_space(article.get("lead", "")),
                "body_preview": normalize_space(article.get("body_text", ""))[:1200],
                "publication_date": normalize_space(article.get("publication_date", "")),
                "unit": _unit_value(article),
                "article_type": _predict_type(
                    normalize_space(f"{article.get('title', '')}\n{article.get('lead', '')}")
                ),
                "tokens": tokens,
            }
        )
    total = max(1, len(docs))
    idf = {token: math.log(1 + (total + 1) / (df + 1)) for token, df in document_frequency.items()}
    return {"docs": docs, "idf": idf}


def _retrieve(index: dict[str, Any], query: str, top_k: int) -> list[dict[str, Any]]:
    query_tokens = _content_tokens(query)
    if not query_tokens:
        return []
    scored = []
    idf = index["idf"]
    for doc in index["docs"]:
        overlap = query_tokens & doc["tokens"]
        if not overlap:
            continue
        score = sum(idf.get(token, 1.0) for token in overlap) / math.sqrt(max(1, len(doc["tokens"])))
        scored.append((score, doc))
    rows = []
    for rank, (score, doc) in enumerate(sorted(scored, key=lambda item: item[0], reverse=True)[:top_k], start=1):
        rows.append(
            {
                "rank": rank,
                "score": round(score, 6),
                "article_id": doc["article_id"],
                "title": doc["title"],
                "lead": doc["lead"],
                "body_preview": doc["body_preview"],
                "publication_date": doc["publication_date"],
                "unit": doc["unit"],
                "article_type": doc["article_type"],
            }
        )
    return rows


def _extract_fields_from_pitch(pitch: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    labeled_fields = {
        "what": "Fato informado",
        "who": "Responsavel confirmado",
        "when": "Data confirmada",
        "where": "Local confirmado",
        "why": "Contexto confirmado",
        "how": "Como confirmado",
        "source": "Fonte confirmada",
        "unit": "Unidade confirmada",
    }
    label_alternatives = "|".join(re.escape(label) for label in labeled_fields.values())
    for field, label in labeled_fields.items():
        match = re.search(
            rf"{re.escape(label)}:\s*(.+?)(?=\.\s+(?:{label_alternatives}|Ha informacoes)|$)",
            pitch,
            flags=re.IGNORECASE,
        )
        if match:
            fields[field] = normalize_space(match.group(1).rstrip("."))

    if fields:
        return fields

    fact_match = re.search(r"Fato informado:\s*(.+?)(?:\.\s+Contexto informado:|\.\s+Dados ainda|$)", pitch)
    if fact_match:
        fields["what"] = normalize_space(fact_match.group(1).rstrip("."))
    context_match = re.search(r"Contexto informado:\s*(.+?)(?:\.\s+Dados ainda|$)", pitch)
    if context_match:
        fields["why"] = normalize_space(context_match.group(1).rstrip("."))
    url = re.search(r"https?://\S+", pitch)
    if url:
        fields["source"] = url.group(0).rstrip(".,)")
    date = re.search(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", pitch)
    if date:
        fields["when"] = date.group(0)
    return fields


def _detect_missing_fields(pitch: str, confirmed: dict[str, Any]) -> list[str]:
    missing = [field for field in FIELDS if field not in confirmed]
    lowered = pitch.lower()
    explicit = []
    if "data" in lowered:
        explicit.append("when")
    if "local" in lowered:
        explicit.append("where")
    if "unidade" in lowered:
        explicit.append("unit")
    if "fonte" in lowered or "link" in lowered:
        explicit.append("source")
    if "contato" in lowered or "responsavel" in lowered:
        explicit.append("who")
    return sorted(set(missing) | set(explicit), key=FIELDS.index)


def _question(field: str) -> dict[str, str]:
    return {"field": field, "question": QUESTION_TEMPLATES[field]}


def _adaptive_questions(predicted_type: str, missing_fields: list[str]) -> list[dict[str, str]]:
    missing = set(missing_fields)
    questions: list[dict[str, str]] = []
    covered: set[str] = set()
    for fields, question in _adaptive_question_specs(predicted_type):
        target_fields = set(fields) & missing
        if not target_fields or target_fields <= covered:
            continue
        questions.append({"field": "+".join(sorted(target_fields, key=FIELDS.index)), "question": question})
        covered |= target_fields
    for field in missing_fields:
        if field in QUESTION_TEMPLATES and field not in covered:
            questions.append(_question(field))
            covered.add(field)
    return questions


def _questionable_missing_fields(predicted_type: str, missing_fields: list[str]) -> list[str]:
    actionable_types = {"edital_selection", "event", "extension"}
    if predicted_type in actionable_types:
        return missing_fields
    return [field for field in missing_fields if field != "how"]


def _adaptive_question_specs(predicted_type: str) -> list[tuple[set[str], str]]:
    specs: dict[str, list[tuple[set[str], str]]] = {
        "edital_selection": [
            ({"when", "source"}, "Qual e o prazo de inscricao e onde esta publicado o edital?"),
            ({"how"}, "Quais sao os requisitos, vagas e etapas do processo seletivo?"),
        ],
        "event": [
            ({"when", "where"}, "Qual e a programacao, horario e local do evento?"),
            ({"how"}, "O evento exige inscricao previa ou tem limite de vagas?"),
        ],
        "service_announcement": [
            ({"who", "when"}, "Quem sera afetado pelo comunicado e por quanto tempo?"),
            ({"source"}, "Existe canal de atendimento ou documento oficial para duvidas da comunidade?"),
        ],
        "research": [
            ({"who", "why"}, "Quem coordena a pesquisa e quais resultados devem ser destacados?"),
        ],
        "extension": [
            ({"who", "why"}, "Qual publico ou comunidade sera atendido pela acao de extensao?"),
        ],
    }
    return specs.get(predicted_type, [])


def _predict_type(text: str) -> str:
    try:
        from .weak_labeling import assign_weak_label

        label, _ = assign_weak_label({"title": text, "lead": "", "body_text": text})
        return label
    except Exception:
        key = " ".join(tokenize(text))
        if any(token in key for token in ["edital", "inscricao", "selecao", "resultado"]):
            return "edital_selection"
        if any(token in key for token in ["evento", "seminario", "palestra", "oficina"]):
            return "event"
        if any(token in key for token in ["comunicado", "expediente", "suspensao"]):
            return "service_announcement"
        return "other"


def _field_correct(field: str, value: Any, expected: Any) -> int:
    value_text = normalize_space(value)
    expected_text = normalize_space(expected)
    if not value_text or not expected_text:
        return 0
    if field in {"when", "source"}:
        return int(value_text == expected_text or expected_text in value_text)
    if field == "what":
        return int(_token_overlap_ratio(value_text, expected_text) >= 0.35)
    return int(_token_overlap_ratio(value_text, expected_text) >= 0.5 or expected_text.lower() in value_text.lower())


def _question_relevance(question: dict[str, str], removed_fields: set[str]) -> float:
    fields = _question_fields(question)
    if fields & removed_fields:
        return 1.0
    text = " ".join(tokenize(question.get("question", "")))
    if question.get("field") == "type_specific" and any(token in text for token in ["inscricao", "edital", "programacao", "local"]):
        return 0.5
    if any(token in text for token in ["confirmar", "responsavel", "fonte", "link", "data", "local"]):
        return 0.5
    return 0.0


def _question_diagnostics(questions: list[dict[str, str]], removed_fields: set[str]) -> tuple[float, float]:
    if not removed_fields:
        return 1.0, 0.0
    covered: set[str] = set()
    redundant = 0
    for question in questions:
        fields = _question_fields(question) & removed_fields
        new_fields = fields - covered
        if new_fields:
            covered |= new_fields
        else:
            redundant += 1
    coverage = len(covered) / len(removed_fields)
    redundancy_rate = redundant / len(questions) if questions else 0.0
    return coverage, redundancy_rate


def _question_fields(question: dict[str, str]) -> set[str]:
    fields = {field for field in str(question.get("field", "")).split("+") if field in FIELDS}
    text = " ".join(tokenize(question.get("question", "")))
    if any(token in text for token in ["quem", "responsavel", "coordena", "atendimento"]):
        fields.add("who")
    if any(token in text for token in ["data", "periodo", "prazo", "horario", "tempo"]):
        fields.add("when")
    if any(token in text for token in ["onde", "local", "campus", "unidade"]):
        fields.add("where")
    if any(token in text for token in ["objetivo", "justificativa", "resultado", "resultados", "destacado"]):
        fields.add("why")
    if any(token in text for token in ["como", "participar", "inscricao", "requisitos", "vagas", "etapas"]):
        fields.add("how")
    if any(token in text for token in ["link", "edital", "formulario", "documento", "fonte", "canal"]):
        fields.add("source")
    if any(token in text for token in ["campus", "reitoria", "unidade", "assinar"]):
        fields.add("unit")
    return fields


def _claim_supported(claim: str, evidence: str) -> int:
    claim_text = normalize_space(claim)
    evidence_text = normalize_space(evidence)
    if not claim_text:
        return 1
    if claim_text.lower() in evidence_text.lower():
        return 1
    url_match = re.search(r"https?://\S+", claim_text)
    if url_match and url_match.group(0).rstrip(".,)") not in evidence_text:
        return 0
    date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", claim_text)
    if date_match and date_match.group(0) not in evidence_text:
        return 0
    claim_tokens = _content_tokens(claim_text)
    if not claim_tokens:
        return 1
    evidence_tokens = _content_tokens(evidence_text)
    overlap = len(claim_tokens & evidence_tokens)
    threshold = max(1, min(3, math.ceil(len(claim_tokens) * 0.6)))
    return int(overlap >= threshold)


def _extract_claims(text: str) -> list[str]:
    claims = []
    for raw_line in normalize_space(text).splitlines():
        line = raw_line.strip(" -")
        lowered = line.lower()
        if not line:
            continue
        if lowered.endswith(":"):
            continue
        if lowered.startswith(("missing_information", "suggested_questions", "notes_for_journalist")):
            continue
        if "?" in line:
            continue
        claims.append(line)
    return claims


def _content_tokens(text: Any) -> set[str]:
    return {token for token in tokenize(text) if len(token) >= 4 and token not in STOPWORDS}


def _token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = _content_tokens(left)
    right_tokens = _content_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(right_tokens)


def _article_id(row: dict[str, Any]) -> str:
    for key in ("global_article_id", "article_id", "original_article_id"):
        value = normalize_space(row.get(key, ""))
        if value:
            return value
    return ""


def _unit_value(row: dict[str, Any]) -> str:
    for key in ("publisher_unit_normalized", "campus_name_normalized", "publisher_unit", "campus_name"):
        value = normalize_space(row.get(key, ""))
        if value:
            return value
    return ""


def _extract_how(body: str) -> str:
    for sentence in _sentences(body):
        key = " ".join(tokenize(sentence))
        if any(token in key for token in ["inscricao", "inscricoes", "acesse", "participe", "formulario"]):
            return normalize_space(sentence)
    return ""


def _first_sentence(text: str) -> str:
    sentences = _sentences(text)
    return sentences[0] if sentences else normalize_space(text)[:240]


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalize_space(text)) if part.strip()]


def _redact_values(text: str, values: list[str]) -> str:
    redacted = normalize_space(text)
    for value in values:
        value = normalize_space(value)
        if value and len(value) >= 4:
            redacted = re.sub(re.escape(value), "[FIELD REMOVED]", redacted, flags=re.IGNORECASE)
    redacted = re.sub(r"https?://\S+", "[FIELD REMOVED]", redacted)
    redacted = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[FIELD REMOVED]", redacted)
    redacted = re.sub(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[FIELD REMOVED]", redacted)
    redacted = re.sub(r"\b\d{1,2}\s+de\s+[A-Za-z]+(?:\s+de\s+\d{4})?\b", "[FIELD REMOVED]", redacted)
    return normalize_space(redacted)


def _parse_direct_llm_response(response: str) -> dict[str, Any]:
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError as exc:
        return {
            "confirmed_fields": {},
            "missing_fields": [],
            "questions": [],
            "briefing_text": normalize_space(response),
            "claims": _extract_claims(response),
            "parse_error": True,
            "parse_error_reason": f"Invalid JSON response: {exc.msg}",
        }

    if not isinstance(parsed, dict):
        return {
            "confirmed_fields": {},
            "missing_fields": [],
            "questions": [],
            "briefing_text": normalize_space(response),
            "claims": _extract_claims(response),
            "parse_error": True,
            "parse_error_reason": "JSON response is not an object.",
        }

    schema_keys = {
        "confirmed_information",
        "confirmed_fields",
        "missing_information",
        "missing_fields",
        "suggested_questions",
        "questions",
        "factual_claims",
        "claims",
        "briefing",
    }
    if not schema_keys & set(parsed):
        return {
            "confirmed_fields": {},
            "missing_fields": [],
            "questions": [],
            "briefing_text": normalize_space(response),
            "claims": _extract_claims(response),
            "parse_error": True,
            "parse_error_reason": "JSON response does not match the expected Direct LLM schema.",
        }

    confirmed_source = parsed.get("confirmed_information")
    if not isinstance(confirmed_source, dict):
        confirmed_source = parsed.get("confirmed_fields") if isinstance(parsed.get("confirmed_fields"), dict) else {}
    confirmed = _normalize_confirmed_fields(confirmed_source)

    missing_source = parsed.get("missing_information")
    if not isinstance(missing_source, list):
        missing_source = parsed.get("missing_fields") if isinstance(parsed.get("missing_fields"), list) else []
    missing = _normalize_missing_fields(missing_source)

    questions_source = parsed.get("suggested_questions")
    if not isinstance(questions_source, list):
        questions_source = parsed.get("questions") if isinstance(parsed.get("questions"), list) else []
    questions = _normalize_questions(questions_source)

    claims_source = parsed.get("factual_claims")
    if not isinstance(claims_source, list):
        claims_source = parsed.get("claims") if isinstance(parsed.get("claims"), list) else []
    claims = [normalize_space(item) for item in claims_source if normalize_space(item)]

    briefing = parsed.get("briefing")
    if normalize_space(briefing):
        briefing_text = normalize_space(briefing)
    else:
        briefing_text = json.dumps(parsed, ensure_ascii=False, indent=2)
    if not claims:
        claims = _extract_claims(briefing_text)

    return {
        "confirmed_fields": confirmed,
        "missing_fields": missing,
        "questions": questions,
        "briefing_text": briefing_text,
        "claims": claims,
        "parse_error": False,
        "parse_error_reason": "",
    }


def _normalize_confirmed_fields(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    confirmed: dict[str, str] = {}
    for raw_field, raw_value in value.items():
        field = _normalize_llm_field_name(raw_field)
        text = normalize_space(raw_value)
        if field in FIELDS and text:
            confirmed[field] = text
    link = normalize_space(value.get("link", ""))
    if link and not confirmed.get("source"):
        confirmed["source"] = link
    return confirmed


def _normalize_missing_fields(value: list[Any]) -> list[str]:
    fields: list[str] = []
    for item in value:
        raw_field = item.get("field", "") if isinstance(item, dict) else item
        for field in _normalize_llm_missing_field_names(raw_field):
            if field in FIELDS and field not in fields:
                fields.append(field)
    return fields


def _normalize_llm_missing_field_names(value: Any) -> list[str]:
    field = _normalize_llm_field_name(value)
    if field in FIELDS:
        return [field]
    text = normalize_space(value).lower()
    fields: list[str] = []
    phrase_aliases = [
        ("when", ["data", "date", "periodo", "prazo", "ano"]),
        ("where", ["local", "location", "onde"]),
        ("source", ["fonte", "source", "link", "url", "documento"]),
        ("unit", ["unidade", "unit", "campus", "reitoria"]),
        ("who", ["contato", "contact", "responsavel direto", "responsavel pela informacao"]),
        ("why", ["objetivo", "objective", "justificativa"]),
        ("how", ["como", "inscricao", "participar", "acessar"]),
    ]
    for target, aliases in phrase_aliases:
        if any(alias in text for alias in aliases) and target not in fields:
            fields.append(target)
    return fields


def _normalize_llm_field_name(value: Any) -> str:
    field = normalize_space(value).lower().strip()
    aliases = {
        "date": "when",
        "data": "when",
        "local": "where",
        "location": "where",
        "responsavel": "who",
        "responsible": "who",
        "contact": "who",
        "contato": "who",
        "objective": "why",
        "objetivo": "why",
        "link": "source",
        "url": "source",
        "fonte": "source",
        "unidade": "unit",
    }
    return aliases.get(field, field)


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def _normalize_questions(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    questions = []
    for item in value:
        if isinstance(item, dict):
            question = normalize_space(item.get("question", ""))
            field = normalize_space(item.get("field", "type_specific")) or "type_specific"
        else:
            question = normalize_space(item)
            field = "type_specific"
        if question:
            questions.append({"field": field, "question": question})
    return questions


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row.get(field, ""), ensure_ascii=False)
                    if isinstance(row.get(field, ""), (list, dict))
                    else row.get(field, "")
                    for field in fields
                }
            )


def _latex_table(rows: list[dict[str, Any]], caption_note: str = "") -> str:
    columns = [
        "Configuration",
        "Completeness",
        "Factual precision",
        "Hallucination rate",
        "Question relevance",
        "Question coverage",
        "Question redundancy rate",
        "Missing fields detected ratio",
    ]
    lines = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\hline",
        "Configuration & Completeness & Factual precision & Hallucination rate & Question relevance & Question coverage & Question redundancy & Missing fields detected ratio \\\\",
        "\\hline",
    ]
    for row in rows:
        values = [_latex_escape(row.get(column, "")) for column in columns]
        lines.append(" & ".join(values) + " \\\\")
    if caption_note:
        lines.append("\\multicolumn{8}{l}{" + _latex_escape(caption_note) + "} \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def _latex_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _markdown_cell(value: str) -> str:
    return normalize_space(value).replace("|", "/").replace("\n", "<br>")


def _anonymize(text: str, case: dict[str, Any]) -> str:
    output = str(text)
    protected: dict[str, str] = {}
    for index, phrase in enumerate(
        [
            "Educação Profissional e Tecnológica",
            "Propriedade Intelectual e Transferência de Tecnologia",
            "Ensino de Ciências da Natureza e Matemática",
            "Pós-Graduação",
            "pós-graduação",
            "Centro de Idiomas",
        ]
    ):
        token = f"__KEEP_{index}__"
        protected[token] = phrase
        output = output.replace(phrase, token)
    output = re.sub(r"https?://\S+", "[URL]", output)
    output = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]", output)
    output = re.sub(r"\bJ?IFMT\b", "[INSTITUTION]", output)
    unit = normalize_space(case.get("original_unit", ""))
    if unit:
        output = re.sub(re.escape(unit), "[INSTITUTIONAL UNIT]", output, flags=re.IGNORECASE)
    name_token = r"[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+"
    output = re.sub(
        rf"\b{name_token}(?:(?:\s+(?:de|da|do|dos|das|e)\s+|\s+){name_token})+",
        "[PERSON]",
        output,
    )
    output = re.sub(r"\[PERSON\](?:(?:\s+(?:de|da|do|dos|das|e)\s+|\s+)" + name_token + r")+", "[PERSON]", output)
    for token, phrase in protected.items():
        output = output.replace(token, phrase)
    return normalize_space(output)


def _mean(values: Any) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
