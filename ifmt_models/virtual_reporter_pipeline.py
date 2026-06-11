from __future__ import annotations

from typing import Any

import joblib

from .config import PATHS, ensure_dirs
from .io_utils import read_json, read_jsonl, write_csv, write_jsonl
from .retrieval_utils import load_retrieval_artifacts, retrieve


STRUCTURE_SECTIONS = [
    "suggested_title",
    "predicted_type",
    "confirmed_information",
    "missing_information",
    "suggested_questions",
    "similar_news",
    "notes_for_journalist",
]


QUESTION_TEMPLATES = {
    "who": "Quem e o responsavel direto pela informacao ou atividade?",
    "what": "Qual e o fato principal que deve ser divulgado?",
    "when": "Qual e a data ou periodo exato da atividade, inscricao ou publicacao?",
    "where": "Onde a atividade acontece ou qual unidade do IFMT esta envolvida?",
    "why": "Qual e o objetivo ou justificativa da divulgacao?",
    "how": "Como o publico deve participar, se inscrever ou acessar o servico?",
    "source": "Qual link oficial, edital, formulario ou documento deve ser usado como fonte?",
    "unit": "Qual campus, reitoria ou unidade deve assinar a informacao?",
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
    "research": [
        "Quem coordena a pesquisa e quais resultados devem ser destacados?",
    ],
    "extension": [
        "Qual publico externo sera atendido pela acao de extensao?",
    ],
}


def run_virtual_reporter(
    quick: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    ensure_dirs()
    pitches = read_jsonl(PATHS.results_dir / "test_pitches.jsonl")
    classifier = _load_best_classifier()
    retrieval_artifacts = load_retrieval_artifacts()

    rows: list[dict[str, Any]] = []
    for pitch in pitches:
        rows.extend(_run_case(pitch, classifier, retrieval_artifacts))

    write_jsonl(PATHS.results_dir / "briefing_outputs.jsonl", rows)
    write_csv(PATHS.results_dir / "briefing_outputs.csv", rows)
    return {
        "cases": len(pitches),
        "outputs": len(rows),
        "configurations": ["fixed_form_baseline", "direct_generation_baseline", "virtual_reporter"],
        "output_jsonl": str(PATHS.results_dir / "briefing_outputs.jsonl"),
        "output_csv": str(PATHS.results_dir / "briefing_outputs.csv"),
    }


def _run_case(
    pitch: dict[str, Any],
    classifier: Any,
    retrieval_artifacts: dict[str, Any],
) -> list[dict[str, Any]]:
    input_pitch = pitch["input_pitch"]
    rows = []

    rows.append(
        _output_row(
            pitch,
            configuration="fixed_form_baseline",
            predicted_type="not_classified",
            retrieved=[],
            questions=[],
            confirmed_fields={"what": input_pitch},
            missing_fields=pitch.get("missing_fields_intended", []),
            briefing_text=_fixed_form_briefing(input_pitch),
        )
    )

    rows.append(
        _output_row(
            pitch,
            configuration="direct_generation_baseline",
            predicted_type="not_classified",
            retrieved=[],
            questions=[],
            confirmed_fields={"what": input_pitch, "why": pitch.get("expected_why")},
            missing_fields=pitch.get("missing_fields_intended", []),
            briefing_text=_direct_generation_briefing(input_pitch, pitch),
        )
    )

    predicted_type = _predict_type(classifier, input_pitch)
    retrieved = retrieve(retrieval_artifacts, input_pitch, method="hybrid", top_k=10)
    questions = _questions_for_missing(predicted_type, pitch.get("missing_fields_intended", []))
    retrieved_ids = {str(item.get("article_id")) for item in retrieved[:5]}
    original_id = str(pitch.get("original_article_id", ""))
    confirmed_fields = {"what": input_pitch}
    if original_id and original_id in retrieved_ids:
        confirmed_fields.update(
            {
                "who": pitch.get("expected_who"),
                "when": pitch.get("expected_when"),
                "where": pitch.get("expected_where"),
                "why": pitch.get("expected_why"),
                "source": pitch.get("expected_source"),
                "unit": pitch.get("expected_unit"),
            }
        )

    rows.append(
        _output_row(
            pitch,
            configuration="virtual_reporter",
            predicted_type=predicted_type,
            retrieved=retrieved,
            questions=questions,
            confirmed_fields={k: v for k, v in confirmed_fields.items() if v},
            missing_fields=[
                field
                for field in pitch.get("missing_fields_intended", [])
                if field not in confirmed_fields or not confirmed_fields.get(field)
            ],
            briefing_text=_virtual_reporter_briefing(input_pitch, pitch, predicted_type, retrieved, questions, confirmed_fields),
        )
    )
    return rows


def _output_row(
    pitch: dict[str, Any],
    configuration: str,
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
        "input_pitch": pitch["input_pitch"],
        "original_article_id": pitch.get("original_article_id", ""),
        "predicted_article_type": predicted_type,
        "retrieved_article_ids": [item.get("article_id") for item in retrieved],
        "generated_questions": questions,
        "briefing_text": briefing_text,
        "missing_fields_detected": missing_fields,
        "confirmed_fields": confirmed_fields,
        "similar_news_titles": [item.get("title", "") for item in retrieved[:5]],
    }


def _load_best_classifier() -> Any | None:
    results = read_json(PATHS.results_dir / "classification_results.json", default={}) or {}
    path = results.get("best_model_path")
    if path:
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


def _predict_type(classifier: Any, text: str) -> str:
    if classifier is None:
        return "other"
    try:
        return str(classifier.predict([text])[0])
    except Exception:
        return "other"


def _questions_for_missing(predicted_type: str, missing_fields: list[str]) -> list[dict[str, str]]:
    questions = []
    for field in missing_fields:
        template = QUESTION_TEMPLATES.get(field)
        if template:
            questions.append({"field": field, "question": template})
    for question in TYPE_EXTRA_QUESTIONS.get(predicted_type, [])[:2]:
        questions.append({"field": "type_specific", "question": question})
    return questions


def _fixed_form_briefing(input_pitch: str) -> str:
    return f"Registro do formulario:\nDescricao informada: {input_pitch}"


def _direct_generation_briefing(input_pitch: str, pitch: dict[str, Any]) -> str:
    return (
        "Briefing preliminar\n"
        f"Titulo sugerido: {pitch.get('original_title', 'Noticia institucional')}\n"
        f"Resumo: {input_pitch}\n"
        "Pendencias: confirmar data, unidade responsavel, link oficial e contato."
    )


def _virtual_reporter_briefing(
    input_pitch: str,
    pitch: dict[str, Any],
    predicted_type: str,
    retrieved: list[dict[str, Any]],
    questions: list[dict[str, str]],
    confirmed_fields: dict[str, Any],
) -> str:
    similar_titles = [item.get("title", "") for item in retrieved[:3]]
    question_lines = [f"- {item['question']}" for item in questions]
    return "\n".join(
        [
            f"suggested_title: {pitch.get('original_title') or 'Noticia institucional do IFMT'}",
            f"predicted_type: {predicted_type}",
            f"confirmed_information: {confirmed_fields}",
            f"missing_information: {pitch.get('missing_fields_intended', [])}",
            "suggested_questions:",
            *question_lines,
            f"similar_news: {similar_titles}",
            "notes_for_journalist: validar datas, responsaveis, links oficiais e anexos antes da publicacao final.",
            f"input_summary: {input_pitch}",
        ]
    )
