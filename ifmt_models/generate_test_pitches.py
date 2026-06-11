from __future__ import annotations

from typing import Any

import pandas as pd

from .config import PATHS, RANDOM_STATE, ensure_dirs
from .io_utils import normalize_space, write_jsonl, write_table
from .load_dataset import load_labeled
from .retrieval_utils import article_id_from_row


def generate_test_pitches(
    pitch_count: int | None = None,
    quick: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    ensure_dirs()
    df = load_labeled().copy()
    target = pitch_count or (30 if quick else 100)
    df["_quality"] = (
        df["title"].fillna("").astype(str).str.len()
        + df["lead"].fillna("").astype(str).str.len()
        + df["body_text"].fillna("").astype(str).str.len().clip(upper=500)
    )
    df = df.sort_values("_quality", ascending=False)
    sample = _balanced_sample(df, min(target, len(df)))

    rows = []
    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        article_type = str(row.get("article_type_weak", "other") or "other")
        title = normalize_space(row.get("title", ""))
        lead = normalize_space(row.get("lead", ""))
        body = normalize_space(row.get("body_text", ""))
        unit = _first_nonempty(
            row.get("publisher_unit_normalized", ""),
            row.get("campus_name_normalized", ""),
            row.get("publisher_unit", ""),
            "IFMT",
        )
        pitch = _make_pitch(title, lead, article_type)
        expected = {
            "expected_who": unit or "IFMT",
            "expected_what": title,
            "expected_when": _empty_to_none(row.get("publication_date", "")),
            "expected_where": _empty_to_none(unit),
            "expected_why": _first_sentence(lead or body),
            "expected_how": None,
            "expected_source": _empty_to_none(row.get("original_url", "")),
            "expected_unit": _empty_to_none(unit),
        }
        missing_fields = [
            field.replace("expected_", "")
            for field, value in expected.items()
            if value is not None and field not in {"expected_what"}
        ]
        rows.append(
            {
                "case_id": f"pitch_{i:04d}",
                "original_article_id": article_id_from_row(row),
                "original_title": title,
                "article_type_weak": article_type,
                "input_pitch": pitch,
                "removed_fields": "date;local;responsavel;link;contato;objetivo;publico_alvo",
                "missing_fields_intended": missing_fields,
                **expected,
            }
        )

    write_jsonl(PATHS.results_dir / "test_pitches.jsonl", rows)
    write_table(PATHS.results_dir / "test_pitches.csv", pd.DataFrame(rows))
    return {
        "test_pitches": len(rows),
        "outputs": {
            "jsonl": str(PATHS.results_dir / "test_pitches.jsonl"),
            "csv": str(PATHS.results_dir / "test_pitches.csv"),
        },
    }


def _balanced_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    labels = list(df["article_type_weak"].fillna("other").value_counts().index)
    per_label = max(1, n // max(1, len(labels)))
    pieces = []
    used = set()
    for label in labels:
        group = df[df["article_type_weak"].fillna("other").eq(label)]
        sample = group.head(per_label)
        used.update(sample.index.tolist())
        pieces.append(sample)
    sampled = pd.concat(pieces)
    if len(sampled) < n:
        remaining = df[~df.index.isin(used)]
        sampled = pd.concat([sampled, remaining.head(n - len(sampled))])
    return sampled.head(n).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def _make_pitch(title: str, lead: str, article_type: str) -> str:
    base = title or _first_sentence(lead)
    if article_type == "edital_selection":
        return normalize_space(
            f"Queremos divulgar uma oportunidade institucional relacionada a {base}. "
            "A ação envolve inscrições ou seleção e precisa virar uma notícia objetiva para o público interessado."
        )
    if article_type == "event":
        return normalize_space(
            f"Queremos divulgar uma atividade do IFMT sobre {base}. "
            "A programação será voltada à comunidade acadêmica e precisa de uma notícia de chamada."
        )
    if article_type == "service_announcement":
        return normalize_space(
            f"Precisamos publicar um comunicado institucional sobre {base}. "
            "O texto deve orientar estudantes, servidores e comunidade externa."
        )
    return normalize_space(
        f"Queremos divulgar uma notícia institucional sobre {base}. "
        "A informação deve ser organizada em um briefing para a equipe de comunicação."
    )


def _first_sentence(text: str) -> str | None:
    text = normalize_space(text)
    if not text:
        return None
    for sep in [". ", "! ", "? "]:
        if sep in text:
            return text.split(sep, 1)[0].strip() + sep.strip()
    return text[:240]


def _empty_to_none(value: Any) -> str | None:
    value = normalize_space(value)
    return value or None


def _first_nonempty(*values: Any) -> str:
    for value in values:
        normalized = normalize_space(value)
        if normalized:
            return normalized
    return ""
