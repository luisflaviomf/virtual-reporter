from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from .io_utils import normalize_space


SAMPLING_VERSION = "stratified-v1"
MASKING_VERSION = "balanced-mask-v1"
FIELDS = ("what", "who", "when", "where", "why", "how", "source", "unit")
MACRO_GROUPS = (
    "edital_selection",
    "event",
    "teaching",
    "research",
    "extension",
    "other",
)
PRIMARY_GROUPS = set(MACRO_GROUPS[:-1])

FIELD_LABELS = {
    "what": "Fato informado",
    "who": "Responsavel confirmado",
    "when": "Data confirmada",
    "where": "Local confirmado",
    "why": "Contexto confirmado",
    "how": "Como confirmado",
    "source": "Fonte confirmada",
    "unit": "Unidade confirmada",
}


@dataclass(frozen=True)
class SelectedArticle:
    article_id: str
    original_label: str
    macro_group: str
    article: dict[str, Any]


def macro_group(label: str) -> str:
    return label if label in PRIMARY_GROUPS else "other"


def article_id(article: dict[str, Any]) -> str:
    for field in ("article_id", "canonical_article_id", "record_id", "id"):
        value = normalize_space(article.get(field, ""))
        if value:
            return value
    return normalize_space(article.get("original_url", "") or article.get("canonical_url", ""))


def select_stratified_articles(
    articles: list[dict[str, Any]],
    pilot_article_ids: set[str],
    per_group: int,
    seed: int,
    labeler: Callable[[dict[str, Any]], str],
) -> list[SelectedArticle]:
    grouped: dict[str, list[SelectedArticle]] = {name: [] for name in MACRO_GROUPS}
    for article in articles:
        current_id = article_id(article)
        if not current_id or current_id in pilot_article_ids:
            continue
        original_label = normalize_space(labeler(article)) or "other"
        assigned_group = macro_group(original_label)
        grouped[assigned_group].append(
            SelectedArticle(
                article_id=current_id,
                original_label=original_label,
                macro_group=assigned_group,
                article=article,
            )
        )

    selected: list[SelectedArticle] = []
    for group in MACRO_GROUPS:
        pool = sorted(grouped[group], key=lambda item: item.article_id)
        random.Random(f"{seed}:{group}").shuffle(pool)
        if len(pool) < per_group:
            raise ValueError(f"{group} requires {per_group} articles, found {len(pool)}")
        selected.extend(pool[:per_group])
    return selected


def make_masked_case(
    article: dict[str, Any],
    number: int,
    assigned_group: str,
    original_label: str,
    seed: int,
    requested_depth: int,
) -> dict[str, Any]:
    ground_truth = build_ground_truth(article)
    candidates = [
        field
        for field in FIELDS
        if field != "what" and normalize_space(ground_truth.get(field, ""))
    ]
    rng = random.Random(_stable_seed(seed, article_id(article), number))
    rng.shuffle(candidates)
    removed = set(candidates[: min(requested_depth, len(candidates))])

    removed_values = {
        _comparable_value(ground_truth[field])
        for field in removed
        if _comparable_value(ground_truth[field])
    }
    for field in candidates:
        if _comparable_value(ground_truth[field]) in removed_values:
            removed.add(field)

    ordered_removed = [field for field in FIELDS if field in removed]
    literal_removed_values = _unique_values(
        [normalize_space(ground_truth[field]) for field in ordered_removed]
    )
    confirmed: dict[str, str] = {}
    for field in FIELDS:
        if field in removed:
            continue
        value = normalize_space(ground_truth.get(field, ""))
        if not value:
            continue
        confirmed[field] = _redact_values(value, literal_removed_values)

    pitch = serialize_pitch(confirmed)
    return {
        "case_id": f"case_{number:04d}",
        "article_id": article_id(article),
        "original_title": normalize_space(article.get("title", "")),
        "original_url": normalize_space(
            article.get("original_url", "") or article.get("canonical_url", "")
        ),
        "original_unit": _unit_value(article),
        "original_date": normalize_space(article.get("publication_date", "")),
        "original_predicted_label": original_label,
        "macro_group": assigned_group,
        "requested_mask_depth": requested_depth,
        "mask_depth": len(ordered_removed),
        "incomplete_pitch": pitch,
        "removed_fields": ordered_removed,
        "available_fields_in_pitch": confirmed,
        "ground_truth": ground_truth,
    }


def build_ground_truth(article: dict[str, Any]) -> dict[str, str]:
    title = normalize_space(article.get("title", ""))
    body = normalize_space(article.get("body_text", ""))
    lead = normalize_space(article.get("lead", "")) or _first_sentence(body)
    unit = _unit_value(article)
    source = normalize_space(
        article.get("original_url", "") or article.get("canonical_url", "")
    )
    who = normalize_space(article.get("author_name", "")) or unit
    return {
        "what": title,
        "who": who,
        "when": normalize_space(article.get("publication_date", "")),
        "where": unit,
        "why": _first_sentence(lead or body),
        "how": _extract_how(body),
        "source": source,
        "unit": unit,
    }


def serialize_pitch(confirmed_fields: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in FIELDS:
        value = normalize_space(confirmed_fields.get(field, ""))
        if value:
            parts.append(f"{FIELD_LABELS[field]}: {value}.")
    parts.append("Ha informacoes ainda pendentes de confirmacao.")
    return normalize_space(" ".join(parts))


def sampling_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    group_counts = Counter(str(case.get("macro_group", "")) for case in cases)
    depth_counts = Counter(str(case.get("mask_depth", "")) for case in cases)
    field_counts: Counter[str] = Counter()
    for case in cases:
        field_counts.update(str(field) for field in case.get("removed_fields", []))
    return {
        "case_count": len(cases),
        "macro_groups": dict(sorted(group_counts.items())),
        "mask_depths": dict(sorted(depth_counts.items())),
        "removed_fields": {
            field: field_counts.get(field, 0)
            for field in FIELDS
        },
    }


def _stable_seed(seed: int, current_article_id: str, number: int) -> int:
    payload = f"{seed}:{current_article_id}:{number}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _unit_value(article: dict[str, Any]) -> str:
    for field in (
        "normalized_unit",
        "campus_name",
        "publisher_unit",
        "source_name",
    ):
        value = normalize_space(article.get(field, ""))
        if value:
            return value
    return ""


def _first_sentence(text: str) -> str:
    normalized = normalize_space(text)
    if not normalized:
        return ""
    return normalize_space(re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0])


def _extract_how(body: str) -> str:
    for sentence in re.split(r"(?<=[.!?])\s+", normalize_space(body)):
        lowered = sentence.casefold()
        if any(token in lowered for token in ("inscri", "acess", "particip", "formulario", "formulário")):
            return normalize_space(sentence)
    return ""


def _comparable_value(value: Any) -> str:
    return normalize_space(value).casefold()


def _unique_values(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in sorted(values, key=len, reverse=True):
        key = value.casefold()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def _redact_values(text: str, values: list[str]) -> str:
    output = text
    for value in values:
        output = re.sub(re.escape(value), "[FIELD REMOVED]", output, flags=re.IGNORECASE)
    return normalize_space(output)
