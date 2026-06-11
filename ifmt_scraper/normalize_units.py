from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import DISCOVERY_SOURCES
from .crawl import NEWS_ARTICLE_FIELDS
from .finalize import build_dataset_report, build_quality_report_markdown
from .normalize import normalize_text, strip_accents

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - optional dependency fallback.
    fuzz = None


NORMALIZED_FIELDS = [
    "publisher_unit_raw",
    "campus_name_raw",
    "publisher_unit_normalized",
    "campus_name_normalized",
    "unit_normalization_status",
    "unit_normalization_score",
    "unit_normalization_source_value",
    "unit_normalization_note",
]

UNIT_REPORT_FIELDS = [
    "original_value",
    "original_value_key",
    "count_full",
    "count_canonical",
    "normalized_value",
    "campus_code",
    "status",
    "score",
    "example_article_id",
    "example_title",
    "example_url",
    "note",
]


@dataclass(frozen=True)
class UnitDefinition:
    canonical_name: str
    code: str
    aliases: tuple[str, ...] = ()


@dataclass
class MatchResult:
    unit: UnitDefinition | None
    status: str
    score: float
    source_value: str
    note: str = ""
    applied: bool = False


@dataclass
class NormalizeUnitsOptions:
    final_dir: Path = Path("data/output/final")
    full_input: Path | None = None
    canonical_input: Path | None = None
    in_place_standard: bool = False
    refresh_dataset_report: bool = True


CANONICAL_UNITS = [
    UnitDefinition("Reitoria", "reitoria", ("reitoria", "pro reitoria")),
    UnitDefinition("Campus Alta Floresta", "alf", ("alta floresta", "campus alta floresta")),
    UnitDefinition(
        "Campus Barra do Gar\u00e7as",
        "bag",
        ("barra do garcas", "barra do gar\u00e7as", "campus barra do garcas"),
    ),
    UnitDefinition(
        "Campus C\u00e1ceres",
        "cas",
        (
            "caceres",
            "c\u00e1ceres",
            "campus caceres",
            "campus c\u00e1ceres",
            "caceres prof olegario baldo",
            "c\u00e1ceres - prof. oleg\u00e1rio baldo",
            "prof olegario baldo",
        ),
    ),
    UnitDefinition(
        "Campus Campo Novo do Parecis",
        "cnp",
        ("campo novo", "campo novo do parecis", "campus campo novo do parecis"),
    ),
    UnitDefinition("Campus Confresa", "cfs", ("confresa", "campus confresa")),
    UnitDefinition(
        "Campus Cuiab\u00e1",
        "cba",
        (
            "cuiaba",
            "cuiab\u00e1",
            "campus cuiaba",
            "campus cuiab\u00e1",
            "cuiaba octayde",
            "cuiab\u00e1 octayde",
            "octayde",
            "octayde jorge da silva",
        ),
    ),
    UnitDefinition(
        "Campus Cuiab\u00e1 - Bela Vista",
        "blv",
        (
            "bela vista",
            "campus bela vista",
            "campus cuiaba bela vista",
            "campus cuiab\u00e1 - bela vista",
        ),
    ),
    UnitDefinition("Campus Ju\u00edna", "jna", ("juina", "ju\u00edna", "campus juina")),
    UnitDefinition(
        "Campus Pontes e Lacerda",
        "plc",
        ("pontes e lacerda", "campus pontes e lacerda"),
    ),
    UnitDefinition(
        "Campus Primavera do Leste",
        "pdl",
        ("primavera do leste", "campus primavera do leste"),
    ),
    UnitDefinition(
        "Campus Rondon\u00f3polis",
        "roo",
        ("rondonopolis", "rondon\u00f3polis", "campus rondonopolis"),
    ),
    UnitDefinition(
        "Campus S\u00e3o Vicente",
        "svc",
        ("sao vicente", "s\u00e3o vicente", "campus sao vicente"),
    ),
    UnitDefinition("Campus Sinop", "snp", ("sinop", "campus sinop", "campus avancado sinop")),
    UnitDefinition("Campus Sorriso", "srs", ("sorriso", "campus sorriso")),
    UnitDefinition(
        "Campus Tangar\u00e1 da Serra",
        "tga",
        ("tangara da serra", "tangar\u00e1 da serra", "campus tangara da serra"),
    ),
    UnitDefinition(
        "Campus V\u00e1rzea Grande",
        "vgd",
        ("varzea grande", "v\u00e1rzea grande", "campus varzea grande"),
    ),
    UnitDefinition(
        "Campus Lucas do Rio Verde",
        "lrv",
        (
            "lucas do rio verde",
            "campus lucas do rio verde",
            "campus avancado lucas do rio verde",
            "campus avancado de lucas do rio verde",
            "calrv",
            "lrv",
        ),
    ),
    UnitDefinition(
        "Centro de Refer\u00eancia de Jaciara",
        "jac",
        ("jaciara", "centro de referencia de jaciara"),
    ),
    UnitDefinition(
        "Centro de Refer\u00eancia de Campo Verde",
        "cvd",
        ("campo verde", "centro de referencia de campo verde", "campus campo verde"),
    ),
    UnitDefinition(
        "Centro de Refer\u00eancia de Canarana",
        "can",
        ("canarana", "centro de referencia de canarana"),
    ),
    UnitDefinition(
        "Centro de Refer\u00eancia de Parana\u00edta",
        "par",
        ("paranaita", "parana\u00edta", "centro de referencia de paranaita"),
    ),
    UnitDefinition("Campus Diamantino", "dmt", ("diamantino", "campus diamantino")),
    UnitDefinition(
        "Campus Guarant\u00e3 do Norte",
        "gta",
        ("guaranta do norte", "guarant\u00e3 do norte", "campus guaranta do norte"),
    ),
]


def _repair_mojibake(value: str) -> str:
    if "Ã" not in value and "Â" not in value:
        return value
    try:
        return value.encode("latin1").decode("utf-8")
    except UnicodeError:
        return value


def comparison_key(value: str) -> str:
    text = _repair_mojibake(normalize_text(str(value or "")))
    text = strip_accents(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


UNIT_BY_CODE = {unit.code: unit for unit in CANONICAL_UNITS}
EXACT_ALIAS_MAP: dict[str, UnitDefinition] = {}
FUZZY_CHOICES: list[tuple[str, UnitDefinition]] = []

for _unit in CANONICAL_UNITS:
    for _alias in (_unit.canonical_name, *_unit.aliases):
        _key = comparison_key(_alias)
        if _key and _key not in EXACT_ALIAS_MAP:
            EXACT_ALIAS_MAP[_key] = _unit
        if _key not in {"ifmt", "lrv"}:
            FUZZY_CHOICES.append((_key, _unit))


RULE_FRAGMENTS = [
    ("bela vista", "blv"),
    ("barra do garcas", "bag"),
    ("alta floresta", "alf"),
    ("campo novo", "cnp"),
    ("pontes e lacerda", "plc"),
    ("primavera do leste", "pdl"),
    ("sao vicente", "svc"),
    ("varzea grande", "vgd"),
    ("tangara da serra", "tga"),
    ("rondonopolis", "roo"),
    ("lucas do rio verde", "lrv"),
    ("campo verde", "cvd"),
    ("canarana", "can"),
    ("paranaita", "par"),
    ("diamantino", "dmt"),
    ("guaranta do norte", "gta"),
    ("caceres", "cas"),
    ("cuiaba octayde", "cba"),
    ("cuiaba", "cba"),
    ("confresa", "cfs"),
    ("juina", "jna"),
    ("sinop", "snp"),
    ("sorriso", "srs"),
    ("jaciara", "jac"),
    ("reitoria", "reitoria"),
]


def run_normalize_units(options: NormalizeUnitsOptions) -> dict[str, Any]:
    options.final_dir.mkdir(parents=True, exist_ok=True)
    full_input = options.full_input or options.final_dir / "news_articles_full.jsonl"
    canonical_input = options.canonical_input or options.final_dir / "news_articles_canonical.jsonl"

    full_records = _read_jsonl(full_input)
    canonical_records = _read_jsonl(canonical_input)
    normalized_full = [normalize_article(record) for record in full_records]
    normalized_canonical = [normalize_article(record) for record in canonical_records]

    output_paths = {
        "full_normalized_jsonl": options.final_dir / "news_articles_full_normalized.jsonl",
        "canonical_normalized_jsonl": options.final_dir / "news_articles_canonical_normalized.jsonl",
        "unit_report_csv": options.final_dir / "unit_normalization_report.csv",
        "unit_report_json": options.final_dir / "unit_normalization_report.json",
    }

    _write_jsonl(output_paths["full_normalized_jsonl"], normalized_full)
    _write_jsonl(output_paths["canonical_normalized_jsonl"], normalized_canonical)

    report = build_unit_normalization_report(
        full_records=normalized_full,
        canonical_records=normalized_canonical,
        final_dir=options.final_dir,
        input_paths={"full": full_input, "canonical": canonical_input},
        output_paths=output_paths,
    )
    _write_csv(output_paths["unit_report_csv"], UNIT_REPORT_FIELDS, report["report_rows"])
    output_paths["unit_report_json"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if options.in_place_standard:
        _write_standard_article_outputs(options.final_dir, normalized_full, normalized_canonical)

    if options.refresh_dataset_report:
        _refresh_dataset_reports(options.final_dir, normalized_full)

    return report


def normalize_article(article: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(article)
    publisher_raw = str(article.get("publisher_unit_raw") or article.get("publisher_unit") or "")
    campus_raw = str(article.get("campus_name_raw") or article.get("campus_name") or "")
    normalized["publisher_unit_raw"] = publisher_raw
    normalized["campus_name_raw"] = campus_raw

    match = resolve_unit(article, publisher_raw, campus_raw)
    if match.unit and match.applied:
        normalized["publisher_unit_normalized"] = match.unit.canonical_name
        normalized["campus_name_normalized"] = (
            "" if match.unit.code == "reitoria" else match.unit.canonical_name
        )
        normalized["campus_code"] = match.unit.code
    else:
        normalized["publisher_unit_normalized"] = ""
        normalized["campus_name_normalized"] = ""

    normalized["unit_normalization_status"] = match.status
    normalized["unit_normalization_score"] = round(match.score, 2)
    normalized["unit_normalization_source_value"] = match.source_value
    normalized["unit_normalization_note"] = match.note
    return normalized


def resolve_unit(article: dict[str, Any], publisher_raw: str, campus_raw: str) -> MatchResult:
    candidates = _candidate_values(article, publisher_raw, campus_raw)
    review_candidate: MatchResult | None = None

    for candidate in candidates:
        match = match_unit_value(candidate, article)
        if match.applied:
            return match
        if match.status == "needs_review" and review_candidate is None:
            review_candidate = match

    source_match = _source_default_match(article)
    if source_match:
        return source_match

    if review_candidate:
        return review_candidate
    return MatchResult(None, "unknown", 0, candidates[0] if candidates else "", "No unit match", False)


def match_unit_value(value: str, article: dict[str, Any] | None = None) -> MatchResult:
    source_value = normalize_text(str(value or ""))
    key = comparison_key(source_value)
    if not key:
        return MatchResult(None, "unknown", 0, source_value, "Empty unit label", False)

    if key in EXACT_ALIAS_MAP:
        return MatchResult(EXACT_ALIAS_MAP[key], "exact_match", 100, source_value, "", True)

    context_key = comparison_key(_context_text(article or {}, source_value))
    ambiguous = _match_ambiguous_advanced_campus(key, context_key, source_value)
    if ambiguous:
        return ambiguous

    for fragment, code in RULE_FRAGMENTS:
        if fragment in key:
            unit = UNIT_BY_CODE[code]
            return MatchResult(unit, "rule_match", 100, source_value, f"Matched fragment '{fragment}'", True)

    best_key, best_unit, best_score = _best_fuzzy_match(key)
    if best_unit and best_score >= 94:
        return MatchResult(
            best_unit,
            "fuzzy_match",
            best_score,
            source_value,
            f"Fuzzy matched '{best_key}'",
            True,
        )
    if best_unit and best_score >= 80:
        return MatchResult(
            best_unit,
            "needs_review",
            best_score,
            source_value,
            f"Low-confidence fuzzy suggestion: {best_unit.canonical_name}",
            False,
        )
    return MatchResult(None, "unknown", best_score, source_value, "No reliable unit match", False)


def build_unit_normalization_report(
    full_records: list[dict[str, Any]],
    canonical_records: list[dict[str, Any]],
    final_dir: Path,
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    report_rows = _build_report_rows(full_records, canonical_records)
    status_counts_full = Counter(record.get("unit_normalization_status") or "unknown" for record in full_records)
    status_counts_canonical = Counter(
        record.get("unit_normalization_status") or "unknown" for record in canonical_records
    )
    generated_at = datetime.now(timezone.utc).isoformat()

    return {
        "generated_at": generated_at,
        "final_dir": str(final_dir),
        "input_files": {key: str(path) for key, path in input_paths.items()},
        "output_files": {key: str(path) for key, path in output_paths.items()},
        "total_full_records": len(full_records),
        "total_canonical_records": len(canonical_records),
        "status_counts_full": dict(status_counts_full),
        "status_counts_canonical": dict(status_counts_canonical),
        "original_value_counts_full": _counter_items(_raw_unit_value(record) for record in full_records),
        "normalized_value_counts_full": _counter_items(_normalized_unit_value(record) for record in full_records),
        "original_value_counts_canonical": _counter_items(_raw_unit_value(record) for record in canonical_records),
        "normalized_value_counts_canonical": _counter_items(
            _normalized_unit_value(record) for record in canonical_records
        ),
        "unknown_or_needs_review_examples": _status_examples(
            full_records,
            statuses={"unknown", "needs_review"},
            limit=25,
        ),
        "corrected_noisy_examples": _corrected_examples(full_records, 25),
        "report_rows": report_rows,
    }


def print_normalize_units_report(report: dict[str, Any], preview_count: int = 10) -> None:
    print("")
    print("IFMT unit normalization report")
    print(f"Generated at: {report['generated_at']}")
    print(f"Full records: {report['total_full_records']}")
    print(f"Canonical records: {report['total_canonical_records']}")
    print(f"Status full: {json.dumps(report['status_counts_full'], ensure_ascii=False)}")
    print(f"Status canonical: {json.dumps(report['status_counts_canonical'], ensure_ascii=False)}")
    print(f"JSON report: {report['output_files']['unit_report_json']}")
    print(f"CSV report: {report['output_files']['unit_report_csv']}")
    print("")
    print(f"Corrected examples ({min(preview_count, len(report['corrected_noisy_examples']))}):")
    for example in report["corrected_noisy_examples"][:preview_count]:
        print(
            "  - "
            f"{example['original_value']} -> {example['normalized_value']} "
            f"({example['status']})"
        )
    print("")


def _candidate_values(article: dict[str, Any], publisher_raw: str, campus_raw: str) -> list[str]:
    values = [
        publisher_raw,
        campus_raw,
        str(article.get("institutional_unit") or ""),
    ]
    source = DISCOVERY_SOURCES.get(str(article.get("source_id") or ""))
    if source and source.campus_name:
        values.append(source.campus_name)
    unique: list[str] = []
    for value in values:
        value = normalize_text(value)
        if value and value not in unique:
            unique.append(value)
    return unique


def _source_default_match(article: dict[str, Any]) -> MatchResult | None:
    source = DISCOVERY_SOURCES.get(str(article.get("source_id") or ""))
    if not source or not source.campus_code:
        return None
    unit = UNIT_BY_CODE.get(source.campus_code)
    if not unit:
        return None
    return MatchResult(
        unit,
        "rule_match",
        100,
        source.campus_name or source.source_name,
        "Matched source campus configuration",
        True,
    )


def _match_ambiguous_advanced_campus(
    key: str,
    context_key: str,
    source_value: str,
) -> MatchResult | None:
    if not (key == "campus avancado de" or key.startswith("campus avancado de ")):
        return None
    if "lucas do rio verde" in context_key or re.search(r"\b(calrv|lrv)\b", context_key):
        return MatchResult(
            UNIT_BY_CODE["lrv"],
            "rule_match",
            100,
            source_value,
            "Resolved truncated advanced campus label from article context",
            True,
        )
    if "sinop" in context_key:
        return MatchResult(
            UNIT_BY_CODE["snp"],
            "rule_match",
            100,
            source_value,
            "Resolved truncated advanced campus label from article context",
            True,
        )
    return MatchResult(
        None,
        "needs_review",
        0,
        source_value,
        "Ambiguous truncated advanced campus label",
        False,
    )


def _best_fuzzy_match(key: str) -> tuple[str, UnitDefinition | None, float]:
    best_key = ""
    best_unit: UnitDefinition | None = None
    best_score = 0.0
    for choice, unit in FUZZY_CHOICES:
        score = _similarity_score(key, choice)
        if score > best_score:
            best_key = choice
            best_unit = unit
            best_score = score
    return best_key, best_unit, best_score


def _similarity_score(left: str, right: str) -> float:
    if fuzz is not None:
        return float(fuzz.WRatio(left, right))
    from difflib import SequenceMatcher

    return SequenceMatcher(None, left, right).ratio() * 100


def _context_text(article: dict[str, Any], source_value: str) -> str:
    return " ".join(
        str(part or "")
        for part in [
            source_value,
            article.get("title"),
            article.get("slug"),
            article.get("original_url"),
            article.get("request_url"),
            article.get("source_name"),
        ]
    )


def _raw_unit_value(record: dict[str, Any]) -> str:
    return (
        record.get("unit_normalization_source_value")
        or record.get("publisher_unit_raw")
        or record.get("publisher_unit")
        or record.get("campus_name_raw")
        or record.get("campus_name")
        or "unknown"
    )


def _normalized_unit_value(record: dict[str, Any]) -> str:
    return (
        record.get("campus_name_normalized")
        or record.get("publisher_unit_normalized")
        or "unknown"
    )


def _build_report_rows(
    full_records: list[dict[str, Any]],
    canonical_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    full_groups = _group_records_for_report(full_records)
    canonical_groups = _group_records_for_report(canonical_records)
    rows: list[dict[str, Any]] = []
    for key, group in sorted(full_groups.items(), key=lambda item: (-item[1]["count"], item[0])):
        example = group["example"]
        rows.append(
            {
                "original_value": group["original_value"],
                "original_value_key": comparison_key(group["original_value"]),
                "count_full": group["count"],
                "count_canonical": canonical_groups.get(key, {}).get("count", 0),
                "normalized_value": group["normalized_value"],
                "campus_code": group["campus_code"],
                "status": group["status"],
                "score": group["score"],
                "example_article_id": example.get("article_id", ""),
                "example_title": example.get("title", ""),
                "example_url": example.get("original_url", ""),
                "note": group["note"],
            }
        )
    return rows


def _group_records_for_report(records: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in records:
        original = _raw_unit_value(record)
        normalized_value = _normalized_unit_value(record)
        status = record.get("unit_normalization_status") or "unknown"
        campus_code = record.get("campus_code") or ""
        key = (original, normalized_value, campus_code, status)
        if key not in groups:
            groups[key] = {
                "original_value": original,
                "normalized_value": normalized_value,
                "campus_code": campus_code,
                "status": status,
                "score": record.get("unit_normalization_score") or 0,
                "note": record.get("unit_normalization_note") or "",
                "example": record,
                "count": 0,
            }
        groups[key]["count"] += 1
    return groups


def _status_examples(
    records: list[dict[str, Any]],
    statuses: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    examples = []
    for record in records:
        if record.get("unit_normalization_status") not in statuses:
            continue
        examples.append(_example_row(record))
        if len(examples) >= limit:
            break
    return examples


def _corrected_examples(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    examples = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        normalized_value = _normalized_unit_value(record)
        original_value = _raw_unit_value(record)
        if not normalized_value or normalized_value == "unknown":
            continue
        if comparison_key(original_value) == comparison_key(normalized_value):
            continue
        key = (original_value, normalized_value)
        if key in seen:
            continue
        seen.add(key)
        examples.append(_example_row(record))
        if len(examples) >= limit:
            break
    return examples


def _example_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "article_id": record.get("article_id", ""),
        "original_value": _raw_unit_value(record),
        "normalized_value": _normalized_unit_value(record),
        "campus_code": record.get("campus_code", ""),
        "status": record.get("unit_normalization_status", ""),
        "score": record.get("unit_normalization_score", ""),
        "title": record.get("title", ""),
        "original_url": record.get("original_url", ""),
        "note": record.get("unit_normalization_note", ""),
    }


def _counter_items(values: Iterable[str]) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in Counter(value or "unknown" for value in values).most_common()
    ]


def _refresh_dataset_reports(final_dir: Path, articles: list[dict[str, Any]]) -> None:
    media_rows = _read_csv_if_exists(final_dir / "media_assets.csv")
    link_rows = _read_csv_if_exists(final_dir / "article_links.csv")
    crawl_log_rows = _read_csv_if_exists(final_dir / "crawl_logs.csv")
    duplicate_rows = _read_csv_if_exists(final_dir / "duplicate_groups.csv")
    report = build_dataset_report(
        articles=articles,
        media_rows=media_rows,
        link_rows=link_rows,
        crawl_log_rows=crawl_log_rows,
        duplicate_rows=duplicate_rows,
        final_dir=final_dir,
    )
    report["unit_normalization_report"] = str(final_dir / "unit_normalization_report.json")
    (final_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (final_dir / "quality_report.md").write_text(
        build_quality_report_markdown(report),
        encoding="utf-8",
    )


def _write_standard_article_outputs(
    final_dir: Path,
    full_records: list[dict[str, Any]],
    canonical_records: list[dict[str, Any]],
) -> None:
    _write_jsonl(final_dir / "news_articles_full.jsonl", full_records)
    _write_jsonl(final_dir / "news_articles_canonical.jsonl", canonical_records)
    fields = _article_fields(full_records, canonical_records)
    _write_csv(final_dir / "news_articles_full.csv", fields, full_records)
    _write_csv(final_dir / "news_articles_canonical.csv", fields, canonical_records)
    _write_parquet_if_available(final_dir / "news_articles_full.parquet", full_records)
    _write_parquet_if_available(final_dir / "news_articles_canonical.parquet", canonical_records)


def _article_fields(*record_sets: list[dict[str, Any]]) -> list[str]:
    fields = list(dict.fromkeys([*NEWS_ARTICLE_FIELDS, *NORMALIZED_FIELDS]))
    for records in record_sets:
        for record in records:
            for key in record:
                if key not in fields:
                    fields.append(key)
    return fields


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Article input not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if line.strip():
                records.append(json.loads(line))
    return records


def _read_csv_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_csv(path: Path, fields: list[str], records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    field: json.dumps(record.get(field, ""), ensure_ascii=False)
                    if isinstance(record.get(field, ""), (list, dict))
                    else record.get(field, "")
                    for field in fields
                }
            )


def _write_parquet_if_available(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    try:
        import pandas as pd

        pd.DataFrame(records).to_parquet(path, index=False)
    except Exception:
        return

