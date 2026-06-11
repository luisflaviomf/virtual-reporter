from __future__ import annotations

import csv
import json
import shutil
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import DISCOVERY_SOURCES
from .crawl import (
    CRAWL_LOG_FIELDS,
    LINK_FIELDS,
    MANUAL_REVIEW_FIELDS,
    MEDIA_FIELDS,
    NEWS_ARTICLE_FIELDS,
    build_manual_review_sample,
)
from .dedupe import DEDUPED_ARTICLE_FIELDS, DUPLICATE_GROUP_FIELDS


@dataclass
class ExportOptions:
    output_dir: Path = Path("data/output")
    final_dir: Path = Path("data/output/final")
    input_prefix: str = "news_articles"


@dataclass
class ReportOptions:
    output_dir: Path = Path("data/output")
    final_dir: Path = Path("data/output/final")
    input_prefix: str = "news_articles"


def run_export(options: ExportOptions) -> dict[str, Any]:
    options.final_dir.mkdir(parents=True, exist_ok=True)
    paths = _input_paths(options.output_dir, options.input_prefix)
    final_paths = _final_paths(options.final_dir)

    article_input = paths["deduped_jsonl"] if paths["deduped_jsonl"].exists() else paths["articles_jsonl"]
    articles = _read_jsonl(article_input)
    article_fields = DEDUPED_ARTICLE_FIELDS if paths["deduped_jsonl"].exists() else NEWS_ARTICLE_FIELDS
    canonical_articles = [article for article in articles if _truthy(article.get("is_canonical", True))]

    _write_jsonl(final_paths["news_full_jsonl"], articles)
    _write_csv(final_paths["news_full_csv"], article_fields, articles)
    _write_jsonl(final_paths["news_canonical_jsonl"], canonical_articles)
    _write_csv(final_paths["news_canonical_csv"], article_fields, canonical_articles)
    _write_parquet_if_available(final_paths["news_full_parquet"], articles)
    _write_parquet_if_available(final_paths["news_canonical_parquet"], canonical_articles)

    media_rows = _read_csv_if_exists(paths["media_csv"])
    link_rows = _read_csv_if_exists(paths["links_csv"])
    crawl_log_rows = _read_csv_if_exists(paths["crawl_logs_csv"])
    duplicate_rows = _read_csv_if_exists(paths["duplicate_groups_csv"])
    _write_csv(final_paths["media_csv"], MEDIA_FIELDS, media_rows)
    _write_csv(final_paths["links_csv"], LINK_FIELDS, link_rows)
    _write_csv(final_paths["crawl_logs_csv"], _fields_from_rows(crawl_log_rows) or CRAWL_LOG_FIELDS, crawl_log_rows)
    _write_csv(final_paths["duplicate_groups_csv"], DUPLICATE_GROUP_FIELDS, duplicate_rows)
    _write_csv(final_paths["sources_csv"], _source_fields(), _source_rows())
    _write_csv(final_paths["manual_review_csv"], MANUAL_REVIEW_FIELDS, build_manual_review_sample(articles))

    report = build_dataset_report(
        articles=articles,
        media_rows=media_rows,
        link_rows=link_rows,
        crawl_log_rows=crawl_log_rows,
        duplicate_rows=duplicate_rows,
        final_dir=options.final_dir,
    )
    final_paths["dataset_report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    final_paths["quality_report_md"].write_text(
        build_quality_report_markdown(report),
        encoding="utf-8",
    )

    return {
        "total_articles": len(articles),
        "total_canonical": len(canonical_articles),
        "total_duplicate_groups": len(duplicate_rows),
        "output_paths": {key: str(path) for key, path in final_paths.items() if path.exists()},
    }


def run_report(options: ReportOptions) -> dict[str, Any]:
    options.final_dir.mkdir(parents=True, exist_ok=True)
    paths = _input_paths(options.output_dir, options.input_prefix)
    final_paths = _final_paths(options.final_dir)
    article_input = paths["deduped_jsonl"] if paths["deduped_jsonl"].exists() else paths["articles_jsonl"]
    articles = _read_jsonl(article_input)
    media_rows = _read_csv_if_exists(paths["media_csv"])
    link_rows = _read_csv_if_exists(paths["links_csv"])
    crawl_log_rows = _read_csv_if_exists(paths["crawl_logs_csv"])
    duplicate_rows = _read_csv_if_exists(paths["duplicate_groups_csv"])

    report = build_dataset_report(
        articles=articles,
        media_rows=media_rows,
        link_rows=link_rows,
        crawl_log_rows=crawl_log_rows,
        duplicate_rows=duplicate_rows,
        final_dir=options.final_dir,
    )
    final_paths["dataset_report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    final_paths["quality_report_md"].write_text(
        build_quality_report_markdown(report),
        encoding="utf-8",
    )
    return report


def print_export_report(report: dict[str, Any]) -> None:
    print("")
    print("IFMT export report")
    print(f"Total articles: {report['total_articles']}")
    print(f"Canonical articles: {report['total_canonical']}")
    print(f"Duplicate groups: {report['total_duplicate_groups']}")
    for label, path in report["output_paths"].items():
        print(f"{label}: {path}")
    print("")


def print_dataset_report(report: dict[str, Any]) -> None:
    print("")
    print("IFMT dataset report")
    print(f"Total collected: {report['total_articles']}")
    print(f"Canonical articles: {report['total_canonical_articles']}")
    print(f"Duplicated articles: {report['total_duplicated_articles']}")
    print(f"Needs review: {report['total_needs_review']}")
    print(f"Errors: {report['total_errors']}")
    print(f"Average word count: {report['word_count_stats']['average']}")
    print(f"By year: {json.dumps(report['total_by_year'], ensure_ascii=False)}")
    print(f"Dataset report: {report['output_files']['dataset_report']}")
    print(f"Quality report: {report['output_files']['quality_report_md']}")
    print("")


def build_dataset_report(
    articles: list[dict[str, Any]],
    media_rows: list[dict[str, Any]],
    link_rows: list[dict[str, Any]],
    crawl_log_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    final_dir: Path = Path("data/output/final"),
) -> dict[str, Any]:
    total = len(articles)
    canonical_articles = [article for article in articles if _truthy(article.get("is_canonical", True))]
    duplicated_articles = [
        article
        for article in articles
        if article.get("duplicate_group_id") and not _truthy(article.get("is_canonical", True))
    ]
    word_counts = [int(article.get("word_count") or 0) for article in articles]
    publication_dates = [str(article.get("publication_date") or "") for article in articles]
    years = [date[:4] for date in publication_dates if len(date) >= 4]
    error_messages = [
        article.get("error_message", "")
        for article in articles
        if article.get("error_message")
    ]

    report = {
        "total_articles": total,
        "total_canonical_articles": len(canonical_articles),
        "total_duplicated_articles": len(duplicated_articles),
        "total_duplicate_groups": len(duplicate_rows),
        "total_needs_review": sum(1 for article in articles if article.get("status") == "needs_review"),
        "total_errors": sum(1 for article in articles if article.get("status") == "error"),
        "total_by_source_environment": _counter_dict(article.get("source_environment", "") for article in articles),
        "total_by_source_host": _counter_dict(article.get("source_host", "") for article in articles),
        "total_by_campus_or_unit": _counter_dict(
            article.get("campus_name") or article.get("publisher_unit") or "unknown"
            for article in articles
        ),
        "total_by_normalized_unit": _counter_dict(_normalized_unit(article) for article in articles),
        "total_by_year": _counter_dict(years),
        "total_by_month": _counter_dict(date[:7] for date in publication_dates if len(date) >= 7),
        "total_with_title": sum(1 for article in articles if article.get("title")),
        "total_with_body": sum(1 for article in articles if article.get("body_text")),
        "total_with_publication_date": sum(1 for article in articles if article.get("publication_date")),
        "total_with_author_or_unit": sum(
            1 for article in articles if article.get("author_name") or article.get("publisher_unit") or article.get("campus_name")
        ),
        "total_with_images": sum(1 for article in articles if _truthy(article.get("has_images"))),
        "total_with_links": sum(1 for article in articles if _truthy(article.get("has_links"))),
        "total_with_attachments": sum(1 for article in articles if _truthy(article.get("has_attachments"))),
        "media_assets_count": len(media_rows),
        "article_links_count": len(link_rows),
        "crawl_logs_count": len(crawl_log_rows),
        "word_count_stats": _word_stats(word_counts),
        "top_20_categories": _top_terms(articles, "categories"),
        "top_20_units_campi": Counter(
            article.get("campus_name") or article.get("publisher_unit") or "unknown"
            for article in articles
        ).most_common(20),
        "top_20_normalized_units": Counter(_normalized_unit(article) for article in articles).most_common(20),
        "top_20_years": Counter(years).most_common(20),
        "top_20_errors": Counter(error_messages).most_common(20),
        "coverage_percent": _coverage_percent(articles),
        "examples_collected": _examples(
            [article for article in articles if article.get("status") == "collected"], 10
        ),
        "examples_duplicates": _examples(duplicated_articles, 10),
        "examples_needs_review": _examples(
            [article for article in articles if article.get("status") == "needs_review"], 10
        ),
        "examples_cross_source_duplicates": _cross_source_duplicate_examples(articles),
        "output_files": {
            "dataset_report": str(final_dir / "dataset_report.json"),
            "quality_report_md": str(final_dir / "quality_report.md"),
        },
    }
    return report


def build_quality_report_markdown(report: dict[str, Any]) -> str:
    total = report["total_articles"]
    success = total - report["total_errors"]
    success_rate = round((success / total) * 100, 2) if total else 0
    return "\n".join(
        [
            "# IFMT Dataset Quality Report",
            "",
            "## Resumo da coleta",
            "",
            f"- Total de noticias coletadas: {total}",
            f"- Noticias canonicas: {report['total_canonical_articles']}",
            f"- Noticias duplicadas: {report['total_duplicated_articles']}",
            f"- Needs review: {report['total_needs_review']}",
            f"- Erros: {report['total_errors']}",
            f"- Taxa de sucesso: {success_rate}%",
            "",
            "## Fontes utilizadas",
            "",
            *[
                f"- {source}: {count}"
                for source, count in report["total_by_source_environment"].items()
            ],
            "",
            "## Top unidades normalizadas",
            "",
            *[
                f"- {unit}: {count}"
                for unit, count in report.get("top_20_normalized_units", [])
            ],
            "",
            "## Observacoes sobre o IP historico",
            "",
            "O portal historico por IP deve continuar sendo acessado como "
            "`200.129.244.212`, com verificacao SSL desabilitada apenas para esse host. "
            "Os links relativos do IP devem permanecer baseados no IP e nao devem ser "
            "reescritos para `ifmt.edu.br`.",
            "",
            "## Cobertura dos principais campos",
            "",
            *[
                f"- {field}: {percent}%"
                for field, percent in report["coverage_percent"].items()
            ],
            "",
            "## Limitacoes conhecidas",
            "",
            "- A deduplicacao aproximada gera grupos revisaveis e nao remove registros.",
            "- Alguns registros historicos podem ter data parcial se a pagina individual nao trouxer ano.",
            "- Imagens sao catalogadas por URL/metadados; o download de midia ainda nao e feito por padrao.",
            "- A normalizacao de unidades usa regras e fuzzy matching; registros `needs_review` exigem validacao humana.",
            "- Noticias do portal atual sem rotulo explicito de unidade ficam como `unknown`, sem inferencia por titulo.",
            "",
            "## Proximos passos recomendados",
            "",
            "- Revisar amostra manual antes da coleta completa.",
            "- Rodar coleta incremental por fonte com `--resume`.",
            "- Inspecionar grupos de duplicidade antes de usar apenas registros canonicos.",
        ]
    )


def _input_paths(output_dir: Path, input_prefix: str = "news_articles") -> dict[str, Path]:
    if input_prefix == "news_articles_antigoportal":
        return {
            "articles_jsonl": output_dir / "news_articles_antigoportal_latest.jsonl",
            "deduped_jsonl": output_dir / "news_articles_antigoportal_deduped_latest.jsonl",
            "media_csv": output_dir / "media_assets_antigoportal_latest.csv",
            "links_csv": output_dir / "article_links_antigoportal_latest.csv",
            "crawl_logs_csv": output_dir / "crawl_logs_antigoportal_latest.csv",
            "duplicate_groups_csv": output_dir / "duplicate_groups_antigoportal_latest.csv",
        }
    return {
        "articles_jsonl": output_dir / "news_articles_latest.jsonl",
        "deduped_jsonl": output_dir / "news_articles_deduped_latest.jsonl",
        "media_csv": output_dir / "media_assets_latest.csv",
        "links_csv": output_dir / "article_links_latest.csv",
        "crawl_logs_csv": output_dir / "crawl_logs_latest.csv",
        "duplicate_groups_csv": output_dir / "duplicate_groups_latest.csv",
    }


def _final_paths(final_dir: Path) -> dict[str, Path]:
    return {
        "news_full_jsonl": final_dir / "news_articles_full.jsonl",
        "news_full_csv": final_dir / "news_articles_full.csv",
        "news_canonical_jsonl": final_dir / "news_articles_canonical.jsonl",
        "news_canonical_csv": final_dir / "news_articles_canonical.csv",
        "news_full_parquet": final_dir / "news_articles_full.parquet",
        "news_canonical_parquet": final_dir / "news_articles_canonical.parquet",
        "media_csv": final_dir / "media_assets.csv",
        "links_csv": final_dir / "article_links.csv",
        "sources_csv": final_dir / "sources.csv",
        "crawl_logs_csv": final_dir / "crawl_logs.csv",
        "duplicate_groups_csv": final_dir / "duplicate_groups.csv",
        "dataset_report": final_dir / "dataset_report.json",
        "manual_review_csv": final_dir / "manual_review_sample.csv",
        "quality_report_md": final_dir / "quality_report.md",
    }


def _source_fields() -> list[str]:
    return [
        "source_id",
        "source_name",
        "source_type",
        "base_url",
        "news_archive_url",
        "campus_code",
        "campus_name",
        "status",
        "ssl_verify",
        "notes",
    ]


def _source_rows() -> list[dict[str, Any]]:
    rows = []
    for source in DISCOVERY_SOURCES.values():
        rows.append(
            {
                "source_id": source.source_id,
                "source_name": source.source_name,
                "source_type": source.source_type,
                "base_url": source.base_url,
                "news_archive_url": source.news_archive_url,
                "campus_code": source.campus_code or "",
                "campus_name": source.campus_name or "",
                "status": "active",
                "ssl_verify": source.ssl_verify,
                "notes": source.notes,
            }
        )
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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
    fields = fields or []
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
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


def _fields_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _word_stats(word_counts: list[int]) -> dict[str, Any]:
    if not word_counts:
        return {"average": 0, "median": 0, "min": 0, "max": 0}
    return {
        "average": round(sum(word_counts) / len(word_counts), 2),
        "median": statistics.median(word_counts),
        "min": min(word_counts),
        "max": max(word_counts),
    }


def _coverage_percent(articles: list[dict[str, Any]]) -> dict[str, float]:
    fields = {
        "title": lambda article: article.get("title"),
        "body_text": lambda article: article.get("body_text"),
        "publication_date": lambda article: article.get("publication_date"),
        "author_or_unit": lambda article: article.get("author_name") or article.get("publisher_unit") or article.get("campus_name"),
        "images": lambda article: _truthy(article.get("has_images")),
        "links": lambda article: _truthy(article.get("has_links")),
        "attachments": lambda article: _truthy(article.get("has_attachments")),
    }
    total = len(articles)
    if not total:
        return {field: 0 for field in fields}
    return {
        field: round((sum(1 for article in articles if checker(article)) / total) * 100, 2)
        for field, checker in fields.items()
    }


def _top_terms(articles: list[dict[str, Any]], key: str) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for article in articles:
        value = article.get(key, [])
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = [value] if value else []
        for item in value or []:
            if item:
                counter[str(item)] += 1
    return counter.most_common(20)


def _examples(articles: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [
        {
            "article_id": article.get("article_id", ""),
            "title": article.get("title", ""),
            "publication_date": article.get("publication_date", ""),
            "source_environment": article.get("source_environment", ""),
            "publisher_unit": article.get("publisher_unit", ""),
            "campus_name": article.get("campus_name", ""),
            "word_count": article.get("word_count", 0),
            "original_url": article.get("original_url", ""),
            "duplicate_group_id": article.get("duplicate_group_id", ""),
            "duplicate_of": article.get("duplicate_of", ""),
            "error_message": article.get("error_message", ""),
        }
        for article in articles[:limit]
    ]


def _normalized_unit(article: dict[str, Any]) -> str:
    return (
        article.get("campus_name_normalized")
        or article.get("publisher_unit_normalized")
        or article.get("campus_name")
        or article.get("publisher_unit")
        or "unknown"
    )


def _cross_source_duplicate_examples(
    articles: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for article in articles:
        group_id = article.get("duplicate_group_id")
        if not group_id:
            continue
        groups.setdefault(str(group_id), []).append(article)

    examples: list[dict[str, Any]] = []
    for group_id, group_articles in groups.items():
        environments = sorted(
            {
                str(article.get("source_environment") or "unknown")
                for article in group_articles
            }
        )
        if len(environments) < 2:
            continue
        examples.append(
            {
                "duplicate_group_id": group_id,
                "source_environments": environments,
                "article_ids": [article.get("article_id", "") for article in group_articles],
                "titles": [article.get("title", "") for article in group_articles],
                "original_urls": [article.get("original_url", "") for article in group_articles],
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _counter_dict(values: Iterable[str]) -> dict[str, int]:
    return dict(Counter(value or "unknown" for value in values))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "sim"}
