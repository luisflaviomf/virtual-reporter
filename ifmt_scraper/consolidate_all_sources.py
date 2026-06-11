from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .dedupe import (
    DisjointSet,
    _add_pair_reason,
    _article_url_key,
    _body_similarity_text,
    _component_reasons,
    _has_dedupable_body,
    _similarity,
    choose_canonical,
)
from .normalize import normalize_text, normalize_title
from .url_utils import slug_from_url


DATASET_SOURCES = (
    ("final_combined", Path("data/output/final_combined")),
    ("final_antigoportal", Path("data/output/final_antigoportal")),
)


@dataclass
class ConsolidateAllSourcesOptions:
    final_combined_dir: Path = Path("data/output/final_combined")
    final_antigoportal_dir: Path = Path("data/output/final_antigoportal")
    output_dir: Path = Path("data/output/final_all_sources")
    title_similarity_threshold: int = 95
    body_similarity_threshold: int = 90


def run_consolidate_all_sources(options: ConsolidateAllSourcesOptions) -> dict[str, Any]:
    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dirs = (
        ("final_combined", options.final_combined_dir),
        ("final_antigoportal", options.final_antigoportal_dir),
    )

    articles, article_id_map = _load_global_articles(dataset_dirs)
    _dedupe_global_articles(
        articles=articles,
        title_similarity_threshold=options.title_similarity_threshold,
        body_similarity_threshold=options.body_similarity_threshold,
    )
    canonical_articles = [
        article for article in articles if _truthy(article.get("is_canonical", True))
    ]
    media_rows = _load_global_child_rows(
        dataset_dirs=dataset_dirs,
        article_id_map=article_id_map,
        filename="media_assets.csv",
        id_field="media_id",
        global_id_field="global_media_id",
        original_id_field="original_media_id",
        global_prefix="gmedia",
    )
    link_rows = _load_global_child_rows(
        dataset_dirs=dataset_dirs,
        article_id_map=article_id_map,
        filename="article_links.csv",
        id_field="link_id",
        global_id_field="global_link_id",
        original_id_field="original_link_id",
        global_prefix="glink",
    )
    duplicate_rows = _build_duplicate_group_rows(articles)

    article_fields = _article_fields(articles)
    media_fields = _child_fields(media_rows, "global_media_id", "media_id", "original_media_id")
    link_fields = _child_fields(link_rows, "global_link_id", "link_id", "original_link_id")
    duplicate_fields = _duplicate_group_fields(duplicate_rows)

    _write_jsonl(output_dir / "news_articles_full.jsonl", articles)
    _write_csv(output_dir / "news_articles_full.csv", article_fields, articles)
    _write_jsonl(output_dir / "news_articles_canonical.jsonl", canonical_articles)
    _write_csv(output_dir / "news_articles_canonical.csv", article_fields, canonical_articles)
    _write_csv(output_dir / "media_assets.csv", media_fields, media_rows)
    _write_csv(output_dir / "article_links.csv", link_fields, link_rows)
    _write_csv(output_dir / "duplicate_groups.csv", duplicate_fields, duplicate_rows)

    report = build_all_sources_report(
        articles=articles,
        canonical_articles=canonical_articles,
        media_rows=media_rows,
        link_rows=link_rows,
        duplicate_rows=duplicate_rows,
        output_dir=output_dir,
    )
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "quality_report.md").write_text(
        build_quality_report_markdown(report),
        encoding="utf-8",
    )
    return report


def print_all_sources_report(report: dict[str, Any]) -> None:
    print("")
    print("IFMT all-sources consolidation report")
    print(f"Expected raw total: {report['expected_raw_total']}")
    print(f"Obtained raw total: {report['total_raw_articles']}")
    print(f"Canonical articles: {report['total_canonical_articles']}")
    print(f"Non-canonical duplicates: {report['total_noncanonical_duplicates']}")
    print(f"Duplicate groups: {report['total_duplicate_groups']}")
    print(f"Cross-dataset duplicate groups: {report['cross_dataset_duplicate_groups']}")
    print(f"By source: {json.dumps(report['total_by_original_dataset_source'], ensure_ascii=False)}")
    print(f"By year: {json.dumps(report['total_by_year'], ensure_ascii=False)}")
    print(f"Dataset report: {report['output_files']['dataset_report']}")
    print(f"Quality report: {report['output_files']['quality_report_md']}")
    print("")


def _load_global_articles(
    dataset_dirs: tuple[tuple[str, Path], ...],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], str]]:
    articles: list[dict[str, Any]] = []
    article_id_map: dict[tuple[str, str], str] = {}

    for dataset_source, dataset_dir in dataset_dirs:
        input_path = dataset_dir / "news_articles_full.jsonl"
        records = _read_jsonl(input_path)
        for record in records:
            original_article_id = str(record.get("article_id", ""))
            global_article_id = f"gart_{len(articles) + 1:08d}"
            article_id_map[(dataset_source, original_article_id)] = global_article_id

            article = dict(record)
            article["global_article_id"] = global_article_id
            article["original_article_id"] = original_article_id
            article["original_dataset_source"] = dataset_source
            article["original_duplicate_group_id"] = record.get("duplicate_group_id", "")
            article["original_duplicate_of"] = record.get("duplicate_of", "")
            article["global_duplicate_group_id"] = ""
            article["article_id"] = global_article_id
            article["duplicate_group_id"] = ""
            article["duplicate_of"] = ""
            article["is_canonical"] = True
            article["duplicate_reason"] = ""
            article["duplicate_confidence"] = 0
            articles.append(article)

    return articles, article_id_map


def _dedupe_global_articles(
    articles: list[dict[str, Any]],
    title_similarity_threshold: int,
    body_similarity_threshold: int,
) -> None:
    article_ids = [article["article_id"] for article in articles]
    dsu = DisjointSet(article_ids)
    pair_reasons: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)

    _union_exact_global_keys(articles, dsu, pair_reasons)
    _union_approximate_global(
        articles=articles,
        dsu=dsu,
        pair_reasons=pair_reasons,
        title_similarity_threshold=title_similarity_threshold,
        body_similarity_threshold=body_similarity_threshold,
    )

    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for article in articles:
        components[dsu.find(article["article_id"])].append(article)

    duplicate_index = 1
    for component in components.values():
        if len(component) == 1:
            component[0]["is_canonical"] = True
            continue

        canonical = choose_canonical(component)
        group_id = f"gdup_{duplicate_index:08d}"
        duplicate_index += 1
        reasons, confidence = _component_reasons(component, pair_reasons)

        for article in component:
            article["global_duplicate_group_id"] = group_id
            article["duplicate_group_id"] = group_id
            article["duplicate_of"] = (
                "" if article["article_id"] == canonical["article_id"] else canonical["article_id"]
            )
            article["is_canonical"] = article["article_id"] == canonical["article_id"]
            article["duplicate_reason"] = "; ".join(reasons)
            article["duplicate_confidence"] = round(confidence, 4)


def _union_exact_global_keys(
    articles: list[dict[str, Any]],
    dsu: DisjointSet,
    pair_reasons: dict[tuple[str, str], list[tuple[str, float]]],
) -> None:
    keys: dict[str, dict[str, list[str]]] = {
        "normalized_url": defaultdict(list),
        "slug": defaultdict(list),
        "title_date": defaultdict(list),
        "content_hash": defaultdict(list),
    }
    for article in articles:
        article_id = article["article_id"]
        url_key = _article_url_key(article)
        if url_key:
            keys["normalized_url"][url_key].append(article_id)
        slug = normalize_title(
            slug_from_url(article.get("original_url", "") or article.get("request_url", ""))
        )
        if slug:
            keys["slug"][slug].append(article_id)
        if article.get("title_hash") and article.get("publication_date"):
            keys["title_date"][f"{article['title_hash']}|{article['publication_date']}"].append(article_id)
        if article.get("content_hash") and _has_dedupable_body(article):
            keys["content_hash"][article["content_hash"]].append(article_id)

    confidence_by_key = {
        "normalized_url": 1.0,
        "slug": 0.86,
        "title_date": 0.95,
        "content_hash": 1.0,
    }
    for reason, groups in keys.items():
        for ids in groups.values():
            if len(ids) < 2:
                continue
            first = ids[0]
            for other in ids[1:]:
                dsu.union(first, other)
                _add_pair_reason(pair_reasons, first, other, reason, confidence_by_key[reason])


def _union_approximate_global(
    articles: list[dict[str, Any]],
    dsu: DisjointSet,
    pair_reasons: dict[tuple[str, str], list[tuple[str, float]]],
    title_similarity_threshold: int,
    body_similarity_threshold: int,
) -> None:
    candidate_pairs = _approximate_candidate_pairs(articles)
    by_id = {article["article_id"]: article for article in articles}

    for left_id, right_id in candidate_pairs:
        left = by_id[left_id]
        right = by_id[right_id]
        title_score = _similarity(
            normalize_title(left.get("title", "")),
            normalize_title(right.get("title", "")),
        )
        body_score = 0
        if _has_dedupable_body(left) and _has_dedupable_body(right):
            body_score = _similarity(
                _body_similarity_text(left.get("body_text", "")),
                _body_similarity_text(right.get("body_text", "")),
            )

        if title_score >= title_similarity_threshold and _title_match_is_safe(left, right, body_score):
            dsu.union(left_id, right_id)
            _add_pair_reason(
                pair_reasons,
                left_id,
                right_id,
                f"title_similarity_{title_score}",
                title_score / 100,
            )
        elif body_score >= body_similarity_threshold:
            dsu.union(left_id, right_id)
            _add_pair_reason(
                pair_reasons,
                left_id,
                right_id,
                f"body_similarity_{body_score}",
                body_score / 100,
            )


def _approximate_candidate_pairs(articles: list[dict[str, Any]]) -> set[tuple[str, str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for article in articles:
        title = normalize_title(article.get("title", ""))
        if not title:
            continue
        year = str(article.get("publication_date") or "")[:4]
        words = title.split()
        keys = {f"title:{title[:24]}"}
        if year:
            keys.add(f"year_title:{year}|{title[:18]}")
        if len(words) >= 4:
            keys.add("tokens:" + " ".join(words[:5]))
        for key in keys:
            buckets[key].append(article["article_id"])

    pairs: set[tuple[str, str]] = set()
    for ids in buckets.values():
        unique_ids = sorted(set(ids))
        if len(unique_ids) < 2 or len(unique_ids) > 350:
            continue
        for index, left_id in enumerate(unique_ids):
            for right_id in unique_ids[index + 1 :]:
                pairs.add((left_id, right_id))
    return pairs


def _title_match_is_safe(left: dict[str, Any], right: dict[str, Any], body_score: int) -> bool:
    left_title = normalize_title(left.get("title", ""))
    right_title = normalize_title(right.get("title", ""))
    if not left_title or not right_title:
        return False
    if _is_generic_title(left_title) or _is_generic_title(right_title):
        left_date = left.get("publication_date", "")
        right_date = right.get("publication_date", "")
        return bool(left_date and right_date and left_date == right_date) or body_score >= 90
    return True


def _is_generic_title(title_key: str) -> bool:
    if len(title_key.split()) <= 3:
        return True
    generic = {
        "nota de falecimento",
        "pregao eletronico",
        "resultado final",
        "retificacao",
        "comunicado",
    }
    return title_key in generic


def _load_global_child_rows(
    dataset_dirs: tuple[tuple[str, Path], ...],
    article_id_map: dict[tuple[str, str], str],
    filename: str,
    id_field: str,
    global_id_field: str,
    original_id_field: str,
    global_prefix: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_source, dataset_dir in dataset_dirs:
        for record in _read_csv_if_exists(dataset_dir / filename):
            original_id = str(record.get(id_field, ""))
            original_article_id = str(record.get("article_id", ""))
            global_id = f"{global_prefix}_{len(rows) + 1:08d}"
            row = dict(record)
            row[global_id_field] = global_id
            row[original_id_field] = original_id
            row["original_article_id"] = original_article_id
            row["original_dataset_source"] = dataset_source
            row["global_article_id"] = article_id_map.get((dataset_source, original_article_id), "")
            row[id_field] = global_id
            row["article_id"] = row["global_article_id"]
            rows.append(row)
    return rows


def _build_duplicate_group_rows(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for article in articles:
        group_id = article.get("duplicate_group_id")
        if group_id:
            groups[group_id].append(article)

    rows = []
    for group_id in sorted(groups):
        component = groups[group_id]
        canonical = next((article for article in component if _truthy(article.get("is_canonical"))), component[0])
        datasets = sorted({article.get("original_dataset_source", "") for article in component})
        rows.append(
            {
                "global_duplicate_group_id": group_id,
                "duplicate_group_id": group_id,
                "canonical_article_id": canonical.get("article_id", ""),
                "global_canonical_article_id": canonical.get("global_article_id", canonical.get("article_id", "")),
                "duplicate_count": len(component),
                "duplicate_confidence": canonical.get("duplicate_confidence", ""),
                "duplicate_reasons": _json_list(_split_reasons(canonical.get("duplicate_reason", ""))),
                "article_ids": _json_list([article.get("article_id", "") for article in component]),
                "global_article_ids": _json_list([article.get("global_article_id", "") for article in component]),
                "original_article_ids": _json_list([article.get("original_article_id", "") for article in component]),
                "original_duplicate_group_ids": _json_list(
                    sorted({article.get("original_duplicate_group_id", "") for article in component if article.get("original_duplicate_group_id")})
                ),
                "original_dataset_sources": _json_list(datasets),
                "source_hosts": _json_list(sorted({article.get("source_host", "") for article in component})),
                "titles": _json_list([article.get("title", "") for article in component]),
                "publication_dates": _json_list([article.get("publication_date", "") for article in component]),
                "original_urls": _json_list([article.get("original_url", "") for article in component]),
                "cross_dataset_duplicate": len(datasets) > 1,
            }
        )
    return rows


def build_all_sources_report(
    articles: list[dict[str, Any]],
    canonical_articles: list[dict[str, Any]],
    media_rows: list[dict[str, Any]],
    link_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    word_counts = [int(article.get("word_count") or 0) for article in articles]
    publication_dates = [str(article.get("publication_date") or "") for article in articles]
    years = [date[:4] for date in publication_dates if len(date) >= 4]
    duplicate_articles = [
        article
        for article in articles
        if article.get("duplicate_group_id") and not _truthy(article.get("is_canonical"))
    ]
    cross_groups = [row for row in duplicate_rows if _truthy(row.get("cross_dataset_duplicate"))]
    errors = [article.get("error_message", "") for article in articles if article.get("error_message")]

    return {
        "expected_raw_total": 10189,
        "total_raw_articles": len(articles),
        "expected_raw_total_matched": len(articles) == 10189,
        "total_canonical_articles": len(canonical_articles),
        "total_noncanonical_duplicates": len(duplicate_articles),
        "total_duplicate_groups": len(duplicate_rows),
        "cross_dataset_duplicate_groups": len(cross_groups),
        "total_by_original_dataset_source": _counter_dict(article.get("original_dataset_source", "") for article in articles),
        "total_by_source_environment": _counter_dict(article.get("source_environment", "") for article in articles),
        "total_by_source_host": _counter_dict(article.get("source_host", "") for article in articles),
        "total_by_year": _counter_dict(years),
        "total_by_normalized_unit": _counter_dict(_normalized_unit(article) for article in articles),
        "total_with_title": sum(1 for article in articles if article.get("title")),
        "total_with_body": sum(1 for article in articles if article.get("body_text")),
        "total_with_date": sum(1 for article in articles if article.get("publication_date")),
        "total_with_author_or_unit": sum(
            1
            for article in articles
            if article.get("author_name")
            or article.get("publisher_unit")
            or article.get("campus_name")
            or article.get("publisher_unit_normalized")
            or article.get("campus_name_normalized")
        ),
        "total_with_images": sum(1 for article in articles if _truthy(article.get("has_images"))),
        "total_with_links": sum(1 for article in articles if _truthy(article.get("has_links"))),
        "total_with_attachments": sum(1 for article in articles if _truthy(article.get("has_attachments"))),
        "total_needs_review": sum(1 for article in articles if article.get("status") == "needs_review"),
        "total_errors": sum(1 for article in articles if article.get("status") == "error"),
        "media_assets_count": len(media_rows),
        "article_links_count": len(link_rows),
        "word_count_stats": _word_stats(word_counts),
        "coverage_percent": _coverage_percent(articles),
        "top_10_errors": Counter(errors).most_common(10),
        "examples_cross_dataset_duplicates": _cross_dataset_examples(cross_groups, 10),
        "output_files": {
            "news_articles_full_jsonl": str(output_dir / "news_articles_full.jsonl"),
            "news_articles_full_csv": str(output_dir / "news_articles_full.csv"),
            "news_articles_canonical_jsonl": str(output_dir / "news_articles_canonical.jsonl"),
            "news_articles_canonical_csv": str(output_dir / "news_articles_canonical.csv"),
            "media_assets_csv": str(output_dir / "media_assets.csv"),
            "article_links_csv": str(output_dir / "article_links.csv"),
            "duplicate_groups_csv": str(output_dir / "duplicate_groups.csv"),
            "dataset_report": str(output_dir / "dataset_report.json"),
            "quality_report_md": str(output_dir / "quality_report.md"),
        },
    }


def build_quality_report_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Relatorio de Qualidade - Todas as Fontes IFMT",
            "",
            "## Fontes consolidadas",
            "",
            "- `final_combined`: portal historico por IP, portal atual e campi atuais.",
            "- `final_antigoportal`: portal antigo em subdominios `*.antigoportal.ifmt.edu.br`.",
            "",
            "## Resumo",
            "",
            f"- Total bruto esperado: {report['expected_raw_total']}",
            f"- Total bruto obtido: {report['total_raw_articles']}",
            f"- Total canonico final: {report['total_canonical_articles']}",
            f"- Duplicados nao-canonicos: {report['total_noncanonical_duplicates']}",
            f"- Grupos de duplicidade: {report['total_duplicate_groups']}",
            f"- Grupos de duplicidade entre datasets: {report['cross_dataset_duplicate_groups']}",
            "",
            "## Cobertura dos campos principais",
            "",
            *[
                f"- {field}: {percent}%"
                for field, percent in report["coverage_percent"].items()
            ],
            "",
            "## Preservacao dos dados",
            "",
            "O arquivo `news_articles_full.jsonl` preserva todos os registros brutos, inclusive duplicatas, "
            "`needs_review` e erros. A deduplicacao apenas marca `duplicate_group_id`, `duplicate_of` "
            "e `is_canonical`; registros nao sao removidos do arquivo full.",
            "",
            "## Limitacoes conhecidas",
            "",
            "- O antigoportal tem mais ruido estrutural e mais paginas com corpo fraco; esses registros foram preservados para rastreabilidade.",
            "- Algumas datas do HTML legado sao ambiguas e podem gerar anos improvaveis, exigindo QA posterior.",
            "- A deduplicacao aproximada e conservadora, mas grupos com titulos genericos ainda devem ser revisados antes de uso definitivo.",
            "- Arquivos de midia e links foram consolidados por metadados; download de anexos/midias nao faz parte desta etapa.",
        ]
    )


def _cross_dataset_examples(duplicate_rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    examples = []
    for row in duplicate_rows[:limit]:
        examples.append(
            {
                "global_duplicate_group_id": row.get("global_duplicate_group_id", ""),
                "duplicate_count": row.get("duplicate_count", 0),
                "canonical_article_id": row.get("canonical_article_id", ""),
                "original_dataset_sources": _loads_json_list(row.get("original_dataset_sources", "[]")),
                "titles": _loads_json_list(row.get("titles", "[]"))[:5],
                "original_urls": _loads_json_list(row.get("original_urls", "[]"))[:5],
            }
        )
    return examples


def _article_fields(articles: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "global_article_id",
        "article_id",
        "original_article_id",
        "original_dataset_source",
        "global_duplicate_group_id",
        "original_duplicate_group_id",
        "original_duplicate_of",
    ]
    return _ordered_fields(preferred, articles)


def _child_fields(rows: list[dict[str, Any]], global_id: str, id_field: str, original_id: str) -> list[str]:
    preferred = [
        global_id,
        id_field,
        original_id,
        "global_article_id",
        "article_id",
        "original_article_id",
        "original_dataset_source",
    ]
    return _ordered_fields(preferred, rows)


def _duplicate_group_fields(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "global_duplicate_group_id",
        "duplicate_group_id",
        "canonical_article_id",
        "global_canonical_article_id",
        "duplicate_count",
        "duplicate_confidence",
        "duplicate_reasons",
        "cross_dataset_duplicate",
        "original_dataset_sources",
        "article_ids",
        "original_article_ids",
        "original_duplicate_group_ids",
        "source_hosts",
        "titles",
        "publication_dates",
        "original_urls",
    ]
    return _ordered_fields(preferred, rows)


def _ordered_fields(preferred: list[str], rows: list[dict[str, Any]]) -> list[str]:
    fields = list(dict.fromkeys(preferred))
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_csv_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as jsonl_file:
        for row in rows:
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            csv_row = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                csv_row[field] = value
            writer.writerow(csv_row)


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
    total = len(articles)
    if not total:
        return {}
    fields = {
        "title": lambda article: bool(article.get("title")),
        "body_text": lambda article: bool(article.get("body_text")),
        "publication_date": lambda article: bool(article.get("publication_date")),
        "author_or_unit": lambda article: bool(
            article.get("author_name") or article.get("publisher_unit") or article.get("campus_name")
        ),
        "links": lambda article: _truthy(article.get("has_links")),
        "attachments": lambda article: _truthy(article.get("has_attachments")),
        "images": lambda article: _truthy(article.get("has_images")),
    }
    return {
        field: round((sum(1 for article in articles if predicate(article)) / total) * 100, 2)
        for field, predicate in fields.items()
    }


def _normalized_unit(article: dict[str, Any]) -> str:
    return (
        article.get("campus_name_normalized")
        or article.get("publisher_unit_normalized")
        or article.get("campus_name")
        or article.get("publisher_unit")
        or "unknown"
    )


def _counter_dict(values: Iterable[str]) -> dict[str, int]:
    return dict(Counter(value or "unknown" for value in values).most_common())


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "sim"}


def _json_list(values: list[Any]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _loads_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value or "[]")
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def _split_reasons(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]
