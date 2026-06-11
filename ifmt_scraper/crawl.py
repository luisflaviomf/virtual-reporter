from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .discover import resolve_source_selection
from .http_client import HttpClient
from .parse_articles import ParsedArticle, build_error_article, parse_article_html


NEWS_ARTICLE_FIELDS = [
    "article_id",
    "discovered_id",
    "source_id",
    "source_name",
    "source_environment",
    "source_base_url",
    "source_host",
    "source_access_mode",
    "discovered_from_url",
    "discovered_page_number",
    "original_url",
    "request_url",
    "final_url_after_redirect",
    "canonical_url",
    "http_status",
    "redirected",
    "used_ip_access",
    "ssl_verify_disabled",
    "slug",
    "title",
    "subtitle",
    "lead",
    "body_text",
    "body_html",
    "publication_date",
    "publication_time",
    "date_label_raw",
    "author_name",
    "author_url",
    "publisher_unit",
    "campus_name",
    "campus_code",
    "institutional_unit",
    "publisher_unit_raw",
    "campus_name_raw",
    "publisher_unit_normalized",
    "campus_name_normalized",
    "unit_normalization_status",
    "unit_normalization_score",
    "unit_normalization_source_value",
    "unit_normalization_note",
    "categories",
    "tags",
    "estimated_reading_time",
    "comments_count",
    "access_count",
    "word_count",
    "has_images",
    "has_links",
    "has_attachments",
    "content_hash",
    "title_hash",
    "title_date_hash",
    "duplicate_group_id",
    "duplicate_of",
    "is_canonical",
    "duplicate_reason",
    "duplicate_confidence",
    "collected_at",
    "updated_at",
    "status",
    "error_message",
]

MEDIA_FIELDS = [
    "media_id",
    "article_id",
    "media_url",
    "normalized_media_url",
    "local_path",
    "media_type",
    "alt_text",
    "caption",
    "position_in_article",
    "should_download",
    "downloaded",
    "error_message",
]

LINK_FIELDS = [
    "link_id",
    "article_id",
    "url",
    "normalized_url",
    "anchor_text",
    "link_type",
    "domain",
    "file_extension",
    "should_download",
    "downloaded",
    "notes",
]

MANUAL_REVIEW_FIELDS = [
    "article_id",
    "source_environment",
    "title",
    "publication_date",
    "date_label_raw",
    "author_name",
    "publisher_unit",
    "campus_name",
    "original_url",
    "final_url_after_redirect",
    "body_text_preview",
    "word_count",
    "has_images",
    "has_links",
    "status",
    "error_message",
]

CRAWL_LOG_FIELDS = [
    "log_id",
    "timestamp",
    "source_id",
    "url",
    "action",
    "status_code",
    "success",
    "error_type",
    "error_message",
    "elapsed_seconds",
    "retry_count",
    "html_length",
    "final_url",
]


@dataclass
class CrawlOptions:
    input_path: Path
    sources: str
    limit: int | None
    output_prefix: str
    output_dir: Path
    raw_html_dir: Path
    resume: bool = False
    force: bool = False
    save_html: bool = False
    dry_run: bool = False
    timeout: float = 30.0
    delay_seconds: float = 0.5
    max_retries: int = 2


def run_crawl(options: CrawlOptions) -> dict[str, Any]:
    options.output_dir.mkdir(parents=True, exist_ok=True)
    if options.save_html and not options.dry_run:
        options.raw_html_dir.mkdir(parents=True, exist_ok=True)

    output_paths = _output_paths(options.output_dir, options.output_prefix)
    input_records = read_discovered_records(options.input_path)
    selected_source_ids = {source.source_id for source in resolve_source_selection(options.sources)}
    filtered_records = [
        record for record in input_records if record.get("source_id") in selected_source_ids
    ]
    total_input_urls = len(filtered_records)
    if options.limit is not None:
        filtered_records = _limited_records(filtered_records, options.limit)

    existing_discovery_keys = set()
    existing_request_urls = set()
    articles: list[dict[str, Any]] = []
    media_assets: list[dict[str, Any]] = []
    article_links: list[dict[str, Any]] = []
    crawl_logs: list[dict[str, Any]] = []
    skipped_existing = 0
    if options.resume and not options.force and output_paths["articles_jsonl"].exists():
        existing_articles = _read_jsonl(output_paths["articles_jsonl"])
        articles.extend(existing_articles)
        existing_discovery_keys = {
            (
                str(article.get("source_id", "")),
                str(article.get("discovered_id", "")),
            )
            for article in existing_articles
            if article.get("source_id") and article.get("discovered_id")
        }
        existing_request_urls = {
            str(article.get("request_url") or article.get("original_url") or "")
            for article in existing_articles
        }
        media_assets = _read_csv_if_exists(output_paths["media_csv"])
        article_links = _read_csv_if_exists(output_paths["links_csv"])
        crawl_logs = _read_csv_if_exists(output_paths["crawl_logs_csv"])

    if not options.dry_run:
        if not options.resume or options.force:
            _initialize_incremental_outputs(output_paths)
        else:
            _ensure_incremental_outputs(output_paths)

    client = HttpClient(
        timeout=options.timeout,
        delay_seconds=options.delay_seconds,
        max_retries=options.max_retries,
    )

    attempted = 0
    for discovered_record in filtered_records:
        request_url = discovered_record.get("discovered_url") or discovered_record.get("normalized_url")
        discovery_key = (
            str(discovered_record.get("source_id", "")),
            str(discovered_record.get("discovered_id", "")),
        )
        if (
            options.resume
            and not options.force
            and (
                discovery_key in existing_discovery_keys
                or request_url in existing_request_urls
            )
        ):
            skipped_existing += 1
            continue

        attempted += 1
        article_id = f"art_{len(articles) + 1:08d}"
        collected_at = datetime.now(timezone.utc).isoformat()
        parsed_media = []
        parsed_links = []
        fetch_log: dict[str, Any] | None = None

        try:
            started = time.perf_counter()
            fetch_result = client.fetch_url(request_url)
            selected = fetch_result.selected
            fetch_log = _crawl_log_record(
                log_id=f"log_{len(crawl_logs) + 1:08d}",
                discovered_record=discovered_record,
                request_url=request_url,
                action="fetch_article",
                status_code=selected.status_code,
                success=bool(selected.status_code and 200 <= selected.status_code < 400),
                error_type=selected.error_type or "",
                error_message=selected.error_message or "",
                elapsed_seconds=selected.elapsed_seconds,
                retry_count=selected.retry_count,
                html_length=selected.html_length,
                final_url=selected.final_url or "",
            )
            if selected.status_code and 200 <= selected.status_code < 400 and selected.html:
                selected = _select_oldportal_article_attempt(
                    selected=selected,
                    discovered_record=discovered_record,
                    request_url=request_url,
                    client=client,
                )
                fetch_log = _crawl_log_record(
                    log_id=f"log_{len(crawl_logs) + 1:08d}",
                    discovered_record=discovered_record,
                    request_url=request_url,
                    action="fetch_article",
                    status_code=selected.status_code,
                    success=bool(selected.status_code and 200 <= selected.status_code < 400),
                    error_type=selected.error_type or "",
                    error_message=selected.error_message or "",
                    elapsed_seconds=selected.elapsed_seconds,
                    retry_count=selected.retry_count,
                    html_length=selected.html_length,
                    final_url=selected.final_url or "",
                )
                parsed = parse_article_html(
                    html=selected.html,
                    discovered_record=discovered_record,
                    request_url=request_url,
                    final_url=selected.final_url,
                    http_status=selected.status_code,
                    redirected=selected.redirected,
                    used_ip_access=selected.used_ip_access,
                    ssl_verify_disabled=selected.ssl_verify_disabled,
                )
                article = parsed.article
                parsed_media = parsed.media_assets
                parsed_links = parsed.article_links
                if options.save_html and not options.dry_run:
                    _save_article_html(
                        options.raw_html_dir,
                        article_id,
                        article.get("source_id", ""),
                        article.get("slug", ""),
                        selected.html,
                    )
            else:
                article = build_error_article(
                    discovered_record=discovered_record,
                    request_url=request_url,
                    final_url=selected.final_url,
                    http_status=selected.status_code,
                    redirected=selected.redirected,
                    used_ip_access=selected.used_ip_access,
                    ssl_verify_disabled=selected.ssl_verify_disabled,
                    error_message=selected.error_message
                    or f"HTTP status {selected.status_code}",
                )
                parsed_media = []
                parsed_links = []
        except Exception as exc:  # Defensive: keep crawl moving and preserve the URL.
            elapsed = time.perf_counter() - started if "started" in locals() else 0
            article = build_error_article(
                discovered_record=discovered_record,
                request_url=request_url,
                final_url="",
                http_status=None,
                redirected=False,
                used_ip_access=False,
                ssl_verify_disabled=False,
                error_message=f"{exc.__class__.__name__}: {exc}",
            )
            parsed_media = []
            parsed_links = []
            fetch_log = _crawl_log_record(
                log_id=f"log_{len(crawl_logs) + 1:08d}",
                discovered_record=discovered_record,
                request_url=request_url,
                action="fetch_article",
                status_code=None,
                success=False,
                error_type=exc.__class__.__name__,
                error_message=str(exc),
                elapsed_seconds=elapsed,
                retry_count=0,
                html_length=0,
                final_url="",
            )

        article["article_id"] = article_id
        article["collected_at"] = collected_at
        article["updated_at"] = collected_at
        _assign_child_ids(article_id, parsed_media, parsed_links, len(media_assets), len(article_links))
        articles.append(article)
        media_assets.extend(parsed_media)
        article_links.extend(parsed_links)
        if fetch_log:
            crawl_logs.append(fetch_log)

        if not options.dry_run:
            _append_jsonl(output_paths["articles_jsonl"], article)
            _append_csv_rows(output_paths["media_csv"], MEDIA_FIELDS, parsed_media)
            _append_csv_rows(output_paths["links_csv"], LINK_FIELDS, parsed_links)
            _append_csv_rows(output_paths["crawl_logs_csv"], CRAWL_LOG_FIELDS, [fetch_log] if fetch_log else [])

    report = build_crawl_report(
        input_records=filtered_records,
        articles=articles[-attempted:] if attempted else [],
        total_input_urls=total_input_urls,
        attempted=attempted,
        skipped_existing=skipped_existing,
        output_paths=output_paths,
        dry_run=options.dry_run,
    )

    if not options.dry_run:
        _write_outputs(articles, media_assets, article_links, crawl_logs, report, output_paths)
    return report


def read_discovered_records(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Discovery input not found: {input_path}")
    if input_path.suffix.lower() == ".csv":
        with input_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            return list(csv.DictReader(csv_file))
    return _read_jsonl(input_path)


def _limited_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit >= len(records):
        return records
    if records and all(
        record.get("source_environment") == "old_subdomain_portal" for record in records
    ):
        return _round_robin_by_source(records, limit)
    return records[:limit]


def _round_robin_by_source(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_order: list[str] = []
    for record in records:
        source_id = str(record.get("source_id") or "")
        if source_id not in by_source:
            source_order.append(source_id)
        by_source[source_id].append(record)

    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(by_source.values()):
        for source_id in source_order:
            if by_source[source_id] and len(selected) < limit:
                selected.append(by_source[source_id].pop(0))
    return selected


def _select_oldportal_article_attempt(
    selected: Any,
    discovered_record: dict[str, Any],
    request_url: str,
    client: HttpClient,
):
    if discovered_record.get("source_environment") != "old_subdomain_portal":
        return selected
    if not _oldportal_article_attempt_is_weak(selected):
        return selected

    best = selected
    for _ in range(2):
        retry = client.fetch_url(request_url).selected
        if _oldportal_article_attempt_score(retry) > _oldportal_article_attempt_score(best):
            best = retry
        if retry.status_code and 200 <= retry.status_code < 400 and not _oldportal_article_attempt_is_weak(retry):
            return retry
    return best


def _oldportal_article_attempt_is_weak(attempt: Any) -> bool:
    html = attempt.html or ""
    if not attempt.status_code or not (200 <= attempt.status_code < 400):
        return True
    if len(html) < 100000:
        return True
    lowered = html.lower()
    return "itens_noticia" not in lowered and "barra_central" not in lowered


def _oldportal_article_attempt_score(attempt: Any) -> int:
    html = attempt.html or ""
    score = len(html)
    lowered = html.lower()
    if attempt.status_code and 200 <= attempt.status_code < 400:
        score += 100000
    if "itens_noticia" in lowered:
        score += 500000
    if "barra_central" in lowered:
        score += 250000
    return score


def print_crawl_report(report: dict[str, Any], preview_count: int = 5) -> None:
    print("")
    print("IFMT crawl report")
    print(f"Generated at: {report['generated_at']}")
    print(f"Total input URLs: {report['total_input_urls']}")
    print(f"Total attempted: {report['total_attempted']}")
    print(f"Total collected: {report['total_collected']}")
    print(f"Total needs review: {report['total_needs_review']}")
    print(f"Total errors: {report['total_errors']}")
    print(f"With title: {report['total_with_title']}")
    print(f"With body: {report['total_with_body']}")
    print(f"With publication date: {report['total_with_publication_date']}")
    print(f"With author or publisher: {report['total_with_author_or_publisher']}")
    print(f"With images: {report['total_with_images']}")
    print(f"With links: {report['total_with_links']}")
    print(f"With attachments: {report['total_with_attachments']}")
    print(f"Average word count: {report['average_word_count']}")
    if not report.get("dry_run"):
        print(f"Articles JSONL: {report['output_paths']['articles_jsonl']}")
        print(f"Articles CSV: {report['output_paths']['articles_csv']}")
        print(f"Media CSV: {report['output_paths']['media_csv']}")
        print(f"Links CSV: {report['output_paths']['links_csv']}")
        print(f"Crawl logs CSV: {report['output_paths']['crawl_logs_csv']}")
        print(f"Manual sample: {report['output_paths']['manual_review_csv']}")
        print(f"Report: {report['output_paths']['report']}")
    else:
        print("Dry run: no output files written")

    print("")
    print(f"Examples collected ({min(preview_count, len(report['examples_collected']))}):")
    for example in report["examples_collected"][:preview_count]:
        print(
            "  - "
            f"{example.get('title') or '(no title)'} | "
            f"{example.get('publication_date') or example.get('date_label_raw') or '(no date)'} | "
            f"{example.get('publisher_unit') or example.get('campus_name') or '(no unit)'} | "
            f"{example.get('word_count')} words"
        )
    print("")


def build_crawl_report(
    input_records: list[dict[str, Any]],
    articles: list[dict[str, Any]],
    total_input_urls: int,
    attempted: int,
    skipped_existing: int,
    output_paths: dict[str, Path],
    dry_run: bool,
) -> dict[str, Any]:
    status_counts = Counter(article.get("status", "") for article in articles)
    source_counts = Counter(article.get("source_environment", "") for article in articles)
    word_counts = [int(article.get("word_count") or 0) for article in articles]
    errors = Counter(
        article.get("error_message", "")
        for article in articles
        if article.get("error_message")
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_input_urls": total_input_urls,
        "total_selected_urls": len(input_records),
        "total_attempted": attempted,
        "total_skipped_existing": skipped_existing,
        "total_collected": status_counts.get("collected", 0),
        "total_needs_review": status_counts.get("needs_review", 0),
        "total_errors": status_counts.get("error", 0),
        "total_by_source_environment": dict(source_counts),
        "total_with_title": sum(1 for article in articles if article.get("title")),
        "total_with_body": sum(1 for article in articles if article.get("body_text")),
        "total_with_publication_date": sum(1 for article in articles if article.get("publication_date")),
        "total_with_author_or_publisher": sum(
            1 for article in articles if article.get("author_name") or article.get("publisher_unit")
        ),
        "total_with_images": sum(1 for article in articles if article.get("has_images")),
        "total_with_links": sum(1 for article in articles if article.get("has_links")),
        "total_with_attachments": sum(1 for article in articles if article.get("has_attachments")),
        "average_word_count": round(sum(word_counts) / len(word_counts), 2) if word_counts else 0,
        "top_error_messages": [
            {"error_message": message, "count": count}
            for message, count in errors.most_common(10)
        ],
        "examples_collected": _example_articles(
            [article for article in articles if article.get("status") == "collected"], 5
        ),
        "examples_needs_review": _example_articles(
            [article for article in articles if article.get("status") == "needs_review"], 5
        ),
        "examples_errors": _example_articles(
            [article for article in articles if article.get("status") == "error"], 5
        ),
        "output_paths": {key: str(path) for key, path in output_paths.items()},
        "dry_run": dry_run,
    }


def _write_outputs(
    articles: list[dict[str, Any]],
    media_assets: list[dict[str, Any]],
    article_links: list[dict[str, Any]],
    crawl_logs: list[dict[str, Any]],
    report: dict[str, Any],
    output_paths: dict[str, Path],
) -> None:
    # JSONL/media/link/log outputs are appended incrementally during crawl. Rewrite
    # them here as a final normalization pass so resumed runs keep stable headers.
    _write_jsonl(output_paths["articles_jsonl"], articles)
    _write_csv(output_paths["articles_csv"], NEWS_ARTICLE_FIELDS, articles)
    _write_csv(output_paths["media_csv"], MEDIA_FIELDS, media_assets)
    _write_csv(output_paths["links_csv"], LINK_FIELDS, article_links)
    _write_csv(output_paths["crawl_logs_csv"], CRAWL_LOG_FIELDS, crawl_logs)
    _write_csv(
        output_paths["manual_review_csv"],
        MANUAL_REVIEW_FIELDS,
        build_manual_review_sample(articles),
    )
    output_paths["report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_manual_review_sample(articles: list[dict[str, Any]], max_records: int = 100) -> list[dict[str, Any]]:
    by_environment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for article in articles:
        by_environment[article.get("source_environment", "")].append(article)

    sample: list[dict[str, Any]] = []
    while len(sample) < max_records and any(by_environment.values()):
        for environment in sorted(list(by_environment)):
            if by_environment[environment] and len(sample) < max_records:
                sample.append(_manual_review_record(by_environment[environment].pop(0)))
    return sample


def _manual_review_record(article: dict[str, Any]) -> dict[str, Any]:
    preview = (article.get("body_text") or "").replace("\n", " ")
    return {
        "article_id": article.get("article_id", ""),
        "source_environment": article.get("source_environment", ""),
        "title": article.get("title", ""),
        "publication_date": article.get("publication_date", ""),
        "date_label_raw": article.get("date_label_raw", ""),
        "author_name": article.get("author_name", ""),
        "publisher_unit": article.get("publisher_unit", ""),
        "campus_name": article.get("campus_name", ""),
        "original_url": article.get("original_url", ""),
        "final_url_after_redirect": article.get("final_url_after_redirect", ""),
        "body_text_preview": preview[:500],
        "word_count": article.get("word_count", 0),
        "has_images": article.get("has_images", False),
        "has_links": article.get("has_links", False),
        "status": article.get("status", ""),
        "error_message": article.get("error_message", ""),
    }


def _assign_child_ids(
    article_id: str,
    media_assets: list[dict[str, Any]],
    article_links: list[dict[str, Any]],
    media_offset: int,
    links_offset: int,
) -> None:
    for index, media in enumerate(media_assets, start=media_offset + 1):
        media["media_id"] = f"media_{index:08d}"
        media["article_id"] = article_id
    for index, link in enumerate(article_links, start=links_offset + 1):
        link["link_id"] = f"link_{index:08d}"
        link["article_id"] = article_id


def _crawl_log_record(
    log_id: str,
    discovered_record: dict[str, Any],
    request_url: str,
    action: str,
    status_code: int | None,
    success: bool,
    error_type: str,
    error_message: str,
    elapsed_seconds: float,
    retry_count: int,
    html_length: int,
    final_url: str,
) -> dict[str, Any]:
    return {
        "log_id": log_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_id": discovered_record.get("source_id", ""),
        "url": request_url,
        "action": action,
        "status_code": status_code,
        "success": success,
        "error_type": error_type,
        "error_message": error_message,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "retry_count": retry_count,
        "html_length": html_length,
        "final_url": final_url,
    }


def _save_article_html(
    raw_html_dir: Path,
    article_id: str,
    source_id: str,
    slug: str,
    html: str,
) -> None:
    source_dir = raw_html_dir / (source_id or "unknown")
    source_dir.mkdir(parents=True, exist_ok=True)
    safe_slug = "".join(char if char.isalnum() or char in "-_" else "-" for char in slug)[:80]
    filename = f"{article_id}_{safe_slug or 'article'}.html"
    (source_dir / filename).write_text(html, encoding="utf-8", errors="ignore")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as jsonl_file:
        jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_csv(path: Path, fields: list[str], records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            csv_record = {}
            for field in fields:
                value = record.get(field, "")
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                csv_record[field] = value
            writer.writerow(csv_record)


def _append_csv_rows(path: Path, fields: list[str], records: Iterable[dict[str, Any]]) -> None:
    rows = list(records)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for record in rows:
            csv_record = {}
            for field in fields:
                value = record.get(field, "")
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                csv_record[field] = value
            writer.writerow(csv_record)


def _initialize_incremental_outputs(output_paths: dict[str, Path]) -> None:
    output_paths["articles_jsonl"].write_text("", encoding="utf-8")
    _write_csv(output_paths["media_csv"], MEDIA_FIELDS, [])
    _write_csv(output_paths["links_csv"], LINK_FIELDS, [])
    _write_csv(output_paths["crawl_logs_csv"], CRAWL_LOG_FIELDS, [])


def _ensure_incremental_outputs(output_paths: dict[str, Path]) -> None:
    output_paths["articles_jsonl"].parent.mkdir(parents=True, exist_ok=True)
    if not output_paths["articles_jsonl"].exists():
        output_paths["articles_jsonl"].write_text("", encoding="utf-8")
    for key, fields in (
        ("media_csv", MEDIA_FIELDS),
        ("links_csv", LINK_FIELDS),
        ("crawl_logs_csv", CRAWL_LOG_FIELDS),
    ):
        if not output_paths[key].exists():
            _write_csv(output_paths[key], fields, [])


def _example_articles(articles: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    examples = []
    for article in articles[:limit]:
        examples.append(
            {
                "article_id": article.get("article_id", ""),
                "title": article.get("title", ""),
                "publication_date": article.get("publication_date", ""),
                "date_label_raw": article.get("date_label_raw", ""),
                "publisher_unit": article.get("publisher_unit", ""),
                "campus_name": article.get("campus_name", ""),
                "word_count": article.get("word_count", 0),
                "status": article.get("status", ""),
                "error_message": article.get("error_message", ""),
                "original_url": article.get("original_url", ""),
            }
        )
    return examples


def _output_paths(output_dir: Path, output_prefix: str) -> dict[str, Path]:
    if output_prefix == "news_articles":
        return {
            "articles_jsonl": output_dir / "news_articles_latest.jsonl",
            "articles_csv": output_dir / "news_articles_latest.csv",
            "media_csv": output_dir / "media_assets_latest.csv",
            "links_csv": output_dir / "article_links_latest.csv",
            "crawl_logs_csv": output_dir / "crawl_logs_latest.csv",
            "report": output_dir / "crawl_report_latest.json",
            "manual_review_csv": output_dir / "manual_review_sample.csv",
        }
    if output_prefix == "news_articles_antigoportal":
        return {
            "articles_jsonl": output_dir / "news_articles_antigoportal_latest.jsonl",
            "articles_csv": output_dir / "news_articles_antigoportal_latest.csv",
            "media_csv": output_dir / "media_assets_antigoportal_latest.csv",
            "links_csv": output_dir / "article_links_antigoportal_latest.csv",
            "crawl_logs_csv": output_dir / "crawl_logs_antigoportal_latest.csv",
            "report": output_dir / "crawl_report_antigoportal_latest.json",
            "manual_review_csv": output_dir / "manual_review_antigoportal_latest.csv",
        }
    return {
        "articles_jsonl": output_dir / f"{output_prefix}_latest.jsonl",
        "articles_csv": output_dir / f"{output_prefix}_latest.csv",
        "media_csv": output_dir / f"{output_prefix}_media_assets_latest.csv",
        "links_csv": output_dir / f"{output_prefix}_article_links_latest.csv",
        "crawl_logs_csv": output_dir / f"{output_prefix}_crawl_logs_latest.csv",
        "report": output_dir / f"{output_prefix}_crawl_report_latest.json",
        "manual_review_csv": output_dir / f"{output_prefix}_manual_review_sample.csv",
    }
