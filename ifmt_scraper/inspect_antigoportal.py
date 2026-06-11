from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePosixPath, Path
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from .crawl import _limited_records, _select_oldportal_article_attempt, read_discovered_records
from .http_client import HttpClient
from .normalize import calculate_word_count, normalize_text
from .parse_articles import (
    DOCUMENT_EXTENSIONS,
    OLDPORTAL_CONTENT_TOKEN,
    _classify_link,
    _extract_oldportal_article_fields,
)
from .parse_articles import parse_article_html
from .url_utils import normalize_url, resolve_link


@dataclass
class AntigoportalInspectionOptions:
    input_path: Path = Path("data/output/discovered_urls_antigoportal_latest.jsonl")
    output_dir: Path = Path("data/output")
    limit: int = 20
    timeout: float = 30.0
    delay_seconds: float = 0.25
    max_retries: int = 2


def run_antigoportal_inspection(options: AntigoportalInspectionOptions) -> dict[str, Any]:
    options.output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        record
        for record in read_discovered_records(options.input_path)
        if record.get("source_environment") == "old_subdomain_portal"
    ]
    selected_records = _limited_records(records, options.limit)
    client = HttpClient(
        timeout=options.timeout,
        delay_seconds=options.delay_seconds,
        max_retries=options.max_retries,
    )

    inspections = []
    for record in selected_records:
        inspections.append(_inspect_record(record, client))

    summary = _inspection_summary(inspections)
    report = {
        "generated_at": summary["generated_at"],
        "input_path": str(options.input_path),
        "limit": options.limit,
        "total_available": len(records),
        "total_inspected": len(inspections),
        "summary": summary,
        "records": inspections,
        "output_files": {
            "json": str(options.output_dir / "antigoportal_inspection_latest.json"),
            "md": str(options.output_dir / "antigoportal_inspection_latest.md"),
        },
    }

    json_path = options.output_dir / "antigoportal_inspection_latest.json"
    md_path = options.output_dir / "antigoportal_inspection_latest.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_inspection_markdown(report), encoding="utf-8")
    return report


def print_antigoportal_inspection_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("")
    print("IFMT antigoportal inspection report")
    print(f"Total available: {report['total_available']}")
    print(f"Total inspected: {report['total_inspected']}")
    print(f"Collected: {summary['status_counts'].get('collected', 0)}")
    print(f"Needs review: {summary['status_counts'].get('needs_review', 0)}")
    print(f"Errors: {summary['status_counts'].get('error', 0)}")
    print(f"With normalized date: {summary['with_normalized_date']}")
    print(f"With links: {summary['with_links']}")
    print(f"With document-like links: {summary['with_document_like_links']}")
    print(f"Average words: {summary['average_word_count']}")
    print(f"Median words: {summary['median_word_count']}")
    print(f"JSON: {report['output_files']['json']}")
    print(f"Markdown: {report['output_files']['md']}")
    print("")


def _inspect_record(record: dict[str, Any], client: HttpClient) -> dict[str, Any]:
    url = record.get("discovered_url") or record.get("normalized_url") or ""
    fetch_result = client.fetch_url(url)
    selected = _select_oldportal_article_attempt(
        selected=fetch_result.selected,
        discovered_record=record,
        request_url=url,
        client=client,
    )
    html = selected.html or ""
    final_url = selected.final_url or url
    soup = BeautifulSoup(html, "lxml")

    parsed = parse_article_html(
        html=html,
        discovered_record=record,
        request_url=url,
        final_url=final_url,
        http_status=selected.status_code,
        redirected=selected.redirected,
        used_ip_access=selected.used_ip_access,
        ssl_verify_disabled=selected.ssl_verify_disabled,
    )
    article = parsed.article
    oldportal_fields = _extract_oldportal_article_fields(soup, record)
    base_url = final_url or url
    link_summary = _link_summary(soup, base_url, final_url)

    return {
        "url": url,
        "host": urlparse(final_url or url).hostname or "",
        "status": selected.status_code,
        "redirected": selected.redirected,
        "final_url": final_url,
        "html_length": selected.html_length,
        "title_from_listing": record.get("title_from_listing", ""),
        "page_title": normalize_text(soup.title.get_text(" ", strip=True)) if soup.title else "",
        "extracted_title": article.get("title", ""),
        "date_raw": article.get("date_label_raw", ""),
        "date_normalized": article.get("publication_date", ""),
        "word_count": article.get("word_count", 0),
        "candidate_content_blocks": _content_block_candidates(soup),
        "selected_content_selector": _node_descriptor(oldportal_fields.get("content_node")),
        "body_text_preview": normalize_text(article.get("body_text", ""))[:500],
        "total_links": link_summary["total_links"],
        "link_type_counts": link_summary["type_counts"],
        "link_extensions": link_summary["extensions"],
        "first_links": link_summary["examples"],
        "article_link_type_counts": dict(Counter(link.get("link_type", "") for link in parsed.article_links)),
        "article_document_like_links": [
            link
            for link in parsed.article_links
            if link.get("link_type") in {"document", "edital", "drive"}
        ][:10],
        "needs_review_reason": article.get("error_message", "") if article.get("status") != "collected" else "",
        "extraction_status": article.get("status", ""),
    }


def _content_block_candidates(soup: BeautifulSoup) -> list[dict[str, Any]]:
    for tag in soup.find_all(["script", "style", "noscript", "svg", "form"]):
        tag.decompose()

    selectors = (
        "main",
        "article",
        ".itens_noticia",
        "#barra_central",
        "#corpo",
        "#corpo_geral",
        "#content",
        "#conteudo",
        ".content",
        ".conteudo",
        ".noticia",
        ".post",
        ".texto",
        ".principal",
        ".miolo",
    )
    nodes = []
    seen: set[int] = set()

    def add(node) -> None:
        if node is None or id(node) in seen:
            return
        seen.add(id(node))
        nodes.append(node)

    for selector in selectors:
        for node in soup.select(selector):
            add(node)
    for node in soup.find_all(["section", "div"]):
        token_text = " ".join(
            str(value)
            for attr in ("class", "id", "role", "aria-label")
            for value in _attr_values(node.get(attr))
        )
        if token_text and OLDPORTAL_CONTENT_TOKEN.search(token_text):
            add(node)

    candidates = []
    for node in nodes:
        text = normalize_text(node.get_text("\n", strip=True))
        candidates.append(
            {
                "selector": _node_descriptor(node),
                "tag": getattr(node, "name", ""),
                "id": str(node.get("id", "")) if hasattr(node, "get") else "",
                "classes": " ".join(_attr_values(node.get("class"))) if hasattr(node, "get") else "",
                "char_count": len(text),
                "word_count": calculate_word_count(text),
                "paragraph_count": len(node.find_all("p")) if hasattr(node, "find_all") else 0,
                "link_count": len(node.find_all("a", href=True)) if hasattr(node, "find_all") else 0,
                "text_preview": text[:240],
            }
        )
    return sorted(candidates, key=lambda item: item["char_count"], reverse=True)[:20]


def _link_summary(soup: BeautifulSoup, base_url: str, article_url: str) -> dict[str, Any]:
    total = 0
    type_counts: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    examples = []
    article_host = urlparse(article_url).hostname or ""
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        resolved = resolve_link(str(anchor.get("href") or ""), base_url)
        if not resolved:
            continue
        normalized = normalize_url(resolved)
        if normalized in seen:
            continue
        seen.add(normalized)
        total += 1
        parsed = urlparse(resolved)
        extension = PurePosixPath(parsed.path).suffix.lower()
        if extension:
            extensions[extension] += 1
        anchor_text = normalize_text(anchor.get_text(" ", strip=True))
        link_type = _classify_link(resolved, anchor_text, article_host, extension)
        type_counts[link_type] += 1
        if len(examples) < 20:
            examples.append(
                {
                    "url": resolved,
                    "anchor_text": anchor_text,
                    "link_type": link_type,
                    "extension": extension,
                    "document_extension": extension in DOCUMENT_EXTENSIONS,
                }
            )

    return {
        "total_links": total,
        "type_counts": dict(type_counts),
        "extensions": dict(extensions),
        "examples": examples,
    }


def _inspection_summary(inspections: list[dict[str, Any]]) -> dict[str, Any]:
    from datetime import datetime, timezone

    word_counts = [int(item.get("word_count") or 0) for item in inspections]
    sorted_words = sorted(word_counts)
    median = 0
    if sorted_words:
        midpoint = len(sorted_words) // 2
        if len(sorted_words) % 2:
            median = sorted_words[midpoint]
        else:
            median = round((sorted_words[midpoint - 1] + sorted_words[midpoint]) / 2, 2)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status_counts": dict(Counter(item.get("extraction_status", "") for item in inspections)),
        "hosts": dict(Counter(item.get("host", "") for item in inspections)),
        "with_normalized_date": sum(1 for item in inspections if item.get("date_normalized")),
        "with_links": sum(1 for item in inspections if item.get("total_links")),
        "with_document_like_links": sum(
            1 for item in inspections if item.get("article_document_like_links")
        ),
        "average_word_count": round(sum(word_counts) / len(word_counts), 2) if word_counts else 0,
        "median_word_count": median,
        "link_type_counts": dict(
            sum((Counter(item.get("link_type_counts", {})) for item in inspections), Counter())
        ),
    }


def _inspection_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Antigoportal inspection",
        "",
        f"- Total available: {report['total_available']}",
        f"- Total inspected: {report['total_inspected']}",
        f"- Status counts: {summary['status_counts']}",
        f"- With normalized date: {summary['with_normalized_date']}",
        f"- With links: {summary['with_links']}",
        f"- With document-like links: {summary['with_document_like_links']}",
        f"- Average words: {summary['average_word_count']}",
        f"- Median words: {summary['median_word_count']}",
        "",
    ]
    for item in report["records"]:
        lines.extend(
            [
                f"## {item['host']} - {item['status']}",
                "",
                f"- URL: {item['url']}",
                f"- Listing title: {item['title_from_listing']}",
                f"- Extracted title: {item['extracted_title']}",
                f"- Date raw/normalized: {item['date_raw']} / {item['date_normalized']}",
                f"- Words: {item['word_count']}",
                f"- Links: {item['total_links']} {item['link_type_counts']}",
                f"- Selected block: {item['selected_content_selector']}",
                f"- Needs review: {item['needs_review_reason'] or '(no)'}",
                "",
                "Body preview:",
                "",
                "```text",
                item["body_text_preview"],
                "```",
                "",
                "Top content candidates:",
            ]
        )
        for candidate in item["candidate_content_blocks"][:5]:
            lines.append(
                f"- {candidate['selector']}: {candidate['char_count']} chars, "
                f"{candidate['word_count']} words, {candidate['paragraph_count']} paragraphs"
            )
        lines.append("")
    return "\n".join(lines)


def _node_descriptor(node) -> str:
    if node is None:
        return ""
    tag = getattr(node, "name", "") or ""
    node_id = node.get("id") if hasattr(node, "get") else ""
    classes = _attr_values(node.get("class")) if hasattr(node, "get") else []
    if node_id:
        return f"{tag}#{node_id}"
    if classes:
        return f"{tag}." + ".".join(classes[:4])
    return tag


def _attr_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []
