from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import DEFAULT_DIAGNOSTIC_SOURCES, LEGACY_IP_HOST, SourceConfig
from .http_client import FetchResult, HttpClient
from .parse_listings import extract_news_links, extract_page_title, page_fingerprint


def run_diagnostics(
    sources: list[SourceConfig] | tuple[SourceConfig, ...] = DEFAULT_DIAGNOSTIC_SOURCES,
    output_dir: Path = Path("data/output"),
    raw_html_dir: Path = Path("data/raw_html"),
    save_html: bool = True,
    timeout: float = 30.0,
    delay_seconds: float = 0.5,
    max_retries: int = 2,
    preview_links: int = 5,
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_run_dir = raw_html_dir / "diagnose" / timestamp
    if save_html:
        raw_run_dir.mkdir(parents=True, exist_ok=True)

    client = HttpClient(
        timeout=timeout,
        delay_seconds=delay_seconds,
        max_retries=max_retries,
    )

    results = [
        diagnose_source(
            source=source,
            client=client,
            raw_run_dir=raw_run_dir if save_html else None,
            preview_links=preview_links,
        )
        for source in sources
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    report_path = output_dir / f"diagnostic_report_{timestamp}.json"
    latest_path = output_dir / "diagnostic_report_latest.json"
    report["report_path"] = str(report_path)
    report["latest_report_path"] = str(latest_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    latest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def diagnose_source(
    source: SourceConfig,
    client: HttpClient,
    raw_run_dir: Path | None,
    preview_links: int = 5,
) -> dict[str, Any]:
    fetch_result = client.fetch_url(source.news_archive_url)
    selected = fetch_result.selected
    link_base_url = _link_base_url(source.news_archive_url, selected.final_url)
    all_candidates = extract_news_links(selected.html, link_base_url)
    preview_candidates = all_candidates[:preview_links]

    raw_html_path = None
    if raw_run_dir is not None and selected.html:
        raw_html_path = raw_run_dir / f"{source.source_id}.html"
        raw_html_path.write_text(selected.html, encoding="utf-8", errors="ignore")

    relative_candidates = [
        candidate for candidate in all_candidates if candidate.relative_href
    ]
    ip_relative_candidates = [
        candidate
        for candidate in relative_candidates
        if urlparse(candidate.url).hostname == LEGACY_IP_HOST
    ]
    is_ip_source = urlparse(source.news_archive_url).hostname == LEGACY_IP_HOST

    return {
        "source": {
            "source_id": source.source_id,
            "source_name": source.source_name,
            "source_type": source.source_type,
            "source_environment": source.source_environment,
            "source_access_mode": source.source_access_mode,
            "base_url": source.base_url,
            "news_archive_url": source.news_archive_url,
            "campus_code": source.campus_code,
            "campus_name": source.campus_name,
            "ssl_verify": source.ssl_verify,
            "notes": source.notes,
        },
        "fetch": fetch_result.to_dict(include_html=False),
        "diagnostics": {
            "tested_url": source.news_archive_url,
            "status_code": selected.status_code,
            "final_url": selected.final_url,
            "redirected": selected.redirected,
            "html_length": selected.html_length,
            "page_title": extract_page_title(selected.html),
            "candidate_news_links_count": len(all_candidates),
            "first_candidate_news_links": [
                candidate.to_dict() for candidate in preview_candidates
            ],
            "ssl_error": _first_error_of_type(fetch_result, "SSLError"),
            "ssl_verify_disabled": selected.ssl_verify_disabled,
            "used_ip_access": selected.used_ip_access,
            "html_fingerprint": page_fingerprint(selected.html) if selected.html else None,
            "raw_html_path": str(raw_html_path) if raw_html_path else None,
            "ip_relative_links_preserved": (
                len(relative_candidates) == len(ip_relative_candidates)
                if is_ip_source and relative_candidates
                else None
            ),
            "relative_candidate_links_count": len(relative_candidates),
            "relative_candidate_links_with_ip_count": len(ip_relative_candidates),
        },
    }


def print_diagnostic_report(report: dict[str, Any]) -> None:
    print("")
    print("IFMT diagnostic report")
    print(f"Generated at: {report['generated_at']}")
    print(f"Saved JSON: {report['latest_report_path']}")
    print("")

    for result in report["results"]:
        source = result["source"]
        diagnostics = result["diagnostics"]
        fetch = result["fetch"]
        selected = fetch["selected"]
        print(f"== {source['source_id']} - {source['source_name']} ==")
        print(f"Tested URL: {diagnostics['tested_url']}")
        for index, attempt in enumerate(fetch["attempts"], start=1):
            print(
                "Attempt "
                f"{index}: status={attempt['status_code']} "
                f"verify={attempt['verify']} "
                f"ssl_verify_disabled={attempt['ssl_verify_disabled']} "
                f"redirected={attempt['redirected']} "
                f"elapsed={attempt['elapsed_seconds']:.2f}s"
            )
            print(f"  request_url: {attempt['request_url']}")
            print(f"  final_url:   {attempt['final_url']}")
            if attempt["error_type"]:
                print(f"  error: {attempt['error_type']}: {attempt['error_message']}")
        print(f"Selected status: {selected['status_code']}")
        print(f"Selected final URL: {diagnostics['final_url']}")
        print(f"Redirected: {diagnostics['redirected']}")
        print(f"HTML length: {diagnostics['html_length']}")
        print(f"Page title: {diagnostics['page_title']}")
        print(f"HTML fingerprint: {diagnostics['html_fingerprint']}")
        print(f"Candidate news links: {diagnostics['candidate_news_links_count']}")
        if diagnostics["ip_relative_links_preserved"] is not None:
            print(
                "Relative IP links preserved: "
                f"{diagnostics['ip_relative_links_preserved']} "
                f"({diagnostics['relative_candidate_links_with_ip_count']}/"
                f"{diagnostics['relative_candidate_links_count']})"
            )
        for candidate in diagnostics["first_candidate_news_links"]:
            label = candidate["text"] or "(no text)"
            print(f"  - [{candidate['pattern']}] {label} -> {candidate['url']}")
        print("")


def _link_base_url(request_url: str, final_url: str | None) -> str:
    if urlparse(request_url).hostname == LEGACY_IP_HOST:
        if final_url and urlparse(final_url).hostname == LEGACY_IP_HOST:
            return final_url
        return request_url
    return final_url or request_url


def _first_error_of_type(fetch_result: FetchResult, error_type: str) -> str | None:
    for attempt in fetch_result.attempts:
        if attempt.error_type == error_type:
            return attempt.error_message
    return None
