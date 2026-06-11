from __future__ import annotations

import csv
import json
import re
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from requests import Response
from urllib3.exceptions import InsecureRequestWarning

from .config import DEFAULT_HEADERS
from .normalize import calculate_word_count, normalize_text
from .parse_articles import parse_article_html
from .parse_listings import CandidateLink, extract_news_links, extract_page_title, page_fingerprint
from .url_utils import has_static_extension, is_relative_href, normalize_url, resolve_link


SOURCE_ENVIRONMENT = "old_subdomain_portal"

ANTIGOPORTAL_LISTED_URLS = [
    "https://antigoportal.ifmt.edu.br/noticias/",
    "http://antigoportal.ifmt.edu.br/noticias/",
    "https://cba.antigoportal.ifmt.edu.br/noticias/",
    "http://cba.antigoportal.ifmt.edu.br/noticias/",
    "https://svc.antigoportal.ifmt.edu.br/noticias/",
    "http://svc.antigoportal.ifmt.edu.br/noticias/",
    "https://lrv.antigoportal.ifmt.edu.br/noticias/",
    "http://lrv.antigoportal.ifmt.edu.br/noticias/",
    "https://tga.antigoportal.ifmt.edu.br/noticias/",
    "http://tga.antigoportal.ifmt.edu.br/noticias/",
    "https://bag.antigoportal.ifmt.edu.br/noticias/",
    "https://pdl.antigoportal.ifmt.edu.br/noticias/",
    "https://cfs.antigoportal.ifmt.edu.br/noticias/",
    "https://cas.antigoportal.ifmt.edu.br/noticias/",
    "https://blv.antigoportal.ifmt.edu.br/noticias/",
]

KNOWN_EXTRA_OLD_SUBDOMAIN_CODES = [
    "alf",
    "bag",
    "blv",
    "cas",
    "cfs",
    "cnp",
    "jna",
    "lrv",
    "pdl",
    "plc",
    "roo",
    "snp",
    "srs",
    "svc",
    "tga",
    "vgd",
]

CSV_FIELDS = [
    "source_environment",
    "tested_url",
    "source_host",
    "status_code",
    "opened",
    "redirected",
    "final_url",
    "html_length",
    "page_title",
    "verify_false_needed",
    "ssl_error",
    "candidate_news_links_count",
    "news_link_pattern_counts",
    "individual_url_path_patterns",
    "pagination_patterns_observed",
    "pagination_works",
    "recommended_for_discover_crawl",
    "first_10_candidate_links",
    "pagination_tests",
    "article_tests",
    "error_type",
    "error_message",
]

_NAV_LABEL_KEYS = {
    "acesso a informacao",
    "institucional",
    "estudante",
    "eventos",
    "servicos",
    "biblioteca",
    "processo seletivo",
    "sistemas",
    "email institucional",
    "ativa incubadora",
    "cursos",
    "a rede",
}


@dataclass
class AntigoportalDiagnosticOptions:
    output_dir: Path = Path("data/output")
    include_known_extra_subdomains: bool = True
    timeout: float = 25.0
    delay_seconds: float = 0.25
    max_article_tests_per_source: int = 3


class AntigoportalDiagnosticClient:
    def __init__(self, timeout: float, delay_seconds: float) -> None:
        self.timeout = timeout
        self.delay_seconds = delay_seconds
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self._last_request_at = 0.0

    def fetch(self, url: str) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        first_attempt = self._request(url, verify=True)
        attempts.append(first_attempt)

        verify_false_needed = False
        if first_attempt.get("error_type") == "SSLError":
            verify_false_needed = True
            attempts.append(self._request(url, verify=False))

        selected = _select_attempt(attempts)
        return {
            "requested_url": url,
            "attempts": attempts,
            "selected": selected,
            "verify_false_needed": verify_false_needed,
            "ssl_error": _first_error(attempts, "SSLError"),
        }

    def _request(self, url: str, verify: bool) -> dict[str, Any]:
        self._respect_delay()
        started = time.perf_counter()
        try:
            response = self._get(url, verify=verify)
            elapsed = time.perf_counter() - started
            return _attempt_from_response(url, response, verify, elapsed)
        except requests.RequestException as exc:
            elapsed = time.perf_counter() - started
            return {
                "request_url": url,
                "final_url": "",
                "status_code": None,
                "success": False,
                "redirected": False,
                "verify": verify,
                "ssl_verify_disabled": not verify,
                "elapsed_seconds": round(elapsed, 4),
                "html": "",
                "html_length": 0,
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            }

    def _get(self, url: str, verify: bool) -> Response:
        if verify:
            return self.session.get(
                url,
                timeout=self.timeout,
                verify=True,
                allow_redirects=True,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InsecureRequestWarning)
            return self.session.get(
                url,
                timeout=self.timeout,
                verify=False,
                allow_redirects=True,
            )

    def _respect_delay(self) -> None:
        if self.delay_seconds <= 0:
            return
        elapsed = time.perf_counter() - self._last_request_at
        wait_seconds = self.delay_seconds - elapsed
        if self._last_request_at and wait_seconds > 0:
            time.sleep(wait_seconds)
        self._last_request_at = time.perf_counter()


def run_antigoportal_diagnostic(options: AntigoportalDiagnosticOptions) -> dict[str, Any]:
    options.output_dir.mkdir(parents=True, exist_ok=True)
    client = AntigoportalDiagnosticClient(
        timeout=options.timeout,
        delay_seconds=options.delay_seconds,
    )

    urls = antigoportal_urls(options.include_known_extra_subdomains)
    listing_results: list[dict[str, Any]] = []
    best_listing_by_host: dict[str, dict[str, Any]] = {}

    for url in urls:
        result = diagnose_listing_url(url, client)
        listing_results.append(result)
        host = result["source_host"]
        current = best_listing_by_host.get(host)
        if current is None or _listing_rank(result) > _listing_rank(current):
            best_listing_by_host[host] = result

    for host, listing in best_listing_by_host.items():
        if not listing["opened"]:
            continue
        pagination_tests = diagnose_pagination(listing, client)
        listing["pagination_tests"] = pagination_tests
        listing["pagination_works"] = any(test["works"] for test in pagination_tests)
        listing["article_tests"] = diagnose_articles(
            listing,
            client,
            max_articles=options.max_article_tests_per_source,
        )
        listing["recommended_for_discover_crawl"] = _recommend_source(listing)

    summary = build_summary(listing_results, best_listing_by_host)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_environment": SOURCE_ENVIRONMENT,
        "tested_urls_count": len(urls),
        "listing_results": listing_results,
        "best_listing_by_host": best_listing_by_host,
        "summary": summary,
        "output_files": {
            "json": str(options.output_dir / "antigoportal_diagnostic_report_latest.json"),
            "csv": str(options.output_dir / "antigoportal_diagnostic_report_latest.csv"),
        },
    }

    json_path = options.output_dir / "antigoportal_diagnostic_report_latest.json"
    csv_path = options.output_dir / "antigoportal_diagnostic_report_latest.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv_report(csv_path, listing_results)
    return report


def diagnose_listing_url(url: str, client: AntigoportalDiagnosticClient) -> dict[str, Any]:
    fetch = client.fetch(url)
    selected = fetch["selected"]
    final_url = selected.get("final_url") or url
    html = selected.get("html") or ""
    base_url = final_url or url
    candidates = extract_antigoportal_news_links(html, base_url)
    patterns = Counter(candidate.pattern for candidate in candidates)
    path_patterns = Counter(_individual_path_pattern(candidate.url) for candidate in candidates)
    pagination_patterns = detect_pagination_patterns(html, base_url)

    return {
        "source_environment": SOURCE_ENVIRONMENT,
        "tested_url": url,
        "source_host": urlparse(url).hostname or "",
        "status_code": selected.get("status_code"),
        "opened": bool(selected.get("success") and html),
        "redirected": bool(selected.get("redirected")),
        "final_url": final_url,
        "html_length": selected.get("html_length") or 0,
        "page_title": extract_page_title(html) or "",
        "verify_false_needed": fetch["verify_false_needed"],
        "ssl_error": fetch["ssl_error"] or "",
        "candidate_news_links_count": len(candidates),
        "news_link_pattern_counts": dict(patterns),
        "individual_url_path_patterns": dict(path_patterns),
        "pagination_patterns_observed": pagination_patterns,
        "pagination_works": False,
        "recommended_for_discover_crawl": False,
        "first_10_candidate_links": [candidate.to_dict() for candidate in candidates[:10]],
        "pagination_tests": [],
        "article_tests": [],
        "html_fingerprint": page_fingerprint(html) if html else "",
        "fetch": _fetch_without_html(fetch),
        "error_type": selected.get("error_type") or "",
        "error_message": selected.get("error_message") or "",
    }


def diagnose_pagination(
    listing: dict[str, Any],
    client: AntigoportalDiagnosticClient,
) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    first_fingerprint = listing.get("html_fingerprint") or ""
    first_link_signature = _link_signature(listing.get("first_10_candidate_links", []))
    base_url = _listing_base_url(listing["final_url"] or listing["tested_url"])

    for pattern_name, url_template in _pagination_templates(base_url):
        for page_number in (2, 10, 50):
            url = url_template.format(page=page_number)
            fetch = client.fetch(url)
            selected = fetch["selected"]
            html = selected.get("html") or ""
            candidates = extract_antigoportal_news_links(html, selected.get("final_url") or url)
            fingerprint = page_fingerprint(html) if html else ""
            link_signature = _candidate_signature(candidates[:10])
            works = bool(
                selected.get("success")
                and candidates
                and fingerprint
                and fingerprint != first_fingerprint
                and link_signature != first_link_signature
            )
            tests.append(
                {
                    "pattern": pattern_name,
                    "page_number": page_number,
                    "url": url,
                    "status_code": selected.get("status_code"),
                    "redirected": bool(selected.get("redirected")),
                    "final_url": selected.get("final_url") or "",
                    "html_length": selected.get("html_length") or 0,
                    "candidate_news_links_count": len(candidates),
                    "first_10_candidate_links": [
                        candidate.to_dict() for candidate in candidates[:10]
                    ],
                    "news_link_pattern_counts": dict(Counter(candidate.pattern for candidate in candidates)),
                    "individual_url_path_patterns": dict(
                        Counter(_individual_path_pattern(candidate.url) for candidate in candidates)
                    ),
                    "same_as_first_page": bool(fingerprint and fingerprint == first_fingerprint),
                    "works": works,
                    "verify_false_needed": fetch["verify_false_needed"],
                    "error_type": selected.get("error_type") or "",
                    "error_message": selected.get("error_message") or "",
                }
            )
    return tests


def diagnose_articles(
    listing: dict[str, Any],
    client: AntigoportalDiagnosticClient,
    max_articles: int,
) -> list[dict[str, Any]]:
    articles: list[dict[str, Any]] = []
    candidates = _article_test_candidates(listing)
    for candidate in candidates[:max_articles]:
        url = candidate["url"]
        fetch = client.fetch(url)
        selected = fetch["selected"]
        html = selected.get("html") or ""
        article_result = {
            "url": url,
            "status_code": selected.get("status_code"),
            "redirected": bool(selected.get("redirected")),
            "final_url": selected.get("final_url") or "",
            "html_length": selected.get("html_length") or 0,
            "verify_false_needed": fetch["verify_false_needed"],
            "extractable": False,
            "title": "",
            "publication_date": "",
            "date_label_raw": "",
            "author_name": "",
            "publisher_unit": "",
            "has_body": False,
            "word_count": 0,
            "image_count": 0,
            "link_count": 0,
            "attachment_count": 0,
            "status": "error",
            "error_message": selected.get("error_message") or "",
        }
        if selected.get("success") and html:
            oldportal_fields = extract_oldportal_article_fields(html, url, candidate)
            parsed = parse_article_html(
                html=html,
                discovered_record=_article_discovered_record(listing, candidate),
                request_url=url,
                final_url=selected.get("final_url") or url,
                http_status=selected.get("status_code"),
                redirected=bool(selected.get("redirected")),
                used_ip_access=False,
                ssl_verify_disabled=fetch["verify_false_needed"],
            )
            article = parsed.article
            word_count = oldportal_fields.get("word_count") or article.get("word_count") or 0
            article_result.update(
                {
                    "extractable": bool((oldportal_fields.get("title") or article.get("title")) and word_count),
                    "title": oldportal_fields.get("title") or article.get("title", ""),
                    "title_from_listing": candidate.get("title") or candidate.get("text") or "",
                    "publication_date": article.get("publication_date", ""),
                    "date_label_raw": oldportal_fields.get("date_label_raw") or article.get("date_label_raw", ""),
                    "author_name": oldportal_fields.get("author_name") or article.get("author_name", ""),
                    "publisher_unit": oldportal_fields.get("publisher_unit") or article.get("publisher_unit", ""),
                    "has_body": bool(word_count),
                    "word_count": word_count,
                    "body_preview": oldportal_fields.get("body_preview", ""),
                    "image_count": len(parsed.media_assets),
                    "link_count": len(parsed.article_links),
                    "attachment_count": sum(
                        1 for link in parsed.article_links if link.get("link_type") == "document"
                    ),
                    "status": article.get("status", ""),
                    "error_message": article.get("error_message", ""),
                }
            )
        articles.append(article_result)
    return articles


def extract_oldportal_article_fields(
    html: str,
    url: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style", "noscript", "svg", "form"]):
        tag.decompose()

    title = candidate.get("title") or candidate.get("text") or _oldportal_title_from_page(soup, url)
    lines = [
        normalize_text(line)
        for line in soup.get_text("\n", strip=True).splitlines()
        if normalize_text(line)
    ]
    title_key = _text_key(title)
    start_index = -1
    for index, line in enumerate(lines):
        if title_key and _text_key(line) == title_key:
            start_index = index
            break
    if start_index < 0 and title_key:
        for index, line in enumerate(lines):
            if title_key in _text_key(line):
                start_index = index
                break

    content_lines: list[str] = []
    if start_index >= 0:
        for line in lines[start_index + 1 :]:
            line_key = _text_key(line)
            if line_key in {"cursos", "a rede"}:
                break
            if line_key.startswith("instituto federal de educacao ciencia e tecnologia de mato grosso - campus"):
                break
            if line_key == "documentos":
                break
            if _looks_like_document_title(line):
                break
            content_lines.append(line)

    author_name = ""
    body_lines = []
    for line in content_lines:
        if not author_name and _looks_like_oldportal_author(line):
            author_name = line
            continue
        body_lines.append(line)

    body_text = "\n\n".join(body_lines)
    return {
        "title": title,
        "date_label_raw": _first_date(body_text or "\n".join(lines)),
        "author_name": author_name,
        "publisher_unit": _publisher_unit_from_author(author_name),
        "word_count": calculate_word_count(body_text),
        "body_preview": normalize_text(body_text)[:500],
    }


def extract_antigoportal_news_links(html: str, base_url: str) -> list[CandidateLink]:
    candidates = list(extract_news_links(html, base_url))
    by_normalized = {normalize_url(candidate.url): candidate for candidate in candidates}
    if not html:
        return candidates

    soup = BeautifulSoup(html, "lxml")
    for anchor in soup.find_all("a", href=True):
        raw_href = str(anchor.get("href") or "")
        resolved = resolve_link(raw_href, base_url)
        if not resolved or has_static_extension(resolved):
            continue
        parsed = urlparse(resolved)
        host = parsed.hostname or ""
        if not _is_antigoportal_host(host):
            continue
        pattern = _oldportal_link_pattern(resolved)
        if not pattern:
            continue
        key = normalize_url(resolved)
        if key in by_normalized:
            continue
        text = _anchor_text(anchor)
        candidate = CandidateLink(
            url=resolved,
            raw_href=raw_href,
            text=text,
            pattern=pattern,
            relative_href=is_relative_href(raw_href),
            title=text,
            summary="",
            date_label_raw=_nearby_date(anchor),
            publisher_unit="",
            image_url=_nearby_image(anchor, base_url),
        )
        by_normalized[key] = candidate

    return list(by_normalized.values())


def detect_pagination_patterns(html: str, base_url: str) -> dict[str, Any]:
    patterns: Counter[str] = Counter()
    examples: dict[str, str] = {}
    if not html:
        return {"pattern_counts": {}, "examples": {}}

    soup = BeautifulSoup(html, "lxml")
    for anchor in soup.find_all("a", href=True):
        resolved = resolve_link(str(anchor.get("href") or ""), base_url)
        if not resolved:
            continue
        parsed = urlparse(resolved)
        path = parsed.path.lower()
        query = parsed.query.lower()
        pattern = ""
        if re.search(r"(^|&)page=\d+", query):
            pattern = "query_page"
        elif re.search(r"(^|&)pagina=\d+", query):
            pattern = "query_pagina"
        elif re.search(r"/page/\d+/?$", path):
            pattern = "path_page"
        elif re.search(r"/pagina/\d+/?$", path):
            pattern = "path_pagina"
        if pattern:
            patterns[pattern] += 1
            examples.setdefault(pattern, resolved)
    return {"pattern_counts": dict(patterns), "examples": examples}


def build_summary(
    listing_results: list[dict[str, Any]],
    best_listing_by_host: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    opened_hosts = sorted(
        host for host, listing in best_listing_by_host.items() if listing.get("opened")
    )
    not_opened_hosts = sorted(
        host for host, listing in best_listing_by_host.items() if not listing.get("opened")
    )
    hosts_with_news = sorted(
        host
        for host, listing in best_listing_by_host.items()
        if _listing_has_any_news(listing)
    )
    hosts_with_working_pagination = sorted(
        host for host, listing in best_listing_by_host.items() if listing.get("pagination_works")
    )
    recommended_hosts = sorted(
        host
        for host, listing in best_listing_by_host.items()
        if listing.get("recommended_for_discover_crawl")
    )

    return {
        "opened_hosts": opened_hosts,
        "not_opened_hosts": not_opened_hosts,
        "hosts_with_news": hosts_with_news,
        "hosts_without_news": sorted(set(best_listing_by_host) - set(hosts_with_news)),
        "hosts_with_working_pagination": hosts_with_working_pagination,
        "recommended_hosts_for_discover_crawl": recommended_hosts,
        "first_page_news_links_by_host": {
            host: listing.get("candidate_news_links_count", 0)
            for host, listing in sorted(best_listing_by_host.items())
        },
        "verify_false_needed_hosts": sorted(
            {
                listing["source_host"]
                for listing in listing_results
                if listing.get("verify_false_needed")
            }
        ),
        "status_counts": dict(Counter(str(result.get("status_code")) for result in listing_results)),
    }


def write_csv_report(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row.get(field, ""), ensure_ascii=False)
                    if isinstance(row.get(field, ""), (list, dict))
                    else row.get(field, "")
                    for field in CSV_FIELDS
                }
            )


def print_antigoportal_diagnostic_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("")
    print("IFMT antigoportal diagnostic report")
    print(f"Generated at: {report['generated_at']}")
    print(f"Source environment: {report['source_environment']}")
    print(f"Tested URLs: {report['tested_urls_count']}")
    print(f"Opened hosts: {', '.join(summary['opened_hosts']) or '(none)'}")
    print(f"Hosts with news: {', '.join(summary['hosts_with_news']) or '(none)'}")
    print(
        "Hosts with working pagination: "
        f"{', '.join(summary['hosts_with_working_pagination']) or '(none)'}"
    )
    print(
        "Recommended for discover/crawl: "
        f"{', '.join(summary['recommended_hosts_for_discover_crawl']) or '(none)'}"
    )
    print(f"Verify=False needed hosts: {', '.join(summary['verify_false_needed_hosts']) or '(none)'}")
    print(f"JSON: {report['output_files']['json']}")
    print(f"CSV: {report['output_files']['csv']}")
    print("")
    print("First-page news links by host:")
    for host, count in summary["first_page_news_links_by_host"].items():
        print(f"  - {host}: {count}")
    print("")


def antigoportal_urls(include_known_extra_subdomains: bool) -> list[str]:
    urls = list(ANTIGOPORTAL_LISTED_URLS)
    if include_known_extra_subdomains:
        for code in KNOWN_EXTRA_OLD_SUBDOMAIN_CODES:
            for scheme in ("https", "http"):
                urls.append(f"{scheme}://{code}.antigoportal.ifmt.edu.br/noticias/")
    return list(dict.fromkeys(urls))


def _attempt_from_response(
    request_url: str,
    response: Response,
    verify: bool,
    elapsed_seconds: float,
) -> dict[str, Any]:
    final_url = response.url
    return {
        "request_url": request_url,
        "final_url": final_url,
        "status_code": response.status_code,
        "success": 200 <= response.status_code < 400,
        "redirected": bool(response.history) or final_url != request_url,
        "verify": verify,
        "ssl_verify_disabled": not verify,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "html": response.text or "",
        "html_length": len(response.text or ""),
        "error_type": "",
        "error_message": "",
    }


def _select_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    for attempt in attempts:
        if attempt.get("success"):
            return attempt
    return attempts[-1]


def _first_error(attempts: list[dict[str, Any]], error_type: str) -> str:
    for attempt in attempts:
        if attempt.get("error_type") == error_type:
            return str(attempt.get("error_message") or "")
    return ""


def _fetch_without_html(fetch: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(fetch)
    cleaned["attempts"] = [_attempt_without_html(attempt) for attempt in fetch["attempts"]]
    cleaned["selected"] = _attempt_without_html(fetch["selected"])
    return cleaned


def _attempt_without_html(attempt: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(attempt)
    cleaned.pop("html", None)
    return cleaned


def _is_antigoportal_host(host: str) -> bool:
    return host == "antigoportal.ifmt.edu.br" or host.endswith(".antigoportal.ifmt.edu.br")


def _oldportal_link_pattern(url: str) -> str:
    path = urlparse(url).path.rstrip("/") + "/"
    if re.fullmatch(r"/noticias/\d+/", path):
        return "oldportal_numeric_news"
    if re.fullmatch(r"/noticias/[a-z0-9-]+/", path.lower()):
        return "oldportal_slug_news"
    if re.search(r"/noticia(s)?/[^/]+/", path.lower()):
        return "oldportal_article_like"
    return ""


def _oldportal_title_from_page(soup: BeautifulSoup, url: str) -> str:
    normalized_url = normalize_url(url)
    for anchor in soup.find_all("a", href=True):
        raw_href = str(anchor.get("href") or "").strip()
        if not raw_href or raw_href.startswith("#"):
            continue
        resolved = resolve_link(raw_href, url)
        if not resolved or normalize_url(resolved) != normalized_url:
            continue
        text = normalize_text(anchor.get_text(" ", strip=True))
        if len(text) >= 12 and _text_key(text) not in _NAV_LABEL_KEYS:
            return text
    return ""


def _text_key(value: str) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9\xC0-\xFF]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_oldportal_author(line: str) -> bool:
    key = _text_key(line)
    if key.startswith("site desativado"):
        return False
    return bool(
        re.search(r"\b(ascom|assessoria|comunicacao|comunica(?:cao|ção))\b", key)
        or (
            re.search(r"\bifmt\b", key)
            and "/" in line
            and len(line) <= 120
            and len(line.split()) <= 8
        )
    )


def _publisher_unit_from_author(author_name: str) -> str:
    key = _text_key(author_name)
    if "reitoria" in key:
        return "Reitoria"
    campus_match = re.search(r"campus\s+([a-z0-9\xC0-\xFF ]+)", author_name, flags=re.IGNORECASE)
    if campus_match:
        return normalize_text(f"Campus {campus_match.group(1)}")
    return ""


def _first_date(text: str) -> str:
    patterns = (
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{1,2}\s+de\s+[A-Za-z\xC0-\xFF]+(?:\s+de\s+\d{4})?\b",
        r"\b\d{1,2}\s+[A-Za-z\xC0-\xFF]{3,}\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_text(match.group(0))
    return ""


def _looks_like_document_title(line: str) -> bool:
    lowered = line.lower()
    return lowered.endswith((".pdf", ".doc", ".docx", ".xls", ".xlsx")) or lowered.startswith(
        ("edital ", "instrucao normativa", "instrução normativa")
    )


def _individual_path_pattern(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") + "/"
    if re.fullmatch(r"/noticias/\d+/", path):
        return "/noticias/{id}/"
    if re.fullmatch(r"/noticias/[a-z0-9-]+/", path.lower()):
        return "/noticias/{slug}/"
    parts = [part for part in path.strip("/").split("/") if part]
    if not parts:
        return path
    parts[-1] = "{article}"
    return "/" + "/".join(parts) + "/"


def _listing_base_url(url: str) -> str:
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if not base.endswith("/"):
        base = f"{base}/"
    return base


def _pagination_templates(base_url: str) -> list[tuple[str, str]]:
    return [
        ("query_page", f"{base_url}?page={{page}}"),
        ("path_page", f"{base_url}page/{{page}}/"),
        ("query_pagina", f"{base_url}?pagina={{page}}"),
    ]


def _link_signature(link_dicts: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted(normalize_url(str(link.get("url") or "")) for link in link_dicts))


def _candidate_signature(candidates: list[CandidateLink]) -> tuple[str, ...]:
    return tuple(sorted(normalize_url(candidate.url) for candidate in candidates))


def _listing_rank(result: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(bool(result.get("opened"))),
        int(result.get("candidate_news_links_count", 0)),
        int(str(result.get("tested_url", "")).startswith("https://")),
        -int(result.get("html_length") or 0),
    )


def _recommend_source(listing: dict[str, Any]) -> bool:
    if not listing.get("opened") or not _listing_has_any_news(listing):
        return False
    article_tests = listing.get("article_tests", [])
    if not article_tests:
        return False
    return any(test.get("extractable") for test in article_tests)


def _listing_has_any_news(listing: dict[str, Any]) -> bool:
    if listing.get("candidate_news_links_count", 0) > 0:
        return True
    return any(
        test.get("candidate_news_links_count", 0) > 0
        for test in listing.get("pagination_tests", [])
    )


def _article_test_candidates(listing: dict[str, Any]) -> list[dict[str, Any]]:
    first_page_candidates = listing.get("first_10_candidate_links", [])
    if first_page_candidates:
        return first_page_candidates
    pagination_tests = sorted(
        listing.get("pagination_tests", []),
        key=lambda test: (
            not bool(test.get("works")),
            int(test.get("page_number") or 999999),
            str(test.get("pattern") or ""),
        ),
    )
    for test in pagination_tests:
        candidates = test.get("first_10_candidate_links", [])
        if candidates:
            return candidates
    return []


def _article_discovered_record(listing: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    host = listing.get("source_host", "")
    return {
        "discovered_id": "diagnostic",
        "source_id": f"oldportal_{host.replace('.', '_')}",
        "source_name": f"IFMT old subdomain portal {host}",
        "source_environment": SOURCE_ENVIRONMENT,
        "source_base_url": f"{urlparse(candidate['url']).scheme}://{host}",
        "source_host": host,
        "source_access_mode": "domain",
        "listing_url": listing.get("tested_url", ""),
        "listing_page_number": 1,
        "discovered_url": candidate.get("url", ""),
        "title_from_listing": candidate.get("title") or candidate.get("text") or "",
        "summary_from_listing": candidate.get("summary") or "",
        "date_label_raw": candidate.get("date_label_raw") or "",
        "publisher_unit": candidate.get("publisher_unit") or "",
        "image_url_from_listing": candidate.get("image_url") or "",
    }


def _anchor_text(anchor) -> str:
    candidates = [
        anchor.get_text(" ", strip=True),
        anchor.get("title"),
        anchor.get("aria-label"),
    ]
    image = anchor.find("img")
    if image is not None:
        candidates.extend([image.get("alt"), image.get("title")])
    for candidate in candidates:
        text = normalize_text(str(candidate or ""))
        if text:
            return text
    return ""


def _nearby_date(anchor) -> str:
    container = _nearby_container(anchor)
    if container is None:
        return ""
    time_tag = container.find("time")
    if time_tag:
        return normalize_text(str(time_tag.get("datetime") or time_tag.get_text(" ", strip=True)))
    text = normalize_text(container.get_text(" ", strip=True))
    patterns = (
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{1,2}\s+de\s+[A-Za-z\xC0-\xFF]+(?:\s+de\s+\d{4})?\b",
        r"\b\d{1,2}\s+[A-Za-z\xC0-\xFF]{3,}\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_text(match.group(0))
    return ""


def _nearby_image(anchor, base_url: str) -> str:
    container = _nearby_container(anchor)
    if container is None:
        return ""
    image = container.find("img")
    if image is None:
        return ""
    for attr in ("src", "data-src", "data-original"):
        value = image.get(attr)
        if value:
            return resolve_link(str(value), base_url) or ""
    return ""


def _nearby_container(anchor):
    current = anchor
    for _ in range(6):
        current = current.parent
        if current is None:
            return anchor.parent
        if current.name in {"article", "li"}:
            return current
        tokens = " ".join(
            str(item)
            for attr in ("class", "id", "role")
            for item in _attr_values(current.get(attr))
        ).lower()
        if any(token in tokens for token in ("noticia", "news", "post", "item", "card")):
            return current
    return anchor.parent


def _attr_values(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
