from __future__ import annotations

import csv
import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import (
    DISCOVERY_SOURCE_GROUPS,
    DISCOVERY_SOURCES,
    LEGACY_IP_HOST,
    SourceConfig,
)
from .http_client import HttpClient
from .parse_listings import (
    CandidateLink,
    detect_last_listing_page,
    extract_news_links,
    page_fingerprint,
)
from .url_utils import normalize_url, slug_from_url


DISCOVERY_FIELDS = [
    "discovered_id",
    "source_id",
    "source_name",
    "source_environment",
    "source_base_url",
    "source_host",
    "source_access_mode",
    "listing_url",
    "listing_page_number",
    "raw_href",
    "discovered_url",
    "normalized_url",
    "url_host",
    "slug",
    "title_from_listing",
    "summary_from_listing",
    "date_label_raw",
    "publisher_unit",
    "image_url_from_listing",
    "discovered_at",
    "http_status_listing",
    "ssl_verify_disabled",
    "redirected_listing",
    "final_listing_url",
    "extraction_status",
    "error_message",
    "alternative_discovery_locations",
]


@dataclass
class DiscoveryOptions:
    sources: str
    max_pages: int
    limit: int | None
    output_prefix: str
    output_dir: Path
    dry_run: bool = False
    resume: bool = False
    timeout: float = 30.0
    delay_seconds: float = 0.5
    max_retries: int = 2


@dataclass
class DiscoveryState:
    records_by_url: OrderedDict[str, dict[str, Any]] = field(default_factory=OrderedDict)
    pages_processed: int = 0
    total_url_observations: int = 0
    duplicates_merged: int = 0
    pages_by_source: dict[str, int] = field(default_factory=dict)
    urls_by_source: dict[str, int] = field(default_factory=dict)
    unique_urls_by_source: dict[str, int] = field(default_factory=dict)
    stop_reasons: list[dict[str, Any]] = field(default_factory=list)
    page_errors: list[dict[str, Any]] = field(default_factory=list)
    legacy_ip_candidates_total: int = 0
    legacy_ip_candidates_with_ip_host: int = 0
    detected_last_pages: dict[str, int] = field(default_factory=dict)


def run_discovery(options: DiscoveryOptions) -> dict[str, Any]:
    sources = resolve_source_selection(options.sources)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = _output_paths(options.output_dir, options.output_prefix)

    state = DiscoveryState()
    if options.resume:
        _load_existing_records(output_paths["jsonl"], state)

    client = HttpClient(
        timeout=options.timeout,
        delay_seconds=options.delay_seconds,
        max_retries=options.max_retries,
    )

    run_started_at = datetime.now(timezone.utc).isoformat()
    for source in sources:
        if _limit_reached(state, options.limit):
            state.stop_reasons.append(
                {
                    "source_id": source.source_id,
                    "page_number": None,
                    "reason": "limit_reached",
                }
            )
            break
        _discover_source(source, client, state, options)

    _assign_discovered_ids(state)
    report = _build_report(
        state=state,
        options=options,
        sources=sources,
        output_paths=output_paths,
        run_started_at=run_started_at,
    )

    report["dry_run"] = options.dry_run
    if not options.dry_run:
        _write_outputs(state.records_by_url.values(), report, output_paths)

    return report


def resolve_source_selection(selection: str) -> list[SourceConfig]:
    requested = [part.strip() for part in selection.split(",") if part.strip()]
    if not requested:
        requested = ["legacy_ip", "current", "campuses"]

    selected_ids: list[str] = []
    for name in requested:
        if name in DISCOVERY_SOURCE_GROUPS:
            selected_ids.extend(DISCOVERY_SOURCE_GROUPS[name])
        elif name in DISCOVERY_SOURCES:
            selected_ids.append(name)
        else:
            valid = sorted(set(DISCOVERY_SOURCE_GROUPS) | set(DISCOVERY_SOURCES))
            raise ValueError(
                f"Unknown discovery source '{name}'. Valid values: {', '.join(valid)}"
            )

    deduped_ids: list[str] = []
    for source_id in selected_ids:
        if source_id not in deduped_ids:
            deduped_ids.append(source_id)
    return [DISCOVERY_SOURCES[source_id] for source_id in deduped_ids]


def print_discovery_report(report: dict[str, Any], preview_count: int = 10) -> None:
    print("")
    print("IFMT discovery report")
    print(f"Generated at: {report['generated_at']}")
    print(f"Sources: {', '.join(report['source_ids'])}")
    print(f"Pages processed: {report['pages_processed']}")
    print(f"Total URL observations: {report['total_urls_discovered']}")
    print(f"Unique URLs: {report['total_unique_urls']}")
    print(f"Duplicates merged: {report['duplicates_merged']}")
    print(f"Legacy IP links preserved: {report['legacy_ip_links_preserved']}")
    if report.get("detected_last_pages"):
        print(f"Detected last pages: {json.dumps(report['detected_last_pages'])}")
    if not report.get("dry_run"):
        print(f"JSONL: {report['output_paths']['jsonl']}")
        print(f"CSV: {report['output_paths']['csv']}")
        print(f"Report: {report['output_paths']['report']}")
    else:
        print("Dry run: no output files written")

    if report["stop_reasons"]:
        print("Stop reasons:")
        for reason in report["stop_reasons"]:
            print(
                "  - "
                f"{reason['source_id']} page {reason.get('page_number')}: "
                f"{reason['reason']}"
            )

    print("")
    print(f"First {min(preview_count, len(report['preview_records']))} records:")
    for record in report["preview_records"][:preview_count]:
        print(
            "  - "
            f"{record['source_id']} p.{record['listing_page_number']} "
            f"{record['title_from_listing'] or '(no title)'} -> "
            f"{record['discovered_url']}"
        )
    print("")


def _discover_source(
    source: SourceConfig,
    client: HttpClient,
    state: DiscoveryState,
    options: DiscoveryOptions,
) -> None:
    seen_page_fingerprints: set[str] = set()
    seen_link_signatures: set[tuple[str, ...]] = set()
    detected_last_page: int | None = None

    for page_number in range(1, options.max_pages + 1):
        if _limit_reached(state, options.limit):
            state.stop_reasons.append(
                {
                    "source_id": source.source_id,
                    "page_number": page_number,
                    "reason": "limit_reached",
                }
            )
            break

        listing_url = build_listing_url(source, page_number)
        fetch_result = client.fetch_url(listing_url)
        selected = fetch_result.selected
        if source.source_type == "old_subdomain_portal":
            listing_url, selected = _select_oldportal_listing_attempt(
                source=source,
                page_number=page_number,
                listing_url=listing_url,
                selected=selected,
                client=client,
            )
        state.pages_processed += 1
        state.pages_by_source[source.source_id] = (
            state.pages_by_source.get(source.source_id, 0) + 1
        )

        if selected.status_code != 200:
            state.page_errors.append(
                {
                    "source_id": source.source_id,
                    "listing_url": listing_url,
                    "page_number": page_number,
                    "status_code": selected.status_code,
                    "error_type": selected.error_type,
                    "error_message": selected.error_message,
                }
            )
            state.stop_reasons.append(
                {
                    "source_id": source.source_id,
                    "page_number": page_number,
                    "reason": "non_200_listing_status",
                    "status_code": selected.status_code,
                }
            )
            break

        if selected.html_length < 500:
            state.stop_reasons.append(
                {
                    "source_id": source.source_id,
                    "page_number": page_number,
                    "reason": "html_too_small",
                    "html_length": selected.html_length,
                }
            )
            break

        fingerprint = page_fingerprint(selected.html)
        if fingerprint in seen_page_fingerprints:
            state.stop_reasons.append(
                {
                    "source_id": source.source_id,
                    "page_number": page_number,
                    "reason": "repeated_page_fingerprint",
                }
            )
            break
        seen_page_fingerprints.add(fingerprint)

        link_base_url = _link_base_url(listing_url, selected.final_url)
        candidates = [
            candidate
            for candidate in extract_news_links(selected.html, link_base_url)
            if _source_accepts_candidate(source, candidate)
        ]
        normalized_candidates = tuple(sorted(normalize_url(candidate.url) for candidate in candidates))
        if normalized_candidates in seen_link_signatures:
            state.stop_reasons.append(
                {
                    "source_id": source.source_id,
                    "page_number": page_number,
                    "reason": "repeated_link_signature",
                }
            )
            break
        seen_link_signatures.add(normalized_candidates)

        if not candidates:
            if source.source_type == "old_subdomain_portal" and page_number > 1:
                state.detected_last_pages[source.source_id] = page_number - 1
            state.stop_reasons.append(
                {
                    "source_id": source.source_id,
                    "page_number": page_number,
                    "reason": "no_candidate_links",
                }
            )
            break

        new_records = 0
        for candidate in candidates:
            if _limit_reached(state, options.limit):
                break
            state.total_url_observations += 1
            state.urls_by_source[source.source_id] = (
                state.urls_by_source.get(source.source_id, 0) + 1
            )
            if source.source_id == "legacy_ip":
                state.legacy_ip_candidates_total += 1
                if urlparse(candidate.url).hostname == LEGACY_IP_HOST:
                    state.legacy_ip_candidates_with_ip_host += 1

            normalized_url = normalize_url(candidate.url)
            if _merge_candidate(
                state=state,
                source=source,
                candidate=candidate,
                normalized_url=normalized_url,
                listing_url=listing_url,
                page_number=page_number,
                listing_status=selected.status_code,
                ssl_verify_disabled=selected.ssl_verify_disabled,
                redirected_listing=selected.redirected,
                final_listing_url=selected.final_url,
            ):
                new_records += 1

        if detected_last_page is None and source.source_type != "old_subdomain_portal":
            detected_last_page = detect_last_listing_page(selected.html, source.source_type)
            if detected_last_page:
                state.detected_last_pages[source.source_id] = detected_last_page

        if new_records == 0:
            if source.source_type == "old_subdomain_portal":
                state.detected_last_pages[source.source_id] = page_number - 1 if page_number > 1 else page_number
            state.stop_reasons.append(
                {
                    "source_id": source.source_id,
                    "page_number": page_number,
                    "reason": "no_new_links",
                }
            )
            break

        if detected_last_page and page_number >= detected_last_page:
            if source.source_type == "old_subdomain_portal":
                state.detected_last_pages[source.source_id] = page_number
            state.stop_reasons.append(
                {
                    "source_id": source.source_id,
                    "page_number": page_number,
                    "reason": "detected_last_page_reached",
                    "detected_last_page": detected_last_page,
                }
            )
            break


def build_listing_url(source: SourceConfig, page_number: int) -> str:
    base = source.news_archive_url.rstrip("/") + "/"
    if source.source_type == "legacy_ip":
        return f"{base}?page={page_number}"
    if source.source_type == "old_subdomain_portal":
        return base if page_number == 1 else f"{base}?page={page_number}"
    if page_number == 1:
        return base
    return f"{base}page/{page_number}/"


def _select_oldportal_listing_attempt(
    source: SourceConfig,
    page_number: int,
    listing_url: str,
    selected: Any,
    client: HttpClient,
) -> tuple[str, Any]:
    if _oldportal_selected_has_candidates(source, selected, listing_url):
        return listing_url, selected

    alternatives = _oldportal_listing_alternatives(source, page_number, listing_url)
    for _attempt_round in range(2):
        for alternative_url in alternatives:
            if alternative_url == listing_url and _attempt_round == 0:
                continue
            alternative = client.fetch_url(alternative_url).selected
            if _oldportal_selected_has_candidates(source, alternative, alternative_url):
                return alternative_url, alternative
    return listing_url, selected


def _oldportal_listing_alternatives(
    source: SourceConfig,
    page_number: int,
    listing_url: str,
) -> list[str]:
    base = source.news_archive_url.rstrip("/") + "/"
    parsed = urlparse(base)
    http_base = parsed._replace(scheme="http").geturl()
    if page_number == 1:
        candidates = [
            base,
            f"{base}?page=1",
            http_base,
            f"{http_base}?page=1",
        ]
    else:
        candidates = [
            f"{base}?page={page_number}",
            f"{http_base}?page={page_number}",
        ]
    return list(dict.fromkeys([listing_url, *candidates]))


def _oldportal_selected_has_candidates(source: SourceConfig, selected: Any, listing_url: str) -> bool:
    if selected.status_code != 200 or not selected.html:
        return False
    link_base_url = _link_base_url(listing_url, selected.final_url)
    return any(
        _source_accepts_candidate(source, candidate)
        for candidate in extract_news_links(selected.html, link_base_url)
    )


def _source_accepts_candidate(source: SourceConfig, candidate: CandidateLink) -> bool:
    parsed = urlparse(candidate.url)
    host = parsed.hostname or ""
    path = parsed.path.rstrip("/") + "/"
    source_host = urlparse(source.base_url).hostname or ""

    if source.source_id == "legacy_ip":
        return host == LEGACY_IP_HOST and path.startswith("/conteudo/noticia/")
    if source.source_id == "current_portal":
        return host == "ifmt.edu.br" and path.startswith("/blog/")
    if source.source_type == "campus_wordpress":
        return host == source_host
    if source.source_type == "old_subdomain_portal":
        return host == source_host and bool(
            re.match(r"^/noticias/[^/]+/$", path, flags=re.IGNORECASE)
        )
    return True


def _merge_candidate(
    state: DiscoveryState,
    source: SourceConfig,
    candidate: CandidateLink,
    normalized_url: str,
    listing_url: str,
    page_number: int,
    listing_status: int | None,
    ssl_verify_disabled: bool,
    redirected_listing: bool,
    final_listing_url: str | None,
) -> bool:
    now = datetime.now(timezone.utc).isoformat()
    location = {
        "source_id": source.source_id,
        "source_name": source.source_name,
        "listing_url": listing_url,
        "listing_page_number": page_number,
        "raw_href": candidate.raw_href,
        "discovered_url": candidate.url,
        "title_from_listing": candidate.title or candidate.text,
        "http_status_listing": listing_status,
        "ssl_verify_disabled": ssl_verify_disabled,
        "redirected_listing": redirected_listing,
        "final_listing_url": final_listing_url,
    }

    if normalized_url in state.records_by_url:
        record = state.records_by_url[normalized_url]
        alternatives = record.setdefault("alternative_discovery_locations", [])
        if not _location_already_recorded(record, alternatives, location):
            alternatives.append(location)
            state.duplicates_merged += 1
        return False

    discovered_url_host = urlparse(candidate.url).hostname or ""
    source_host = urlparse(source.base_url).hostname or ""
    record = {
        "discovered_id": "",
        "source_id": source.source_id,
        "source_name": source.source_name,
        "source_environment": source.source_environment,
        "source_base_url": source.base_url,
        "source_host": source_host,
        "source_access_mode": source.source_access_mode,
        "listing_url": listing_url,
        "listing_page_number": page_number,
        "raw_href": candidate.raw_href,
        "discovered_url": candidate.url,
        "normalized_url": normalized_url,
        "url_host": discovered_url_host,
        "slug": slug_from_url(candidate.url),
        "title_from_listing": candidate.title or candidate.text,
        "summary_from_listing": candidate.summary,
        "date_label_raw": candidate.date_label_raw,
        "publisher_unit": candidate.publisher_unit,
        "image_url_from_listing": candidate.image_url,
        "discovered_at": now,
        "http_status_listing": listing_status,
        "ssl_verify_disabled": ssl_verify_disabled,
        "redirected_listing": redirected_listing,
        "final_listing_url": final_listing_url,
        "extraction_status": "discovered",
        "error_message": "",
        "alternative_discovery_locations": [],
    }
    state.records_by_url[normalized_url] = record
    state.unique_urls_by_source[source.source_id] = (
        state.unique_urls_by_source.get(source.source_id, 0) + 1
    )
    return True


def _location_already_recorded(
    record: dict[str, Any],
    alternatives: list[dict[str, Any]],
    location: dict[str, Any],
) -> bool:
    primary = {
        "source_id": record.get("source_id"),
        "listing_url": record.get("listing_url"),
        "listing_page_number": record.get("listing_page_number"),
        "raw_href": record.get("raw_href"),
    }
    candidate_key = {
        "source_id": location.get("source_id"),
        "listing_url": location.get("listing_url"),
        "listing_page_number": location.get("listing_page_number"),
        "raw_href": location.get("raw_href"),
    }
    if primary == candidate_key:
        return True
    for alternative in alternatives:
        alternative_key = {
            "source_id": alternative.get("source_id"),
            "listing_url": alternative.get("listing_url"),
            "listing_page_number": alternative.get("listing_page_number"),
            "raw_href": alternative.get("raw_href"),
        }
        if alternative_key == candidate_key:
            return True
    return False


def _link_base_url(request_url: str, final_url: str | None) -> str:
    if urlparse(request_url).hostname == LEGACY_IP_HOST:
        if final_url and urlparse(final_url).hostname == LEGACY_IP_HOST:
            return final_url
        return request_url
    return final_url or request_url


def _limit_reached(state: DiscoveryState, limit: int | None) -> bool:
    return limit is not None and len(state.records_by_url) >= limit


def _assign_discovered_ids(state: DiscoveryState) -> None:
    for index, record in enumerate(state.records_by_url.values(), start=1):
        if not record.get("discovered_id"):
            record["discovered_id"] = f"disc_{index:08d}"


def _build_report(
    state: DiscoveryState,
    options: DiscoveryOptions,
    sources: list[SourceConfig],
    output_paths: dict[str, Path],
    run_started_at: str,
) -> dict[str, Any]:
    latest_paths = {key: str(path) for key, path in output_paths.items()}
    records = list(state.records_by_url.values())
    legacy_total = state.legacy_ip_candidates_total
    legacy_preserved = (
        state.legacy_ip_candidates_with_ip_host == legacy_total if legacy_total else None
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_started_at": run_started_at,
        "source_ids": [source.source_id for source in sources],
        "max_pages": options.max_pages,
        "limit": options.limit,
        "pages_processed": state.pages_processed,
        "pages_by_source": state.pages_by_source,
        "total_urls_discovered": state.total_url_observations,
        "total_unique_urls": len(state.records_by_url),
        "duplicates_merged": state.duplicates_merged,
        "urls_by_source": state.urls_by_source,
        "unique_urls_by_source": state.unique_urls_by_source,
        "legacy_ip_links_preserved": legacy_preserved,
        "legacy_ip_candidates_total": legacy_total,
        "legacy_ip_candidates_with_ip_host": state.legacy_ip_candidates_with_ip_host,
        "detected_last_pages": state.detected_last_pages,
        "stop_reasons": state.stop_reasons,
        "page_errors": state.page_errors,
        "output_paths": latest_paths,
        "preview_records": records[:10],
    }


def _write_outputs(
    records: Any,
    report: dict[str, Any],
    output_paths: dict[str, Path],
) -> None:
    records_list = list(records)
    with output_paths["jsonl"].open("w", encoding="utf-8") as jsonl_file:
        for record in records_list:
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    with output_paths["csv"].open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=DISCOVERY_FIELDS)
        writer.writeheader()
        for record in records_list:
            csv_record = dict(record)
            csv_record["alternative_discovery_locations"] = json.dumps(
                csv_record.get("alternative_discovery_locations", []),
                ensure_ascii=False,
            )
            writer.writerow({field: csv_record.get(field, "") for field in DISCOVERY_FIELDS})

    output_paths["report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_existing_records(path: Path, state: DiscoveryState) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if not line.strip():
                continue
            record = json.loads(line)
            normalized_url = record.get("normalized_url")
            if normalized_url:
                state.records_by_url[normalized_url] = record


def _output_paths(output_dir: Path, output_prefix: str) -> dict[str, Path]:
    if output_prefix == "discovered_urls":
        report_name = "discovery_report_latest.json"
    else:
        report_name = f"{output_prefix}_discovery_report_latest.json"
    return {
        "jsonl": output_dir / f"{output_prefix}_latest.jsonl",
        "csv": output_dir / f"{output_prefix}_latest.csv",
        "report": output_dir / report_name,
    }
