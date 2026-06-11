from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_DIAGNOSTIC_SOURCES, SourceConfig
from .crawl import CrawlOptions, print_crawl_report, run_crawl
from .dedupe import DedupeOptions, print_dedupe_report, run_dedupe
from .consolidate_all_sources import (
    ConsolidateAllSourcesOptions,
    print_all_sources_report,
    run_consolidate_all_sources,
)
from .diagnose import print_diagnostic_report, run_diagnostics
from .diagnose_antigoportal import (
    AntigoportalDiagnosticOptions,
    print_antigoportal_diagnostic_report,
    run_antigoportal_diagnostic,
)
from .discover import (
    DiscoveryOptions,
    print_discovery_report,
    resolve_source_selection,
    run_discovery,
)
from .finalize import (
    ExportOptions,
    ReportOptions,
    print_dataset_report,
    print_export_report,
    run_export,
    run_report,
)
from .inspect_antigoportal import (
    AntigoportalInspectionOptions,
    print_antigoportal_inspection_report,
    run_antigoportal_inspection,
)
from .normalize_units import (
    NormalizeUnitsOptions,
    print_normalize_units_report,
    run_normalize_units,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ifmt-scraper",
        description="Scraper tools for IFMT institutional news.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnose = subparsers.add_parser(
        "diagnose",
        help="Test access to IFMT news sources and summarize listing links.",
    )
    diagnose.add_argument(
        "--url",
        action="append",
        default=None,
        help="Custom URL to diagnose. May be repeated. Defaults to the required IFMT URLs.",
    )
    diagnose.add_argument(
        "--output-dir",
        default="data/output",
        help="Directory for JSON diagnostic reports.",
    )
    diagnose.add_argument(
        "--raw-html-dir",
        default="data/raw_html",
        help="Directory for raw HTML snapshots.",
    )
    diagnose.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds.",
    )
    diagnose.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Minimum delay between HTTP requests in seconds.",
    )
    diagnose.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per attempted URL for transient request failures.",
    )
    diagnose.add_argument(
        "--preview-links",
        type=int,
        default=5,
        help="Number of candidate news links to print per source.",
    )
    diagnose.add_argument(
        "--no-save-html",
        action="store_true",
        help="Do not save raw HTML snapshots.",
    )
    diagnose.set_defaults(func=run_diagnose_command)

    diagnose_antigoportal = subparsers.add_parser(
        "diagnose-antigoportal",
        help="Diagnose old antigoportal.ifmt.edu.br news listings without crawling the full source.",
    )
    diagnose_antigoportal.add_argument(
        "--output-dir",
        default="data/output",
        help="Directory for antigoportal diagnostic reports.",
    )
    diagnose_antigoportal.add_argument(
        "--timeout",
        type=float,
        default=25.0,
        help="Request timeout in seconds.",
    )
    diagnose_antigoportal.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Minimum delay between diagnostic requests in seconds.",
    )
    diagnose_antigoportal.add_argument(
        "--max-article-tests-per-source",
        type=int,
        default=3,
        help="Number of individual article pages to test per opened host.",
    )
    diagnose_antigoportal.add_argument(
        "--no-extra-known-subdomains",
        action="store_true",
        help="Only test the URL list explicitly requested, without extra known campus subdomains.",
    )
    diagnose_antigoportal.set_defaults(func=run_diagnose_antigoportal_command)

    inspect_antigoportal = subparsers.add_parser(
        "inspect-antigoportal",
        help="Inspect a limited sample of antigoportal article pages and parser candidates.",
    )
    inspect_antigoportal.add_argument(
        "--input",
        default="data/output/discovered_urls_antigoportal_latest.jsonl",
        help="Antigoportal discovery JSONL or CSV input file.",
    )
    inspect_antigoportal.add_argument(
        "--output-dir",
        default="data/output",
        help="Directory for antigoportal inspection reports.",
    )
    inspect_antigoportal.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of URLs to inspect, distributed round-robin by host.",
    )
    inspect_antigoportal.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds.",
    )
    inspect_antigoportal.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Minimum delay between inspection requests in seconds.",
    )
    inspect_antigoportal.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per article URL for transient request failures.",
    )
    inspect_antigoportal.set_defaults(func=run_inspect_antigoportal_command)

    discover = subparsers.add_parser(
        "discover",
        help="Discover article URLs from IFMT listing pages without crawling articles.",
    )
    discover.add_argument(
        "--sources",
        default="legacy_ip,current,campuses",
        help="Comma-separated source groups or source ids. Examples: legacy_ip,current,campuses.",
    )
    discover.add_argument(
        "--max-pages",
        type=int,
        default=1000,
        help="Maximum listing pages to process per source.",
    )
    discover.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many unique URLs.",
    )
    discover.add_argument(
        "--output-prefix",
        default="discovered_urls",
        help="Prefix for JSONL/CSV output files.",
    )
    discover.add_argument(
        "--output-dir",
        default="data/output",
        help="Directory for discovery outputs.",
    )
    discover.add_argument(
        "--dry-run",
        action="store_true",
        help="Run discovery and print the report without writing output files.",
    )
    discover.add_argument(
        "--resume",
        action="store_true",
        help="Merge with an existing latest JSONL file for the same output prefix.",
    )
    discover.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds.",
    )
    discover.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Minimum delay between HTTP requests in seconds.",
    )
    discover.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per listing URL for transient request failures.",
    )
    discover.set_defaults(func=run_discover_command)

    crawl = subparsers.add_parser(
        "crawl",
        help="Fetch discovered article URLs and extract structured article data.",
    )
    crawl.add_argument(
        "--input",
        default="data/output/discovered_urls_latest.jsonl",
        help="Discovery JSONL or CSV input file.",
    )
    crawl.add_argument(
        "--sources",
        default="legacy_ip,current,campuses",
        help="Comma-separated source groups or source ids to crawl.",
    )
    crawl.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of discovered URLs to crawl after filtering.",
    )
    crawl.add_argument(
        "--resume",
        action="store_true",
        help="Skip discovered ids already present in the article output JSONL.",
    )
    crawl.add_argument(
        "--force",
        action="store_true",
        help="When used with --resume, crawl records even if they already exist.",
    )
    crawl.add_argument(
        "--save-html",
        action="store_true",
        help="Save raw article HTML under data/raw_html/articles/.",
    )
    crawl.add_argument(
        "--output-prefix",
        default="news_articles",
        help="Prefix for article output files.",
    )
    crawl.add_argument(
        "--output-dir",
        default="data/output",
        help="Directory for crawl outputs.",
    )
    crawl.add_argument(
        "--raw-html-dir",
        default="data/raw_html/articles",
        help="Directory for raw article HTML snapshots.",
    )
    crawl.add_argument(
        "--dry-run",
        action="store_true",
        help="Run crawl and print the report without writing output files.",
    )
    crawl.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds.",
    )
    crawl.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Minimum delay between article requests in seconds.",
    )
    crawl.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per article URL for transient request failures.",
    )
    crawl.set_defaults(func=run_crawl_command)

    dedupe = subparsers.add_parser(
        "dedupe",
        help="Detect duplicate article records without deleting them.",
    )
    dedupe.add_argument(
        "--sources",
        default="",
        help="Optional source group/source ids for auto-selecting source-specific files. Example: antigoportal.",
    )
    dedupe.add_argument(
        "--input",
        default="data/output/news_articles_latest.jsonl",
        help="Article JSONL or CSV input file.",
    )
    dedupe.add_argument(
        "--output-dir",
        default="data/output",
        help="Directory for dedupe outputs.",
    )
    dedupe.add_argument(
        "--title-threshold",
        type=int,
        default=95,
        help="Approximate duplicate threshold for title similarity.",
    )
    dedupe.add_argument(
        "--body-threshold",
        type=int,
        default=90,
        help="Approximate duplicate threshold for body similarity.",
    )
    dedupe.set_defaults(func=run_dedupe_command)

    export = subparsers.add_parser(
        "export",
        help="Write final consolidated dataset files.",
    )
    export.add_argument(
        "--sources",
        default="",
        help="Optional source group/source ids for auto-selecting source-specific files. Example: antigoportal.",
    )
    export.add_argument(
        "--output-dir",
        default="data/output",
        help="Directory containing intermediate outputs.",
    )
    export.add_argument(
        "--final-dir",
        default="data/output/final",
        help="Directory for final consolidated outputs.",
    )
    export.set_defaults(func=run_export_command)

    report = subparsers.add_parser(
        "report",
        help="Generate final dataset and quality reports.",
    )
    report.add_argument(
        "--sources",
        default="",
        help="Optional source group/source ids for auto-selecting source-specific files. Example: antigoportal.",
    )
    report.add_argument(
        "--output-dir",
        default="data/output",
        help="Directory containing intermediate outputs.",
    )
    report.add_argument(
        "--final-dir",
        default="data/output/final",
        help="Directory for final report outputs.",
    )
    report.set_defaults(func=run_report_command)

    consolidate_all_sources = subparsers.add_parser(
        "consolidate-all-sources",
        help="Create final_all_sources by merging final_combined and final_antigoportal.",
    )
    consolidate_all_sources.add_argument(
        "--final-combined-dir",
        default="data/output/final_combined",
        help="Input directory for the previously combined dataset.",
    )
    consolidate_all_sources.add_argument(
        "--final-antigoportal-dir",
        default="data/output/final_antigoportal",
        help="Input directory for the antigoportal dataset.",
    )
    consolidate_all_sources.add_argument(
        "--output-dir",
        default="data/output/final_all_sources",
        help="Output directory for the all-sources final dataset.",
    )
    consolidate_all_sources.add_argument(
        "--title-threshold",
        type=int,
        default=95,
        help="Approximate duplicate threshold for title similarity.",
    )
    consolidate_all_sources.add_argument(
        "--body-threshold",
        type=int,
        default=90,
        help="Approximate duplicate threshold for body similarity.",
    )
    consolidate_all_sources.set_defaults(func=run_consolidate_all_sources_command)

    normalize_units = subparsers.add_parser(
        "normalize-units",
        help="Normalize publisher_unit/campus fields in final article outputs.",
    )
    normalize_units.add_argument(
        "--final-dir",
        default="data/output/final",
        help="Directory containing final article JSONL files.",
    )
    normalize_units.add_argument(
        "--full-input",
        default=None,
        help="Optional full article JSONL input. Defaults to final-dir/news_articles_full.jsonl.",
    )
    normalize_units.add_argument(
        "--canonical-input",
        default=None,
        help="Optional canonical article JSONL input. Defaults to final-dir/news_articles_canonical.jsonl.",
    )
    normalize_units.add_argument(
        "--in-place-standard",
        action="store_true",
        help="Also rewrite standard final JSONL/CSV/parquet files with normalized fields.",
    )
    normalize_units.add_argument(
        "--no-refresh-report",
        action="store_true",
        help="Do not refresh dataset_report.json and quality_report.md from normalized records.",
    )
    normalize_units.set_defaults(func=run_normalize_units_command)

    return parser


def run_diagnose_command(args: argparse.Namespace) -> int:
    sources = _sources_from_urls(args.url) if args.url else DEFAULT_DIAGNOSTIC_SOURCES
    report = run_diagnostics(
        sources=sources,
        output_dir=Path(args.output_dir),
        raw_html_dir=Path(args.raw_html_dir),
        save_html=not args.no_save_html,
        timeout=args.timeout,
        delay_seconds=args.delay,
        max_retries=args.max_retries,
        preview_links=args.preview_links,
    )
    print_diagnostic_report(report)
    return 0


def run_diagnose_antigoportal_command(args: argparse.Namespace) -> int:
    if args.max_article_tests_per_source < 1:
        raise ValueError("--max-article-tests-per-source must be at least 1")
    report = run_antigoportal_diagnostic(
        AntigoportalDiagnosticOptions(
            output_dir=Path(args.output_dir),
            include_known_extra_subdomains=not args.no_extra_known_subdomains,
            timeout=args.timeout,
            delay_seconds=args.delay,
            max_article_tests_per_source=args.max_article_tests_per_source,
        )
    )
    print_antigoportal_diagnostic_report(report)
    return 0


def run_inspect_antigoportal_command(args: argparse.Namespace) -> int:
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")
    report = run_antigoportal_inspection(
        AntigoportalInspectionOptions(
            input_path=Path(args.input),
            output_dir=Path(args.output_dir),
            limit=args.limit,
            timeout=args.timeout,
            delay_seconds=args.delay,
            max_retries=args.max_retries,
        )
    )
    print_antigoportal_inspection_report(report)
    return 0


def run_discover_command(args: argparse.Namespace) -> int:
    if args.max_pages < 1:
        raise ValueError("--max-pages must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1 when provided")

    output_prefix = _auto_discovery_prefix(args.sources, args.output_prefix)
    report = run_discovery(
        DiscoveryOptions(
            sources=args.sources,
            max_pages=args.max_pages,
            limit=args.limit,
            output_prefix=output_prefix,
            output_dir=Path(args.output_dir),
            dry_run=args.dry_run,
            resume=args.resume,
            timeout=args.timeout,
            delay_seconds=args.delay,
            max_retries=args.max_retries,
        )
    )
    print_discovery_report(report)
    return 0


def run_crawl_command(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1 when provided")

    input_path = _auto_crawl_input(args.sources, args.input)
    output_prefix = _auto_crawl_prefix(args.sources, args.output_prefix)
    report = run_crawl(
        CrawlOptions(
            input_path=Path(input_path),
            sources=args.sources,
            limit=args.limit,
            output_prefix=output_prefix,
            output_dir=Path(args.output_dir),
            raw_html_dir=Path(args.raw_html_dir),
            resume=args.resume,
            force=args.force,
            save_html=args.save_html,
            dry_run=args.dry_run,
            timeout=args.timeout,
            delay_seconds=args.delay,
            max_retries=args.max_retries,
        )
    )
    print_crawl_report(report)
    return 0


def run_dedupe_command(args: argparse.Namespace) -> int:
    input_path = _auto_dedupe_input(args.sources, args.input)
    output_prefix = _auto_dedupe_prefix(args.sources)
    report = run_dedupe(
        DedupeOptions(
            input_path=Path(input_path),
            output_dir=Path(args.output_dir),
            output_prefix=output_prefix,
            title_similarity_threshold=args.title_threshold,
            body_similarity_threshold=args.body_threshold,
        )
    )
    print_dedupe_report(report)
    return 0


def run_export_command(args: argparse.Namespace) -> int:
    input_prefix = _auto_finalize_input_prefix(args.sources)
    final_dir = _auto_final_dir(args.sources, args.final_dir)
    report = run_export(
        ExportOptions(
            output_dir=Path(args.output_dir),
            final_dir=Path(final_dir),
            input_prefix=input_prefix,
        )
    )
    print_export_report(report)
    return 0


def run_report_command(args: argparse.Namespace) -> int:
    input_prefix = _auto_finalize_input_prefix(args.sources)
    final_dir = _auto_final_dir(args.sources, args.final_dir)
    report = run_report(
        ReportOptions(
            output_dir=Path(args.output_dir),
            final_dir=Path(final_dir),
            input_prefix=input_prefix,
        )
    )
    print_dataset_report(report)
    return 0


def run_consolidate_all_sources_command(args: argparse.Namespace) -> int:
    report = run_consolidate_all_sources(
        ConsolidateAllSourcesOptions(
            final_combined_dir=Path(args.final_combined_dir),
            final_antigoportal_dir=Path(args.final_antigoportal_dir),
            output_dir=Path(args.output_dir),
            title_similarity_threshold=args.title_threshold,
            body_similarity_threshold=args.body_threshold,
        )
    )
    print_all_sources_report(report)
    return 0


def run_normalize_units_command(args: argparse.Namespace) -> int:
    report = run_normalize_units(
        NormalizeUnitsOptions(
            final_dir=Path(args.final_dir),
            full_input=Path(args.full_input) if args.full_input else None,
            canonical_input=Path(args.canonical_input) if args.canonical_input else None,
            in_place_standard=args.in_place_standard,
            refresh_dataset_report=not args.no_refresh_report,
        )
    )
    print_normalize_units_report(report)
    return 0


def not_implemented_command(args: argparse.Namespace) -> int:
    print(
        f"The '{args.command}' command is not implemented in this diagnostic slice yet.",
        file=sys.stderr,
    )
    return 2


def _auto_discovery_prefix(sources: str, output_prefix: str) -> str:
    if output_prefix != "discovered_urls":
        return output_prefix
    if _only_old_subdomain_sources(sources):
        return "discovered_urls_antigoportal"
    return output_prefix


def _auto_crawl_prefix(sources: str, output_prefix: str) -> str:
    if output_prefix != "news_articles":
        return output_prefix
    if _only_old_subdomain_sources(sources):
        return "news_articles_antigoportal"
    return output_prefix


def _auto_crawl_input(sources: str, input_path: str) -> str:
    if input_path != "data/output/discovered_urls_latest.jsonl":
        return input_path
    if _only_old_subdomain_sources(sources):
        return "data/output/discovered_urls_antigoportal_latest.jsonl"
    return input_path


def _auto_dedupe_input(sources: str, input_path: str) -> str:
    if input_path != "data/output/news_articles_latest.jsonl":
        return input_path
    if sources and _only_old_subdomain_sources(sources):
        return "data/output/news_articles_antigoportal_latest.jsonl"
    return input_path


def _auto_dedupe_prefix(sources: str) -> str:
    if sources and _only_old_subdomain_sources(sources):
        return "news_articles_antigoportal"
    return "news_articles"


def _auto_finalize_input_prefix(sources: str) -> str:
    if sources and _only_old_subdomain_sources(sources):
        return "news_articles_antigoportal"
    return "news_articles"


def _auto_final_dir(sources: str, final_dir: str) -> str:
    if final_dir != "data/output/final":
        return final_dir
    if sources and _only_old_subdomain_sources(sources):
        return "data/output/final_antigoportal"
    return final_dir


def _only_old_subdomain_sources(sources: str) -> bool:
    try:
        selected = resolve_source_selection(sources)
    except ValueError:
        return False
    return bool(selected) and all(
        source.source_environment == "old_subdomain_portal" for source in selected
    )


def _sources_from_urls(urls: list[str]) -> list[SourceConfig]:
    sources: list[SourceConfig] = []
    for index, url in enumerate(urls, start=1):
        sources.append(
            SourceConfig(
                source_id=f"custom_{index}",
                source_name=f"Custom diagnostic URL {index}",
                source_type="custom",
                source_environment="custom",
                base_url=url,
                news_archive_url=url,
                source_access_mode="custom",
                ssl_verify=True,
            )
        )
    return sources


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
