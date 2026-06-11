from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    final_all_sources_dir: Path = Path("data/output/final_all_sources")
    canonical_jsonl: Path = Path("data/output/final_all_sources/news_articles_canonical.jsonl")
    full_jsonl: Path = Path("data/output/final_all_sources/news_articles_full.jsonl")
    article_links_csv: Path = Path("data/output/final_all_sources/article_links.csv")
    media_assets_csv: Path = Path("data/output/final_all_sources/media_assets.csv")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("data/models")
    results_dir: Path = Path("data/results")
    figures_dir: Path = Path("data/results/figures")
    tables_dir: Path = Path("data/results/tables")
    retrieval_dir: Path = Path("data/models/retrieval")


PATHS = Paths()
RANDOM_STATE = 42


def ensure_dirs() -> None:
    for path in (
        PATHS.processed_dir,
        PATHS.models_dir,
        PATHS.results_dir,
        PATHS.figures_dir,
        PATHS.tables_dir,
        PATHS.retrieval_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

