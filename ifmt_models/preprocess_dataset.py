from __future__ import annotations

from collections import Counter
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .config import PATHS, ensure_dirs
from .io_utils import normalize_space, safe_int, save_dataframe, truthy, write_table
from .load_dataset import load_articles


def prepare_data(sample_size: int | None = None, quick: bool = False, force: bool = False) -> dict[str, Any]:
    ensure_dirs()
    df = load_articles(canonical=True)
    full_df = load_articles(canonical=False)
    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=42)
    elif quick:
        df = df.sample(min(1000, len(df)), random_state=42)

    df = df.copy()
    df["word_count"] = df["word_count"].apply(safe_int)
    if "status" in df.columns:
        df = df[df["status"].fillna("collected").eq("collected")]
    df = df[df["body_text"].fillna("").astype(str).str.strip().ne("")]
    df = df[df["word_count"] >= 30]

    for col in ("title", "lead", "body_text"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").map(normalize_space)
    df["text_for_model"] = (
        df["title"].fillna("") + "\n" + df["lead"].fillna("") + "\n" + df["body_text"].fillna("")
    ).map(normalize_space)
    df["short_text_for_query"] = (df["title"].fillna("") + "\n" + df["lead"].fillna("")).map(normalize_space)
    df["model_record_id"] = [f"model_{i:08d}" for i in range(1, len(df) + 1)]

    save_dataframe(
        df,
        PATHS.processed_dir / "articles_model_ready.parquet",
        PATHS.processed_dir / "articles_model_ready.csv",
    )

    _write_dataset_tables(df, full_df)
    _make_dataset_figures(df)
    return {
        "canonical_input_records": len(load_articles(canonical=True)),
        "full_input_records": len(full_df),
        "model_ready_records": len(df),
        "outputs": {
            "parquet": str(PATHS.processed_dir / "articles_model_ready.parquet"),
            "csv": str(PATHS.processed_dir / "articles_model_ready.csv"),
        },
    }


def _write_dataset_tables(df: pd.DataFrame, full_df: pd.DataFrame) -> None:
    summary = pd.DataFrame(
        [
            {"metric": "full_records", "value": len(full_df)},
            {"metric": "canonical_model_ready_records", "value": len(df)},
            {"metric": "avg_word_count", "value": round(df["word_count"].mean(), 2)},
            {"metric": "median_word_count", "value": round(df["word_count"].median(), 2)},
            {"metric": "with_links", "value": int(df["has_links"].apply(truthy).sum()) if "has_links" in df else 0},
            {"metric": "with_attachments", "value": int(df["has_attachments"].apply(truthy).sum()) if "has_attachments" in df else 0},
            {"metric": "with_images", "value": int(df["has_images"].apply(truthy).sum()) if "has_images" in df else 0},
        ]
    )
    write_table(PATHS.tables_dir / "dataset_summary.csv", summary)

    years = df["publication_date"].fillna("").astype(str).str[:4]
    write_table(PATHS.tables_dir / "articles_by_year.csv", _counter_frame(Counter(y for y in years if y)))
    write_table(PATHS.tables_dir / "articles_by_source.csv", _counter_frame(Counter(df["source_environment"].fillna("unknown"))))
    unit_col = _unit_series(df)
    write_table(PATHS.tables_dir / "articles_by_unit.csv", _counter_frame(Counter(unit_col)))


def _make_dataset_figures(df: pd.DataFrame) -> None:
    years = df["publication_date"].fillna("").astype(str).str[:4]
    _bar_figure(
        Counter(y for y in years if y),
        PATHS.figures_dir / "articles_by_year.png",
        "Noticias por ano",
        "Ano",
        "Artigos",
        top_n=25,
    )
    _bar_figure(
        Counter(df["source_environment"].fillna("unknown")),
        PATHS.figures_dir / "articles_by_source.png",
        "Noticias por fonte",
        "Fonte",
        "Artigos",
        top_n=10,
    )
    coverage = {
        "title": df["title"].fillna("").astype(str).str.strip().ne("").mean() * 100,
        "body": df["body_text"].fillna("").astype(str).str.strip().ne("").mean() * 100,
        "date": df["publication_date"].fillna("").astype(str).str.strip().ne("").mean() * 100,
        "links": df["has_links"].apply(truthy).mean() * 100 if "has_links" in df else 0,
        "attachments": df["has_attachments"].apply(truthy).mean() * 100 if "has_attachments" in df else 0,
        "images": df["has_images"].apply(truthy).mean() * 100 if "has_images" in df else 0,
    }
    _bar_figure(Counter({k: round(v, 2) for k, v in coverage.items()}), PATHS.figures_dir / "metadata_coverage.png", "Cobertura de metadados", "Campo", "%", top_n=10)


def _unit_series(df: pd.DataFrame) -> pd.Series:
    for col in ("campus_name_normalized", "publisher_unit_normalized", "campus_name", "publisher_unit"):
        if col in df.columns:
            series = df[col].fillna("").astype(str).str.strip()
            if series.ne("").any():
                return series.mask(series.eq(""), "unknown")
    return pd.Series(["unknown"] * len(df))


def _counter_frame(counter: Counter) -> pd.DataFrame:
    return pd.DataFrame([{"label": k, "count": v} for k, v in counter.most_common()])


def _bar_figure(counter: Counter, path, title: str, xlabel: str, ylabel: str, top_n: int) -> None:
    items = counter.most_common(top_n)
    labels = [str(k) for k, _ in items]
    values = [v for _, v in items]
    plt.figure(figsize=(max(8, len(labels) * 0.45), 5))
    plt.bar(labels, values, color="#3f6f8f")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    try:
        plt.savefig(path.with_suffix(".pdf"))
    except Exception:
        pass
    plt.close()

