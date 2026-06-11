from __future__ import annotations

import pandas as pd

from .config import PATHS
from .io_utils import read_jsonl


def load_articles(canonical: bool = True) -> pd.DataFrame:
    path = PATHS.canonical_jsonl if canonical else PATHS.full_jsonl
    return pd.DataFrame(read_jsonl(path))


def load_model_ready() -> pd.DataFrame:
    return pd.read_parquet(PATHS.processed_dir / "articles_model_ready.parquet")


def load_labeled() -> pd.DataFrame:
    return pd.read_parquet(PATHS.processed_dir / "articles_labeled.parquet")

