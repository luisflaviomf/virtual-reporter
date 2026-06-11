from __future__ import annotations

from typing import Any

from .config import PATHS, RANDOM_STATE, ensure_dirs
from .load_dataset import load_labeled, load_model_ready
from .retrieval_utils import build_retrieval_artifacts


def build_retrieval_index(
    sample_size: int | None = None,
    quick: bool = False,
    force: bool = False,
    no_heavy_models: bool = False,
) -> dict[str, Any]:
    ensure_dirs()
    try:
        df = load_labeled().copy()
    except FileNotFoundError:
        df = load_model_ready().copy()

    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=RANDOM_STATE)
    elif quick and len(df) > 1000:
        df = df.sample(1000, random_state=RANDOM_STATE)

    df = df[df["text_for_model"].fillna("").astype(str).str.strip().ne("")].copy()
    result = build_retrieval_artifacts(df)
    result["outputs"] = {
        "bm25_index": str(PATHS.retrieval_dir / "bm25_index.pkl"),
        "faiss_index": str(PATHS.retrieval_dir / "faiss.index"),
        "article_id_mapping": str(PATHS.retrieval_dir / "article_id_mapping.json"),
        "embeddings": str(PATHS.retrieval_dir / "embeddings.npy"),
    }
    return result
