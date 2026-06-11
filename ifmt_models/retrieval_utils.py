from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

from .config import PATHS, RANDOM_STATE
from .io_utils import normalize_space, read_json, write_json
from .text_utils import tokenize


class SimpleBM25:
    def __init__(self, tokenized_docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_docs = tokenized_docs
        self.k1 = k1
        self.b = b
        self.doc_len = np.array([len(doc) for doc in tokenized_docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        self.doc_freq: Counter[str] = Counter()
        self.term_freqs: list[Counter[str]] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(tokenized_docs):
            tf = Counter(doc)
            self.term_freqs.append(tf)
            self.doc_freq.update(tf.keys())
            for term, freq in tf.items():
                self.postings[term].append((i, freq))
        self.n_docs = len(tokenized_docs)
        self.idf = {
            term: math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            for term, df in self.doc_freq.items()
        }

    def score(self, query: str) -> np.ndarray:
        query_terms = tokenize(query)
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if not query_terms or not self.n_docs or not self.avgdl:
            return scores
        for term in query_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freq in self.postings.get(term, []):
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (freq * (self.k1 + 1) / denom)
        return scores


def article_id_from_row(row: pd.Series | dict[str, Any]) -> str:
    for key in ("global_article_id", "article_id", "model_record_id", "original_article_id"):
        value = row.get(key, "")
        if str(value).strip():
            return str(value)
    return ""


def build_mapping(df: pd.DataFrame, include_text: bool = False) -> list[dict[str, Any]]:
    mapping = []
    for i, row in df.reset_index(drop=True).iterrows():
        article_id = article_id_from_row(row) or f"doc_{i:08d}"
        item = {
            "index": i,
            "article_id": article_id,
            "title": row.get("title", ""),
            "lead": row.get("lead", ""),
            "publication_date": row.get("publication_date", ""),
            "article_type_weak": row.get("article_type_weak", ""),
            "source_environment": row.get("source_environment", ""),
            "source_host": row.get("source_host", ""),
            "publisher_unit_normalized": row.get("publisher_unit_normalized", ""),
            "campus_name_normalized": row.get("campus_name_normalized", ""),
            "original_url": row.get("original_url", ""),
        }
        if include_text:
            item["text_for_model"] = row.get("text_for_model", "")
        mapping.append(item)
    return mapping


def create_retrieval_artifacts(df: pd.DataFrame, include_text: bool = False) -> dict[str, Any]:
    texts = df["text_for_model"].fillna("").map(normalize_space).tolist()
    mapping = build_mapping(df, include_text=include_text)

    bm25 = SimpleBM25([tokenize(text) for text in texts])

    vectorizer = _fit_vectorizer(texts)
    sparse = vectorizer.transform(texts)
    n_components = max(2, min(256, sparse.shape[0] - 1, sparse.shape[1] - 1))
    if n_components >= 2:
        reducer = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        normalizer = Normalizer(copy=False)
        embeddings = normalizer.fit_transform(reducer.fit_transform(sparse)).astype("float32")
        dense_model = {"backend": "tfidf_svd", "vectorizer": vectorizer, "reducer": reducer, "normalizer": normalizer}
    else:
        normalizer = Normalizer(copy=False)
        embeddings = normalizer.fit_transform(sparse.toarray()).astype("float32")
        dense_model = {"backend": "tfidf_dense", "vectorizer": vectorizer, "reducer": None, "normalizer": normalizer}

    return {
        "bm25": bm25,
        "dense_model": dense_model,
        "embeddings": embeddings,
        "mapping": mapping,
        "dense_backend": dense_model["backend"],
    }


def build_retrieval_artifacts(df: pd.DataFrame) -> dict[str, Any]:
    PATHS.retrieval_dir.mkdir(parents=True, exist_ok=True)
    artifacts = create_retrieval_artifacts(df, include_text=False)

    joblib.dump(artifacts["bm25"], PATHS.retrieval_dir / "bm25_index.pkl")
    np.save(PATHS.retrieval_dir / "embeddings.npy", artifacts["embeddings"])
    joblib.dump(artifacts["dense_model"], PATHS.retrieval_dir / "dense_vectorizer.joblib")
    joblib.dump(
        {"backend": artifacts["dense_backend"], "note": "Local numpy index saved because FAISS is optional."},
        PATHS.retrieval_dir / "faiss.index",
    )
    write_json(PATHS.retrieval_dir / "article_id_mapping.json", artifacts["mapping"])

    return {
        "records": len(artifacts["mapping"]),
        "dense_backend": artifacts["dense_backend"],
        "embedding_dimensions": int(artifacts["embeddings"].shape[1]) if artifacts["embeddings"].ndim == 2 else 0,
    }


def load_retrieval_artifacts() -> dict[str, Any]:
    return {
        "bm25": joblib.load(PATHS.retrieval_dir / "bm25_index.pkl"),
        "dense_model": joblib.load(PATHS.retrieval_dir / "dense_vectorizer.joblib"),
        "embeddings": np.load(PATHS.retrieval_dir / "embeddings.npy"),
        "mapping": read_json(PATHS.retrieval_dir / "article_id_mapping.json", default=[]),
    }


def retrieve(artifacts: dict[str, Any], query: str, method: str = "hybrid", top_k: int = 10) -> list[dict[str, Any]]:
    if method == "bm25":
        return _rank_from_scores(artifacts, _bm25_scores(artifacts, query), top_k, method)
    if method == "dense":
        return _rank_from_scores(artifacts, _dense_scores(artifacts, query), top_k, method)
    if method == "hybrid":
        return _hybrid_rank(artifacts, query, top_k)
    raise ValueError(f"Metodo de recuperacao desconhecido: {method}")


def _fit_vectorizer(texts: list[str]) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=60000,
        sublinear_tf=True,
    )
    try:
        vectorizer.fit(texts)
    except ValueError:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=60000, sublinear_tf=True)
        vectorizer.fit(texts)
    return vectorizer


def _bm25_scores(artifacts: dict[str, Any], query: str) -> np.ndarray:
    return artifacts["bm25"].score(query)


def _dense_scores(artifacts: dict[str, Any], query: str) -> np.ndarray:
    model = artifacts["dense_model"]
    vector = model["vectorizer"].transform([query])
    if model.get("reducer") is not None:
        dense = model["reducer"].transform(vector)
    else:
        dense = vector.toarray()
    dense = model["normalizer"].transform(dense).astype("float32")
    return np.dot(artifacts["embeddings"], dense[0])


def _rank_from_scores(
    artifacts: dict[str, Any],
    scores: np.ndarray,
    top_k: int,
    method: str,
) -> list[dict[str, Any]]:
    if len(scores) == 0:
        return []
    top = np.argsort(scores)[::-1][:top_k]
    rows = []
    for rank, idx in enumerate(top, start=1):
        item = dict(artifacts["mapping"][int(idx)])
        item.update({"rank": rank, "score": float(scores[int(idx)]), "retrieval_method": method})
        rows.append(item)
    return rows


def _hybrid_rank(artifacts: dict[str, Any], query: str, top_k: int) -> list[dict[str, Any]]:
    bm25 = _rank_from_scores(artifacts, _bm25_scores(artifacts, query), min(100, len(artifacts["mapping"])), "bm25")
    dense = _rank_from_scores(artifacts, _dense_scores(artifacts, query), min(100, len(artifacts["mapping"])), "dense")
    rrf: dict[int, float] = defaultdict(float)
    rank_constant = 60
    for ranked in (bm25, dense):
        for item in ranked:
            rrf[int(item["index"])] += 1.0 / (rank_constant + int(item["rank"]))
    ordered = sorted(rrf.items(), key=lambda item: item[1], reverse=True)[:top_k]
    rows = []
    for rank, (idx, score) in enumerate(ordered, start=1):
        item = dict(artifacts["mapping"][int(idx)])
        item.update({"rank": rank, "score": float(score), "retrieval_method": "hybrid"})
        rows.append(item)
    return rows
