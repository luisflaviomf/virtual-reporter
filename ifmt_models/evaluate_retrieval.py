from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .config import PATHS, RANDOM_STATE, ensure_dirs
from .io_utils import write_json, write_table
from .retrieval_utils import load_retrieval_artifacts, retrieve
from .text_utils import remove_query_details


def evaluate_retrieval(
    query_count: int | None = None,
    quick: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    ensure_dirs()
    artifacts = load_retrieval_artifacts()
    mapping_df = pd.DataFrame(artifacts["mapping"])
    if mapping_df.empty:
        raise RuntimeError("Indice de recuperacao vazio.")

    target_count = query_count or (100 if quick else 500)
    eval_df = _sample_queries(mapping_df, min(target_count, len(mapping_df)))

    methods = ["bm25", "dense", "hybrid"]
    case_rows = []
    qualitative_rows = []
    for _, row in eval_df.iterrows():
        query = remove_query_details(f"{row.get('title', '')}\n{row.get('lead', '')}")
        expected_id = str(row["article_id"])
        case_result: dict[str, Any] = {
            "query_article_id": expected_id,
            "query": query,
            "expected_title": row.get("title", ""),
        }
        for method in methods:
            top = retrieve(artifacts, query, method=method, top_k=10)
            top_ids = [str(item["article_id"]) for item in top]
            rank = top_ids.index(expected_id) + 1 if expected_id in top_ids else None
            case_rows.append(
                {
                    "query_article_id": expected_id,
                    "method": method,
                    "rank": rank,
                    "recall_at_1": int(rank is not None and rank <= 1),
                    "recall_at_5": int(rank is not None and rank <= 5),
                    "recall_at_10": int(rank is not None and rank <= 10),
                    "mrr_at_10": round(1 / rank, 4) if rank is not None and rank <= 10 else 0.0,
                    "ndcg_at_10": round(1 / _log2(rank + 1), 4) if rank is not None and rank <= 10 else 0.0,
                    "top_5_article_ids": json.dumps(top_ids[:5], ensure_ascii=False),
                    "top_5_titles": json.dumps([item.get("title", "") for item in top[:5]], ensure_ascii=False),
                }
            )
            case_result[f"{method}_top5"] = [item.get("title", "") for item in top[:5]]
            case_result[f"{method}_hit_top5"] = bool(rank is not None and rank <= 5)
        qualitative_rows.append(case_result)

    case_df = pd.DataFrame(case_rows)
    metrics = []
    for method, group in case_df.groupby("method"):
        metrics.append(
            {
                "method": method,
                "queries": int(len(group)),
                "recall_at_1": round(float(group["recall_at_1"].mean()), 4),
                "recall_at_5": round(float(group["recall_at_5"].mean()), 4),
                "recall_at_10": round(float(group["recall_at_10"].mean()), 4),
                "mrr_at_10": round(float(group["mrr_at_10"].mean()), 4),
                "ndcg_at_10": round(float(group["ndcg_at_10"].mean()), 4),
            }
        )
    metrics_df = pd.DataFrame(metrics).sort_values("recall_at_5", ascending=False)
    write_table(PATHS.tables_dir / "retrieval_metrics.csv", metrics_df)
    write_table(PATHS.tables_dir / "retrieval_examples.csv", case_df)
    write_table(PATHS.tables_dir / "retrieval_qualitative_examples.csv", pd.DataFrame(qualitative_rows).head(20))
    _plot_metrics(metrics_df)

    result = {
        "evaluation_type": "sanity_check",
        "queries": int(len(eval_df)),
        "best_retriever": str(metrics_df.iloc[0]["method"]),
        "metrics": metrics,
        "note": "Sanity check: the query article is present in the retrieval index, so scores can be artificially high.",
    }
    write_json(PATHS.results_dir / "retrieval_results.json", result)
    write_json(PATHS.results_dir / "retrieval_sanity_results.json", result)
    return result


def _sample_queries(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if "article_type_weak" not in df.columns or df["article_type_weak"].fillna("").eq("").all():
        return df.sample(n, random_state=RANDOM_STATE)
    pieces = []
    labels = list(df["article_type_weak"].fillna("unknown").value_counts().index)
    per_label = max(1, n // max(1, len(labels)))
    used_indexes = set()
    for label in labels:
        group = df[df["article_type_weak"].fillna("unknown").eq(label)]
        sample = group.sample(min(per_label, len(group)), random_state=RANDOM_STATE)
        used_indexes.update(sample.index.tolist())
        pieces.append(sample)
    sampled = pd.concat(pieces)
    if len(sampled) < n:
        remaining = df[~df.index.isin(used_indexes)]
        if not remaining.empty:
            sampled = pd.concat([sampled, remaining.sample(min(n - len(sampled), len(remaining)), random_state=RANDOM_STATE)])
    return sampled.head(n).reset_index(drop=True)


def _plot_metrics(metrics_df: pd.DataFrame) -> None:
    path = PATHS.figures_dir / "retrieval_metrics.png"
    plot_df = metrics_df.set_index("method")[["recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10"]]
    plot_df.plot(kind="bar", figsize=(9, 5), color=["#476f95", "#7a9b68", "#c59a44", "#8a6f9b", "#b7665c"])
    plt.title("Metricas de recuperacao")
    plt.xlabel("Metodo")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    try:
        plt.savefig(path.with_suffix(".pdf"))
    except Exception:
        pass
    plt.close()


def _log2(value: float) -> float:
    import math

    return math.log(value, 2)
