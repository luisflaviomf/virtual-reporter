from __future__ import annotations

from collections import Counter
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from .config import PATHS, RANDOM_STATE, ensure_dirs
from .io_utils import write_json, write_table
from .load_dataset import load_labeled


MODEL_SPECS = {
    "tfidf_logreg": PATHS.models_dir / "news_type_tfidf_logreg.joblib",
    "tfidf_linearsvc": PATHS.models_dir / "news_type_tfidf_linearsvc.joblib",
    "embedding_logreg": PATHS.models_dir / "news_type_embedding_logreg.joblib",
}


def train_classifier(
    sample_size: int | None = None,
    quick: bool = False,
    force: bool = False,
    no_heavy_models: bool = False,
) -> dict[str, Any]:
    ensure_dirs()
    df = load_labeled().copy()
    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=RANDOM_STATE)
    elif quick and len(df) > 1000:
        df = df.sample(1000, random_state=RANDOM_STATE)

    train_df, decision = _training_frame(df)
    if train_df["article_type_weak"].nunique() < 2:
        raise RuntimeError("Nao ha classes suficientes para treinar classificadores.")

    x = train_df["text_for_model"].fillna("").astype(str)
    y = train_df["article_type_weak"].astype(str)
    split = _split_data(x, y)

    models = _build_models(no_heavy_models=no_heavy_models)
    metrics: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    predictions_rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}

    for name, model in models.items():
        model.fit(split["x_train"], split["y_train"])
        fitted[name] = model
        joblib.dump(model, MODEL_SPECS[name])

        val_pred = model.predict(split["x_val"])
        test_pred = model.predict(split["x_test"])
        metrics.append(_metric_row(name, "validation", split["y_val"], val_pred))
        metrics.append(_metric_row(name, "test", split["y_test"], test_pred))
        reports[name] = classification_report(
            split["y_test"],
            test_pred,
            output_dict=True,
            zero_division=0,
        )

        for idx, (text, expected, predicted) in enumerate(
            zip(split["x_test"].head(100), split["y_test"].head(100), test_pred[:100])
        ):
            predictions_rows.append(
                {
                    "model_name": name,
                    "sample_index": idx,
                    "expected": expected,
                    "predicted": predicted,
                    "text_preview": text[:300],
                }
            )

    metrics_df = pd.DataFrame(metrics)
    write_table(PATHS.tables_dir / "classification_metrics.csv", metrics_df)

    best_row = metrics_df[metrics_df["split"].eq("test")].sort_values(
        ["macro_f1", "weighted_f1", "accuracy"], ascending=False
    ).iloc[0]
    best_model_name = str(best_row["model_name"])
    best_model = fitted[best_model_name]
    best_pred = best_model.predict(split["x_test"])
    by_class_df = _classification_report_frame(split["y_test"], best_pred)
    write_table(PATHS.tables_dir / "classification_report_by_class.csv", by_class_df)
    write_table(PATHS.tables_dir / "classifier_predictions_sample.csv", pd.DataFrame(predictions_rows))
    _plot_confusion_matrix(split["y_test"], best_pred, best_model_name)

    result = {
        "input_records": len(df),
        "training_records": len(train_df),
        "label_distribution_input": dict(Counter(df["article_type_weak"].astype(str))),
        "label_distribution_training": dict(Counter(train_df["article_type_weak"].astype(str))),
        "training_decision": decision,
        "best_model": best_model_name,
        "best_model_path": str(MODEL_SPECS[best_model_name]),
        "best_macro_f1": float(best_row["macro_f1"]),
        "metrics": metrics,
        "classification_reports": reports,
    }
    write_json(PATHS.results_dir / "classification_results.json", result)
    return result


def _training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = df[df["article_type_weak"].fillna("").ne("")].copy()
    counts = working["article_type_weak"].value_counts()
    min_count = 8
    removed_sparse = counts[counts < min_count].index.tolist()
    working = working[~working["article_type_weak"].isin(removed_sparse)]

    removed_other = False
    other_share = 0.0
    if not working.empty:
        other_share = float((working["article_type_weak"].eq("other")).mean())
    if other_share > 0.45 and working["article_type_weak"].nunique() > 2:
        candidate = working[~working["article_type_weak"].eq("other")]
        if candidate["article_type_weak"].nunique() >= 2:
            working = candidate
            removed_other = True

    return working, {
        "min_class_count": min_count,
        "removed_sparse_classes": removed_sparse,
        "other_share_before_optional_removal": round(other_share, 4),
        "removed_other_from_training": removed_other,
    }


def _split_data(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    stratify = y if y.value_counts().min() >= 3 else None
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    stratify_train_val = y_train_val if y_train_val.value_counts().min() >= 3 else None
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.15 / 0.85,
        random_state=RANDOM_STATE,
        stratify=stratify_train_val,
    )
    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def _build_models(no_heavy_models: bool = False) -> dict[str, Pipeline]:
    tfidf_common = {
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_features": 50000,
        "sublinear_tf": True,
    }
    return {
        "tfidf_logreg": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**tfidf_common)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=None,
                    ),
                ),
            ]
        ),
        "tfidf_linearsvc": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**tfidf_common)),
                ("classifier", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000)),
            ]
        ),
        "embedding_logreg": Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)),
                ("svd", TruncatedSVD(n_components=128, random_state=RANDOM_STATE)),
                ("normalize", Normalizer(copy=False)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def _metric_row(model_name: str, split: str, y_true: pd.Series, y_pred) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "split": split,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }


def _classification_report_frame(y_true: pd.Series, y_pred) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=sorted(set(y_true) | set(y_pred)),
        zero_division=0,
    )
    return pd.DataFrame(
        [
            {
                "article_type_weak": label,
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
                "support": int(s),
            }
            for label, p, r, f, s in zip(sorted(set(y_true) | set(y_pred)), precision, recall, f1, support)
        ]
    )


def _plot_confusion_matrix(y_true: pd.Series, y_pred, model_name: str) -> None:
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    path = PATHS.figures_dir / "confusion_matrix_best_model.png"

    plt.figure(figsize=(max(7, len(labels) * 0.75), max(6, len(labels) * 0.65)))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(f"Matriz de confusao - {model_name}")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predito")
    plt.ylabel("Esperado")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    try:
        plt.savefig(path.with_suffix(".pdf"))
    except Exception:
        pass
    plt.close()
