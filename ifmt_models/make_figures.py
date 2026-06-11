from __future__ import annotations

import shutil
from typing import Any

from .config import PATHS, ensure_dirs


FIGURE_ALIASES = {
    "articles_by_year.png": "articles_by_year.png",
    "weak_label_distribution.png": "weak_label_distribution.png",
    "confusion_matrix_best_model.png": "classification_confusion_matrix.png",
    "retrieval_metrics.png": "retrieval_metrics.png",
    "retrieval_strict_metrics.png": "retrieval_strict_metrics.png",
    "briefing_metrics_comparison.png": "briefing_metrics_comparison.png",
    "briefing_strict_metrics_comparison.png": "briefing_strict_metrics_comparison.png",
}


def make_paper_figures() -> dict[str, Any]:
    ensure_dirs()
    outputs = []
    for source_name, target_name in FIGURE_ALIASES.items():
        source = PATHS.figures_dir / source_name
        target = PATHS.figures_dir / target_name
        if source.exists() and source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        if target.exists():
            outputs.append(str(target))
        source_pdf = source.with_suffix(".pdf")
        target_pdf = target.with_suffix(".pdf")
        if source_pdf.exists() and source_pdf.resolve() != target_pdf.resolve():
            shutil.copyfile(source_pdf, target_pdf)
        if target_pdf.exists():
            outputs.append(str(target_pdf))
    return {"figures": outputs}


def make_paper_assets(quick: bool = False, force: bool = False) -> dict[str, Any]:
    from .make_tables import make_paper_tables

    table_result = make_paper_tables()
    figure_result = make_paper_figures()
    return {**table_result, **figure_result}
