from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PATHS, ensure_dirs
from .io_utils import read_jsonl, write_table


TABLE_ALIASES = {
    "dataset_summary.csv": "dataset_summary_table.csv",
    "classification_metrics.csv": "classification_metrics_table.csv",
    "retrieval_metrics.csv": "retrieval_metrics_table.csv",
    "retrieval_strict_metrics.csv": "paper_retrieval_strict_results.csv",
    "briefing_metrics_by_configuration.csv": "briefing_metrics_table.csv",
    "briefing_strict_metrics_by_configuration.csv": "paper_briefing_strict_results.csv",
}


def make_paper_tables() -> dict[str, Any]:
    ensure_dirs()
    outputs = []
    for source_name, target_name in TABLE_ALIASES.items():
        source = PATHS.tables_dir / source_name
        target = PATHS.tables_dir / target_name
        if source.exists():
            shutil.copyfile(source, target)
            outputs.append(str(target))

    outputs.append(str(_write_generated_question_examples()))
    outputs.extend(_write_paper_specific_tables())
    return {"tables": outputs}


def _write_generated_question_examples() -> Path:
    outputs = read_jsonl(PATHS.results_dir / "briefing_outputs.jsonl")
    rows = []
    for output in outputs:
        if output.get("configuration") != "virtual_reporter":
            continue
        for question in output.get("generated_questions") or []:
            rows.append(
                {
                    "case_id": output.get("case_id"),
                    "predicted_article_type": output.get("predicted_article_type"),
                    "field": question.get("field"),
                    "question": question.get("question"),
                    "input_pitch": output.get("input_pitch"),
                }
            )
    path = PATHS.tables_dir / "examples_of_generated_questions.csv"
    write_table(path, pd.DataFrame(rows).head(100))
    return path


def _write_paper_specific_tables() -> list[str]:
    outputs = []
    classification = PATHS.tables_dir / "classification_metrics.csv"
    if classification.exists():
        df = pd.read_csv(classification)
        paper = df[df["split"].eq("test")].copy() if "split" in df else df
        path = PATHS.tables_dir / "paper_classification_results.csv"
        write_table(path, paper)
        outputs.append(str(path))

    dataset = PATHS.tables_dir / "dataset_summary.csv"
    if dataset.exists():
        df = pd.read_csv(dataset)
        path = PATHS.tables_dir / "paper_dataset_processing_summary.csv"
        write_table(path, df)
        outputs.append(str(path))
    return outputs
