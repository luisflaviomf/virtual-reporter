from __future__ import annotations

from collections import Counter
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .config import PATHS, RANDOM_STATE, ensure_dirs
from .io_utils import save_dataframe, write_table
from .load_dataset import load_model_ready
from .text_utils import text_key


LABEL_PRIORITY = [
    "edital_selection",
    "award_result",
    "event",
    "research",
    "extension",
    "teaching",
    "institutional_management",
    "service_announcement",
    "other",
]


KEYWORDS: dict[str, list[str]] = {
    "edital_selection": [
        "edital",
        "editais",
        "seleção",
        "processo seletivo",
        "inscrição",
        "inscrições",
        "resultado",
        "convocação",
        "homologação",
        "retificação",
        "chamada pública",
        "vaga",
        "vagas",
        "bolsista",
        "professor substituto",
        "tutor",
    ],
    "event": [
        "evento",
        "seminário",
        "palestra",
        "encontro",
        "reunião",
        "audiência",
        "workshop",
        "oficina",
        "jornada",
        "congresso",
        "semana acadêmica",
        "simpósio",
        "live",
    ],
    "research": [
        "pesquisa",
        "pesquisador",
        "iniciação científica",
        "ciência",
        "projeto de pesquisa",
        "inovação",
        "patente",
        "tecnologia",
        "publicação científica",
    ],
    "extension": [
        "extensão",
        "comunidade",
        "curso gratuito",
        "ação social",
        "projeto de extensão",
        "oficina para comunidade",
        "programa de extensão",
    ],
    "teaching": [
        "ensino",
        "curso",
        "estudante",
        "aluno",
        "aula",
        "avaliação do MEC",
        "licenciatura",
        "bacharelado",
        "técnico",
        "formação",
    ],
    "award_result": [
        "premiação",
        "prêmio",
        "medalha",
        "olimpíada",
        "conquista",
        "aprovado",
        "destaque",
        "resultado final",
        "classificação",
    ],
    "institutional_management": [
        "reitoria",
        "conselho",
        "codir",
        "consepe",
        "posse",
        "nomeação",
        "portaria",
        "reunião ordinária",
        "gestão",
        "planejamento",
        "parceria institucional",
    ],
    "service_announcement": [
        "comunicado",
        "suspensão",
        "expediente",
        "manutenção",
        "férias",
        "calendário",
        "funcionamento",
        "aviso",
    ],
}

NORMALIZED_KEYWORDS = {
    label: [text_key(keyword) for keyword in keywords] for label, keywords in KEYWORDS.items()
}


def weak_label(sample_size: int | None = None, quick: bool = False, force: bool = False) -> dict[str, Any]:
    ensure_dirs()
    df = load_model_ready().copy()
    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=RANDOM_STATE)
    elif quick and len(df) > 1000:
        df = df.sample(1000, random_state=RANDOM_STATE)

    labels = []
    matches = []
    for _, row in df.iterrows():
        label, row_matches = assign_weak_label(row)
        labels.append(label)
        matches.append(row_matches)
    df["article_type_weak"] = labels
    df["article_type_weak_matches"] = matches

    save_dataframe(
        df,
        PATHS.processed_dir / "articles_labeled.parquet",
        PATHS.processed_dir / "articles_labeled.csv",
    )

    distribution = (
        df["article_type_weak"]
        .value_counts()
        .rename_axis("article_type_weak")
        .reset_index(name="count")
    )
    distribution["percent"] = (distribution["count"] / len(df) * 100).round(2)
    write_table(PATHS.tables_dir / "weak_label_distribution.csv", distribution)
    _plot_distribution(distribution)

    review_sample = _stratified_sample(df, 200)
    cols = [
        c
        for c in [
            "global_article_id",
            "title",
            "lead",
            "publication_date",
            "publisher_unit_normalized",
            "source_environment",
            "source_host",
            "article_type_weak",
            "article_type_weak_matches",
            "original_url",
        ]
        if c in review_sample.columns
    ]
    write_table(PATHS.results_dir / "manual_label_review_sample.csv", review_sample[cols])

    return {
        "labeled_records": len(df),
        "distribution": distribution.to_dict(orient="records"),
        "outputs": {
            "parquet": str(PATHS.processed_dir / "articles_labeled.parquet"),
            "csv": str(PATHS.processed_dir / "articles_labeled.csv"),
            "distribution": str(PATHS.tables_dir / "weak_label_distribution.csv"),
            "manual_review": str(PATHS.results_dir / "manual_label_review_sample.csv"),
        },
    }


def assign_weak_label(row: pd.Series | dict[str, Any]) -> tuple[str, dict[str, list[str]]]:
    title = row.get("title", "")
    body = row.get("body_text", "")
    lead = row.get("lead", "")
    normalized = text_key(f"{title} {lead} {body}")

    found: dict[str, list[str]] = {}
    for label, keywords in NORMALIZED_KEYWORDS.items():
        hits = [keyword for keyword in keywords if keyword and keyword in normalized]
        if hits:
            found[label] = hits

    for label in LABEL_PRIORITY:
        if label in found:
            return label, found
    return "other", found


def _stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty:
        return df
    pieces = []
    labels = list(df["article_type_weak"].value_counts().index)
    per_label = max(1, n // max(1, len(labels)))
    for label in labels:
        group = df[df["article_type_weak"].eq(label)]
        pieces.append(group.sample(min(per_label, len(group)), random_state=RANDOM_STATE))
    sample = pd.concat(pieces, ignore_index=True)
    if len(sample) < min(n, len(df)):
        remaining = df.drop(sample.index, errors="ignore")
        if not remaining.empty:
            sample = pd.concat(
                [
                    sample,
                    remaining.sample(min(n - len(sample), len(remaining)), random_state=RANDOM_STATE),
                ],
                ignore_index=True,
            )
    return sample.head(n)


def _plot_distribution(distribution: pd.DataFrame) -> None:
    path = PATHS.figures_dir / "weak_label_distribution.png"
    plt.figure(figsize=(10, 5))
    plt.bar(distribution["article_type_weak"], distribution["count"], color="#61734a")
    plt.title("Distribuicao dos rotulos fracos")
    plt.xlabel("Tipo de noticia")
    plt.ylabel("Artigos")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    try:
        plt.savefig(path.with_suffix(".pdf"))
    except Exception:
        pass
    plt.close()
