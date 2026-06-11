from __future__ import annotations

from typing import Any

import pandas as pd

from .config import PATHS, ensure_dirs
from .io_utils import read_json, write_json


def report_results(quick: bool = False, force: bool = False) -> dict[str, Any]:
    ensure_dirs()
    dataset_summary = _read_table("dataset_summary.csv")
    weak_labels = _read_table("weak_label_distribution.csv")
    classification_metrics = _read_table("classification_metrics.csv")
    retrieval_metrics = _read_table("retrieval_metrics.csv")
    retrieval_strict_metrics = _read_table("retrieval_strict_metrics.csv")
    briefing_metrics = _read_table("briefing_metrics_by_configuration.csv")
    briefing_strict_metrics = _read_table("briefing_strict_metrics_by_configuration.csv")

    classification_results = read_json(PATHS.results_dir / "classification_results.json", default={}) or {}
    retrieval_sanity_results = (
        read_json(PATHS.results_dir / "retrieval_sanity_results.json", default={})
        or read_json(PATHS.results_dir / "retrieval_results.json", default={})
        or {}
    )
    retrieval_strict_results = read_json(PATHS.results_dir / "retrieval_strict_results.json", default={}) or {}
    briefing_sanity_results = read_json(PATHS.results_dir / "briefing_evaluation_results.json", default={}) or {}
    briefing_strict_results = read_json(PATHS.results_dir / "briefing_strict_evaluation_results.json", default={}) or {}

    markdown = _build_markdown(
        dataset_summary=dataset_summary,
        weak_labels=weak_labels,
        classification_metrics=classification_metrics,
        retrieval_metrics=retrieval_metrics,
        retrieval_strict_metrics=retrieval_strict_metrics,
        briefing_metrics=briefing_metrics,
        briefing_strict_metrics=briefing_strict_metrics,
        classification_results=classification_results,
        retrieval_sanity_results=retrieval_sanity_results,
        retrieval_strict_results=retrieval_strict_results,
    )
    report_path = PATHS.results_dir / "processing_report.md"
    report_path.write_text(markdown, encoding="utf-8")

    intro_path = PATHS.results_dir / "introduction_draft.md"
    intro_path.write_text(_build_introduction(), encoding="utf-8")

    result = {
        "dataset_summary": dataset_summary.to_dict(orient="records"),
        "weak_labels": weak_labels.to_dict(orient="records"),
        "classification": classification_results,
        "retrieval_sanity": retrieval_sanity_results,
        "retrieval_strict": retrieval_strict_results,
        "briefing_sanity": briefing_sanity_results,
        "briefing_strict": briefing_strict_results,
        "generated_tables": sorted(str(path) for path in PATHS.tables_dir.glob("*.csv")),
        "generated_figures": sorted(str(path) for path in PATHS.figures_dir.glob("*.*")),
        "processing_report": str(report_path),
        "introduction_draft": str(intro_path),
    }
    write_json(PATHS.results_dir / "processing_report.json", result)
    return result


def _read_table(name: str) -> pd.DataFrame:
    path = PATHS.tables_dir / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _build_markdown(
    dataset_summary: pd.DataFrame,
    weak_labels: pd.DataFrame,
    classification_metrics: pd.DataFrame,
    retrieval_metrics: pd.DataFrame,
    retrieval_strict_metrics: pd.DataFrame,
    briefing_metrics: pd.DataFrame,
    briefing_strict_metrics: pd.DataFrame,
    classification_results: dict[str, Any],
    retrieval_sanity_results: dict[str, Any],
    retrieval_strict_results: dict[str, Any],
) -> str:
    summary = _metric_dict(dataset_summary)
    best_model = classification_results.get("best_model", "")
    best_classification = _best_classification_row(classification_metrics, best_model)
    best_sanity_retriever = retrieval_sanity_results.get("best_retriever", "")
    best_sanity_retrieval = _row_by_value(retrieval_metrics, "method", best_sanity_retriever)
    best_strict_retriever = retrieval_strict_results.get("best_retriever", "")
    best_strict_retrieval = _row_by_value(retrieval_strict_metrics, "method", best_strict_retriever)

    lines = [
        "# Relatorio de processamento - Virtual Reporter",
        "",
        "## 1. Dataset used",
        "",
        "- Registros brutos no dataset consolidado: 10.189.",
        "- Registros canonicos no dataset consolidado: 8.681.",
        f"- Registros usados apos filtros: {summary.get('canonical_model_ready_records', 'n/d')}.",
        f"- Media de palavras: {summary.get('avg_word_count', 'n/d')}.",
        f"- Mediana de palavras: {summary.get('median_word_count', 'n/d')}.",
        "",
        "## 2. Weak labels",
        "",
        "Os tipos de noticia sao weak labels criados por regras. Eles servem para experimentos iniciais e nao substituem anotacao humana.",
        "",
        _markdown_table(weak_labels.head(20)),
        "",
        "## 3. Sanity check results",
        "",
        "### 3.1 Classification",
        "",
        f"- Melhor modelo por macro-F1: {best_model or 'n/d'}.",
        f"- Accuracy: {best_classification.get('accuracy', 'n/d')}.",
        f"- Macro-F1: {best_classification.get('macro_f1', 'n/d')}.",
        f"- Weighted-F1: {best_classification.get('weighted_f1', 'n/d')}.",
        "- A classificacao avalia a capacidade de reproduzir weak labels, nao categorias humanas definitivas.",
        "",
        "### 3.2 Original retrieval evaluation",
        "",
        f"- Melhor recuperador: {best_sanity_retriever or 'n/d'}.",
        f"- Recall@5: {best_sanity_retrieval.get('recall_at_5', 'n/d')}.",
        f"- MRR@10: {best_sanity_retrieval.get('mrr_at_10', 'n/d')}.",
        f"- nDCG@10: {best_sanity_retrieval.get('ndcg_at_10', 'n/d')}.",
        "- Este resultado e um sanity check: o artigo usado para formar a query esta presente no indice.",
        "",
        "### 3.3 Original briefing evaluation",
        "",
        _markdown_table(briefing_metrics),
        "",
        "## 4. Strict evaluation results",
        "",
        "### 4.1 Strict retrieval evaluation",
        "",
        "- O indice strict e construido apenas com artigos de treino.",
        "- As queries sao geradas a partir de artigos de teste.",
        "- O artigo original da query nao esta no indice.",
        "- A query nao usa o titulo completo e remove datas, URLs e alguns nomes especificos quando possivel.",
        f"- Melhor recuperador strict: {best_strict_retriever or 'n/d'}.",
        f"- same_type_at_5: {best_strict_retrieval.get('same_type_at_5', 'n/d')}.",
        f"- same_type_ratio_at_5: {best_strict_retrieval.get('same_type_ratio_at_5', 'n/d')}.",
        f"- same_unit_at_5: {best_strict_retrieval.get('same_unit_at_5', 'n/d')}.",
        f"- nDCG@10 aproximado: {best_strict_retrieval.get('ndcg_at_10_approx', 'n/d')}.",
        "",
        _markdown_table(retrieval_strict_metrics),
        "",
        "### 4.2 Strict briefing evaluation",
        "",
        _markdown_table(briefing_strict_metrics),
        "",
        "A configuracao virtual_reporter usa classificacao, deteccao de campos faltantes, perguntas por template e recuperacao de noticias semelhantes sem incluir a noticia original no indice strict.",
        "",
        "## 5. Figures and tables generated",
        "",
        "Tabelas principais em `data/results/tables/` e figuras em `data/results/figures/`.",
        "",
        "## 6. Limitations",
        "",
        "- Os rotulos de tipo de noticia sao weak labels heuristicos, nao labels humanos.",
        "- A classificacao avalia reproducao desses weak labels, nao categorias jornalisticas definitivas.",
        "- A primeira avaliacao de retrieval e apenas um sanity check.",
        "- A avaliacao strict e mais proxima do cenario real, porque a noticia de teste nao esta disponivel no indice.",
        "- A avaliacao automatica de factualidade e aproximada e deve ser complementada por avaliacao humana.",
        "- O sistema gera briefings estruturados, nao textos finais para publicacao.",
    ]
    return "\n".join(lines) + "\n"


def _build_introduction() -> str:
    return """# Introducao

Instituicoes publicas de ensino produzem diariamente comunicados, editais, chamadas de eventos, resultados, informes de servico e noticias sobre ensino, pesquisa e extensao. Embora esse fluxo seja essencial para a transparencia institucional e para o relacionamento com estudantes, servidores e comunidade externa, a producao jornalistica tende a depender de informacoes iniciais incompletas, enviadas por diferentes setores e em formatos heterogeneos. Esse cenario aumenta o trabalho de triagem das equipes de comunicacao e pode atrasar a transformacao de demandas internas em pautas estruturadas.

Este artigo apresenta o Virtual Reporter, uma abordagem experimental para apoiar a etapa inicial da producao de noticias institucionais. O sistema nao substitui a decisao editorial nem gera automaticamente uma publicacao final. Em vez disso, organiza pitches incompletos, identifica o tipo provavel da noticia, aponta campos ausentes, sugere perguntas complementares e recupera noticias semelhantes previamente publicadas para oferecer contexto e exemplos de cobertura.

O estudo utiliza um corpus consolidado de noticias do IFMT coletadas em portais historicos e atuais, incluindo registros do portal legado por IP, do portal atual, de portais de campi e de subdominios antigos. A versao bruta contem 10.189 registros e a versao canonica contem 8.681 noticias apos deduplicacao. A partir desse corpus, sao avaliados tres componentes: classificacao de tipo de noticia institucional, recuperacao de noticias semelhantes e geracao de briefings estruturados a partir de pitches incompletos.

Como ainda nao ha rotulos humanos para todas as noticias, os experimentos iniciais usam rotulagem fraca baseada em regras para classes como editais e selecoes, eventos, pesquisa, extensao, ensino, resultados/premiacoes, gestao institucional, comunicados de servico e outros. Essa escolha permite obter uma primeira linha de base quantitativa, mas tambem delimita a principal restricao metodologica: os resultados medem a consistencia do pipeline sob labels heuristicos e precisam ser complementados por avaliacao humana em trabalhos futuros.
"""


def _metric_dict(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or not {"metric", "value"}.issubset(df.columns):
        return {}
    return {str(row["metric"]): row["value"] for _, row in df.iterrows()}


def _best_classification_row(df: pd.DataFrame, model_name: str) -> dict[str, Any]:
    if df.empty:
        return {}
    candidates = df[df["split"].eq("test")]
    if model_name:
        candidates = candidates[candidates["model_name"].eq(model_name)]
    if candidates.empty:
        return {}
    return candidates.iloc[0].to_dict()


def _row_by_value(df: pd.DataFrame, column: str, value: str) -> dict[str, Any]:
    if df.empty or column not in df.columns:
        return {}
    candidates = df[df[column].astype(str).eq(str(value))]
    if candidates.empty:
        return {}
    return candidates.iloc[0].to_dict()


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "Sem dados disponiveis."
    shown = df.copy().fillna("")
    columns = [str(col) for col in shown.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in shown.iterrows():
        values = [str(row[col]).replace("|", "/") for col in shown.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
