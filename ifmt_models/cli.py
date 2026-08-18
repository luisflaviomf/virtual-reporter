from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from .baseline_comparison import run_baseline_comparison
from .build_retrieval_index import build_retrieval_index
from .evaluate_briefings import evaluate_briefings
from .evaluate_retrieval import evaluate_retrieval
from .expanded_evaluation import ExpandedEvaluationConfig, run_expanded_evaluation
from .generate_test_pitches import generate_test_pitches
from .make_figures import make_paper_assets
from .preprocess_dataset import prepare_data
from .report_results import report_results
from .strict_evaluation import evaluate_briefings_strict, evaluate_retrieval_strict
from .train_classifier import train_classifier
from .virtual_reporter_pipeline import run_virtual_reporter
from .weak_labeling import weak_label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ifmt_models", description="Pipeline experimental do Virtual Reporter.")
    subparsers = parser.add_subparsers(dest="command")

    _add_common(subparsers.add_parser("prepare-data"))
    _add_common(subparsers.add_parser("weak-label"))
    _add_common(subparsers.add_parser("train-classifier"), include_no_heavy=True)
    _add_common(subparsers.add_parser("build-retrieval-index"), include_no_heavy=True)

    retrieval_parser = subparsers.add_parser("evaluate-retrieval")
    _add_common_flags(retrieval_parser)
    retrieval_parser.add_argument("--query-count", type=int, default=None)

    strict_retrieval_parser = subparsers.add_parser("evaluate-retrieval-strict")
    _add_common_flags(strict_retrieval_parser)
    strict_retrieval_parser.add_argument("--query-count", type=int, default=None)

    pitch_parser = subparsers.add_parser("generate-test-pitches")
    _add_common_flags(pitch_parser)
    pitch_parser.add_argument("--pitch-count", type=int, default=None)

    _add_common_flags(subparsers.add_parser("run-virtual-reporter"))
    _add_common_flags(subparsers.add_parser("evaluate-briefings"))
    _add_common_flags(subparsers.add_parser("evaluate-briefings-strict"))
    _add_common_flags(subparsers.add_parser("make-paper-assets"))
    _add_common_flags(subparsers.add_parser("report-results"))

    baseline_parser = subparsers.add_parser("run-baseline-comparison")
    _add_common_flags(baseline_parser)
    baseline_parser.add_argument("--case-count", type=int, default=30)
    baseline_parser.add_argument("--seed", type=int, default=42)

    expanded_parser = subparsers.add_parser("run-expanded-evaluation")
    expanded_parser.add_argument("--case-count", type=int, default=300)
    expanded_parser.add_argument("--seed", type=int, default=42)
    expanded_parser.add_argument("--output-dir", required=True)
    expanded_parser.add_argument(
        "--pilot-dir", default="data/results/evaluation"
    )
    expanded_parser.add_argument("--resume", action="store_true")

    run_all = subparsers.add_parser("run-all")
    _add_common(run_all, include_no_heavy=True)

    return parser


def main() -> None:
    parser = build_parser()

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "run-all":
        result = run_all_pipeline(args)
    else:
        result = run_command(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _add_common(parser: argparse.ArgumentParser, include_no_heavy: bool = False) -> None:
    _add_common_flags(parser)
    parser.add_argument("--sample-size", type=int, default=None)
    if include_no_heavy:
        parser.add_argument("--no-heavy-models", action="store_true")


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "run-expanded-evaluation":
        return run_expanded_evaluation(
            ExpandedEvaluationConfig(
                output_dir=Path(args.output_dir),
                pilot_dir=Path(args.pilot_dir),
                case_count=args.case_count,
                seed=args.seed,
                resume=args.resume,
            )
        )
    command_map: dict[str, Callable[..., dict[str, Any]]] = {
        "prepare-data": prepare_data,
        "weak-label": weak_label,
        "train-classifier": train_classifier,
        "build-retrieval-index": build_retrieval_index,
        "evaluate-retrieval": evaluate_retrieval,
        "evaluate-retrieval-strict": evaluate_retrieval_strict,
        "generate-test-pitches": generate_test_pitches,
        "run-virtual-reporter": run_virtual_reporter,
        "evaluate-briefings": evaluate_briefings,
        "evaluate-briefings-strict": evaluate_briefings_strict,
        "make-paper-assets": make_paper_assets,
        "report-results": report_results,
        "run-baseline-comparison": run_baseline_comparison,
    }
    kwargs = {"quick": getattr(args, "quick", False), "force": getattr(args, "force", False)}
    if hasattr(args, "sample_size"):
        kwargs["sample_size"] = args.sample_size
    if hasattr(args, "no_heavy_models"):
        kwargs["no_heavy_models"] = args.no_heavy_models
    if hasattr(args, "query_count"):
        kwargs["query_count"] = args.query_count
    if hasattr(args, "pitch_count"):
        kwargs["pitch_count"] = args.pitch_count
    if hasattr(args, "case_count"):
        kwargs["case_count"] = args.case_count
    if hasattr(args, "seed"):
        kwargs["seed"] = args.seed
    return command_map[args.command](**kwargs)


def run_all_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    steps: list[tuple[str, Callable[..., dict[str, Any]], dict[str, Any]]] = [
        ("prepare-data", prepare_data, {"sample_size": args.sample_size}),
        ("weak-label", weak_label, {}),
        ("train-classifier", train_classifier, {"no_heavy_models": args.no_heavy_models}),
        ("build-retrieval-index", build_retrieval_index, {"no_heavy_models": args.no_heavy_models}),
        ("evaluate-retrieval", evaluate_retrieval, {}),
        ("generate-test-pitches", generate_test_pitches, {}),
        ("run-virtual-reporter", run_virtual_reporter, {}),
        ("evaluate-briefings", evaluate_briefings, {}),
        ("make-paper-assets", make_paper_assets, {}),
        ("report-results", report_results, {}),
    ]
    results = {}
    for name, func, extra in steps:
        print(f"[ifmt_models] running {name}...")
        kwargs = {"quick": args.quick, "force": args.force}
        kwargs.update(extra)
        result = func(**kwargs)
        results[name] = result
    return {"mode": "quick" if args.quick else "full", "steps": results}


if __name__ == "__main__":
    main()
