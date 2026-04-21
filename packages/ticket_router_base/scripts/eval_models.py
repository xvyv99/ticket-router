"""Evaluate existing model predictions and output a comparison table.

Usage:
    uv run python scripts/eval_models.py --dataset multilingual-customer-support
    uv run python scripts/eval_models.py --dataset french-gov-oss --pred-files lr:xgb
"""

from pathlib import Path
import sys
from argparse import ArgumentParser

# add project root to path so ticket_router_base is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ticket_router_base.eval import evaluate_file
from ticket_router_base.datasets import get_dataset, DATASET_REGISTRY


def main() -> None:
    parser = ArgumentParser(description="Evaluate model predictions")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()),
        default="multilingual-customer-support",
        help="Dataset to evaluate against",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("../../outputs/supervised"),
        help="Directory containing prediction JSONL files",
    )
    parser.add_argument(
        "--pred-files",
        type=str,
        default="lr:xgb:mbert",
        help="Colon-separated list of prediction file stems (e.g. lr:xgb)",
    )
    args = parser.parse_args()

    dataset = get_dataset(args.dataset)
    stems = [s.strip() for s in args.pred_files.split(":") if s.strip()]

    results = []
    for stem in stems:
        path = args.pred_dir / f"{stem}_predictions.jsonl"
        if not path.exists():
            print(f"[SKIP] {stem}: {path} not found")
            continue
        try:
            report = evaluate_file(path, dataset, model_name=stem)
            results.append((stem, report))
        except Exception as e:
            print(f"[ERROR] {stem}: {e}")
            continue

    if not results:
        print("No valid prediction files found.")
        return

    task_names = [tr.task_name for tr in results[0][1].task_results]

    # overall comparison table — one row per model, columns per task
    col_width = 10
    print("\n" + "=" * (30 + len(task_names) * (col_width * 2 + 4)))
    print("Model Evaluation Comparison — Overall Metrics")
    print("=" * (30 + len(task_names) * (col_width * 2 + 4)))
    header = f"{'Model':<22}"
    for tn in task_names:
        header += f" {tn[:8]+' Acc':>10} {tn[:8]+' MF1':>10}"
    print(header)
    print("-" * len(header))
    for name, report in results:
        line = f"{name:<22}"
        for tr in report.task_results:
            line += f" {tr.overall.accuracy:>10.4f} {tr.overall.macro_f1:>10.4f}"
        print(line)
    print("=" * len(header))

    # by-language fairness check (using first task as representative)
    if results[0][1].task_results and results[0][1].task_results[0].by_language:
        first_task = results[0][1].task_results[0]
        languages = sorted(first_task.by_language.keys())
        print("\n" + "=" * (30 + len(languages) * 10))
        print(f"{first_task.task_name} Macro-F1 by Language (Fairness Check)")
        print("=" * (30 + len(languages) * 10))
        header = f"{'Model':<22}"
        for lang in languages:
            header += f" {lang:>8}"
        print(header)
        print("-" * len(header))
        for name, report in results:
            line = f"{name:<22}"
            tr = report.task_results[0]
            for lang in languages:
                m = tr.by_language.get(lang)
                val = m.macro_f1 if m else 0.0
                line += f" {val:>8.4f}"
            print(line)
        print("=" * len(header))

    # error summary
    print("\n" + "=" * 60)
    print("Error Summary")
    print("=" * 60)
    for name, report in results:
        if report.error_summary != {"SUCCESS": report.total_samples}:
            print(f"{name:<22} {report.error_summary}")
    print("=" * 60)


if __name__ == "__main__":
    main()
