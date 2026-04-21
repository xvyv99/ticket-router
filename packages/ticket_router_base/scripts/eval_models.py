"""Evaluate existing model predictions and output a comparison table."""

from pathlib import Path
import sys

# add project root to path so ticket_router_base is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ticket_router_base.eval import evaluate_file


PRED_FILES = [
    ("LR", Path("../../outputs/supervised/lr_predictions.jsonl")),
    ("XGB", Path("../../outputs/supervised/xgb_predictions.jsonl")),
    ("mBERT", Path("../../outputs/supervised/mbert_predictions.jsonl")),
    ("Qwen3-0.6B (Agent)", Path("../../outputs/goal_based/Qwen_Qwen3-0.6B_few_shot_predictions.jsonl")),
]


def main() -> None:
    results = []
    for name, path in PRED_FILES:
        if not path.exists():
            print(f"[SKIP] {name}: {path} not found")
            continue
        try:
            report = evaluate_file(path, model_name=name)
            results.append((name, report))
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            continue

    if not results:
        print("No valid prediction files found.")
        return

    # print overall comparison table
    print("\n" + "=" * 100)
    print("Model Evaluation Comparison — Overall Metrics")
    print("=" * 100)
    print(f"{'Model':<22} {'Queue Acc':>10} {'Queue MF1':>10} {'Queue WF1':>10} {'Prio Acc':>10} {'Prio MF1':>10} {'Prio WF1':>10} {'Samples':>8}")
    print("-" * 100)
    for name, report in results:
        q = report.queue_result.overall
        p = report.priority_result.overall
        print(
            f"{name:<22} {q.accuracy:>10.4f} {q.macro_f1:>10.4f} {q.weighted_f1:>10.4f} "
            f"{p.accuracy:>10.4f} {p.macro_f1:>10.4f} {p.weighted_f1:>10.4f} {report.total_samples:>8}"
        )
    print("=" * 100)

    # print by-language queue macro-f1 table
    print("\n" + "=" * 100)
    print("Queue Macro-F1 by Language (Fairness Check)")
    print("=" * 100)
    languages = sorted(next(iter(results))[1].queue_result.by_language.keys())
    header = f"{'Model':<22}"
    for lang in languages:
        header += f" {lang:>8}"
    print(header)
    print("-" * 100)
    for name, report in results:
        line = f"{name:<22}"
        for lang in languages:
            m = report.queue_result.by_language.get(lang)
            val = m.macro_f1 if m else 0.0
            line += f" {val:>8.4f}"
        print(line)
    print("=" * 100)

    # print by-language priority macro-f1 table
    print("\n" + "=" * 100)
    print("Priority Macro-F1 by Language (Fairness Check)")
    print("=" * 100)
    print(header)
    print("-" * 100)
    for name, report in results:
        line = f"{name:<22}"
        for lang in languages:
            m = report.priority_result.by_language.get(lang)
            val = m.macro_f1 if m else 0.0
            line += f" {val:>8.4f}"
        print(line)
    print("=" * 100)

    # print error summary
    print("\n" + "=" * 100)
    print("Error Summary")
    print("=" * 100)
    for name, report in results:
        if report.error_summary != {"SUCCESS": report.total_samples}:
            print(f"{name:<22} {report.error_summary}")
    print("=" * 100)


if __name__ == "__main__":
    main()
