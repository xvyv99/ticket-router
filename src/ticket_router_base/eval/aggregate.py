"""Aggregate evaluation results across multiple runs."""

from typing import Dict, List, Tuple

import numpy as np

from .evaluator import TaskEvaluationResult
from .metrics import OrdinalMetrics, PerClassMetrics
from .fairness_metrics import FairnessMetrics
from .report import EvaluationReport


def _mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def _aggregate_fairness(
    fairness_list: List[Dict[str, FairnessMetrics]],
) -> Tuple[Dict[str, FairnessMetrics], Dict[str, FairnessMetrics]]:
    """Aggregate fairness metrics across runs.

    Returns (mean_fairness, std_fairness) mapping sensitive_attr -> FairnessMetrics.
    """
    # Collect all sensitive attributes present across runs
    all_attrs = set()
    for fdict in fairness_list:
        all_attrs.update(fdict.keys())

    mean_fairness: Dict[str, FairnessMetrics] = {}
    std_fairness: Dict[str, FairnessMetrics] = {}

    # FIXME: should depend on actual fields of FairnessMetrics, not hardcoded list
    scalar_fields = [
        "accuracy_gap",
        "accuracy_ratio",
        "macro_f1_gap",
        "macro_f1_ratio",
        "avg_disparate_impact",
        "avg_statistical_parity_difference",
        "avg_equal_opportunity_difference",
        "avg_average_odds_difference",
    ]

    for attr in all_attrs:
        fms = [fdict[attr] for fdict in fairness_list if attr in fdict]
        base_fm = fms[0]
        base_dump = base_fm.model_dump()

        mean_dump = dict(base_dump)
        std_dump = dict(base_dump)

        # by_group metrics: collect all groups across runs
        all_acc_groups: set[str] = set()
        all_f1_groups: set[str] = set()
        for fm in fms:
            all_acc_groups.update(fm.accuracy_by_group.keys())
            all_f1_groups.update(fm.macro_f1_by_group.keys())

        mean_dump["accuracy_by_group"] = {}
        std_dump["accuracy_by_group"] = {}
        for g in all_acc_groups:
            vals = [fm.accuracy_by_group[g] for fm in fms if g in fm.accuracy_by_group]
            if vals:
                mean_dump["accuracy_by_group"][g] = _mean_std(vals)[0]
                std_dump["accuracy_by_group"][g] = _mean_std(vals)[1]

        mean_dump["macro_f1_by_group"] = {}
        std_dump["macro_f1_by_group"] = {}
        for g in all_f1_groups:
            vals = [fm.macro_f1_by_group[g] for fm in fms if g in fm.macro_f1_by_group]
            if vals:
                mean_dump["macro_f1_by_group"][g] = _mean_std(vals)[0]
                std_dump["macro_f1_by_group"][g] = _mean_std(vals)[1]

        # scalar fields
        for field in scalar_fields:
            vals = [getattr(fm, field) for fm in fms if getattr(fm, field) is not None]
            if vals:
                mean_dump[field], std_dump[field] = _mean_std(vals)
            else:
                mean_dump[field] = None
                std_dump[field] = None

        mean_fairness[attr] = FairnessMetrics(**mean_dump)
        std_fairness[attr] = FairnessMetrics(**std_dump)

    return mean_fairness, std_fairness


def aggregate_task_results(
    run_results: List[List[TaskEvaluationResult]],
) -> Tuple[List[TaskEvaluationResult], List[TaskEvaluationResult]]:
    """Aggregate task results across multiple runs.

    Args:
        run_results: List of task result lists, one per run.

    Returns:
        (mean_results, std_results) where both are List[TaskEvaluationResult]
        with performance and fairness fields holding mean and std respectively.
    """
    if not run_results:
        raise ValueError("No run results to aggregate")

    n_tasks = len(run_results[0])

    mean_results: List[TaskEvaluationResult] = []
    std_results: List[TaskEvaluationResult] = []

    for task_idx in range(n_tasks):
        task_name = run_results[0][task_idx].task_name
        perfs = [run[task_idx].performance for run in run_results]
        base_perf = perfs[0]

        # FIXME: should depend on actual fields of PerformanceMetrics, not hardcoded list
        # Top-level scalar fields
        perf_fields = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
        if isinstance(base_perf, OrdinalMetrics):
            perf_fields.extend(["mae", "qwk"])

        mean_values: dict[str, float] = {}
        std_values: dict[str, float] = {}
        for field in perf_fields:
            values = [getattr(p, field) for p in perfs]
            mean_val, std_val = _mean_std(values)
            mean_values[field] = mean_val
            std_values[field] = std_val

        # Per-class std (safe: some runs may lack a label)
        std_per_class: dict[str, dict] = {}
        for label in base_perf.labels:
            precisions = [
                p.per_class[label].precision for p in perfs if label in p.per_class
            ]
            recalls = [p.per_class[label].recall for p in perfs if label in p.per_class]
            f1s = [p.per_class[label].f1 for p in perfs if label in p.per_class]
            if not precisions:
                continue
            support = (
                base_perf.per_class[label].support
                if label in base_perf.per_class
                else 0
            )
            std_per_class[label] = PerClassMetrics(
                precision=_mean_std(precisions)[1],
                recall=_mean_std(recalls)[1],
                f1=_mean_std(f1s)[1],
                support=support,
            ).model_dump()

        # Build mean performance
        mean_dump = base_perf.model_dump()
        for field in perf_fields:
            mean_dump[field] = mean_values[field]
        mean_perf = type(base_perf)(**mean_dump)

        # Build std performance (same structure, values are std deviations)
        std_dump = base_perf.model_dump()
        for field in perf_fields:
            std_dump[field] = std_values[field]
        std_dump["per_class"] = std_per_class
        std_perf = type(base_perf)(**std_dump)

        # Fairness aggregation
        fairness_list = [run[task_idx].fairness for run in run_results]
        mean_fairness, std_fairness = _aggregate_fairness(fairness_list)

        mean_results.append(
            TaskEvaluationResult(
                task_name=task_name,
                performance=mean_perf,
                fairness=mean_fairness,
            )
        )
        std_results.append(
            TaskEvaluationResult(
                task_name=task_name,
                performance=std_perf,
                fairness=std_fairness,
            )
        )

    return mean_results, std_results


def aggregate_reports(reports: List[EvaluationReport]) -> EvaluationReport:
    """Aggregate multiple EvaluationReports (one per run) into a single report.

    Args:
        reports: List of EvaluationReports, one per run.

    Returns:
        A single EvaluationReport with aggregated mean metrics and std deviations.
    """
    if not reports:
        raise ValueError("No reports to aggregate")

    run_results = [r.task_results for r in reports]
    mean_results, std_results = aggregate_task_results(run_results)

    # Aggregate error_summary by summing counts across runs
    agg_errors: dict[str, int] = {}
    for r in reports:
        for err_name, count in r.error_summary.items():
            agg_errors[err_name] = agg_errors.get(err_name, 0) + count

    return EvaluationReport(
        model_name=reports[0].model_name,
        dataset=reports[0].dataset,
        file_path=None,
        task_results=mean_results,
        task_stds=std_results,
        run_results=run_results,
        total_samples=reports[0].total_samples,
        n_runs=len(reports),
        error_summary=agg_errors,
    )
