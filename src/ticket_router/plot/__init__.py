"""Publication-quality visualization package for ticket router results.

Usage:
    from ticket_router.plot.style import set_academic_style
    from ticket_router.plot.eda import plot_class_distribution
    from ticket_router.plot.fairness import plot_pareto_frontier
"""

# Core utilities
from .style import save_figure, set_academic_style

# EDA plots
from .eda import (
    plot_class_distribution,
    plot_inferred_attributes,
    plot_language_queue_heatmap,
    plot_priority_by_attributes,
    plot_priority_by_language,
    plot_queue_by_attributes,
    plot_queue_priority_heatmap,
    plot_tag_coverage,
    plot_text_lengths,
    plot_text_vs_label,
)

# Fairness plots
from .fairness import (
    plot_cross_task_fairness,
    plot_gap_with_errorbars,
    plot_language_di_matrices,
    plot_llm_scale_fairness,
    plot_pareto_frontier,
    plot_spearman_correlation,
    plot_thinking_vs_nothinking,
    plot_unfairness_radar,
    plot_user_type_eod,
)

# Evaluation plots
from .evaluation import (
    plot_accuracy_priority,
    plot_accuracy_queue,
    plot_fairness_heatmap,
    plot_macro_f1_priority,
    plot_macro_f1_queue,
    plot_scaling_curve,
)

# Robustness plots
from .robustness import (
    plot_robustness_accuracy_drop_by_lang,
    plot_robustness_attack_success_by_lang,
    plot_robustness_clean_vs_perturbed_by_lang,
    plot_robustness_recipe_attack_success,
    plot_robustness_recipe_clean_vs_perturbed,
)

__all__ = [
    # Style
    "save_figure",
    "set_academic_style",
    # EDA
    "plot_class_distribution",
    "plot_inferred_attributes",
    "plot_language_queue_heatmap",
    "plot_priority_by_attributes",
    "plot_priority_by_language",
    "plot_queue_by_attributes",
    "plot_queue_priority_heatmap",
    "plot_tag_coverage",
    "plot_text_lengths",
    "plot_text_vs_label",
    # Fairness
    "plot_cross_task_fairness",
    "plot_gap_with_errorbars",
    "plot_language_di_matrices",
    "plot_llm_scale_fairness",
    "plot_pareto_frontier",
    "plot_spearman_correlation",
    "plot_thinking_vs_nothinking",
    "plot_unfairness_radar",
    "plot_user_type_eod",
    # Evaluation
    "plot_accuracy_priority",
    "plot_accuracy_queue",
    "plot_fairness_heatmap",
    "plot_macro_f1_priority",
    "plot_macro_f1_queue",
    "plot_scaling_curve",
    # Robustness
    "plot_robustness_accuracy_drop_by_lang",
    "plot_robustness_attack_success_by_lang",
    "plot_robustness_clean_vs_perturbed_by_lang",
    "plot_robustness_recipe_attack_success",
    "plot_robustness_recipe_clean_vs_perturbed",
]
