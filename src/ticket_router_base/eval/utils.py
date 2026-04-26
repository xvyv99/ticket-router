from typing import Type

from ticket_router_base.data import BaseDataset
from ticket_router_base.predictor import Predictor
from .evaluator import TaskEvaluator
from .report import EvaluationReport


def evaluate_model_dataset(
    model: Type[Predictor], dataset: BaseDataset, sub_name: str | None = None
) -> EvaluationReport:
    pred_saves = model.load_pred(dataset, sub_name)
    pred_file_path = model.get_save_path(dataset, sub_name)

    evaluator = TaskEvaluator()
    task_results = evaluator.evaluate(pred_saves, dataset)

    # TODO: error summary

    return EvaluationReport(
        model_name=model.name,
        dataset=dataset,
        file_path=pred_file_path,
        task_results=task_results,
        error_summary={},
        total_samples=len(pred_saves),
    )
