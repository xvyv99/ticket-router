from typing import Type

from ticket_router_base.data import BaseDataset
from ticket_router_base.predictor import Predictor
from .evaluator import TaskEvaluator
from .report import EvaluationReport


def evaluate_model(model: Type[Predictor]):
    pred_saves = model.load_pred()
    pred_file_path = model.get_save_path()

    evaluator = TaskEvaluator()
    task_results = evaluator.evaluate(pred_saves, model.dataset)

    # TODO: error summary

    return EvaluationReport(
        model_name=model.name,
        dataset=model.dataset,
        file_path=pred_file_path,
        task_results=task_results,
        error_summary={},
        total_samples=len(pred_saves),
    )
