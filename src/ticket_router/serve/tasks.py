"""Background task submission via ThreadPoolExecutor."""

import concurrent.futures
import uuid
from logging import getLogger
from typing import Any

from ticket_router_base.types import Record, Language
from ticket_router_eval.interpret import HFInterpretabilityEvaluator

from ticket_router.serve.cache import get_cache_entry, update_cache_entry, set_cache_entry
from ticket_router.serve.models import get_pool

logger = getLogger(__name__)

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def submit_task(title: str | None, body: str, model: str) -> str:
    """Submit a prediction task. Returns req_id."""
    req_id = str(uuid.uuid4())

    entry: dict[str, Any] = {
        "req_id": req_id,
        "status": "PENDING",
        "result": None,
        "attribution": None,
        "error": None,
        "model": model,
        "title": title,
        "body": body,
    }
    set_cache_entry(req_id, entry)

    _executor.submit(run_prediction, req_id, title, body, model)
    logger.info(f"Task {req_id} submitted for model {model}")
    return req_id


def run_prediction(req_id: str, title: str | None, body: str, model: str) -> None:
    """Execute prediction in background thread."""
    try:
        entry = get_cache_entry(req_id)
        entry["status"] = "PROCESSING"
        update_cache_entry(req_id, entry)

        # Re-read immediately before writing to avoid race with concurrent readers
        entry = get_cache_entry(req_id)

        record = Record(
            request_id=req_id,
            title=title,
            body=body,
            language=None,
            labels={},
            discrete_features={},
            generation_target=None,
            sensitive_attributes={},
        )

        # Get predictor for non-qwen3 models
        if model == "qwen3":
            # qwen3 uses DashScope API — only generate answer
            answer = _call_qwen3_for_answer(title, body)
            # For qwen3, we skip queue/priority classification since it's API-only
            # Use Unknown as placeholder (or skip if not needed)
            queue = "General Inquiry"
            priority = "medium"
            queue_conf = 0.5
            priority_conf = 0.5
        else:
            pool = get_pool()
            predictor = pool.get_predictor(model)
            predictions = predictor.predict([record], run_id=0)
            pred = predictions[0]

            queue = pred.labels.get("queue", "Unknown")
            priority = pred.labels.get("priority", "Unknown")
            queue_conf = pred.confidences.get("queue", 0.0) if pred.confidences else 0.0
            priority_conf = pred.confidences.get("priority", 0.0) if pred.confidences else 0.0
            answer = None

        result = {
            "queue": queue,
            "priority": priority,
            "answer": answer,
            "confidence": {
                "queue": queue_conf,
                "priority": priority_conf,
            },
        }

        entry = get_cache_entry(req_id)
        entry["status"] = "COMPLETED"
        entry["result"] = result
        update_cache_entry(req_id, entry)
        logger.info(f"Task {req_id} completed")

    except Exception as e:
        logger.exception(f"Task {req_id} failed: {e}")
        entry = get_cache_entry(req_id)
        entry["status"] = "FAILED"
        entry["error"] = str(e)
        update_cache_entry(req_id, entry)


def _call_qwen3_for_answer(title: str | None, body: str) -> str | None:
    """Call DashScope API to generate a preliminary answer."""
    pool = get_pool()
    user_prompt = f"Ticket subject: {title or 'N/A'}\n\nTicket body: {body}"
    messages = [
        {"role": "system", "content": "You are a helpful customer support assistant. Based on the ticket, provide a brief preliminary answer."},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return pool.call_qwen3(messages)
    except Exception as e:
        logger.warning(f"DashScope API call failed: {e}")
        return None


def submit_attribution(req_id: str) -> None:
    """Submit an attribution task for rembert/xlm-roberta."""
    _executor.submit(run_attribution, req_id)


def run_attribution(req_id: str) -> None:
    """Compute attribution in background thread for rembert/xlm-roberta."""
    try:
        entry = get_cache_entry(req_id)
        model = entry.get("model", "")
        if model not in ("rembert", "xlm-roberta"):
            logger.info(f"Attribution not supported for model {model}, skipping")
            return

        title = entry.get("title")
        body = entry.get("body", "")

        pool = get_pool()
        predictor = pool.get_predictor(model)

        n_steps = 8 if model == "rembert" else 10

        record = Record(
            request_id=req_id,
            title=title,
            body=body,
            language=None,
            labels={},
            discrete_features={},
            generation_target=None,
            sensitive_attributes={},
        )

        from ticket_router_base.data import get_dataset

        dataset = get_dataset("multilingual-customer-support")()

        evaluator = HFInterpretabilityEvaluator(
            predictor=predictor,
            dataset=dataset,
            device="cuda",
            n_steps=n_steps,
        )

        reports = evaluator.evaluate(records=[record], top_k=10)

        attribution: dict[str, Any] = {}
        for task_name, task_report in reports.items():
            if not task_report.sample_attributions:
                continue
            sample = task_report.sample_attributions[0]
            attribution[task_name] = {
                "predicted_label": sample.predicted_label,
                "confidence": sample.confidence,
                "top_positive": [
                    {"token": t.token, "score": t.score}
                    for t in sample.top_positive
                ],
                "top_negative": [
                    {"token": t.token, "score": t.score}
                    for t in sample.top_negative
                ],
            }

        entry = get_cache_entry(req_id)
        entry["attribution"] = attribution
        update_cache_entry(req_id, entry)
        logger.info(f"Attribution for {req_id} completed")

    except Exception as e:
        logger.exception(f"Attribution for {req_id} failed: {e}")
        entry = get_cache_entry(req_id)
        entry["attribution"] = None
        entry["error"] = str(e)
        update_cache_entry(req_id, entry)
