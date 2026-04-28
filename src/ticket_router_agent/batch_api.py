"""SiliconFlow Batch API: generate requests and parse results."""

import json
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

from ticket_router_base.data import BaseDataset
from ticket_router_base.predictor import Predictor, register_model
from ticket_router_base.types import (
    ErrorFlag,
    Prediction,
    Record,
)
from ticket_router_base.utils import JSONLLogger

from .config import SAVE_DIR
from .prompt import build_conversation
from .utils import normalize_model_id
from .infer import vLLMCfg

logger = getLogger(__name__)

BATCH_FILE_DIR = SAVE_DIR / "batch_file"
BATCH_FILE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_id: str
    thinking_budget: int | None
    max_tokens: int
    temperature: float


def build_request_body(model_cfg: ModelConfig, messages: list) -> dict:
    body = {
        "model": model_cfg.model_id,
        "messages": messages,
        "temperature": model_cfg.temperature,
        "max_tokens": model_cfg.max_tokens,
    }
    if model_cfg.thinking_budget is not None:
        body["thinking_budget"] = model_cfg.thinking_budget
        body["enable_thinking"] = True
    return body


def generate_batch_jsonl(
    model_cfg: ModelConfig,
    records: List[Record],
    dataset: BaseDataset,
    few_shot_examples: List[Record] | None = None,
) -> List[Dict[str, Any]]:
    """Build SiliconFlow batch request JSONL objects from records."""
    requests: List[Dict[str, Any]] = []
    for rec in records:
        messages = build_conversation(rec, dataset, few_shot_examples)
        body = build_request_body(model_cfg, messages)
        request = {
            "custom_id": f"req-{rec.request_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        requests.append(request)
    return requests


def parse_batch_result(result_path: Path) -> Generator[Tuple[str, str], None, None]:
    """Yield (custom_id, raw_content) pairs from a batch result JSONL."""
    with JSONLLogger(result_path, mode="r") as logger:
        for line in logger.read():
            obj = json.loads(line)
            custom_id = obj.get("custom_id")
            assert isinstance(custom_id, str), "Missing custom_id in batch result line"
            content = None
            response = obj.get("response")
            if response:
                body = response.get("body")
                assert body is not None, (
                    f"Missing body in batch result for custom_id: {custom_id}"
                )
                choices = body.get("choices")
                assert isinstance(choices, list) and choices, (
                    f"Missing choices in batch result for custom_id: {custom_id}"
                )
                if choices:
                    content = choices[0].get("message")
                    assert content is not None, (
                        f"Missing message content in batch result for custom_id: {custom_id}"
                    )
                    messages = choices[0].get("message")
                    content = messages.get("content")
            assert isinstance(content, str), (
                f"Missing content in batch result for custom_id: {custom_id}"
            )
            yield custom_id, content


def _parse_llm_json(
    raw: str, dataset: BaseDataset
) -> Tuple[Dict[str, str], str | None]:
    """Parse raw LLM JSON output into labels dict and generation target."""
    td = dataset.task_descriptor
    loaded = json.loads(raw)

    labels: Dict[str, str] = {}
    for task in td.classification_tasks + td.ordinal_tasks:
        val = loaded.get(task.name)
        if val is not None:
            labels[task.name] = str(val)

    gen_target = None
    if td.generation_task:
        gen_target = loaded.get(td.generation_task.name)

    return labels, gen_target


@register_model
class BatchAPIPredictor(Predictor[vLLMCfg]):
    name = "batch-api"
    sub_name_required = True
    DEFAULT_SAVE_DIR = SAVE_DIR

    dataset: BaseDataset
    model_cfg: ModelConfig

    def __init__(
        self, dataset: BaseDataset, model_cfg: ModelConfig, few_shot: bool = True
    ):
        self.dataset = dataset
        self.model_cfg = model_cfg
        self.few_shot = few_shot
        self.sub_name = normalize_model_id(model_cfg.name)

        self.cfg = vLLMCfg(few_shot=self.few_shot)

    def predict(self, records: List[Record], run_id: int = 0) -> List[Prediction]:
        raise NotImplementedError(
            "Batch API is async; use gen_batch() and parse_result() instead."
        )

    def gen_batch(
        self,
        records: List[Record],
        save_path: Path | None = None,
    ) -> Path:
        """Generate batch request JSONL and save to disk.

        Returns the path to the saved file.
        """
        if save_path is None:
            save_path = BATCH_FILE_DIR / f"batch_{self.sub_name}.jsonl"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        few_shot_examples = None
        if self.few_shot:
            few_shot_examples = self.dataset.sample_few_shot_examples(
                max_per_stratum=1,
                max_total=5,
            )

        requests = generate_batch_jsonl(
            self.model_cfg, records, self.dataset, few_shot_examples
        )
        with JSONLLogger(save_path) as jsonl:
            for req in requests:
                jsonl.write(req)

        size_kb = save_path.stat().st_size // 1024
        logger.info(f"{save_path.name}: {len(requests)} lines, {size_kb} KB")
        return save_path

    def parse_result(
        self,
        result_path: Path,
        records: List[Record],
        run_id: int = 0,
        save_path: Path | None = None,
    ) -> List[Prediction]:
        """Parse a batch result JSONL and save predictions.

        Returns the list of Prediction objects, one per input record.
        """
        # Build a lookup from request_id -> raw LLM output
        result_map: Dict[str, str] = {}
        for custom_id, raw in parse_batch_result(result_path):
            request_id = custom_id.replace("req-", "", 1)
            result_map[request_id] = raw

        predictions: List[Prediction] = []
        json_err_count = 0
        missing_count = 0
        td = self.dataset.task_descriptor

        for rec in records:
            raw = result_map.get(rec.request_id)
            if raw is None:
                logger.warning(f"Result not found for request_id: {rec.request_id}")
                missing_count += 1
                labels = {
                    task.name: "" for task in td.classification_tasks + td.ordinal_tasks
                }
                pred = Prediction(
                    request_id=rec.request_id,
                    discrete_features=rec.discrete_features,
                    sensitive_attributes=rec.sensitive_attributes,
                    labels=labels,
                    generation_target=None,
                    confidences=None,
                    raw_output=None,
                    error=ErrorFlag.JSON_ERR,
                )
            else:
                try:
                    labels, gen_target = _parse_llm_json(raw, self.dataset)
                    pred = Prediction(
                        request_id=rec.request_id,
                        discrete_features=rec.discrete_features,
                        sensitive_attributes=rec.sensitive_attributes,
                        labels=labels,
                        generation_target=gen_target,
                        confidences=None,
                        raw_output=raw,
                        error=ErrorFlag.SUCCESS,
                    )
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning(f"Failed to parse result for {rec.request_id}: {e}")
                    json_err_count += 1
                    labels = {
                        task.name: ""
                        for task in td.classification_tasks + td.ordinal_tasks
                    }
                    pred = Prediction(
                        request_id=rec.request_id,
                        discrete_features=rec.discrete_features,
                        sensitive_attributes=rec.sensitive_attributes,
                        labels=labels,
                        generation_target=None,
                        confidences=None,
                        raw_output=None,
                        error=ErrorFlag.JSON_ERR,
                    )
            predictions.append(pred)

        # Save using the base class mechanism
        self.save_pred_inst(predictions, records, run_id=run_id, save_path=save_path)

        logger.info(
            f"Parsed {len(predictions)} predictions ({missing_count} missing, {json_err_count} JSON errors)"
        )
        return predictions
