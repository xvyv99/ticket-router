"""vLLM local inference with structured JSON output."""

import json
from pathlib import Path
from typing import Dict, List

from vllm import SamplingParams, LLM
from vllm.sampling_params import StructuredOutputsParams

from ticket_router_base.datasets.base import BaseDataset
from ticket_router_base.types import (
    ErrorFlag,
    Record,
    PredictionBatch,
    Prediction,
)

from .config import MAX_TOKEN_LENGTH
from .prompt import build_prompt
from .types import build_ticket_schema


def parse_llm_output(raw: str, dataset: BaseDataset) -> Dict[str, str]:
    """Parse vLLM structured JSON output into a labels dict.

    Raises JSONDecodeError on failure.
    """
    loaded = json.loads(raw)
    labels: Dict[str, str] = {}
    for task in dataset.classification_tasks:
        val = loaded.get(task.name)
        if val is not None:
            labels[task.name] = str(val)
    return labels


class vLLMPredictor:
    supports_tags: bool = True
    supports_preliminary_answer: bool = True

    model_name_or_path: Path | str
    dataset: BaseDataset
    few_shot: bool

    def __init__(
        self,
        model_name_or_path: Path | str,
        dataset: BaseDataset,
        few_shot: bool = True,
    ):
        if isinstance(model_name_or_path, Path):
            assert model_name_or_path.exists(), (
                f"Model path not found: {model_name_or_path}"
            )

        self.model_name_or_path = model_name_or_path
        self.dataset = dataset
        self.few_shot = few_shot

    def predict(self, records: List[Record]) -> PredictionBatch:
        # Build prompts from records
        prompts = []
        for rec in records:
            prompt = build_prompt(
                self.dataset,
                rec.title or "",
                rec.body,
                rec.language,
            )
            prompts.append(prompt)

        llm = LLM(
            model=str(self.model_name_or_path),
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len=MAX_TOKEN_LENGTH,
        )

        schema = build_ticket_schema(self.dataset)
        sampling_params = SamplingParams(
            max_tokens=512,
            temperature=0.7,
            stop=["<|im_end|>"],
            structured_outputs=StructuredOutputsParams(json=schema),
        )

        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

        json_err_count = 0
        results: List[Prediction] = []

        for rec, output in zip(records, outputs):
            raw = output.outputs[0].text.strip()
            try:
                labels = parse_llm_output(raw, self.dataset)

                # generation target
                gen_target = None
                if self.dataset.generation_task:
                    gen_target = json.loads(raw).get(self.dataset.generation_task.name)

                pred = Prediction(
                    request_id=rec.request_id,
                    labels=labels,
                    discrete_features={},
                    generation_target=gen_target,
                    confidences={
                        task.name: None for task in self.dataset.classification_tasks
                    },
                    raw_output=raw,
                    error=ErrorFlag.SUCCESS,
                )
                results.append(pred)
            except json.JSONDecodeError:
                json_err_count += 1

                pred = Prediction(
                    request_id=rec.request_id,
                    labels={
                        task.name: task.labels[0]
                        for task in self.dataset.classification_tasks
                    },
                    discrete_features={},
                    generation_target=None,
                    confidences={
                        task.name: None for task in self.dataset.classification_tasks
                    },
                    raw_output=None,
                    error=ErrorFlag.JSON_ERR,
                )
                results.append(pred)

        return PredictionBatch(
            predictions=results,
            parse_err_count=json_err_count,
            parse_json_err_count=json_err_count,
        )
