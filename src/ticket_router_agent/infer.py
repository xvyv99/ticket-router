"""vLLM local inference with structured JSON output."""

import json
from pathlib import Path
from typing import Dict, List

from vllm import SamplingParams, LLM
from vllm.sampling_params import StructuredOutputsParams

from ticket_router_base.data import BaseDataset
from ticket_router_base.predictor import Predictor
from ticket_router_base.types import (
    ErrorFlag,
    Record,
    Prediction,
)

from .config import MAX_TOKEN_LENGTH, SAVE_DIR
from .types import build_ticket_schema
from .utils import normalize_model_id


def parse_llm_output(raw: str, dataset: BaseDataset) -> Dict[str, str]:
    """Parse vLLM structured JSON output into a labels dict.

    Raises JSONDecodeError on failure.
    """
    loaded = json.loads(raw)
    labels: Dict[str, str] = {}
    for task in dataset.classification_tasks + dataset.ordinal_tasks:
        val = loaded.get(task.name)
        if val is not None:
            labels[task.name] = str(val)
    return labels


class vLLMPredictor(Predictor):
    name = "vllm"
    sub_name_required = True  # require sub_name to distinguish different model choices

    model_name_or_path: Path | str
    sub_name: str | None  # set to model choice for save name formatting

    dataset: BaseDataset
    few_shot: bool

    DEFAULT_SAVE_DIR = SAVE_DIR

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
        self.sub_name = normalize_model_id(
            model_name_or_path.stem
            if isinstance(model_name_or_path, Path)
            else model_name_or_path
        )

        self.dataset = dataset
        self.few_shot = few_shot

    def predict(self, records: List[Record]) -> List[Prediction]:
        # Build prompts from records
        prompts = []
        for rec in records:
            prompt = self.dataset.build_system_prompt()
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

                all_tasks = (
                    self.dataset.classification_tasks + self.dataset.ordinal_tasks
                )
                pred = Prediction(
                    request_id=rec.request_id,
                    discrete_features=rec.discrete_features,
                    sensitive_attributes=rec.sensitive_attributes,
                    # results
                    labels=labels,
                    generation_target=gen_target,
                    confidences=None,
                    raw_output=raw,
                    error=ErrorFlag.SUCCESS,
                )
                results.append(pred)
            except json.JSONDecodeError:
                json_err_count += 1
                all_tasks = (
                    self.dataset.classification_tasks + self.dataset.ordinal_tasks
                )

                pred = Prediction(
                    request_id=rec.request_id,
                    discrete_features=rec.discrete_features,
                    sensitive_attributes=rec.sensitive_attributes,
                    # results
                    labels={task.name: "" for task in all_tasks},
                    generation_target=None,
                    confidences=None,
                    raw_output=None,
                    error=ErrorFlag.JSON_ERR,
                )
                results.append(pred)

        return results
