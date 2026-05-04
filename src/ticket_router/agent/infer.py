"""vLLM local inference with structured JSON output."""

import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from logging import getLogger

from vllm import SamplingParams, LLM
from vllm.sampling_params import StructuredOutputsParams

from ticket_router.base.config import SEED
from ticket_router.base.data import BaseDataset
from ticket_router.base.data.desc import TaskDescriptor
from ticket_router.base.predictor import Predictor, register_model
from ticket_router.base.types import (
    ErrorFlag,
    Record,
    Prediction,
)
from ticket_router.base.cfg import Cfg

from .config import MAX_TOKEN_LENGTH, SAVE_DIR
from .prompt import build_conversation
from .types import build_ticket_schema
from .utils import normalize_model_id

logger = getLogger(__name__)


def parse_llm_output(raw: str, task_descriptor: TaskDescriptor) -> Dict[str, str]:
    """Parse vLLM structured JSON output into a labels dict.

    Raises JSONDecodeError on failure.
    """
    loaded = json.loads(raw)
    labels: Dict[str, str] = {}
    for task in task_descriptor.classification_tasks + task_descriptor.ordinal_tasks:
        val = loaded.get(task.name)
        if val is not None:
            labels[task.name] = str(val)
    return labels


@dataclass(frozen=True)
class vLLMCfg(Cfg):
    few_shot: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@register_model
class vLLMPredictor(Predictor[vLLMCfg]):
    name = "vllm"
    sub_name_required = True  # require sub_name to distinguish different model choices

    model_name_or_path: Path | str
    sub_name: str | None  # set to model choice for save name formatting

    dataset: BaseDataset

    DEFAULT_SAVE_DIR = SAVE_DIR

    _llm: LLM
    _schema: dict

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

        # Load model once during init to avoid reloading on every predict() call
        self._llm = LLM(
            model=str(self.model_name_or_path),
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len=MAX_TOKEN_LENGTH,
        )

        # tokenizer = self._llm.get_tokenizer()

        # Pre-build schema since it doesn't change across runs
        self._schema = build_ticket_schema(self.dataset.task_descriptor)

        self.cfg = vLLMCfg(few_shot=few_shot)

    def predict(self, records: List[Record], run_id: int = 0) -> List[Prediction]:
        # Sample few-shot examples once (shared across all records in this batch)
        few_shot_examples = None
        if self.few_shot:
            few_shot_examples = self.dataset.sample_few_shot_examples(
                max_per_stratum=1,
                max_total=5,
            )

        # Build conversation prompts for vLLM chat API
        conversations: List[List[Dict[str, str]]] = []
        for rec in records:
            conv = build_conversation(rec, self.dataset, few_shot_examples)
            conversations.append(conv)

        td = self.dataset.task_descriptor
        sampling_params = SamplingParams(
            max_tokens=1024,
            temperature=0.7,
            seed=SEED + run_id,
            stop=["<|im_end|>"],
            structured_outputs=StructuredOutputsParams(json=self._schema),
        )

        outputs = self._llm.chat(
            conversations,  # type: ignore
            sampling_params=sampling_params,
            use_tqdm=True,  # type: ignore
        )

        json_err_count = 0
        results: List[Prediction] = []

        for rec, output in zip(records, outputs):
            raw = output.outputs[0].text.strip()
            try:
                labels = parse_llm_output(raw, td)

                # generation target
                gen_target = None
                if td.generation_task:
                    gen_target = json.loads(raw).get(td.generation_task.name)

                all_tasks = td.classification_tasks + td.ordinal_tasks
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
                all_tasks = td.classification_tasks + td.ordinal_tasks

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

        logger.info(
            f"vLLM prediction completed with {json_err_count} JSON decode errors out of {len(records)} samples"
        )

        return results
