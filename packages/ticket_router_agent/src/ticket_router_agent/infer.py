import json
from pathlib import Path
from typing import List

from vllm import SamplingParams, LLM
from vllm.sampling_params import StructuredOutputsParams

from ticket_router_base.utils import to_records
from ticket_router_base.predictor import Predictor
from ticket_router_base.types import (
    ErrorFlag,
    Priority,
    Queue,
    Record,
    RecordDF,
    PredictionBatch,
    Prediction,
)

from .prompt import build_prompt
from .types import TICKET_SCHEMA, TicketOutput

DEFAULT_SAMPLING_PARAMS = SamplingParams(
    max_tokens=512,
    temperature=0.7,
    stop=["<|im_end|>"],
    structured_outputs=StructuredOutputsParams(json=TICKET_SCHEMA),
)


def parse_llm_output(raw: str) -> TicketOutput:
    """Parse vLLM structured output into TicketOutput. Raises JSONDecodeError on failure."""
    loaded = json.loads(raw)
    return TicketOutput.model_validate(loaded)


class vLLMPredictor(Predictor):
    supports_tags: bool = True
    supports_preliminary_answer: bool = True

    model_path: Path
    few_shot: bool

    def __init__(self, model_path: Path, few_shot: bool = True):
        assert model_path.exists(), f"Model path not found: {model_path}"
        self.model_path = model_path
        self.few_shot = few_shot

        self.llm = LLM(
            model=str(model_path),
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len=8092,
        )

    def predict(self, records: List[Record] | RecordDF) -> PredictionBatch:
        records = to_records(records)

        llm = LLM(
            model=str(self.model_path),
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len=8092,
        )

        prompts = [
            build_prompt(
                rec.subject or "",
                rec.body or "",
                None,
                few_shot=self.few_shot,
            )
            for rec in records
        ]

        outputs = llm.generate(prompts, DEFAULT_SAMPLING_PARAMS, use_tqdm=True)

        json_err_count = 0

        results = []
        for rec, output in zip(records, outputs):
            raw = output.outputs[0].text.strip()
            try:
                out = parse_llm_output(raw)

                pred = Prediction(
                    request_id=rec.request_id,
                    queue=out.queue,
                    priority=out.priority,
                    tag_1=None,
                    # TODO: handle tag prediction properly
                    tag_2=None,
                    answer=None,
                    queue_confidence=None,
                    priority_confidence=None,
                    raw_output=raw,
                    error=ErrorFlag.SUCCESS,
                )

                results.append(pred)
            except json.JSONDecodeError:
                json_err_count += 1

                pred = Prediction(
                    request_id="",
                    queue=Queue.CUSTOMER_SERVICE,
                    priority=Priority.LOW,
                    tag_1=None,
                    tag_2=None,
                    answer=None,
                    queue_confidence=None,
                    priority_confidence=None,
                    raw_output=None,
                    error=ErrorFlag.JSON_ERR,
                )
                results.append(pred)
                # TODO: add regex parsing fallback here in the future to reduce JSON parsing errors

        pred_batch = PredictionBatch(
            predictions=results,
            parse_err_count=json_err_count,
            # TODO: distinguish between JSON parsing errors and regex parsing errors in the future
            parse_json_err_count=json_err_count,
        )

        return pred_batch
