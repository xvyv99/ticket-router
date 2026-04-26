"""Main inference logic using vLLM."""

from pathlib import Path
from typing import List, Optional, Literal

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from .base import get_inferrer
from .schema import AttributePrediction, InferenceResult, UserType, TechProficiency


def infer_attributes(
    model: str,
    dataset_name: str = "multilingual-customer-support",
    dataset_path: Optional[Path] = None,
    limit: Optional[int] = None,
    split: Literal["train", "test", "valid"] = "test",
    max_tokens: int = 1024,
    temperature: float = 0.3,
    output_path: Optional[Path] = None,
) -> InferenceResult:
    """Run attribute inference using vLLM.

    Args:
        model: vLLM model path or name.
        dataset_name: Name of the dataset to infer on.
        dataset_path: Optional path to dataset CSV file.
        limit: Optional limit on number of records.
        split: Which data split to use (train/test/valid).
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        output_path: Optional output path for JSONL results.

    Returns:
        InferenceResult containing predictions and statistics.
    """
    inferrer = get_inferrer(dataset_name)

    print(f"Loading records from dataset '{dataset_name}' (split={split})...")

    records = inferrer.load_records(
        dataset_path=dataset_path,
        limit=limit,
        split=split,
    )
    print(f"Loaded {len(records)} records")

    llm = LLM(
        model=model,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=8092,
    )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<|im_end|>"],
        structured_outputs=StructuredOutputsParams(json=inferrer.output_schema),
    )

    conversations = [inferrer.build_conversation(rec) for rec in records]

    print(f"Running inference on {len(conversations)} records...")
    outputs = llm.chat(
        conversations,  # type: ignore
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    predictions: List[AttributePrediction] = []
    error_count = 0

    for rec, output in zip(records, outputs):
        raw = output.outputs[0].text.strip()
        pred = inferrer.parse_output(raw, rec.request_id)

        if pred.user_type == UserType.UNKNOWN:
            pred.user_type = None
            if "parse failed" in pred.reason.lower():
                error_count += 1

        if pred.tech_proficiency == TechProficiency.UNKNOWN:
            pred.tech_proficiency = None
            if "parse failed" in pred.reason.lower():
                error_count += 1

        predictions.append(pred)

    print(f"\nTotal errors: {error_count}/{len(records)}")

    if output_path:
        inferrer.save_predictions(predictions, output_path)
        print(f"Saved predictions to {output_path}")

    return InferenceResult(
        predictions=predictions,
        error_count=error_count,
        total_count=len(records),
    )
