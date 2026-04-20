import json
from enum import Enum
from pathlib import Path
from typing import Annotated, List

from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

MODEL_SIZES = ["0.6B", "1.7B", "4B"]
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
CALIB_OUTPUT = (
    Path(__file__).parent.parent.parent / "outputs" / "goal_based" / "calibration.jsonl"
)
TEST_SET = Path(__file__).parent.parent.parent / "outputs" / "test_set.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "goal_based"

QUEUES = [
    "Billing and Payments",
    "Customer Service",
    "General Inquiry",
    "Human Resources",
    "IT Support",
    "Product Support",
    "Returns and Exchanges",
    "Sales and Pre-Sales",
    "Service Outages and Maintenance",
    "Technical Support",
]
PRIORITIES = ["high", "medium", "low"]

FEW_SHOT_EXAMPLES = [
    {
        "queue": "Billing and Payments",
        "priority": "high",
        "tags": ["Billing Issue", "Invoice Review"],
        "preliminary_answer": "Dear customer, we have received your billing inquiry and are reviewing the charges on your account. We will provide a detailed breakdown within 24 hours. Thank you for your patience.",
    },
    {
        "queue": "Technical Support",
        "priority": "high",
        "tags": ["Network Hardware", "Router Issue"],
        "preliminary_answer": "Sehr geehrter Kunde, wir haben Ihre Meldung zu den Netzwerkproblemen mit Ihrem Cisco Router erhalten. Unser technisches Team wird sich schnellstmöglich mit Ihnen in Verbindung setzen.",
    },
    {
        "queue": "Customer Service",
        "priority": "low",
        "tags": ["Account Assistance", "General Inquiry"],
        "preliminary_answer": "Olá, agradecemos o seu contacto. Estamos a rever a sua solicitação e responderemos em breve com as informações necessárias.",
    },
    {
        "queue": "Product Support",
        "priority": "medium",
        "tags": ["Hardware Issue", "Device Troubleshooting"],
        "preliminary_answer": "Bonjour, nous avons bien reçu votre signalement concernant les problèmes techniques avec votre appareil. Notre équipe de support produit analysera votre dossier et reviendra vers vous sous 48 heures.",
    },
    {
        "queue": "IT Support",
        "priority": "medium",
        "tags": ["Cloud Infrastructure", "Performance Optimization"],
        "preliminary_answer": "Hello, thank you for reaching out regarding your cloud infrastructure needs. Our IT consulting team will prepare a customized optimization plan and contact you within 2 business days.",
    },
    {
        "queue": "Returns and Exchanges",
        "priority": "low",
        "tags": ["Product Return", "Order Issue"],
        "preliminary_answer": "Sehr geehrte Kundin, vielen Dank für Ihre Nachricht. Wir haben Ihre Rückgabeanfrage erhalten und werden Ihnen innerhalb von 3-5 Werktagen ein Rücksendeetikett zusenden.",
    },
    {
        "queue": "Service Outages and Maintenance",
        "priority": "high",
        "tags": ["Service Outage", "Incident Report"],
        "preliminary_answer": "Olá, lamentamos sinceramente pela interrupção do serviço. Nossa equipa de operações está a trabalhar ativamente. Fürneceremos uma atualização dentro de 60 minutos.",
    },
    {
        "queue": "Sales and Pre-Sales",
        "priority": "low",
        "tags": ["Feature Request", "Software Integration"],
        "preliminary_answer": "Estimado cliente, gracias por su interés en nuestra solución. Nuestro equipo de preventas analizará su solicitud y se pondrá en contacto con usted en breve para coordinar una demostración personalizada.",
    },
    {
        "queue": "Human Resources",
        "priority": "medium",
        "tags": ["HR Inquiry", "Employee Support"],
        "preliminary_answer": "Dear employee, thank you for reaching out to HR. We have received your inquiry and will follow up with the relevant department. Please expect a response within 3 business days.",
    },
    {
        "queue": "General Inquiry",
        "priority": "low",
        "tags": ["General Information", "Customer Inquiry"],
        "preliminary_answer": "Estimado cliente, gracias por ponerse en contacto con nosotros. Hemos registrado su consulta y le proporcionaremos la información solicitada en breve. Atentamente, el equipo de atención al cliente.",
    },
]


# ---------------------------------------------------------------------------
# Pydantic model for vLLM structured outputs
# ---------------------------------------------------------------------------
class QueueEnum(str, Enum):
    BILLING = "Billing and Payments"
    CUSTOMER_SERVICE = "Customer Service"
    GENERAL_INQUIRY = "General Inquiry"
    HUMAN_RESOURCES = "Human Resources"
    IT_SUPPORT = "IT Support"
    PRODUCT_SUPPORT = "Product Support"
    RETURNS = "Returns and Exchanges"
    SALES = "Sales and Pre-Sales"
    SERVICE_OUTAGES = "Service Outages and Maintenance"
    TECHNICAL_SUPPORT = "Technical Support"


class PriorityEnum(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TicketOutput(BaseModel):
    queue: QueueEnum
    priority: PriorityEnum
    tags: Annotated[List[str], Field(min_length=1, max_length=3)]
    preliminary_answer: str


TICKET_SCHEMA = TicketOutput.model_json_schema()


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_system_prompt(few_shot: bool = True) -> str:
    base = (
        "You are a multilingual customer support assistant. "
        "Analyze the user's email (subject and body) and respond ONLY with a valid JSON object. "
        "Do not include markdown formatting outside the JSON.\n\n"
        "The JSON must have these exact keys:\n"
        "- queue: one of " + ", ".join(QUEUES) + "\n"
        "- priority: one of " + ", ".join(PRIORITIES) + "\n"
        "- tags: an array of 1-3 relevant tags (strings)\n"
        "- preliminary_answer: a polite, helpful customer service reply in the same language as the user's email\n"
    )
    if not few_shot:
        return base + (
            "\nExample:\n"
            '{"queue": "Billing and Payments", "priority": "high", "tags": ["Billing Issue"], "preliminary_answer": "Dear customer, we are reviewing your charges and will respond within 24 hours."}'
        )
    examples_str = "\n".join(
        '{{"queue": "{q}", "priority": "{p}", "tags": {t}, "preliminary_answer": "{a}"}}'.format(
            q=e["queue"],
            p=e["priority"],
            t=json.dumps(e["tags"]),
            a=e["preliminary_answer"],
        )
        for e in FEW_SHOT_EXAMPLES
    )
    return base + "\nExamples:\n" + examples_str


def build_prompt(subject: str, body: str, language: str, few_shot: bool = True) -> str:
    system = build_system_prompt(few_shot=few_shot)
    user_text = f"Language: {language}\nSubject: {subject}\nBody: {body}"
    return f"{system}\n\n{user_text}"


def make_sampling_params(few_shot: bool, max_tokens: int = 256):
    if few_shot:
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["<|im_end|>"],
            structured_outputs=StructuredOutputsParams(json=TICKET_SCHEMA),
        )
    return SamplingParams(
        max_tokens=512,
        temperature=0.7,
        stop=["<|im_end|>"],
        structured_outputs=StructuredOutputsParams(json=TICKET_SCHEMA),
    )


def parse_llm_output(raw: str) -> TicketOutput:
    """Parse vLLM structured output into TicketOutput. Raises JSONDecodeError on failure."""
    loaded = json.loads(raw)
    return TicketOutput.model_validate(loaded)


def make_result(rec: dict, out: TicketOutput, raw: str) -> dict:
    return {
        "request_id": rec.get("request_id", "unknown"),
        "language": rec.get("language", "en"),
        "predicted": {
            "queue": out.queue.value,
            "priority": out.priority.value,
            "tags": out.tags,
            "preliminary_answer": out.preliminary_answer,
        },
        "ground_truth": rec.get("ground_truth"),
        "raw_response": raw,
        "success": True,
    }


def make_error_result(rec: dict, raw: str) -> dict:
    return {
        "request_id": rec.get("request_id", "unknown"),
        "language": rec.get("language", "en"),
        "predicted": {
            "queue": "parse_error",
            "priority": "parse_error",
            "tags": [],
            "preliminary_answer": "",
        },
        "ground_truth": rec.get("ground_truth"),
        "raw_response": raw,
        "success": False,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def test_inference(size, n_samples=20, few_shot: bool = True):
    suffix = "fewshot" if few_shot else "zero"

    quant_dir = MODEL_DIR / f"qwen3-{size}-awq"
    if not quant_dir.exists():
        print(f"Quantized model not found: {quant_dir}, skipping.")
        return

    print(f"\n=== Testing: Qwen3-{size} ({n_samples} samples, {suffix}) ===")
    llm = LLM(
        model=str(quant_dir),
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
    )

    records = []
    with open(CALIB_OUTPUT, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    records = records[:n_samples]

    prompts = [rec["messages"][0]["content"] for rec in records]
    sampling_params = make_sampling_params(few_shot=few_shot, max_tokens=256)
    outputs = llm.generate(prompts, sampling_params)

    print(f"\n--- Qwen3-{size} samples ---")
    results = []
    for i, (rec, output) in enumerate(zip(records, outputs)):
        text = rec["messages"][0]["content"]
        generated = output.outputs[0].text.strip()
        print(f"\n[{i + 1}] Input: {text[:80]}...")
        print(f"    Output: {generated[:200]}")
        try:
            out = parse_llm_output(generated)
            results.append(make_result(rec, out, generated))
        except (json.JSONDecodeError, Exception):
            results.append(make_error_result(rec, generated))
    print("\n--- End ---")

    out_path = OUTPUT_DIR / f"qwen35_{size}_{suffix}_predictions.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} results -> {out_path}")


def run_full_evaluation(size, few_shot: bool = True):
    suffix = "fewshot" if few_shot else "zero"

    quant_dir = MODEL_DIR / f"qwen3-{size}-awq"
    if not quant_dir.exists():
        print(f"Quantized model not found: {quant_dir}, skipping.")
        return

    records = [
        json.loads(line)
        for line in TEST_SET.read_text(encoding="utf-8").strip().split("\n")
        if line.strip()
    ]
    print(f"Loaded {len(records)} test records")
    print(f"\n=== Full Evaluation: Qwen3-{size} ({len(records)} samples, {suffix}) ===")

    llm = LLM(
        model=str(quant_dir),
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=8092,
    )

    prompts = [
        build_prompt(
            rec["subject"], rec["body"], rec.get("language", "en"), few_shot=few_shot
        )
        for rec in records
    ]

    sampling_params = make_sampling_params(few_shot=few_shot, max_tokens=256)
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, (rec, output) in enumerate(zip(records, outputs)):
        raw = output.outputs[0].text.strip()
        try:
            out = parse_llm_output(raw)
            results.append(make_result(rec, out, raw))
        except (json.JSONDecodeError, Exception):
            results.append(make_error_result(rec, raw))

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(records)}...")

    out_path = OUTPUT_DIR / f"qwen35_{size}_{suffix}_predictions.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    success_count = sum(1 for o in results if o.get("success"))
    print(f"\nDone: {success_count}/{len(results)} success -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3 vLLM inference")
    parser.add_argument(
        "sizes",
        nargs="*",
        choices=MODEL_SIZES,
        default=MODEL_SIZES,
        help="Model sizes to run (default: all)",
    )
    parser.add_argument(
        "--full", action="store_true", help="Run full evaluation (1200 samples)"
    )
    parser.add_argument(
        "--no-few-shot",
        dest="few_shot",
        action="store_false",
        default=True,
        help="Disable few-shot prompting (zero-shot mode)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of samples for test mode (default: 20)",
    )
    args = parser.parse_args()

    for size in args.sizes:
        if args.full:
            run_full_evaluation(size, few_shot=args.few_shot)
        else:
            test_inference(size, n_samples=args.n_samples, few_shot=args.few_shot)
