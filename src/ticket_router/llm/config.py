from ticket_router.base.config import OUTPUT_DIR

MODEL_CHOICES_RAW = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]
MODEL_CHOICES_QUANT = [f"qwen3-{m}-awq" for m in ["0.6B", "1.7B", "4B"]]

MODEL_CHOICES = MODEL_CHOICES_RAW + MODEL_CHOICES_QUANT

MAX_TOKEN_LENGTH = 8092

SAVE_DIR = OUTPUT_DIR / "goal_based"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
