import torch

from ticket_router_base.config import OUTPUT_DIR

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = OUTPUT_DIR / "supervised"
MODEL_SAVE_DIR = SAVE_DIR / "models"
