"""Environment configuration loaded from .env."""

import os
from logging import getLogger
from pathlib import Path
from dotenv import load_dotenv

logger = getLogger(__name__)

load_dotenv()

# API Keys for service authentication (comma-separated in .env)
_api_keys_raw = os.getenv("API_KEYS", "")
API_KEYS: set[str] = set(k.strip() for k in _api_keys_raw.split(",") if k.strip())

if not API_KEYS:
    logger.warning("No API_KEYS configured — service will reject all requests")

DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
if not DASHSCOPE_API_KEY:
    logger.warning("DASHSCOPE_API_KEY not set — qwen3 model will not work")

# Cache directory for JSON file storage
CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", "src/ticket_router_serve/cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
