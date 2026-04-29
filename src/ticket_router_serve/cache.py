"""JSON file cache for prediction results."""

import fcntl
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from ticket_router_serve.config import CACHE_DIR


def _ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def compute_fingerprint(title: str | None, body: str, model: str) -> str:
    """Compute SHA256 fingerprint for request deduplication."""
    text = f"{title or ''}|{body}|{model}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _index_path() -> Path:
    return _ensure_cache_dir() / "index.json"


def _lock_path() -> Path:
    return _ensure_cache_dir() / ".index.lock"


def load_fingerprint_index() -> dict[str, str]:
    """Load fingerprint → req_id index."""
    path = _index_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_fingerprint_index(index: dict[str, str]) -> None:
    path = _index_path()
    lock_path = _lock_path()
    with lock_path.open("w", encoding="utf-8") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def save_fingerprint(fingerprint: str, req_id: str) -> str | None:
    """Save fingerprint → req_id mapping. Returns existing req_id if already cached."""
    lock_path = _lock_path()
    with lock_path.open("w", encoding="utf-8") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            index = load_fingerprint_index()
            if fingerprint in index:
                return index[fingerprint]
            index[fingerprint] = req_id
            path = _index_path()
            with path.open("w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
            return None
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def find_by_fingerprint(fingerprint: str) -> str | None:
    """Look up req_id by fingerprint. Returns None if not cached."""
    index = load_fingerprint_index()
    return index.get(fingerprint)


def _entry_path(req_id: str) -> Path:
    return _ensure_cache_dir() / f"{req_id}.json"


def get_cache_entry(req_id: str) -> dict[str, Any] | None:
    """Load a cache entry by req_id."""
    path = _entry_path(req_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def set_cache_entry(req_id: str, data: dict[str, Any]) -> None:
    """Atomically write a cache entry (temp file + rename)."""
    path = _entry_path(req_id)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    shutil.move(str(temp_path), str(path))


def update_cache_entry(req_id: str, data: dict[str, Any]) -> None:
    """Atomically update a cache entry (write to temp then rename)."""
    path = _entry_path(req_id)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    shutil.move(str(temp_path), str(path))