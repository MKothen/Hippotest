
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from platformdirs import user_cache_dir

APP_NAME = "hippo3d"

def cache_root() -> Path:
    return Path(user_cache_dir(APP_NAME))

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
