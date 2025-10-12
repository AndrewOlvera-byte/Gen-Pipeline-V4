from __future__ import annotations
from pathlib import Path
import os


def resolve_repo_root() -> Path:

    return Path(__file__).resolve().parents[2]


def ensure_data_dir(repo_root: Path) -> Path:

    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # Hint Minari to store datasets under this directory
    os.environ.setdefault("MINARI_DATASET_PATH", str(data_dir))
    return data_dir

