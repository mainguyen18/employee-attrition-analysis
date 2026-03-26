from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists (đảm bảo thư mục tồn tại)."""
    path.mkdir(parents=True, exist_ok=True)
    return path






