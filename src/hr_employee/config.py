from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Centralized project paths (tập trung đường dẫn) để tránh hardcode."""

    project_root: Path
    data_dir: Path
    figures_dir: Path
    outputs_dir: Path


def get_default_paths(project_root: Path | None = None) -> ProjectPaths:
    root = project_root or Path(__file__).resolve().parents[2]
    return ProjectPaths(
        project_root=root,
        data_dir=root / "data",
        figures_dir=root / "figures",
        outputs_dir=root / "outputs",
    )


