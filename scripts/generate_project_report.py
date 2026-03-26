from __future__ import annotations

from hr_employee.config import get_default_paths
from hr_employee.reporting.project_report import write_project_report


def main() -> None:
    paths = get_default_paths()
    out_path = write_project_report(paths)
    # Use ASCII-only output for Windows console compatibility.
    print(f"[OK] Generated PROJECT_REPORT.md: {out_path}")


if __name__ == "__main__":
    main()


