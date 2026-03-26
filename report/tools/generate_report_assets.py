from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ProjectPaths:
    """All paths are resolved relative to this file (report/tools/...)."""

    repo_root: Path
    figures_dir: Path
    outputs_dir: Path

    report_dir: Path
    report_assets_figures: Path
    report_assets_outputs: Path
    report_tables_dir: Path


def _resolve_paths() -> ProjectPaths:
    report_dir = Path(__file__).resolve().parents[1]
    repo_root = report_dir.parent

    return ProjectPaths(
        repo_root=repo_root,
        figures_dir=repo_root / "figures",
        outputs_dir=repo_root / "outputs",
        report_dir=report_dir,
        report_assets_figures=report_dir / "assets" / "figures",
        report_assets_outputs=report_dir / "assets" / "outputs",
        report_tables_dir=report_dir / "tables",
    )


def _ensure_dirs(paths: ProjectPaths) -> None:
    paths.report_assets_figures.mkdir(parents=True, exist_ok=True)
    paths.report_assets_outputs.mkdir(parents=True, exist_ok=True)
    paths.report_tables_dir.mkdir(parents=True, exist_ok=True)


def _copy_tree_files(source_dir: Path, dest_dir: Path, patterns: Iterable[str]) -> int:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    copied = 0
    for pattern in patterns:
        for src in source_dir.glob(pattern):
            if not src.is_file():
                continue
            dst = dest_dir / src.name
            shutil.copy2(src, dst)
            copied += 1
    return copied


def _latex_scientific(value: float, digits: int = 2) -> str:
    """
    Convert a float to LaTeX scientific notation.
    Example: 2.521e-175 -> \\(2.52\\times 10^{-175}\\)
    """
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10**exponent)
    mantissa_str = f"{mantissa:.{digits}f}"
    return rf"\({mantissa_str}\times 10^{{{exponent}}}\)"


def _format_p_value(p: float) -> str:
    if p < 1e-3:
        return _latex_scientific(p, digits=2)
    return f"{p:.4f}"


def _format_float(x: float) -> str:
    # Keep high precision when values are extremely close to 1
    if abs(x) < 1e-3 and x != 0.0:
        return f"{x:.6f}"
    if abs(x - 1.0) < 1e-3:
        return f"{x:.7f}".rstrip("0").rstrip(".")
    return f"{x:.4f}"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def _generate_metrics_table(metrics: dict[str, Any]) -> str:
    roc_auc = float(metrics["roc_auc"])
    accuracy = float(metrics["accuracy"])
    precision = float(metrics["precision"])
    recall = float(metrics["recall"])
    f1 = float(metrics["f1"])

    return "\n".join(
        [
            r"\begin{table}[H]",
            r"  \centering",
            r"  \caption{Kết quả đánh giá mô hình trên tập test}",
            r"  \begin{tabular}{@{}ll@{}}",
            r"    \toprule",
            r"    \textbf{Metric} & \textbf{Giá trị} \\",
            r"    \midrule",
            f"    ROC AUC   & {roc_auc:.4f} \\\\",
            f"    Accuracy  & {accuracy:.4f} \\\\",
            f"    Precision & {precision:.4f} \\\\",
            f"    Recall    & {recall:.4f} \\\\",
            f"    F1-score  & {f1:.4f} \\\\",
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def _generate_confusion_matrix_table(metrics: dict[str, Any]) -> str:
    cm = metrics["confusion_matrix"]
    tn = int(cm[0][0])
    fp = int(cm[0][1])
    fn = int(cm[1][0])
    tp = int(cm[1][1])

    return "\n".join(
        [
            r"\begin{table}[H]",
            r"  \centering",
            r"  \caption{Confusion matrix}",
            r"  \begin{tabular}{@{}lcc@{}}",
            r"    \toprule",
            r"    & \textbf{Dự đoán: 0 (Ở lại)} & \textbf{Dự đoán: 1 (Rời bỏ)} \\",
            r"    \midrule",
            f"    \\textbf{{Thực tế: 0 (Ở lại)}} & TN = {tn} & FP = {fp} \\\\",
            f"    \\textbf{{Thực tế: 1 (Rời bỏ)}} & FN = {fn} & TP = {tp} \\\\",
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def _generate_odds_ratio_longtable(csv_path: Path) -> str:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    lines: list[str] = [
        r"\begin{longtable}{@{}p{4.2cm}p{2.4cm}p{5.2cm}p{2.2cm}@{}}",
        r"  \caption{Bảng Odds Ratio (OR), 95\% CI và p-value} \\",
        r"  \toprule",
        r"  \textbf{Feature} & \textbf{OR} & \textbf{95\% CI} & \textbf{p-value} \\",
        r"  \midrule",
        r"  \endfirsthead",
        r"  \toprule",
        r"  \textbf{Feature} & \textbf{OR} & \textbf{95\% CI} & \textbf{p-value} \\",
        r"  \midrule",
        r"  \endhead",
    ]

    for r in rows:
        feature = (r["feature"] or "").replace("_", r"\_")
        or_val = float(r["odds_ratio"])
        ci_l = float(r["ci_lower"])
        ci_u = float(r["ci_upper"])
        p = float(r["p_value"])

        or_str = _format_float(or_val)
        ci_str = rf"[{_format_float(ci_l)},\;{_format_float(ci_u)}]"
        p_str = _format_p_value(p)

        lines.append(f"  {feature} & {or_str} & {ci_str} & {p_str} \\\\")

    lines.extend([r"  \bottomrule", r"\end{longtable}", ""])
    return "\n".join(lines)


def main() -> None:
    paths = _resolve_paths()
    _ensure_dirs(paths)

    # 1) Sync artifacts into report/assets (self-contained report folder)
    copied_fig = _copy_tree_files(paths.figures_dir, paths.report_assets_figures, patterns=["*.png"])
    copied_out = _copy_tree_files(
        paths.outputs_dir,
        paths.report_assets_outputs,
        patterns=["*.json", "*.csv", "*.txt"],
    )

    # 2) Generate LaTeX tables directly from synced outputs
    metrics_path = paths.report_assets_outputs / "logistic_metrics.json"
    or_table_path = paths.report_assets_outputs / "odds_ratio_table.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing required metrics: {metrics_path}")
    if not or_table_path.exists():
        raise FileNotFoundError(f"Missing required OR table: {or_table_path}")

    metrics = _read_json(metrics_path)

    _write_text(paths.report_tables_dir / "metrics.tex", _generate_metrics_table(metrics))
    _write_text(paths.report_tables_dir / "confusion_matrix.tex", _generate_confusion_matrix_table(metrics))
    _write_text(paths.report_tables_dir / "odds_ratio_table.tex", _generate_odds_ratio_longtable(or_table_path))

    print("OK - Synced artifacts and generated tables.")
    print(f"- Copied figures: {copied_fig} -> {paths.report_assets_figures}")
    print(f"- Copied outputs: {copied_out} -> {paths.report_assets_outputs}")
    print(f"- Generated tables -> {paths.report_tables_dir}")


if __name__ == "__main__":
    main()


