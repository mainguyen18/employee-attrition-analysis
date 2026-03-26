from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True, slots=True)
class OddsRatioInterpretation:
    feature: str
    odds_ratio: float
    ci_lower: float
    ci_upper: float
    p_value: float
    direction: str
    magnitude_pct: float
    significance: str
    strength: str
    ci_contains_one: bool


def _significance_label(p_value: float) -> str:
    if p_value < 0.001:
        return "rất cao (p < 0.001)"
    if p_value < 0.01:
        return "cao (p < 0.01)"
    if p_value < 0.05:
        return "có ý nghĩa (p < 0.05)"
    return "không có ý nghĩa thống kê (p >= 0.05)"


def _strength_label(odds_ratio: float) -> str:
    # Heuristic: khoảng cách tới 1 càng lớn => tác động càng mạnh
    abs_change = abs(odds_ratio - 1.0)
    if abs_change > 1.0:
        return "rất mạnh"
    if abs_change > 0.5:
        return "mạnh"
    if abs_change > 0.2:
        return "vừa phải"
    return "yếu"


def interpret_odds_ratio(
    *,
    feature: str,
    odds_ratio: float,
    ci_lower: float,
    ci_upper: float,
    p_value: float,
) -> OddsRatioInterpretation:
    """Diễn giải OR (Odds Ratio - tỷ số odds) theo hướng + độ lớn.

    Lưu ý khoa học:
    - Với biến liên tục, OR là theo 1 đơn vị tăng.
    - Với biến dummy (ví dụ: Geography_Germany), OR là so với baseline (nhóm tham chiếu).
    """
    if odds_ratio >= 1:
        direction = "tăng"
        magnitude_pct = (odds_ratio - 1.0) * 100.0
    else:
        direction = "giảm"
        magnitude_pct = (1.0 - odds_ratio) * 100.0

    return OddsRatioInterpretation(
        feature=feature,
        odds_ratio=float(odds_ratio),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=float(p_value),
        direction=direction,
        magnitude_pct=float(magnitude_pct),
        significance=_significance_label(float(p_value)),
        strength=_strength_label(float(odds_ratio)),
        ci_contains_one=bool(ci_lower <= 1.0 <= ci_upper),
    )


def _category_for_feature(feature: str) -> str:
    if feature == "const":
        return "Hằng số"

    # Demographics
    if "Age" in feature or "Gender" in feature:
        return "Nhân khẩu học"

    # Geography
    if "Geography" in feature:
        return "Địa lý"

    # Financial
    if any(token in feature for token in ("Balance", "CreditScore", "EstimatedSalary")):
        return "Tài chính"

    # Product/usage behavior
    if any(
        token in feature
        for token in ("IsActiveMember", "HasCrCard", "NumOfProducts", "Tenure")
    ):
        return "Hành vi sử dụng dịch vụ"

    return "Khác"


def _format_feature_context(feature: str) -> str | None:
    # Gợi ý diễn giải cho biến dummy
    if feature.startswith("Geography_"):
        return "So với nhóm baseline (nhóm tham chiếu) của Geography."
    if feature.startswith("Gender_"):
        return "So với nhóm baseline (nhóm tham chiếu) của Gender."
    return None


def generate_odds_ratio_analysis_text(or_df: pd.DataFrame) -> str:
    """Sinh báo cáo text (UTF-8) từ bảng odds ratio."""
    df = or_df.copy()
    required_cols = {"feature", "odds_ratio", "ci_lower", "ci_upper", "p_value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Odds ratio table missing columns: {sorted(missing)}")

    df = df[df["feature"] != "const"].sort_values("p_value", ascending=True)

    report: list[str] = []
    report.append("=" * 80)
    report.append("PHÂN TÍCH CHI TIẾT ODDS RATIO - CÁC YẾU TỐ ẢNH HƯỞNG ĐẾN CHURN")
    report.append("=" * 80)
    report.append("")

    # Group by categories for readability
    df["category"] = df["feature"].map(_category_for_feature)
    category_order = [
        "Nhân khẩu học",
        "Tài chính",
        "Hành vi sử dụng dịch vụ",
        "Địa lý",
        "Khác",
    ]
    for category in category_order:
        chunk = df[df["category"] == category]
        if chunk.empty:
            continue

        report.append("\n" + "=" * 80)
        report.append(category.upper())
        report.append("=" * 80 + "\n")

        for _, row in chunk.iterrows():
            interp = interpret_odds_ratio(
                feature=str(row["feature"]),
                odds_ratio=float(row["odds_ratio"]),
                ci_lower=float(row["ci_lower"]),
                ci_upper=float(row["ci_upper"]),
                p_value=float(row["p_value"]),
            )

            report.append(f"[{interp.feature}]")
            report.append(f"   Odds Ratio: {interp.odds_ratio:.4f}")
            report.append(f"   95% CI: [{interp.ci_lower:.4f}, {interp.ci_upper:.4f}]")
            report.append(f"   p-value: {interp.p_value:.4e}")
            report.append(f"   Ý nghĩa thống kê: {interp.significance}")
            context = _format_feature_context(interp.feature)
            if context is not None:
                report.append(f"   Ghi chú: {context}")
            report.append("")

            if interp.p_value >= 0.05 or interp.ci_contains_one:
                report.append("   [!] KHÔNG CÓ Ý NGHĨA THỐNG KÊ")
                report.append("   -> Không có bằng chứng thống kê rằng biến này ảnh hưởng churn.")
            else:
                report.append(f"   [+] TÁC ĐỘNG: {interp.direction.upper()} odds churn")
                report.append(f"   -> Độ mạnh: {interp.strength}")
                report.append(f"   -> Ước lượng thay đổi: {interp.magnitude_pct:.2f}% (theo 1 đơn vị)")
            report.append("")

    # Summary: top effects among significant variables
    report.append("\n" + "=" * 80)
    report.append("TÓM TẮT: CÁC YẾU TỐ ẢNH HƯỞNG MẠNH NHẤT (p < 0.05, CI không chứa 1)")
    report.append("=" * 80 + "\n")

    sig = df[(df["p_value"] < 0.05) & ~((df["ci_lower"] <= 1.0) & (1.0 <= df["ci_upper"]))].copy()
    if sig.empty:
        report.append("Không có biến nào đạt ý nghĩa thống kê theo tiêu chí hiện tại.")
        report.append("")
        return "\n".join(report)

    protect = sig[sig["odds_ratio"] < 1.0].copy()
    protect["protect_strength"] = 1.0 / protect["odds_ratio"]
    protect = protect.sort_values("protect_strength", ascending=False).head(5)

    risk = sig[sig["odds_ratio"] > 1.0].copy()
    risk = risk.sort_values("odds_ratio", ascending=False).head(5)

    report.append("[+] Yếu tố bảo vệ (giảm churn):")
    if protect.empty:
        report.append("   - (Không có)")
    else:
        for _, row in protect.iterrows():
            report.append(
                f"   - {row['feature']}: OR={row['odds_ratio']:.3f} (95%CI {row['ci_lower']:.3f}–{row['ci_upper']:.3f}, p={row['p_value']:.2e})"
            )

    report.append("")
    report.append("[!] Yếu tố rủi ro (tăng churn):")
    if risk.empty:
        report.append("   - (Không có)")
    else:
        for _, row in risk.iterrows():
            report.append(
                f"   - {row['feature']}: OR={row['odds_ratio']:.3f} (95%CI {row['ci_lower']:.3f}–{row['ci_upper']:.3f}, p={row['p_value']:.2e})"
            )

    report.append("")
    report.append("Khuyến nghị: ưu tiên can thiệp vào các biến vừa có ý nghĩa thống kê vừa có hiệu ứng lớn (|OR-1| lớn).")
    report.append("")
    return "\n".join(report)


def write_odds_ratio_analysis(or_table_path: Path, output_path: Path) -> str:
    df = pd.read_csv(or_table_path)
    text = generate_odds_ratio_analysis_text(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return text


def plot_odds_ratio_forest(or_table_path: Path, output_path: Path) -> None:
    """Vẽ forest plot cho Odds Ratio (chuẩn y khoa/dịch tễ học)."""
    df = pd.read_csv(or_table_path)
    df = df[df["feature"] != "const"].sort_values("p_value", ascending=True).copy()
    if df.empty:
        raise ValueError("Odds ratio table has no rows after filtering 'const'.")

    df["significant"] = (df["p_value"] < 0.05) & ~(
        (df["ci_lower"] <= 1.0) & (1.0 <= df["ci_upper"])
    )

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    y_pos = list(range(len(df)))
    colors = ["#2ecc71" if bool(sig) else "#95a5a6" for sig in df["significant"]]

    # CI bars
    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot(
            [float(row["ci_lower"]), float(row["ci_upper"])],
            [i, i],
            color=colors[i],
            linewidth=2,
            alpha=0.8,
        )

    # OR points
    ax.scatter(
        df["odds_ratio"].astype(float),
        y_pos,
        s=90,
        c=colors,
        edgecolors="black",
        linewidths=1.2,
        zorder=3,
        alpha=0.95,
    )

    ax.axvline(
        x=1.0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="OR = 1 (No effect)",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"].astype(str))
    ax.set_xlabel("Odds Ratio (95% CI)")
    ax.set_title("Forest Plot: Odds Ratio các yếu tố ảnh hưởng churn\n(Xanh = p<0.05, Xám = không ý nghĩa)")
    ax.set_xscale("log")
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


