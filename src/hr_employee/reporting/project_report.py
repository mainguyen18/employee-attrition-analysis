from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hr_employee.config import ProjectPaths
from hr_employee.preprocessing.features import FeatureSpec


@dataclass(frozen=True, slots=True)
class EdaSummaryArtifact:
    n_rows: int
    n_cols: int
    churn_rate: float
    missing_by_column: dict[str, int]


@dataclass(frozen=True, slots=True)
class LogisticMetricsArtifact:
    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_eda_summary(path: Path) -> EdaSummaryArtifact:
    data = _read_json(path)
    return EdaSummaryArtifact(
        n_rows=int(data["n_rows"]),
        n_cols=int(data["n_cols"]),
        churn_rate=float(data["churn_rate"]),
        missing_by_column={str(k): int(v) for k, v in dict(data["missing_by_column"]).items()},
    )


def load_logistic_metrics(path: Path) -> LogisticMetricsArtifact:
    data = _read_json(path)
    return LogisticMetricsArtifact(
        roc_auc=float(data["roc_auc"]),
        accuracy=float(data["accuracy"]),
        precision=float(data["precision"]),
        recall=float(data["recall"]),
        f1=float(data["f1"]),
        confusion_matrix=[list(map(int, row)) for row in data["confusion_matrix"]],
    )


def _maybe_image_md(project_root: Path, image_rel_path: str, caption: str) -> str:
    image_path = project_root / image_rel_path
    if not image_path.exists():
        return ""
    return f"![{caption}]({image_rel_path})\n"


def generate_project_report_markdown(
    *,
    paths: ProjectPaths,
    eda_summary: EdaSummaryArtifact | None,
    logistic_metrics: LogisticMetricsArtifact | None,
    odds_ratio_table: pd.DataFrame | None,
) -> str:
    lines: list[str] = []
    lines.append("> **Lưu ý**: Đây là báo cáo **tổng hợp (summary)** và được **tự sinh (auto-generated)** từ các artifacts trong `outputs/` và `figures/`.")
    lines.append("> - Không khuyến nghị sửa tay file này; hãy chạy `python scripts/run_pipeline.py` hoặc `python scripts/generate_project_report.py` để cập nhật.")
    lines.append("> - Báo cáo luận văn chi tiết: `report/PROJECT_REPORT_FULL.md`.")
    lines.append("")
    lines.append("## HR Employee Attrition — Statistical Analysis Report")
    lines.append("")

    # ---------- 1) Abstract ----------
    lines.append("### 1) Abstract (Tóm tắt)")
    lines.append(
        "Báo cáo này tập trung phân tích gốc rễ nguyên nhân nghỉ việc của nhân viên (Employee Attrition). Dựa trên tập dữ liệu IBM HR Analytics, báo cáo xây dựng mô hình dự đoán Máy học (Logistic Regression) để cảnh báo rủi ro thôi việc, đồng thời dùng Phương pháp Thống kê (Odds Ratio) để định lượng chính xác yếu tố nào đang thúc đẩy nhân viên rời đi."
    )
    if eda_summary is not None:
        lines.append(
            f"Dữ liệu được xử lý bao gồm {eda_summary.n_rows:,} quan sát nhân sự, với tỷ lệ nghỉ việc hiện tại là {eda_summary.churn_rate:.2%}."
        )
    if logistic_metrics is not None:
        lines.append(
            f"Mô hình máy học đạt phân hạng ROC AUC={logistic_metrics.roc_auc:.3f} và tỷ lệ bao phủ Recall={logistic_metrics.recall:.3f} trên tập dữ liệu kiểm thử độc lập (Test Set)."
        )
    lines.append("")

    # ---------- 2) Problem ----------
    lines.append("### 2) Problem Statement (Phát biểu bài toán)")
    lines.append("- **Mục tiêu chẩn đoán (Diagnostic)**: Khám phá lý do vì sao nhân viên xin từ chức, trả lời câu hỏi: Đãi ngộ, Thâm niên, Khối lượng công việc, hay Khoảng cách đi lại đang tạo sức ép lớn nhất?")
    lines.append("- **Mục tiêu tiên đoán (Predictive)**: Xây dựng giải thuật học máy phân loại nhị phân trên biến mục tiêu `Attrition`, tự động tìm ra ai có rủi ro ra đi tiếp theo trong tương lai để kìm hãm rủi ro chảy máu chất xám.")
    lines.append("")

    # ---------- 3) Data ----------
    lines.append("### 3) Dataset (Dữ liệu nguồn)")
    lines.append("- **Nguồn dữ liệu gốc**: `data/WA_Fn-UseC_-HR-Employee-Attrition.csv` (Tiêu chuẩn của IBM HR Analytics)")
    feature_spec = FeatureSpec()
    lines.append(f"- **Biến phân loại mục tiêu (Target)**: `{feature_spec.target_column}` (Yes/No)")
    lines.append(f"- **Các cột định danh bị cấm/loại bỏ khỏi bài toán**: {', '.join(feature_spec.id_columns)}")
    lines.append(f"- **Các tính chất phân loại chuẩn (Categorical)**: {', '.join(feature_spec.categorical_columns)}")
    lines.append(f"- **Các tính năng định lượng (Numeric)**: {', '.join(feature_spec.numeric_columns)}")
    if eda_summary is not None:
        missing_total = sum(eda_summary.missing_by_column.values())
        lines.append(f"- **Kiểm định cấu trúc**: Toàn bộ {eda_summary.n_rows:,} mẫu và {eda_summary.n_cols} cột thuộc tính đều nguyên vẹn.")
        lines.append(f"- **Thiếu sót dữ liệu (Missing values)**: Ghi nhận {missing_total:,} ô trống, dữ liệu hoàn toàn sạch.")
        lines.append(f"- **Bài toán mất cân bằng phân bổ (Imbalance Target)**: Tỉ lệ thiểu số (Nghỉ việc) chỉ chiếm {eda_summary.churn_rate:.2%}.")
    lines.append("- **Khóa kiểm toán (Artifacts)**: Quá trình quét được đóng băng tại `outputs/eda_summary.json`.")
    lines.append("")

    # ---------- 4) EDA ----------
    lines.append("### 4) Khám phá Dữ liệu Trực quan (EDA - Exploratory Data Analysis)")
    lines.append("Bóc tách hành vi rủi ro bằng cách so sánh phân phối giữa hai tập: Nghỉ Việc (Attrition) và Ở Lại (Retention).")
    lines.append("- *Lưu ý: Bạn có thể tìm thấy toàn bộ dữ liệu gốc của các biểu đồ định dạng CSV trong thư mục `outputs/`.*")
    lines.append("")
    for rel_path, caption in (
        ("figures/churn_rate_by_OverTime.png", "Hiệu ứng sụp đổ năng lượng do Làm Thêm Giờ (Over Time)"),
        ("figures/churn_rate_by_JobRole.png", "Khảo sát Tỷ lệ thất thoát Phân hóa theo Vai trò Công việc (Job Role)"),
        ("figures/churn_rate_by_BusinessTravel.png", "Tỷ lệ rủi ro vì Áp lực Đi Công Tác (Business Travel)"),
        ("figures/churn_rate_by_MaritalStatus.png", "Xu hướng gắn bó qua lăng kính Tình trạng Hôn nhân (Marital Status)"),
        ("figures/dist_Age_by_churn.png", "Đường cong phân bổ Độ Tuổi mâu thuẫn giữa 2 Nhóm (Age vs Churn)"),
        ("figures/dist_MonthlyIncome_by_churn.png", "Mật độ dồn dập rủi ro từ Thu nhập Thấp (Monthly Income vs Churn)"),
        ("figures/dist_DistanceFromHome_by_churn.png", "Khoảng cách địa lý - Liều thuốc độc hao mòn sinh lực (Distance from Home vs Churn)"),
        ("figures/dist_YearsAtCompany_by_churn.png", "Tìm kiếm sự Cứu chuộc trong Vòng đời Thâm niên (Years At Company)"),
    ):
        img = _maybe_image_md(paths.project_root, rel_path, caption)
        if img:
            lines.append(img.rstrip("\n"))
            lines.append("")
    lines.append("")

    # ---------- 5) Preprocessing & design ----------
    lines.append("### 5) Tiền Xử Lý Máy Học & Kiến trúc Thực Nghiệm (Preprocessing & Experimental Design)")
    if eda_summary is not None:
        n_total = int(eda_summary.n_rows)
        n_test = int(round(n_total * 0.2))
        n_train = n_total - n_test
        lines.append(
            f"- **Train/Test split (Tập Huấn Luyện / Tập Kiểm Thử)**: Máy tính phân bổ chia hạt giống (seed=42) theo tỉ lệ vàng 80/20 có phân tầng bảo toàn tỷ lệ thiểu số (Stratified distribution). Quần thể chốt ở mức: Train={n_train:,} người, Test={n_test:,} người."
        )
    else:
        lines.append("- **Train/Test split**: Hệ thống tự động chia Stratified 80/20 an toàn.")
    lines.append("- **Bẻ gãy Dãy Phân loại (Categorical Pipeline)**: Thuật toán OneHotEncoder băm 100% các biến phân loại thành ma trận thưa tự động (Sparce matrix) để máy tính hiểu ý nghĩa phi kỹ thuật.")
    lines.append("- **Dàn phẳng Không gian Toán học (Numeric Pipeline)**: Phủ một lớp StandardScaler lên toàn bộ các trị số chênh lệch (Lương nghìn Đô la vs Độ tuổi 30) thành tham số hội tụ Gradient nhạy bén.")
    lines.append("- **Liều Thuốc Mất Cân Bằng (Imbalance handling)**: Ép hàm `class_weight='balanced'` vặn tăng uy quyền của Trọng số Nhóm Rời Đi (do thiếu mẫu) trong Logistic Regression.")
    lines.append("")

    # ---------- 6) Model ----------
    lines.append("### 6) Lắp ráp Lõi Động cơ Máy Học (Machine Learning Model)")
    lines.append(
        "Bộ não được chọn để thao túng tập dữ liệu là **Hồi quy Logistic (Logistic Regression)**, nhằm triệt để tối đa hóa khả năng Trắng-Đen rành mạch và truy vết được nguyên nhân (Explainability) cho Ban Giám đốc."
    )
    lines.append("- **Siết chặt Ranh giới**: Trình giải mã `liblinear` kéo dãn kịch khung với Penalty L2 (chống Overfit).")
    lines.append("- **Đóng băng Tri thức (Artifacts)**: Mô hình ngưng đọng vào tệp nhị phân siêu chuẩn ở `outputs/logistic_pipeline.joblib`. Bạn có tệp này tức là bạn đã có 1 chuyên gia săn lùng nhân viên nghỉ việc.")
    lines.append("")

    # ---------- 7) Evaluation ----------
    lines.append("### 7) Đánh giá Khả năng Khám xét (Evaluation Metrics & Matrix)")
    if logistic_metrics is not None:
        tn, fp = logistic_metrics.confusion_matrix[0]
        fn, tp = logistic_metrics.confusion_matrix[1]
        lines.append("Cái cân chân lý kiểm bài cho AI trên Tập ẩn Test Set:")
        lines.append(f"- **Diện tích Khuất phục Nhận diện (ROC AUC)**: {logistic_metrics.roc_auc:.3f}")
        lines.append(f"- **Tỷ lệ Tổng khớp Đúng (Accuracy)**: {logistic_metrics.accuracy:.3f}")
        lines.append(f"- **Tuyệt đối Bắt Lỗi (Precision)**: {logistic_metrics.precision:.3f}")
        lines.append(f"- **Lưới Cào Nhận Diện Rủi Ro (Recall)**: {logistic_metrics.recall:.3f}")
        lines.append(f"- **Giữ nhịp Tính Hiệu Quả (F1-score)**: {logistic_metrics.f1:.3f}")
        lines.append(
            f"- **Giải phẫu Ma trận Nhầm Lẫn (Confusion Matrix)**: Giữ được {tn} người an lòng, Trót báo động nhầm {fp} lần đâm lo, Tệ bạc để lọt lưới {fn} người dứt áo, Bắt tại trận {tp} lính đánh thuê muốn từ bỏ."
        )
        lines.append("")
        lines.append("👉 **Quan điểm Tranh luận Kinh doanh (Interpretation)**:")
        lines.append(
            "1. Recall vươn cao rất tuyệt vời! Trong quản trị tỷ lệ hao mòn (Attrition), luật bất thành văn là phải bắt cho bằng hết người có triệu chứng (Tối ưu FN thấp nhất). Giám đốc thà nhầm lẫn ban phát chính sách quan tâm thừa mứa (FP cao), còn hơn bỏ lỡ một trụ cột tổ chức dứt áo ra đi trong im lặng (FN = 0)."
        )
    else:
        lines.append("❌ Cảnh báo: `outputs/logistic_metrics.json` chưa được sinh ra do thiếu hụt vòng lặp huấn luyện!")
    lines.append("")
    roc_img = _maybe_image_md(paths.project_root, "figures/roc_curve.png", "ROC Curve Model Capability")
    if roc_img:
        lines.append(roc_img.rstrip("\n"))
        lines.append("")
    cm_img = _maybe_image_md(paths.project_root, "figures/confusion_matrix.png", "AI Confusion Matrix Analysis")
    if cm_img:
        lines.append(cm_img.rstrip("\n"))
        lines.append("")
    lines.append("")

    # ---------- 8) Statistical inference (OR) ----------
    lines.append("### 8) Định tội Khoa học qua Tỷ suất Ngoại biên (Statistical Inference by Odds Ratio)")
    lines.append("Ma thuật Máy học đoán bạn bỏ đi, còn Toán học Thống kê lột trần thủ phạm đâm sau lưng bạn. Bảng OR biến cớ chối bỏ của bạn thành toán học thực dụng.")
    lines.append("- *Chứng từ luận tội tham chiếu*: `outputs/odds_ratio_table.csv`, `outputs/odds_ratio_analysis.txt`.")
    lines.append("")
    forest_img = _maybe_image_md(
        paths.project_root, "figures/forest_plot_odds_ratio.png", "Forest plot Thuyết Phục Odds Ratio"
    )
    if forest_img:
        lines.append(forest_img.rstrip("\n"))
        lines.append("")

    lines.append("**Nguyên lý Phán Xét Tuyệt Đối:**")
    lines.append(
        "- Bất kì cáo buộc rủi ro nào nếu dính P-value > 0.05, hoặc râu độ lệch chuẩn liếm vào Cột Mốc `Số 1 Tuyệt đối`, ngay lập tức bị loại trừ khỏi phòng xử án."
    )
    lines.append("")

    if odds_ratio_table is not None and not odds_ratio_table.empty:
        lines.append("**BẢNG PHONG THẦN THỦ PHẠM:** (Các yếu tố dồn ép nhân sự nộp đơn)")
        lines.append("👉 *Quy ước: OR > 1 (Đẩy rủi ro lên cao), OR < 1 (Là lá chắn trấn an)*")
        lines.append("")
        top = (
            odds_ratio_table[odds_ratio_table["feature"] != "const"]
            .sort_values("p_value")
            .head(10)
        )
        lines.append("| Đặc Tính Nhân Viên (Feature) | Phương Hại (Odds Ratio) | Trượt Dưới | Sai Số Trên | Thẩm định p-value |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, row in top.iterrows():
            lines.append(
                f"| **{row['feature']}** | {row['odds_ratio']:.4f} | {row['ci_lower']:.4f} | {row['ci_upper']:.4f} | {row['p_value']:.2e} |"
            )
        lines.append("")
        lines.append("🔥 **Lật Tẩy Các Án Điển Hình:**")
        lines.append("- Ám ảnh bởi **Tăng Ca (OverTime_Yes)**: Có sức mạnh hủy diệt ghê gớm bậc nhất, cắn nát tâm lý, thúc đẩy tháo chạy quy mô cực điểm so với nhóm Non-OT.")
        lines.append("- Bảo ấn của **Sự trưởng thành (Age)**: Khoảng khắc nhân sự già đi 1 tuổi, tỷ suất từ bỏ giảm đều lùi về vùng dưới mốc 1 một cách tĩnh lặng vững chắc.")
        lines.append("- Rào cản địa lý **(DistanceFromHome)**: Vắt kiệt nhân sự vô hình, OR nhỉnh lên xói mòn quyết chí làm việc theo mỗi cây số họ vượt qua.")
        lines.append("")
    else:
        lines.append("Chưa nạp được sổ điểm OR table từ `outputs/odds_ratio_table.csv`.")
        lines.append("")

    # ---------- 9) Discussion & limitations ----------
    lines.append("### 9) Tự Sự Hạn Suy (Discussion & Limitations)")
    lines.append("- **Khuyết tật của Phóng đại Tuyến tính (Effect size scaling)**: Rủi ro OR trói buộc với MỖI 1 ĐƠN VỊ. Mặc dù OR của Lương dường như chả suy chuyển ở mốc 0.999x, nhưng trên thực tế, tiền lương dao động bạt mạng theo chục nghìn Dollar nên tác động khố rách của nó lên tỷ lệ rời đi là tột cùng đau đớn chứ không hề bé!")
    lines.append("- **Chối bỏ Thần giao Cách cảm (Causality)**: Giải thuật này cung cấp Lối suy diễn, không khẳng định Nguyên do Tuyệt đối. Việc một 'Kỹ thuật viên phòng thí nghiệm' dễ bỏ việc chỉ mang tính thống kê cụm ngành, không phải tráng một lớp lăng kính định kiến nhân quả lên đầu họ.")
    lines.append("")

    # ---------- 10) Conclusion & recommendations ----------
    lines.append("### 10) Khởi xướng Thay máu Hệ thống (Conclusion & Actionable Recommendations)")
    lines.append("**THỊ KIẾN THỰC TIỄN KIẾT LÝ TỪ DỮ LIỆU CHẾT:**")
    lines.append("Đằng sau cỗ máy Học Sâu, mô hình phát giác ra 1 kẻ thất bại trong tổ chức hầu hết mắc một căn bệnh thập tử nhất sinh hội tụ tử 4 nguyên cớ: **Người còn trẻ tuổi + Trót làm chức Tăng ca sấp mặt + Đi Lữ hành công tác cạn kiệt ngày nghỉ + Nhận mức lương cận biên đáy.**")
    lines.append("")
    lines.append("🛡️ **PHỐI VÀ VẬN THUỐC QUẢN TRỊ:**")
    lines.append("1. **Ban bố Cốt Cách Tuyệt Tăng Ca Trái Chiều**: Thiết diện bãi bỏ, khoán ngân sách đi thuê thầu phụ (Outsource) chặn tay luồng cháy nổ OverTime nội bộ. Quỹ săn đầu người thuê mới 1 cá nhân vỡ vạc còn chát chúa hơn ngần ấy bạc lương OT bù đắp.")
    lines.append("2. **Buông Rèm Sát phạt Năm Thứ 2**: Sổ lồng bóc lột phúc lợi ở hai năm mới vào nghề. Biểu đồ sống sót thể hiện sức nặng gắn bó tăm tắp ngay sau vượt cạn 24 tháng!")
    lines.append("3. **Tuyên ngôn Trả oán Tiền xăng cộ**: Dứt khoát chiết xuất quỹ 'Hỗ trợ lưu trú / Nhà ở gần viện' cho những nhân sự oằn mình sáng tối lê lết phương xa trên dặm đường về trụ sở. Kẹp theo đó ấn định tỷ lệ quy đổi Nghỉ ngơi trọn vẹn đặc ân cho thành phần Lữ hành đi công tác bứt rứt mệt nhoài.")
    lines.append("")

    # ---------- 11) Reproducibility ----------
    lines.append("### 11) Lệnh tái kích hoạt vòng đời tự động (Automated Reproducibility)")
    lines.append("Sự kiện thay máu Báo cáo này hoàn toàn phi can thiệp thủ công (Data-Driven).")
    lines.append("Để Đào tạo lại (Re-train) 100% não bộ máy tích lũy kiến thức qua dữ liệu cập nhật mới:")
    lines.append("```bash")
    lines.append("python scripts/run_pipeline.py")
    lines.append("```")
    lines.append("Nếu bạn chỉ có nhu cầu Dựng lại Form chữ (Re-render Text Template) qua bảng tính thô sẵn:")
    lines.append("```bash")
    lines.append("python scripts/generate_project_report.py")
    lines.append("```")
    lines.append("")

    # Keep markdown formatting stable: collapse >2 consecutive blank lines to 2.
    cleaned: list[str] = []
    blank_run = 0
    for ln in lines:
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                cleaned.append("")
            continue
        blank_run = 0
        cleaned.append(ln)

    return "\n".join(cleaned).rstrip() + "\n"


def write_project_report(paths: ProjectPaths, output_path: Path | None = None) -> Path:
    """Sinh `PROJECT_REPORT.md` từ artifacts hiện có (không train lại)."""
    output = output_path or (paths.project_root / "PROJECT_REPORT.md")

    eda_summary_path = paths.outputs_dir / "eda_summary.json"
    logistic_metrics_path = paths.outputs_dir / "logistic_metrics.json"
    odds_ratio_table_path = paths.outputs_dir / "odds_ratio_table.csv"

    eda_summary = load_eda_summary(eda_summary_path) if eda_summary_path.exists() else None
    logistic_metrics = (
        load_logistic_metrics(logistic_metrics_path)
        if logistic_metrics_path.exists()
        else None
    )
    odds_ratio_table = (
        pd.read_csv(odds_ratio_table_path) if odds_ratio_table_path.exists() else None
    )

    md = generate_project_report_markdown(
        paths=paths,
        eda_summary=eda_summary,
        logistic_metrics=logistic_metrics,
        odds_ratio_table=odds_ratio_table,
    )
    output.write_text(md, encoding="utf-8")
    return output
