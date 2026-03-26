# HR Employee Statistical Analysis

**Repository**: `hr-employee-statistical-analysis`

Nghiên cứu các yếu tố ảnh hưởng đến quyết định nghỉ việc của nhân viên (Employee Attrition) bằng mô hình Hồi quy Logistic (Logistic Regression).

**Dataset**: `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`  
Nguồn: [Kaggle - IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

## Tổng quan

Project này sử dụng:
- **Logistic Regression** (sklearn) cho dự đoán khả năng nghỉ việc
- **Logit model** (statsmodels) cho suy luận thống kê và Odds Ratio
- **LaTeX** cho báo cáo và presentation slides (Beamer)

## Cấu trúc thư mục

- **`data/`**: dữ liệu thô (raw data) — giữ nguyên.
- **`figures/`**: nơi lưu toàn bộ hình ảnh xuất ra (plots/figures) từ EDA, mô hình, đánh giá.
- **`src/`**: mã nguồn theo dạng package (module hoá) để tái sử dụng và maintain tốt.
- **`scripts/`**: entry-point scripts chạy pipeline.
- **`outputs/`**: artifacts (bảng kết quả .csv, model dumps, metrics JSON, v.v.).
- **`report/`**: 
  - Báo cáo LaTeX chi tiết (`main.tex`, `main.pdf`)
  - Presentation slides Beamer (`presentation.tex`, `presentation.pdf`)

## Quy ước xuất kết quả

- Hình ảnh: lưu vào `figures/` (ví dụ: `figures/roc_curve.png`).
- Bảng kết quả / metrics: lưu vào `outputs/`.
- Báo cáo LaTeX: trong `report/` (PDF được giữ lại trong repo).

## Yêu cầu môi trường

- Python 3.10+ (khuyến nghị)
- LaTeX distribution (MiKTeX hoặc TeX Live) để compile báo cáo và presentation

## Cài dependencies

### Option A: Conda (khuyến nghị nếu bạn đang dùng `data_science`)

```bash
conda activate data_science
pip install -r requirements.txt
pip install -e .
```

Nếu bạn không muốn activate trực tiếp, có thể dùng `conda run`:

```bash
conda run -n data_science pip install -r requirements.txt
conda run -n data_science pip install -e .
```

### Option B: venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Chạy pipeline (end-to-end)

```bash
python scripts/run_pipeline.py
```

Nếu bạn **chưa** cài `pip install -e .`, có thể chạy nhanh bằng `PYTHONPATH` (đặt đường dẫn import tạm thời):

```bash
export PYTHONPATH="src"
python scripts/run_pipeline.py
```

## Compile LaTeX Reports

### Báo cáo chính (main report)

```bash
cd report
pdflatex main.tex
# Hoặc dùng latexmk để tự động compile nhiều lần
latexmk -pdf main.tex
```

### Presentation slides

```bash
cd report
pdflatex presentation.tex
# Hoặc
latexmk -pdf presentation.tex
```

## Artifacts đầu ra

### Python outputs

- **`outputs/eda_summary.json`**: tóm tắt EDA.
- **`outputs/logistic_metrics.json`**: metrics mô hình logistic.
- **`outputs/logistic_pipeline.joblib`**: pipeline sklearn đã train (binary artifact - file nhị phân, có thể tái tạo).
- **`outputs/odds_ratio_table.csv`**, **`outputs/logit_summary.txt`**: kết quả statsmodels Logit + Odds Ratio.
- **`outputs/odds_ratio_analysis.txt`**: báo cáo diễn giải OR (data-driven).
- **`figures/`**: ROC, confusion matrix, EDA distributions, forest plot OR.

### LaTeX outputs

- **`report/main.pdf`**: Báo cáo LaTeX chính
- **`report/presentation.pdf`**: Presentation slides (Beamer)

Ghi chú: repo đã cấu hình `.gitignore` để **không commit** `outputs/*.joblib` (model binary) và các file build LaTeX (`.aux`, `.log`, v.v.) nhằm giữ repo gọn, vì có thể tái tạo bằng `python scripts/run_pipeline.py` và compile LaTeX.

## Sinh báo cáo riêng (không chạy lại train)

```bash
python scripts/generate_project_report.py
```
