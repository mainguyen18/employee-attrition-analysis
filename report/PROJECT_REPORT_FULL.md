# HR Employee Attrition Analysis — Báo cáo Markdown chi tiết (Full Project Report)

**Repository**: `data-analysis`

## Vai trò của các báo cáo (Report structure)

- `PROJECT_REPORT.md`: **báo cáo tổng hợp (summary)**, tự sinh từ artifacts sau khi chạy pipeline.
- `report/PROJECT_REPORT_FULL.md` (file này): **báo cáo luận văn chi tiết**, phục vụ bảo vệ và biện luận học thuật.
- `report/main.pdf`: **Báo cáo LaTeX chính** (báo cáo đầy đủ dạng PDF)
- `report/presentation.pdf`: **Presentation slides** (Beamer) cho thuyết trình

## 1) Tóm tắt đề tài (Project summary)

- **Đề tài**: Nghiên cứu các yếu tố ảnh hưởng đến quyết định rời bỏ của nhân viên (attrition) bằng Logistic Regression.
- **Dataset**: `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`
- **Đầu ra chính (artifacts - sản phẩm đầu ra)**:
  
  **Python outputs:**
  - Metrics mô hình: `outputs/logistic_metrics.json`
  - Model pipeline: `outputs/logistic_pipeline.joblib` (binary artifact - file nhị phân, có thể tái tạo; thường không commit theo `.gitignore`)
  - Odds Ratio + CI + p-value: `outputs/odds_ratio_table.csv`
  - Statsmodels summary: `outputs/logit_summary.txt`
  - Báo cáo diễn giải OR: `outputs/odds_ratio_analysis.txt`
  - Báo cáo tổng hợp ngắn: `PROJECT_REPORT.md`
  - Hình minh hoạ: `figures/*.png`
  
  **LaTeX outputs:**
  - Báo cáo LaTeX chính: `report/main.pdf`
  - Presentation slides: `report/presentation.pdf`

## 2) Cấu trúc dự án (Project structure)

Thư mục chính:

```text
data-analysis/
  data/                       # dữ liệu gốc
  outputs/                    # bảng kết quả + metrics + model artifacts
  figures/                    # biểu đồ (EDA, ROC, confusion matrix, forest plot OR)
  scripts/                    # entry-point scripts chạy pipeline
  src/hr_employee/             # package: logic phân tích/model/reporting
  report/                     # báo cáo và presentation
    main.tex                  # Báo cáo LaTeX chính
    main.pdf                  # Báo cáo PDF
    presentation.tex          # Presentation slides (Beamer)
    presentation.pdf          # Presentation PDF
    sections/                 # Các section của báo cáo LaTeX
    slides/                   # Các slide của presentation
    PROJECT_REPORT_FULL.md    # (file này) báo cáo markdown chi tiết
    assets/                   # Hình ảnh và bảng cho LaTeX
```

Các module quan trọng trong `src/hr_employee/`:

- **FeatureSpec** (đặc tả biến): `src/hr_employee/preprocessing/features.py` (L6–L22)
- **Preprocessing** (tiền xử lý): `src/hr_employee/preprocessing/pipeline.py` (L20–L37)
- **Train/Test split + Pipeline**: `src/hr_employee/training/pipeline.py` (L22–L35)
- **Logistic model** (sklearn): `src/hr_employee/model/logistic.py` (L8–L31)
- **Evaluation metrics**: `src/hr_employee/evaluation/metrics.py` (L16–L37)
- **Statsmodels Logit + Odds Ratio**: `src/hr_employee/stats/logit_odds_ratio.py` (L21–L64)
- **Orchestration pipeline**: `scripts/run_pipeline.py` (L53–L203)

## 3) Quy trình làm việc (Workflow end-to-end)

### 3.1 Pipeline tổng (1 lệnh chạy)

Chạy end-to-end được điều phối bởi `scripts/run_pipeline.py`:

- **Bước 1: Load data**: `scripts/run_pipeline.py` (L65–L68)
- **Bước 2: EDA**: `scripts/run_pipeline.py` (L70–L95)
- **Bước 3: Train + Evaluate (sklearn LogisticRegression)**: `scripts/run_pipeline.py` (L97–L165)
- **Bước 4: Fit statsmodels Logit + xuất Odds Ratio table**: `scripts/run_pipeline.py` (L167–L174)
- **Bước 5: Sinh báo cáo OR analysis + PROJECT_REPORT.md**: `scripts/run_pipeline.py` (L176–L187)

### 3.2 Data → Preprocess

**Mục tiêu**: tách $X$ (features - biến độc lập) và $y$ (target - biến mục tiêu), đồng thời xây dựng preprocessor cho categorical/numeric.

- Tách $X, y$: `src/hr_employee/preprocessing/pipeline.py` (hàm `split_xy`, L40–L57)
  - Loại bỏ `id_columns` và `target_column`.
  - Kiểm tra thiếu cột bắt buộc trước khi train (L42–L53).
- Đặc tả biến: `src/hr_employee/preprocessing/features.py` (L6–L22)
  - `target_column="Attrition"` (L11)
  - categorical: `("Department", "Gender", "JobRole")` (L12)
  - numeric: 8 biến (L13–L22)
- Preprocessor:
  - OneHotEncoder cho categorical và StandardScaler cho numeric:
    `src/hr_employee/preprocessing/pipeline.py` (hàm `build_preprocessor`, L20–L37)

### 3.3 Train → Evaluate

- Split train/test có phân tầng (stratified split - chia có phân tầng):
  `src/hr_employee/training/pipeline.py` (hàm `stratified_split`, L22–L26)
- Xây pipeline sklearn:
  `src/hr_employee/training/pipeline.py` (hàm `build_sklearn_pipeline`, L29–L35)
  - Step `"preprocess"`: ColumnTransformer (từ `build_preprocessor`)
  - Step `"model"`: LogisticRegression (từ `build_logistic_model`)
- Logistic model spec + instantiate:
  `src/hr_employee/model/logistic.py` (L8–L31)
  - `solver="liblinear"`, `penalty="l2"`, `C=1.0`, `max_iter=2000` (L13–L17)
  - `class_weight="balanced"` được truyền từ pipeline script (ví dụ `scripts/run_pipeline.py` L107–L111)
- Evaluation metrics:
  `src/hr_employee/evaluation/metrics.py` (hàm `compute_metrics`, L28–L37)
  - Dùng threshold mặc định 0.5 để suy ra $\hat{y}$ (L28–L30)
  - Tính ROC AUC, accuracy, precision, recall, f1, confusion matrix (L31–L37)

### 3.4 Interpret → Odds Ratio (statsmodels)

Fit mô hình Logit (statsmodels) và xuất Odds Ratio:

- Chuẩn bị design matrix (ma trận thiết kế): `src/hr_employee/stats/logit_odds_ratio.py` (L21–L35)
  - `pd.get_dummies(..., drop_first=True)` (L29–L31) tạo dummy variables; baseline là nhóm bị drop.
  - `sm.add_constant(...)` (L34) thêm hệ số chặn $\beta_0$.
- Fit Logit và chuyển hệ số sang Odds Ratio:
  `src/hr_employee/stats/logit_odds_ratio.py` (L38–L64)
  - OR = `np.exp(params.values)` (L54–L60)
  - CI của OR = exp(CI của beta) (L58–L59)

## 4) Công thức Logistic Regression (chi tiết)

### 4.1 Mô hình xác suất (Probability model)

Với $x \in \mathbb{R}^d$, xác suất churn:

$$
p(y=1 \mid x) = \sigma(z) = \frac{1}{1 + e^{-z}},\quad z = \beta_0 + \beta^T x
$$

Tương đương dạng logit (log-odds):

$$
\log\left(\frac{p(y=1\mid x)}{1 - p(y=1\mid x)}\right) = \beta_0 + \beta^T x
$$

**Triển khai trong code**:

- Mô hình LogisticRegression của sklearn được tạo tại `src/hr_employee/model/logistic.py` (L22–L31).
- Pipeline ghép preprocess + model: `src/hr_employee/training/pipeline.py` (L29–L35).

### 4.2 Hàm mục tiêu (Objective / Loss)

Logistic Regression thường tối ưu Negative Log-Likelihood (NLL - âm log-likelihood):

$$
\mathcal{L}(\beta) = -\sum_{i=1}^{n}\Big[y_i\log p_i + (1-y_i)\log(1-p_i)\Big]
$$

Với regularization L2 (phạt bình phương):

$$
\mathcal{L}_{L2}(\beta) = \mathcal{L}(\beta) + \lambda \|\beta\|_2^2
$$

Trong sklearn, cấu hình `penalty="l2"` (tương ứng L2) nằm ở:
`src/hr_employee/model/logistic.py` (L13–L17, L24–L31).

**Ghi chú quan trọng về tham số regularization (`C` vs \(\lambda\))**:

- sklearn dùng tham số `C` (inverse regularization strength - nghịch đảo độ mạnh regularization), tức là **`C` lớn → regularization yếu**, `C` nhỏ → regularization mạnh.
- Vì vậy, ký hiệu \(\lambda\) trong công thức trên là cách viết chuẩn sách vở; khi đối chiếu với sklearn thì cần hiểu \(\lambda\) tỉ lệ nghịch với `C` (và có thể khác hệ số hằng tuỳ solver/implementation).
- Trong dự án này: `C=1.0` được đặt tại `src/hr_employee/model/logistic.py` (L15–L16, L22–L31).

### 4.3 Xử lý mất cân bằng lớp (class imbalance)

Trong churn, lớp $y=1$ thường ít hơn (tỷ lệ churn ~ 20%). Ở đây dùng:

- `class_weight="balanced"` trong `scripts/run_pipeline.py` (L107–L111)

Ý nghĩa: trọng số lớp được điều chỉnh để loss “phạt” lỗi của lớp minority mạnh hơn.

### 4.4 Kiểm định giả thuyết (Hypothesis Testing - kiểm định giả thuyết)

Trong báo cáo thống kê, khi ta báo cáo p-value cho từng hệ số $\beta_j$ (tương ứng biến $x_j$), ta đang kiểm định:

**Giả thuyết không (Null hypothesis - $H_0$):**

$$
H_0: \beta_j = 0 \quad \Leftrightarrow \quad OR_j = e^{\beta_j} = e^0 = 1
$$

Nghĩa là biến $x_j$ **không có ảnh hưởng** đến log-odds churn (và odds churn) trong điều kiện các biến khác giữ nguyên (ceteris paribus - các yếu tố khác không đổi).

**Giả thuyết đối (Alternative hypothesis - $H_1$):**

$$
H_1: \beta_j \neq 0 \quad \Leftrightarrow \quad OR_j \neq 1
$$

Nghĩa là biến $x_j$ **có ảnh hưởng** đến churn.

**Quy tắc quyết định (Decision rule - quy tắc ra quyết định):**

- Chọn mức ý nghĩa $\alpha = 0.05$ (thường dùng trong nghiên cứu).
- Nếu $p\text{-value} < \alpha$ thì **bác bỏ $H_0$** ⇒ biến có ý nghĩa thống kê (statistically significant - có ý nghĩa thống kê).
- Nếu $p\text{-value} \ge \alpha$ thì **không đủ bằng chứng** bác bỏ $H_0$.

**Liên hệ với code/artifacts**:

- p-value và CI được lấy từ statsmodels (MLE + Wald z-test) trong `outputs/logit_summary.txt`, được sinh bởi:
  `src/hr_employee/stats/logit_odds_ratio.py` (L45–L52) và được lưu bởi `scripts/run_pipeline.py` (L167–L174).

## 5) Công thức Odds Ratio (OR) và diễn giải

### 5.1 Odds và Odds Ratio

Odds (odds - tỷ số odds) của churn:

$$
\text{odds}(x) = \frac{p(y=1\mid x)}{1 - p(y=1\mid x)}
$$

Với logit tuyến tính, khi tăng feature $x_j$ thêm 1 đơn vị, OR của $x_j$:

$$
OR_j = \exp(\beta_j)
$$

**Triển khai trong code**:

- OR được tính bằng `np.exp(params.values)` tại:
  `src/hr_employee/stats/logit_odds_ratio.py` (L54–L60)

### 5.2 95% Confidence Interval (CI) cho OR

Nếu $CI(\beta_j) = [l_j, u_j]$ thì:

$$
CI(OR_j) = [\exp(l_j), \exp(u_j)]
$$

**Triển khai**:

- `result.conf_int()` lấy CI của beta (L50–L52)
- exponentiate để ra CI của OR: `np.exp(conf["ci_lower"])`, `np.exp(conf["ci_upper"])` (L58–L59)
  trong `src/hr_employee/stats/logit_odds_ratio.py` (L49–L60)

### 5.3 Baseline (nhóm tham chiếu) khi dummy hoá

Vì dùng `drop_first=True`:

- Mỗi biến dummy (ví dụ `MaritalStatus_Single`) được hiểu **so với baseline** của cột đó (category bị drop).
- Quy tắc này được tạo tại:
  `src/hr_employee/stats/logit_odds_ratio.py` (L29–L31)

**Ghi chú kỹ thuật (pandas `get_dummies`)**:

- Baseline chính xác là **category bị drop** sau khi dummy hoá; thứ tự category có thể phụ thuộc vào kiểu dữ liệu và thứ tự levels (ví dụ nếu cột là `category` có order, hoặc nếu là `object` thì thường theo thứ tự levels nội bộ).
- Cách kiểm chứng chắc chắn nhất là nhìn trực tiếp các cột dummy đã sinh trong `outputs/odds_ratio_table.csv` (đây là “source of truth” cho baseline trong báo cáo này).

Trong dataset này, ta quan sát được các cột dummy được tạo ra trong `outputs/odds_ratio_table.csv` (L1–L13):

- **Geography**: có `MaritalStatus_Single` và `MaritalStatus_Married` ⇒ baseline là **France**.
- **Gender**: có `Gender_Male` ⇒ baseline là **Female**.

Khi diễn giải chuẩn học thuật, cần viết rõ “so với ai”:

> Ví dụ: `MaritalStatus_Single` có OR=3.11 nghĩa là nhân viên thường xuyên đi công tác có odds churn cao gấp 2.17 lần so với nhân viên ít đi công tác (nhóm tham chiếu), trong điều kiện các biến khác không đổi (ceteris paribus).

## 6) Kết quả hiện tại (Results) và giải thích

### 6.1 Metrics mô hình (sklearn)

Đọc từ `outputs/logistic_metrics.json` (L1–L16):

- ROC AUC: 0.7771
- Accuracy: 0.7135
- Precision: 0.3872
- Recall: 0.7002
- F1: 0.4987
- Confusion matrix (dạng bảng):

|  | **Dự đoán: Ở lại (0)** | **Dự đoán: Rời bỏ (1)** |
| --- | --- | --- |
| **Thực tế: Ở lại (0)** | **1142 (TN)** | 451 (FP) |
| **Thực tế: Rời bỏ (1)** | 122 (FN) | **285 (TP)** |

**Diễn giải để trình bày**:

- Recall ~ 0.70: bắt được ~70% nhân viên nghỉ việc thật (giảm FN).
- Precision ~ 0.39: trong các cảnh báo churn, ~39% đúng (FP còn nhiều).
- Với bài toán churn, thường ưu tiên Recall (giữ chân càng nhiều càng tốt), nhưng cần cân đối chi phí gọi/ưu đãi (cost trade-off - đánh đổi chi phí).

**Biểu đồ liên quan**:

- ROC curve: `figures/roc_curve.png`
- Confusion matrix: `figures/confusion_matrix.png`

### 6.2 Odds Ratio (statsmodels) — các yếu tố nổi bật

Đọc từ `outputs/odds_ratio_table.csv` (L1–L13):

- **Age**: OR=1.0754, p rất nhỏ → tuổi tăng làm odds churn tăng.
- **OverTime_Yes**: OR=0.3411 → nhân viên có mức độ hài lòng cao giảm odds churn mạnh.
- **MaritalStatus_Single**: OR=3.1158 → odds churn của nhân viên ở Đức cao hơn **so với France (baseline)**, giữ nguyên các biến khác (ceteris paribus).
- **Gender_Male**: OR=1.4830 → odds churn của nhân viên nam thấp hơn **so với Female (baseline)**, giữ nguyên các biến khác (ceteris paribus).

Đọc từ `outputs/logit_summary.txt` (L13–L24) để có hệ số beta, z-score, CI của beta.

**Diễn giải hệ số chặn (Intercept / Constant - hệ số chặn)**:

- Hệ số chặn $\beta_0$ (const) là **log-odds churn** của “nhân viên cơ sở” khi tất cả biến độc lập bằng 0:
  - MaritalStatus baseline: Divorced (vì `MaritalStatus_Single=0`, `MaritalStatus_Married=0`)
  - Gender baseline: Female (vì `Gender_Male=0`)
  - Các biến nhị phân như `OverTime_Yes`, `WorkLifeBalance` bằng 0
  - Các biến liên tục (`Age`, `MonthlyIncome`, ...) bằng 0 (mang tính quy ước toán học; không phải nhân viên “thực tế”)
- Trong `outputs/logit_summary.txt`, const $\approx -3.3923$ (L13) ⇒ log-odds nền tảng âm lớn ⇒ xác suất churn nền tảng thấp:

$$
p_0 = \sigma(\beta_0) = \frac{1}{1+e^{-\beta_0}} \approx \frac{1}{1+e^{3.3923}} \approx 0.0326
$$

Ghi chú học thuật: để intercept “có nghĩa thực tế” hơn, thường center/standardize biến liên tục (ví dụ theo mean) trước khi fit mô hình suy luận.

**Phương trình hồi quy ước lượng (Estimated regression equation - phương trình ước lượng)**:

Từ `outputs/logit_summary.txt` (L13–L24), phương trình logit ước lượng:

$$
\log\left(\frac{p}{1-p}\right) =
-3.3923
-0.0007 \cdot \text{YearsAtCompany}
+0.0727 \cdot \text{Age}
-0.0159 \cdot \text{TotalWorkingYears}
+2.637\times 10^{-6} \cdot \text{MonthlyIncome}
-0.1015 \cdot \text{NumCompaniesWorked}
-0.0447 \cdot \text{WorkLifeBalance}
-1.0754 \cdot \text{OverTime_Yes}
+4.807\times 10^{-7} \cdot \text{HourlyRate}
+0.7747 \cdot \text{Geography\_Germany}
+0.0352 \cdot \text{Geography\_Spain}
-0.5285 \cdot \text{Gender\_Male}
$$

Sau đó:

$$
p = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}
$$

Diễn giải nhanh: hệ số dương (+) làm tăng log-odds (tăng xác suất churn), hệ số âm (-) làm giảm log-odds (giảm xác suất churn), trong điều kiện các biến khác không đổi (ceteris paribus).

**Biểu đồ OR**:

- Forest plot: `figures/forest_plot_odds_ratio.png`

## 7) “Kịch bản trình bày” theo từng bước (Teacher presentation script)

Bạn có thể trình bày theo format 7 phút:

1) **Giới thiệu bài toán**: churn prediction + interpretability.
2) **Mô tả dữ liệu**: 1470 nhân viên, attrition ~16%, biến numeric/categorical; nêu FeatureSpec:
   `src/hr_employee/preprocessing/features.py` (L6–L22).
3) **EDA**: churn theo MaritalStatus/Gender + phân phối Age/MonthlyIncome/DistanceFromHome; nêu file EDA:
   `src/hr_employee/analysis/eda.py` (L20–L41) và hình `figures/*.png`.
4) **Tiền xử lý**: OneHotEncoder + StandardScaler:
   `src/hr_employee/preprocessing/pipeline.py` (L20–L37).
5) **Train/test**: stratified split:
   `src/hr_employee/training/pipeline.py` (L22–L26).
6) **Mô hình**: LogisticRegression (L2 regularization, class_weight balanced):
   `src/hr_employee/model/logistic.py` (L8–L31) và `scripts/run_pipeline.py` (L107–L111).
7) **Đánh giá**: compute_metrics và confusion matrix:
   `src/hr_employee/evaluation/metrics.py` (L28–L37); nêu ROC AUC/Recall.
8) **Suy luận thống kê**: Logit + OR=exp(beta):
   `src/hr_employee/stats/logit_odds_ratio.py` (L38–L64).
9) **Khuyến nghị**: tập trung cân bằng công việc, phân tích theo phòng ban, tối ưu threshold theo chi phí.

## 8) Cách chạy lại (Reproducibility)

### 8.1 Chạy Python pipeline (macOS)

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
python3 scripts/run_pipeline.py
```

Ghi chú: file `outputs/logistic_pipeline.joblib` là artifact sinh tự động (binary). Repo được cấu hình `.gitignore` để không commit `outputs/*.joblib`.

### 8.3 Compile LaTeX Reports

**Yêu cầu**: LaTeX distribution (MiKTeX hoặc TeX Live) đã được cài đặt.

**Báo cáo chính (main report)**:

```bash
cd report
pdflatex main.tex
# Hoặc dùng latexmk để tự động compile nhiều lần
latexmk -pdf main.tex
```

**Presentation slides**:

```bash
cd report
pdflatex presentation.tex
# Hoặc
latexmk -pdf presentation.tex
```

**Lưu ý**: Các file build LaTeX (`.aux`, `.log`, `.fdb_latexmk`, v.v.) được cấu hình trong `.gitignore` để không commit vào repository.

## 9) Phụ lục: Nơi code sinh artifacts (traceability)

**Python artifacts:**

- **Metrics JSON**: `scripts/run_pipeline.py` (L122–L154) gọi `compute_metrics`:
  `src/hr_employee/evaluation/metrics.py` (L28–L37)
- **Model joblib**: `scripts/run_pipeline.py` (L163–L165)
- **Odds ratio table**: `scripts/run_pipeline.py` (L167–L174) gọi:
  `src/hr_employee/stats/logit_odds_ratio.py` (L38–L64)
- **OR analysis text + forest plot**: `scripts/run_pipeline.py` (L176–L185)

**LaTeX artifacts:**

- **Báo cáo chính**: `report/main.tex` → `report/main.pdf`
  - Các section được tổ chức trong `report/sections/`
  - Hình ảnh từ `report/assets/figures/`
  - Bảng từ `report/tables/` (nếu có)
- **Presentation slides**: `report/presentation.tex` → `report/presentation.pdf`
  - Các slide được tổ chức trong `report/slides/`
  - Sử dụng theme Warsaw với color scheme whale
  - Hình ảnh từ `report/assets/figures/`

## 10) Ghi chú quan trọng khi trình bày (Notes for presentation)

### 10.1 Vì sao dùng cả sklearn LogisticRegression và statsmodels Logit?

- **sklearn LogisticRegression**: tối ưu cho prediction/inference (dự đoán), dễ đóng gói thành pipeline và lưu `.joblib`.
  - Nơi ghép pipeline: `src/hr_employee/training/pipeline.py` (L29–L35)
  - Nơi build model: `src/hr_employee/model/logistic.py` (L22–L31)
- **statsmodels Logit**: tối ưu cho inference (suy luận thống kê) — cung cấp p-value, CI, summary rõ ràng cho báo cáo.
  - Nơi fit Logit + lấy CI/p-value: `src/hr_employee/stats/logit_odds_ratio.py` (L45–L52)

### 10.2 Vì sao cần StandardScaler cho Logistic Regression?

Với regularization L2, việc đưa các numeric features về cùng thang đo giúp:

- Ổn định tối ưu (optimization stability - ổn định tối ưu)
- Hệ số $\beta$ bớt bị “lệch” do đơn vị đo quá lớn

Nơi scaler được áp dụng:
`src/hr_employee/preprocessing/pipeline.py` (L26–L33).

### 10.3 “OR của MonthlyIncome gần 1” có nghĩa gì?

Trong OR table:

- `MonthlyIncome` có OR ~ 1.0000026 (rất gần 1).

Điều này thường xảy ra vì:

- Đơn vị của `MonthlyIncome` quá lớn (1 đơn vị tiền) → OR theo “1 đơn vị” nhỏ xíu.
- Đúng cách diễn giải là cần **rescale**: ví dụ OR theo 10,000 hoặc 100,000 đơn vị tiền.

Gợi ý: khi viết luận văn, nên chuẩn hoá/scale lại biến continuous (liên tục) theo đơn vị dễ hiểu.

### 10.4 Bàn luận về ngưỡng quyết định (Decision Threshold Discussion - bàn luận threshold)

Trong `compute_metrics`, dự án dùng threshold mặc định 0.5 để chuyển xác suất $\hat{p}$ thành nhãn dự đoán $\hat{y}$:

- `src/hr_employee/evaluation/metrics.py` (L28–L30)

Tuy nhiên, với bài toán churn (mất cân bằng lớp) trong nhân sự, threshold=0.5 **hiếm khi tối ưu**. Cần bàn luận theo business trade-off (đánh đổi kinh doanh):

- **Ưu tiên High Recall (không bỏ sót nhân viên sắp churn)**:
  - Giảm threshold (ví dụ 0.3–0.4) ⇒ tăng TP nhưng cũng tăng FP.
  - Phù hợp khi chi phí bỏ sót nhân viên churn (FN) cao hơn chi phí chăm sóc nhầm (FP).
- **Ưu tiên High Precision (tiết kiệm chi phí ưu đãi/marketing)**:
  - Tăng threshold ⇒ giảm FP nhưng tăng FN.
  - Phù hợp khi chi phí chăm sóc/khuyến mãi cao.

Trong bảo vệ luận văn, việc nêu rõ “tối ưu metric nào” (Recall/F1/expected cost) thể hiện tư duy thống kê + ứng dụng tốt hơn so với chỉ báo cáo Accuracy.

## 11) Hạn chế và hướng phát triển (Limitations & Future Work)

### 11.1 Hạn chế (Limitations)

- **Rescale biến liên tục để OR dễ hiểu**: `MonthlyIncome` và `HourlyRate` có đơn vị lớn ⇒ OR theo “1 đơn vị” gần 1. Nên rescale theo 10k/100k để diễn giải.
- **Chưa kiểm định đa cộng tuyến (multicollinearity)**: chưa tính VIF (Variance Inflation Factor - hệ số phóng đại phương sai). Nếu VIF cao (thường >5 hoặc >10), ước lượng $\beta$ có thể kém ổn định.
- **Giả định tuyến tính của logit (linearity of logit)**: Logistic Regression giả định quan hệ tuyến tính giữa biến liên tục và log-odds, có thể bị vi phạm nếu quan hệ phi tuyến.

### 11.2 Hướng phát triển (Future work)

- **VIF diagnostics**: tính VIF cho design matrix (sau dummy hoá) và báo cáo biến có nguy cơ đa cộng tuyến.
- **Linearity checks**: kiểm tra tuyến tính logit (ví dụ thêm spline/biến bậc hai hoặc binning cho biến continuous).
- **Threshold optimization**: chọn threshold theo mục tiêu (maximize Recall/F1, hoặc minimize expected cost) thay vì cố định 0.5.
