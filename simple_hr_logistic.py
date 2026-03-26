import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# 1. TẢI DỮ LIỆU
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Xử lý cột Attrition (Mục tiêu) thành 0 và 1
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# 2. KHAI BÁO BIẾN
# Bỏ đi các cột Id/không có ý nghĩa
cols_to_drop = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours", "Department", "Attrition"]
y = df['Attrition']
X_raw = df.drop(columns=cols_to_drop)

# Tách biến phân loại (Chữ) và biến số (Số)
categorical_cols = ["BusinessTravel", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]
numeric_cols = [c for c in X_raw.columns if c not in categorical_cols]

# 3. TIỀN XỬ LÝ DỮ LIỆU
# Biến phân loại: Tạo biến Dummy (0/1) bằng pandas
X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)

# Biến số: Chuẩn hóa theo phân phối chuẩn (StandardScaler) để tránh lệnh lệch trọng số
scaler = StandardScaler()
X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])

# 4. THỐNG KÊ TOÁN HỌC VỚI STATSMODELS (Để tìm hiểu TẠI SAO nhân viên nghỉ việc)
# Statsmodels cần phải add cứng hệ số tự do (constant)
X_stats = sm.add_constant(X_encoded.astype(float))
logit_model = sm.Logit(y, X_stats)
result = logit_model.fit(disp=False)

print("="*60)
print("1. KET QUA THONG KE (LOGISTIC REGRESSION)")
print("="*60)
# In ra P-value và Hệ số
print(result.summary().tables[1])

import numpy as np
# Tính Tỷ số chênh (Odds Ratio) = e^(Hệ số b)
odds_ratios = pd.DataFrame({
    'Odds Ratio': round(result.params.apply(lambda x: np.exp(x)), 3),
    'P-Value': round(result.pvalues, 4)
})
print("\n[CHI TIET TY SO CHENH - ODDS RATIO]")
print(odds_ratios.sort_values(by='Odds Ratio', ascending=False).head(5))


# 5. MACHINE LEARNING VỚI SCIKIT-LEARN (Để DỰ ĐOÁN nhân viên ở ranh giới nghỉ việc)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

ml_model = LogisticRegression(class_weight='balanced', max_iter=500)
ml_model.fit(X_train, y_train)

y_pred = ml_model.predict(X_test)
y_prob = ml_model.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("2. DANH GIA MO HINH DU DOAN (AI)")
print("="*60)
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.3f}")
