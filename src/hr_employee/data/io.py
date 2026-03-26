from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Dataset specification (đặc tả dataset) để chuẩn hoá đường dẫn & schema."""

    csv_path: Path
    target_column: str = "Exited"


# def load_churn_dataset(dataset_spec: DatasetSpec) -> pd.DataFrame:
#     """Load dataset từ CSV.

#     Lưu ý: hàm này chỉ đọc dữ liệu; làm sạch/tiền xử lý đặt ở module khác.
#     """
#     if not dataset_spec.csv_path.exists():
#         raise FileNotFoundError(f"CSV not found: {dataset_spec.csv_path}")
#     return pd.read_csv(dataset_spec.csv_path)
def load_churn_dataset(spec: DatasetSpec) -> pd.DataFrame:
    df = pd.read_csv(spec.csv_path)
    # Xử lý riêng cột Attrition của file HR
    if 'Attrition' in df.columns:
        df['Attrition'] = df['Attrition'].replace({'Yes': 1, 'No': 0})
        # Ép kiểu về số (nếu còn là object/string)
        df['Attrition'] = pd.to_numeric(df['Attrition'])
    return df



