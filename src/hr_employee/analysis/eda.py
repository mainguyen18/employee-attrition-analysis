from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hr_employee.preprocessing.features import FeatureSpec


@dataclass(frozen=True, slots=True)
class EdaSummary:
    """Key EDA summary (tóm tắt EDA chính)."""

    n_rows: int
    n_cols: int
    churn_rate: float
    missing_by_column: dict[str, int]


def compute_eda_summary(raw_df: pd.DataFrame, feature_spec: FeatureSpec) -> EdaSummary:
    missing_counts = raw_df.isna().sum().sort_values(ascending=False)
    churn_rate = float(raw_df[feature_spec.target_column].mean())
    return EdaSummary(
        n_rows=int(raw_df.shape[0]),
        n_cols=int(raw_df.shape[1]),
        churn_rate=churn_rate,
        missing_by_column={str(k): int(v) for k, v in missing_counts.items()},
    )


def churn_rate_by_category(
    raw_df: pd.DataFrame, *, category_column: str, feature_spec: FeatureSpec
) -> pd.DataFrame:
    """Compute churn rate by category (tỷ lệ churn theo nhóm)."""
    grouped = (
        raw_df.groupby(category_column, dropna=False)[feature_spec.target_column]
        .agg(["count", "mean"])
        .rename(columns={"count": "n", "mean": "churn_rate"})
        .sort_values("churn_rate", ascending=False)
    )
    return grouped.reset_index()


def describe_numeric_by_churn(raw_df: pd.DataFrame, feature_spec: FeatureSpec) -> pd.DataFrame:
    """Numeric descriptive stats split by churn (thống kê mô tả theo churn)."""
    numeric_cols = list(feature_spec.numeric_columns)
    return raw_df.groupby(feature_spec.target_column)[numeric_cols].describe().transpose()






