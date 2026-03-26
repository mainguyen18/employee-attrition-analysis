from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hr_employee.preprocessing.features import FeatureSpec


@dataclass(frozen=True, slots=True)
class PreprocessArtifacts:
    """Returned artifacts from preprocessing (kết quả tiền xử lý)."""

    transformer: ColumnTransformer
    feature_spec: FeatureSpec


def build_preprocessor(feature_spec: FeatureSpec) -> PreprocessArtifacts:
    """Build ColumnTransformer for preprocessing.

    - One-hot encode categorical columns.
    - Standardize numeric columns (beneficial for regularized logistic regression).
    """
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numeric_transformer = StandardScaler()

    transformer = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, list(feature_spec.categorical_columns)),
            ("numeric", numeric_transformer, list(feature_spec.numeric_columns)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return PreprocessArtifacts(transformer=transformer, feature_spec=feature_spec)


def split_xy(raw_df: pd.DataFrame, feature_spec: FeatureSpec) -> tuple[pd.DataFrame, pd.Series]:
    """Split raw DataFrame into X and y (tách biến độc lập/phụ thuộc)."""
    missing_cols = [
        col
        for col in (
            set(feature_spec.id_columns)
            | {feature_spec.target_column}
            | set(feature_spec.categorical_columns)
            | set(feature_spec.numeric_columns)
        )
        if col not in raw_df.columns
    ]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    y = raw_df[feature_spec.target_column]
    x = raw_df.drop(columns=[*feature_spec.id_columns, feature_spec.target_column])
    return x, y


