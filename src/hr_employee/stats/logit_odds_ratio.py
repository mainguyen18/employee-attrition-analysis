from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from hr_employee.preprocessing.features import FeatureSpec


@dataclass(frozen=True, slots=True)
class OddsRatioRow:
    feature: str
    odds_ratio: float
    ci_lower: float
    ci_upper: float
    p_value: float


def _prepare_design_matrix(
    raw_df: pd.DataFrame, feature_spec: FeatureSpec
) -> tuple[pd.DataFrame, pd.Series]:
    y = raw_df[feature_spec.target_column].astype(int)

    x = raw_df.drop(
        columns=[*feature_spec.id_columns, feature_spec.target_column]
    ).copy()
    x = pd.get_dummies(
        x, columns=list(feature_spec.categorical_columns), drop_first=True
    )
    # Ensure numeric dtype for statsmodels (avoid object dtypes)
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    x = sm.add_constant(x, has_constant="add")
    return x, y


def fit_logit_and_odds_ratio(
    raw_df: pd.DataFrame, feature_spec: FeatureSpec
) -> tuple[sm.Logit, sm.discrete.discrete_model.BinaryResults, pd.DataFrame]:
    """Fit statsmodels Logit and return Odds Ratio table.

    Lưu ý: OR của biến liên tục là theo 1 đơn vị tăng; có thể cân nhắc rescale khi viết báo cáo.
    """
    x, y = _prepare_design_matrix(raw_df, feature_spec)
    model = sm.Logit(y, x)
    result = model.fit(disp=False, maxiter=200)

    params = result.params
    conf = result.conf_int()
    conf.columns = ["ci_lower", "ci_upper"]
    p_values = result.pvalues

    or_df = pd.DataFrame(
        {
            "feature": params.index,
            "odds_ratio": np.exp(params.values),
            "ci_lower": np.exp(conf["ci_lower"].values),
            "ci_upper": np.exp(conf["ci_upper"].values),
            "p_value": p_values.values,
        }
    ).sort_values("p_value", ascending=True)

    return model, result, or_df.reset_index(drop=True)




