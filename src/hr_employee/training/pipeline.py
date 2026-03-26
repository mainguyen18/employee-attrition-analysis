from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from hr_employee.model.logistic import LogisticSpec, build_logistic_model
from hr_employee.preprocessing.features import FeatureSpec
from hr_employee.preprocessing.pipeline import build_preprocessor


@dataclass(frozen=True, slots=True)
class TrainTestSplit:
    x_train: object
    x_test: object
    y_train: np.ndarray
    y_test: np.ndarray


def stratified_split(x, y, *, test_size: float = 0.2, random_state: int = 42) -> TrainTestSplit:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return TrainTestSplit(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


def build_sklearn_pipeline(
    feature_spec: FeatureSpec,
    logistic_spec: LogisticSpec,
) -> Pipeline:
    preprocessor = build_preprocessor(feature_spec).transformer
    model = build_logistic_model(logistic_spec)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])






