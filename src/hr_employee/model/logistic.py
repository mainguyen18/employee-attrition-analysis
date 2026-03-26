from __future__ import annotations

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True, slots=True)
class LogisticSpec:
    """Logistic regression specification (đặc tả mô hình)."""

    # 'liblinear' ổn cho binary + dataset vừa; có thể đổi sang 'lbfgs' nếu cần.
    solver: str = "liblinear"
    penalty: str = "l2"
    c: float = 1.0
    max_iter: int = 2000
    class_weight: str | dict[int, float] | None = None
    random_state: int = 42
    verbose: int = 0  # Set to 1 for convergence messages


def build_logistic_model(spec: LogisticSpec) -> LogisticRegression:
    return LogisticRegression(
        solver=spec.solver,
        penalty=spec.penalty,
        C=spec.c,
        max_iter=spec.max_iter,
        class_weight=spec.class_weight,
        random_state=spec.random_state,
        verbose=spec.verbose,
    )
