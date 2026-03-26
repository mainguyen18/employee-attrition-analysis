from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np


def _ensure_src_on_path() -> None:
    """
    Ensure the project `src` directory is on `sys.path` when running this script directly.

    This allows `python scripts/run_pipeline.py` to resolve `hr_employee.*` imports
    even if the package has not been installed in editable mode.
    """
    script_path: Path = Path(__file__).resolve()
    project_root: Path = script_path.parent.parent
    src_path: Path = project_root / "src"

    src_str = str(src_path)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


_ensure_src_on_path()

from hr_employee.analysis.eda import compute_eda_summary, churn_rate_by_category
from hr_employee.config import get_default_paths
from hr_employee.data.io import DatasetSpec, load_churn_dataset
from hr_employee.evaluation.metrics import compute_metrics
from hr_employee.model.logistic import LogisticSpec
from hr_employee.preprocessing.features import FeatureSpec
from hr_employee.preprocessing.pipeline import split_xy
from hr_employee.reporting.odds_ratio import plot_odds_ratio_forest, write_odds_ratio_analysis
from hr_employee.reporting.project_report import write_project_report
from hr_employee.stats.logit_odds_ratio import fit_logit_and_odds_ratio
from hr_employee.training.pipeline import build_sklearn_pipeline, stratified_split
from hr_employee.utils.fs import ensure_dir
from hr_employee.visualization.plots import (
    plot_churn_rate_bar,
    plot_confusion_matrix,
    plot_numeric_distribution,
    plot_roc_curve,
)

# Numeric columns to visualize more closely for churn separation
EDA_NUMERIC_HIGHLIGHTS: tuple[str, ...] = (
    "Age",
    "MonthlyIncome",    
    "YearsAtCompany",   
    "DistanceFromHome", 
)


def save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def overwrite_csv(df, path: Path) -> None:
    """Write CSV, forcing overwrite. If locked, instruct user to close the file and retry."""
    try:
        path.unlink(missing_ok=True)
        df.to_csv(path, index=False)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot overwrite {path}. Close any application using the file (e.g., Excel/Viewer) and rerun."
        ) from exc


def main() -> None:
    print("=" * 60)
    print("HR Employee Analysis Pipeline")
    print("=" * 60)

    paths = get_default_paths()
    ensure_dir(paths.figures_dir)
    ensure_dir(paths.outputs_dir)

    dataset_spec = DatasetSpec(csv_path=paths.data_dir / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    feature_spec = FeatureSpec()

    # 1) Load raw data
    print("\n[1/5] Loading data...")
    raw_df = load_churn_dataset(dataset_spec)
    print(f"  [OK] Loaded {len(raw_df):,} rows, {len(raw_df.columns)} columns")

    # 2) EDA summaries + charts
    print("\n[2/5] Running EDA...")
    summary = compute_eda_summary(raw_df, feature_spec)
    save_json(asdict(summary), paths.outputs_dir / "eda_summary.json")
    print(f"  [OK] Churn rate: {summary.churn_rate:.2%}")

    for cat in feature_spec.categorical_columns:
        df_rate = churn_rate_by_category(
            raw_df, category_column=cat, feature_spec=feature_spec
        )
        overwrite_csv(df_rate, paths.outputs_dir / f"churn_rate_by_{cat}.csv")
        plot_churn_rate_bar(
            df_rate,
            category_column=cat,
            out_path=paths.figures_dir / f"churn_rate_by_{cat}.png",
        )
        print(f"  [OK] Churn analysis by {cat}")

    for col in EDA_NUMERIC_HIGHLIGHTS:
        plot_numeric_distribution(
            raw_df,
            numeric_column=col,
            target_column=feature_spec.target_column,
            out_path=paths.figures_dir / f"dist_{col}_by_churn.png",
        )
    print(f"  [OK] Generated {len(EDA_NUMERIC_HIGHLIGHTS)} distribution plots")

    # 3) Train + evaluate logistic pipeline
    print("\n[3/5] Training Logistic Regression...")
    x, y = split_xy(raw_df, feature_spec)
    split = stratified_split(x, y, test_size=0.2, random_state=42)

    train_churn = np.mean(split.y_train)
    test_churn = np.mean(split.y_test)
    print(f"  [OK] Train: {len(split.x_train):,} samples (churn: {train_churn:.2%})")
    print(f"  [OK] Test:  {len(split.x_test):,} samples (churn: {test_churn:.2%})")

    logistic_spec = LogisticSpec(class_weight="balanced", random_state=42)
    pipeline = build_sklearn_pipeline(feature_spec, logistic_spec)

    print("\n  Training model (with class_weight='balanced' for imbalanced data)...")
    pipeline.fit(split.x_train, split.y_train)

    # Get feature count after preprocessing
    n_features = (
        pipeline.named_steps["preprocess"].transform(split.x_train[:1]).shape[1]
    )
    print(f"  [OK] Model trained on {n_features} features")
    print(
        f"  [OK] Solver: {logistic_spec.solver}, Penalty: {logistic_spec.penalty}, C: {logistic_spec.c}"
    )

    # Evaluate on both train and test to check overfitting
    y_prob_train = pipeline.predict_proba(split.x_train)[:, 1]
    metrics_train = compute_metrics(
        np.asarray(split.y_train), np.asarray(y_prob_train), threshold=0.5
    )

    y_prob = pipeline.predict_proba(split.x_test)[:, 1]
    metrics = compute_metrics(
        np.asarray(split.y_test), np.asarray(y_prob), threshold=0.5
    )

    print("\n  Performance Metrics:")
    print(
        f"  Train - ROC AUC: {metrics_train.roc_auc:.3f}, Accuracy: {metrics_train.accuracy:.3f}, Recall: {metrics_train.recall:.3f}"
    )
    print(
        f"  Test  - ROC AUC: {metrics.roc_auc:.3f}, Accuracy: {metrics.accuracy:.3f}, Recall: {metrics.recall:.3f}"
    )
    print(
        f"  [OK] Confusion Matrix (Test): TN={metrics.confusion_matrix[0,0]}, FP={metrics.confusion_matrix[0,1]}, FN={metrics.confusion_matrix[1,0]}, TP={metrics.confusion_matrix[1,1]}"
    )

    save_json(
        {
            "roc_auc": metrics.roc_auc,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "confusion_matrix": metrics.confusion_matrix.tolist(),
        },
        paths.outputs_dir / "logistic_metrics.json",
    )

    plot_roc_curve(
        np.asarray(split.y_test), y_prob, paths.figures_dir / "roc_curve.png"
    )
    plot_confusion_matrix(
        metrics.confusion_matrix, paths.figures_dir / "confusion_matrix.png"
    )

    # Persist the trained model for reuse (e.g., scoring script later)
    joblib.dump(pipeline, paths.outputs_dir / "logistic_pipeline.joblib")
    print("  [OK] Model saved to outputs/logistic_pipeline.joblib")

    # 4) Statsmodels Logit for odds ratios (for LaTeX tables / appendix)
    print("\n[4/5] Computing Odds Ratios (statsmodels)...")
    _, result, or_df = fit_logit_and_odds_ratio(raw_df, feature_spec)
    overwrite_csv(or_df, paths.outputs_dir / "odds_ratio_table.csv")
    (paths.outputs_dir / "logit_summary.txt").write_text(
        str(result.summary()), encoding="utf-8"
    )
    print("  [OK] Odds ratio table and summary saved")

    # 5) Human-readable report artifacts
    print("\n[5/5] Generating reports (OR analysis + PROJECT_REPORT.md)...")
    write_odds_ratio_analysis(
        paths.outputs_dir / "odds_ratio_table.csv",
        paths.outputs_dir / "odds_ratio_analysis.txt",
    )
    plot_odds_ratio_forest(
        paths.outputs_dir / "odds_ratio_table.csv",
        paths.figures_dir / "forest_plot_odds_ratio.png",
    )
    write_project_report(paths)
    print("  [OK] Reports generated")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  - Metrics: outputs/logistic_metrics.json")
    print(f"  - Model: outputs/logistic_pipeline.joblib")
    print(f"  - Odds Ratios: outputs/odds_ratio_table.csv")
    print(f"  - Odds Ratio analysis: outputs/odds_ratio_analysis.txt")
    print(f"  - Project report: PROJECT_REPORT.md")
    print(f"  - Figures: figures/ (ROC, confusion matrix, distributions)")
    print()


if __name__ == "__main__":
    main()
