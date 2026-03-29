from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import RocCurveDisplay


def save_current_figure(path: Path, *, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_churn_rate_bar(df_rate, *, category_column: str, out_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(data=df_rate, x=category_column, y="churn_rate")
    ax.set_title(f"Churn rate by {category_column}")
    ax.set_xlabel(category_column)
    ax.set_ylabel("Churn rate")
    plt.xticks(rotation=45, ha='right')
    save_current_figure(out_path)


def plot_numeric_distribution(raw_df, *, numeric_column: str, target_column: str, out_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))
    ax = sns.kdeplot(
        data=raw_df,
        x=numeric_column,
        hue=target_column,
        common_norm=False,
        fill=True,
        alpha=0.35,
    )
    ax.set_title(f"Distribution of {numeric_column} by {target_column}")
    save_current_figure(out_path)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title("ROC Curve")
    save_current_figure(out_path)


def plot_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(5, 4))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    save_current_figure(out_path)






