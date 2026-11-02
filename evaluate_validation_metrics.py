from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from determine_training_thresholds import main as prepare_predictions

OUTPUT_DIR = Path("evaluation_outputs")
CONFUSION_DIR = OUTPUT_DIR / "confusion_matrices"
N_BOOTSTRAPS = 1000
RANDOM_SEED = 123
DISPLAY_NAMES = {
    "Logistic": "Logistic regression",
    "DecisionTree": "Decision tree",
    "RandomForest": "Random forest",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "SVM": "SVM",
    "ANN": "ANN",
}


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = map(float, (tn, fp, fn, tp))
    total = tn + fp + fn + tp
    accuracy = safe_ratio(tp + tn, total)
    sensitivity = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    precision = safe_ratio(tp, tp + fp)
    expected_agreement = safe_ratio(
        (tn + fp) * (tn + fn) + (fn + tp) * (fp + tp), total ** 2
    )
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "Kappa": safe_ratio(accuracy - expected_agreement, 1 - expected_agreement),
        "Youden_J": sensitivity + specificity - 1,
        "PPV": precision,
        "NPV": safe_ratio(tn, tn + fn),
    }


def calculate_metrics_with_ci(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape or len(y_true) == 0:
        raise ValueError("Outcomes and predictions must be matching nonempty vectors")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Outcomes and predictions must be coded 0/1")

    estimates = calculate_metrics(y_true, y_pred)
    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_values = {name: np.empty(N_BOOTSTRAPS) for name in estimates}
    for i in range(N_BOOTSTRAPS):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        sample_metrics = calculate_metrics(y_true[indices], y_pred[indices])
        for name, value in sample_metrics.items():
            bootstrap_values[name][i] = value

    results = {}
    for name, estimate in estimates.items():
        samples = bootstrap_values[name]
        valid_samples = samples[np.isfinite(samples)]
        lower, upper = (
            np.percentile(valid_samples, [2.5, 97.5])
            if len(valid_samples)
            else (np.nan, np.nan)
        )
        results[name] = estimate
        results[f"{name}_CI_Lower"] = lower
        results[f"{name}_CI_Upper"] = upper
        results[f"{name}_Valid_Bootstraps"] = len(valid_samples)
    return results


def plot_confusion_matrix(cm, model_key, model_name):
    row_totals = cm.sum(axis=1, keepdims=True)
    percentages = np.divide(
        cm.astype(float),
        row_totals,
        out=np.zeros_like(cm, dtype=float),
        where=row_totals != 0,
    )
    annotations = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm[i, j]}\n{percentages[i, j]:.2%}"
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        percentages,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=["Survival", "In-hospital death"],
        yticklabels=["Survival", "In-hospital death"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.savefig(
        CONFUSION_DIR / f"{model_key}_confusion_matrix.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    _, thresholds, _, validation_probabilities, _, y_val = prepare_predictions()
    y_val = np.asarray(y_val)
    if y_val.ndim != 1 or len(y_val) == 0 or set(np.unique(y_val)) != {0, 1}:
        raise ValueError("Validation outcomes must be a nonempty vector containing both 0 and 1")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    count_rows = []
    prediction_columns = {"Outcome": y_val}
    for key, model_name in DISPLAY_NAMES.items():
        if key not in thresholds or key not in validation_probabilities:
            raise ValueError(f"{model_name}: missing threshold or validation probabilities")
        cutoff = float(thresholds[key])
        if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
            raise ValueError(f"{model_name}: training cutoff must be finite and between 0 and 1")
        probabilities = np.asarray(validation_probabilities[key], dtype=float)
        if probabilities.shape != y_val.shape or not np.isfinite(probabilities).all():
            raise ValueError(f"{model_name}: probabilities must be finite and match outcome shape")
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError(f"{model_name}: probabilities must be between 0 and 1")
        predicted_classes = (probabilities >= cutoff).astype(int)
        cm = confusion_matrix(y_val, predicted_classes, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        total = int(cm.sum())
        if total != len(y_val):
            raise ValueError(f"{model_name}: confusion matrix sample count mismatch")
        metric_rows.append({
            "Model": model_name,
            "Training_Cutoff": cutoff,
            **calculate_metrics_with_ci(y_val, predicted_classes),
        })
        count_rows.append({
            "Model": model_name, "Cutoff": cutoff,
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn), "Total": total,
        })
        prediction_columns[f"{key}_Probability"] = probabilities
        prediction_columns[f"{key}_Prediction"] = predicted_classes
        plot_confusion_matrix(cm, key, model_name)
    metrics_table = pd.DataFrame(metric_rows)
    counts_table = pd.DataFrame(count_rows)
    metrics_path = OUTPUT_DIR / "validation_classification_metrics.csv"
    counts_path = CONFUSION_DIR / "validation_confusion_matrix_counts.csv"
    metrics_table.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    counts_table.to_csv(counts_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_columns).to_csv(
        OUTPUT_DIR / "validation_classification_predictions.csv", index=False, encoding="utf-8-sig"
    )
    print(metrics_table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nValidation-set confusion matrix counts:")
    print(counts_table.to_string(index=False, formatters={"Cutoff": "{:.4f}".format}))
    print(f"\nAll models evaluated on {len(y_val)} validation patients.")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Confusion matrix outputs saved to: {CONFUSION_DIR}")
    return metrics_table, counts_table


if __name__ == "__main__":
    validation_metrics, confusion_count_table = main()
