from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from determine_training_thresholds import main as prepare_predictions


OUTPUT_DIR = Path("evaluation_outputs")
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


def main():
    _, thresholds, _, validation_probabilities, _, y_val = prepare_predictions()
    rows = []
    for key, display_name in DISPLAY_NAMES.items():
        cutoff = thresholds[key]
        predictions = (validation_probabilities[key] >= cutoff).astype(int)
        rows.append({
            "Model": display_name,
            "Training_Cutoff": cutoff,
            **calculate_metrics_with_ci(y_val, predictions),
        })

    table = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "validation_classification_metrics.csv"
    table.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nResults saved to: {output_path}")
    return table


if __name__ == "__main__":
    validation_metrics = main()
