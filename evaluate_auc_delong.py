from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from determine_training_thresholds import main as prepare_predictions
from evaluate_validation_metrics import (
    DISPLAY_NAMES,
    N_BOOTSTRAPS,
    RANDOM_SEED,
    calculate_metrics_with_ci,
)


OUTPUT_DIR = Path("evaluation_outputs")


def calculate_auc(y_true, probabilities):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.ndim != 1 or y_true.shape != probabilities.shape:
        raise ValueError("Outcomes and probabilities must be matching vectors")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Outcomes must contain both 0 and 1")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite")
    rng = np.random.default_rng(RANDOM_SEED)
    samples = []
    for _ in range(N_BOOTSTRAPS):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[indices])) == 2:
            samples.append(roc_auc_score(y_true[indices], probabilities[indices]))
    if not samples:
        raise ValueError("No bootstrap samples contain both outcome classes")
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return {
        "AUC": roc_auc_score(y_true, probabilities),
        "AUC_CI_Lower": lower,
        "AUC_CI_Upper": upper,
        "AUC_Valid_Bootstraps": len(samples),
    }


def compute_midrank(values):
    order = np.argsort(values)
    sorted_values = np.asarray(values)[order]
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[i:j] = (i + j - 1) / 2 + 1
        i = j
    restored_ranks = np.empty(len(values), dtype=float)
    restored_ranks[order] = ranks
    return restored_ranks


def fast_delong(predictions, positive_count):
    m = positive_count
    n = predictions.shape[1] - m
    if m < 2 or n < 2:
        raise ValueError("DeLong covariance requires at least two patients per class")
    positive_ranks = np.array([compute_midrank(row[:m]) for row in predictions])
    negative_ranks = np.array([compute_midrank(row[m:]) for row in predictions])
    pooled_ranks = np.array([compute_midrank(row) for row in predictions])
    aucs = pooled_ranks[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    positive_placements = (pooled_ranks[:, :m] - positive_ranks) / n
    negative_placements = 1 - (pooled_ranks[:, m:] - negative_ranks) / m
    covariance = (
        np.atleast_2d(np.cov(positive_placements, bias=False)) / m
        + np.atleast_2d(np.cov(negative_placements, bias=False)) / n
    )
    return aucs, covariance


def delong_test(y_true, prediction_1, prediction_2):
    y_true = np.asarray(y_true)
    predictions = np.asarray([prediction_1, prediction_2], dtype=float)
    if y_true.ndim != 1 or predictions.shape != (2, len(y_true)):
        raise ValueError("Outcomes and predictions must have matching lengths")
    if set(np.unique(y_true)) != {0, 1} or not np.isfinite(predictions).all():
        raise ValueError("Outcomes must be binary and predictions finite")
    order = np.argsort(-y_true)
    aucs, covariance = fast_delong(predictions[:, order], int(np.sum(y_true == 1)))
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast)
    difference = float(aucs[0] - aucs[1])
    if variance < -1e-12:
        raise ValueError("DeLong variance is negative")
    if variance <= 0:
        p_value = np.nan
    else:
        p_value = float(2 * norm.sf(abs(difference) / np.sqrt(variance)))
    return float(aucs[0]), float(aucs[1]), p_value


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan)
    valid_indices = np.flatnonzero(np.isfinite(p_values))
    order = valid_indices[np.argsort(p_values[valid_indices])]
    corrected = np.maximum.accumulate(
        (len(p_values) - np.arange(len(order))) * p_values[order]
    )
    adjusted[order] = np.minimum(corrected, 1)
    return adjusted


def main():
    _, thresholds, _, probabilities, _, y_val = prepare_predictions()
    performance_rows = []
    for key, name in DISPLAY_NAMES.items():
        predictions = (probabilities[key] >= thresholds[key]).astype(int)
        performance_rows.append({
            "Model": name,
            "Training_Cutoff": thresholds[key],
            **calculate_auc(y_val, probabilities[key]),
            **calculate_metrics_with_ci(y_val, predictions),
        })

    comparison_rows = []
    for key_1, key_2 in combinations(DISPLAY_NAMES, 2):
        auc_1, auc_2, p_value = delong_test(y_val, probabilities[key_1], probabilities[key_2])
        comparison_rows.append({
            "Model_1": DISPLAY_NAMES[key_1],
            "Model_2": DISPLAY_NAMES[key_2],
            "AUC_1": auc_1,
            "AUC_2": auc_2,
            "AUC_Difference": auc_1 - auc_2,
            "P_Unadjusted": p_value,
        })

    performance = pd.DataFrame(performance_rows)
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["P_Adjusted_Holm"] = holm_adjust(comparisons["P_Unadjusted"])
    comparisons["Significant_Holm_0_05"] = (
        comparisons["P_Adjusted_Holm"] < 0.05
    ).astype("boolean")
    comparisons.loc[
        comparisons["P_Adjusted_Holm"].isna(), "Significant_Holm_0_05"
    ] = pd.NA
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    performance.to_csv(
        OUTPUT_DIR / "model_performance_validation.csv", index=False, encoding="utf-8-sig"
    )
    comparisons.to_csv(
        OUTPUT_DIR / "pairwise_delong_holm.csv", index=False, encoding="utf-8-sig"
    )
    print(performance.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(comparisons.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    return performance, comparisons


if __name__ == "__main__":
    model_performance, pairwise_comparisons = main()
