from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from scipy.stats import norm
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from determine_training_thresholds import main as prepare_predictions
from evaluate_validation_metrics import (
    DISPLAY_NAMES,
    N_BOOTSTRAPS,
    RANDOM_SEED,
    calculate_metrics_with_ci,
)

OUTPUT_DIR = Path("evaluation_outputs")
ROC_DIR = OUTPUT_DIR / "roc_curves"
CALIBRATION_DIR = OUTPUT_DIR / "calibration_curves"
N_BINS = 5
CALIBRATION_BOOTSTRAPS = 1000
CALIBRATION_SEED = 123


def validate_predictions(y_true, probabilities):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.ndim != 1 or y_true.shape != probabilities.shape:
        raise ValueError("Outcomes and probabilities must be matching vectors")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Outcomes must contain both 0 and 1")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and between 0 and 1")
    return y_true, probabilities


def calculate_auc_and_brier(y_true, probabilities):
    y_true, probabilities = validate_predictions(y_true, probabilities)
    rng = np.random.default_rng(RANDOM_SEED)
    auc_samples = []
    brier_samples = []
    for _ in range(N_BOOTSTRAPS):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        observed = y_true[indices]
        predicted = probabilities[indices]
        brier_samples.append(brier_score_loss(observed, predicted))
        if len(np.unique(observed)) == 2:
            auc_samples.append(roc_auc_score(observed, predicted))
    if not auc_samples:
        raise ValueError("No bootstrap samples contain both outcome classes")
    auc_lower, auc_upper = np.percentile(auc_samples, [2.5, 97.5])
    brier_lower, brier_upper = np.percentile(brier_samples, [2.5, 97.5])
    return {
        "AUC": roc_auc_score(y_true, probabilities),
        "AUC_CI_Lower": auc_lower,
        "AUC_CI_Upper": auc_upper,
        "AUC_Valid_Bootstraps": len(auc_samples),
        "Brier": brier_score_loss(y_true, probabilities),
        "Brier_CI_Lower": brier_lower,
        "Brier_CI_Upper": brier_upper,
        "Brier_Valid_Bootstraps": len(brier_samples),
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
    y_true, prediction_1 = validate_predictions(y_true, prediction_1)
    _, prediction_2 = validate_predictions(y_true, prediction_2)
    predictions = np.asarray([prediction_1, prediction_2])
    order = np.argsort(-y_true.astype(int))
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


def fit_logistic_calibration(y_true, probabilities):
    y_true, probabilities = validate_predictions(y_true, probabilities)
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p))
    if np.ptp(logit_p) == 0:
        raise ValueError("Calibration slope is undefined for constant predictions")
    citl_fit = sm.GLM(y_true, np.ones((len(y_true), 1)),
                      family=sm.families.Binomial(), offset=logit_p).fit()
    joint_fit = sm.GLM(y_true, sm.add_constant(logit_p, has_constant="add"),
                       family=sm.families.Binomial()).fit()
    estimates = np.array([citl_fit.params[0], joint_fit.params[0], joint_fit.params[1]])
    if not citl_fit.converged or not joint_fit.converged or not np.isfinite(estimates).all():
        raise ValueError("Calibration model did not converge to finite estimates")
    return estimates


def calculate_calibration_ci(y_true, probabilities, model_key):
    y_true, probabilities = validate_predictions(y_true, probabilities)
    point = fit_logistic_calibration(y_true, probabilities)
    rng = np.random.default_rng(CALIBRATION_SEED)
    samples = []
    for _ in range(CALIBRATION_BOOTSTRAPS):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        try:
            estimates = fit_logistic_calibration(y_true[indices], probabilities[indices])
        except (ValueError, RuntimeError, np.linalg.LinAlgError, PerfectSeparationError):
            continue
        samples.append(estimates)
    if len(samples) < 100:
        raise RuntimeError(f"Too few successful calibration bootstrap samples: {len(samples)}/{CALIBRATION_BOOTSTRAPS}")
    lower, upper = np.percentile(np.asarray(samples), [2.5, 97.5], axis=0)
    return pd.DataFrame([
        {"Model": DISPLAY_NAMES[model_key], "Metric": metric, "Estimate": point[i],
         "CI_Lower": lower[i], "CI_Upper": upper[i], "Valid_Bootstraps": len(samples)}
        for i, metric in enumerate(["CITL", "Calibration_Intercept", "Calibration_Slope"])
    ])

def format_auc(result):
    return (
        f"AUC = {result['AUC']:.3f} "
        f"(95% CI: {result['AUC_CI_Lower']:.3f}-"
        f"{result['AUC_CI_Upper']:.3f})"
    )


def format_brier(result):
    return (
        f"Brier = {result['Brier']:.3f} "
        f"(95% CI: {result['Brier_CI_Lower']:.3f}-"
        f"{result['Brier_CI_Upper']:.3f})"
    )


def configure_roc_axes(ax, title):
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(
        xlabel="1 - Specificity",
        ylabel="Sensitivity",
        title=title,
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="lower right", fontsize=9)


def configure_calibration_axes(ax, title):
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        color="gray",
        label="Perfect calibration",
    )
    ax.set(
        xlabel="Predicted Probability",
        ylabel="Observed Probability",
        title=title,
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="lower right", fontsize=8)


def plot_curves(y_true, probabilities, results):
    roc_curves = {}
    calibration_curves = {}
    calibration_rows = []
    with plt.rc_context({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12}):
        for key, name in DISPLAY_NAMES.items():
            fpr, tpr, _ = roc_curve(y_true, probabilities[key])
            roc_curves[key] = (fpr, tpr)
            observed, predicted = calibration_curve(y_true, probabilities[key], n_bins=N_BINS, strategy="quantile")
            calibration_curves[key] = (observed, predicted)
            calibration_rows.extend([
                {"Model": name, "Group": i + 1, "Mean_Predicted": p, "Observed_Proportion": o}
                for i, (p, o) in enumerate(zip(predicted, observed))
            ])
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, label=format_auc(results[key]))
            configure_roc_axes(ax, f"ROC Curve - {name}")
            fig.savefig(ROC_DIR / f"{key}_ROC_curve.pdf", bbox_inches="tight")
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(predicted, observed, marker="o", label=format_brier(results[key]))
            configure_calibration_axes(ax, f"Calibration Curve - {name}")
            fig.savefig(CALIBRATION_DIR / f"{key}_calibration_curve.pdf", bbox_inches="tight")
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 6))
        for key, name in DISPLAY_NAMES.items():
            fpr, tpr = roc_curves[key]
            ax.plot(fpr, tpr, label=f"{name}: {format_auc(results[key])}")
        configure_roc_axes(ax, "ROC Curves - Internal Validation Cohort")
        ax.grid(True, alpha=0.3)
        fig.savefig(ROC_DIR / "ROC_curves_allmodel_validation.pdf", bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 8))
        for key, name in DISPLAY_NAMES.items():
            observed, predicted = calibration_curves[key]
            ax.plot(predicted, observed, marker="o", label=f"{name}: {format_brier(results[key])}")
        configure_calibration_axes(ax, "Calibration Curves - Internal Validation Cohort")
        ax.grid(True, alpha=0.3)
        fig.savefig(CALIBRATION_DIR / "calibration_curves_allmodel_validation.pdf", bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(calibration_rows).to_csv(CALIBRATION_DIR / "calibration_curve_points.csv", index=False, encoding="utf-8-sig")
def main():
    for directory in (OUTPUT_DIR, ROC_DIR, CALIBRATION_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    _, thresholds, _, probabilities, _, y_val = prepare_predictions()
    missing_models = set(DISPLAY_NAMES).difference(probabilities)
    if missing_models:
        raise ValueError(f"Missing model predictions: {sorted(missing_models)}")
    results = {}
    performance_rows = []
    for key, name in DISPLAY_NAMES.items():
        results[key] = calculate_auc_and_brier(y_val, probabilities[key])
        if not np.isfinite(thresholds[key]):
            raise ValueError(f"{name}: training threshold is not finite")
        predictions = (probabilities[key] >= thresholds[key]).astype(int)
        performance_rows.append({
            "Model": name, "Training_Cutoff": thresholds[key],
            **results[key], **calculate_metrics_with_ci(y_val, predictions),
        })
    comparison_rows = []
    for key_1, key_2 in combinations(DISPLAY_NAMES, 2):
        auc_1, auc_2, p_value = delong_test(y_val, probabilities[key_1], probabilities[key_2])
        comparison_rows.append({
            "Model_1": DISPLAY_NAMES[key_1], "Model_2": DISPLAY_NAMES[key_2],
            "AUC_1": auc_1, "AUC_2": auc_2, "AUC_Difference": auc_1 - auc_2,
            "P_Unadjusted": p_value,
        })
    performance = pd.DataFrame(performance_rows)
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["P_Adjusted_Holm"] = holm_adjust(comparisons["P_Unadjusted"])
    comparisons["Significant_Holm_0_05"] = (comparisons["P_Adjusted_Holm"] < 0.05).astype("boolean")
    comparisons.loc[comparisons["P_Adjusted_Holm"].isna(), "Significant_Holm_0_05"] = pd.NA
    performance.to_csv(OUTPUT_DIR / "model_performance_validation.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(OUTPUT_DIR / "pairwise_delong_holm.csv", index=False, encoding="utf-8-sig")
    plot_curves(y_val, probabilities, results)
    calibration_tables = []
    for key in DISPLAY_NAMES:
        try:
            table = calculate_calibration_ci(y_val, probabilities[key], key)
            table["Status"] = "OK"
        except (ValueError, RuntimeError, np.linalg.LinAlgError, PerfectSeparationError) as error:
            table = pd.DataFrame([
                {"Model": DISPLAY_NAMES[key], "Metric": metric, "Estimate": np.nan,
                 "CI_Lower": np.nan, "CI_Upper": np.nan, "Valid_Bootstraps": np.nan,
                 "Status": str(error)}
                for metric in ["CITL", "Calibration_Intercept", "Calibration_Slope"]
            ])
            print(f"Calibration estimation failed for {DISPLAY_NAMES[key]}: {error}")
        calibration_tables.append(table)
    calibration_metrics = pd.concat(calibration_tables, ignore_index=True)
    calibration_metrics.to_csv(OUTPUT_DIR / "calibration_metrics_all_models.csv", index=False, encoding="utf-8-sig")
    wide_calibration = calibration_metrics.pivot(index="Model", columns="Metric",
                                                values=["Estimate", "CI_Lower", "CI_Upper"])
    wide_calibration.columns = [f"{metric}_{statistic}" for statistic, metric in wide_calibration.columns]
    wide_calibration = wide_calibration.join(calibration_metrics.groupby("Model")[["Valid_Bootstraps", "Status"]].first())
    wide_calibration.reindex(list(DISPLAY_NAMES.values())).reset_index().to_csv(
        OUTPUT_DIR / "calibration_intercept_slope_bootstrap.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"Outcome": np.asarray(y_val), **{key: probabilities[key] for key in DISPLAY_NAMES}}).to_csv(
        OUTPUT_DIR / "validation_probabilities.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"N_Bootstraps": N_BOOTSTRAPS, "Random_Seed": RANDOM_SEED,
                   "Calibration_Bins": N_BINS, "Calibration_Strategy": "quantile",
                   "Calibration_Bootstraps": CALIBRATION_BOOTSTRAPS, "Calibration_Seed": CALIBRATION_SEED,
                   "Bootstrap_Sampling": "ordinary paired resampling"}]).to_csv(
        OUTPUT_DIR / "auc_roc_calibration_settings.csv", index=False, encoding="utf-8-sig")
    print(performance.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(comparisons.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(calibration_metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")
    return performance, comparisons, calibration_metrics


if __name__ == "__main__":
    model_performance, pairwise_comparisons, calibration_results = main()