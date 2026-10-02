from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from determine_training_thresholds import main as prepare_predictions
from evaluate_validation_metrics import DISPLAY_NAMES


OUTPUT_DIR = Path("evaluation_outputs/decision_curves")
THRESHOLDS = np.linspace(0.05, 0.50, 100)
N_BOOTSTRAPS = 1000
RANDOM_SEED = 123
COLORS = {
    "Logistic": "#2878B5",
    "DecisionTree": "#9AC9DB",
    "RandomForest": "#F8AC8C",
    "XGBoost": "#C82423",
    "LightGBM": "#FF8884",
    "SVM": "#6F4E7C",
    "ANN": "#3B9B72",
}


def calculate_net_benefit(y_true, probabilities, thresholds=THRESHOLDS):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if y_true.ndim != 1 or y_true.shape != probabilities.shape or len(y_true) == 0:
        raise ValueError("Outcomes and probabilities must be matching nonempty vectors")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Outcomes must be coded 0/1")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and between 0 and 1")
    if thresholds.ndim != 1 or not len(thresholds) or not np.isfinite(thresholds).all():
        raise ValueError("Thresholds must be a nonempty finite vector")
    if np.any((thresholds <= 0) | (thresholds >= 1)):
        raise ValueError("Thresholds must be strictly between 0 and 1")
    positive = probabilities[:, None] >= thresholds[None, :]
    true_positive = (positive & (y_true[:, None] == 1)).sum(axis=0)
    false_positive = (positive & (y_true[:, None] == 0)).sum(axis=0)
    odds = thresholds / (1 - thresholds)
    prevalence = y_true.mean()
    return pd.DataFrame({
        "Threshold": thresholds,
        "Model_Net_Benefit": (true_positive - false_positive * odds) / len(y_true),
        "Treat_All_Net_Benefit": prevalence - (1 - prevalence) * odds,
        "Treat_None_Net_Benefit": np.zeros(len(thresholds)),
    })


def bootstrap_dca(y_true, probabilities):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    result = calculate_net_benefit(y_true, probabilities)
    rng = np.random.default_rng(RANDOM_SEED)
    samples = np.empty((N_BOOTSTRAPS, len(THRESHOLDS)))
    for i in range(N_BOOTSTRAPS):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        samples[i] = calculate_net_benefit(
            y_true[indices], probabilities[indices]
        )["Model_Net_Benefit"].to_numpy()
    result["Model_CI_Lower"] = np.percentile(samples, 2.5, axis=0)
    result["Model_CI_Upper"] = np.percentile(samples, 97.5, axis=0)
    return result


def configure_axes(ax, title):
    ax.set(
        xlabel="Threshold probability",
        ylabel="Net benefit",
        title=title,
        xlim=(THRESHOLDS.min(), THRESHOLDS.max()),
    )
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="best", frameon=False)


def add_references(ax, result):
    ax.plot(
        result["Threshold"], result["Treat_All_Net_Benefit"],
        linestyle="--", color="black", label="Treat all",
    )
    ax.plot(
        result["Threshold"], result["Treat_None_Net_Benefit"],
        linestyle=":", color="gray", label="Treat none",
    )


def main():
    _, _, _, probabilities, _, y_val = prepare_predictions()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    with plt.rc_context({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12}):
        for key, name in DISPLAY_NAMES.items():
            result = (
                bootstrap_dca(y_val, probabilities[key])
                if key == "Logistic"
                else calculate_net_benefit(y_val, probabilities[key])
            )
            results[key] = result
            result.to_csv(OUTPUT_DIR / f"{key}_DCA_results.csv", index=False, encoding="utf-8-sig")
            fig, ax = plt.subplots(figsize=(8, 6))
            if key == "Logistic":
                ax.fill_between(
                    result["Threshold"].to_numpy(),
                    result["Model_CI_Lower"].to_numpy(),
                    result["Model_CI_Upper"].to_numpy(),
                    color=COLORS[key], alpha=0.20, linewidth=0, label="Pointwise 95% CI",
                )
            ax.plot(
                result["Threshold"], result["Model_Net_Benefit"],
                color=COLORS[key], linewidth=2.2, label=name,
            )
            add_references(ax, result)
            configure_axes(ax, f"Decision Curve Analysis – {name}")
            fig.savefig(OUTPUT_DIR / f"{key}_DCA.pdf", bbox_inches="tight")
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 8))
        for key, name in DISPLAY_NAMES.items():
            ax.plot(
                results[key]["Threshold"], results[key]["Model_Net_Benefit"],
                color=COLORS[key], linewidth=2, label=name,
            )
        add_references(ax, results["Logistic"])
        configure_axes(ax, "Decision Curve Analysis – Internal Validation Cohort")
        fig.savefig(OUTPUT_DIR / "DCA_curves_all_models_validation.pdf", bbox_inches="tight")
        plt.close(fig)
    print(f"Decision-curve results saved to: {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    decision_curve_results = main()
