from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from statsmodels.nonparametric.smoothers_lowess import lowess

from determine_training_thresholds import (
    FEATURES,
    MODEL_DIR,
    TRAIN_FILE,
    VALIDATION_FILE,
    load_cohort,
)


OUTPUT_DIR = Path("evaluation_outputs/shap")
RANDOM_SEED = 123
N_BOOTSTRAPS = 1000
WATERFALL_INDEX = 0
FORCE_INDEX = 2
DISPLAY_NAMES = {
    "PCT": "PCT",
    "CD3.CD4": "CD3+CD4+ T-cells",
    "SFTSV_RNA": "SFTSV RNA load",
    "IL_6": "IL-6",
}


def global_importance_intervals(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("SHAP values must be a nonempty finite matrix")
    absolute_values = np.abs(values)
    rng = np.random.default_rng(RANDOM_SEED)
    samples = np.empty((N_BOOTSTRAPS, values.shape[1]))
    for i in range(N_BOOTSTRAPS):
        indices = rng.choice(len(values), size=len(values), replace=True)
        samples[i] = absolute_values[indices].mean(axis=0)
    bounds = np.percentile(samples, [2.5, 97.5], axis=0)
    return absolute_values.mean(axis=0), bounds[0], bounds[1]


def plot_global_importance(table):
    ordered = table.sort_values("Mean_Absolute_SHAP")
    positions = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(positions, ordered["Mean_Absolute_SHAP"], color="#2878B5", alpha=0.8)
    ax.hlines(positions, ordered["CI_Lower"], ordered["CI_Upper"], color="black", linewidth=1.5)
    ax.vlines(ordered["CI_Lower"], positions - 0.08, positions + 0.08, color="black")
    ax.vlines(ordered["CI_Upper"], positions - 0.08, positions + 0.08, color="black")
    ax.set_yticks(positions, ordered["Display_Name"])
    ax.set_xlabel("Mean absolute SHAP value (predicted probability)")
    ax.set_title("SVM Global Importance with 95% Uncertainty Intervals")
    fig.savefig(OUTPUT_DIR / "SVM_global_importance.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_dependence(predictors, values):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, feature in enumerate(FEATURES):
        ax = axes.flat[i]
        x = predictors[feature].to_numpy()
        y = values[:, i]
        scatter = ax.scatter(x, y, s=25, alpha=0.7, c=x, cmap="viridis")
        ax.axhline(0, color="darkorange", linestyle="--", linewidth=1.2)
        if len(np.unique(x)) > 1:
            fitted = lowess(y, x, frac=0.4)
            ax.plot(fitted[:, 0], fitted[:, 1], color="darkred", linewidth=2)
        ax.set_xlabel(DISPLAY_NAMES[feature])
        ax.set_ylabel("SHAP value (predicted probability)")
        ax.spines[["top", "right"]].set_visible(False)
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        colorbar.set_label("Feature value")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "SVM_dependence.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    X_train, _ = load_cohort(TRAIN_FILE)
    X_val, _ = load_cohort(VALIDATION_FILE)
    with (MODEL_DIR / "svm_model.pkl").open("rb") as model_file:
        model = pickle.load(model_file)
    class_indices = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(class_indices) != 1:
        raise ValueError("SVM must contain class 1 for in-hospital death")
    class_index = int(class_indices[0])

    def predict_probability(values):
        predictors = pd.DataFrame(np.asarray(values), columns=FEATURES)
        return model.predict_proba(predictors)

    np.random.seed(RANDOM_SEED)
    explainer = shap.KernelExplainer(predict_probability, X_train, link="identity")
    raw_values = explainer.shap_values(X_val)
    if isinstance(raw_values, list):
        values = np.asarray(raw_values[class_index], dtype=float)
    else:
        raw_values = np.asarray(raw_values)
        if raw_values.ndim != 3:
            raise ValueError("Expected SHAP values for a multi-output probability function")
        values = raw_values[:, :, class_index]
    if values.shape != X_val.shape or not np.isfinite(values).all():
        raise ValueError("SHAP dimensions or values are invalid")
    background_probability = float(np.asarray(explainer.expected_value)[class_index])
    explanation = shap.Explanation(
        values=values,
        base_values=np.full(len(X_val), background_probability),
        data=X_val.to_numpy(),
        feature_names=[DISPLAY_NAMES[feature] for feature in FEATURES],
    )
    means, lower, upper = global_importance_intervals(values)
    table = pd.DataFrame({
        "Feature": FEATURES,
        "Display_Name": [DISPLAY_NAMES[feature] for feature in FEATURES],
        "Mean_Absolute_SHAP": means,
        "CI_Lower": lower,
        "CI_Upper": upper,
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "SVM_global_importance.csv", index=False, encoding="utf-8-sig")

    with plt.rc_context({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13}):
        plot_global_importance(table)
        shap.plots.beeswarm(explanation, max_display=4, show=False)
        plt.gcf().savefig(OUTPUT_DIR / "SVM_beeswarm.pdf", bbox_inches="tight")
        plt.close()
        plot_dependence(X_val, values)
        if WATERFALL_INDEX < len(X_val):
            shap.plots.waterfall(explanation[WATERFALL_INDEX], max_display=4, show=False)
            plt.title("Class 1: In-hospital death")
            plt.gcf().savefig(OUTPUT_DIR / "SVM_waterfall.pdf", bbox_inches="tight")
            plt.close()
    if FORCE_INDEX < len(X_val):
        force_plot = shap.plots.force(explanation[FORCE_INDEX])
        shap.save_html(str(OUTPUT_DIR / "SVM_force_plot.html"), force_plot)
    print(table.to_string(index=False))
    print(f"SHAP results saved to: {OUTPUT_DIR}")
    return explanation, table


if __name__ == "__main__":
    svm_explanation, global_importance = main()
