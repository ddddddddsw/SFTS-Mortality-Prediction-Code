"""Plot validation confusion matrices using fixed training-derived cutoffs.

Keep determine_training_thresholds.py in the same directory. Run after model
training. Outcomes: 0 = survival; 1 = in-hospital death. Percentages are
normalized within each actual-outcome row. Classification uses probability
>= cutoff; thresholds retain full precision for classification.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from determine_training_thresholds import main as prepare_predictions


OUTPUT_DIR = Path("evaluation_outputs/confusion_matrices")
DISPLAY_NAMES = {
    "Logistic": "Logistic regression",
    "DecisionTree": "Decision tree",
    "RandomForest": "Random forest",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "SVM": "SVM",
    "ANN": "ANN",
}


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
        OUTPUT_DIR / f"{model_key}_confusion_matrix.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    # Recalculate cutoffs from training predictions, without refitting models.
    # Validation outcomes do not contribute to cutoff selection.
    _, thresholds, _, validation_probabilities, _, y_val = prepare_predictions()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for key, model_name in DISPLAY_NAMES.items():
        cutoff = thresholds[key]
        predicted_classes = (validation_probabilities[key] >= cutoff).astype(int)
        cm = confusion_matrix(y_val, predicted_classes, labels=[0, 1])
        # Fixed layout: [[TN, FP], [FN, TP]].
        tn, fp, fn, tp = cm.ravel()
        total = int(cm.sum())
        if total != len(y_val):
            raise ValueError(f"{model_name}: confusion matrix sample count mismatch")

        plot_confusion_matrix(cm, key, model_name)
        rows.append({
            "Model": model_name,
            "Cutoff": cutoff,
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "Total": total,
        })

    table = pd.DataFrame(rows)
    # Round only for display; preserve full precision in the exported table.
    print("\nValidation-set confusion matrix counts:")
    print(table.to_string(index=False, formatters={"Cutoff": "{:.4f}".format}))
    table.to_csv(
        OUTPUT_DIR / "validation_confusion_matrix_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"\nAll models evaluated on {len(y_val)} validation patients.")
    print(f"Results saved to: {OUTPUT_DIR}")
    return table


if __name__ == "__main__":
    confusion_count_table = main()
