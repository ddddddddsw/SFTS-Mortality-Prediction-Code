from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from determine_training_thresholds import main as prepare_predictions
from evaluate_auc_delong import calculate_auc
from evaluate_validation_metrics import DISPLAY_NAMES


OUTPUT_DIR = Path("evaluation_outputs/roc_curves")


def format_auc(result):
    return (
        f"AUC = {result['AUC']:.3f} "
        f"(95% CI: {result['AUC_CI_Lower']:.3f}–{result['AUC_CI_Upper']:.3f})"
    )


def configure_axes(ax, title):
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(
        xlabel="1 − Specificity",
        ylabel="Sensitivity",
        title=title,
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="lower right", fontsize=9)


def main():
    _, _, _, probabilities, _, y_val = prepare_predictions()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curves = {}

    with plt.rc_context({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12}):
        for key, name in DISPLAY_NAMES.items():
            result = calculate_auc(y_val, probabilities[key])
            fpr, tpr, _ = roc_curve(y_val, probabilities[key])
            curves[key] = (fpr, tpr, result)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, label=format_auc(result))
            configure_axes(ax, f"ROC Curve – {name}")
            fig.savefig(OUTPUT_DIR / f"{key}_ROC_curve.pdf", bbox_inches="tight")
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        for key, name in DISPLAY_NAMES.items():
            fpr, tpr, result = curves[key]
            ax.plot(fpr, tpr, label=f"{name}: {format_auc(result)}")
        configure_axes(ax, "ROC Curves – Internal Validation Cohort")
        ax.grid(True, alpha=0.3)
        fig.savefig(OUTPUT_DIR / "ROC_curves_allmodel_validation.pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"ROC plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
