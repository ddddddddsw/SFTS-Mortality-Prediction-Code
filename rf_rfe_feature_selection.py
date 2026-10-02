from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


TRAIN_FILE = Path("cleaned_data.csv")
OUTPUT_DIR = Path("feature_selection_outputs")
RANDOM_SEED = 123
N_FOLDS = 5
DISPLAY_NAMES = {
    "CNS_symptoms": "CNS symptoms",
    "Onset_to_admission_interval": "Admission delay",
    "SFTSV_RNA_load": "SFTSV RNA load",
    "SFTSV_RNA": "SFTSV RNA load",
    "IL_2": "IL-2",
    "IL_4": "IL-4",
    "IL_6": "IL-6",
    "IL_10": "IL-10",
    "IL_17A": "IL-17A",
    "IFN_gamma": "IFN-γ",
    "TNF_alpha": "TNF-α",
    "TGF_beta": "TGF-β",
    "CD3.CD4": "CD3+CD4+ T-cells",
    "CD3.CD4.": "CD3+CD4+ T-cells",
    "CD3.CD8": "CD3+CD8+ T-cells",
    "CD3.CD8.": "CD3+CD8+ T-cells",
    "CD3.CD19": "CD3−CD19+ B-cells",
    "CD3.CD19.": "CD3−CD19+ B-cells",
}


def load_training_data():
    data = pd.read_csv(TRAIN_FILE, encoding="GBK")
    if "Outcome" not in data.columns:
        raise ValueError("Training data must contain Outcome")
    data = data.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Training data must be complete and finite")
    if set(data["Outcome"].unique()) != {0, 1}:
        raise ValueError("Outcome must contain both 0 and 1")
    predictors = data.drop(columns="Outcome")
    if predictors.shape[1] == 0:
        raise ValueError("No candidate predictors found")
    return predictors, data["Outcome"].astype(int)


def select_features(predictors, outcome):
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds = list(splitter.split(predictors))
    for training_indices, evaluation_indices in folds:
        if outcome.iloc[training_indices].nunique() != 2 or outcome.iloc[evaluation_indices].nunique() != 2:
            raise ValueError("Each cross-validation training and evaluation fold must contain both classes")
    ranking_model = RFE(
        estimator=RandomForestClassifier(random_state=RANDOM_SEED),
        n_features_to_select=1,
        step=1,
    ).fit(predictors, outcome)
    ranking = pd.DataFrame({
        "Feature": predictors.columns,
        "Ranking": ranking_model.ranking_,
    }).sort_values("Ranking").reset_index(drop=True)
    selected = []
    rows = []
    for feature in ranking["Feature"]:
        selected.append(feature)
        scores = []
        for training_indices, evaluation_indices in folds:
            model = RandomForestClassifier(random_state=RANDOM_SEED)
            model.fit(predictors.iloc[training_indices][selected], outcome.iloc[training_indices])
            probabilities = model.predict_proba(predictors.iloc[evaluation_indices][selected])[:, 1]
            scores.append(roc_auc_score(outcome.iloc[evaluation_indices], probabilities))
        model = RandomForestClassifier(random_state=RANDOM_SEED)
        model.fit(predictors[selected], outcome)
        rows.append({
            "Feature": feature,
            "Subset_Size": len(selected),
            "Step_Feature_Importance": model.feature_importances_[-1],
            "Mean_AUC": np.mean(scores),
            "Fold_AUC_SD": np.std(scores, ddof=1),
            **{f"Fold_{i + 1}_AUC": score for i, score in enumerate(scores)},
        })
    results = pd.DataFrame(rows)
    total_importance = results["Step_Feature_Importance"].sum()
    results["Normalized_Step_Importance"] = (
        results["Step_Feature_Importance"] / total_importance if total_importance else 0
    )
    results["Display_Feature"] = results["Feature"].map(lambda name: DISPLAY_NAMES.get(name, name))
    best_size = int(results.loc[results["Mean_AUC"].idxmax(), "Subset_Size"])
    results["Retained"] = results["Subset_Size"] <= best_size
    return ranking, results, best_size


def plot_selection(results, best_size):
    positions = np.arange(len(results))
    importance = results["Normalized_Step_Importance"].to_numpy()
    mean_auc = results["Mean_AUC"].to_numpy()
    colors = plt.cm.Blues(plt.Normalize(importance.min(), importance.max())(importance))
    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax1.bar(positions, importance, color=colors)
    ax1.set_xlabel("Features", fontsize=16)
    ax1.set_ylabel("Normalized step-specific importance", fontsize=16)
    ax1.set_xticks(positions, results["Display_Feature"], rotation=90, fontsize=12)
    for i, label in enumerate(ax1.get_xticklabels()):
        label.set_color("red" if i < best_size else "black")
    ax2 = ax1.twinx()
    ax2.plot(positions[:best_size], mean_auc[:best_size], color="red", marker="o", linewidth=1.5)
    if best_size < len(results):
        ax2.plot(positions[best_size - 1:], mean_auc[best_size - 1:], color="black", marker="o", linewidth=1.5)
    ax2.set_ylabel("Mean cross-validated AUC", fontsize=16)
    ax2.yaxis.set_major_formatter(lambda value, position: f"{value:.3f}")
    ax1.set_xlim(-0.6, len(results) - 0.4)
    ax1.set_title(f"RF-RFE: Selected {best_size}-Predictor Subset", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "RF_RFE.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    predictors, outcome = load_training_data()
    ranking, results, best_size = select_features(predictors, outcome)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUTPUT_DIR / "RF_RFE_ranking.csv", index=False, encoding="utf-8-sig")
    results.to_csv(OUTPUT_DIR / "RF_RFE_subset_scores.csv", index=False, encoding="utf-8-sig")
    results.loc[results["Retained"], ["Feature"]].to_csv(
        OUTPUT_DIR / "RF_RFE_selected_features.csv", index=False, encoding="utf-8-sig"
    )
    plot_selection(results, best_size)
    print(results.to_string(index=False))
    print(f"Selected subset size: {best_size}")
    print(f"Results saved to: {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    selection_results = main()
