from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests


INPUT_FILE = Path("traindata.csv")
INPUT_ENCODING = "GBK"
RFE_INPUT_FILE = Path("cleaned_data.csv")
RFE_INPUT_ENCODING = "GBK"
OUTPUT_DIR = Path("results")
CONTINUOUS_START_COLUMN = 10
FDR_THRESHOLD = 0.05
SEED = 123
N_SPLITS = 5
HIGHLIGHT_COUNT = 5
DISPLAY_NAMES = {
    "CNS_symptoms": "CNS symptoms",
    "Onset_to_admission_interval": "Admission delay",
    "SFTSV_RNA": "SFTSV RNA load",
    "SFTSV_RNA_load": "SFTSV RNA load",
    "IL_2": "IL-2",
    "IL_4": "IL-4",
    "IL_6": "IL-6",
    "IL_10": "IL-10",
    "IL_17A": "IL-17A",
    "IFN_gamma": "IFN-\u03b3",
    "TNF_alpha": "TNF-\u03b1",
    "TGF_beta": "TGF-\u03b2",
    "CD3.CD4": "CD3+CD4+ T-cells",
    "CD3.CD4.": "CD3+CD4+ T-cells",
    "CD3.CD8": "CD3+CD8+ T-cells",
    "CD3.CD8.": "CD3+CD8+ T-cells",
    "CD3.CD19": "CD3-CD19+ B-cells",
    "CD3.CD19.": "CD3-CD19+ B-cells",
    "CD3.CD56": "CD3-CD56+ NK cells",
}


def display_name(name):
    key = str(name).strip()
    return DISPLAY_NAMES.get(key, key)


def calculate_correlations(data):
    values = data.to_numpy(dtype=float)
    count = values.shape[1]
    correlations = np.eye(count)
    raw_p_values = np.zeros((count, count))
    adjusted_p_values = np.zeros((count, count))
    pairs = []
    p_values = []
    for i in range(count):
        for j in range(i + 1, count):
            coefficient, p_value = stats.pearsonr(values[:, i], values[:, j])
            if not np.isfinite(coefficient) or not np.isfinite(p_value):
                raise ValueError(f"Invalid correlation for {data.columns[i]} and {data.columns[j]}.")
            correlations[i, j] = correlations[j, i] = coefficient
            raw_p_values[i, j] = raw_p_values[j, i] = p_value
            pairs.append((i, j))
            p_values.append(p_value)
    corrected = multipletests(p_values, alpha=FDR_THRESHOLD, method="fdr_bh")[1]
    for (i, j), p_value in zip(pairs, corrected):
        adjusted_p_values[i, j] = adjusted_p_values[j, i] = p_value
    return correlations, raw_p_values, adjusted_p_values


def plot_heatmap(correlations, columns, output_path):
    labels = [display_name(name) for name in columns]
    count = len(labels)
    frame = pd.DataFrame(correlations, index=labels, columns=labels)
    mask = np.triu(np.ones_like(correlations, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.15, right=0.80, top=0.93, bottom=0.07)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="white")
    sns.heatmap(frame, mask=mask, annot=False, cmap=cmap, vmin=-1, vmax=1,
                center=0, square=True, linewidths=0, cbar=False, ax=ax)
    ax.set_xticks([])
    ax.set_yticks(np.arange(1, count) + 0.5)
    ax.set_yticklabels(labels[1:], rotation=0, va="center", ha="right", fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(0, count)
    ax.set_ylim(count, 0)
    colorbar_ax = fig.add_axes([0.73, 0.28, 0.02, 0.5])
    colorbar = fig.colorbar(ax.collections[0], cax=colorbar_ax,
                            orientation="vertical", ticks=np.arange(-1.0, 1.1, 0.2))
    colorbar.ax.tick_params(labelsize=8)
    colorbar.set_label("Pearson correlation coefficient", size=9)
    for spine in colorbar_ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    for i in range(count - 1):
        ax.text(i + 0.08, i + 0.5, labels[i], ha="left", va="center",
                fontsize=7, rotation=0, color="black", weight="normal")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_rf_rfe(predictors, outcome):
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(predictors))
    for train_index, validation_index in splits:
        if outcome.iloc[train_index].nunique() < 2 or outcome.iloc[validation_index].nunique() < 2:
            raise ValueError("A KFold partition contains only one outcome class; review the split without silently changing CV.")
    estimator = RandomForestClassifier(random_state=SEED)
    rfe = RFE(estimator=estimator, n_features_to_select=1, step=1)
    rfe.fit(predictors, outcome)
    ranking = pd.DataFrame({"Feature": predictors.columns, "Ranking": rfe.ranking_}).sort_values("Ranking")
    ranking.to_csv(OUTPUT_DIR / "rf_rfe_ranking.csv", index=False)
    selected = []
    records = []
    for feature in ranking["Feature"]:
        selected.append(feature)
        scores = []
        for train_index, validation_index in splits:
            estimator.fit(predictors.iloc[train_index][selected], outcome.iloc[train_index])
            probabilities = estimator.predict_proba(predictors.iloc[validation_index][selected])[:, 1]
            scores.append(roc_auc_score(outcome.iloc[validation_index], probabilities))
        estimator.fit(predictors[selected], outcome)
        mean_auc = float(np.mean(scores))
        half_width = stats.t.ppf(0.975, df=len(scores) - 1) * stats.sem(scores)
        record = {
            "Feature": feature, "Subset_Size": len(selected),
            "Added_Feature_Subset_Importance": estimator.feature_importances_[-1],
            "Mean_ROC": mean_auc, "Fold_T_Lower": mean_auc - half_width,
            "Fold_T_Upper": mean_auc + half_width,
        }
        record.update({f"Fold_{i + 1}_ROC": score for i, score in enumerate(scores)})
        records.append(record)
    results = pd.DataFrame(records)
    total = results["Added_Feature_Subset_Importance"].sum()
    if total <= 0:
        raise ValueError("The sum of added-feature importances is zero.")
    results["Normalized_Added_Feature_Subset_Importance"] = results["Added_Feature_Subset_Importance"] / total
    results["Display_Feature"] = results["Feature"].map(display_name)
    results.to_csv(OUTPUT_DIR / "rf_rfe_subset_results.csv", index=False)
    return results


def plot_rf_rfe(results, output_path):
    highlighted = min(HIGHLIGHT_COUNT, len(results))
    positions = np.arange(len(results))
    importance = results["Normalized_Added_Feature_Subset_Importance"].to_numpy()
    mean_auc = results["Mean_ROC"].to_numpy()
    norm = plt.Normalize(importance.min(), importance.max())
    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax1.bar(positions, importance, color=plt.cm.Blues(norm(importance)))
    ax1.set_xlabel("Features", fontsize=18, fontweight="bold")
    ax1.set_ylabel("Normalized added-feature importance\nwithin cumulative subsets", fontsize=14, fontweight="bold")
    ax1.set_xticks(positions)
    ax1.set_xticklabels(results["Display_Feature"], rotation=90, ha="center", fontsize=12)
    ax1.tick_params(axis="y", labelsize=12)
    for i, label in enumerate(ax1.get_xticklabels()):
        label.set_color("red" if i < highlighted else "black")
    ax2 = ax1.twinx()
    ax2.plot(positions[:highlighted], mean_auc[:highlighted], color="red", marker="o", linewidth=1.5, markersize=5)
    if highlighted < len(results):
        ax2.plot(positions[highlighted - 1:], mean_auc[highlighted - 1:], color="black", marker="o", linewidth=1.5, markersize=5)
    ax2.set_ylabel("Mean AUC", fontsize=18, fontweight="bold")
    ax2.tick_params(axis="y", labelsize=12)
    ax2.yaxis.set_major_formatter(lambda value, position: f"{value:.3f}")
    ax1.set_xlim(-0.6, len(results) - 0.4)
    ax1.set_title(f"Cumulative Subset AUC (Top {highlighted} Features Highlighted)", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "Times New Roman"
    train = pd.read_csv(INPUT_FILE, encoding=INPUT_ENCODING)
    if "Outcome" not in train.columns:
        raise ValueError("Outcome is missing from the training data.")
    outcome = train["Outcome"]
    if outcome.isna().any() or set(outcome.unique()) != {0, 1}:
        raise ValueError("Outcome must contain both 0 = survival and 1 = in-hospital death, without missing values.")
    predictors = train.iloc[:, CONTINUOUS_START_COLUMN:].drop(columns=["Outcome"], errors="ignore")
    if predictors.shape[1] < 2:
        raise ValueError("At least two numeric predictors are required; review CONTINUOUS_START_COLUMN.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in predictors.dtypes):
        raise ValueError("All selected predictors must be numeric; review the column selection.")
    if not np.isfinite(predictors.to_numpy(dtype=float)).all():
        raise ValueError("Predictors contain missing or infinite values.")
    constant = predictors.columns[predictors.nunique() < 2].tolist()
    if constant:
        raise ValueError(f"Constant predictors require review: {constant}")
    correlations, raw_p_values, adjusted_p_values = calculate_correlations(predictors)
    for filename, values in [("pearson_correlations.csv", correlations), ("raw_p_values.csv", raw_p_values),
                             ("fdr_p_values.csv", adjusted_p_values)]:
        pd.DataFrame(values, index=predictors.columns, columns=predictors.columns).to_csv(OUTPUT_DIR / filename)
    plot_heatmap(correlations, predictors.columns, OUTPUT_DIR / "correlation_heatmap.pdf")
    if not RFE_INPUT_FILE.is_file():
        raise FileNotFoundError(f"Provide the manually screened training data: {RFE_INPUT_FILE}")
    cleaned = pd.read_csv(RFE_INPUT_FILE, encoding=RFE_INPUT_ENCODING)
    if "Outcome" not in cleaned.columns:
        raise ValueError("The manually screened training data must include Outcome.")
    rfe_outcome = cleaned["Outcome"]
    if rfe_outcome.isna().any() or set(rfe_outcome.unique()) != {0, 1}:
        raise ValueError("Outcome in the manually screened training data must contain both 0 and 1, without missing values.")
    rfe_predictors = cleaned.drop(columns=["Outcome"])
    if rfe_predictors.shape[1] < 1:
        raise ValueError("The manually screened training data contains no predictors.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in rfe_predictors.dtypes):
        raise ValueError("All manually screened predictors must be numeric.")
    if not np.isfinite(rfe_predictors.to_numpy(dtype=float)).all():
        raise ValueError("The manually screened predictors contain missing or infinite values.")
    constant = rfe_predictors.columns[rfe_predictors.nunique() < 2].tolist()
    if constant:
        raise ValueError(f"Constant manually screened predictors require review: {constant}")
    results = run_rf_rfe(rfe_predictors, rfe_outcome)
    plot_rf_rfe(results, OUTPUT_DIR / "rf_rfe.pdf")
    print(f"Predictors supplied in the manually screened training data: {rfe_predictors.shape[1]}")
    print(f"Plot highlight count: {min(HIGHLIGHT_COUNT, len(results))}; no automatic subset selection was applied.")
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()