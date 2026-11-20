"""Generate tables and vector-derived 600-dpi figures from saved predictions."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import shap

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_private"
OUT = ROOT / "outputs"
SPLITS = ("training", "validation", "test")
MODELS = ("Clinical", "Dose", "Interaction", "Clinical+Dose", "Clinical+Dose+Interaction")
LABELS = {
    "Clinical": "Clin", "Dose": "Dose", "Interaction": "Int",
    "Clinical+Dose": "Clin–Dose", "Clinical+Dose+Interaction": "Clin–Dose–Int",
}
COLORS = plt.get_cmap("tab10").colors


def table1(cohort: pd.DataFrame) -> None:
    variables = [
        ("Age, years", "Age", "numeric"), ("Male sex", "Gender", "binary"),
        ("Height, cm", "Height", "numeric"), ("Weight, kg", "Weight", "numeric"),
        ("Smoking (code 1)", "Smoking", "binary"), ("Alcohol (code 1)", "Alcohol", "binary"),
        ("Hypertension (code 1)", "Hypertension", "binary"),
        ("Diabetes (code 1)", "Diabetes", "binary"),
        ("Hemoglobin", "Hemoglobin", "numeric"), ("Albumin", "Albumin", "numeric"),
        ("Total protein", "TotalProtein", "numeric"), ("CEA", "CEA", "numeric"),
    ]
    rows = []
    for name, column, kind in variables:
        row = {"Characteristic": name}
        for split in SPLITS:
            part = cohort[cohort.cohort == split]
            neg = pd.to_numeric(part.loc[part.label == 0, column], errors="coerce").dropna()
            pos = pd.to_numeric(part.loc[part.label == 1, column], errors="coerce").dropna()
            if kind == "binary":
                def binary_text(values: pd.Series) -> str:
                    count = int((values == 1).sum())
                    return f"{count}/{len(values)} ({100 * count / len(values):.1f}%)" if len(values) else "NA"
                row[f"{split} Non-RP"] = binary_text(neg)
                row[f"{split} RP"] = binary_text(pos)
                table = [[int((pos == 1).sum()), int((pos == 0).sum())],
                         [int((neg == 1).sum()), int((neg == 0).sum())]]
                p = fisher_exact(table).pvalue
            else:
                def numeric_text(values: pd.Series) -> str:
                    return f"{values.median():.2f} [{values.quantile(.25):.2f}, {values.quantile(.75):.2f}]" if len(values) else "NA"
                row[f"{split} Non-RP"] = numeric_text(neg)
                row[f"{split} RP"] = numeric_text(pos)
                p = mannwhitneyu(pos, neg, alternative="two-sided").pvalue if len(pos) and len(neg) else np.nan
            row[f"{split} p"] = f"{p:.3f}" if np.isfinite(p) else "NA"
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "Table1_Patient_Characteristics.csv", index=False, encoding="utf-8-sig")


def table2() -> None:
    clinical = ["TotalProtein", "CEA"]
    dose = ["dose_original_glcm_Imc2", "dose_original_glrlm_RunLengthNonUniformity",
            "dose_original_glcm_MaximumProbability", "dose_original_glcm_InverseVariance"]
    rows = ([{"Feature set": "Clinical", "Feature": x, "Definition": "Original clinical variable"} for x in clinical]
            + [{"Feature set": "Dose", "Feature": x, "Definition": "PyRadiomics, whole-lung dose map"} for x in dose]
            + [{"Feature set": "Interaction", "Feature": f"sqrt__{c}__x__{d}",
                "Definition": f"sqrt({c} × {d})"} for c in clinical for d in dose])
    pd.DataFrame(rows).to_csv(OUT / "Table2_Selected_Features.csv", index=False, encoding="utf-8-sig")


def table3(metrics: pd.DataFrame) -> None:
    order = {name: i for i, name in enumerate(MODELS)}
    frame = metrics.copy()
    frame["AUC (95% CI)"] = frame.apply(lambda x: f"{x.auc:.3f} ({x.auc_ci_lower:.3f}–{x.auc_ci_upper:.3f})", axis=1)
    for key in ("accuracy", "precision", "recall_sensitivity", "specificity", "f1", "brier"):
        frame[key] = frame[key].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NA")
    frame["order"] = frame.feature_set.map(order)
    frame["split_order"] = frame.cohort.map({name: i for i, name in enumerate(SPLITS)})
    frame.sort_values(["split_order", "order"])[["cohort", "feature_set", "n", "predictor_n", "AUC (95% CI)",
        "accuracy", "precision", "recall_sensitivity", "specificity", "f1", "brier"]].to_csv(
            OUT / "Table3_Model_Performance.csv", index=False, encoding="utf-8-sig")


def figure2(pred: pd.DataFrame, destination: Path | None = None) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12.5})
    fig, axes = plt.subplots(2, 3, figsize=(8.3, 5.8))
    handles = []
    for j, split in enumerate(SPLITS):
        part = pred[pred.cohort == split]
        y = part.label.to_numpy(dtype=int)
        roc_ax, cal_ax = axes[:, j]
        for i, name in enumerate(MODELS):
            p = part[f"probability__{name}"].to_numpy()
            fpr, tpr, _ = roc_curve(y, p)
            # Join observed ROC threshold points with straight segments. This
            # retains the empirical coordinates and the reported AUC.
            line = roc_ax.plot(fpr, tpr, color=COLORS[i], lw=1.7,
                        marker="o" if split != "training" else None,
                        markersize=2.7, label=LABELS[name])[0]
            if j == 0:
                handles.append(line)
            frac, mean = calibration_curve(y, p, n_bins=4, strategy="quantile")
            cal_ax.plot(mean, frac, marker="o", ms=3.4, lw=1.5, color=COLORS[i], label=LABELS[name])
        roc_ax.plot([0,1],[0,1],"k--",lw=1)
        cal_ax.plot([0,1],[0,1],"k--",lw=1)
        for ax in (roc_ax,cal_ax):
            ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.grid(alpha=.15)
        roc_ax.set_title(split.title(), fontsize=13.5, fontweight="bold", pad=9)
        roc_ax.set_xlabel("False Positive Rate", fontsize=12)
        roc_ax.set_ylabel("True Positive Rate", fontsize=12)
        cal_ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        cal_ax.set_ylabel("Fraction of Positives", fontsize=12)
        roc_ax.tick_params(labelsize=11)
        cal_ax.tick_params(labelsize=11)
    fig.legend(handles,[LABELS[name] for name in MODELS],loc="lower center",ncol=3,
               fontsize=11.5,frameon=False,bbox_to_anchor=(.5,-.01))
    fig.subplots_adjust(left=.11,right=.99,top=.88,bottom=.22,wspace=.54,hspace=.72)
    for row, letters in enumerate(("abc", "def")):
        for col, letter in enumerate(letters):
            box = axes[row, col].get_position()
            fig.text(box.x0-.037, box.y1+.015, letter,
                     fontsize=14, fontweight="bold", ha="left", va="bottom")
    if destination is None:
        fig.savefig(OUT / "Fig2_ROC_Calibration.tiff", dpi=600, bbox_inches="tight")
        fig.savefig(OUT / "Fig2_ROC_Calibration.png", dpi=300, bbox_inches="tight")
    else:
        save_tiff(fig, destination)
    plt.close(fig)


def figure3(cohort: pd.DataFrame, submission_dir: Path | None = None) -> None:
    clinical = pd.read_csv(DATA / "selected_clinical_features.csv", encoding="utf-8-sig").set_index("patient_id")
    dose = pd.read_csv(DATA / "selected_dose_radiomics.csv", encoding="utf-8-sig").set_index("patient_id")
    from model_analysis import make_interactions
    clinical = clinical.loc[cohort.patient_id].drop(columns="label").reset_index(drop=True)
    dose = dose.loc[cohort.patient_id].drop(columns="label").reset_index(drop=True)
    interaction = make_interactions(clinical, dose)
    x = pd.concat([clinical,dose,interaction],axis=1)
    model = RandomForestClassifier(n_estimators=500,max_depth=5,min_samples_leaf=2,
                                    class_weight="balanced",n_jobs=-1,random_state=20261008)
    train = cohort.cohort.eq("training").to_numpy()
    model.fit(x.loc[train],cohort.loc[train,"label"])
    # For a sklearn random forest, TreeExplainer's raw output is the class
    # probability. Check the additive identity before interpreting it.
    x_train = x.loc[train]
    explainer = shap.TreeExplainer(model, model_output="raw")
    values = explainer.shap_values(x_train)
    positive = values[:, :, 1] if isinstance(values, np.ndarray) and values.ndim == 3 else values[1]
    base = float(np.asarray(explainer.expected_value).reshape(-1)[1])
    risk = model.predict_proba(x_train)[:, 1]
    if not np.allclose(base + positive.sum(axis=1), risk, atol=1e-6, rtol=0):
        raise AssertionError("Positive-class SHAP values do not sum to predicted RP probability")
    means = np.abs(positive).mean(axis=0)
    order = np.argsort(means)[::-1]
    short = {"TotalProtein":"TP", "CEA":"CEA",
             "dose_original_glcm_Imc2":"Imc2",
             "dose_original_glrlm_RunLengthNonUniformity":"RLNU",
             "dose_original_glcm_MaximumProbability":"MaxProb",
             "dose_original_glcm_InverseVariance":"InvVar"}
    def display(name: str) -> str:
        if name.startswith("sqrt__"):
            left, right = name[6:].split("__x__")
            return rf"$\sqrt{{\mathrm{{{short[left]}}}\,\times\,\mathrm{{{short[right]}}}}}$"
        return short[name]
    top = order[:8][::-1]
    if submission_dir is None:
        fig, (bar_ax, scatter_ax) = plt.subplots(1, 2, figsize=(8.2, 4.7))
    else:
        fig, grid = plt.subplots(3, 2, figsize=(8.8, 8.6), layout="constrained")
        fig.get_layout_engine().set(w_pad=.12, h_pad=.12, hspace=.12, wspace=.06)
        bar_ax, scatter_ax = grid[0]
    yy = np.arange(len(top))
    names = [display(x.columns[k]) for k in top]
    bar_ax.barh(yy, means[top], color="#3278A8", height=.65)
    bar_ax.set_yticks(yy, names, fontsize=11)
    bar_ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    bar_ax.set_title("Global importance", fontsize=13, weight="bold")
    bar_ax.tick_params(axis="x", labelsize=10.5)
    bar_ax.text(-.20, 1.07, "a", transform=bar_ax.transAxes, fontsize=15, weight="bold")

    jitter = np.random.default_rng(20261008)
    for row, k in enumerate(top):
        raw = x_train.iloc[:, k].to_numpy(dtype=float)
        # Within-feature percentile makes the red/blue direction explicit;
        # values from different clinical and dose units are not pooled.
        percentiles = pd.Series(raw).rank(pct=True).to_numpy()
        scatter_ax.scatter(positive[:, k], row + jitter.uniform(-.22, .22, len(raw)),
                           c=percentiles, cmap="coolwarm", vmin=0, vmax=1,
                           s=19, alpha=.85, linewidths=0, rasterized=True)
    scatter_ax.axvline(0, color="#555555", lw=1)
    scatter_ax.set_yticks(yy, names, fontsize=11)
    scatter_ax.set_xlabel("SHAP value for RP probability", fontsize=12)
    scatter_ax.set_title("Feature-value direction", fontsize=13, weight="bold")
    scatter_ax.tick_params(axis="x", labelsize=10.5)
    scatter_ax.text(-.20, 1.07, "b", transform=scatter_ax.transAxes, fontsize=15, weight="bold")
    mapper = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="coolwarm")
    colorbar = fig.colorbar(mapper, ax=scatter_ax, fraction=.045, pad=.03)
    colorbar.set_label("Within-feature percentile", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    if submission_dir is None:
        fig.subplots_adjust(left=.19, right=.96, top=.84, bottom=.16, wspace=.64)
        fig.savefig(OUT / "Fig3_SHAP_Three_Component_Model.tiff", dpi=600, bbox_inches="tight")
        fig.savefig(OUT / "Fig3_SHAP_Three_Component_Model.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Choose examples solely by fitted probability rank (25th/75th
    # percentiles), never by the observed RP label or correctness.
    ranking = np.argsort(risk, kind="stable")
    examples = [ranking[round(.25 * (len(ranking) - 1))],
                ranking[round(.75 * (len(ranking) - 1))]]
    if submission_dir is None:
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.6))
    else:
        axes = grid[1]
    for panel, (ax, patient_row) in enumerate(zip(axes, examples)):
        contributions = positive[patient_row]
        indices = np.argsort(np.abs(contributions))[::-1][:5]
        remainder = float(contributions.sum() - contributions[indices].sum())
        deltas = [float(contributions[k]) for k in indices] + [remainder]
        labels = [display(x.columns[k]) for k in indices] + ["Other 9 features"]
        current = base
        path = [current]
        for row, delta in enumerate(deltas):
            next_value = current + delta
            ax.barh(row, abs(delta), left=min(current, next_value), height=.62,
                    color="#D94A58" if delta >= 0 else "#3278A8", edgecolor="white")
            current = next_value
            path.append(current)
        ax.axvline(base, color="#777777", linestyle="--", lw=1.2)
        ax.axvline(risk[patient_row], color="black", lw=1.2)
        ax.set_xlim(min(0, min(path)-.03), max(1, max(path)+.03))
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel("RP probability", fontsize=12)
        ax.set_title(("Lower" if panel == 0 else "Higher") +
                     f" fitted risk: {risk[patient_row]:.2f}", fontsize=13, weight="bold")
        ax.tick_params(axis="x", labelsize=10.5)
        ax.text(-.20, 1.07, ("ab" if submission_dir is None else "cd")[panel], transform=ax.transAxes,
                fontsize=15, weight="bold")
    if submission_dir is None:
        fig.subplots_adjust(left=.19, right=.98, top=.84, bottom=.16, wspace=.66)
        fig.savefig(OUT / "Fig4_SHAP_Individual_Examples.tiff", dpi=600, bbox_inches="tight")
        fig.savefig(OUT / "Fig4_SHAP_Individual_Examples.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Figure 5: dependence plots for two clinical–dose products.  Percentile
    # axes avoid squashing most points when a source variable is skewed.
    interaction_names = (
        "sqrt__TotalProtein__x__dose_original_glcm_Imc2",
        "sqrt__CEA__x__dose_original_glcm_Imc2",
    )
    clinical_names = ("TotalProtein", "CEA")
    if submission_dir is None:
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    else:
        axes = grid[2]
    for panel, (ax, interaction_name, clinical_name) in enumerate(zip(
            axes, interaction_names, clinical_names)):
        feature_idx = x.columns.get_loc(interaction_name)
        value_rank = x_train.iloc[:, feature_idx].rank(pct=True).to_numpy()
        clinical_rank = x_train[clinical_name].rank(pct=True).to_numpy()
        ax.scatter(value_rank, positive[:, feature_idx],
                   c=clinical_rank, cmap="coolwarm", vmin=0, vmax=1,
                   s=38, alpha=.8, linewidths=.25, edgecolors="white", rasterized=True)
        ax.axhline(0, color="#555555", lw=1, linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Interaction value percentile", fontsize=12)
        ax.set_ylabel("SHAP value for RP probability", fontsize=12)
        ax.set_title(display(interaction_name), fontsize=13, weight="bold")
        ax.tick_params(labelsize=10.5)
        ax.text(-.17, 1.08, ("ab" if submission_dir is None else "ef")[panel], transform=ax.transAxes,
                fontsize=15, weight="bold")
    mapper = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="coolwarm")
    colorbar = fig.colorbar(mapper, ax=axes if submission_dir is None else axes[1], fraction=.04, pad=.025)
    colorbar.set_label("Clinical component percentile", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    if submission_dir is None:
        fig.subplots_adjust(left=.11, right=.87, top=.80, bottom=.23, wspace=.42)
        fig.savefig(OUT / "Fig5_SHAP_Dependence.tiff", dpi=600, bbox_inches="tight")
        fig.savefig(OUT / "Fig5_SHAP_Dependence.png", dpi=300, bbox_inches="tight")
    else:
        save_tiff(fig, submission_dir / "Fig4.tiff")
    plt.close(fig)

    # Figure 6: signed Spearman correlations among the six highest-ranked
    # predictors, including deterministic products. This documents redundancy.
    leading = order[:6]
    correlations = x_train.iloc[:, leading].corr(method="spearman").to_numpy()
    if not np.allclose(correlations, correlations.T, atol=1e-12):
        raise AssertionError("Feature correlation matrix is not symmetric")
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    matrix = ax.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
    names = [display(x.columns[k]) for k in leading]
    ax.set_xticks(np.arange(len(leading)), names, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(np.arange(len(leading)), names, fontsize=11)
    for row in range(len(leading)):
        for col in range(len(leading)):
            value = correlations[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                    color="white" if abs(value) >= .65 else "#222222", fontsize=10)
    ax.set_title("Correlation among leading predictors", fontsize=13, weight="bold")
    colorbar = fig.colorbar(matrix, ax=ax, fraction=.045, pad=.025)
    colorbar.set_label("Spearman correlation", fontsize=11)
    colorbar.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=.26, right=.88, top=.88, bottom=.24)
    if submission_dir is None:
        fig.savefig(OUT / "Fig6_Feature_Correlations.tiff", dpi=600, bbox_inches="tight")
        fig.savefig(OUT / "Fig6_Feature_Correlations.png", dpi=300, bbox_inches="tight")
    else:
        save_tiff(fig, submission_dir / "FigS1.tiff")
    plt.close(fig)


def save_tiff(fig, destination: Path, dpi: int = 600) -> None:
    fig.savefig(destination, dpi=dpi, bbox_inches="tight", pad_inches=.12,
                pil_kwargs={"compression": "tiff_lzw"})


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cohort = pd.read_csv(DATA / "cohort.csv",encoding="utf-8-sig")
    pred = pd.read_csv(OUT / "patient_predictions_PRIVATE.csv",encoding="utf-8-sig")
    metrics = pd.read_csv(OUT / "five_models.csv",encoding="utf-8-sig")
    assert cohort.patient_id.tolist() == pred.patient_id.tolist()
    table1(cohort); table2(); table3(metrics)
    figure2(pred); figure3(cohort)
    print("Created Table1–3 and model figures in",OUT)


if __name__ == "__main__":
    main()
