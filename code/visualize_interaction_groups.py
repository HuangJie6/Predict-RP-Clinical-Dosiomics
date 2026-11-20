"""Describe four TP/CEA-by-Imc2 strata using existing, frozen predictions.

Uses saved predictions, fixed cut points, and existing cohort assignments.
Example: python visualize_interaction_groups.py
All cut points are medians of the original 108 training patients.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
IMC2 = "dose_original_glcm_Imc2"
SPLITS = ("training", "validation", "test")
TITLES = ("Training Cohort", "Validation Cohort", "Internal Test Cohort")
COLORS = ("#4477AA", "#66AABB", "#CCBB44", "#AA3377")
GROUPS = ("Low/Low", "Low/High", "High/Low", "High/High")
MODELS = {"combined": "Clinical+Dose+Interaction", "interaction": "Interaction"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(events: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf(0.975)
    p = events / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return max(0., center-half), min(1., center+half)


def load_data(root: Path, model: str):
    paths = {
        "cohort": root / "data_private/cohort.csv",
        "clinical": root / "data_private/selected_clinical_features.csv",
        "dose": root / "data_private/selected_dose_radiomics.csv",
        "predictions": root / "outputs/patient_predictions_PRIVATE.csv",
        "metrics": root / "outputs/five_models.csv",
    }
    hashes = {key: sha256(path) for key, path in paths.items()}
    cohort = pd.read_csv(paths["cohort"])
    if len(cohort) != 155 or cohort.patient_id.duplicated().any():
        raise ValueError("Expected the fixed 155-patient cohort with unique IDs.")
    if cohort.cohort.value_counts().to_dict() != {"training": 108, "test": 24, "validation": 23}:
        raise ValueError("The original cohort allocation has changed.")
    if cohort.groupby("cohort").label.sum().to_dict() != {"training": 48, "validation": 10, "test": 11}:
        raise ValueError("The expected outcomes do not match.")
    data = cohort[["patient_id", "cohort", "label"]].set_index("patient_id")
    for key in ("clinical", "dose", "predictions"):
        frame = pd.read_csv(paths[key])
        if frame.patient_id.duplicated().any() or set(frame.patient_id) != set(data.index):
            raise ValueError(f"Patient IDs do not match: {key}")
        frame = frame.set_index("patient_id").loc[data.index]
        for column in ("label", "cohort"):
            if column in frame and not np.array_equal(frame[column], data[column]):
                raise ValueError(f"{column} differs between cohort and {key}.")
        data = data.join(frame.drop(columns=["cohort", "label"], errors="ignore"), validate="one_to_one")
    probability = f"probability__{model}"
    required = ["TotalProtein", "CEA", IMC2, probability]
    if not np.isfinite(data[required].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite source values; do not silently remove patients.")
    if not data[probability].between(0, 1).all():
        raise ValueError("Invalid model probabilities.")
    train = data.cohort.eq("training")
    cutoffs = data.loc[train, ["TotalProtein", "CEA", IMC2]].median().to_dict()
    metrics = pd.read_csv(paths["metrics"])
    for split in SPLITS:
        subset = data.loc[data.cohort.eq(split)]
        auc = roc_auc_score(subset.label, subset[probability])
        reference = metrics.loc[metrics.feature_set.eq(model) & metrics.cohort.eq(split), "auc"]
        if len(reference) != 1 or not np.isclose(auc, reference.iloc[0], atol=1e-12):
            raise ValueError(f"Stored AUC and patient predictions disagree: {split}.")
    return data, cutoffs, probability, paths, hashes


def summarize(data, clinical, cutoffs, probability):
    grouped = data.copy()
    # >= median is high. This tie rule is fixed for every cohort.
    grouped["stratum"] = (grouped[clinical].ge(cutoffs[clinical]).astype(int) * 2
                            + grouped[IMC2].ge(cutoffs[IMC2]).astype(int))
    rows = []
    for split in SPLITS:
        for index, name in enumerate(GROUPS):
            sub = grouped.loc[grouped.cohort.eq(split) & grouped.stratum.eq(index)]
            n = len(sub)
            events = int(sub.label.sum())
            lo, hi = wilson(events, n)
            scores = sub[probability]
            rows.append({
                "clinical_feature": clinical, "cohort": split, "group": name,
                "group_index": index, "n": n, "rp_n": events,
                "mean_predicted_probability": float(scores.mean()),
                "median_predicted_probability": float(scores.median()),
                "probability_q25": float(scores.quantile(.25)),
                "probability_q75": float(scores.quantile(.75)),
                "observed_rp_proportion": events/n if n else np.nan,
                "observed_rp_ci_lower": lo, "observed_rp_ci_upper": hi,
                "clinical_training_median": cutoffs[clinical],
                "imc2_training_median": cutoffs[IMC2],
                "prediction_model": probability.removeprefix("probability__"),
            })
    result = pd.DataFrame(rows)
    assert result.n.sum() == 155 and result.rp_n.sum() == 69
    return grouped, result


def plot_strata(data, summary, clinical, probability, out):
    short = {"TotalProtein": "TP", "CEA": "CEA"}[clinical]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.titlesize": 16, "axes.labelsize": 14,
        "xtick.labelsize": 11, "ytick.labelsize": 12,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.7))
    fig.subplots_adjust(left=.075, right=.985, bottom=.105, top=.93,
                        wspace=.31, hspace=.42)
    ticklabels = [f"{short} {a.lower()}\nImc2 {b.lower()}"
                  for a, b in (item.split("/") for item in GROUPS)]
    rng = np.random.default_rng(20261111)  # Display jitter only.
    for col, (split, title) in enumerate(zip(SPLITS, TITLES)):
        part = data.loc[data.cohort.eq(split)]
        stats = summary.loc[summary.cohort.eq(split)].set_index("group_index")
        upper, lower = axes[:, col]
        upper.set_title(f"{title} (n = {len(part)})", pad=15)
        for index in range(4):
            s = stats.loc[index]
            scores = part.loc[part.stratum.eq(index), probability].to_numpy()
            if len(scores):
                boxes = upper.boxplot([scores], positions=[index], widths=.5,
                                      patch_artist=True, showfliers=False,
                                      medianprops={"color": "black", "linewidth": 1.4},
                                      whiskerprops={"color": COLORS[index]},
                                      capprops={"color": COLORS[index]})
                boxes["boxes"][0].set(facecolor=COLORS[index], alpha=.20,
                                      edgecolor=COLORS[index])
                upper.scatter(index+rng.uniform(-.14, .14, len(scores)), scores,
                              s=26, color=COLORS[index], alpha=.76,
                              linewidths=.4, edgecolors="white", zorder=3)
                upper.scatter(index, s.mean_predicted_probability, marker="D",
                              s=47, facecolor="white", edgecolor="black", zorder=5)
                upper.text(index, 1.06, f"{s.mean_predicted_probability:.1%}\nn = {int(s.n)}",
                           ha="center", va="bottom", fontsize=10.5)
                value = s.observed_rp_proportion
                lower.errorbar(index, value,
                               yerr=[[value-s.observed_rp_ci_lower], [s.observed_rp_ci_upper-value]],
                               fmt="o", ms=8, color=COLORS[index], ecolor=COLORS[index],
                               capsize=5, elinewidth=2, zorder=3)
                lower.text(index, 1.06, f"{int(s.rp_n)}/{int(s.n)}",
                           ha="center", va="bottom", fontsize=12)
            else:
                upper.text(index, .55, "No patients", rotation=90, ha="center")
                lower.text(index, .55, "No patients", rotation=90, ha="center")
        for row, ax in enumerate((upper, lower)):
            ax.set_xlim(-.55, 3.55)
            ax.set_ylim(-.035, 1.20)
            ax.set_xticks(range(4), ticklabels)
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            ax.grid(axis="y", color="#dddddd", linewidth=.65, alpha=.65)
            ax.set_axisbelow(True)
            ax.text(-.17, 1.045, chr(ord("a")+row*3+col), transform=ax.transAxes,
                    fontsize=20, fontweight="bold", va="top", ha="left", clip_on=False)
        if col == 0:
            upper.set_ylabel("Predicted RP Probability")
            lower.set_ylabel("Observed RP Proportion")
    for extension in ("png", "pdf", "tiff"):
        kwargs = {"dpi": 400, "facecolor": "white", "bbox_inches": "tight"}
        if extension == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(out / f"Figure_{short}_Imc2_Four_Groups.{extension}", **kwargs)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--model", choices=list(MODELS), default="combined")
    args = parser.parse_args()
    root = args.project.resolve()
    out = root / "outputs" / "interaction_stratification" / args.model
    data, cutoffs, probability, paths, before = load_data(root, MODELS[args.model])
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for clinical in ("TotalProtein", "CEA"):
        grouped, summary = summarize(data, clinical, cutoffs, probability)
        results.append(summary)
        plot_strata(grouped, summary, clinical, probability, out)
    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(out / "Four_Group_Summary.csv", index=False, encoding="utf-8-sig")
    after = {key: sha256(path) for key, path in paths.items()}
    if before != after:
        raise RuntimeError("An input file changed during analysis; review before using outputs.")
    audit = {
        "analysis": "Exploratory stratification of existing fixed-cohort model predictions",
        "model": MODELS[args.model], "training_medians": cutoffs,
        "group_definition": "low < training median; high >= training median",
        "cohorts": {"training": 108, "validation": 23, "test": 24},
        "interval_method": "95% Wilson binomial CI for observed RP proportions",
        "boxplot": "median, 25th/75th percentiles, whiskers at 1.5 IQR; all patients shown as dots; diamond = mean",
        "source_sha256": before, "source_files_unchanged": before == after,
        "no_training_or_split_changes": True,
        "interpretation": [
            "Training predictions are fitted in-sample probabilities, not out-of-fold predictions.",
            "Groups describe observed patients with different other predictors; effects are unadjusted.",
            "Cutoffs are sample-relative, not clinical reference limits.",
            "These plots do not isolate the effect of the product feature or prove biological synergy.",
            "Observed proportions describe the sampled cohort, not source-population incidence.",
            "No hypothesis tests or significance claims are produced.",
        ],
    }
    (out / "analysis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    notes = f"""Four-group visualization
Model: {MODELS[args.model]}
Data: existing 155 cases, fixed training/validation/test allocation (108/23/24).
The analysis uses the existing model predictions and cohort assignments.

Training cutoffs (exact values used):
TP: {cutoffs['TotalProtein']:.12g}
CEA: {cutoffs['CEA']:.12g}
Imc2: {cutoffs[IMC2]:.15g}
Low is strictly below the training median; high is at or above it.

Panels a-c: existing model predictions in training, validation and internal test cohorts.
Dots show individual patients; boxes show median and IQR; diamonds show the mean.
Numbers above boxes are mean probability and sample size. Jitter is display-only.
Panels d-f: observed RP proportions with 95% Wilson confidence intervals.
Labels show RP cases/total patients in each stratum. Empty strata are kept visible.

TP is total protein concentration, not target-protein expression. Imc2 is a whole-lung
dose GLCM feature, not a molecular-expression measurement. High/low are relative
strata, not clinical abnormality thresholds. Four-group comparisons are descriptive
and do not establish biological synergy or isolate a product-feature effect.
The positive/negative case sampling means observed proportions are not estimates
of RP incidence in the original clinical population.
"""
    (out / "README.txt").write_text(notes, encoding="utf-8")
    print(json.dumps(cutoffs, indent=2))
    print(combined[["clinical_feature", "cohort", "group", "n", "rp_n",
                    "mean_predicted_probability", "observed_rp_proportion"]].to_string(index=False))
    print(f"Saved figures and aggregate results to: {out}")
    print("Verified: all source file hashes unchanged.")


if __name__ == "__main__":
    main()
