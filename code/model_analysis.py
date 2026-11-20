"""Five fixed random-forest feature sets with eight square-root interactions.

This exploratory experiment reuses the 155 reviewed cases, frozen 108/23/24
split, training-only selected original six predictors, and original RF settings.
Nothing is selected, tuned, or excluded using validation/test outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
COHORT = ROOT / "data_private" / "cohort.csv"
CLINICAL_PATH = ROOT / "data_private" / "selected_clinical_features.csv"
DOSE_PATH = ROOT / "data_private" / "selected_dose_radiomics.csv"
SEED = 20261008
SPLITS = ("training", "validation", "test")
SCHEMES = {
    "Clinical": ("clinical",),
    "Dose": ("dose",),
    "Interaction": ("interaction",),
    "Clinical+Dose": ("clinical", "dose"),
    "Clinical+Dose+Interaction": ("clinical", "dose", "interaction"),
}


def read_cohort() -> pd.DataFrame:
    cohort = pd.read_csv(COHORT, encoding="utf-8-sig")
    if len(cohort) != 155 or cohort.patient_id.duplicated().any():
        raise ValueError("The reviewed 155-patient cohort has changed")
    if cohort.cohort.value_counts().to_dict() != {"training": 108, "validation": 23, "test": 24}:
        raise ValueError("The frozen split has changed")
    if cohort.label.value_counts().to_dict() != {0: 86, 1: 69}:
        raise ValueError("The reviewed labels have changed")
    return cohort


def read_selected(path: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig")
    if data.patient_id.duplicated().any() or set(data.patient_id) != set(cohort.patient_id):
        raise ValueError(f"Patient IDs do not match the cohort: {path}")
    data = data.set_index("patient_id").loc[cohort.patient_id]
    if "label" in data:
        if not np.array_equal(data.label.to_numpy(dtype=int), cohort.label.to_numpy(dtype=int)):
            raise ValueError(f"Labels do not match the cohort: {path}")
        data = data.drop(columns="label")
    if data.columns.duplicated().any():
        raise ValueError(f"Duplicate predictor names: {path}")
    data = data.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError(f"The original selected predictors contain missing/non-finite values: {path}")
    data.index = cohort.index
    return data


def make_interactions(clinical: pd.DataFrame, dose: pd.DataFrame) -> pd.DataFrame:
    if list(clinical) != ["TotalProtein", "CEA"] or list(dose) != [
        "dose_original_glcm_Imc2", "dose_original_glrlm_RunLengthNonUniformity",
        "dose_original_glcm_MaximumProbability", "dose_original_glcm_InverseVariance",
    ]:
        raise ValueError("Expected the original two clinical and four dose predictors")
    if (clinical.to_numpy() < 0).any() or (dose.to_numpy() < 0).any():
        raise ValueError("sqrt(clinical*dose) requires nonnegative source values")
    columns = {}
    for clinical_name in clinical:
        for dose_name in dose:
            name = f"sqrt__{clinical_name}__x__{dose_name}"
            columns[name] = np.sqrt(clinical[clinical_name].to_numpy() * dose[dose_name].to_numpy())
    interactions = pd.DataFrame(columns, index=clinical.index)
    if interactions.shape[1] != 8 or not np.isfinite(interactions.to_numpy()).all():
        raise ValueError("The eight interactions were not generated correctly")
    # Independently check the row-level arithmetic for the first and last case.
    for row in (0, len(interactions) - 1):
        for clinical_name in clinical:
            for dose_name in dose:
                expected = clinical.loc[row, clinical_name] * dose.loc[row, dose_name]
                observed = interactions.loc[row, f"sqrt__{clinical_name}__x__{dose_name}"] ** 2
                if not np.isclose(observed, expected, rtol=1e-12, atol=1e-12):
                    raise AssertionError("An interaction does not equal sqrt(clinical*dose)")
    return interactions


def bootstrap_auc_ci(y: np.ndarray, probability: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    positive = probability[y == 1]
    negative = probability[y == 0]
    p = positive[rng.integers(len(positive), size=(2000, len(positive)))]
    n = negative[rng.integers(len(negative), size=(2000, len(negative)))]
    auc = (p[:, :, None] > n[:, None, :]).mean(axis=(1, 2))
    auc += 0.5 * (p[:, :, None] == n[:, None, :]).mean(axis=(1, 2))
    return tuple(float(value) for value in np.quantile(auc, (0.025, 0.975)))


def paired_auc_delta_ci(y: np.ndarray, old: np.ndarray, new: np.ndarray,
                        seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    pos = rng.integers(len(positive), size=(2000, len(positive)))
    neg = rng.integers(len(negative), size=(2000, len(negative)))
    old_p, old_n = old[positive[pos]], old[negative[neg]]
    new_p, new_n = new[positive[pos]], new[negative[neg]]
    old_auc = (old_p[:, :, None] > old_n[:, None, :]).mean(axis=(1, 2))
    old_auc += 0.5 * (old_p[:, :, None] == old_n[:, None, :]).mean(axis=(1, 2))
    new_auc = (new_p[:, :, None] > new_n[:, None, :]).mean(axis=(1, 2))
    new_auc += 0.5 * (new_p[:, :, None] == new_n[:, None, :]).mean(axis=(1, 2))
    return tuple(float(value) for value in np.quantile(new_auc - old_auc, (0.025, 0.975)))


def main() -> int:
    cohort = read_cohort()
    clinical = read_selected(CLINICAL_PATH, cohort)
    dose = read_selected(DOSE_PATH, cohort)
    interactions = make_interactions(clinical, dose)
    modalities = {"clinical": clinical, "dose": dose, "interaction": interactions}
    y = cohort.label.to_numpy(dtype=int)
    train = cohort.cohort.eq("training").to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([cohort[["patient_id"]], interactions], axis=1).to_csv(
        OUT / "interaction_features_PRIVATE.csv", index=False, encoding="utf-8-sig")

    predictions = cohort[["patient_id", "cohort", "label"]].copy()
    result_rows = []
    for scheme_index, (scheme, parts) in enumerate(SCHEMES.items()):
        features = pd.concat([modalities[name] for name in parts], axis=1)
        predictor = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("forest", RandomForestClassifier(
                n_estimators=500, max_depth=5, min_samples_leaf=2,
                class_weight="balanced", n_jobs=-1, random_state=SEED,
            )),
        ])
        predictor.fit(features.loc[train], y[train])
        probability = predictor.predict_proba(features)[:, 1]
        predictions[f"probability__{scheme}"] = probability
        for split_index, split in enumerate(SPLITS):
            where = cohort.cohort.eq(split).to_numpy()
            truth = y[where]
            values = probability[where]
            predicted = values >= 0.5
            positive = truth == 1
            negative = ~positive
            tp = int(np.sum(predicted & positive))
            tn = int(np.sum(~predicted & negative))
            fp = int(np.sum(predicted & negative))
            fn = int(np.sum(~predicted & positive))
            lower, upper = bootstrap_auc_ci(truth, values, SEED + scheme_index * 10 + split_index)
            result_rows.append({
                "feature_set": scheme, "cohort": split, "n": int(where.sum()),
                "rp_n": int(truth.sum()), "predictor_n": features.shape[1],
                "auc": float(roc_auc_score(truth, values)),
                "auc_ci_lower": lower, "auc_ci_upper": upper,
                "accuracy": float(accuracy_score(truth, predicted)),
                "precision": tp / (tp + fp) if tp + fp else np.nan,
                "recall_sensitivity": tp / (tp + fn),
                "specificity": tn / (tn + fp),
                "f1": float(f1_score(truth, predicted, zero_division=0)),
                "brier": float(brier_score_loss(truth, values)),
            })

    baseline = "probability__Clinical+Dose"
    for split_index, split in enumerate(SPLITS):
        where = cohort.cohort.eq(split).to_numpy()
        old = predictions.loc[where, baseline].to_numpy()
        baseline_auc = float(roc_auc_score(y[where], old))
        for scheme_index, scheme in enumerate(SCHEMES):
            if scheme == "Clinical+Dose":
                continue
            new = predictions.loc[where, f"probability__{scheme}"].to_numpy()
            row = next(item for item in result_rows if item["feature_set"] == scheme and item["cohort"] == split)
            row["auc_delta_vs_clinical_dose"] = row["auc"] - baseline_auc
            lower, upper = paired_auc_delta_ci(y[where], old, new,
                                               SEED + 100 + scheme_index * 10 + split_index)
            row["auc_delta_ci_lower"] = lower
            row["auc_delta_ci_upper"] = upper

    results = pd.DataFrame(result_rows)
    results.to_csv(OUT / "five_models.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT / "patient_predictions_PRIVATE.csv", index=False, encoding="utf-8-sig")
    (OUT / "experiment.json").write_text(json.dumps({
        "cohort": "155 fixed reviewed cases, unchanged 108/23/24 split",
        "clinical_features": list(clinical), "dose_features": list(dose),
        "interaction_formula": "sqrt(clinical_feature * dose_feature)",
        "interaction_features": list(interactions),
        "model": {"type": "RandomForestClassifier", "n_estimators": 500,
                  "max_depth": 5, "min_samples_leaf": 2,
                  "class_weight": "balanced", "random_state": SEED},
        "threshold": 0.5, "bootstrap": "2000 stratified patient resamples; fitted model held fixed",
        "analysis_note": "Exploratory; the same test cohort has been viewed in previous experiments.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[["feature_set", "cohort", "predictor_n", "auc", "auc_ci_lower",
                   "auc_ci_upper", "brier"]].to_string(index=False))
    print(f"Saved: {OUT / 'five_models.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
