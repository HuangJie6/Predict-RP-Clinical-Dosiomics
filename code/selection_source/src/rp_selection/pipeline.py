from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger("rp_selection")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _normalise_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def _path(value: str, project_dir: Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (project_dir / candidate).resolve()


def _configure_logging(output_root: Path) -> Path:
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    return log_path


def _load_clinical(config: dict[str, Any], project_dir: Path) -> tuple[pd.DataFrame, str, str]:
    source = _path(config["input"]["clinical_csv"], project_dir)
    if not source.is_file():
        raise FileNotFoundError(f"Clinical CSV not found: {source}")
    clinical = pd.read_csv(source, encoding="utf-8-sig")
    clinical.columns = [str(column).strip() for column in clinical.columns]
    if clinical.columns.duplicated().any():
        duplicates = clinical.columns[clinical.columns.duplicated()].tolist()
        raise ValueError(f"Clinical CSV has duplicate column names after trimming spaces: {duplicates}")
    patient_column = config["input"]["patient_id_column"]
    label_column = config["input"]["label_column"]
    missing = {patient_column, label_column}.difference(clinical.columns)
    if missing:
        raise ValueError(f"Clinical CSV is missing required columns: {sorted(missing)}")
    clinical = clinical.copy()
    clinical[patient_column] = _normalise_id(clinical[patient_column])
    if clinical[patient_column].duplicated().any():
        duplicates = clinical.loc[clinical[patient_column].duplicated(), patient_column].head(5).tolist()
        raise ValueError(f"Duplicate patient_id values in clinical CSV: {duplicates}")
    clinical[label_column] = pd.to_numeric(clinical[label_column], errors="raise").astype(int)
    labels = set(clinical[label_column].unique())
    if labels != {0, 1}:
        raise ValueError(f"This binary-selection pipeline requires labels {{0, 1}}, got {sorted(labels)}")
    exclusion_csv = str(config["input"].get("confirmed_exclusions_csv", "")).strip()
    if exclusion_csv:
        exclusion_path = _path(exclusion_csv, project_dir)
        if not exclusion_path.is_file():
            raise FileNotFoundError(f"Confirmed-exclusion CSV not found: {exclusion_path}")
        exclusions = pd.read_csv(exclusion_path, encoding="utf-8-sig")
        exclusions.columns = [str(column).strip() for column in exclusions.columns]
        required = {patient_column, "review_decision", "reason", "reviewed_by", "review_date"}
        if missing := required.difference(exclusions.columns):
            raise ValueError(f"Confirmed-exclusion CSV is missing columns: {sorted(missing)}")
        exclusions[patient_column] = _normalise_id(exclusions[patient_column])
        confirmed = exclusions.loc[
            exclusions["review_decision"].astype(str).str.strip().eq("exclude_confirmed_by_clinical_review")
        ].copy()
        if confirmed.empty:
            raise ValueError("No clinically confirmed exclusions were found in the configured exclusion CSV.")
        if confirmed[patient_column].duplicated().any():
            raise ValueError("Confirmed-exclusion CSV contains duplicate patient IDs.")
        for column in ("reason", "reviewed_by", "review_date"):
            if confirmed[column].isna().any() or confirmed[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"Confirmed exclusions require non-empty '{column}' values.")
        unknown = set(confirmed[patient_column]).difference(clinical[patient_column])
        if unknown:
            raise ValueError(f"Confirmed exclusion IDs not found in the clinical cohort: {sorted(unknown)}")
        clinical = clinical.loc[~clinical[patient_column].isin(confirmed[patient_column])].copy().reset_index(drop=True)
        labels = set(clinical[label_column].unique())
        if labels != {0, 1}:
            raise ValueError("Clinical exclusions left fewer than two outcome classes.")
        LOGGER.info("Applied %d clinically confirmed exclusions; retained %d cases.", len(confirmed), len(clinical))
    return clinical, patient_column, label_column


def _fit_ids(clinical: pd.DataFrame, config: dict[str, Any], patient_column: str) -> tuple[list[str], str]:
    cohort_column = str(config["input"].get("cohort_column", "")).strip()
    if not cohort_column:
        LOGGER.warning("No cohort column configured: selection is fitted on all cases. Repeat selection inside training folds during model development.")
        return clinical[patient_column].tolist(), "all_cases_no_external_holdout"
    if cohort_column not in clinical.columns:
        raise ValueError(f"Configured cohort_column is absent from clinical CSV: {cohort_column}")
    allowed = {str(value).casefold() for value in config["input"]["training_values"]}
    values = clinical[cohort_column].fillna("").astype(str).str.strip().str.casefold()
    selected = clinical.loc[values.isin(allowed), patient_column].tolist()
    if not selected:
        raise ValueError(f"No training/development cases found in {cohort_column!r} using {sorted(allowed)}")
    return selected, f"development_only:{cohort_column}"


def _read_features(path: Path, patient_column: str, label_column: str) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Feature CSV not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"{path.name} has duplicate feature names after trimming spaces: {duplicates}")
    if patient_column not in frame.columns:
        raise ValueError(f"{path.name} is missing {patient_column!r}")
    frame = frame.copy()
    frame[patient_column] = _normalise_id(frame[patient_column])
    if frame[patient_column].duplicated().any():
        raise ValueError(f"{path.name} has duplicate patient_id values")
    # Cohort assignment is a split-control field, never a predictor.
    excluded = {patient_column, label_column, "cohort", "status", "error"}
    feature_columns = [column for column in frame.columns if column not in excluded]
    if not feature_columns:
        raise ValueError(f"{path.name} contains no feature columns")
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = pd.concat([frame[[patient_column]], numeric], axis=1)
    return frame, feature_columns


def _icc_2_1(matrix: np.ndarray) -> float:
    """Two-way random, absolute-agreement, single-measure ICC(2,1)."""
    n, k = matrix.shape
    if n < 2 or k < 2:
        return float("nan")
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    grand_mean = matrix.mean()
    ms_row = k * np.square(row_means - grand_mean).sum() / (n - 1)
    ms_col = n * np.square(col_means - grand_mean).sum() / (k - 1)
    residual = matrix - row_means[:, None] - col_means[None, :] + grand_mean
    ms_error = np.square(residual).sum() / ((n - 1) * (k - 1))
    denominator = ms_row + (k - 1) * ms_error + (k * (ms_col - ms_error) / n)
    return float((ms_row - ms_error) / denominator) if denominator else float("nan")


def _icc_filter(
    modality: str,
    candidate_columns: list[str],
    fit_ids: list[str],
    config: dict[str, Any],
    project_dir: Path,
    patient_column: str,
    label_column: str,
) -> tuple[set[str], pd.DataFrame, dict[str, Any]]:
    cfg = config["icc"]
    paths = [_path(value, project_dir) for value in cfg.get("repeat_feature_files", {}).get(modality, []) if str(value).strip()]
    if not cfg.get("enabled", True) or len(paths) < 2:
        audit = pd.DataFrame({"feature": candidate_columns, "icc_2_1": np.nan, "icc_status": "not_run_no_repeat_feature_tables", "icc_keep": True})
        return set(candidate_columns), audit, {"status": "not_run", "reason": "at least two repeat feature tables are required"}
    repeats = [_read_features(path, patient_column, label_column)[0].set_index(patient_column) for path in paths]
    common_ids = [case for case in fit_ids if all(case in table.index for table in repeats)]
    min_cases = int(cfg["minimum_common_cases"])
    if len(common_ids) < min_cases:
        audit = pd.DataFrame({"feature": candidate_columns, "icc_2_1": np.nan, "icc_status": "not_run_insufficient_common_cases", "icc_keep": True})
        return set(candidate_columns), audit, {"status": "not_run", "reason": f"only {len(common_ids)} common cases; need {min_cases}"}
    threshold = float(cfg["minimum_icc_2_1"])
    rows: list[dict[str, Any]] = []
    keep: set[str] = set()
    for feature in candidate_columns:
        if not all(feature in table.columns for table in repeats):
            rows.append({"feature": feature, "icc_2_1": np.nan, "icc_status": "missing_in_repeat_table", "icc_keep": False})
            continue
        matrix = np.column_stack([pd.to_numeric(table.loc[common_ids, feature], errors="coerce").to_numpy() for table in repeats])
        matrix = matrix[np.isfinite(matrix).all(axis=1)]
        value = _icc_2_1(matrix) if len(matrix) >= min_cases else float("nan")
        accepted = bool(np.isfinite(value) and value >= threshold)
        rows.append({"feature": feature, "icc_2_1": value, "icc_status": "calculated", "icc_keep": accepted})
        if accepted:
            keep.add(feature)
    return keep, pd.DataFrame(rows), {"status": "calculated", "repeat_tables": [str(path) for path in paths], "common_cases": len(common_ids), "threshold": threshold}


def _select_modality(
    modality: dict[str, Any],
    clinical: pd.DataFrame,
    fit_ids: list[str],
    fit_scope: str,
    config: dict[str, Any],
    project_dir: Path,
    patient_column: str,
    label_column: str,
    output_root: Path,
) -> dict[str, Any]:
    name = modality["name"]
    feature_path = _path(modality["feature_csv"], project_dir)
    features, original_columns = _read_features(feature_path, patient_column, label_column)
    feature_indexed = features.set_index(patient_column)
    clinical_ids = clinical[patient_column].tolist()
    missing_features = [case for case in clinical_ids if case not in feature_indexed.index]
    extra_features = [case for case in feature_indexed.index if case not in set(clinical_ids)]
    if config["input"].get("require_exact_case_match", True) and (missing_features or extra_features):
        raise ValueError(f"{name}: feature/clinical IDs differ; missing={len(missing_features)}, extra={len(extra_features)}")
    available_ids = [case for case in clinical_ids if case in feature_indexed.index]
    fit_ids = [case for case in fit_ids if case in feature_indexed.index]
    if len(fit_ids) < 20:
        raise ValueError(f"{name}: fewer than 20 eligible fit cases")
    y = clinical.set_index(patient_column).loc[fit_ids, label_column].to_numpy(dtype=int)
    if min(np.bincount(y)) < 5:
        raise ValueError(f"{name}: each class needs at least 5 fit cases for five-fold selection")
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_cfg = config["selection"]
    all_x = feature_indexed.loc[available_ids, original_columns].copy()
    fit_x = all_x.loc[fit_ids]
    audit = pd.DataFrame({"feature": original_columns})
    missing_fraction = fit_x.isna().mean()
    audit["missing_fraction_fit"] = audit["feature"].map(missing_fraction)
    keep_missing = set(missing_fraction[missing_fraction <= float(selection_cfg["maximum_missing_fraction"])].index)
    audit["keep_missingness"] = audit["feature"].isin(keep_missing)
    if not keep_missing:
        raise ValueError(f"{name}: no features remain after missingness filter")
    imputer = SimpleImputer(strategy="median")
    kept_columns = [column for column in original_columns if column in keep_missing]
    fit_imputed = pd.DataFrame(imputer.fit_transform(fit_x[kept_columns]), index=fit_ids, columns=kept_columns)
    all_imputed = pd.DataFrame(imputer.transform(all_x[kept_columns]), index=available_ids, columns=kept_columns)
    variance = pd.Series(fit_imputed.var(axis=0, ddof=0), index=kept_columns)
    keep_variance = set(variance[variance > float(selection_cfg["variance_threshold"])].index)
    audit["variance_fit"] = audit["feature"].map(variance)
    audit["keep_variance"] = audit["feature"].isin(keep_variance)
    if not keep_variance:
        raise ValueError(f"{name}: no features remain after variance filter")
    variance_columns = [column for column in kept_columns if column in keep_variance]
    icc_keep, icc_audit, icc_summary = _icc_filter(name, variance_columns, fit_ids, config, project_dir, patient_column, label_column)
    audit = audit.merge(icc_audit, on="feature", how="left")
    audit["keep_icc"] = audit["feature"].isin(icc_keep)
    if not icc_keep:
        raise ValueError(f"{name}: no features remain after ICC stability filter")
    stable_columns = [column for column in variance_columns if column in icc_keep]
    stable_x = fit_imputed[stable_columns]
    mi = mutual_info_classif(stable_x, y, discrete_features=False, n_neighbors=int(selection_cfg["mutual_information_neighbors"]), random_state=int(selection_cfg["random_seed"]))
    mi_series = pd.Series(mi, index=stable_columns).sort_values(ascending=False, kind="mergesort")
    audit["mutual_information"] = audit["feature"].map(mi_series)
    audit["mi_rank"] = audit["feature"].map(pd.Series(np.arange(1, len(mi_series) + 1), index=mi_series.index))
    pre_corr_count = min(len(mi_series), int(selection_cfg["maximum_pre_correlation_candidates"]))
    pre_corr_columns = mi_series.index[:pre_corr_count].tolist()
    audit["keep_pre_correlation_mi"] = audit["feature"].isin(pre_corr_columns)
    correlation_limit = float(selection_cfg["spearman_correlation_threshold"])
    corr = stable_x[pre_corr_columns].corr(method="spearman").abs()
    correlation_keep: list[str] = []
    correlation_dropped: dict[str, str] = {}
    for feature in pre_corr_columns:
        conflict = next((retained for retained in correlation_keep if corr.loc[feature, retained] >= correlation_limit), None)
        if conflict is None:
            correlation_keep.append(feature)
        else:
            correlation_dropped[feature] = conflict
    audit["correlation_redundant_with"] = audit["feature"].map(correlation_dropped)
    audit["keep_correlation"] = audit["feature"].isin(correlation_keep)
    max_mi = min(len(correlation_keep), int(selection_cfg["maximum_mi_candidates"]))
    min_mi = min(len(correlation_keep), int(selection_cfg["minimum_mi_candidates"]))
    candidate_count = max(min_mi, max_mi)
    mi_candidates = correlation_keep[:candidate_count]
    if len(mi_candidates) < 2:
        raise ValueError(f"{name}: fewer than two non-redundant MI candidates")
    audit["keep_mi_candidate"] = audit["feature"].isin(mi_candidates)
    cv_folds = min(int(selection_cfg["lasso_cv_folds"]), int(np.bincount(y).min()))
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=int(selection_cfg["random_seed"]))
    lasso = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegressionCV(
            Cs=int(selection_cfg["lasso_c_values"]), cv=splitter, penalty="l1", solver="saga", scoring="roc_auc",
            class_weight="balanced", max_iter=10000, n_jobs=-1, refit=True, random_state=int(selection_cfg["random_seed"]),
        )),
    ])
    lasso.fit(fit_imputed[mi_candidates], y)
    coefficients = pd.Series(lasso.named_steps["model"].coef_.ravel(), index=mi_candidates)
    lasso_keep = set(coefficients[coefficients.abs() > 1e-10].index)
    audit["lasso_coefficient"] = audit["feature"].map(coefficients)
    audit["keep_lasso"] = audit["feature"].isin(lasso_keep)
    forest = RandomForestClassifier(n_estimators=1000, class_weight="balanced", max_depth=5, n_jobs=-1, random_state=int(selection_cfg["random_seed"]))
    boruta = BorutaPy(forest, n_estimators="auto", max_iter=int(selection_cfg["boruta_max_iter"]), random_state=int(selection_cfg["random_seed"]), verbose=0)
    boruta.fit(fit_imputed[mi_candidates].to_numpy(), y)
    boruta_confirmed = {feature for feature, flag in zip(mi_candidates, boruta.support_) if flag}
    boruta_tentative = {feature for feature, flag in zip(mi_candidates, boruta.support_weak_) if flag}
    decisions = {feature: "confirmed" if feature in boruta_confirmed else "tentative" if feature in boruta_tentative else "rejected" for feature in mi_candidates}
    audit["boruta_decision"] = audit["feature"].map(decisions).fillna("not_run")
    audit["keep_boruta_confirmed"] = audit["feature"].isin(boruta_confirmed)
    audit["keep_boruta_tentative"] = audit["feature"].isin(boruta_tentative)
    mi_rank = {feature: rank for rank, feature in enumerate(mi_candidates, start=1)}
    score: dict[str, float] = {}
    for feature in mi_candidates:
        score[feature] = (1 - (mi_rank[feature] - 1) / max(1, len(mi_candidates))) + (1 if feature in lasso_keep else 0) + (2 if feature in boruta_confirmed else 1 if feature in boruta_tentative else 0)
    consensus = set(lasso_keep).intersection(boruta_confirmed)
    fallback_pool = consensus or (set(lasso_keep).union(boruta_confirmed).union(boruta_tentative)) or set(mi_candidates)
    minor_class_count = int(np.bincount(y).min())
    event_cap = max(int(selection_cfg["minimum_final_features"]), minor_class_count // int(selection_cfg["events_per_feature"]))
    final_cap = min(int(selection_cfg["maximum_final_features"]), event_cap, len(fallback_pool))
    final_features = sorted(fallback_pool, key=lambda feature: (-score[feature], mi_rank[feature], feature))[:final_cap]
    audit["consensus_score"] = audit["feature"].map(score)
    audit["selected_final"] = audit["feature"].isin(final_features)
    audit["selection_pool"] = "not_candidate"
    audit.loc[audit["feature"].isin(consensus), "selection_pool"] = "lasso_and_boruta_confirmed"
    if not consensus:
        audit.loc[audit["feature"].isin(fallback_pool), "selection_pool"] = "fallback_union_or_mi"
    clinical_indexed = clinical.set_index(patient_column)
    output_frame = pd.DataFrame({patient_column: available_ids, label_column: clinical_indexed.loc[available_ids, label_column].to_numpy()})
    for feature in final_features:
        output_frame[feature] = all_imputed.loc[available_ids, feature].to_numpy()
    _write_csv(output_dir / "selected_features.csv", output_frame)
    _write_csv(output_dir / "feature_selection_audit.csv", audit.sort_values(["selected_final", "consensus_score", "feature"], ascending=[False, False, True], na_position="last"))
    _write_json(output_dir / "selected_feature_names.json", {"modality": name, "selected_features": final_features})
    _write_json(output_dir / "feature_preprocessing.json", {
        "fit_scope": fit_scope,
        "imputation": "median_fitted_on_selection_cohort",
        "median_by_selected_feature": {feature: float(imputer.statistics_[kept_columns.index(feature)]) for feature in final_features},
        "variance_threshold": float(selection_cfg["variance_threshold"]),
        "maximum_missing_fraction": float(selection_cfg["maximum_missing_fraction"]),
    })
    summary = {
        "modality": name,
        "feature_source": str(feature_path),
        "fit_scope": fit_scope,
        "fit_cases": len(fit_ids),
        "all_available_cases": len(available_ids),
        "class_counts_fit": {str(key): int(value) for key, value in pd.Series(y).value_counts().sort_index().items()},
        "initial_features": len(original_columns),
        "after_missingness": len(keep_missing),
        "after_variance": len(keep_variance),
        "after_icc": len(icc_keep),
        "after_correlation": len(correlation_keep),
        "mi_candidates": len(mi_candidates),
        "lasso_selected": len(lasso_keep),
        "boruta_confirmed": len(boruta_confirmed),
        "boruta_tentative": len(boruta_tentative),
        "final_cap": final_cap,
        "final_features": final_features,
        "icc": icc_summary,
        "correlation_threshold": correlation_limit,
        "selection_pool": "lasso_and_boruta_confirmed" if consensus else "fallback_union_or_mi",
        "warning": "Selection was fitted on all cases; repeat it inside each training fold during modelling." if fit_scope.startswith("all_cases") else None,
    }
    _write_json(output_dir / "summary.json", summary)
    LOGGER.info("%s | selected %d features from %d input features", name, len(final_features), len(original_columns))
    return summary


def run_pipeline(config_path: Path, stage: str, modalities_requested: list[str], project_dir: Path) -> int:
    config = _read_json(config_path)
    output_root = _path(config["output_root"], project_dir)
    log_path = _configure_logging(output_root)
    LOGGER.info("RP project step 03 started | stage=%s | log=%s", stage, log_path)
    clinical, patient_column, label_column = _load_clinical(config, project_dir)
    fit_ids, fit_scope = _fit_ids(clinical, config, patient_column)
    configured = config["modalities"]
    requested = set(modalities_requested)
    selected_modalities = [item for item in configured if not requested or item["name"] in requested]
    unknown = requested.difference({item["name"] for item in configured})
    if unknown:
        raise ValueError(f"Unknown modality name(s): {sorted(unknown)}")
    if not selected_modalities:
        raise ValueError("No modalities selected")
    preflight_rows = []
    for modality in selected_modalities:
        path = _path(modality["feature_csv"], project_dir)
        preflight_rows.append({"modality": modality["name"], "feature_csv": str(path), "exists": path.is_file()})
    _write_csv(output_root / "manifests" / "preflight.csv", pd.DataFrame(preflight_rows))
    absent = [row["modality"] for row in preflight_rows if not row["exists"]]
    if absent:
        raise FileNotFoundError(f"Missing required step-02 feature table(s): {absent}. Finish step 02 before step 03.")
    LOGGER.info("Preflight passed | clinical=%d | fit=%d | modalities=%s", len(clinical), len(fit_ids), ", ".join(item["name"] for item in selected_modalities))
    if stage == "preflight":
        return 0
    summaries = [_select_modality(item, clinical, fit_ids, fit_scope, config, project_dir, patient_column, label_column, output_root) for item in selected_modalities]
    _write_json(output_root / "summary.json", {"completed_at": datetime.now().isoformat(timespec="seconds"), "fit_scope": fit_scope, "modalities": summaries})
    LOGGER.info("RP project step 03 completed successfully.")
    return 0
