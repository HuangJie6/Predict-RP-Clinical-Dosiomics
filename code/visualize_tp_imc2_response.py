"""Four representative TP/Imc2 settings and a derived-feature-aware response surface.

Reconstructs the locked 14-feature model and requires agreement with all 155
saved predictions. Low/high settings are training Q25/Q75, NOT new patient
groups. All four settings use the same 108 training background profiles.
Original files and the manuscript are never modified.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from model_analysis import make_interactions
from visualize_interaction_groups import (
    ROOT, IMC2, COLORS, GROUPS, load_data, sha256, summarize, plot_strata,
)

CLINICAL = ["TotalProtein", "CEA"]
DOSE = [IMC2, "dose_original_glrlm_RunLengthNonUniformity",
        "dose_original_glcm_MaximumProbability", "dose_original_glcm_InverseVariance"]
MODEL = "Clinical+Dose+Interaction"


def design(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.reset_index(drop=True)
    c, d = raw[CLINICAL], raw[DOSE]
    interaction = make_interactions(c, d)
    result = pd.concat([c, d, interaction], axis=1)
    assert result.shape == (len(raw), 14)
    # Verify every derived column after every change to the raw variables.
    for cname in CLINICAL:
        for dname in DOSE:
            assert np.allclose(result[f"sqrt__{cname}__x__{dname}"] ** 2,
                               raw[cname] * raw[dname], rtol=1e-12, atol=1e-12)
    return result


def scenario_predictions(model, background, tp_values, imc_values):
    """Each requested pair is applied to every identical background profile."""
    if len(tp_values) != len(imc_values):
        raise ValueError("Every scenario must specify both TP and Imc2.")
    n = len(background)
    raw = pd.concat([background] * len(tp_values), ignore_index=True)
    raw["TotalProtein"] = np.repeat(tp_values, n)
    raw[IMC2] = np.repeat(imc_values, n)
    values = model.predict_proba(design(raw))[:, 1].reshape(len(tp_values), n)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Invalid response probabilities.")
    return values


def make_figure(out, settings, scenario, tp_grid, imc_grid, surface, support,
                background, line_x, line_predictions):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.labelsize": 14, "axes.titlesize": 15,
                         "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(13.3, 10))
    fig.subplots_adjust(left=.09, right=.97, top=.93, bottom=.09,
                        wspace=.40, hspace=.42)
    a, b, c, d = axes.ravel()
    mean = scenario.mean(axis=1)
    heat = a.imshow(mean.reshape(2, 2), origin="lower", cmap="YlOrRd",
                    vmin=0, vmax=1, aspect="auto")
    a.set_xticks([0, 1], [f"Low\n{settings['imc_low']:.7f}", f"High\n{settings['imc_high']:.7f}"])
    a.set_yticks([0, 1], [f"Low\n{settings['tp_low']:.3g}", f"High\n{settings['tp_high']:.3g}"])
    a.set_xlabel("Imc2 Setting")
    a.set_ylabel("Total Protein Setting")
    a.set_title("Four Representative Combinations", pad=15)
    for index, value in enumerate(mean):
        a.text(index % 2, index // 2, f"{value:.1%}", ha="center", va="center",
               fontsize=24, fontweight="bold", color="white" if value > .60 else "black")
    cb = fig.colorbar(heat, ax=a, fraction=.046, pad=.035,
                      format=PercentFormatter(xmax=1, decimals=0))
    cb.set_label("Mean Predicted RP Probability", fontsize=11)

    rng = np.random.default_rng(20261111)
    for index, values in enumerate(scenario):
        boxes = b.boxplot([values], positions=[index], widths=.5,
                          patch_artist=True, showfliers=False,
                          medianprops={"color": "black", "linewidth": 1.5})
        boxes['boxes'][0].set(facecolor=COLORS[index], alpha=.25)
        b.scatter(index+rng.uniform(-.14, .14, len(values)), values, s=13,
                  color=COLORS[index], alpha=.50, edgecolors="none")
        b.scatter(index, values.mean(), marker="D", facecolors="white",
                  edgecolors="black", s=55, zorder=5)
    b.set_xticks(range(4), ["TP low\nImc2 low", "TP low\nImc2 high",
                            "TP high\nImc2 low", "TP high\nImc2 high"])
    b.set_ylim(0, 1)
    b.set_xlim(-.55, 3.55)
    b.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    b.set_ylabel("Predicted RP Probability")
    b.set_title("Same 108 Background Profiles", pad=15)
    b.grid(axis="y", alpha=.2)
    b.set_axisbelow(True)

    # Do not extrapolate the displayed surface beyond the training TP/Imc2 hull.
    shown = np.ma.masked_where(~support, surface)
    mesh = c.pcolormesh(tp_grid, imc_grid, shown, cmap="YlOrRd", vmin=0, vmax=1,
                        shading="nearest", rasterized=True)
    within = (background.TotalProtein.between(tp_grid.min(), tp_grid.max())
              & background[IMC2].between(imc_grid.min(), imc_grid.max()))
    c.scatter(background.loc[within, "TotalProtein"], background.loc[within, IMC2],
              s=16, facecolors="white", edgecolors="#444444", linewidths=.4, alpha=.65)
    c.set_xlim(tp_grid.min(), tp_grid.max())
    c.set_ylim(imc_grid.min(), imc_grid.max())
    c.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    c.set_xlabel("Total Protein")
    c.set_ylabel("Imc2")
    c.set_title("Joint Model-Response Surface", pad=15)
    cb = fig.colorbar(mesh, ax=c, fraction=.046, pad=.035,
                      format=PercentFormatter(xmax=1, decimals=0))
    cb.set_label("Mean Predicted RP Probability", fontsize=11)

    for index, (tp, color, style) in enumerate([
            (settings['tp_low'], COLORS[0], "-"), (settings['tp_high'], COLORS[3], "--")]):
        d.plot(line_x, line_predictions[index], color=color, linestyle=style,
               linewidth=2.3, label=f"TP = {tp:.3g} ({'low' if index == 0 else 'high'})")
        d.scatter([settings['imc_low'], settings['imc_high']], mean[index*2:index*2+2],
                  color=color, s=44, zorder=4)
    d.axvline(settings['imc_low'], color="gray", linestyle=":", alpha=.5)
    d.axvline(settings['imc_high'], color="gray", linestyle=":", alpha=.5)
    d.set_xlabel("Imc2")
    d.set_ylabel("Mean Predicted RP Probability")
    d.set_title("Imc2 Response at Two TP Levels", pad=15)
    d.set_ylim(0, 1)
    d.set_xlim(line_x.min(), line_x.max())
    d.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    d.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    d.tick_params(axis="x", labelrotation=25)
    d.grid(alpha=.2)
    d.legend(loc="upper left", frameon=False, fontsize=11)
    for letter, ax in zip("abcd", axes.ravel()):
        ax.text(-.17, 1.09, letter, transform=ax.transAxes, fontsize=21,
                fontweight="bold", ha="left", va="top", clip_on=False)
    for ext in ("png", "pdf", "tiff"):
        kw = {"dpi": 400, "bbox_inches": "tight", "facecolor": "white"}
        if ext == "tiff":
            kw['pil_kwargs'] = {"compression": "tiff_lzw"}
        fig.savefig(out / f"Figure_TP_Imc2_Model_Response.{ext}", **kw)
    plt.close(fig)


def main():
    out = ROOT / "outputs" / "tp_imc2_joint_response"
    data, medians, probability, paths, hashes = load_data(ROOT, MODEL)
    experiment_path = ROOT / "outputs/experiment.json"
    hashes['experiment'] = sha256(experiment_path)
    paths['experiment'] = experiment_path
    specification = json.loads(experiment_path.read_text(encoding="utf-8"))
    params = {k: v for k, v in specification['model'].items() if k != 'type'}
    model = Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("forest", RandomForestClassifier(**params, n_jobs=-1))])
    raw = data[CLINICAL+DOSE].reset_index(drop=True)
    train = data.cohort.eq("training").to_numpy()
    design_all = design(raw)
    model.fit(design_all.loc[train], data.label.to_numpy()[train])
    reconstructed = model.predict_proba(design_all)[:, 1]
    error = float(np.max(np.abs(reconstructed-data[probability].to_numpy())))
    if error > 1e-12:
        raise ValueError(f"Model does not reproduce saved predictions: max error {error}")
    print(f"Original model reproduced; maximum probability difference: {error:.3g}", flush=True)
    background = raw.loc[train].reset_index(drop=True)
    q = background[["TotalProtein", IMC2]].quantile([.05, .25, .5, .75, .95])
    settings = {"tp_low": float(q.loc[.25, "TotalProtein"]),
                "tp_high": float(q.loc[.75, "TotalProtein"]),
                "imc_low": float(q.loc[.25, IMC2]), "imc_high": float(q.loc[.75, IMC2])}
    tp_values = [settings['tp_low'], settings['tp_low'], settings['tp_high'], settings['tp_high']]
    imc_values = [settings['imc_low'], settings['imc_high'], settings['imc_low'], settings['imc_high']]
    scenario = scenario_predictions(model, background, tp_values, imc_values)
    summary = pd.DataFrame({
        "TP_level_Imc2_level": GROUPS, "TP_value": tp_values, "Imc2_value": imc_values,
        "sqrt_TP_Imc2": np.sqrt(np.array(tp_values)*imc_values),
        "n_background_profiles": len(background),
        "mean_predicted_probability": scenario.mean(axis=1),
        "median_predicted_probability": np.median(scenario, axis=1),
        "q25_across_background_profiles": np.quantile(scenario, .25, axis=1),
        "q75_across_background_profiles": np.quantile(scenario, .75, axis=1),
    })
    # Use central training ranges specified before inspecting model responses.
    tp_axis = np.linspace(q.loc[.05, "TotalProtein"], q.loc[.95, "TotalProtein"], 31)
    imc_axis = np.linspace(q.loc[.05, IMC2], q.loc[.95, IMC2], 31)
    tg, ig = np.meshgrid(tp_axis, imc_axis)
    print("Computing the joint response surface and conditional response curves...", flush=True)
    surface = scenario_predictions(model, background, tg.ravel(), ig.ravel()).mean(axis=1).reshape(tg.shape)
    # Scale coordinates for numerical stability in the two-dimensional hull.
    origin = background[["TotalProtein", IMC2]].min().to_numpy()
    span = background[["TotalProtein", IMC2]].max().to_numpy()-origin
    hull = Delaunay((background[["TotalProtein", IMC2]].to_numpy()-origin)/span)
    support = (hull.find_simplex((np.c_[tg.ravel(), ig.ravel()]-origin)/span) >= 0).reshape(tg.shape)
    pair_support = hull.find_simplex((np.c_[tp_values, imc_values]-origin)/span) >= 0
    if not pair_support.all():
        raise ValueError("A representative pair lies outside observed TP/Imc2 support; review before plotting.")
    line_x = np.unique(np.r_[np.linspace(imc_axis.min(), imc_axis.max(), 81),
                             settings['imc_low'], settings['imc_high']])
    curves = scenario_predictions(model, background,
                                  np.repeat([settings['tp_low'], settings['tp_high']], len(line_x)),
                                  np.tile(line_x, 2)).mean(axis=1).reshape(2, -1)
    for ti in range(2):
        for ii, value in enumerate([settings['imc_low'], settings['imc_high']]):
            assert np.isclose(curves[ti, np.flatnonzero(line_x == value)[0]], scenario[ti*2+ii].mean())
    out.mkdir(parents=True, exist_ok=True)
    make_figure(out, settings, scenario, tg, ig, surface, support, background, line_x, curves)
    summary.to_csv(out / "Four_Representative_Settings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"TP": tg.ravel(), "Imc2": ig.ravel(),
                  "mean_probability": surface.ravel(), "within_training_pair_hull": support.ravel()}).to_csv(
                      out / "Response_Surface.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"Imc2": line_x, "mean_probability_at_low_TP": curves[0],
                  "mean_probability_at_high_TP": curves[1]}).to_csv(
                      out / "Response_Curves.csv", index=False, encoding="utf-8-sig")
    # Keep actual patient stratification explicitly separate from synthetic settings.
    actual, observed_summary = summarize(data, "TotalProtein", medians, probability)
    actual_out = out / "actual_patient_groups"
    actual_out.mkdir(exist_ok=True)
    plot_strata(actual, observed_summary, "TotalProtein", probability, actual_out)
    observed_summary.to_csv(actual_out / "Observed_Groups.csv", index=False, encoding="utf-8-sig")
    means = scenario.mean(axis=1)
    contrasts = {
        "Imc2_high_minus_low_at_low_TP": float(means[1]-means[0]),
        "Imc2_high_minus_low_at_high_TP": float(means[3]-means[2]),
        "probability_scale_difference_in_differences": float(means[3]-means[2]-means[1]+means[0]),
    }
    after = {key: sha256(path) for key, path in paths.items()}
    if hashes != after:
        raise RuntimeError("Source files changed during analysis.")
    audit = {
        "model": MODEL, "frozen_model_settings": params,
        "maximum_difference_from_saved_predictions": error,
        "background": "The same 108 training profiles for every setting; not four independent patient groups",
        "representative_values": "Training Q25 and Q75, chosen without outcome/prediction optimization",
        "settings": settings, "all_derived_features_recomputed": True,
        "raw_features_held_fixed": [c for c in CLINICAL+DOSE if c not in ['TotalProtein', IMC2]],
        "surface_range": "Training Q05-Q95; 31x31 grid; shown only inside observed TP/Imc2 convex hull",
        "contrasts_descriptive_not_hypothesis_tests": contrasts,
        "limits": ["Not a single-interaction-feature model or a causal intervention.",
                   "Joint changes include direct TP/Imc2 effects and all their derived product features.",
                   "Related original dose features can remain correlated; synthetic settings may be atypical.",
                   "Across-profile IQR is heterogeneity, not a confidence interval for model accuracy.",
                   "This is training-model interpretation, not new validation of predictive performance."],
        "source_sha256": hashes, "sources_unchanged": True,
    }
    (out / "Analysis_Audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    caption = """Joint model-response visualization for total protein (TP) and Imc2. Low and high
settings represent the training-cohort 25th and 75th percentiles, respectively.
Each combination was applied to the same 108 training background profiles, while
the remaining original predictors were retained at their observed values. All
derived square-root product features were recalculated before prediction with
the reconstructed clinical-dosiomic-interaction model. (a) Mean predicted RP
probabilities for the four representative settings. (b) Distributions across
background profiles; points show individual profile predictions, boxes indicate
interquartile ranges, horizontal lines indicate medians, and diamonds indicate
means. Whiskers extend to the most extreme values within 1.5 interquartile ranges.
(c) Mean model-response surface over the central training ranges (5th-95th
percentiles), masked outside the observed TP/Imc2 convex hull. Dots show training
patients within the displayed ranges. (d) Mean Imc2-response curves at the two TP
settings; markers indicate the representative Imc2 values. These are hypothetical
model-input settings, not four clinical cohorts or estimates of causal effects.
"""
    (out / "Figure_Caption_EN.txt").write_text(caption, encoding="utf-8")
    notes = f"""本次分析：TP与Imc2的联合模型响应（探索性）。
与旧图的区别：旧图按中位数分真实患者；新主图将同一批108例背景资料依次代入四个数值情景。
低/高代表训练集Q25/Q75，不代表正常/异常，也不是蛋白表达状态。
TP低/高：{settings['tp_low']:.12g} / {settings['tp_high']:.12g}
Imc2低/高：{settings['imc_low']:.12g} / {settings['imc_high']:.12g}
四组合平均概率（TP低Imc2低、TP低Imc2高、TP高Imc2低、TP高Imc2高）：
{', '.join(f'{v:.2%}' for v in means)}
所有原始特征及预测文件均未改动。Word未改动。
复现模型与原155例保存概率的最大绝对误差：{error:.3g}。
使用的是14输入的原三组分模型，不是仅用TP×Imc2的单输入模型。
其他4项原始变量保持每个人自己的取值；依赖TP/Imc2的全部交互项同步重算。
图a：四个情景平均预测概率。图b：相同108个背景资料在四个情景下的概率分布。
图c：联合响应曲面。图d：分别固定TP低/高水平时Imc2变化对应的平均概率。
这是模型内部响应，不是新的验证结果；不能证明生物学协同或干预后的实际风险。
背景资料的四分位范围不是模型置信区间。变量间相关性可能使某些设定不常见。
真实患者分组结果单独保存在actual_patient_groups，仍采用训练中位数为分界。
"""
    (out / "结果说明.txt").write_text(notes, encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(contrasts, indent=2))
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
