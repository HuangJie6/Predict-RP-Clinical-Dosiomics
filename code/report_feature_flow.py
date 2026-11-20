"""Report extracted dose-feature classes and the recorded training-only selection flow."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
CLASS_LABELS = {
    "firstorder": "First-order",
    "glcm": "Gray-level co-occurrence matrix (GLCM)",
    "glrlm": "Gray-level run-length matrix (GLRLM)",
    "glszm": "Gray-level size-zone matrix (GLSZM)",
    "gldm": "Gray-level dependence matrix (GLDM)",
    "ngtdm": "Neighboring gray-tone difference matrix (NGTDM)",
}


def main(figure_path: Path | None = None) -> None:
    OUT.mkdir(exist_ok=True)
    headers = pd.read_csv(ROOT / "data_private" / "dose_radiomics.csv", nrows=0).columns
    dose_names = [name for name in headers if name.startswith("dose_original_")]
    if len(dose_names) != 93:
        raise ValueError(f"Expected 93 original dose features, found {len(dose_names)}")
    classes = [name.removeprefix("dose_original_").split("_", 1)[0] for name in dose_names]
    counts = pd.Series(classes).value_counts()
    if set(counts.index) != set(CLASS_LABELS):
        raise ValueError(f"Unexpected dose feature classes: {set(counts.index)}")
    rows = [
        {"Image type": "Original dose map", "Feature class": label,
         "Number of extracted features": int(counts[key])}
        for key, label in CLASS_LABELS.items()
    ]
    rows.append({"Image type": "Original dose map", "Feature class": "Total",
                 "Number of extracted features": len(dose_names)})
    if figure_path is None:
        pd.DataFrame(rows).to_csv(OUT / "Table_Extracted_Dose_Feature_Classes.csv",
                                  index=False, encoding="utf-8-sig")

    summary = json.loads((ROOT / "selection_source" / "original_selection_summary.json").read_text(encoding="utf-8"))
    by_name = {entry["modality"]: entry for entry in summary["modalities"]}
    clinical = by_name["clinical_features"]
    dose = by_name["dose_radiomics"]
    if dose["initial_features"] != len(dose_names) or clinical["fit_cases"] != 108 or dose["fit_cases"] != 108:
        raise ValueError("Feature extraction and selection summaries do not match the frozen analysis")

    fig, ax = plt.subplots(figsize=(10.2, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 3.45)
    ax.axis("off")
    stages = ["Candidate\nfeatures", "Missing value +\nvariance filters",
              "MI ranking +\nSpearman filter", "L1-CV /\nBoruta", "Final\nfeatures"]
    x_positions = [0.15, 2.16, 4.17, 6.18, 8.19]
    box_w = 1.78
    for x, stage in zip(x_positions, stages):
        ax.text(x + box_w / 2, 3.24, stage, ha="center", va="top", fontsize=9.8, weight="bold")
    for label, data, y, color in [
        ("Clinical", clinical, 1.85, "#DDEBF6"),
        ("Dosiomic", dose, 0.52, "#FBE5D6"),
    ]:
        ax.text(0.15, y + 0.82, label, ha="left", va="bottom", fontsize=11.4, weight="bold")
        text_values = [
            str(data["initial_features"]),
            str(data["after_variance"]),
            str(data["after_correlation"]),
            f'L1: {data["lasso_selected"]}   Boruta: {data["boruta_confirmed"]}',
            str(len(data["final_features"])),
        ]
        for idx, (x, value) in enumerate(zip(x_positions, text_values)):
            box = FancyBboxPatch((x, y), box_w, 0.67, boxstyle="round,pad=0.03,rounding_size=0.08",
                                 facecolor=color, edgecolor="#555555", linewidth=1)
            ax.add_patch(box)
            ax.text(x + box_w / 2, y + 0.335, value, ha="center", va="center",
                    fontsize=11.2 if idx != 3 else 9.6, weight="bold" if idx == 4 else "normal")
            if idx < len(x_positions) - 1:
                arrow = FancyArrowPatch((x + box_w + 0.04, y + 0.335),
                                        (x_positions[idx + 1] - 0.06, y + 0.335),
                                        arrowstyle="-|>", mutation_scale=12,
                                        linewidth=1, color="#555555")
                ax.add_patch(arrow)
    ax.text(5.1, 0.10, "All screening steps used the 108-patient training cohort; L1-CV and Boruta were evaluated in parallel.",
            ha="center", va="center", fontsize=9.8, color="#333333")
    if figure_path is None:
        fig.savefig(OUT / "Fig_Feature_Selection_Flow.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUT / "Fig_Feature_Selection_Flow.tiff", dpi=600, bbox_inches="tight")
    else:
        fig.savefig(figure_path, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        fig.savefig(figure_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Created feature-selection figure:", figure_path or OUT / "Fig_Feature_Selection_Flow.tiff")


if __name__ == "__main__":
    main()
