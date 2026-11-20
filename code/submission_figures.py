"""Generate separate figures for clinical–dosiomic model analysis."""
import argparse
from pathlib import Path

import reporting as report
from report_feature_flow import main as feature_flow
import pandas as pd


def main(destination):
    destination.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(report.DATA / "cohort.csv")
    pred = pd.read_csv(report.OUT / "patient_predictions_PRIVATE.csv")
    if not cohort.patient_id.equals(pred.patient_id):
        raise ValueError("Prediction rows differ from the analysis cohort")
    feature_flow(destination / "Fig2.tiff")
    report.figure2(pred, destination / "Fig3.tiff")
    report.figure3(cohort, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=report.OUT / "journal_figures")
    main(parser.parse_args().out_dir)
