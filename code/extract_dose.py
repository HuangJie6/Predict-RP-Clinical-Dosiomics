"""Optional re-extraction of 93 whole-lung dose features from private NRRD data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from radiomics import featureextractor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_private"


def main() -> None:
    cohort = pd.read_csv(DATA / "cohort.csv", encoding="utf-8-sig")
    extractor = featureextractor.RadiomicsFeatureExtractor(str(ROOT / "config" / "dose_radiomics.yaml"))
    rows = []
    for _, case in cohort.iterrows():
        folder = DATA / "volumes" / case.patient_id
        result = extractor.execute(str(folder / "Dose_resampled.nrrd"), str(folder / "LungMask_resampled.nrrd"))
        values = {f"dose_{key}": float(value) for key, value in result.items()
                  if key.startswith("original_")}
        rows.append({"patient_id": case.patient_id, **values, "label": int(case.label)})
    frame = pd.DataFrame(rows)
    assert frame.shape == (155, 95), frame.shape
    reference_columns = pd.read_csv(DATA / "dose_radiomics.csv", nrows=0, encoding="utf-8-sig").columns
    assert set(frame.columns) == set(reference_columns)
    frame = frame.loc[:, reference_columns]
    destination = ROOT / "outputs" / "dose_radiomics_recomputed.csv"
    destination.parent.mkdir(exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8-sig")
    print(destination)


if __name__ == "__main__":
    main()
