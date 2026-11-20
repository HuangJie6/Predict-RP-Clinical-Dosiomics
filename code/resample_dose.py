"""Resample a registered whole-lung mask and cumulative dose to the planning CT grid."""
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

DATA = Path(__file__).resolve().parent / "data_private"


def resample_case(source, destination):
    ct = sitk.ReadImage(str(source / "CT.nrrd"), sitk.sitkFloat32)
    mask = sitk.Cast(sitk.ReadImage(str(source / "LungMask.nrrd")) > 0, sitk.sitkUInt8)
    dose = sitk.ReadImage(str(source / "Dose_original.nrrd"), sitk.sitkFloat32)
    if ct.GetSize() != mask.GetSize() or any(
        not np.allclose(a, b, atol=1e-5, rtol=0) for a, b in [
            (ct.GetOrigin(), mask.GetOrigin()), (ct.GetDirection(), mask.GetDirection()),
            (ct.GetSpacing(), mask.GetSpacing())]
    ):
        raise ValueError("The input lung mask must share the CT geometry")
    spacing = (2., 2., 2.)
    size = [max(1, round((n - 1) * s / step) + 1)
            for n, s, step in zip(ct.GetSize(), ct.GetSpacing(), spacing)]
    reference = sitk.Image(size, sitk.sitkFloat32)
    reference.SetSpacing(spacing)
    reference.SetOrigin(ct.GetOrigin())
    reference.SetDirection(ct.GetDirection())
    lung = sitk.Resample(mask, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    gy = sitk.Resample(dose, reference, sitk.Transform(), sitk.sitkLinear, 0., sitk.sitkFloat32)
    values = sitk.GetArrayViewFromImage(gy)[sitk.GetArrayViewFromImage(lung) > 0]
    if values.size < 64 or values.size * np.prod(spacing) / 1000 < 100:
        raise ValueError("Implausible whole-lung volume")
    if not np.isfinite(values).all() or values.max() <= 0:
        raise ValueError("Invalid whole-lung dose")
    destination.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(lung, str(destination / "LungMask_resampled.nrrd"), True)
    sitk.WriteImage(gy, str(destination / "Dose_resampled.nrrd"), True)


if __name__ == "__main__":
    for case in pd.read_csv(DATA / "cohort.csv", dtype={"patient_id": str}).patient_id:
        resample_case(DATA / "slicer_volumes" / case, DATA / "volumes" / case)
