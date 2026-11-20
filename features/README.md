# Whole-lung dosiomic feature table

`whole_lung_dosiomic_features.csv` contains 155 rows and 94 columns: a newly assigned, shuffled public sample identifier and 93 original-dose PyRadiomics features (18 first-order, 24 GLCM, 16 GLRLM, 16 GLSZM, 14 GLDM, and 5 NGTDM).

Extraction used whole-lung cumulative dose in Gy, 2-mm isotropic spacing, linear dose interpolation, nearest-neighbor mask interpolation, and a fixed bin width of 1 Gy. Feature values are not standardized. Names follow `dose_original_<class>_<feature>`.

Patient identifiers, clinical measurements, outcomes, cohort assignments, images, masks, predictions, and the public-to-private ID mapping are excluded. Row order is shuffled and does not encode the analysis split. The table alone cannot reproduce supervised prediction results: authorized outcome and clinical data are required.
