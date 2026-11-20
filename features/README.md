# Whole-lung dosiomic feature table

`whole_lung_dosiomic_features.csv` contains 155 rows and 95 columns: an anonymous sample identifier, 93 original-dose PyRadiomics features (18 first-order, 24 GLCM, 16 GLRLM, 16 GLSZM, 14 GLDM, and 5 NGTDM), and a binary `label` column.

Extraction used whole-lung cumulative dose in Gy, 2-mm isotropic spacing, linear dose interpolation, nearest-neighbor mask interpolation, and a fixed bin width of 1 Gy. Feature values are not standardized. Names follow `dose_original_<class>_<feature>`.

`label = 1` denotes RP (69 samples); `label = 0` denotes non-RP (86 samples). The label is the target, not an input feature.

Patient identifiers, clinical measurements, cohort assignments, images, masks, predictions, and the public-to-private ID mapping are excluded. Row order is shuffled and does not encode the analysis split. The table can be used for dose-only supervised analyses; reproducing the original combined models requires the private clinical data and cohort assignments.
