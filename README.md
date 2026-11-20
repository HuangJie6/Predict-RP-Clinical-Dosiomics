# Clinical and whole-lung dosiomic interactions for radiation pneumonitis prediction

Analysis code and extracted whole-lung dose features for **Clinical and whole-lung dosiomic features for radiation pneumonitis prediction in esophageal cancer receiving concurrent chemoradiotherapy**.

Authors: Jie Huang, Xu Chen, Jiantao Wang, Shuai Liu, and Xueling Li.
Correspondence: Xueling Li, Hefei Cancer Hospital, Chinese Academy of Sciences; xlli@cmpt.ac.cn.
Target journal: International Journal of Computer Assisted Radiology and Surgery. This repository does not imply journal acceptance.

## Repository contents

- `code/`: feature extraction, training-only selection, five random-forest models, figure/table generation, SHAP and exploratory logistic interaction tests.
- `features/whole_lung_dosiomic_features.csv`: the only public patient-level table, containing 155 anonymous samples and 93 original-dose features.
- `requirements.txt`: analysis dependencies. `requirements_extraction.txt`: optional dose-extraction dependencies.

The table contains no RP labels, clinical variables, patient identifiers, split assignments, raw images, dose volumes or masks. Full prediction-model reproduction therefore requires authorized access to the private clinical/outcome and split information; the public table alone is not a labeled machine-learning benchmark. Other study data may be requested from the corresponding author, subject to institutional approval and patient privacy requirements.

## Inspect the public data

Python 3.10 or newer:

```cmd
python code/run_analysis.py --stage validate-public
```

This validation uses the standard library and checks row/column counts, unique anonymous IDs and finite feature values.

## Reproduce with authorized private inputs

Install `requirements.txt` in an isolated Python environment. Put `cohort.csv` and `dose_radiomics.csv` in `code/data_private/` (ignored by Git).

`cohort.csv` requires `patient_id`, `label` (0/1), `cohort` (training/validation/test) and the clinical candidate columns listed in `code/config/private_input_schema.json`. `dose_radiomics.csv` requires the same private `patient_id`, the 93 `dose_original_*` columns, and `label`. Public IDs must be linked through the separately controlled mapping by an authorized custodian; never infer an analysis split from public row order.

```cmd
python -m pip install -r requirements.txt
python code/run_analysis.py --stage all
```

Stages may also be run separately: `select`, `models`, `figures`, `supplement`. To rerun only figures/models, provide the frozen `selected_clinical_features.csv` and `selected_dose_radiomics.csv` under `code/data_private/`.

The study-specific checks preserve the 155-patient cohort (108/23/24), six selected source predictors and original model settings. They intentionally stop if the selected feature set changes. For a new study, design and validate a new cohort rather than silently bypassing these checks.

## Methods and outputs

Selection uses median imputation, missing-value/variance filters, mutual information, between-feature Spearman filtering, and parallel L1 logistic regression and Boruta. Up to 40 candidates per modality enter the latter methods. The six source predictors generate eight native-scale square-root products. Five random forests use 2, 4, 8, 6 and 14 inputs. Training-only fitting, fixed 0.50 classification, stratified bootstrap AUC intervals and paired AUC differences are retained.

`outputs/journal_figures/` contains Fig2 (feature-selection flow), Fig3 (ROC/calibration), Fig4 (six-panel SHAP), and FigS1 (correlations). FigS2 and logistic tests are in `outputs/supplement_S2_logistic/`. Fig1 is the authors' enrollment diagram and is not generated from invented exclusion counts. Table CSVs are generated in `outputs/`; manuscript table numbering maps extracted feature classes to Table2, selected features to Table3, and performance to Table4. All generated/private files remain local.

Chi-square stratification is an association analysis, not a test of additive interaction. Logistic product tests differ from the square-root predictors and remain exploratory after feature discovery. The original scaling-before-LogisticRegressionCV behavior is preserved for fidelity; full selection was restricted to training patients. This software is research code and is not a clinically validated decision-support tool.

## Optional extraction from authorized images

Use 3D Slicer with SlicerRT and `code/slicer/extract_lung_ct_dose.py` for local DICOM processing. `code/resample_dose.py` aligns dose/masks at 2-mm spacing. `code/extract_dose.py` extracts the 93 original-dose features using the 1-Gy bin-width settings in `code/config/dose_radiomics.yaml`. These stages require private imaging inputs and a separate PyRadiomics-capable environment. No images or masks are distributed here.

## Citation and availability

Please cite the associated article once published and identify the repository revision used. No DOI or acceptance status has been invented. Reuse and further data access may be discussed with the corresponding author.
