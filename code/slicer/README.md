# Optional Slicer export

Copy config.example.json to config.json and configure authorized local input directories. Place DICOM files in one directory per private patient ID. Run the script inside 3D Slicer with SlicerRT, for example:

```cmd
Slicer.exe --no-splash --no-main-window --python-script extract_lung_ct_dose.py
```

The script reads source images and writes CT.nrrd, LungMask.nrrd and Dose_original.nrrd for complete cases. Only cumulative RTDOSE files matching the study's dose-file convention are selected. Review the generated manifest for excluded or failed cases before running resample_dose.py and extract_dose.py. Existing imaging extraction was not repeated during final analytical reproduction; private source DICOM is required to rerun this optional stage.
