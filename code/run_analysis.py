"""Portable entry point; public data inspection is the default action."""
from pathlib import Path
import argparse
import csv
import json
import math
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
EXPECTED={'clinical_features':['TotalProtein','CEA'], 'dose_radiomics':[
    'dose_original_glcm_Imc2','dose_original_glrlm_RunLengthNonUniformity',
    'dose_original_glcm_MaximumProbability','dose_original_glcm_InverseVariance']}

def validate_public(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle);columns=reader.fieldnames;rows=list(reader)
    assert columns and columns[0]=='sample_id'
    assert len(rows)==155 and len(columns)==95
    assert columns[-1]=='label'
    assert all(c.startswith('dose_original_') for c in columns[1:-1])
    assert len({r['sample_id'] for r in rows})==155
    assert all(math.isfinite(float(r[c])) for r in rows for c in columns[1:-1])
    assert all(r['label'] in {'0','1'} for r in rows)
    assert sum(r['label']=='1' for r in rows)==69
    print('Public table passed: 155 samples, 93 numeric dose features; label: 69 RP, 86 non-RP.')

def require_private():
    required=[ROOT/'data_private/cohort.csv', ROOT/'data_private/dose_radiomics.csv']
    absent=[p.name for p in required if not p.is_file()]
    if absent:
        raise SystemExit('Authorized private inputs are required: '+', '.join(absent)+'. See README; the public feature table contains labels but not clinical variables or original cohort assignments.')

def select():
    subprocess.run([sys.executable,str(ROOT/'selection_source/run_03.py'),'--stage','all'],check=True)
    for modality,names in EXPECTED.items():
        folder=ROOT/'selection_source/outputs'/modality
        actual=json.loads((folder/'selected_feature_names.json').read_text(encoding='utf-8'))['selected_features']
        if actual!=names:
            raise RuntimeError('Selected features differ from the frozen study; inspect software/input changes before proceeding: '+modality)
        filename='selected_clinical_features.csv' if modality=='clinical_features' else 'selected_dose_radiomics.csv'
        shutil.copy2(folder/'selected_features.csv',ROOT/'data_private'/filename)
    shutil.copy2(ROOT/'selection_source/outputs/summary.json',ROOT/'selection_source/original_selection_summary.json')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['validate-public','select','models','figures','supplement','all'],default='validate-public')
    parser.add_argument('--public-table',type=Path,default=ROOT.parent/'features/whole_lung_dosiomic_features.csv')
    args=parser.parse_args()
    if args.stage=='validate-public':validate_public(args.public_table);return
    require_private()
    if args.stage in ['select','all']:select()
    if args.stage in ['models','all']:
        from model_analysis import main as fit
        fit()
    if args.stage in ['figures','all']:
        import pandas as pd
        import reporting
        from report_feature_flow import main as flow
        from submission_figures import main as figures
        cohort=pd.read_csv(ROOT/'data_private/cohort.csv')
        reporting.OUT.mkdir(exist_ok=True)
        reporting.table1(cohort)
        reporting.table2()
        reporting.table3(pd.read_csv(ROOT/'outputs/five_models.csv'))
        flow()
        figures(ROOT/'outputs/journal_figures')
    if args.stage in ['supplement','all']:
        from supplement_interaction_tests import main as supplementary
        supplementary()

if __name__=='__main__':main()
