"""S2 observed outcomes and locked exploratory logistic interaction tests.
Run with the original modelling virtual environment (statsmodels==0.14.5).
Uses the existing selected features and cohort assignments.
"""
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import shap
from visualize_interaction_groups import ROOT, IMC2, load_data, summarize, sha256
from visualize_tp_imc2_response import CLINICAL, DOSE, design

OUT = ROOT / 'outputs/supplement_S2_logistic'
PAIRS = [('TotalProtein', IMC2), ('CEA', IMC2)]


def exp_scientific(value):
    exponent = int(np.floor(value / np.log(10)))
    mantissa = np.exp(value-exponent*np.log(10))
    return f'{mantissa:.4f}e{exponent:+d}'


def savefig(fig, name):
    for ext in ['png', 'pdf', 'tiff']:
        kwargs = dict(dpi=400, bbox_inches='tight', facecolor='white')
        if ext == 'tiff':
            kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(OUT / (name+'.'+ext), **kwargs)
    plt.close(fig)


def fit_checked(x, y):
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always')
        result = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=500, tol=1e-10)
    score = float(np.max(np.abs(result.model.score(result.params))))
    rank = np.linalg.matrix_rank(x)
    if not result.converged or rank != x.shape[1] or score > 1e-5:
        raise ValueError(f'Invalid MLE: converged={result.converged}; rank={rank}; score={score}')
    if any('separation' in str(m.message).lower() for m in messages):
        raise ValueError('Separation detected; ordinary Wald inference is not valid.')
    eta=np.asarray(x) @ np.asarray(result.params)
    weight=np.exp(-np.abs(eta))/(1+np.exp(-np.abs(eta)))**2
    information=np.asarray(x).T @ (weight[:,None]*np.asarray(x))
    stable_se=np.sqrt(np.diag(np.linalg.inv(information)))
    relative_se_difference=float(np.max(np.abs(stable_se-result.bse)/np.maximum(stable_se,1e-12)))
    if relative_se_difference>0.01:
        raise ValueError(f'GLM covariance cross-check differs by {relative_se_difference}')
    return result, {'converged': bool(result.converged), 'score_max_abs': score,
                    'condition_number': float(np.linalg.cond(x)),
                    'information_condition_number': float(np.linalg.cond(information)),
                    'covariance_crosscheck_max_relative_SE_difference':relative_se_difference,
                    'warnings': [str(m.message) for m in messages]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, medians, probability, paths, hashes = load_data(ROOT, 'Clinical+Dose+Interaction')
    train = data.cohort.eq('training')
    grouped, summaries = summarize(data, 'TotalProtein', medians, probability)
    # Observed outcomes ONLY, no predicted probabilities in Figure S2.
    records = []
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12,
                         'axes.titlesize':16, 'axes.labelsize':14, 'pdf.fonttype':42})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(left=.09, right=.98, bottom=.09, top=.88, hspace=.42, wspace=.24)
    labels = ['TP low\nImc2 low','TP low\nImc2 high','TP high\nImc2 low','TP high\nImc2 high']
    for ax, split, title, letter in zip(axes.ravel(), ['all','training','validation','test'],
            ['Entire Sample','Training Cohort','Validation Cohort','Internal Test Cohort'], 'abcd'):
        sub = grouped if split == 'all' else grouped.loc[grouped.cohort.eq(split)]
        for g in range(4):
            part = sub.loc[sub.stratum.eq(g)]
            n = len(part); positive = int(part.label.sum()); negative = n-positive
            p = positive/n if n else np.nan
            records.append(dict(cohort=split, group=labels[g].replace('\n',' / '),
                                n=n, RP=positive, non_RP=negative, RP_proportion=p,
                                non_RP_proportion=1-p))
            if n:
                ax.bar(g, p, color='#CC6677', width=.65)
                ax.bar(g, 1-p, bottom=p, color='#4477AA', width=.65)
                if positive: ax.text(g,p/2,f'{positive}/{n}\n{p:.1%}',ha='center',va='center',color='white',fontsize=11)
                if negative: ax.text(g,p+(1-p)/2,f'{negative}/{n}\n{1-p:.1%}',ha='center',va='center',color='white',fontsize=11)
                ax.text(g,1.025,f'n = {n}',ha='center',fontsize=11)
            else: ax.text(g,.5,'No patients',rotation=90,ha='center')
        ax.set_title(f'{title} (n = {len(sub)})',pad=17)
        ax.set_xticks(range(4),labels); ax.set_ylim(0,1.10)
        ax.set_yticks([0,.25,.5,.75,1]); ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel('Observed Patient Proportion')
        ax.spines[['top','right']].set_visible(False)
        ax.text(-.15,1.1,letter,transform=ax.transAxes,fontsize=20,fontweight='bold')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color='#CC6677',label='RP (positive)'),Patch(color='#4477AA',label='Non-RP (negative)')],
               loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5,.985),fontsize=13)
    savefig(fig, 'Figure_S2_Observed_RP_Composition')
    pd.DataFrame(records).to_csv(OUT/'Table_S2_Observed_Counts.csv',index=False,encoding='utf-8-sig')

    # Lock both specified pairs before tests; no search over pairs or transformations.
    scaler = StandardScaler().fit(data.loc[train,CLINICAL+DOSE])
    z = pd.DataFrame(scaler.transform(data[CLINICAL+DOSE]),index=data.index,columns=CLINICAL+DOSE)
    pd.DataFrame({'feature':CLINICAL+DOSE,'training_mean':scaler.mean_,
                  'training_sd_ddof0':scaler.scale_}).to_csv(OUT/'Logistic_Training_Scaling.csv',index=False)
    results=[]; diagnostics={}; coefrows=[]
    for scope, mask in [('training_primary',train),('all155_sensitivity',np.ones(len(data),dtype=bool))]:
        y=data.loc[mask,'label']; base=sm.add_constant(z.loc[mask],has_constant='add')
        fit0, diagnostic=fit_checked(base,y); diagnostics[scope+'/main_effects']=diagnostic
        for a,b in PAIRS:
            x=base.copy(); term=a+':Imc2'; x[term]=z.loc[mask,a]*z.loc[mask,b]
            fit1, diagnostic=fit_checked(x,y); diagnostics[scope+'/'+term]=diagnostic
            beta=float(fit1.params[term]); se=float(fit1.bse[term]); lower,upper=fit1.conf_int().loc[term]
            lr=2*(fit1.llf-fit0.llf)
            if lr < -1e-7: raise ValueError('Nested model likelihood decreased')
            row=dict(scope=scope,pair=term,n=len(y),RP=int(y.sum()),beta=beta,SE=se,
                     OR=exp_scientific(beta),OR_CI_lower=exp_scientific(lower),OR_CI_upper=exp_scientific(upper),
                     log_OR_CI_lower=float(lower),log_OR_CI_upper=float(upper),
                     Wald_P=float(fit1.pvalues[term]),LRT_chi2=max(0,lr),LRT_df=1,LRT_P=float(chi2.sf(max(0,lr),1)),
                     AIC_main=float(fit0.aic),AIC_interaction=float(fit1.aic))
            results.append(row)
            for name in fit1.params.index:
                coefrows.append(dict(scope=scope,interaction_test=term,term=name,beta=fit1.params[name],SE=fit1.bse[name],P=fit1.pvalues[name]))
            # Cross-check the MLE using a distinct optimizer.
            with warnings.catch_warnings(record=True) as check_warnings:
                warnings.simplefilter('always')
                check=sm.Logit(y,x).fit(method='bfgs',maxiter=2000,disp=False,gtol=1e-9)
            if abs(check.llf-fit1.llf)>1e-6: raise ValueError('Independent MLE cross-check failed')
            diagnostic['BFGS_crosscheck_converged']=bool(check.mle_retvals['converged'])
            diagnostic['BFGS_loglik_difference']=float(check.llf-fit1.llf)
            diagnostic['BFGS_warnings']=[str(w.message) for w in check_warnings]
    result=pd.DataFrame(results)
    for scope in result.scope.unique():
        where=result.scope.eq(scope)
        for col in ['Wald_P','LRT_P']:
            result.loc[where,col+'_Holm']=multipletests(result.loc[where,col],method='holm')[1]
    result.to_csv(OUT/'Table_S3_Logistic_Interaction_Tests.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(coefrows).to_csv(OUT/'Logistic_Coefficients.csv',index=False)
    print(result.to_string(index=False),flush=True)

    # Direct SHAP interaction values for the two raw-variable pairs in the locked RF.
    config=json.loads((ROOT/'outputs/experiment.json').read_text(encoding='utf-8'))
    params={k:v for k,v in config['model'].items() if k!='type'}
    xx=design(data[CLINICAL+DOSE].reset_index(drop=True))
    model=RandomForestClassifier(**params,n_jobs=-1).fit(xx.loc[train.to_numpy()],data.loc[train,'label'])
    assert np.allclose(model.predict_proba(xx)[:,1],data[probability],atol=1e-12,rtol=0)
    explainer=shap.TreeExplainer(model,model_output='raw',feature_perturbation='tree_path_dependent')
    si=explainer.shap_interaction_values(xx.loc[train.to_numpy()])
    si=si[1] if isinstance(si,list) else np.asarray(si)[...,1]
    sv=explainer.shap_values(xx.loc[train.to_numpy()])
    sv=sv[1] if isinstance(sv,list) else np.asarray(sv)[...,1]
    assert np.allclose(si.sum(axis=2),sv,atol=1e-7)
    assert np.allclose(si,si.transpose(0,2,1),atol=1e-7)
    rows=[]
    for i in range(14):
        for j in range(i+1,14):
            rows.append(dict(feature_a=xx.columns[i],feature_b=xx.columns[j],
                             mean_absolute_phi_ij=float(np.abs(si[:,i,j]).mean())))
    ranks=pd.DataFrame(rows).sort_values('mean_absolute_phi_ij',ascending=False)
    ranks['rank_among_91_pairs']=np.arange(1,len(ranks)+1)
    locked=ranks.loc[(ranks.feature_a.isin(CLINICAL)) & ranks.feature_b.eq(IMC2)]
    locked.to_csv(OUT/'Locked_Pairs_SHAP_Interaction.csv',index=False)
    print('Locked pair SHAP interactions (single off-diagonal entry):',flush=True)
    print(locked.to_string(index=False),flush=True)
    assert hashes=={k:sha256(v) for k,v in paths.items()}
    audit={'primary':'108 training cases; six main effects + one standardized bilinear product at a time',
           'secondary':'all 155 cases, exploratory sensitivity only; not independent confirmation',
           'candidate_pairs':PAIRS,'scaling':'train-only StandardScaler, ddof=0; product of standardized source variables',
           'CI':'Wald 95% CI','multiplicity':'Holm across two pairs separately for Wald and LRT within each scope',
           'diagnostics':diagnostics,'source_hashes':hashes,
           'warnings':['Same-data feature/candidate selection means nominal p-values are exploratory, not post-selection-adjusted.',
                       'The bilinear logit product differs from the RF sqrt(raw clinical * raw dose) feature.',
                       'Non-significant bilinear tests do not establish nonlinearity.',
                       'SHAP of an engineered feature is distinct from off-diagonal SHAP interaction values.',
                       'OR is the ratio of one-SD clinical odds ratios for a one-SD increase in Imc2, not high-high vs low-low OR.']}
    (OUT/'Analysis_Audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('Analysis complete. Output:',OUT)


if __name__=='__main__': main()
