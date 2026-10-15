"""Step 06 - dynamic phenotypes along the metabolic continuum (HbA1c, T2D status, BMI).

Donors: controls and T2D (type 1 diabetes donors excluded). Outcomes: donor GLS estimates (unshrunk; measurement
error only widens confidence intervals) of capacity and each stimulus contrast, per lab and hormone, plus a combined
glucose-suppression score (mean of the available lab-specific z-scores, so a single-lab z for donors measured at one lab).
Models (OLS, HC3 standard errors):
  HbA1c   outcome ~ HbA1c + age + sex + BMI + isolation year
  T2D     outcome ~ T2D + age + sex + BMI + isolation year
  BMI     outcome ~ BMI + age + sex + isolation year
Benjamini-Hochberg FDR within each predictor over the phenotype outcomes (capacity, contrasts, combined score).
Capacity-adjusted window levels (interpretation) and conventional summary traits computed from the same curves
(comparison) are reported with their own FDR families. Sensitivity analyses for the HbA1c association (donors at both
labs, Vanderbilt runs without censored values, controls only, T2D donors only, excluding islet autoantibody-positive
donors, adjustment for T2D status, age 30 years or older,
adjustment for islet viability and cold ischaemia time) form a separate family.
Direction. Glucose suppression contrasts are log(16.7 mM level) - log(lower-glucose level). Values closer to zero or
positive mean weaker suppression.
Outputs: results/tables/clinical_associations.tsv, data/processed/donor_phenotypes.tsv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from glucv2.data import CONTRASTS, WINDOWS
WINDOWS_NAMES = {k: [w[0] for w in v] for k, v in WINDOWS.items()}


def main():
    D = pd.read_csv(ROOT / 'data/processed/donors.tsv', sep='\t').set_index('donor')
    D = D[D.group.isin(['control', 'T2D'])].copy(); D['T2D'] = (D.group == 'T2D').astype(int); D['male'] = (D.sex == 'Male').astype(int)
    D['isolation_year'] = D.isolation_year.fillna(D.isolation_year.median())
    P = pd.DataFrame(index=D.index); outcomes = []
    for lab in ['Penn', 'Vanderbilt']:
        for hormone in ['glucagon', 'insulin']:
            E = pd.read_csv(ROOT / f'data/processed/donor_effects_{lab}_{hormone}.tsv', sep='\t', index_col=0)
            for name in ['capacity'] + list(CONTRASTS[lab]):
                col = f'{lab} | {hormone} | {name}'; P[col] = E[f'gls_{name}'].reindex(P.index); outcomes.append(col)
            # capacity-adjusted window levels (window effect minus the donor's capacity), for interpretation
            wins = [c[4:] for c in E.columns if c.startswith('gls_') and c[4:] in WINDOWS_NAMES[lab]]
            for w in wins:
                col = f'{lab} | {hormone} | window {w} (capacity-adjusted)'; P[col] = (E[f'gls_{w}'] - E['gls_capacity']).reindex(P.index); outcomes.append(col)
    for hormone in ['glucagon', 'insulin']:
        a = P[f'Penn | {hormone} | Glucose suppression (3 to 16.7 mM)']; b = P[f'Vanderbilt | {hormone} | Glucose suppression (5.6 to 16.7 mM)']
        z = pd.concat([(a - a.mean()) / a.std(), (b - b.mean()) / b.std()], axis=1).mean(1)
        col = f'Combined | {hormone} | Glucose suppression (z, mean of labs)'; P[col] = z; outcomes.append(col)
    # conventional summary traits from the same curves (log scale), for comparison
    import importlib.util
    spec = importlib.util.spec_from_file_location('xl', ROOT / 'workflow/05_cross_lab.py'); xl = importlib.util.module_from_spec(spec); spec.loader.exec_module(xl)
    for lab in ['Penn', 'Vanderbilt']:
        Tc = xl.conv_table(lab, 'glucagon')
        for c in Tc.columns:
            col = f'{lab} | glucagon | conventional {c}'; P[col] = np.log(Tc[c].clip(lower=1e-6)).reindex(P.index); outcomes.append(col)
    P.to_csv(ROOT / 'data/processed/donor_phenotypes.tsv', sep='\t')
    rows = []
    for pred, form in [('HbA1c', 'y ~ hba1c + age + male + bmi + isolation_year'), ('T2D', 'y ~ T2D + age + male + bmi + isolation_year'),
                       ('BMI', 'y ~ bmi + age + male + isolation_year')]:
        term = {'HbA1c': 'hba1c', 'T2D': 'T2D', 'BMI': 'bmi'}[pred]
        for o in outcomes:
            d = pd.concat([P[o].rename('y'), D], axis=1).dropna(subset=['y', 'hba1c', 'age', 'bmi'])
            if len(d) < 20: continue
            f = smf.ols(form, data=d).fit(cov_type='HC3'); sd = d.y.std()
            rows.append({'predictor': pred, 'outcome': o, 'n': len(d), 'n_T2D': int(d.T2D.sum()), 'beta': f.params[term], 'se': f.bse[term],
                         'ci_low': f.conf_int().loc[term, 0], 'ci_high': f.conf_int().loc[term, 1], 'p': f.pvalues[term],
                         'beta_per_outcome_SD': f.params[term] / sd})
    # sensitivity: glucose suppression restricted to donors perifused at both labs; Vanderbilt runs without censored values
    both = P[['Penn | glucagon | Glucose suppression (3 to 16.7 mM)', 'Vanderbilt | glucagon | Glucose suppression (5.6 to 16.7 mM)']].dropna().index
    EV = pd.read_csv(ROOT / 'data/processed/donor_effects_Vanderbilt_glucagon.tsv', sep='\t', index_col=0); nocens = EV.index[EV.n_censored == 0]
    for o, idx, tag in [('Penn | glucagon | Glucose suppression (3 to 16.7 mM)', both, 'donors at both labs'),
                        ('Vanderbilt | glucagon | Glucose suppression (5.6 to 16.7 mM)', both, 'donors at both labs'),
                        ('Vanderbilt | glucagon | Glucose suppression (5.6 to 16.7 mM)', nocens, 'Vanderbilt runs without censored values')]:
        d = pd.concat([P.loc[P.index.isin(idx), o].rename('y'), D], axis=1, join='inner').dropna(subset=['y', 'hba1c', 'age', 'bmi'])
        f = smf.ols('y ~ hba1c + age + male + bmi + isolation_year', data=d).fit(cov_type='HC3')
        rows.append({'predictor': 'HbA1c', 'outcome': f'{o} [{tag}]', 'n': len(d), 'n_T2D': int(d.T2D.sum()), 'beta': f.params['hba1c'], 'se': f.bse['hba1c'],
                     'ci_low': f.conf_int().loc['hba1c', 0], 'ci_high': f.conf_int().loc['hba1c', 1], 'p': f.pvalues['hba1c'], 'beta_per_outcome_SD': f.params['hba1c'] / d.y.std()})
    # sensitivity: is the HbA1c gradient present beyond the diagnosis, within controls, at overlapping ages, and after islet-quality adjustment?
    base = 'y ~ hba1c + age + male + bmi + isolation_year'
    for o in ['Penn | glucagon | Glucose suppression (3 to 16.7 mM)', 'Combined | glucagon | Glucose suppression (z, mean of labs)',
              'Penn | glucagon | AA response', 'Penn | glucagon | window AA+G3 (capacity-adjusted)']:
        for tag, sub, form in [('controls only', lambda d: d[d.T2D == 0], base), ('T2D donors only', lambda d: d[d.T2D == 1], base),
                               ('excluding autoantibody-positive donors', lambda d: d[d.aab_any != 1], base), ('adjusted for T2D status', lambda d: d, base + ' + T2D'),
                               ('age 30 years or older', lambda d: d[d.age >= 30], base),
                               ('adjusted for viability and cold ischaemia', lambda d: d.dropna(subset=['viability', 'cold_ischemia_h']), base + ' + viability + cold_ischemia_h')]:
            d = sub(pd.concat([P[o].rename('y'), D], axis=1).dropna(subset=['y', 'hba1c', 'age', 'bmi']))
            f = smf.ols(form, data=d).fit(cov_type='HC3')
            rows.append({'predictor': 'HbA1c', 'outcome': f'{o} [{tag}]', 'n': len(d), 'n_T2D': int(d.T2D.sum()), 'beta': f.params['hba1c'], 'se': f.bse['hba1c'],
                         'ci_low': f.conf_int().loc['hba1c', 0], 'ci_high': f.conf_int().loc['hba1c', 1], 'p': f.pvalues['hba1c'], 'beta_per_outcome_SD': f.params['hba1c'] / d.y.std()})
    T = pd.DataFrame(rows)
    T['family'] = np.where(T.outcome.str.contains('\\['), 'sensitivity',
                           np.where(T.outcome.str.contains('window '), 'window levels (interpretation)',
                                    np.where(T.outcome.str.contains('conventional'), 'conventional traits (comparison)', 'phenotypes')))
    T['fdr'] = T.groupby(['predictor', 'family']).p.transform(lambda p: multipletests(p, method='fdr_bh')[1])
    T.to_csv(ROOT / 'results/tables/clinical_associations.tsv', sep='\t', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
    print(T[T.outcome.str.contains('glucagon')].sort_values(['predictor', 'p']).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
