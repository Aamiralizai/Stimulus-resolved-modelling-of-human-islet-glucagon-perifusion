"""Step 10 - robustness of the HbA1c association (Gate 1.5): demography, cohort year, shape, influence and censoring.

Outcomes: the Penn glucose-suppression contrast and the capacity-adjusted amino acids + 3 mM window, from the Gate 1
(uncensored) donor estimates, in controls and T2D donors. The base model is y ~ HbA1c + age + male + BMI + isolation
year with HC3 standard errors.
  demography      race added as a factor, isolation year as a factor instead of a linear term, and both together
  shape           linear HbA1c against a natural cubic spline (3 df) and against a T2D-range-only slope, compared by F
                  test, plus the slope within controls and within T2D donors separately
  influence       leave-one-donor-out range of the coefficient and the largest P, Cook's distance, DFBETA for HbA1c
  robust          Huber M-estimation and median (quantile) regression
  censoring       complete-case analysis restricted to runs with no censored fraction, and to runs with none in the
                  amino-acid windows, with the Spearman correlation between censored-fraction counts and HbA1c
Outputs: results/tables/robustness_checks.tsv, robustness_leave_one_out.tsv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from glucv2.data import load_matrix, contrast_vector, WINDOWS

OUT = ROOT / 'results/tables'
GS = 'Glucose suppression (3 to 16.7 mM)'
BASE = 'y ~ hba1c + age + male + bmi + isolation_year'


def frame():
    D = pd.read_csv(ROOT / 'data/processed/donors.tsv', sep='\t').set_index('donor')
    E = pd.read_csv(ROOT / 'data/processed/donor_effects_Penn_glucagon.tsv', sep='\t', index_col=0)
    d = D.loc[[x for x in E.index if x in D.index]].copy()
    d = d[d.group.isin(['control', 'T2D'])].copy()
    d['T2D'] = (d.group == 'T2D').astype(int); d['male'] = (d.sex == 'Male').astype(int)
    d['isolation_year'] = d.isolation_year.fillna(d.isolation_year.median())
    d['year_f'] = d.isolation_year.round().astype(int).astype(str)
    d['race_f'] = d.race.fillna('unknown').astype(str)
    d['supp'] = E[f'gls_{GS}'].reindex(d.index)
    w = [x[0] for x in WINDOWS['Penn']]
    d['aag3'] = (E['gls_AA+G3'] - E['gls_capacity']).reindex(d.index)
    # censoring counts, overall and in the amino-acid windows
    M = load_matrix('Penn', 'glucagon'); C = pd.DataFrame(M['censored'], index=M['donors'], columns=M['frac'])
    aa = [f for f, k in zip(M['frac'], M['win']) if w[k] in ('AA', 'AA+G3')]
    d['n_cens'] = C.sum(axis=1).reindex(d.index); d['n_cens_aa'] = C[aa].sum(axis=1).reindex(d.index)
    return d.dropna(subset=['supp', 'aag3', 'hba1c', 'age', 'bmi'])


def row(d, form, label, group, outcome, term='hba1c', method='ols'):
    dd = d.rename(columns={outcome: 'y'})
    if method == 'ols': f = smf.ols(form, data=dd).fit(cov_type='HC3')
    elif method == 'huber': f = smf.rlm(form, data=dd, M=sm.robust.norms.HuberT()).fit()
    else: f = smf.quantreg(form, data=dd).fit(q=0.5)
    sd = dd.y.std(); ci = f.conf_int()
    return {'outcome': outcome, 'group': group, 'check': label, 'n': int(f.nobs), 'beta': f.params[term], 'se': f.bse[term],
            'p': f.pvalues[term], 'beta_SD': f.params[term] / sd, 'ci_low_SD': ci.loc[term, 0] / sd, 'ci_high_SD': ci.loc[term, 1] / sd}


def main():
    d = frame(); rows = []
    for outcome in ['supp', 'aag3']:
        rows.append(row(d, BASE, 'base', 'demography', outcome))
        rows.append(row(d, BASE + ' + C(race_f)', 'race added', 'demography', outcome))
        rows.append(row(d, 'y ~ hba1c + age + male + bmi + C(year_f)', 'isolation year as a factor', 'demography', outcome))
        rows.append(row(d, 'y ~ hba1c + age + male + bmi + C(year_f) + C(race_f)', 'race and isolation year factor', 'demography', outcome))
        rows.append(row(d.dropna(subset=['viability', 'cold_ischemia_h']), BASE + ' + viability + cold_ischemia_h', 'islet quality added', 'demography', outcome))
        # shape of the HbA1c relation
        dd = d.rename(columns={outcome: 'y'})
        lin = smf.ols(BASE, data=dd).fit(); spl = smf.ols(BASE.replace('hba1c', "cr(hba1c, df=3, constraints='center')"), data=dd).fit()
        dnum = spl.df_model - lin.df_model
        Fstat = ((lin.ssr - spl.ssr) / dnum) / (spl.ssr / spl.df_resid)
        ft = (Fstat, float(stats.f.sf(Fstat, dnum, spl.df_resid)))
        rows.append({'outcome': outcome, 'group': 'shape', 'check': 'natural spline (3 df) against linear, F test',
                     'n': int(lin.nobs), 'beta': np.nan, 'se': np.nan, 'p': ft[1], 'beta_SD': np.nan, 'ci_low_SD': np.nan, 'ci_high_SD': np.nan})
        rows.append(row(d[d.T2D == 0], BASE, 'controls only', 'shape', outcome))
        rows.append(row(d[d.T2D == 1], BASE, 'T2D donors only', 'shape', outcome))
        # influence
        b = []
        for i in d.index:
            f = smf.ols(BASE, data=d.drop(index=i).rename(columns={outcome: 'y'})).fit(cov_type='HC3')
            b.append({'dropped': i, 'beta': f.params['hba1c'], 'p': f.pvalues['hba1c']})
        B = pd.DataFrame(b); B['outcome'] = outcome
        B.to_csv(OUT / f'robustness_leave_one_out_{outcome}.tsv', sep='\t', index=False)
        base = smf.ols(BASE, data=d.rename(columns={outcome: 'y'})).fit()
        infl = base.get_influence(); cook = infl.cooks_distance[0]
        j = list(base.params.index).index('hba1c'); dfb = infl.dfbetas[:, j]
        rows.append({'outcome': outcome, 'group': 'influence', 'check': f'leave one donor out, {len(B)} fits', 'n': len(B),
                     'beta': B.beta.median(), 'se': np.nan, 'p': B.p.max(), 'beta_SD': B.beta.min(), 'ci_low_SD': B.beta.max(),
                     'ci_high_SD': np.nan})
        rows.append({'outcome': outcome, 'group': 'influence', 'check': 'largest Cook distance and |DFBETA| for HbA1c',
                     'n': len(d), 'beta': float(cook.max()), 'se': float(np.abs(dfb).max()), 'p': np.nan,
                     'beta_SD': np.nan, 'ci_low_SD': np.nan, 'ci_high_SD': np.nan})
        rows.append(row(d, BASE, 'Huber M-estimation', 'robust', outcome, method='huber'))
        rows.append(row(d, BASE, 'median regression', 'robust', outcome, method='median'))
        # censoring
        rows.append(row(d[d.n_cens == 0], BASE, 'runs with no censored fraction', 'censoring', outcome))
        rows.append(row(d[d.n_cens_aa == 0], BASE, 'runs with no censored amino-acid fraction', 'censoring', outcome))
    rc, pc = stats.spearmanr(d.n_cens, d.hba1c); ra, pa = stats.spearmanr(d.n_cens_aa, d.hba1c)
    rows.append({'outcome': 'censoring', 'group': 'censoring', 'check': 'Spearman, censored fractions against HbA1c',
                 'n': len(d), 'beta': rc, 'se': np.nan, 'p': pc, 'beta_SD': np.nan, 'ci_low_SD': np.nan, 'ci_high_SD': np.nan})
    rows.append({'outcome': 'censoring', 'group': 'censoring', 'check': 'Spearman, censored amino-acid fractions against HbA1c',
                 'n': len(d), 'beta': ra, 'se': np.nan, 'p': pa, 'beta_SD': np.nan, 'ci_low_SD': np.nan, 'ci_high_SD': np.nan})
    T = pd.DataFrame(rows).drop_duplicates(subset=['outcome', 'group', 'check'])
    T.to_csv(OUT / 'robustness_checks.tsv', sep='\t', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 55); print(T.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
