"""Step 08 - confound checks on the HbA1c association of Penn glucagon dynamics (Gate 1b).

Outcomes (Penn, unshrunk GLS donor estimates from step 03): glucose suppression (3 to 16.7 mM), amino-acid response,
the capacity-adjusted amino acids + 3 mM glucose window, and capacity. Controls and T2D donors.
Base model (as in step 06): y ~ HbA1c + age + male + BMI + isolation year, OLS with HC3 standard errors.
Checks
  duration      diabetes duration (years, parsed from the donor summary) added to the base model, T2D donors only.
                Also the base model in the same T2D donors without duration, and duration alone.
  composition   histological alpha and delta cell fractions added (beta is the remainder).
  paracrine     Penn insulin capacity and the Penn insulin response to 16.7 mM glucose (same runs) added.
  medication    mention of insulin treatment in the clinical record added, and donors with it excluded.
                Text mining. Absence of a mention does not prove absence of treatment.
  sex           HbA1c x sex interaction and sex-stratified HbA1c slopes. T2D x sex for capacity.
For each check the HbA1c slope is reported in SD of the outcome per 1% HbA1c, with the base slope in the same donors.
Outputs: results/tables/followup_checks.tsv, data/processed/donor_covariates_extra.tsv
"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
ROOT = Path(__file__).resolve().parents[1]

INSULIN_RX = r'insulin|novolog|glargine|lantus|humalog|basaglar|levemir|lispro|aspart|detemir|degludec|toujeo|tresiba'
OTHER_RX = {'metformin': r'metformin|glucophage', 'sulfonylurea': r'glipizide|glyburide|glimepiride|glibenclamide|gliclazide',
            'DPP4': r'sitagliptin|linagliptin|saxagliptin|alogliptin|januvia|tradjenta',
            'GLP1': r'liraglutide|semaglutide|dulaglutide|exenatide|ozempic|trulicity|victoza|byetta|tirzepatide|mounjaro',
            'SGLT2': r'empagliflozin|dapagliflozin|canagliflozin|jardiance|farxiga|invokana'}


def years(s):
    """Parse a free-text diabetes duration into years. Ranges give the midpoint, bounds give the bound, months are divided by 12."""
    if not isinstance(s, str) or 'pregnan' in s.lower(): return np.nan
    t = s.lower().replace('years', 'year').replace('months', 'month')
    nums = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', t)]
    if not nums: return np.nan
    v = np.mean(nums[:2]) if '-' in t else nums[0]
    if t.strip().startswith('<'): v = v / 2 if 'year' in t else v
    return v / 12 if 'month' in t else v


def extra_covariates():
    X = pd.read_excel(ROOT / 'data/raw/hpap/Donor_Summary_216.xlsx')
    rec = (X.past_clinical_record.astype(str) + ' ' + X.surgical_history.astype(str)).str.lower()
    E = pd.DataFrame({'donor': X.donor_ID, 'duration_years': X.disease_duration.map(years),
                      'insulin_mention': rec.str.contains(INSULIN_RX).astype(int)})
    for k, rx in OTHER_RX.items(): E[f'{k}_mention'] = rec.str.contains(rx).astype(int)
    return E.set_index('donor')


def fit(d, form, term='hba1c'):
    f = smf.ols(form, data=d).fit(cov_type='HC3'); sd = d.y.std()
    return {'n': len(d), 'n_T2D': int(d.T2D.sum()), 'beta': f.params[term], 'se': f.bse[term], 'p': f.pvalues[term],
            'beta_SD': f.params[term] / sd, 'ci_low_SD': f.conf_int().loc[term, 0] / sd, 'ci_high_SD': f.conf_int().loc[term, 1] / sd}


def main():
    D = pd.read_csv(ROOT / 'data/processed/donors.tsv', sep='\t').set_index('donor')
    D = D[D.group.isin(['control', 'T2D'])].copy(); D['T2D'] = (D.group == 'T2D').astype(int); D['male'] = (D.sex == 'Male').astype(int)
    D['isolation_year'] = D.isolation_year.fillna(D.isolation_year.median())
    E = extra_covariates(); D = D.join(E)
    EG = pd.read_csv(ROOT / 'data/processed/donor_effects_Penn_glucagon.tsv', sep='\t', index_col=0)
    EI = pd.read_csv(ROOT / 'data/processed/donor_effects_Penn_insulin.tsv', sep='\t', index_col=0)
    D['ins_capacity'] = EI.gls_capacity.reindex(D.index); D['ins_glucose_response'] = EI['gls_Glucose suppression (3 to 16.7 mM)'].reindex(D.index)
    D.to_csv(ROOT / 'data/processed/donor_covariates_extra.tsv', sep='\t')
    outs = {'Glucose suppression (3 to 16.7 mM)': EG['gls_Glucose suppression (3 to 16.7 mM)'], 'AA response': EG['gls_AA response'],
            'AA+G3 window (capacity-adjusted)': EG['gls_AA+G3'] - EG['gls_capacity'], 'capacity': EG['gls_capacity']}
    base = 'y ~ hba1c + age + male + bmi + isolation_year'
    rows = []
    for oname, y in outs.items():
        d0 = pd.concat([y.rename('y'), D], axis=1, join='inner').dropna(subset=['y', 'hba1c', 'age', 'bmi'])
        add = lambda check, model, d, form, term='hba1c': rows.append({'outcome': oname, 'check': check, 'model': model, 'term': term, **fit(d, form, term)})
        add('base', 'all controls and T2D', d0, base)
        # duration, T2D donors with a parsed duration
        t = d0[(d0.T2D == 1) & d0.duration_years.notna()]
        add('duration', 'T2D with duration, base', t, base)
        add('duration', 'T2D with duration, + duration', t, base + ' + duration_years')
        add('duration', 'T2D with duration, + duration', t, base + ' + duration_years', term='duration_years')
        add('duration', 'T2D with duration, duration without HbA1c', t, 'y ~ duration_years + age + male + bmi + isolation_year', term='duration_years')
        # composition
        c = d0.dropna(subset=['alpha_frac_hist', 'delta_frac_hist'])
        add('composition', 'with histology, base', c, base)
        add('composition', 'with histology, + alpha and delta fractions', c, base + ' + alpha_frac_hist + delta_frac_hist')
        add('composition', 'with histology, + alpha and delta fractions', c, base + ' + alpha_frac_hist + delta_frac_hist', term='alpha_frac_hist')
        # paracrine insulin
        q = d0.dropna(subset=['ins_capacity', 'ins_glucose_response'])
        add('paracrine', 'with Penn insulin, base', q, base)
        add('paracrine', 'with Penn insulin, + insulin capacity and glucose response', q, base + ' + ins_capacity + ins_glucose_response')
        add('paracrine', 'with Penn insulin, + insulin capacity and glucose response', q, base + ' + ins_capacity + ins_glucose_response', term='ins_glucose_response')
        # medication
        add('medication', 'all, + insulin mention', d0, base + ' + insulin_mention')
        add('medication', 'excluding insulin mention', d0[d0.insulin_mention == 0], base)
        # sex
        add('sex', 'all, HbA1c x male interaction', d0, base + ' + hba1c:male', term='hba1c:male')
        add('sex', 'females', d0[d0.male == 0], 'y ~ hba1c + age + bmi + isolation_year')
        add('sex', 'males', d0[d0.male == 1], 'y ~ hba1c + age + bmi + isolation_year')
        add('sex', 'all, T2D x male interaction (T2D model)', d0, 'y ~ T2D + age + male + bmi + isolation_year + T2D:male', term='T2D:male')
        add('sex', 'females, T2D effect', d0[d0.male == 0], 'y ~ T2D + age + bmi + isolation_year', term='T2D')
        add('sex', 'males, T2D effect', d0[d0.male == 1], 'y ~ T2D + age + bmi + isolation_year', term='T2D')
    T = pd.DataFrame(rows); T.to_csv(ROOT / 'results/tables/followup_checks.tsv', sep='\t', index=False)
    # descriptive summaries
    t2d = D[D.T2D == 1]
    S = {'T2D donors': len(t2d), 'T2D with parsed duration': int(t2d.duration_years.notna().sum()),
         'corr HbA1c vs duration (T2D, Spearman)': t2d[['hba1c', 'duration_years']].corr(method='spearman').iloc[0, 1],
         'insulin mention, T2D': int(t2d.insulin_mention.sum()), 'insulin mention, controls': int(D[D.T2D == 0].insulin_mention.sum()),
         **{f'{k} mention, T2D': int(t2d[f'{k}_mention'].sum()) for k in OTHER_RX},
         'any glucose-lowering drug mention, T2D': int(((t2d.insulin_mention + t2d[[f'{k}_mention' for k in OTHER_RX]].sum(1)) > 0).sum())}
    pd.Series(S).to_csv(ROOT / 'results/tables/followup_checks_summary.tsv', sep='\t', header=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
    print(pd.Series(S).to_string()); print(T.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
