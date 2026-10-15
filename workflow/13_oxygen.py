"""Step 13 - islet oxygen consumption, rebuilt on the correct protocol (Gate 2 stage 1, revision 2).

PANC-DB oxygen consumption is respirometry with perifusion, measured with an OxyLED phosphorescence system on islets
from the same donor preparation as the Penn hormone run. The first version of this step assumed the oxygen run
followed the hormone protocol and ended with IBMX and KCl. That was wrong. The protocol is a mitochondrial one:

    no substrate -> amino acid mixture -> + 3 mM glucose -> + 16.7 mM glucose -> FCCP -> sodium azide

confirmed by the figure legend of Chen et al., which describes the HPAP oxygen tracings with AAM, glucose, FCCP (an
uncoupler), NaN3 (a cytochrome oxidase inhibitor) and antimycin A. There is no IBMX or KCl phase.

Phase timing. The files carry no stimulus column, so the switch times are measured from the traces. For each usable
trace the smoothed derivative is searched for its largest rise inside each candidate interval. The onsets are tight
across donors, with medians and interquartile ranges of 45 (43 to 48), 75 (71 to 81), 107 (102 to 110) and 135 (133
to 139) minutes, and a terminal fall at 165 (163 to 170). Five events for the five additions after basal, about thirty
minutes apart, which is the staircase the published sequence predicts. The small step at 75 minutes is the 3 mM
glucose addition, which is below the beta-cell threshold and should give little, and the largest step at 135 minutes
is the uncoupler, as expected. Phase windows start about ten minutes after each onset and end about three minutes
before the next. The assignment should still be checked against the companion file
HPAP-xxx_Oxygen-consumption_figure.pdf, which PANC-DB lists for each donor and which was not in the download.

Features, on the log scale, named for the standard mitochondrial quantities they correspond to:
    basal, aam, glucose_3, glucose_16, maximal (FCCP), terminal (azide, often truncated)
    aam_response, glucose_3_response, glucose_16_response, spare_capacity (maximal minus basal), level
Quality control flags short traces, flat traces, traces that duplicate another donor exactly or up to a scale factor,
and traces where the terminal inhibitor phase does not appear. Both members of a duplicated pair are excluded, because
there is no way to tell which donor the trace belongs to. Their hormone data are unaffected and stay in every other
analysis.
Models, following the second review:
    1 oxygen ~ HbA1c + covariates
    2 oxygen ~ T2D * sex + covariates
    3 glucagon suppression ~ oxygen + HbA1c + covariates
    4 glucagon suppression ~ oxygen * sex + HbA1c + covariates
Attenuation, not mediation. Adding a covariate and watching the HbA1c coefficient move is reported as attenuation
after adjustment. No formal mediation assumptions are claimed.
Outputs: results/tables/oxygen_qc.tsv, oxygen_features.tsv, oxygen_stability.tsv, oxygen_associations.tsv,
         oxygen_covariate_attenuation.tsv, data/processed/donor_oxygen.tsv, data/processed/oxygen_long.tsv.gz
"""
import sys, re, glob, hashlib
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))

OUT = ROOT / 'results/tables'
RAW = ROOT / 'data/raw/hpap/hpapdata'
GS = 'Glucose suppression (3 to 16.7 mM)'
COV = 'age + male + bmi + isolation_year'
# Phase windows in minutes. The switch times were measured, not guessed. For every usable trace the smoothed
# derivative was searched for its largest rise inside each candidate interval, giving onsets with a median and
# interquartile range of 45 (43 to 48), 75 (71 to 81), 107 (102 to 110), 135 (133 to 139) and, for the terminal fall,
# 165 (163 to 170) minutes. Five events for the five additions after basal, spaced about thirty minutes apart. Each
# window starts about ten minutes after its onset, to clear the chamber delay, and ends about three minutes before the
# next onset. ONSETS records the intervals searched.
PHASE = {'basal': (20, 42), 'aam': (52, 72), 'glucose_3': (82, 104), 'glucose_16': (114, 131), 'maximal': (145, 161)}
ONSETS = {'aam': (35, 55), 'glucose_3': (60, 95), 'glucose_16': (98, 125), 'maximal': (126, 148)}
GRID = np.arange(0, 200, 1.0)


def load_one(f):
    xl = pd.ExcelFile(f); sh = 'Calculations' if 'Calculations' in xl.sheet_names else xl.sheet_names[0]
    d = pd.read_excel(f, sheet_name=sh, header=None, skiprows=2, names=['t', 'pin', 'pout', 'ocr'])
    return d.apply(pd.to_numeric, errors='coerce').dropna(subset=['t', 'ocr'])


def parse():
    recs, traces = [], {}
    for f in sorted(glob.glob(str(RAW / '*/Islet Studies/Islet physiology studies/Oxygen consumption/*.xlsx'))):
        donor = re.search(r'HPAP-\d+', f).group(0)
        d = load_one(f)
        r = {'donor': donor, 'md5': hashlib.md5(open(f, 'rb').read()).hexdigest(), 'n_points': len(d),
             't_max': float(d.t.max()) if len(d) else np.nan, 'ocr_median': float(d.ocr.median()) if len(d) else np.nan}
        if len(d) >= 600 and d.ocr.median() > 0 and d.t.max() >= 160:
            v = np.interp(GRID, d.t.values, d.ocr.values, left=np.nan, right=np.nan)
            traces[donor] = v
            g = pd.Series(np.log(np.where(v > 0, v, np.nan)), index=GRID)
            for k, (a, b) in PHASE.items(): r[k] = float(g.loc[a:b].median())
            tail = d[d.t >= d.t.max() - 5]
            r['terminal'] = float(np.log(max(tail.ocr.median(), 1e-6)))
            r['level'] = float(g.loc[20:160].median())
            r['range_raw'] = float(np.nanmax(v[20:161]) - np.nanmin(v[20:161]))
        recs.append(r)
    F = pd.DataFrame(recs)
    # quality control
    F['flag_short'] = ~((F.n_points >= 600) & (F.t_max >= 160) & (F.ocr_median > 0))
    F['flag_flat'] = False; F['flag_duplicate'] = False; F['duplicate_of'] = ''; F['variability'] = np.nan
    ks = list(traces)
    for k in ks:
        v = traces[k]; m = ~np.isnan(v)
        rel = np.ptp(v[m]) / max(np.nanmedian(v), 1e-9)       # range over median, the flatness statistic
        F.loc[F.donor == k, 'variability'] = rel
        if rel < 0.05: F.loc[F.donor == k, 'flag_flat'] = True
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = traces[ks[i]], traces[ks[j]]; m = ~np.isnan(a) & ~np.isnan(b)
            if m.sum() < 100: continue
            c = np.median(b[m] / np.where(a[m] == 0, np.nan, a[m]))
            if np.nanmax(np.abs(b[m] - c * a[m])) / max(np.nanmax(np.abs(b[m])), 1e-9) < 0.01:
                # neither member of a duplicated pair can be trusted, because there is no way to tell which donor the
                # trace belongs to, so both are excluded from the oxygen analyses
                F.loc[F.donor == ks[j], ['flag_duplicate', 'duplicate_of']] = [True, ks[i]]
                F.loc[F.donor == ks[i], ['flag_duplicate', 'duplicate_of']] = [True, ks[j]]
    # The terminal inhibitor phase is only usable if respiration falls well below basal by the end of the run. It
    # mostly does not. The ratio does not improve in longer runs, so this is not simple truncation, and a
    # non-mitochondrial floor cannot be read from these files at any run length.
    F['terminal_ratio'] = np.exp(F.terminal - F.basal) if 'terminal' in F else np.nan
    F['flag_no_inhibitor'] = ~(F.terminal_ratio < 0.8) if 'terminal' in F else True
    F['usable'] = ~(F.flag_short | F.flag_flat | F.flag_duplicate)
    F['aam_response'] = F.aam - F.basal
    F['glucose_3_response'] = F.glucose_3 - F.aam
    F['glucose_16_response'] = F.glucose_16 - F.glucose_3
    F['spare_capacity'] = F.maximal - F.basal
    L = pd.concat([pd.DataFrame({'donor': k, 't': GRID, 'ocr': traces[k]}) for k in traces])
    return F, L


FEATS = ['level', 'basal', 'aam', 'glucose_3', 'glucose_16', 'maximal', 'spare_capacity',
         'aam_response', 'glucose_3_response', 'glucose_16_response']


def onsets(L, F):
    """Measure the stimulus switch times from the traces themselves. Each trace is divided by its own basal median,
    the derivative is smoothed over five minutes, and the time of the largest rise inside each candidate interval is
    taken as that switch. The terminal fall uses the largest drop between 150 and 180 minutes. Reporting the spread
    across donors is the check that the protocol is common to all runs rather than fitted to the average trace."""
    P = L.pivot(index='donor', columns='t', values='ocr')
    P = P.loc[[d for d in P.index if d in set(F[F.usable].donor)]]
    N = P.div(P.loc[:, 20:42].median(1), axis=0)
    tt = N.columns.values.astype(float)
    rows = []
    for nm, (lo, hi) in list(ONSETS.items()) + [('terminal fall', (150, 180))]:
        w = (tt >= lo) & (tt <= hi); got = []
        for dn in N.index:
            g = np.convolve(np.gradient(N.loc[dn].values.astype(float), tt), np.ones(5) / 5, mode='same')
            if np.isnan(g[w]).all(): continue
            got.append(tt[w][np.nanargmin(g[w]) if nm == 'terminal fall' else np.nanargmax(g[w])])
        got = np.array(got)
        rows.append({'switch': nm, 'searched_from': lo, 'searched_to': hi, 'n': len(got),
                     'onset_median': float(np.median(got)), 'onset_q25': float(np.percentile(got, 25)),
                     'onset_q75': float(np.percentile(got, 75))})
    return pd.DataFrame(rows)


def stability(L, F):
    """Within-trace stability, not reproducibility. Alternate minutes of each phase form two halves, correlated across
    donors and corrected by the Spearman-Brown formula. Adjacent points are autocorrelated, so this measures how
    steady the trace is inside a phase. It is not technical or biological replication, for which there is one run."""
    from scipy import stats as st
    P = L.pivot(index='donor', columns='t', values='ocr').apply(np.log)
    P = P.loc[[d for d in P.index if d in set(F[F.usable].donor)]]
    rows = []
    h = {}
    for k, (a, b) in PHASE.items():
        cols = [c for c in P.columns if a <= c <= b]
        h[k] = (P[cols[0::2]].median(1), P[cols[1::2]].median(1))
    cols = [c for c in P.columns if 20 <= c <= 160]
    h['level'] = (P[cols[0::2]].median(1), P[cols[1::2]].median(1))
    h['spare_capacity'] = (h['maximal'][0] - h['basal'][0], h['maximal'][1] - h['basal'][1])
    h['aam_response'] = (h['aam'][0] - h['basal'][0], h['aam'][1] - h['basal'][1])
    h['glucose_3_response'] = (h['glucose_3'][0] - h['aam'][0], h['glucose_3'][1] - h['aam'][1])
    h['glucose_16_response'] = (h['glucose_16'][0] - h['glucose_3'][0], h['glucose_16'][1] - h['glucose_3'][1])
    for k, (x, y) in h.items():
        m = x.notna() & y.notna()
        r = float(st.pearsonr(x[m], y[m])[0])
        rows.append({'feature': k, 'n': int(m.sum()), 'half_half_r': r, 'within_trace_stability': 2 * r / (1 + r)})
    return pd.DataFrame(rows)


def sex_checks(d, feats, reps=20000, seed=7):
    """The T2D by sex interaction rests on a small and unbalanced cell, so it gets three checks that do not rely on
    the HC3 standard error. First, the sex difference among donors without diabetes, which is the baseline contrast
    Chen et al. report. Second, leave-one-donor-out, reporting the range of the interaction estimate and its P value.
    Third, a permutation test that shuffles the sex label within diabetes group, which holds the group sizes and the
    covariate structure fixed and tests the interaction against an exact null. One set of permuted labels is shared
    across features, so the permutation P values are comparable, and the interaction coefficient is taken by least
    squares rather than by refitting with robust errors, which agrees to machine precision and is much faster."""
    rng = np.random.default_rng(seed)
    fit = lambda x, f: smf.ols(f'{f} ~ T2D*male + age + bmi + isolation_year', data=x).fit(cov_type='HC3')
    Y = d[feats].values
    T, M = d.T2D.values.astype(float), d.male.values.astype(float)
    base = np.column_stack([np.ones(len(d)), T, d.age.values, d.bmi.values, d.isolation_year.values])

    def coef(m):
        """Interaction coefficients for every feature at once, given a sex label vector."""
        X = np.column_stack([base[:, :2], m, T * m, base[:, 2:]])
        return np.linalg.lstsq(X, Y, rcond=None)[0][3]

    b0 = coef(M)
    grp = d.group.values
    idx = [np.where(grp == g)[0] for g in np.unique(grp)]
    cnt = np.zeros(len(feats))
    for _ in range(reps):
        m = M.copy()
        for ix in idx: m[ix] = rng.permutation(m[ix])
        cnt += np.abs(coef(m)) >= np.abs(b0) - 1e-12
    pperm = (cnt + 1) / (reps + 1)

    rows = []
    for j, f_ in enumerate(feats):
        mm = fit(d, f_)
        mc = smf.ols(f'{f_} ~ male + age + bmi + isolation_year', data=d[d.T2D == 0]).fit(cov_type='HC3')
        loo = [fit(d.drop(i), f_) for i in d.index]
        lb = np.array([x.params['T2D:male'] for x in loo]); lp = np.array([x.pvalues['T2D:male'] for x in loo])
        rows.append({'feature': f_, 'n': int(mm.nobs), 'interaction': b0[j], 'p_hc3': mm.pvalues['T2D:male'],
                     'p_permutation': pperm[j], 'permutation_reps': reps,
                     'loo_estimate_min': lb.min(), 'loo_estimate_max': lb.max(), 'loo_p_max': lp.max(),
                     'loo_n_below_05': int((lp < 0.05).sum()),
                     'control_male_minus_female': mc.params['male'], 'control_p': mc.pvalues['male'],
                     'n_control': int(mc.nobs), 'n_male_T2D': int(((d.T2D == 1) & (d.male == 1)).sum())})
    R = pd.DataFrame(rows)
    R['fdr_permutation'] = multipletests(R.p_permutation, method='fdr_bh')[1]
    return R


def main():
    F, L = parse()
    F.to_csv(OUT / 'oxygen_features.tsv', sep='\t', index=False)
    L.to_csv(ROOT / 'data/processed/oxygen_long.tsv.gz', sep='\t', index=False)
    qc = F[['donor', 'md5', 'n_points', 't_max', 'ocr_median', 'terminal_ratio', 'variability', 'flag_short',
            'flag_flat', 'flag_duplicate', 'duplicate_of', 'flag_no_inhibitor', 'usable']]
    qc.to_csv(OUT / 'oxygen_qc.tsv', sep='\t', index=False)
    print(f'parsed {len(F)} files, usable {int(F.usable.sum())}')
    print('  duplicates:', F[F.flag_duplicate][['donor', 'duplicate_of']].values.tolist())
    print('  flat:', F[F.flag_flat].donor.tolist(), ' short:', F[F.flag_short].donor.tolist())
    print(f'  traces without a clear terminal inhibitor phase: {int(F.flag_no_inhibitor.sum())} of {len(F)}')
    print(f'  terminal rate relative to basal, median {F[F.usable].terminal_ratio.median():.3f}; '
          f'relative to the uncoupled maximum, median {np.exp(F[F.usable].terminal - F[F.usable].maximal).median():.3f}')
    print('  terminal inhibition against run length (not simple truncation):')
    for lo, hi in [(160, 185), (185, 195), (195, 230), (230, 400)]:
        x = F[F.usable & (F.t_max >= lo) & (F.t_max < hi)]
        if len(x): print(f'    t_max {lo} to {hi}: n={len(x):2d}, median ratio {x.terminal_ratio.median():.3f}, '
                         f'fraction below 0.8 {np.mean(x.terminal_ratio < 0.8):.2f}')
    print(f'  duplicated pairs are distinct files, md5 collisions across all {len(F)}: {int(F.md5.duplicated().sum())}')
    O = onsets(L, F); O.to_csv(OUT / 'oxygen_onsets.tsv', sep='\t', index=False)
    print('\nmeasured switch onsets, minutes'); print(O.round(1).to_string(index=False))
    S = stability(L, F); S.to_csv(OUT / 'oxygen_stability.tsv', sep='\t', index=False)
    print(S.round(3).to_string(index=False))

    D = pd.read_csv(ROOT / 'data/processed/donors.tsv', sep='\t').set_index('donor')
    E = pd.read_csv(ROOT / 'data/processed/donor_effects_censored_Penn_glucagon.tsv', sep='\t', index_col=0)
    EI = pd.read_csv(ROOT / 'data/processed/donor_effects_censored_Penn_insulin.tsv', sep='\t', index_col=0)
    use = set(F[F.usable].donor)
    d = D.loc[[x for x in D.index if x in use]].copy()
    d = d[d.group.isin(['control', 'T2D'])].copy()
    d['T2D'] = (d.group == 'T2D').astype(int); d['male'] = (d.sex == 'Male').astype(int)
    d['isolation_year'] = d.isolation_year.fillna(d.isolation_year.median())
    d = d.join(F.set_index('donor')[FEATS])
    d['suppression'] = E[f'gls_{GS}'].reindex(d.index)
    d['insulin_glucose_response'] = EI[f'gls_{GS}'].reindex(d.index)
    d = d.dropna(subset=['hba1c', 'age', 'bmi'] + FEATS)
    d.to_csv(ROOT / 'data/processed/donor_oxygen.tsv', sep='\t')
    ns = int(d.suppression.notna().sum())
    print(f'donors with oxygen and covariates {len(d)} ({int(d.male.sum())} male), with a Penn glucagon curve {ns}')

    rows = []
    for f_ in FEATS:
        x = d.rename(columns={f_: 'y'})
        m1 = smf.ols(f'y ~ hba1c + {COV}', data=x).fit(cov_type='HC3')
        rows.append({'model': '1 oxygen against HbA1c', 'feature': f_, 'term': 'hba1c', 'n': int(m1.nobs),
                     'estimate': m1.params['hba1c'], 'se': m1.bse['hba1c'], 'p': m1.pvalues['hba1c'],
                     'estimate_SD': m1.params['hba1c'] / x.y.std()})
        m2 = smf.ols(f'y ~ T2D*male + age + bmi + isolation_year', data=x).fit(cov_type='HC3')
        rows.append({'model': '2 oxygen against T2D by sex', 'feature': f_, 'term': 'T2D:male', 'n': int(m2.nobs),
                     'estimate': m2.params['T2D:male'], 'se': m2.bse['T2D:male'], 'p': m2.pvalues['T2D:male'],
                     'estimate_SD': m2.params['T2D:male'] / x.y.std()})
        rows.append({'model': '2 oxygen against T2D by sex', 'feature': f_, 'term': 'T2D in females', 'n': int(m2.nobs),
                     'estimate': m2.params['T2D'], 'se': m2.bse['T2D'], 'p': m2.pvalues['T2D'],
                     'estimate_SD': m2.params['T2D'] / x.y.std()})
        rows.append({'model': '2 oxygen against T2D by sex', 'feature': f_, 'term': 'T2D in males', 'n': int(m2.nobs),
                     'estimate': m2.params['T2D'] + m2.params['T2D:male'], 'se': np.nan, 'p': np.nan,
                     'estimate_SD': (m2.params['T2D'] + m2.params['T2D:male']) / x.y.std()})
        ds = d.dropna(subset=['suppression']).rename(columns={f_: 'ox'})
        m3 = smf.ols(f'suppression ~ ox + hba1c + {COV}', data=ds).fit(cov_type='HC3')
        rows.append({'model': '3 suppression against oxygen', 'feature': f_, 'term': 'oxygen', 'n': int(m3.nobs),
                     'estimate': m3.params['ox'], 'se': m3.bse['ox'], 'p': m3.pvalues['ox'],
                     'estimate_SD': m3.params['ox'] * ds.ox.std() / ds.suppression.std()})
        m4 = smf.ols(f'suppression ~ ox*male + hba1c + age + bmi + isolation_year', data=ds).fit(cov_type='HC3')
        rows.append({'model': '4 suppression against oxygen by sex', 'feature': f_, 'term': 'oxygen by male', 'n': int(m4.nobs),
                     'estimate': m4.params['ox:male'], 'se': m4.bse['ox:male'], 'p': m4.pvalues['ox:male'],
                     'estimate_SD': m4.params['ox:male'] * ds.ox.std() / ds.suppression.std()})
    T = pd.DataFrame(rows)
    T['fdr'] = T.groupby(['model', 'term']).p.transform(lambda p: multipletests(p.fillna(1), method='fdr_bh')[1])
    T.to_csv(OUT / 'oxygen_associations.tsv', sep='\t', index=False)

    ds = d.dropna(subset=['suppression'])
    b0 = smf.ols(f'suppression ~ hba1c + {COV}', data=ds).fit(cov_type='HC3'); sd = ds.suppression.std()
    att = [{'added': 'nothing', 'n': int(b0.nobs), 'beta_hba1c': b0.params['hba1c'], 'p': b0.pvalues['hba1c'],
            'beta_SD': b0.params['hba1c'] / sd, 'change_pct': 0.0}]
    for f_ in FEATS + ['insulin_glucose_response']:
        m = smf.ols(f'suppression ~ hba1c + {COV} + {f_}', data=ds).fit(cov_type='HC3')
        att.append({'added': f_, 'n': int(m.nobs), 'beta_hba1c': m.params['hba1c'], 'p': m.pvalues['hba1c'],
                    'beta_SD': m.params['hba1c'] / sd,
                    'change_pct': 100 * (m.params['hba1c'] - b0.params['hba1c']) / b0.params['hba1c']})
    A = pd.DataFrame(att); A.to_csv(OUT / 'oxygen_covariate_attenuation.tsv', sep='\t', index=False)
    X = sex_checks(d, FEATS, reps=int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
    X.to_csv(OUT / 'oxygen_sex_checks.tsv', sep='\t', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 36)
    for mdl in T.model.unique():
        print('\n==', mdl); print(T[T.model == mdl].drop(columns='model').round(4).to_string(index=False))
    print('\n== attenuation'); print(A.round(4).to_string(index=False))
    print('\n== sex interaction checks'); print(X.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
