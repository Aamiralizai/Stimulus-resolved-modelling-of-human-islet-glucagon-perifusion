"""Step 09 - left-censored fit of the window model, and the HbA1c association under censoring (Gate 1.5).

Assay-floor values are left-censored. At Penn they are not spread evenly: 17.7% of basal glucagon fractions sit at the
floor, and the number of censored fractions in the amino-acid windows rises with donor HbA1c. Those are the windows
that carry the HbA1c association, so the floor has to be modelled rather than ignored.

Part 1, validation on simulated data with a known truth.
  scenario NULL    donors differ in overall glucagon output, output falls with a simulated HbA1c at the rate estimated
                   from the real data, and there is NO HbA1c effect on glucose suppression. Tests whether the floor
                   alone can manufacture one.
  scenario EFFECT  the same, plus the suppression effect estimated from the real data. Tests recovery.
  Effects are expressed per SD of HbA1c, because the simulated HbA1c is standard normal.
  Each replicate is analysed three ways: no floor (the unreachable ideal), floor ignored (the Gate 1 analysis) and
  floor modelled by EM.
Part 2, real data. The censored model is fitted to each laboratory and hormone. Donor estimates come from 20 multiple
imputations of the censored values, and the HbA1c regressions are combined with Rubin's rules, so the reported
standard errors include the uncertainty that censoring adds. A complete-case analysis restricted to runs with no
censored fraction is reported alongside.
Outputs: results/tables/censored_validation.tsv, censored_model_summary.tsv, censored_clinical.tsv,
         data/processed/donor_effects_censored_{lab}_{hormone}.tsv
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from glucv2.model import WindowModel, decompose
from glucv2.data import load_matrix, contrast_vector, CONTRASTS, WINDOWS
from glucv2.censored import CensoredFit, rubin

OUT = ROOT / 'results/tables'
GS = {'Penn': 'Glucose suppression (3 to 16.7 mM)', 'Vanderbilt': 'Glucose suppression (5.6 to 16.7 mM)'}


# ---------------------------------------------------------------- part 1
def simulate(D, f, beta_cap, beta_supp, N, rng):
    """Curves for N donors with a standard-normal HbA1c, an effect on overall output and one on glucose suppression."""
    m = WindowModel(np.zeros((N, D['R'].shape[1])), D['frac'], D['win'])
    R, _ = m.simulate(f, N=N, rng=rng.integers(1 << 31))
    h = rng.standard_normal(N); W = len(D['windows'])
    a = contrast_vector('Penn', GS['Penn'])
    shift = np.outer(h, beta_cap * np.ones(W) / 1.0 + beta_supp * a / (a @ a))
    return D['mu'] + R + shift @ m.Z.T, h


def analyse(L, thr, cens, D, h, kind='full'):
    """Regression of the suppression contrast on the simulated HbA1c, for one completed data set."""
    a = contrast_vector('Penn', GS['Penn'])
    if cens is None:
        mu = L.mean(0); m = WindowModel(L - mu, D['frac'], D['win']); f = m.fit(kind)
        gls, _, _ = m.donor_estimates(f)
    else:
        cf = CensoredFit(L, cens, thr, D['frac'], D['win'], seed=1); f = cf.fit(kind, iters=12)
        gls, _, _ = cf.donor_estimates(); m = cf.model
    y = gls @ a
    b = np.polyfit(h, y, 1)[0]; sdy = y.std()
    return {'beta': b, 'beta_SD': b / sdy, 'supp_SD': float(np.sqrt(max(a @ f['Sigma'] @ a, 0))),
            'specific_share': decompose(f['Sigma'])['specific_share']}


def validate(R=8, seed=3):
    D = load_matrix('Penn', 'glucagon'); m = WindowModel(D['R'], D['frac'], D['win']); f = m.fit('full')
    a = contrast_vector('Penn', GS['Penn'])
    # the floor is placed to reproduce the observed share of censored fractions
    target = D['censored'].mean(); floor = np.quantile(D['Y'][~D['censored']], target)
    # effects are the real Penn estimates rescaled to one SD of HbA1c, since the simulated HbA1c is standard normal
    CL = pd.read_csv(OUT / 'clinical_associations.tsv', sep='\t'); CL = CL[CL.predictor == 'HbA1c'].set_index('outcome')
    hb_sd = 1.539   # SD of HbA1c in the analysed Penn donors
    b_cap = float(CL.loc['Penn | glucagon | capacity', 'beta']) * hb_sd
    b_sup = float(CL.loc[f'Penn | glucagon | {GS["Penn"]}', 'beta']) * hb_sd
    rng = np.random.default_rng(seed); rows = []
    for name, bc, bs in [('NULL (no HbA1c effect on suppression)', b_cap, 0.0), ('EFFECT (suppression effect as observed)', b_cap, b_sup)]:
        for r in range(R):
            L0, h = simulate(D, f, bc, bs, len(D['donors']), rng)
            cens = L0 <= floor; L = np.where(cens, floor, L0); thr = np.full_like(L, floor)
            for how, args in [('no floor', (L0, None, None)), ('floor ignored', (L, None, None)), ('floor modelled', (L, thr, cens))]:
                Lx, tx, cx = args
                res = analyse(Lx, tx, cx, D, h)
                rows.append({'scenario': name, 'rep': r, 'analysis': how, 'true_beta': bs, **res})
                print(name, r, how, round(res['beta'], 4), flush=True)
    T = pd.DataFrame(rows); T.to_csv(OUT / 'censored_validation.tsv', sep='\t', index=False)
    S = T.groupby(['scenario', 'analysis']).agg(reps=('beta', 'size'), true_beta=('true_beta', 'first'),
                                                beta_median=('beta', 'median'), beta_lo=('beta', lambda x: np.quantile(x, 0.1)),
                                                beta_hi=('beta', lambda x: np.quantile(x, 0.9)), supp_SD_median=('supp_SD', 'median'),
                                                share_median=('specific_share', 'median')).reset_index()
    S.to_csv(OUT / 'censored_validation_summary.tsv', sep='\t', index=False)
    print(S.round(4).to_string(index=False))


# ---------------------------------------------------------------- part 2
def fit_real(n_imp=20):
    summ, clin = [], []
    D0 = pd.read_csv(ROOT / 'data/processed/donors.tsv', sep='\t').set_index('donor')
    for lab in ['Penn', 'Vanderbilt']:
        for hormone in ['glucagon', 'insulin']:
            D = load_matrix(lab, hormone)
            cf = CensoredFit(D['Y'], D['censored'], D['threshold'], D['frac'], D['win'], seed=11)
            f = cf.fit('full', iters=25, verbose=True)
            print(f'[{lab} {hormone}] censored EM done, {f["em_iters"]} iterations, loglik {f["loglik"]:.1f}', flush=True)
            gls, blup, Ainv = cf.donor_estimates(); dec = decompose(f['Sigma'])
            idx = pd.Index(D['donors'], name='donor')
            E = pd.DataFrame(gls, columns=[f'gls_{w}' for w in D['windows']], index=idx)
            E['gls_capacity'] = gls.mean(1)
            for name in CONTRASTS[lab]:
                v = contrast_vector(lab, name); E[f'gls_{name}'] = gls @ v; E[f'se_{name}'] = np.sqrt(v @ Ainv @ v)
            E['n_censored'] = D['censored'].sum(1)
            E.to_csv(ROOT / f'data/processed/donor_effects_censored_{lab}_{hormone}.tsv', sep='\t')
            summ.append({'lab': lab, 'hormone': hormone, 'N': len(idx), 'censored_share': float(D['censored'].mean()),
                         'em_iterations': f['em_iters'], 'capacity_variance': dec['capacity_var'],
                         'specific_share': dec['specific_share'], 'rho': f['rho'], 'eta': f['eta'],
                         **{f'{n}_true_SD': float(np.sqrt(max(contrast_vector(lab, n) @ f['Sigma'] @ contrast_vector(lab, n), 0))) for n in CONTRASTS[lab]}})
            json.dump({'Sigma': f['Sigma'].tolist(), 'windows': D['windows'], 'mu': f['mu'].tolist(),
                       's2': np.atleast_1d(f['s2']).tolist(), 'rho': f['rho'], 'eta': f['eta']},
                      open(ROOT / f'data/processed/model_censored_{lab}_{hormone}.json', 'w'))
            if hormone == 'glucagon':
                clin += clinical(cf, D, lab, idx, D0, n_imp)
    pd.DataFrame(summ).to_csv(OUT / 'censored_model_summary.tsv', sep='\t', index=False)
    C = pd.DataFrame(clin); C.to_csv(OUT / 'censored_clinical.tsv', sep='\t', index=False)
    pd.set_option('display.width', 250); print(pd.DataFrame(summ).round(3).to_string(index=False)); print(C.round(4).to_string(index=False))


def clinical(cf, D, lab, idx, D0, n_imp):
    """HbA1c regressions on the censored donor estimates, pooled over multiple imputations (Rubin's rules)."""
    cov = D0.loc[[d for d in idx if d in D0.index]].copy()
    cov = cov[cov.group.isin(['control', 'T2D'])].copy()
    cov['T2D'] = (cov.group == 'T2D').astype(int); cov['male'] = (cov.sex == 'Male').astype(int)
    cov['isolation_year'] = cov.isolation_year.fillna(cov.isolation_year.median())
    pos = {d: k for k, d in enumerate(idx)}
    names = {'capacity': np.ones(len(D['windows'])) / len(D['windows'])}
    for n in CONTRASTS[lab]: names[n] = contrast_vector(lab, n)
    if lab == 'Penn':
        w = list(D['windows'])
        names['window AA+G3 (capacity-adjusted)'] = np.eye(len(w))[w.index('AA+G3')] - np.ones(len(w)) / len(w)
        names['window AA (capacity-adjusted)'] = np.eye(len(w))[w.index('AA')] - np.ones(len(w)) / len(w)
    imps = cf.imputations(n_imp)
    form = 'y ~ hba1c + age + male + bmi + isolation_year'
    rows = []
    for name, a in names.items():
        est, var, sds = [], [], []
        for im in imps:
            y = pd.Series(im['gls'] @ a, index=idx)
            d = cov.copy(); d['y'] = y.reindex(d.index); d = d.dropna(subset=['y', 'hba1c', 'age', 'bmi'])
            fit = smf.ols(form, data=d).fit(cov_type='HC3')
            est.append(fit.params['hba1c']); var.append(fit.bse['hba1c'] ** 2); sds.append(d.y.std())
        p = rubin(est, var); s = float(np.mean(sds))
        rows.append({'lab': lab, 'outcome': name, 'n': int(len(d)), 'n_imputations': n_imp, 'beta': p['estimate'], 'se': p['se'],
                     'p': p['p'], 'beta_SD': p['estimate'] / s, 'ci_low_SD': (p['estimate'] - 1.96 * p['se']) / s,
                     'ci_high_SD': (p['estimate'] + 1.96 * p['se']) / s, 'fraction_missing_information': p['fmi']})
        print('  ', lab, name, round(rows[-1]['beta_SD'], 3), 'P', f"{p['p']:.3g}", flush=True)
    return rows


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('validate', 'all'): validate(*[int(x) for x in sys.argv[2:3]] if len(sys.argv) > 2 else [])
    if what in ('real', 'all'): fit_real()
