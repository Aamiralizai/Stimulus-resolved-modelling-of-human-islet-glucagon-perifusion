"""Step 12 - insulin and glucagon coordination within the same perifusion run (Gate 1.5).

Insulin and glucagon are measured in the same Penn run, so the reciprocal relation between them can be read per donor
rather than treating insulin only as a covariate. Two coupling measures are derived, both on the log scale.
  window slope   the slope of a donor's seven glucagon window effects on the same donor's seven insulin window
                 effects. A reciprocal islet gives a negative slope. Window effects average many fractions, so this is
                 much less affected by within-run noise than a fraction-level correlation.
  fraction corr  the correlation across fractions between the donor's insulin and glucagon residual trajectories, once
                 the population mean curve of each hormone is removed. This is more sensitive but shares the technical
                 noise of the run (flow rate, fraction collection, islet number), which biases it towards +1.
Both are regressed on HbA1c and on T2D status with the usual covariates. A reciprocal-response composite, the mean of
the z scores of glucagon suppression and the insulin response to glucose, is reported for comparison.
Reliability of the window slope is estimated by a parametric bootstrap: donors are simulated from the fitted models
with no donor-level coupling differences, and the spread of the resulting slopes gives the null variation.
Outputs: results/tables/bihormonal_coordination.tsv, data/processed/donor_bihormonal.tsv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from glucv2.model import WindowModel
from glucv2.data import load_matrix, contrast_vector, WINDOWS

OUT = ROOT / 'results/tables'
GS = 'Glucose suppression (3 to 16.7 mM)'
BASE = 'y ~ {pred} + age + male + bmi + isolation_year'


def slope(x, y):
    x = x - x.mean(); return float(x @ (y - y.mean()) / (x @ x)) if x @ x > 0 else np.nan


def main(B=200, seed=4):
    Dg, Di = load_matrix('Penn', 'glucagon'), load_matrix('Penn', 'insulin')
    common = [d for d in Dg['donors'] if d in set(Di['donors'])]
    pg = {d: k for k, d in enumerate(Dg['donors'])}; pi = {d: k for k, d in enumerate(Di['donors'])}
    mg = WindowModel(Dg['R'], Dg['frac'], Dg['win']); fg = mg.fit('full')
    mi = WindowModel(Di['R'], Di['frac'], Di['win']); fi = mi.fit('full')
    gg, _, _ = mg.donor_estimates(fg); gi, _, _ = mi.donor_estimates(fi)
    rows = []
    for d in common:
        wg, wi = gg[pg[d]], gi[pi[d]]
        rg, ri = Dg['R'][pg[d]], Di['R'][pi[d]]
        rows.append({'donor': d, 'window_slope': slope(wi, wg), 'fraction_corr': float(np.corrcoef(ri, rg)[0, 1]),
                     'glucagon_suppression': float(wg @ contrast_vector('Penn', GS)),
                     'insulin_glucose_response': float(wi @ contrast_vector('Penn', GS)),
                     'glucagon_capacity': float(wg.mean()), 'insulin_capacity': float(wi.mean())})
    P = pd.DataFrame(rows).set_index('donor')
    z = lambda s: (s - s.mean()) / s.std()
    P['reciprocal_composite'] = (z(-P.glucagon_suppression) + z(P.insulin_glucose_response)) / 2
    # null spread of the window slope when donors differ only in level, not in coupling
    rng = np.random.default_rng(seed); null = []
    for b in range(B):
        Rg, _ = mg.simulate(fg, N=len(common), rng=rng.integers(1 << 31))
        Ri, _ = mi.simulate(fi, N=len(common), rng=rng.integers(1 << 31))
        mgb = WindowModel(Rg - Rg.mean(0), mg.frac, mg.win); mib = WindowModel(Ri - Ri.mean(0), mi.frac, mi.win)
        gb, _, _ = mgb.donor_estimates(fg); ib, _, _ = mib.donor_estimates(fi)
        null.append(np.std([slope(ib[k], gb[k]) for k in range(len(common))]))
    null_sd = float(np.mean(null))
    P.to_csv(ROOT / 'data/processed/donor_bihormonal.tsv', sep='\t')
    D = pd.read_csv(ROOT / 'data/processed/donors.tsv', sep='\t').set_index('donor')
    d = D.loc[[x for x in P.index if x in D.index]].copy(); d = d[d.group.isin(['control', 'T2D'])].copy()
    d['T2D'] = (d.group == 'T2D').astype(int); d['male'] = (d.sex == 'Male').astype(int)
    d['isolation_year'] = d.isolation_year.fillna(d.isolation_year.median())
    d = d.join(P)
    out = []
    for outcome in ['window_slope', 'fraction_corr', 'reciprocal_composite']:
        for pred in ['hba1c', 'T2D']:
            dd = d.rename(columns={outcome: 'y'}).dropna(subset=['y', 'hba1c', 'age', 'bmi'])
            f = smf.ols(BASE.format(pred=pred), data=dd).fit(cov_type='HC3'); sd = dd.y.std(); ci = f.conf_int()
            out.append({'outcome': outcome, 'predictor': pred, 'n': int(f.nobs), 'mean': float(dd.y.mean()), 'sd': sd,
                        'beta': f.params[pred], 'se': f.bse[pred], 'p': f.pvalues[pred], 'beta_SD': f.params[pred] / sd,
                        'ci_low_SD': ci.loc[pred, 0] / sd, 'ci_high_SD': ci.loc[pred, 1] / sd})
    T = pd.DataFrame(out)
    T.loc[len(T)] = {'outcome': 'window_slope', 'predictor': 'observed SD against the null SD', 'n': len(P),
                     'mean': float(P.window_slope.mean()), 'sd': float(P.window_slope.std()), 'beta': null_sd,
                     'se': np.nan, 'p': np.nan, 'beta_SD': float(P.window_slope.std() / null_sd), 'ci_low_SD': np.nan, 'ci_high_SD': np.nan}
    r, p = stats.spearmanr(P.window_slope, P.fraction_corr)
    T.loc[len(T)] = {'outcome': 'agreement of the two measures', 'predictor': 'Spearman', 'n': len(P), 'mean': np.nan,
                     'sd': np.nan, 'beta': r, 'se': np.nan, 'p': p, 'beta_SD': np.nan, 'ci_low_SD': np.nan, 'ci_high_SD': np.nan}
    T.to_csv(OUT / 'bihormonal_coordination.tsv', sep='\t', index=False)
    pd.set_option('display.width', 250); print(f'null SD of the window slope {null_sd:.3f}, observed {P.window_slope.std():.3f}')
    print(T.round(4).to_string(index=False))


if __name__ == '__main__':
    main(*[int(x) for x in sys.argv[1:]])
