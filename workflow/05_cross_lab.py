"""Step 05 - do stimulus-specific responses replicate between labs? (HPAP donors perifused at Penn and at Vanderbilt)

Islets from the same donor were perifused independently at Penn and at Vanderbilt with different protocols. A donor
trait should agree between the two runs. A run-level or technical effect should not.
For matched constructs we report the Pearson correlation of the donor GLS estimates (unshrunk), a donor-bootstrap 95% CI,
and the correlation of the true effects: cov(GLS_Penn, GLS_Vanderbilt) / sqrt(var_true_Penn * var_true_Vanderbilt),
with the true-effect variances from each lab's fitted Sigma (sampling errors are independent between labs). Its donor-
bootstrap CI holds the true-effect variances fixed, so it ignores their estimation error and is too narrow.
The same is reported for conventional summary traits computed from the same curves.
Outputs: results/tables/crosslab_replication.tsv
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from glucv2.data import contrast_vector, WINDOWS
from glucv2.traits import vanderbilt_traits, penn_traits

PAIRS = [('capacity', 'capacity', 'matched'),
         ('Glucose suppression (3 to 16.7 mM)', 'Glucose suppression (5.6 to 16.7 mM)', 'matched'),
         ('cAMP (IBMX) response', 'cAMP (IBMX) response', 'matched'),
         ('KCl response', 'KCl response', 'matched'),
         ('KCl response', 'Glucose suppression (5.6 to 16.7 mM)', 'mismatched'),
         ('cAMP (IBMX) response', 'Glucose suppression (5.6 to 16.7 mM)', 'mismatched'),
         ('Glucose suppression (3 to 16.7 mM)', 'KCl response', 'mismatched'),
         ('AA response', 'Glucose suppression (5.6 to 16.7 mM)', 'cross-stimulus'),
         ('AA response', 'Adrenaline response', 'cross-stimulus'),
         ('cAMP (IBMX) response', 'cAMP (IBMX) response, 16.7 mM glucose reference', 'matched, glucose-matched reference')]
# Sensitivity: the pre-specified Vanderbilt IBMX contrast uses the 5.6 mM wash as reference, so it also contains the glucose
# step. The Penn contrast is IBMX against the preceding 16.7 mM glucose window. This extra pair uses the Vanderbilt 16.7 mM
# glucose window as reference, which matches the Penn contrast in glucose.
EXTRA = {('Vanderbilt', 'cAMP (IBMX) response, 16.7 mM glucose reference'): {'IBMX': 1, 'G16.7': -1}}
CONV = [('Basal', 'Basal'), ('G16.7 II', 'G16.7 II'), ('IBMX SI', 'IBMX SI'), ('IBMX AUC', 'IBMX AUC'), ('KCl SI', 'KCl SI'), ('KCl AUC', 'KCl AUC')]


def vec(lab, name, windows):
    if name == 'capacity': return np.ones(len(windows)) / len(windows)
    if (lab, name) in EXTRA:
        a = np.zeros(len(windows))
        for k, w in EXTRA[(lab, name)].items(): a[windows.index(k)] = w
        return a
    return contrast_vector(lab, name)


def true_var(lab, hormone, name):
    M = json.load(open(ROOT / f'data/processed/model_{lab}_{hormone}.json')); S = np.array(M['Sigma'])
    a = vec(lab, name, M['windows']); return float(a @ S @ a)


def gls(E, lab, name):
    if (lab, name) not in EXTRA: return E[f'gls_{name}']
    return sum(w * E[f'gls_{k}'] for k, w in EXTRA[(lab, name)].items())


def boot_ci(x, y, B=2000, seed=1, tv=None):
    """Donor-bootstrap 95% CI of the Pearson correlation, or of cov(x, y) / tv (true-effect correlation, tv held fixed)."""
    rng = np.random.default_rng(seed); n = len(x); r = []
    for _ in range(B):
        i = rng.integers(0, n, n); r.append(np.corrcoef(x[i], y[i])[0, 1] if tv is None else np.cov(x[i], y[i])[0, 1] / tv)
    return np.nanquantile(r, [0.025, 0.975])


def conv_table(lab, hormone):
    L = pd.read_csv(ROOT / 'data/processed/perifusion_long.tsv.gz', sep='\t'); R = pd.read_csv(ROOT / 'data/processed/perifusion_runs.tsv', sep='\t')
    ok = set(R[(R.lab == lab) & (R.hormone == hormone) & R.usable].donor)
    x = L[(L.lab == lab) & (L.hormone == hormone) & L.donor.isin(ok)]
    Y = x.pivot_table(index='donor', columns='fraction', values='value').interpolate(axis=1, limit_direction='both')
    Y = Y.clip(lower=Y[Y > 0].stack().quantile(0.01))
    T = (penn_traits if lab == 'Penn' else vanderbilt_traits)(Y.values); T.index = Y.index
    return T.replace([np.inf, -np.inf], np.nan)


def main():
    rows = []
    for hormone in ['glucagon', 'insulin']:
        EP = pd.read_csv(ROOT / f'data/processed/donor_effects_Penn_{hormone}.tsv', sep='\t', index_col=0)
        EV = pd.read_csv(ROOT / f'data/processed/donor_effects_Vanderbilt_{hormone}.tsv', sep='\t', index_col=0)
        com = EP.index.intersection(EV.index)
        for a, b, kind in PAIRS:
            x = gls(EP, 'Penn', a).loc[com].values; y = gls(EV, 'Vanderbilt', b).loc[com].values
            r = np.corrcoef(x, y)[0, 1]; lo, hi = boot_ci(x, y)
            tv = np.sqrt(true_var('Penn', hormone, a) * true_var('Vanderbilt', hormone, b)); tlo, thi = boot_ci(x, y, tv=tv)
            rows.append({'hormone': hormone, 'measure': 'model', 'kind': kind, 'penn': a, 'vanderbilt': b, 'n': len(com), 'r_observed': r,
                         'ci_low': lo, 'ci_high': hi, 'p': stats.pearsonr(x, y)[1], 'spearman': stats.spearmanr(x, y)[0],
                         'r_true_effects': np.cov(x, y)[0, 1] / tv if tv > 0 else np.nan, 'r_true_ci_low': tlo, 'r_true_ci_high': thi})
        TP, TV = conv_table('Penn', hormone), conv_table('Vanderbilt', hormone); cc = TP.index.intersection(TV.index)
        for a, b in CONV:
            xy = pd.concat([np.log(TP.loc[cc, a].clip(lower=1e-6)), np.log(TV.loc[cc, b].clip(lower=1e-6))], axis=1).dropna()
            x, y = xy.iloc[:, 0].values, xy.iloc[:, 1].values; lo, hi = boot_ci(x, y)
            rows.append({'hormone': hormone, 'measure': 'conventional', 'kind': 'matched', 'penn': a, 'vanderbilt': b, 'n': len(xy),
                         'r_observed': np.corrcoef(x, y)[0, 1], 'ci_low': lo, 'ci_high': hi, 'p': stats.pearsonr(x, y)[1],
                         'spearman': stats.spearmanr(x, y)[0], 'r_true_effects': np.nan})
    T = pd.DataFrame(rows); T.to_csv(ROOT / 'results/tables/crosslab_replication.tsv', sep='\t', index=False)
    pd.set_option('display.width', 250); print(T.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
