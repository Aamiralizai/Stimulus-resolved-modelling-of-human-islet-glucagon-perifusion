"""Step 03 - fit the stimulus-resolved window model to real perifusion data.

For each lab (Penn, Vanderbilt) and hormone (glucagon primary, insulin secondary):
  * maximum-likelihood fits of three covariance models for the donor window effects: capacity only (tau2 * 11'),
    capacity + independent window effects (tau2 * 11' + diag), unstructured.
  * likelihood-ratio tests with parametric-bootstrap p-values (data simulated from the fitted null model).
  * decomposition of between-donor variance into capacity and stimulus-specific parts, and the true-effect variance,
    sampling variance and reliability of each stimulus contrast.
  * donor bootstrap (resampling donors) for confidence intervals.
  * donor-level GLS (unshrunk) and BLUP estimates of window effects, capacity and contrasts.
Outputs: results/tables/model_*.tsv, data/processed/donor_effects_<lab>_<hormone>.tsv
Usage: python workflow/03_fit_models.py <lab> [B_lrt] [B_boot]   (lab = Penn or Vanderbilt; run both, then 03b merges)
"""
import sys, json, zlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from glucv2.model import WindowModel, decompose
from glucv2.data import load_matrix, CONTRASTS, contrast_vector

OUT = ROOT / 'results/tables'


def lrt(m, f_alt, f_null, kind_alt, kind_null, B, seed):
    obs = 2 * (f_alt['loglik'] - f_null['loglik']); rng = np.random.default_rng(seed); null = []
    for b in range(B):
        R, _ = m.simulate(f_null, rng=rng.integers(1 << 31)); mb = WindowModel(R - R.mean(0), m.frac, m.win)
        fa, fn = mb.fit(kind_alt), mb.fit(kind_null)
        null.append(2 * (fa['loglik'] - fn['loglik']))
    null = np.array(null)
    return obs, (1 + (null >= obs).sum()) / (B + 1), null


def summarise(m, f, lab, windows):
    dec = decompose(f['Sigma']); _, _, Ainv = m.donor_estimates(f)
    rows = [{'quantity': 'capacity variance', 'value': dec['capacity_var']},
            {'quantity': 'stimulus-specific share of between-donor variance', 'value': dec['specific_share']},
            {'quantity': 'within-run noise variance s2 (mean over windows)', 'value': float(np.mean(f['s2']))}, {'quantity': 'AR(1) rho per fraction', 'value': f['rho']},
            {'quantity': 'nugget share eta', 'value': f['eta']}]
    for w, v in zip(windows, dec['specific_var_per_window']): rows.append({'quantity': f'stimulus-specific variance, {w}', 'value': v})
    for name in CONTRASTS[lab]:
        a = contrast_vector(lab, name); tv = float(a @ f['Sigma'] @ a); sv = float(a @ Ainv @ a)
        rows += [{'quantity': f'{name}: true SD', 'value': np.sqrt(tv)}, {'quantity': f'{name}: reliability', 'value': tv / (tv + sv)}]
    # correlations among true contrasts and with capacity
    names = list(CONTRASTS[lab]); Amat = np.array([contrast_vector(lab, n) for n in names] + [np.ones(len(windows)) / len(windows)])
    C = Amat @ f['Sigma'] @ Amat.T; d = np.sqrt(np.diag(C)); Rho = C / np.outer(d, d)
    labs = names + ['capacity']
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)): rows.append({'quantity': f'corr: {labs[i]} | {labs[j]}', 'value': Rho[i, j]})
    return rows


def main(lab, B_lrt=100, B_boot=200):
    allrows, tests = [], []
    for lab in [lab]:
        for hormone in ['glucagon', 'insulin']:
            D = load_matrix(lab, hormone); m = WindowModel(D['R'], D['frac'], D['win']); W = m.W
            fits = {k: m.fit(k) for k in ['capacity', 'capacity_diag', 'full']}
            print(f"[{lab} {hormone}] N={m.N} T={m.T} W={W} loglik:", {k: round(v['loglik'], 1) for k, v in fits.items()},
                  'converged', {k: v['converged'] for k, v in fits.items()}, flush=True)
            Bl = B_lrt if hormone == 'glucagon' else max(B_lrt // 2, 50)
            for alt, nul in [('capacity_diag', 'capacity'), ('full', 'capacity_diag')]:
                obs, p, null = lrt(m, fits[alt], fits[nul], alt, nul, Bl, seed=zlib.crc32(f'{lab}|{hormone}|{alt}'.encode()))   # deterministic (Python hash() is salted per process)
                tests.append({'lab': lab, 'hormone': hormone, 'N': m.N, 'test': f'{alt} vs {nul}', 'LR': obs,
                              'df': fits[alt]['n_par'] - fits[nul]['n_par'], 'p_parametric_bootstrap': p, 'B': Bl,
                              'null_LR_95pct': float(np.quantile(null, 0.95))})
                print('  ', tests[-1], flush=True)
            f = fits['full']; rows = summarise(m, f, lab, D['windows'])
            # donor bootstrap
            rng = np.random.default_rng(7); boot = {r['quantity']: [] for r in rows}
            for b in range(B_boot if hormone == 'glucagon' else B_boot // 2):
                idx = rng.integers(0, m.N, m.N); R = D['Y'][idx]; R = R - R.mean(0)
                mb = WindowModel(R, D['frac'], D['win']); fb = mb.fit('full', th0=f['th'])
                for r in summarise(mb, fb, lab, D['windows']): boot[r['quantity']].append(r['value'])
            for r in rows:
                v = np.array(boot[r['quantity']]); r.update({'lab': lab, 'hormone': hormone, 'N': m.N,
                                                            'ci_low': np.nanquantile(v, 0.025), 'ci_high': np.nanquantile(v, 0.975)})
            allrows += rows
            gls, blup, Ainv = m.donor_estimates(f)
            E = pd.DataFrame(gls, columns=[f'gls_{w}' for w in D['windows']], index=D['donors'])
            for k, w in enumerate(D['windows']): E[f'blup_{w}'] = blup[:, k]
            E['gls_capacity'] = gls.mean(1); E['blup_capacity'] = blup.mean(1)
            for name in CONTRASTS[lab]:
                a = contrast_vector(lab, name); E[f'gls_{name}'] = gls @ a; E[f'blup_{name}'] = blup @ a; E[f'se_{name}'] = np.sqrt(a @ Ainv @ a)
            E['n_censored'] = D['censored'].sum(1)
            E.index.name = 'donor'; E.to_csv(ROOT / f'data/processed/donor_effects_{lab}_{hormone}.tsv', sep='\t')
            json.dump({'Sigma': f['Sigma'].tolist(), 's2': np.atleast_1d(f['s2']).tolist(), 'rho': f['rho'], 'eta': f['eta'], 'windows': D['windows'],
                       'frac': D['frac'].tolist(), 'win': D['win'].tolist(), 'mu': D['mu'].tolist(), 'offset': D['offset'],
                       'fits': {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()} for k, v in fits.items()}},
                      open(ROOT / f'data/processed/model_{lab}_{hormone}.json', 'w'))
    pd.DataFrame(allrows)[['lab', 'hormone', 'N', 'quantity', 'value', 'ci_low', 'ci_high']].to_csv(OUT / f'model_variance_decomposition_{lab}.tsv', sep='\t', index=False)
    pd.DataFrame(tests).to_csv(OUT / f'model_lrt_{lab}.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main(sys.argv[1], *[int(x) for x in sys.argv[2:]])
