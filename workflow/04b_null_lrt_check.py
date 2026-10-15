"""Step 04b - calibration check of the likelihood-ratio test for stimulus-specific variance under the capacity-only null.

Step 04 found LR(capacity_diag vs capacity) above the real-data 95% threshold in more scenario A replicates than expected.
Step 04 back-transforms simulated curves to the linear scale (exp, minus the assay offset, clipped at 1e-6) before the
conventional traits and the model refit. This step separates the two possible causes.
  gaussian         simulated log-scale curves are refitted directly (the model is correctly specified)
  back-transformed curves go through the same linear-scale round trip as in step 04 (small floor effects)
Scenario A (capacity only, fitted to HPAP Vanderbilt glucagon), N = the real sample size and N = 400, R replicates.
Outputs: results/tables/null_lrt_calibration.tsv (per replicate), null_lrt_calibration_summary.tsv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from glucv2.model import WindowModel
from glucv2.data import load_matrix

OUT = ROOT / 'results/tables'


def run(N, R=40, seed=7):
    D = load_matrix('Vanderbilt', 'glucagon'); m = WindowModel(D['R'], D['frac'], D['win']); f0 = m.fit('capacity')
    lrt = pd.read_csv(OUT / 'model_lrt_Vanderbilt.tsv', sep='\t').query("hormone == 'glucagon'").set_index('test')
    thr_d, thr_f = lrt.loc['capacity_diag vs capacity', 'null_LR_95pct'], lrt.loc['full vs capacity_diag', 'null_LR_95pct']
    msim = WindowModel(np.zeros((N, m.T)), D['frac'], D['win']); rng = np.random.default_rng(seed + N); rows = []
    for r in range(R):
        Rs, _ = msim.simulate(f0, N=N, rng=rng.integers(1 << 31))
        Ylin = np.maximum(np.exp(D['mu'] + Rs) - D['offset'], 1e-6); LY = np.log(Ylin + D['offset'])
        for ver, X in [('gaussian', Rs), ('back-transformed', LY)]:
            mm = WindowModel(X - X.mean(0), D['frac'], D['win']); ff = {k: mm.fit(k) for k in ['capacity', 'capacity_diag', 'full']}
            rows.append({'N': N, 'rep': r, 'version': ver, 'LR_diag_vs_capacity': 2 * (ff['capacity_diag']['loglik'] - ff['capacity']['loglik']),
                         'LR_full_vs_diag': 2 * (ff['full']['loglik'] - ff['capacity_diag']['loglik']),
                         'threshold_diag': thr_d, 'threshold_full': thr_f})
        print(N, r, flush=True)
    return pd.DataFrame(rows)


def main(R=40):
    N0 = len(load_matrix('Vanderbilt', 'glucagon')['donors'])
    T = pd.concat([run(N0, R), run(400, R)])
    T.to_csv(OUT / 'null_lrt_calibration.tsv', sep='\t', index=False)
    S = T.groupby(['N', 'version']).apply(lambda g: pd.Series({
        'replicates': len(g), 'LR_diag_median': g.LR_diag_vs_capacity.median(), 'LR_diag_max': g.LR_diag_vs_capacity.max(),
        'exceed_diag_threshold': (g.LR_diag_vs_capacity > g.threshold_diag).mean(),
        'LR_full_median': g.LR_full_vs_diag.median(), 'exceed_full_threshold': (g.LR_full_vs_diag > g.threshold_full).mean()}), include_groups=False).reset_index()
    S.to_csv(OUT / 'null_lrt_calibration_summary.tsv', sep='\t', index=False); print(S.round(3).to_string(index=False))


if __name__ == '__main__':
    main(*[int(x) for x in sys.argv[1:]])
