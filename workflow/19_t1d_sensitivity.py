"""Step 19 - type 1 diabetes exclusion sensitivity for the primary association.

The primary model is fitted to every usable run, including runs from donors with
type 1 diabetes, because the larger sample sharpens the shared covariance and the
population mean curve.  Those donors are then dropped from the clinical
associations.  A reviewer may ask whether including them influences the estimated
model parameters, and so the association.

This step refits the left-censored model on donors without diabetes and with type
2 diabetes only, then repeats the HbA1c association.  The comparison of interest
is the coefficient on glucose suppression, in SD units.

Usage: python workflow/19_t1d_sensitivity.py [n_imp]
Output: results/tables/t1d_exclusion_sensitivity.tsv
"""
import sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from glucv2 import inputs
from glucv2.data import load_matrix, contrast_vector, CONTRASTS
from glucv2.model import decompose
from glucv2.censored import CensoredFit

spec = importlib.util.spec_from_file_location('cens9', ROOT / 'workflow/09_censored_model.py')
cens9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(cens9)

OUT = ROOT / 'results/tables'
GS = {'Penn': 'Glucose suppression (3 to 16.7 mM)',
      'Vanderbilt': 'Glucose suppression (5.6 to 16.7 mM)'}


def main(n_imp=20):
    D0 = inputs.donor_table(str(ROOT)).reset_index().set_index('donor')
    keep = list(D0.index[D0.group.isin(['control', 'T2D'])])
    rows = []
    for lab in ['Penn', 'Vanderbilt']:
        for arm, donors in [('all usable runs', None), ('type 1 diabetes excluded', keep)]:
            D = load_matrix(lab, 'glucagon', donors=donors)
            cf = CensoredFit(D['Y'], D['censored'], D['threshold'], D['frac'], D['win'], seed=11)
            f = cf.fit('full', iters=25, verbose=False)
            dec = decompose(f['Sigma'])
            idx = pd.Index(D['donors'], name='donor')
            cl = cens9.clinical(cf, D, lab, idx, D0, n_imp)
            gs = [r for r in cl if r['outcome'] == GS[lab]][0]
            cap = [r for r in cl if r['outcome'] == 'capacity'][0]
            v = contrast_vector(lab, GS[lab])
            rows.append({
                'lab': lab, 'arm': arm, 'N_model_fit': len(idx),
                'n_association': gs['n'],
                'loglik': f['loglik'], 'specific_share': dec['specific_share'],
                'suppression_true_SD': float(np.sqrt(max(v @ f['Sigma'] @ v, 0))),
                'suppression_beta_SD': gs['beta_SD'], 'suppression_ci_low_SD': gs['ci_low_SD'],
                'suppression_ci_high_SD': gs['ci_high_SD'], 'suppression_p': gs['p'],
                'capacity_beta_SD': cap['beta_SD'], 'capacity_p': cap['p'],
            })
            print(f"[{lab}] {arm:26} N={len(idx):4d} n={gs['n']:4d} "
                  f"suppression {gs['beta_SD']:+.3f} SD  p={gs['p']:.3g}", flush=True)
    T = pd.DataFrame(rows)
    for lab in T.lab.unique():
        a = T[(T.lab == lab) & (T.arm == 'all usable runs')].iloc[0]
        b = T[(T.lab == lab) & (T.arm == 'type 1 diabetes excluded')].iloc[0]
        print(f'{lab}: suppression {a.suppression_beta_SD:+.3f} -> {b.suppression_beta_SD:+.3f} SD, '
              f'change {100*(b.suppression_beta_SD-a.suppression_beta_SD)/abs(a.suppression_beta_SD):+.1f}%')
    T.to_csv(OUT / 't1d_exclusion_sensitivity.tsv', sep='\t', index=False)
    print('written', OUT / 't1d_exclusion_sensitivity.tsv')


if __name__ == '__main__':
    main(*[int(x) for x in sys.argv[1:2]])
