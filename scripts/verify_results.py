#!/usr/bin/env python3
"""Check that the shipped result tables still carry every number the paper reports.

Runs in a few seconds and needs nothing beyond what the repository ships.  Each
check names the table it reads, the quantity, the value in the manuscript and
the value found.  Exits non-zero if any check fails.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / 'results/tables'
PER10 = 10 / 10.929          # per 1% HbA1c to per 10 mmol/mol
CHECKS, FAILS = [], 0


def check(label, found, expected, tol, unit=''):
    global FAILS
    ok = abs(found - expected) <= tol
    FAILS += 0 if ok else 1
    CHECKS.append((ok, label, expected, found, unit))
    return ok


def main():
    # 1. primary association, laboratory 1
    cc = pd.read_csv(T / 'censored_clinical.tsv', sep='\t')
    p = cc[(cc.lab == 'Penn') & cc.outcome.str.startswith('Glucose suppression')].iloc[0]
    check('suppression, laboratory 1, SD per 10 mmol/mol', p.beta_SD * PER10, 0.26, 0.005)
    check('  lower 95% CI', p.ci_low_SD * PER10, 0.17, 0.005)
    check('  upper 95% CI', p.ci_high_SD * PER10, 0.34, 0.005)
    check('  p value', p.p, 2.9e-9, 1e-10)
    check('  n donors', p.n, 112, 0)

    cap = cc[(cc.lab == 'Penn') & (cc.outcome == 'capacity')].iloc[0]
    check('capacity, laboratory 1, SD per 10 mmol/mol', cap.beta_SD * PER10, -0.20, 0.005)
    check('  p value', cap.p, 5.1e-5, 1e-6)

    v = cc[(cc.lab == 'Vanderbilt') & cc.outcome.str.startswith('Glucose suppression')].iloc[0]
    check('suppression, laboratory 2, n donors', v.n, 83, 0)

    # 2. cross-laboratory agreement
    x = pd.read_csv(T / 'crosslab_replication.tsv', sep='\t')
    m = x[(x.hormone == 'glucagon') & (x.measure == 'model') & (x.kind == 'matched')
          & x.penn.str.startswith('Glucose')].iloc[0]
    check('cross-laboratory suppression, r', m.r_observed, 0.33, 0.005)
    check('  p value', m.p, 0.0045, 0.0002)
    check('  n donors', m.n, 74, 0)

    # 3. joint latent model
    j = pd.read_csv(T / 'joint_latent_model.tsv', sep='\t').set_index('parameter')
    check('joint latent model, SD per 10 mmol/mol',
          float(j.loc['hba1c, in SD of the latent phenotype', 'estimate']) * PER10, 0.49, 0.005)
    check('  p value', float(j.loc['hba1c', 'p']), 3.2e-5, 1e-6)
    check('  n donors', float(j.loc['hba1c', 'n_donors']), 127, 0)
    check('laboratory 2 loading', float(j.loc['lambda_Vanderbilt', 'estimate']), 0.54, 0.005)
    check('  p value', float(j.loc['lambda_Vanderbilt', 'p']), 0.013, 0.0005)

    # 4. construction null
    n = pd.read_csv(T / 'null_summary.tsv', sep='\t')
    col = next((c for c in n.columns if 'congruence' in c and 'axis1' in c), None)
    if col:
        a = n[n.iloc[:, 0].astype(str).str.startswith('A')]
        if len(a):
            check('construction null, axis 1 Tucker congruence', float(a[col].iloc[0]), 0.91, 0.01)

    # 5. covariance sensitivity, laboratory 1 glucagon
    cv = pd.read_csv(T / 'covariance_sensitivity.tsv', sep='\t')
    g = cv[(cv.lab == 'Penn') & (cv.hormone == 'glucagon')]
    check('stimulus-specific share, laboratory 1, minimum %', g.specific_share.min() * 100, 9.2, 0.15)
    check('stimulus-specific share, laboratory 1, maximum %', g.specific_share.max() * 100, 10.3, 0.15)

    # 6. model fit
    full = g[g.structure == 'full'].iloc[0]
    check('laboratory 1 glucagon, full model log likelihood', full.loglik, -189.33, 0.02)
    check('  free parameters', full.n_par, 37, 0)
    check('  runs in the model', full.N, 130, 0)

    # 7. robustness
    rb = pd.read_csv(T / 'robustness_checks.tsv', sep='\t')
    hub = rb[(rb.outcome == 'supp') & (rb.check == 'Huber M-estimation')].iloc[0]
    med = rb[(rb.outcome == 'supp') & (rb.check == 'median regression')].iloc[0]
    check('Huber M-estimation, beta', hub.beta, 0.0701, 0.0005)
    check('median regression, beta', med.beta, 0.0680, 0.0005)
    lo = pd.read_csv(T / 'robustness_leave_one_out_supp.tsv', sep='\t')
    check('leave-one-donor-out refits', len(lo), 112, 0)
    check('  refits keeping p below 0.05', (lo.p < 0.05).sum(), 112, 0)

    # 8. type 1 diabetes exclusion
    t1 = pd.read_csv(T / 't1d_exclusion_sensitivity.tsv', sep='\t').set_index(['lab', 'arm'])
    a = float(t1.loc[('Penn', 'all usable runs'), 'suppression_beta_SD'])
    b = float(t1.loc[('Penn', 'type 1 diabetes excluded'), 'suppression_beta_SD'])
    check('type 1 exclusion, laboratory 1, SD per 1% before', a, 0.281, 0.003)
    check('  after', b, 0.272, 0.003)
    check('  change, %', 100 * (b - a) / a, -3.4, 0.5)

    # 9. respiratory attenuation
    at = pd.read_csv(T / 'oxygen_covariate_attenuation.tsv', sep='\t')
    pc = next((c for c in at.columns if 'pct' in c.lower() or 'change' in c.lower()), None)
    if pc:
        resp = at[~at.iloc[:, 0].astype(str).str.contains('insulin', case=False)]
        check('largest respiratory attenuation, %', resp[pc].abs().max(), 2.7, 0.3)

    w = max(len(c[1]) for c in CHECKS)
    print(f'{"check".ljust(w)}   paper      found      ')
    print('-' * (w + 26))
    for ok, label, exp, got, unit in CHECKS:
        mark = 'ok  ' if ok else 'FAIL'
        print(f'{label.ljust(w)}   {exp:<10.6g} {got:<10.6g} {mark}')
    print('-' * (w + 26))
    print(f'{len(CHECKS)} checks, {FAILS} failed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
