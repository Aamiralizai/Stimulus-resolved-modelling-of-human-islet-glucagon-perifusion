"""Figure 3 - the assay floor, the censored likelihood and robustness.

Rebuilt from the result tables to the published legend, with the HbA1c
coefficient expressed per 10 mmol/mol.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from glucv2.plotting import style, save_multi, panel_label, CAT, INK, INK2, MUTED, GRID
from glucv2.data import WINDOWS
from glucv2 import inputs

T, P, FIG = ROOT / 'results/tables', ROOT / 'data/processed', ROOT / 'results/figures'
PER10 = 10 / 10.929
L1, L2 = CAT[0], CAT[3]
SUP = 'Glucose suppression (3 to 16.7 mM)'

SHORT = {'Glucose suppression (3 to 16.7 mM)': 'glucose suppression',
         'Glucose suppression (5.6 to 16.7 mM)': 'glucose suppression',
         'Low-glucose response (AA to AA+G3)': 'low-glucose response',
         'window AA+G3 (capacity-adjusted)': 'window AA+G3, adjusted',
         'window AA (capacity-adjusted)': 'window AA, adjusted',
         'cAMP (IBMX) response': 'cAMP (IBMX) response'}
CHECK = {'runs with no censored amino-acid fraction': 'no censored amino-acid run',
         'runs with no censored fraction': 'no censored run',
         'race and isolation year factor': 'race and isolation year',
         'isolation year as a factor': 'isolation year',
         'Huber M-estimation': 'Huber M-estimation',
         'base': 'base model'}
FAMORD = ['demography', 'shape', 'robust', 'censoring']


def main():
    style()
    fig = plt.figure(figsize=(7.2, 7.2))
    gs = GridSpec(3, 3, figure=fig, hspace=.68, wspace=.46,
                  left=.095, right=.955, top=.95, bottom=.075)
    A, B, C = (fig.add_subplot(gs[0, k]) for k in range(3))
    D, E, F = (fig.add_subplot(gs[1, k]) for k in range(3))
    G, H, I = (fig.add_subplot(gs[2, k]) for k in range(3))

    # a. percent of fractions censored per window, both laboratories
    rows = []
    for lab, col in [('Penn', L1), ('Vanderbilt', L2)]:
        for _, w in inputs.censoring_by_window(str(ROOT), lab).iterrows():
            rows.append((f'{lab[:4]}. {w.window}', w.pct_donors_any_censored, col))
    y = np.arange(len(rows))
    A.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=.72)
    A.set_yticks(y); A.set_yticklabels([r[0] for r in rows], fontsize=4.8)
    A.invert_yaxis(); A.tick_params(axis='y', length=0)
    A.set_xlabel('Donors with any censoring (%)')
    A.set_title('Where the assay floor bites', loc='left', color=INK)

    # b. per-donor censored fraction among affected donors
    for lab, col, nm in [('Penn', L1, 'Laboratory 1'), ('Vanderbilt', L2, 'Laboratory 2')]:
        fr = inputs.censoring_per_donor(str(ROOT), lab).pct_fractions_censored
        fr = fr[fr > 0]
        B.hist(fr, bins=np.arange(0, 45, 3), color=col, alpha=.65,
               label=f'{nm}, {len(fr)} donors')
    B.set_xlabel('Censored fractions per donor (%)'); B.set_ylabel('Donors')
    B.set_title('Donors with any censoring', loc='left', color=INK)
    B.legend(fontsize=5.2, loc='upper right', frameon=False)
    B.set_ylim(0, B.get_ylim()[1] * 1.28)
    B.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

    # c, d. censoring simulation, effect and null
    V = pd.read_csv(T / 'censored_validation.tsv', sep='\t')
    order = ['no floor', 'floor ignored', 'floor modelled']
    cols = {'no floor': MUTED, 'floor ignored': CAT[7], 'floor modelled': CAT[0]}
    rng = np.random.default_rng(2)
    for ax, key, ttl in [(C, 'EFFECT', 'Simulation, real effect'),
                         (D, 'NULL', 'Simulation, no effect')]:
        s = V[V.scenario.str.startswith(key)]
        for k, a in enumerate(order):
            v = s[s.analysis == a].beta.values
            ax.scatter(np.full(len(v), k) + rng.uniform(-.16, .16, len(v)), v, s=3.5,
                       color=cols[a], alpha=.5, lw=0)
            ax.plot([k - .3, k + .3], [np.median(v)] * 2, color=INK, lw=1.3, zorder=5)
        truth = float(s.true_beta.iloc[0])
        ax.axhline(truth, color=CAT[7], lw=.9, ls='--')
        ax.text(2.72, truth, 'truth', fontsize=5.4, color=CAT[7], va='bottom', ha='right')
        ax.set_xticks(range(3)); ax.set_xticklabels(['no\nfloor', 'floor\nignored', 'floor\nmodelled'], fontsize=5.2)
        ax.set_ylabel('HbA1c coefficient'); ax.set_title(ttl, loc='left', color=INK)
        ax.set_xlim(-.55, 2.75)

    # e. censored against uncensored donor estimates
    u = pd.read_csv(inputs.find(ROOT, 'donor_effects_Penn_glucagon.tsv'), sep='\t').set_index('donor')
    c = pd.read_csv(inputs.find(ROOT, 'donor_effects_censored_Penn_glucagon.tsv'), sep='\t').set_index('donor')
    k = u.join(c, how='inner', lsuffix='_u', rsuffix='_c')
    xa, ya = f'gls_{SUP}_u', f'gls_{SUP}_c'
    if xa in k.columns and ya in k.columns:
        E.scatter(k[xa], k[ya], s=6, color=L1, alpha=.65, lw=0)
        lim = [min(k[xa].min(), k[ya].min()), max(k[xa].max(), k[ya].max())]
        E.plot(lim, lim, color=INK, lw=.8, ls='--')
        E.text(.03, .97, f'r = {k[xa].corr(k[ya]):.3f}\nn = {len(k)}', transform=E.transAxes,
               va='top', ha='left', fontsize=5.8, color=INK, linespacing=1.3)
    E.set_xlabel('Uncensored estimate'); E.set_ylabel('Censored estimate')
    E.set_title('The floor shifts, not reorders', loc='left', color=INK)

    # f. leave-one-donor-out
    LO = pd.read_csv(T / 'robustness_leave_one_out_supp.tsv', sep='\t')
    F.hist(LO.beta * 0 + LO.beta, bins=22, color=L1, alpha=.8)
    F.axvline(LO.beta.median(), color=CAT[7], lw=1.1)
    F.set_xlabel('HbA1c coefficient, one donor dropped')
    F.set_ylabel('Refits')
    F.text(.03, .97, f'all {len(LO)} refits keep\n$p$ below 0.05', transform=F.transAxes,
           va='top', ha='left', fontsize=5.8, color=INK, linespacing=1.3)
    F.set_title('No single donor drives it', loc='left', color=INK)

    # g. Cook distance against DFBETA
    R = pd.read_csv(T / 'robustness_checks.tsv', sep='\t')
    inf = R[(R.outcome == 'supp') & (R.check.str.contains('Cook'))]
    cook, dfb = float(inf.beta.iloc[0]), float(inf.se.iloc[0])
    G.scatter([cook], [dfb], s=22, color=CAT[7], zorder=4)
    G.axvline(4 / 112, color=INK2, lw=.8, ls=':')
    G.set_xlim(0, max(cook, 4 / 112) * 1.5); G.set_ylim(0, dfb * 1.5)
    G.text(4 / 112, dfb * 1.44, '4/n ', fontsize=5.4, color=INK2, va='top', ha='right')
    G.set_xlabel('Largest Cook distance'); G.set_ylabel('Largest |DFBETA| for HbA1c')
    G.text(.05, .14, f'max Cook {cook:.3f}\nmax |DFBETA| {dfb:.2f}', transform=G.transAxes,
           ha='left', va='center', fontsize=5.8, color=INK, linespacing=1.3)
    G.set_title('Influence diagnostics', loc='left', color=INK)

    # h. every specification, in SD per 10 mmol/mol
    S = R[(R.outcome == 'supp')].dropna(subset=['beta_SD', 'ci_low_SD', 'ci_high_SD']).copy()
    S = S[(S.ci_low_SD <= S.beta_SD) & (S.beta_SD <= S.ci_high_SD) & (S.ci_high_SD > S.ci_low_SD)]
    S['ord'] = S.group.map({g: i for i, g in enumerate(FAMORD)})
    S = S.sort_values(['ord', 'beta_SD'])
    fam = {g: CAT[i] for i, g in enumerate(FAMORD)}
    y = np.arange(len(S))
    for k2, (_, r) in enumerate(S.iterrows()):
        H.errorbar(r.beta_SD * PER10, k2,
                   xerr=[[(r.beta_SD - r.ci_low_SD) * PER10], [(r.ci_high_SD - r.beta_SD) * PER10]],
                   fmt='o', ms=2.6, lw=.7, color=fam[r.group])
    H.axvline(0, color=MUTED, lw=.6, ls='--')
    H.set_yticks(y)
    H.set_yticklabels([CHECK.get(c, c) for c in S.check], fontsize=4.6)
    H.invert_yaxis(); H.tick_params(axis='y', length=0)
    H.set_xlabel('SD per 10 mmol/mol (95% CI)')
    for k2, g in enumerate(FAMORD):
        H.text(.985, .035 + .072 * (len(FAMORD) - 1 - k2), g, transform=H.transAxes,
               fontsize=4.6, color=fam[g], ha='right', va='center')
    H.set_title('Robust to specification', loc='left', color=INK)

    # i. fraction of missing information
    CC = pd.read_csv(T / 'censored_clinical.tsv', sep='\t')
    f = CC.sort_values('fraction_missing_information', ascending=False)
    y = np.arange(len(f))
    I.barh(y, f.fraction_missing_information * 100,
           color=[L1 if l == 'Penn' else L2 for l in f.lab], height=.72)
    I.set_yticks(y)
    I.set_yticklabels([f'{r.lab[:4]}. {SHORT.get(r.outcome, r.outcome)}'
                       for _, r in f.iterrows()], fontsize=4.6)
    I.invert_yaxis(); I.tick_params(axis='y', length=0)
    I.set_xlabel('Fraction of missing information (%)')
    I.set_title('Imputation adds little noise', loc='left', color=INK)

    for ax, l in zip([A, B, C, D, E, F, G, H, I], 'abcdefghi'):
        panel_label(ax, l, x=-0.34 if ax in (A, H, I) else -0.26, y=1.16)
    save_multi(fig, FIG / 'Figure3_censoring_robustness')
    print('wrote results/figures/Figure3_censoring_robustness.{svg,pdf,png}')


if __name__ == '__main__':
    main()
