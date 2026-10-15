"""Figure 4 - cross-laboratory agreement, the joint latent model and
coordination between the two hormones.

Rebuilt from the result tables to the published legend. The HbA1c effect is
expressed per 10 mmol/mol, as Diabetologia requires for figures.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from glucv2 import inputs
from glucv2.plotting import style, save_multi, panel_label, CAT, INK, INK2, MUTED, GRID

T, P, FIG = ROOT / 'results/tables', ROOT / 'data/processed', ROOT / 'results/figures'
PER10 = 10 / 10.929
L1, L2 = CAT[0], CAT[3]
SP = 'Penn | glucagon | Glucose suppression (3 to 16.7 mM)'
SV = 'Vanderbilt | glucagon | Glucose suppression (5.6 to 16.7 mM)'

SHORTEN = [('Glucose suppression', 'suppression'),
           ('cAMP (IBMX) response, 16.7', 'cAMP at 16.7 mM'),
           ('cAMP (IBMX) response', 'cAMP'),
           ('Adrenaline response', 'adrenaline'),
           ('AA response', 'amino acids'),
           ('KCl response', 'KCl'),
           ('capacity', 'capacity')]


def short(s):
    for a, b in SHORTEN:
        if s.startswith(a):
            return b
    return s


def main():
    style()
    fig = plt.figure(figsize=(7.2, 7.2))
    gs = GridSpec(3, 3, figure=fig, hspace=.72, wspace=.52,
                  left=.10, right=.955, top=.95, bottom=.075)
    A, B, C = (fig.add_subplot(gs[0, k]) for k in range(3))
    D, E, F = (fig.add_subplot(gs[1, k]) for k in range(3))
    G, H, I = (fig.add_subplot(gs[2, k]) for k in range(3))

    X = pd.read_csv(T / 'crosslab_replication.tsv', sep='\t')
    XG = X[(X.hormone == 'glucagon') & (X.measure == 'model')]

    # a. glucose suppression at both laboratories, same donors
    ep = pd.read_csv(inputs.find(ROOT, 'donor_effects_Penn_glucagon.tsv'), sep='\t', index_col=0)
    ev = pd.read_csv(inputs.find(ROOT, 'donor_effects_Vanderbilt_glucagon.tsv'), sep='\t', index_col=0)
    cp0 = [c for c in ep.columns if c.startswith('gls_Glucose suppression')][0]
    cv0 = [c for c in ev.columns if c.startswith('gls_Glucose suppression')][0]
    m = ep[[cp0]].join(ev[[cv0]], how='inner').dropna()
    SP, SV = cp0, cv0
    A.scatter(m[SP], m[SV], s=7, color=L1, alpha=.7, lw=0)
    b, a0 = np.polyfit(m[SP], m[SV], 1)
    xs = np.linspace(m[SP].min(), m[SP].max(), 2)
    A.plot(xs, a0 + b * xs, color=INK, lw=.9)
    r = XG[(XG.kind == 'matched') & XG.penn.str.startswith('Glucose')].iloc[0]
    A.text(.03, .97, f'r = {r.r_observed:.2f} (95% CI {r.ci_low:.2f}, {r.ci_high:.2f})\n'
                     f'$p$ = {r.p:.4f},  $n$ = {int(r.n)}', transform=A.transAxes,
           va='top', ha='left', fontsize=5.4, color=INK, linespacing=1.35,
           bbox=dict(fc='white', ec='none', alpha=.78, pad=1.4))
    A.set_xlabel('Laboratory 1, suppression'); A.set_ylabel('Laboratory 2, suppression')
    A.set_title('Same donors, two laboratories', loc='left', color=INK)

    # b. matched against negative-control pairings
    kc = {'matched': CAT[0], 'matched, glucose-matched reference': CAT[0],
          'mismatched': CAT[1], 'cross-stimulus': CAT[2]}
    S = XG.copy()
    S['k'] = S.kind.map(lambda s: 0 if s.startswith('matched') else (1 if s == 'mismatched' else 2))
    S = S.sort_values(['k', 'r_observed'])
    y = np.arange(len(S))
    for k2, (_, w) in enumerate(S.iterrows()):
        B.errorbar(w.r_observed, k2,
                   xerr=[[w.r_observed - w.ci_low], [w.ci_high - w.r_observed]],
                   fmt='o', ms=2.6, lw=.7, color=kc[w.kind])
    B.axvline(0, color=MUTED, lw=.6, ls='--')
    B.set_yticks(y)
    B.set_yticklabels([f'{short(w.penn)} / {short(w.vanderbilt)}' for _, w in S.iterrows()],
                      fontsize=4.4)
    B.invert_yaxis(); B.tick_params(axis='y', length=0)
    for k2, g in enumerate(['matched', 'mismatched', 'cross-stimulus']):
        B.text(.985, .035 + .075 * (2 - k2), g, transform=B.transAxes,
               fontsize=4.6, color=kc[g], ha='right', va='center')
    B.set_xlabel('Correlation between laboratories')
    B.set_title('Only matched stimuli agree', loc='left', color=INK)

    # c. bootstrap, observed and disattenuated
    BT = pd.read_csv(T / 'crosslab_bootstrap.tsv', sep='\t')
    bins = np.linspace(-.6, 2.0, 46)
    C.hist(BT.r_observed, bins=bins, color=L1, alpha=.75, label='observed')
    C.hist(BT.r_true_effects, bins=bins, color=CAT[7], alpha=.55, label='disattenuated')
    C.axvline(1, color=INK2, lw=.8, ls=':')
    C.text(1, C.get_ylim()[1] * .62, ' r = 1', fontsize=5.2, color=INK2, va='top', ha='left')
    C.set_xlabel('Correlation'); C.set_ylabel('Bootstrap replicates')
    C.legend(fontsize=5.2, frameon=False, loc='upper right')
    C.set_title('Disattenuation is uninformative', loc='left', color=INK)

    # d. loadings on the shared latent phenotype
    J = pd.read_csv(T / 'joint_latent_model.tsv', sep='\t').set_index('parameter')
    lv, sv = float(J.loc['lambda_Vanderbilt', 'estimate']), float(J.loc['lambda_Vanderbilt', 'se'])
    D.errorbar([1.0], [1], xerr=None, fmt='o', ms=3.4, color=L1)
    D.errorbar([lv], [0], xerr=[[1.96 * sv], [1.96 * sv]], fmt='o', ms=3.4, lw=.8, color=L2)
    D.axvline(0, color=MUTED, lw=.6, ls='--')
    D.set_yticks([0, 1]); D.set_yticklabels(['Laboratory 2', 'Laboratory 1'], fontsize=5.6)
    D.set_ylim(-.7, 1.7); D.tick_params(axis='y', length=0)
    D.text(1.0, 1.26, 'fixed at 1', fontsize=5.2, color=INK2, ha='center')
    D.text(lv, -.44, f'{lv:.2f} (SE {sv:.2f}), $p$ = {float(J.loc["lambda_Vanderbilt","p"]):.3f}',
           fontsize=5.2, color=INK2, ha='center')
    D.set_xlabel('Loading on the shared phenotype')
    D.set_title('Shared construct, different gain', loc='left', color=INK)

    # e. donors contributing to the joint model
    nb, no = int(J.loc['lambda_Vanderbilt', 'n_both_labs']), int(J.loc['lambda_Vanderbilt', 'n_one_lab'])
    E.bar([0, 1], [nb, no], color=[L1, L2], width=.62)
    for k2, v in enumerate([nb, no]):
        E.text(k2, v + 1.5, str(v), ha='center', va='bottom', fontsize=5.8, color=INK)
    E.set_xticks([0, 1]); E.set_xticklabels(['both\nlaboratories', 'one\nlaboratory'], fontsize=5.4)
    E.set_ylabel('Donors'); E.set_ylim(0, max(nb, no) * 1.28)
    E.text(.5, .93, f'{nb + no} donors in the joint model', transform=E.transAxes,
           ha='center', va='top', fontsize=5.4, color=INK2)
    E.set_title('Who the joint model uses', loc='left', color=INK)

    # f. the HbA1c effect three ways, SD per 10 mmol/mol
    CC = pd.read_csv(T / 'censored_clinical.tsv', sep='\t')
    cp = CC[(CC.lab == 'Penn') & CC.outcome.str.startswith('Glucose suppression')].iloc[0]
    cv = CC[(CC.lab == 'Vanderbilt') & CC.outcome.str.startswith('Glucose suppression')].iloc[0]
    jl = float(J.loc['hba1c, in SD of the latent phenotype', 'estimate'])
    js = jl / float(J.loc['hba1c, in SD of the latent phenotype', 'z'])
    ways = [('Laboratory 1 alone', cp.beta_SD, cp.ci_low_SD, cp.ci_high_SD, int(cp.n), L1),
            ('Laboratory 2 alone', cv.beta_SD, cv.ci_low_SD, cv.ci_high_SD, int(cv.n), L2),
            ('Joint latent model', jl, jl - 1.96 * js, jl + 1.96 * js,
             int(J.loc['hba1c', 'n_donors']), CAT[2])]
    for k2, (nm, bb, lo, hi, nn, col) in enumerate(ways):
        F.errorbar(bb * PER10, 2 - k2,
                   xerr=[[(bb - lo) * PER10], [(hi - bb) * PER10]],
                   fmt='o', ms=3.2, lw=.8, color=col)
        F.text(.985, (2 - k2) + .32, f'n = {nn}', transform=F.get_yaxis_transform(),
               fontsize=5.0, color=INK2, ha='right', va='center')
    F.axvline(0, color=MUTED, lw=.6, ls='--')
    F.set_yticks([0, 1, 2]); F.set_yticklabels([w[0] for w in ways][::-1], fontsize=5.4)
    F.set_ylim(-.6, 2.75); F.tick_params(axis='y', length=0)
    F.set_xlabel('SD per 10 mmol/mol HbA1c (95% CI)')
    F.set_title('Three routes, one answer', loc='left', color=INK)

    # g. stimulus-specific share under four covariance structures
    CV = pd.read_csv(T / 'covariance_sensitivity.tsv', sep='\t')
    CV = CV[CV.hormone == 'glucagon']
    order = ['full', 'factor2', 'factor1', 'capacity_diag']
    lbl = {'full': 'full', 'factor2': '2 factors', 'factor1': '1 factor', 'capacity_diag': 'capacity only'}
    w = .36
    for k2, (lab, col, nm) in enumerate([('Penn', L1, 'Laboratory 1'), ('Vanderbilt', L2, 'Laboratory 2')]):
        s = CV[CV.lab == lab].set_index('structure').reindex(order)
        G.bar(np.arange(4) + (k2 - .5) * w, s.specific_share * 100, width=w, color=col, label=nm)
    G.set_xticks(range(4)); G.set_xticklabels([lbl[o] for o in order], fontsize=5.0)
    G.set_ylabel('Stimulus-specific variance (%)')
    G.legend(fontsize=5.2, frameon=False, loc='upper right')
    G.set_ylim(0, 13.5)
    G.set_title('Stable in laboratory 1 only', loc='left', color=INK)

    # h. coordination against HbA1c and type 2 diabetes
    BH0 = pd.read_csv(T / 'bihormonal_coordination.tsv', sep='\t')
    BH = BH0[BH0.predictor.isin(['hba1c', 'T2D'])].dropna(subset=['ci_low_SD'])
    nm = {'window_slope': 'window slope', 'fraction_corr': 'fraction correlation',
          'reciprocal_composite': 'composite'}
    rows = [(o, p) for o in ['window_slope', 'fraction_corr', 'reciprocal_composite']
            for p in ['hba1c', 'T2D']]
    for k2, (o, p) in enumerate(rows):
        r2 = BH[(BH.outcome == o) & (BH.predictor == p)].iloc[0]
        sc = PER10 if p == 'hba1c' else 1.0
        col = CAT[0] if p == 'hba1c' else CAT[1]
        H.errorbar(r2.beta_SD * sc, k2,
                   xerr=[[(r2.beta_SD - r2.ci_low_SD) * sc], [(r2.ci_high_SD - r2.beta_SD) * sc]],
                   fmt='o', ms=2.8, lw=.7, color=col)
    H.axvline(0, color=MUTED, lw=.6, ls='--')
    H.set_yticks(range(len(rows)))
    H.set_yticklabels([f'{nm[o]}, {"HbA1c" if p == "hba1c" else "T2D"}' for o, p in rows],
                      fontsize=4.8)
    H.invert_yaxis(); H.tick_params(axis='y', length=0)
    H.set_xlabel('SD per 10 mmol/mol HbA1c or for T2D')
    H.set_title('Coordination is preserved', loc='left', color=INK)

    # i. the two independent coordination measures
    BD = pd.read_csv(inputs.find(ROOT, 'donor_bihormonal.tsv'), sep='\t').dropna(subset=['window_slope', 'fraction_corr'])
    I.scatter(BD.window_slope, BD.fraction_corr, s=6, color=L1, alpha=.65, lw=0)
    ag = BH0[BH0.predictor == 'Spearman'].iloc[0]
    rho, nrho = float(ag.beta), int(ag.n)
    I.axhline(0, color=MUTED, lw=.6, ls='--'); I.axvline(0, color=MUTED, lw=.6, ls='--')
    I.text(.03, .97, f'Spearman $\\rho$ = {rho:.2f}\n$n$ = {nrho}', transform=I.transAxes,
           va='top', ha='left', fontsize=5.6, color=INK, linespacing=1.35)
    I.set_xlabel('Window-slope measure'); I.set_ylabel('Fraction-correlation measure')
    I.set_title('Two measures of coordination', loc='left', color=INK)

    for ax in (C, E, G):
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    for ax, l in zip([A, B, C, D, E, F, G, H, I], 'abcdefghi'):
        panel_label(ax, l, x=-0.40 if ax in (B, D, F, H) else -0.26, y=1.16)
    save_multi(fig, FIG / 'Figure4_crosslab_joint')
    print('wrote results/figures/Figure4_crosslab_joint.{svg,pdf,png}')


if __name__ == '__main__':
    main()
