"""Figure 2 - dynamic glucagon phenotype and its association with glycaemia.

Rebuilt from the result tables to the published legend, with HbA1c expressed in
mmol/mol as the journal requires.  Coefficients are rescaled, not relabelled.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from glucv2.plotting import style, save_multi, panel_label, CAT, INK, INK2, MUTED, GRID
from glucv2.data import WINDOWS
from glucv2.data import CONTRASTS
from glucv2 import inputs

T, P, FIG = ROOT / 'results/tables', ROOT / 'data/processed', ROOT / 'results/figures'
PCT_TO_MMOL = 10.929                      # mmol/mol = (DCCT% - 2.15) * 10.929
PER10 = 10 / PCT_TO_MMOL                  # per 1% coefficient -> per 10 mmol/mol
ND, T2D, T1D = CAT[0], CAT[7], CAT[3]
GC = {'control': ND, 'T2D': T2D, 'T1D': T1D}
SUP = {'Penn': 'Glucose suppression (3 to 16.7 mM)',
       'Vanderbilt': 'Glucose suppression (5.6 to 16.7 mM)'}
SHORT = {'Glucose suppression (3 to 16.7 mM)': 'Glucose supp.',
         'Glucose suppression (5.6 to 16.7 mM)': 'Glucose supp.',
         'Low-glucose response (AA to AA+G3)': 'Low glucose',
         'cAMP (IBMX) response': 'cAMP (IBMX)', 'AA response': 'Amino acids',
         'KCl response': 'KCl', 'Adrenaline response': 'Adrenaline', 'capacity': 'Capacity',
         'window AA+G3 (capacity-adjusted)': 'AA+G3 window (adj.)',
         'window AA (capacity-adjusted)': 'AA window (adj.)'}


def mmol(pct):
    return (np.asarray(pct, float) - 2.15) * PCT_TO_MMOL


def donors():
    D = inputs.donor_table(str(ROOT))
    D['hba1c_mmol'] = mmol(D.hba1c)
    return D


def eff(lab):
    return pd.read_csv(inputs.find(ROOT, f'donor_effects_censored_{lab}_glucagon.tsv'), sep='\t').set_index('donor')


def curves(ax, lab, L, D, title):
    C = inputs.curve_summary(str(ROOT), lab)
    tm = C.drop_duplicates('fraction').set_index('fraction').time_min
    for g in ['control', 'T2D', 'T1D']:
        s = C[C.group == g]
        if s.empty:
            continue
        ax.fill_between(s.time_min, s['mean'] - s['sem'], s['mean'] + s['sem'],
                        color=GC[g], alpha=.18, lw=0)
        ax.plot(s.time_min, s['mean'], color=GC[g], lw=1.1,
                label={'control': 'No diabetes', 'T2D': 'Type 2 diabetes', 'T1D': 'Type 1 diabetes'}[g])
    for nm, a, b in WINDOWS[lab]:
        if a in tm.index and b in tm.index:
            ax.axvspan(tm[a], tm[b], color=GRID, alpha=.6, lw=0, zorder=0)
    ax.set_xlabel('Time (min)'); ax.set_ylabel('Glucagon release')
    ax.set_title(title, loc='left', color=INK)
    ax.legend(fontsize=5.2, loc='upper left', handlelength=1.1)


def panel_b(ax):
    M = pd.read_csv(T / 'model_variance_decomposition_Penn.tsv', sep='\t')
    g = M[(M.hormone == 'glucagon')]
    names = ['capacity'] + [c for c in CONTRASTS['Penn']]
    sd, lo, hi, rel, labs = [], [], [], [], []
    for n in names:
        q = 'capacity variance' if n == 'capacity' else f'{n}: true SD'
        r = g[g.quantity == q]
        if r.empty:
            continue
        v = float(r.value.iloc[0]); l = float(r.ci_low.iloc[0]); h = float(r.ci_high.iloc[0])
        if n == 'capacity':
            v, l, h = np.sqrt(max(v, 0)), np.sqrt(max(l, 0)), np.sqrt(max(h, 0))
            rl = np.nan
        else:
            rr = g[g.quantity == f'{n}: reliability']
            rl = float(rr.value.iloc[0]) if not rr.empty else np.nan
        sd.append(v); lo.append(l); hi.append(h); rel.append(rl); labs.append(SHORT.get(n, n))
    y = np.arange(len(sd))
    ax.errorbar(sd, y, xerr=[np.array(sd) - np.array(lo), np.array(hi) - np.array(sd)],
                fmt='o', ms=3, lw=.8, color=ND)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=5.4); ax.invert_yaxis()
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Between-preparation SD (log units)')
    ax.set_title('Capacity and stimulus contrasts', loc='left', color=INK)
    for k, r in enumerate(rel):
        if np.isfinite(r):
            ax.text(1.02, k, f'{r:.2f}', transform=ax.get_yaxis_transform(),
                    fontsize=5.2, va='center', ha='left', color=INK2)
    ax.text(1.02, len(sd) - 0.1, 'reliability', transform=ax.get_yaxis_transform(),
            fontsize=5.2, va='center', ha='left', color=INK2, weight='bold')


def main():
    style()
    L = None
    D = donors(); E = eff('Penn')
    j = E.join(D[['group', 'sex', 'hba1c', 'hba1c_mmol']], how='inner')
    jc = j[j.group.isin(['control', 'T2D'])].dropna(subset=['hba1c_mmol'])
    sup = f'gls_{SUP["Penn"]}'

    fig = plt.figure(figsize=(7.2, 7.4))
    gs = GridSpec(3, 3, figure=fig, hspace=.62, wspace=.46,
                  left=.085, right=.955, top=.955, bottom=.07)
    A, B, C = (fig.add_subplot(gs[0, k]) for k in range(3))
    Dp, Ep, Fp = (fig.add_subplot(gs[1, k]) for k in range(3))
    G, H, I = (fig.add_subplot(gs[2, k]) for k in range(3))

    curves(A, 'Penn', L, D, 'Laboratory 1 glucagon')
    panel_b(B)
    curves(C, 'Vanderbilt', L, D, 'Laboratory 2 glucagon')

    # d. suppression by donor group, censored model, with medians
    rng = np.random.default_rng(5)
    for k, g in enumerate(['control', 'T2D']):
        v = j.loc[j.group == g, sup].dropna().values
        Dp.scatter(np.full(len(v), k) + rng.uniform(-.17, .17, len(v)), v, s=5,
                   color=GC[g], alpha=.6, lw=0)
        Dp.plot([k - .3, k + .3], [np.median(v)] * 2, color=INK, lw=1.4, zorder=5)
    Dp.set_xticks([0, 1]); Dp.set_xticklabels(['No\ndiabetes', 'Type 2\ndiabetes'], fontsize=6)
    Dp.axhline(0, color=MUTED, lw=.5); Dp.set_xlim(-.55, 1.55)
    Dp.set_ylabel('Glucose suppression\n(log units)')
    Dp.set_title('Suppression by group', loc='left', color=INK)

    # e. capacity against suppression
    for g in ['control', 'T2D', 'T1D']:
        s = j[j.group == g]
        Ep.scatter(s['gls_capacity'], s[sup], s=6, color=GC[g], alpha=.65, lw=0)
    M = pd.read_csv(T / 'model_variance_decomposition_Penn.tsv', sep='\t')
    cr = M[(M.hormone == 'glucagon') & (M.quantity == f'corr: {SUP["Penn"]} | capacity')]
    if not cr.empty:
        Ep.text(.03, .97, f'r = {float(cr.value.iloc[0]):.2f}', transform=Ep.transAxes,
                va='top', ha='left', fontsize=6, color=INK)
    Ep.set_xlabel('Capacity (log units)'); Ep.set_ylabel('Glucose suppression\n(log units)')
    Ep.set_title('Two contrasts, one random effect', loc='left', color=INK)

    # f. suppression by HbA1c tertile, in mmol/mol
    q = jc.hba1c_mmol.quantile([1/3, 2/3]).values
    band = np.digitize(jc.hba1c_mmol, q)
    labs = [f'$\\leq${q[0]:.0f}', f'{q[0]:.0f} to {q[1]:.0f}', f'$>${q[1]:.0f}']
    for k in range(3):
        v = jc.loc[band == k, sup].dropna().values
        Fp.scatter(np.full(len(v), k) + rng.uniform(-.15, .15, len(v)), v, s=5,
                   color=CAT[k if k < 2 else 7], alpha=.6, lw=0)
        Fp.plot([k - .3, k + .3], [np.median(v)] * 2, color=INK, lw=1.4, zorder=5)
    Fp.set_xticks(range(3)); Fp.set_xticklabels(labs, fontsize=5.6)
    Fp.axhline(0, color=MUTED, lw=.5)
    Fp.set_xlabel('HbA1c tertile (mmol/mol)'); Fp.set_ylabel('Glucose suppression\n(log units)')
    Fp.set_title('Suppression by HbA1c tertile', loc='left', color=INK)

    # g. suppression against HbA1c by sex
    for sx, col in [('Male', CAT[0]), ('Female', CAT[4])]:
        s = jc[jc.sex == sx].dropna(subset=[sup])
        G.scatter(s.hba1c_mmol, s[sup], s=6, color=col, alpha=.6, lw=0, label=sx)
        if len(s) > 3:
            b = np.polyfit(s.hba1c_mmol, s[sup], 1)
            xs = np.linspace(s.hba1c_mmol.min(), s.hba1c_mmol.max(), 30)
            G.plot(xs, np.polyval(b, xs), color=col, lw=1.0)
    G.axhline(0, color=MUTED, lw=.5)
    G.set_xlabel('HbA1c (mmol/mol)'); G.set_ylabel('Glucose suppression\n(log units)')
    G.set_title('Both sexes', loc='left', color=INK); G.legend(fontsize=5.2, loc='lower right')

    # h. suppression against HbA1c, with the censored adjusted estimate
    CC = pd.read_csv(T / 'censored_clinical.tsv', sep='\t')
    c = CC[(CC.lab == 'Penn') & (CC.outcome == SUP['Penn'])].iloc[0]
    for g in ['control', 'T2D']:
        s = jc[jc.group == g].dropna(subset=[sup])
        H.scatter(s.hba1c_mmol, s[sup], s=6, color=GC[g], alpha=.65, lw=0)
    b = np.polyfit(jc.hba1c_mmol, jc[sup], 1)
    xs = np.linspace(jc.hba1c_mmol.min(), jc.hba1c_mmol.max(), 40)
    H.plot(xs, np.polyval(b, xs), color=INK, lw=1.2)
    H.axhline(0, color=MUTED, lw=.5)
    H.text(.03, .97, f'{c.beta_SD * PER10:.2f} SD per 10 mmol/mol\n'
                     f'95% CI {c.ci_low_SD * PER10:.2f}, {c.ci_high_SD * PER10:.2f}\n'
                     f'$p$ = {c.p:.1e}, censored fit',
           transform=H.transAxes, va='top', ha='left', fontsize=5.6, color=INK, linespacing=1.3)
    lo_, hi_ = H.get_ylim(); H.set_ylim(lo_, hi_ + .42 * (hi_ - lo_))
    H.set_xlabel('HbA1c (mmol/mol)'); H.set_ylabel('Glucose suppression\n(log units)')
    H.set_title('Weaker suppression with HbA1c', loc='left', color=INK)

    # i. every glucagon phenotype, both laboratories, censored fit
    f = CC[CC.lab.isin(['Penn', 'Vanderbilt'])].copy()
    f = f[~f.outcome.str.contains('capacity-adjusted')].sort_values(['lab', 'outcome'])
    sig = multipletests(f.p.values, method='fdr_bh')[1] < 0.05
    y = np.arange(len(f))
    for k, (_, r) in enumerate(f.iterrows()):
        col = ND if r.lab == 'Penn' else T1D
        I.errorbar(r.beta_SD * PER10, k,
                   xerr=[[(r.beta_SD - r.ci_low_SD) * PER10], [(r.ci_high_SD - r.beta_SD) * PER10]],
                   fmt='o', ms=3, lw=.8, color=col)
        if sig[k]:
            I.text(1.02, k, '*', transform=I.get_yaxis_transform(), fontsize=8,
                   va='center', color=CAT[7])
    I.axvline(0, color=MUTED, lw=.6, ls='--')
    I.set_yticks(y)
    I.set_yticklabels([f"{r.lab[:4]}. {SHORT.get(r.outcome, r.outcome)}" for _, r in f.iterrows()],
                      fontsize=5.2)
    I.invert_yaxis(); I.tick_params(axis='y', length=0)
    I.set_xlabel('SD per 10 mmol/mol HbA1c (95% CI)')
    I.set_title('HbA1c associations, glucagon', loc='left', color=INK)

    for ax, l in zip([A, B, C, Dp, Ep, Fp, G, H, I], 'abcdefghi'):
        panel_label(ax, l, x=-0.3 if ax in (B, I) else -0.26, y=1.16)
    save_multi(fig, FIG / 'Figure2_dynamic_glucagon')
    print('wrote results/figures/Figure2_dynamic_glucagon.{svg,pdf,png}')


if __name__ == '__main__':
    main()
