"""Step 07 - Figure 1 (3 x 3). SVG keeps all text editable."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from glucv2.plotting import style, save_multi, panel_label, CAT, INK, INK2, MUTED, GRID
from glucv2.data import WINDOWS

T = ROOT / 'results/tables'; P = ROOT / 'data/processed'; FIG = ROOT / 'results/figures'
BLUE, ORANGE, GREEN, GOLD, PINK, DKGREEN, PURPLE, RED = CAT
GC = {'control': BLUE, 'T2D': RED, 'T1D': MUTED}
SHORT = {'Glucose suppression (3 to 16.7 mM)': 'Glucose suppression', 'Glucose suppression (5.6 to 16.7 mM)': 'Glucose suppression',
         'Low-glucose response (AA to AA+G3)': 'Low glucose (AA→G3)', 'cAMP (IBMX) response': 'IBMX (cAMP)', 'Adrenaline response': 'Adrenaline',
         'KCl response': 'KCl', 'AA response': 'Amino acids', 'capacity': 'Capacity'}


def curves(ax, lab, hormone, ylab):
    L = pd.read_csv(P / 'perifusion_long.tsv.gz', sep='\t'); R = pd.read_csv(P / 'perifusion_runs.tsv', sep='\t')
    D = pd.read_csv(P / 'donors.tsv', sep='\t').set_index('donor')
    ok = set(R[(R.lab == lab) & (R.hormone == hormone) & R.usable].donor)
    fe = P / f'donor_effects_{lab}_{hormone}.tsv'   # draw the analysed curves only
    if fe.exists(): ok &= set(pd.read_csv(fe, sep='\t').donor)
    x = L[(L.lab == lab) & (L.hormone == hormone) & L.donor.isin(ok)].copy(); x['g'] = x.donor.map(D.group)
    off = x.value[x.value > 0].quantile(0.01); x['ly'] = np.log(x.value.clip(lower=off) + off)
    x = x[x.fraction < 141]
    for g in ['control', 'T2D']:
        y = x[x.g == g].groupby('fraction').ly.agg(['mean', 'sem'])
        ax.plot(y.index, y['mean'], color=GC[g], lw=1.1, label=f'{g} (n = {x[x.g == g].donor.nunique()})')
        ax.fill_between(y.index, y['mean'] - y['sem'], y['mean'] + y['sem'], color=GC[g], alpha=0.18, lw=0)
    lo, hi = ax.get_ylim()
    for k, (w, a, b) in enumerate(WINDOWS[lab]):
        if k % 2: ax.axvspan(a - 0.5, b + 0.5, color=GRID, alpha=0.6, lw=0, zorder=0)
        ax.text((a + b) / 2, hi, w.replace('AA+', '+').replace('G16.7', '16.7'), fontsize=4.6, ha='center', va='bottom', rotation=0, color=INK2)
    ax.set_xlabel('Fraction (1 min)'); ax.set_ylabel(ylab); ax.legend(loc='lower right', fontsize=5)


def main():
    style(); FIG.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(3, 3, figsize=(7.6, 8.8), gridspec_kw={'hspace': 0.9, 'wspace': 0.85})
    A, B, C, D_, E, F, G, H, I = axs.ravel()

    # A. Penn glucagon time courses
    curves(A, 'Penn', 'glucagon', 'log glucagon secretion'); A.set_title('Penn protocol, glucagon', loc='left', pad=12)

    # B. variance decomposition: true SD of stimulus contrasts (glucagon), both labs, with reliability
    V = pd.concat([pd.read_csv(T / f'model_variance_decomposition_{lab}.tsv', sep='\t') for lab in ['Penn', 'Vanderbilt']])
    v = V[(V.hormone == 'glucagon') & V.quantity.str.endswith(': true SD')].copy(); v['name'] = v.quantity.str.replace(': true SD', '')
    rel = V[(V.hormone == 'glucagon') & V.quantity.str.endswith(': reliability')].copy(); rel['name'] = rel.quantity.str.replace(': reliability', '')
    v = v.merge(rel[['lab', 'name', 'value']].rename(columns={'value': 'rel'}), on=['lab', 'name'])
    cap = V[(V.hormone == 'glucagon') & (V.quantity == 'capacity variance')]
    labels, y = [], 0
    for lab, col in [('Penn', BLUE), ('Vanderbilt', GOLD)]:
        c = cap[cap.lab == lab].iloc[0]
        B.errorbar(np.sqrt(c.value), y, xerr=[[np.sqrt(c.value) - np.sqrt(c.ci_low)], [np.sqrt(c.ci_high) - np.sqrt(c.value)]], fmt='s', color=col, ms=3.5, lw=0.8)
        labels.append(f'{lab[:4]}. capacity'); y += 1
        for r in v[v.lab == lab].itertuples():
            B.errorbar(r.value, y, xerr=[[r.value - r.ci_low], [r.ci_high - r.value]], fmt='o', color=col, ms=3, lw=0.8)
            B.text(B.get_xlim()[1] if False else 1.02, y, f'{r.rel:.2f}', fontsize=4.8, va='center', transform=B.get_yaxis_transform(), color=INK2)
            labels.append(f'{lab[:4]}. {SHORT.get(r.name, r.name)}'); y += 1
    B.set_yticks(range(len(labels))); B.set_yticklabels(labels, fontsize=5.2); B.invert_yaxis(); B.tick_params(axis='y', length=0)
    B.set_xlabel('Between-preparation SD (log units)'); B.set_title('Capacity and stimulus effects', loc='left')

    # C. construction null: conventional axes retained, model stimulus-specific share
    S = pd.read_csv(T / 'null_simulations.tsv', sep='\t'); real = pd.read_csv(T / 'null_real_axes.tsv', sep='\t').iloc[0]
    sc = list(S.scenario.unique()); sl = ['A null', 'B indep.', 'C fitted', 'D pos. ctrl']
    for k, s in enumerate(sc):
        d = S[S.scenario == s]
        C.scatter(np.full(len(d), k) + np.random.default_rng(k).uniform(-0.15, 0.15, len(d)), d.conv_axes_retained, s=6, color=MUTED, alpha=0.6, lw=0)
    C.axhline(real.pankbase_axes, color=RED, ls='--', lw=0.8); C.text(len(sc) - 0.5, real.pankbase_axes + 0.1, f'PanKbase real data ({int(real.pankbase_axes)})', fontsize=5, color=RED, ha='right', va='bottom')
    C.set_xticks(range(len(sc))); C.set_xticklabels(sl, fontsize=5.4, rotation=30, ha='right'); C.set_ylabel('Axes retained, conventional traits')
    C.set_ylim(0, max(S.conv_axes_retained.max(), real.pankbase_axes) + 1); C.set_title('Conventional axes, simulated data', loc='left')

    # D. model under the null: estimated vs true stimulus-specific share and glucose-suppression SD
    for k, s in enumerate(sc):
        d = S[S.scenario == s]; x0 = k
        D_.scatter(np.full(len(d), x0 - 0.12) + np.random.default_rng(k).uniform(-0.06, 0.06, len(d)), d.glucose_suppression_SD_est, s=6, color=BLUE, alpha=0.6, lw=0)
        D_.plot([x0 - 0.3, x0 + 0.3], [d.glucose_suppression_SD_true.iloc[0]] * 2, color=RED, lw=1.2)
    D_.set_xticks(range(len(sc))); D_.set_xticklabels(sl, fontsize=5.4, rotation=30, ha='right'); D_.set_ylabel('Glucose-suppression SD, model')
    D_.plot([], [], color=RED, label='simulated truth'); D_.scatter([], [], s=6, color=BLUE, label='estimate (N = 400)'); D_.legend(fontsize=5, loc='upper left')
    D_.set_title('Model recovers the truth', loc='left')

    # E. congruence of null-simulated conventional axes with the real PanKbase axes
    cols = [c for c in S.columns if c.startswith('congruence_pankbase_axis')]
    dA = S[S.scenario == sc[0]]
    E.boxplot([dA[c].dropna() for c in cols], widths=0.5, patch_artist=True, boxprops=dict(facecolor=GRID, color=MUTED), medianprops=dict(color=RED),
              whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED), flierprops=dict(markersize=2))
    E.axhline(0.85, color=INK2, ls=':', lw=0.7); E.text(len(cols) + 0.45, 0.86, '0.85', fontsize=5, color=INK2, ha='right')
    E.set_xticklabels([f'Axis {i + 1}' for i in range(len(cols))]); E.set_ylim(0, 1.05); E.set_ylabel('Congruence with PanKbase axes')
    E.set_title('Null vs PanKbase axes', loc='left')

    # F. cross-lab replication
    X = pd.read_csv(T / 'crosslab_replication.tsv', sep='\t'); X = X[(X.kind == 'matched')]
    yy = 0; labs = []
    for hormone, mk in [('glucagon', 'o')]:
        for r in X[X.hormone == hormone].itertuples():
            col = BLUE if r.measure == 'model' else MUTED
            F.errorbar(r.r_observed, yy, xerr=[[r.r_observed - r.ci_low], [r.ci_high - r.r_observed]], fmt=mk, color=col, ms=3, lw=0.8)
            labs.append(f"{SHORT.get(r.penn, r.penn) + ' (model)' if r.measure == 'model' else r.penn + ' (conv.)'}"); yy += 1
    F.axvline(0, color=MUTED, lw=0.6, ls='--'); F.set_yticks(range(len(labs))); F.set_yticklabels(labs, fontsize=5); F.invert_yaxis(); F.tick_params(axis='y', length=0)
    F.set_xlabel('Penn vs Vanderbilt, r (95% CI)'); F.set_title('Replication between labs, glucagon', loc='left')

    # G. glucose suppression, Penn vs Vanderbilt
    Ph = pd.read_csv(P / 'donor_phenotypes.tsv', sep='\t', index_col=0); Dn = pd.read_csv(P / 'donors.tsv', sep='\t').set_index('donor')
    a, b = 'Penn | glucagon | Glucose suppression (3 to 16.7 mM)', 'Vanderbilt | glucagon | Glucose suppression (5.6 to 16.7 mM)'
    d = Ph[[a, b]].dropna(); g = Dn.loc[d.index, 'group']
    for grp in ['control', 'T2D']:
        G.scatter(d.loc[g == grp, a], d.loc[g == grp, b], s=8, color=GC[grp], lw=0, alpha=0.8, label=grp)
    r = X[(X.hormone == 'glucagon') & (X.penn == 'Glucose suppression (3 to 16.7 mM)') & (X.measure == 'model')].iloc[0]
    G.text(0.03, 0.97, f'r = {r.r_observed:.2f} (P = {r.p:.2g})\ntrue-effect r = {r.r_true_effects:.2f}\nn = {int(r.n)}', transform=G.transAxes, fontsize=5.3, va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1))
    lo_, hi_ = G.get_ylim(); G.set_ylim(lo_, hi_ + 0.35 * (hi_ - lo_))   # headroom for the annotation
    G.set_xlabel('Penn, 3 to 16.7 mM glucose'); G.set_ylabel('Vanderbilt, 5.6 to 16.7 mM glucose'); G.legend(fontsize=5, loc='lower right')
    G.set_title('Glucagon suppression by glucose', loc='left')

    # H. Penn glucose suppression vs HbA1c
    d = pd.concat([Ph[a].rename('y'), Dn], axis=1, join='inner').dropna(subset=['y', 'hba1c']); d = d[d.group.isin(['control', 'T2D'])]
    for grp in ['control', 'T2D']:
        H.scatter(d.loc[d.group == grp, 'hba1c'], d.loc[d.group == grp, 'y'], s=8, color=GC[grp], lw=0, alpha=0.8, label=grp)
    co = np.polyfit(d.hba1c, d.y, 1); xs = np.linspace(d.hba1c.min(), d.hba1c.max(), 50); H.plot(xs, np.polyval(co, xs), color=INK, lw=0.8)
    Cl = pd.read_csv(T / 'clinical_associations.tsv', sep='\t'); c = Cl[(Cl.predictor == 'HbA1c') & (Cl.outcome == a)].iloc[0]
    H.text(0.03, 0.97, f'β = {c.beta:.3f} log units per 1% HbA1c\nP = {c.p:.1e} (adjusted)', transform=H.transAxes, fontsize=5.3, va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1))
    lo_, hi_ = H.get_ylim(); H.set_ylim(lo_, hi_ + 0.3 * (hi_ - lo_))
    H.axhline(0, color=MUTED, lw=0.5, ls=':'); H.set_xlabel('HbA1c (%)'); H.set_ylabel('Suppression, log(16.7 / 3 mM)'); H.legend(fontsize=5, loc='lower right')
    H.set_title('Weaker suppression with HbA1c', loc='left')

    # I. forest: HbA1c associations, glucagon phenotypes, both labs
    f = Cl[(Cl.predictor == 'HbA1c') & Cl.outcome.str.contains('glucagon') & (Cl.family == 'phenotypes')].copy()
    f['lab'] = f.outcome.str.split(' \\| ').str[0]; f['name'] = f.outcome.str.split(' \\| ').str[2]
    f = f.sort_values(['lab', 'name']); yy = np.arange(len(f))
    for k, r in enumerate(f.itertuples()):
        col = {'Penn': BLUE, 'Vanderbilt': GOLD, 'Combined': PURPLE}[r.lab]; sd = r.beta / r.beta_per_outcome_SD if r.beta_per_outcome_SD else 1
        I.errorbar(r.beta_per_outcome_SD, k, xerr=[[(r.beta - r.ci_low) / sd], [(r.ci_high - r.beta) / sd]], fmt='o', color=col, ms=3, lw=0.8)
        if r.fdr < 0.05: I.text(1.02, k, '*', transform=I.get_yaxis_transform(), fontsize=7, va='center', color=RED)
    I.axvline(0, color=MUTED, lw=0.6, ls='--'); I.set_yticks(yy)
    I.set_yticklabels([f"{r.lab[:4]}. {SHORT.get(r.name, r.name.replace('Glucose suppression (z, mean of labs)', 'Glucose supp. (z)'))}" for r in f.itertuples()], fontsize=5)
    I.invert_yaxis(); I.tick_params(axis='y', length=0); I.set_xlabel('SD per 1% HbA1c (95% CI)'); I.set_title('HbA1c associations, glucagon', loc='left')

    for ax, l in zip([A, B, C, D_, E, F, G, H, I], 'ABCDEFGHI'):
        panel_label(ax, l, x={'A': -0.32, 'B': -0.3, 'C': -0.3, 'F': -0.3, 'I': -0.3}.get(l, -0.25), y=1.14)
    save_multi(fig, FIG / 'Figure1_dynamic_glucagon'); print('wrote', FIG / 'Figure1_dynamic_glucagon.*')


if __name__ == '__main__':
    main()
