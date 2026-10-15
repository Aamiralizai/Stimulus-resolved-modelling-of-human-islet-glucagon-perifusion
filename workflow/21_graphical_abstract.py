"""Step 21 - graphical abstract, card layout.

Every number is taken from the result tables.  The respiratory panel reports the
qualified negative the paper actually found, not a mechanism it did not support.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from glucv2 import inputs
from glucv2.plotting import style, save_multi, CAT, INK, INK2, MUTED

NAVY = '#1f3b5c'
ND, T2D = '#2a78d6', '#d94f4f'
TINT = dict(rose=('#fdeef1', '#e9b9c2'), blue=('#e9f1fc', '#b4cdea'), violet=('#efeafb', '#c3b7e6'),
            green=('#e8f6ef', '#a9d8c1'), amber=('#fdf1e3', '#edc89a'), lav=('#f1eefb', '#c6bce8'))
SUP_P = 'gls_Glucose suppression (3 to 16.7 mM)'
SUP_V = 'gls_Glucose suppression (5.6 to 16.7 mM)'
W, H = 7.6, 5.15                                   # figure size, inches


def card(ax, x, y, w, h, title, tint):
    fc, ec = TINT[tint]
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.0,
                                boxstyle='round,pad=0,rounding_size=1.2', zorder=1))
    ax.text(x + w / 2, y + h - 2.0, title, ha='center', va='top', fontsize=7.6,
            weight='bold', color=NAVY, linespacing=1.25, zorder=3)


def inset(fig, x, y, w, h):
    """Card coordinates (0-100 x, 0-100 y) to a figure-fraction axes."""
    return fig.add_axes([x / 100, y / 100, w / 100, h / 100], zorder=4)


def islet(ax, cx, cy, r, shell, cells, n=13, seed=0):
    rng = np.random.default_rng(seed)
    ax.add_patch(Circle((cx, cy), r, facecolor=shell, edgecolor='none', zorder=2))
    for _ in range(n):
        a, d = rng.uniform(0, 2 * np.pi), r * np.sqrt(rng.uniform(0, .62))
        ax.add_patch(Circle((cx + d * np.cos(a), cy + d * np.sin(a) * .95), r * .17,
                            facecolor=cells, edgecolor='white', linewidth=.4, zorder=3))


def load():
    E = pd.read_csv(inputs.find(ROOT, 'donor_effects_censored_Penn_glucagon.tsv'), sep='\t')
    D = inputs.donor_table(str(ROOT)).reset_index()
    j = E.merge(D[['donor', 'hba1c', 'age', 'bmi', 'sex', 'group', 'isolation_year']], on='donor')
    j = j[j.group.isin(['control', 'T2D'])].dropna(subset=['hba1c', 'age', 'bmi'])
    # Diabetologia requires HbA1c in mmol/mol in figures.  Convert the values
    # themselves, so the plotted coefficient is rescaled rather than relabelled.
    j['hba1c_mmol'] = (j['hba1c'] - 2.15) * 10.929
    return j


def partial(d, y, x, covars=('age', 'bmi', 'male', 'isolation_year')):
    d = d.copy(); d['male'] = (d.sex == 'Male').astype(int)
    d['isolation_year'] = d.isolation_year.fillna(d.isolation_year.median())
    d = d.dropna(subset=[y, x] + list(covars))
    C = sm.add_constant(d[list(covars)].astype(float))
    return (sm.OLS(d[x].astype(float), C).fit().resid.values,
            sm.OLS(d[y].astype(float), C).fit().resid.values, d)


def tidy(ax, fs=5.6):
    ax.tick_params(labelsize=fs, length=2, pad=1.4)
    ax.xaxis.label.set_size(6.0); ax.yaxis.label.set_size(6.0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_facecolor('none')


def main():
    style()
    j = load()
    fig = plt.figure(figsize=(W, H))
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 100); bg.set_ylim(0, 100); bg.axis('off')

    # ---- title band -------------------------------------------------
    bg.add_patch(FancyBboxPatch((1.4, 88.6), 97.2, 10.2, facecolor=NAVY, edgecolor='none',
                                boxstyle='round,pad=0,rounding_size=1.2'))
    bg.text(50, 95.6, 'Stimulus-resolved modelling reveals weaker glucose suppression of glucagon '
                      'with higher HbA1c in human islets',
            ha='center', va='center', fontsize=8.8, weight='bold', color='white')
    bg.text(50, 91.1, 'Two laboratories, the same donors, with matched islet respirometry',
            ha='center', va='center', fontsize=7.2, color='#c6d6e8')

    # ---- row 1 ------------------------------------------------------
    card(bg, 1.4, 58.6, 23.0, 28.2, 'Human islet donors', 'rose')
    islet(bg, 8.4, 73.0, 3.5, '#f6d7c8', ND, seed=1)
    islet(bg, 18.0, 73.0, 3.5, '#f3c3bd', T2D, seed=2)
    bg.text(8.4, 67.6, 'No diabetes\n$n$ = 107', ha='center', va='top', fontsize=6.6, color=INK, linespacing=1.3)
    bg.text(18.0, 67.6, 'Type 2 diabetes\n$n$ = 58', ha='center', va='top', fontsize=6.6, color=INK, linespacing=1.3)
    bg.text(12.9, 61.0, '112 enter the primary association', ha='center', va='center',
            fontsize=6.1, color=INK2, style='italic')

    card(bg, 25.8, 58.6, 23.0, 28.2, 'Stimulus-resolved perifusion', 'blue')
    bg.add_patch(FancyBboxPatch((30.2, 65.0), 14.2, 8.0, facecolor='#d7e7f8', edgecolor='#8fb6de',
                                linewidth=.9, boxstyle='round,pad=0,rounding_size=.8'))
    for k, (dx, dy) in enumerate([(-3.6, 1.4), (-.3, -1.0), (3.1, 1.2), (.9, 2.3), (-1.9, -1.9)]):
        islet(bg, 37.3 + dx, 69.0 + dy, 1.35, '#f3dccb', '#e39a6e', n=5, seed=10 + k)
    bg.add_patch(FancyArrowPatch((27.6, 69.0), (30.0, 69.0), arrowstyle='-|>', mutation_scale=7,
                                 lw=1.0, color=INK2))
    bg.add_patch(FancyArrowPatch((44.6, 69.0), (47.0, 69.0), arrowstyle='-|>', mutation_scale=7,
                                 lw=1.0, color=INK2))
    bg.text(37.3, 79.4, 'Glucose  3 $\\rightarrow$ 16.7 mmol/l, on amino acids',
            ha='center', va='center', fontsize=6.2, color=INK)
    step_x = [30.4, 34.8, 34.8, 39.6, 39.6, 44.2]
    step_y = [76.2, 76.2, 77.8, 77.8, 76.2, 76.2]
    bg.plot(step_x, step_y, color=ND, lw=1.3, solid_joinstyle='miter')
    bg.text(37.3, 62.6, 'Seven windows, laboratory 1\nNine windows, laboratory 2',
            ha='center', va='center', fontsize=6.2, color=INK2, linespacing=1.35)

    card(bg, 50.2, 58.6, 23.0, 28.2, 'Phenotype by donor group', 'violet')
    ax1 = inset(fig, 54.8, 64.4, 16.0, 14.0)
    rng = np.random.default_rng(3)
    for k, (g, col, lab) in enumerate([('control', ND, 'No\ndiabetes'), ('T2D', T2D, 'Type 2\ndiabetes')]):
        v = j.loc[j.group == g, SUP_P].values
        ax1.scatter(np.full(len(v), k) + rng.uniform(-.17, .17, len(v)), v, s=5,
                    color=col, alpha=.55, linewidths=0)
        ax1.plot([k - .3, k + .3], [v.mean()] * 2, color=INK, lw=1.4, zorder=5)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(['No\ndiabetes', 'Type 2\ndiabetes'])
    ax1.set_ylabel('Glucose suppression\n(log units, higher is weaker)')
    ax1.axhline(0, color=MUTED, lw=.5)
    ax1.set_xlim(-.55, 1.55); tidy(ax1)

    card(bg, 74.6, 58.6, 24.0, 28.2, 'Agreement across two\nlaboratories, same donors', 'green')
    P = pd.read_csv(inputs.find(ROOT, 'donor_effects_censored_Penn_glucagon.tsv'), sep='\t')
    V = pd.read_csv(inputs.find(ROOT, 'donor_effects_censored_Vanderbilt_glucagon.tsv'), sep='\t')
    m = P[['donor', SUP_P]].merge(V[['donor', SUP_V]], on='donor').dropna()
    ax2 = inset(fig, 79.8, 64.6, 15.2, 13.2)
    ax2.scatter(m[SUP_P], m[SUP_V], s=5.5, color='#1baf7a', alpha=.65, linewidths=0)
    b = np.polyfit(m[SUP_P], m[SUP_V], 1)
    xs = np.linspace(m[SUP_P].min(), m[SUP_P].max(), 30)
    ax2.plot(xs, np.polyval(b, xs), color=INK, lw=1.1)
    ax2.set_xlabel('Laboratory 1'); ax2.set_ylabel('Laboratory 2')
    ax2.text(.97, .05, f'$r$ = 0.33,  $p$ = 0.0045\n$n$ = {len(m)}', transform=ax2.transAxes,
             va='bottom', ha='right', fontsize=5.9, color=INK, linespacing=1.3)
    tidy(ax2)

    # ---- row 2 ------------------------------------------------------
    card(bg, 1.4, 29.4, 48.0, 27.6, 'Continuous association with long-term glycaemia', 'amber')
    rx, ry, d = partial(j, SUP_P, 'hba1c_mmol')
    ax3 = inset(fig, 7.4, 35.8, 19.4, 14.0)
    for g, col in [('control', ND), ('T2D', T2D)]:
        k = (d.group == g).values
        ax3.scatter(rx[k], ry[k], s=6, color=col, alpha=.6, linewidths=0, label='No diabetes' if g == 'control' else 'Type 2 diabetes')
    xs = np.linspace(rx.min(), rx.max(), 40)
    ax3.plot(xs, np.polyval(np.polyfit(rx, ry, 1), xs), color=INK, lw=1.3)
    ax3.axhline(0, color=MUTED, lw=.5)
    ax3.set_xlabel('HbA1c (mmol/mol), adjusted'); ax3.set_ylabel('Glucose suppression\n(log units, adjusted)')
    ax3.legend(fontsize=5.4, loc='lower right', handlelength=.9, borderpad=.3)
    tidy(ax3)
    bg.text(38.8, 47.2, '0.26 SD weaker suppression\nper 10 mmol/mol higher HbA1c',
            ha='center', va='center', fontsize=7.2, weight='bold', color=NAVY, linespacing=1.35)
    bg.text(38.8, 41.0, '95% CI 0.17 to 0.34\n$p$ = 2.9 × 10$^{-9}$,   $n$ = 112',
            ha='center', va='center', fontsize=6.3, color=INK, linespacing=1.45)
    bg.text(38.8, 34.6, 'Holds within type 2 diabetes and\nacross 14 robustness checks',
            ha='center', va='center', fontsize=6.1, color=INK2, linespacing=1.45)

    card(bg, 50.8, 29.4, 47.8, 27.6, 'Matched islet respiration does not explain it', 'green')
    F = pd.read_csv(ROOT / 'results/tables/oxygen_features.tsv', sep='\t')
    Q = pd.read_csv(ROOT / 'results/tables/oxygen_qc.tsv', sep='\t')
    F = F[F.donor.isin(Q.loc[Q.usable.astype(bool), 'donor'])]
    k = j.merge(F[['donor', 'glucose_16_response']], on='donor')
    rx2, ry2, d2 = partial(k, SUP_P, 'glucose_16_response')
    ax4 = inset(fig, 56.4, 35.8, 16.4, 14.0)
    ax4.scatter(rx2, ry2, s=6, color='#1baf7a', alpha=.65, linewidths=0)
    xs = np.linspace(rx2.min(), rx2.max(), 40)
    ax4.plot(xs, np.polyval(np.polyfit(rx2, ry2, 1), xs), color=INK, lw=1.3)
    ax4.axhline(0, color=MUTED, lw=.5)
    ax4.set_xlabel('Islet respiration,\nglucose response')
    ax4.set_ylabel('Glucose suppression\n(log units, adjusted)')
    tidy(ax4)
    bg.text(86.8, 47.2, 'No respiratory measure\npredicts the phenotype',
            ha='center', va='center', fontsize=7.2, weight='bold', color=NAVY, linespacing=1.35)
    bg.text(86.8, 41.6, 'smallest $p$ = 0.26,   $n$ = 59',
            ha='center', va='center', fontsize=6.3, color=INK)
    bg.text(86.8, 35.4, 'Adding any respiratory measure\n'
                        'moves the HbA1c coefficient by\n'
                        'at most 2.7%. A qualified negative.',
            ha='center', va='center', fontsize=6.0, color=INK2, linespacing=1.5)

    # ---- row 3 ------------------------------------------------------
    card(bg, 1.4, 1.6, 97.2, 26.2, 'Where the mechanism must be sought, within the sensitivity of these data', 'lav')
    cy = 13.0
    cells = [(22.0, '#f0a3a3', 'Alpha cell', 'glucagon'),
             (43.0, '#9ec5f4', 'Beta cell', 'insulin'),
             (64.0, '#9bd9bb', 'Delta cell', 'somatostatin')]
    for cx, col, nm, prod in cells:
        bg.add_patch(Circle((cx, cy), 4.1, facecolor=col, edgecolor='white', linewidth=1.1, zorder=3))
        bg.add_patch(Circle((cx, cy), 1.5, facecolor='white', alpha=.55, edgecolor='none', zorder=4))
        bg.text(cx, cy - 6.3, nm, ha='center', va='center', fontsize=6.6, weight='bold', color=NAVY)
        bg.text(cx, cy - 9.0, prod, ha='center', va='center', fontsize=6.1, color=INK2)
    for x1, x2 in [(38.7, 26.4), (59.7, 47.4)]:          # beta inhibits alpha, delta inhibits beta
        bg.plot([x1, x2], [cy + 1.4, cy + 1.4], color=INK2, lw=1.1, zorder=5)
        bg.plot([x2, x2], [cy - .5, cy + 3.3], color=INK2, lw=1.6, zorder=5)
    bg.text(43.0, cy + 6.6, 'paracrine restraint on the alpha cell', ha='center', va='center',
            fontsize=6.2, color=INK2, style='italic')
    bg.add_patch(Rectangle((73.0, 4.4), .35, 17.2, facecolor=TINT['lav'][1], edgecolor='none'))
    bg.text(86.0, 18.6, 'Whole-islet respiration is ruled out\nas a detectable explanation',
            ha='center', va='center', fontsize=6.8, weight='bold', color=NAVY, linespacing=1.35)
    bg.text(86.0, 10.0, 'The remaining candidates are cell-type resolved.\n'
                        'Intrinsic alpha-cell glucose sensing, and\n'
                        'beta- and delta-cell paracrine regulation.\n'
                        'This is an association in donor islets, not causal proof.',
            ha='center', va='center', fontsize=6.0, color=INK, linespacing=1.5)

    save_multi(fig, ROOT / 'results/figures/Graphical_abstract', formats=('pdf', 'svg', 'png'))
    print('wrote results/figures/Graphical_abstract.{svg,pdf,png}')


if __name__ == '__main__':
    main()
