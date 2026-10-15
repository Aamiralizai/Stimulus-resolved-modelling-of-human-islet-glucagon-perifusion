"""Step 25 - graphical abstract, single-panel version.

One message, one graph, minimal text, no mechanistic network and no statistics
beyond the effect itself.  Built for the revised version, which is when
Diabetologia asks for a graphical abstract.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from glucv2 import inputs
from glucv2.plotting import style, save_multi, CAT, INK, INK2, MUTED

NAVY, BLUE, TEAL, SAND = '#1f3b5c', '#2a78d6', '#1baf7a', '#e8a33d'
SUP_P = 'gls_Glucose suppression (3 to 16.7 mM)'
W, H = 7.4, 3.9


def load():
    E = pd.read_csv(inputs.find(ROOT, 'donor_effects_censored_Penn_glucagon.tsv'), sep='\t')
    D = inputs.donor_table(str(ROOT)).reset_index()
    j = E.merge(D[['donor', 'hba1c', 'age', 'bmi', 'sex', 'group', 'isolation_year']], on='donor')
    j = j[j.group.isin(['control', 'T2D'])].dropna(subset=['hba1c', 'age', 'bmi'])
    j['hba1c_mmol'] = (j['hba1c'] - 2.15) * 10.929
    return j


def partial(d, y, x):
    d = d.copy()
    d['male'] = (d.sex == 'Male').astype(int)
    d['isolation_year'] = d.isolation_year.fillna(d.isolation_year.median())
    C = sm.add_constant(d[['age', 'bmi', 'male', 'isolation_year']].astype(float))
    rx = sm.OLS(d[x].astype(float), C).fit().resid
    ry = sm.OLS(d[y].astype(float), C).fit().resid
    return rx + d[x].mean(), ry, d


AR = W / H          # the background axes spans 0 to 100 on both axes, so a circle
                    # of radius r needs height 2 r AR to look round


def islet(ax, cx, cy, r, seed=0):
    rng = np.random.default_rng(seed)
    ax.add_patch(Ellipse((cx, cy), 2 * r, 2 * r * AR, facecolor='#f0e3d4',
                         edgecolor='#d8bfa4', lw=.9, zorder=2))
    for i in range(16):
        a, dd = rng.uniform(0, 2 * np.pi), np.sqrt(rng.uniform(0, .55))
        col = SAND if i % 3 else BLUE
        ax.add_patch(Ellipse((cx + r * dd * np.cos(a), cy + r * AR * dd * np.sin(a)),
                             2 * r * .17, 2 * r * AR * .17, facecolor=col,
                             edgecolor='white', lw=.4, zorder=3))


def step(ax, x, y, w, h, text, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.0,
                                boxstyle='round,pad=0,rounding_size=1.4', zorder=1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7.2,
            color=NAVY, linespacing=1.45, zorder=3)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=9,
                                 lw=1.1, color=INK2, shrinkA=0, shrinkB=0, zorder=4))


def main():
    style()
    fig = plt.figure(figsize=(W, H))
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 100); bg.set_ylim(0, 100); bg.axis('off')

    bg.text(50, 95.5, 'Higher HbA1c, weaker glucose suppression of glucagon in human islets',
            ha='center', va='top', fontsize=10.2, weight='bold', color=NAVY)

    # left column, the chain
    islet(bg, 11.0, 78.0, 4.6, seed=3)
    bg.text(11.0, 66.0, 'Human islets, 150 donors', ha='center', va='top', fontsize=7.2,
            color=NAVY)
    arrow(bg, 11.0, 62.5, 11.0, 59.0)
    step(bg, 1.6, 42.5, 18.8, 16.0, 'Stimulus-resolved model\nseparates capacity from\neach stimulus response',
         '#e9f1fc', '#b4cdea')
    arrow(bg, 11.0, 42.0, 11.0, 38.5)
    step(bg, 1.6, 22.0, 18.8, 16.0, 'Replicated at a second\nlaboratory on islets\nfrom the same donors',
         '#e8f6ef', '#a9d8c1')
    arrow(bg, 11.0, 21.5, 11.0, 18.0)
    step(bg, 1.6, 1.5, 18.8, 16.0, 'Whole-islet respiration\ndoes not detectably\naccount for it',
         '#f3f2ef', '#d6d4ce')

    # the single graph
    j = load()
    rx, ry, d = partial(j, SUP_P, 'hba1c_mmol')
    ax = fig.add_axes([0.315, 0.145, 0.60, 0.605])
    cols = np.where(d.group.values == 'T2D', '#d94f4f', BLUE)
    ax.scatter(rx, ry, s=17, c=cols, alpha=.72, linewidths=0)
    b = np.polyfit(rx, ry, 1)
    xs = np.linspace(np.min(rx), np.max(rx), 30)
    ax.plot(xs, np.polyval(b, xs), color=INK, lw=1.6)
    ax.set_xlabel('HbA1c (mmol/mol)', fontsize=8.4, color=INK)
    ax.set_ylabel('Glucose suppression of glucagon\nweaker  $\\longrightarrow$', fontsize=8.4,
                  color=INK, linespacing=1.3)
    ax.tick_params(labelsize=7.2, colors=INK2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c9c7c2')
    ax.scatter([], [], s=17, c=BLUE, label='no diabetes')
    ax.scatter([], [], s=17, c='#d94f4f', label='type 2 diabetes')
    ax.legend(fontsize=7.0, frameon=False, loc='lower right', handletextpad=.2)
    ax.set_ylim(np.min(ry) - .06, np.max(ry) + .20)
    ax.text(.02, .975, '0.26 SD weaker suppression per 10 mmol/mol higher HbA1c',
            transform=ax.transAxes, fontsize=7.8, color=NAVY, va='top', ha='left', weight='bold')

    save_multi(fig, ROOT / 'results/figures/Graphical_abstract_simple')
    print('wrote results/figures/Graphical_abstract_simple.{svg,pdf,png}')


if __name__ == '__main__':
    main()
