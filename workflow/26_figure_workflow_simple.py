"""Step 26 - simplified study flow diagram.

Donors in, quality control, analysis samples out.  No research-question boxes
and no result panels.  Intended for the ESM or the repository rather than the
main manuscript.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from glucv2.plotting import style, save_multi, INK, INK2

NAVY = '#1f3b5c'
W, H = 7.2, 4.6
TINT = {'top': ('#eef2f7', '#c3d0de'), 'lab1': ('#e9f1fc', '#b4cdea'),
        'lab2': ('#fdf1e3', '#edc89a'), 'ox': ('#e8f6ef', '#a9d8c1'),
        'out': ('#f4f3f0', '#d2d0ca')}


def box(ax, x, y, w, h, head, lines, tint, hs=8.0, bs=7.2):
    fc, ec = TINT[tint]
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.0,
                                boxstyle='round,pad=0,rounding_size=1.3', zorder=1))
    ax.text(x + w / 2, y + h - 2.6, head, ha='center', va='top', fontsize=hs,
            weight='bold', color=NAVY, zorder=3)
    ax.text(x + w / 2, y + h - 8.0, lines, ha='center', va='top', fontsize=bs,
            color=INK, linespacing=1.7, zorder=3)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=10,
                                 lw=1.2, color=INK2, shrinkA=0, shrinkB=0, zorder=4))


def main():
    style()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

    box(ax, 30, 84, 40, 13, 'Human Pancreas Analysis Program',
        '216 donors deposited in PANC-DB', 'top', hs=8.4, bs=7.6)

    for x in (17, 50, 83):
        arrow(ax, x if x == 50 else 50, 83.5, x, 72.5) if False else None
    arrow(ax, 50, 83.5, 50, 79.0)
    ax.plot([17, 83], [79, 79], color=INK2, lw=1.2, zorder=4)
    for x in (17, 50, 83):
        arrow(ax, x, 79.0, x, 72.0)

    box(ax, 2, 42, 30, 30, 'Laboratory 1 perifusion',
        '161 deposited runs\n158 on the standard schedule\n134 pass quality control\n130 enter the model',
        'lab1')
    box(ax, 35, 42, 30, 30, 'Laboratory 2 perifusion',
        '97 deposited runs\n97 on the standard schedule\n92 pass quality control\n91 enter the model',
        'lab2')
    box(ax, 68, 42, 30, 30, 'Islet respirometry',
        '96 deposited traces\n7 excluded at quality control\n89 usable traces\nmatched to the same donors',
        'ox')

    for x in (17, 50, 83):
        arrow(ax, x, 41.5, x, 36.0)
    ax.plot([17, 83], [36, 36], color=INK2, lw=1.2, zorder=4)
    arrow(ax, 50, 36.0, 50, 30.5)

    box(ax, 12, 4, 76, 26, 'Analysis samples',
        '112 donors, primary HbA1c association at laboratory 1\n'
        '83 donors, the same association at laboratory 2\n'
        '74 donors studied at both laboratories\n'
        '127 donors in the joint two-laboratory model\n'
        '59 donors with matched respirometry and complete covariates',
        'out', hs=8.4, bs=7.4)

    save_multi(fig, ROOT / 'results/figures/Figure_workflow_simple')
    print('wrote results/figures/Figure_workflow_simple.{svg,pdf,png}')


if __name__ == '__main__':
    main()
