"""Step 20 - Figure 1, study design and analysis workflow.

A schematic only.  No data panels, because every quantity it would plot already
has its own figure later in the paper.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from glucv2.plotting import style, save_multi, CAT, INK, INK2, MUTED

L1, L2, OX = CAT[0], CAT[3], CAT[2]
RND = 'round,pad=0.5,rounding_size=0.9'


def box(ax, x, y, w, h, text, fc='white', ec=MUTED, fs=7.0, weight='normal', tc=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, facecolor=fc, edgecolor=ec,
                                linewidth=0.8, boxstyle=RND))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=tc,
            weight=weight, linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, color=MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=8,
                                 linewidth=0.9, color=color, shrinkA=1, shrinkB=1))


def main():
    style()
    fig, ax = plt.subplots(figsize=(7.2, 5.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    cols = [18.5, 50, 81.5]

    box(ax, 50, 96, 56, 7.4, 'Human Pancreas Analysis Program, PANC-DB\n216 donors with clinical data',
        fc='#eceae6', ec=INK2, fs=7.4, weight='bold')
    for x in cols:
        arrow(ax, 50, 92, x, 86.6)
    for x, t, c in zip(cols, ['Laboratory 1 perifusion\n161 runs',
                              'Laboratory 2 perifusion\n97 runs',
                              'Islet respirometry\n96 traces'], [L1, L2, OX]):
        box(ax, x, 82, 29, 7.6, t, fc=c + '22', ec=c)
    for x in cols:
        arrow(ax, x, 77.9, x, 72.8)

    box(ax, 50, 67.5, 92, 8.6,
        'Quality control\n'
        'Standard stimulus schedule  ·  no more than one tenth of fractions missing  ·  '
        'no more than one quarter at the assay floor\npositive median  ·  duplicated and flat '
        'respirometry traces excluded', fc='#eceae6', ec=INK2, fs=6.8)
    for x in cols:
        arrow(ax, x, 62.9, x, 57.8)

    for x, t, c in zip(cols, ['Usable runs\n134 glucagon  ·  153 insulin',
                              'Usable runs\n92 glucagon  ·  92 insulin',
                              'Usable traces\n89'], [L1, L2, OX]):
        box(ax, x, 53, 29, 7.6, t, fc=c + '22', ec=c)
    for x in cols:
        arrow(ax, x, 48.9, x, 43.8)
    for x, t, c in zip(cols, ['Stimulus-resolved model\n130 donors',
                              'Stimulus-resolved model\n91 donors',
                              'Respiratory features\nper donor'], [L1, L2, OX]):
        box(ax, x, 39, 29, 7.6, t, fc=c + '22', ec=c)
    for x in cols:
        arrow(ax, x, 34.9, x, 30.4)

    box(ax, 50, 25, 92, 9.2,
        'Analysis samples   ·   donors without diabetes and with type 2 diabetes, complete covariates\n'
        '112  primary association, laboratory 1        83  laboratory 2        127  joint two-laboratory model\n'
        '74  studied at both laboratories        59  respirometry with a laboratory 1 glucagon curve',
        fc='white', ec=INK, fs=6.9)
    ax.text(50, 17.6, 'Donors with type 1 diabetes contribute to the model fit and to no association reported here.',
            ha='center', va='center', fontsize=6.4, color=INK2, style='italic')

    arrow(ax, 50, 15.0, 50, 12.6)
    rows = [('Construction null', 'Can capacity alone reproduce\nthe dominant conventional axis?', CAT[6]),
            ('HbA1c association', 'Does stimulus-specific response\ntrack glycaemia?', CAT[7]),
            ('Left-censored likelihood', 'Does the assay floor create\nthe effect, or hide it?', CAT[0]),
            ('Two-laboratory comparison', 'Does it agree across protocols\nin the same donors?', CAT[3]),
            ('Influence and specification', 'Does it depend on any choice\nor any donor?', CAT[5]),
            ('Matched respirometry', 'Does whole-islet respiration\nexplain it?', CAT[2])]
    xs = [16.3, 50, 83.7]
    for k, (name, q, c) in enumerate(rows):
        x = xs[k % 3]; y = 6.6 if k < 3 else -5.9
        ax.add_patch(FancyBboxPatch((x - 15.2, y - 5.0), 30.4, 10.0, facecolor=c + '18',
                                    edgecolor=c, linewidth=0.8, boxstyle=RND))
        ax.text(x, y + 2.3, name, ha='center', va='center', fontsize=6.9, weight='bold', color=INK)
        ax.text(x, y - 1.9, q, ha='center', va='center', fontsize=6.2, color=INK2, linespacing=1.3)
    ax.set_ylim(-12.2, 100)
    save_multi(fig, ROOT / 'results/figures/Figure1_workflow')
    print('wrote results/figures/Figure1_workflow.{svg,pdf,png}')


if __name__ == '__main__':
    main()
