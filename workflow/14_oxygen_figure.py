"""Step 14 - Figure 2 (3 x 3), islet oxygen consumption. SVG keeps all text editable.

Panel A is the protocol correction, which is the main point of this figure. The median trace is shown with the
switch times measured from the traces themselves, not assumed from the hormone protocol.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import statsmodels.formula.api as smf
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from glucv2.plotting import style, save_multi, panel_label, CAT, INK, INK2, MUTED, GRID

T = ROOT / 'results/tables'; P = ROOT / 'data/processed'; FIG = ROOT / 'results/figures'
BLUE, ORANGE, GREEN, GOLD, PINK, DKGREEN, PURPLE, RED = CAT
PHASE = {'basal': (20, 42), 'aam': (52, 72), 'glucose_3': (82, 104), 'glucose_16': (114, 131), 'maximal': (145, 161)}
LBL = {'basal': 'basal', 'aam': 'AAM', 'glucose_3': 'G3', 'glucose_16': 'G16.7', 'maximal': 'FCCP'}
NAME = {'level': 'Overall level', 'basal': 'Basal', 'aam': 'Amino acids', 'glucose_3': '3 mM glucose',
        'glucose_16': '16.7 mM glucose', 'maximal': 'Maximal (FCCP)', 'spare_capacity': 'Spare capacity',
        'aam_response': 'AAM response', 'glucose_3_response': '3 mM G response', 'glucose_16_response': '16.7 mM G response'}


def norm_traces():
    L = pd.read_csv(P / 'oxygen_long.tsv.gz', sep='\t')
    Q = pd.read_csv(T / 'oxygen_qc.tsv', sep='\t')
    ALL = L.pivot(index='donor', columns='t', values='ocr')          # every parsed trace, duplicates included
    M = ALL.loc[[d for d in ALL.index if d in set(Q[Q.usable].donor)]]  # analysed set only
    return ALL, M.div(M.loc[:, 20:42].median(1), axis=0), Q


def pA(ax, N, O):
    t = N.columns.values.astype(float)
    q = N.quantile([0.25, 0.5, 0.75])
    ax.fill_between(t, q.loc[0.25], q.loc[0.75], color=BLUE, alpha=0.18, lw=0)
    ax.plot(t, q.loc[0.5], color=BLUE, lw=1.6)
    for k, (a, b) in PHASE.items():
        ax.axvspan(a, b, color=GRID, alpha=0.8, lw=0, zorder=0)
        ax.text((a + b) / 2, 0.645, LBL[k], ha='center', va='bottom', fontsize=5.2, color=INK2)
    for r in O.itertuples():
        ax.axvline(r.onset_median, color=ORANGE if r.switch != 'terminal fall' else RED, lw=0.8, ls=(0, (3, 2)))
        ax.text(r.onset_median, 2.17, f'{r.onset_median:.0f}', ha='center', fontsize=5.4,
                color=ORANGE if r.switch != 'terminal fall' else RED)
    ax.set(xlim=(0, 195), ylim=(0.6, 2.3), xlabel='Time (min)', ylabel='OCR relative to basal')
    ax.set_title('Protocol recovered from the traces', loc='left')
    ax.text(0.03, 0.80, f'n = {len(N)}\nmedian, IQR', transform=ax.transAxes, ha='left', va='bottom',
            fontsize=5.8, color=INK2)


def pB(ax, O):
    o = O[O.switch != 'terminal fall']
    y = np.arange(len(O))[::-1]
    for i, r in zip(y, O.itertuples()):
        c = ORANGE if r.switch != 'terminal fall' else RED
        ax.plot([r.onset_q25, r.onset_q75], [i, i], color=c, lw=3, solid_capstyle='round', alpha=0.45)
        ax.plot(r.onset_median, i, 'o', color=c, ms=4.5, mec='white', mew=0.6)
        ax.text(r.onset_median, i + 0.26, f'{r.onset_median:.0f}', ha='center', fontsize=5.8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([s.replace('_', ' ').replace('glucose 3', '3 mM glucose')
                                          .replace('glucose 16', '16.7 mM glucose').replace('maximal', 'FCCP')
                                          .replace('aam', 'AAM') for s in O.switch])
    ax.set(xlabel='Measured onset (min)', xlim=(35, 180))
    ax.set_title('Switch times, median and IQR', loc='left')


def pC(ax, M):
    pairs = [('HPAP-049', 'HPAP-052'), ('HPAP-106', 'HPAP-108'), ('HPAP-185', 'HPAP-190')]
    t = M.columns.values.astype(float)
    for i, (a, b) in enumerate(pairs):
        off = i * 0.55
        ax.plot(t, M.loc[a].values + off, color=INK, lw=1.0)
        ax.plot(t, M.loc[b].values + off, color=RED, lw=1.0, ls=(0, (2, 1.6)))
        ax.text(3, off + 0.40, f'{a} / {b}', fontsize=5.6, color=INK2, ha='left')
    ax.set(xlabel='Time (min)', ylabel='OCR + offset', xlim=(0, 198), ylim=(0, 1.92))
    ax.set_title('Duplicated traces, both excluded', loc='left')


def pD(ax, Q):
    q = Q[Q.usable]
    ax.scatter(q.t_max, q.terminal_ratio, s=11, c=np.where(q.terminal_ratio < 0.8, RED, BLUE),
               alpha=0.8, lw=0.3, edgecolors='white')
    ax.axhline(0.8, color=INK2, lw=0.7, ls=(0, (3, 2)))
    ax.text(315, 0.84, 'inhibition threshold', fontsize=5.8, color=INK2, ha='right')
    ax.set(xlabel='Run length (min)', ylabel='Final OCR / basal', xlim=(160, 325))
    ax.set_xticks([175, 200, 250, 300])
    ax.set_title('No usable inhibitor floor', loc='left')
    ax.text(0.97, 0.95, f'{int((q.terminal_ratio >= 0.8).sum())} of {len(q)}\nend above 0.8', transform=ax.transAxes,
            fontsize=5.8, color=INK2, va='top', ha='right')


def strip(ax, d, col, ylab, ttl):
    rng = np.random.default_rng(3)
    xs = {('control', 'Female'): 0, ('control', 'Male'): 1, ('T2D', 'Female'): 2.6, ('T2D', 'Male'): 3.6}
    for (g, s), x0 in xs.items():
        v = d[(d.group == g) & (d.sex == s)][col].dropna().values
        c = BLUE if s == 'Female' else ORANGE
        ax.scatter(x0 + rng.uniform(-0.17, 0.17, len(v)), v, s=12, color=c, alpha=0.75, lw=0.3, edgecolors='white')
        ax.plot([x0 - 0.3, x0 + 0.3], [np.median(v)] * 2, color=INK, lw=1.4)
        ax.text((x0 + 0.6) / 4.8, 1.015, f'n={len(v)}', transform=ax.transAxes, ha='center', fontsize=5.4, color=MUTED)
    ax.set_xticks([0, 1, 2.6, 3.6]); ax.set_xticklabels(['F', 'M', 'F', 'M'])
    ax.set(ylabel=ylab, xlim=(-0.6, 4.2)); ax.set_title(ttl, loc='left', pad=11)
    for lb, xc in [('control', 0.5), ('T2D', 3.1)]:
        ax.annotate(lb, ((xc + 0.6) / 4.8, -0.17), xycoords='axes fraction', ha='center', fontsize=6.2, color=INK2)


def pG(ax, X):
    x = X.sort_values('interaction')
    y = np.arange(len(x))
    lo, hi = x.loc[:, 'loo_estimate_min'], x.loc[:, 'loo_estimate_max']
    for i, r in zip(y, x.itertuples()):
        sig = r.p_permutation < 0.05
        c = RED if sig else MUTED
        ax.plot([r.loo_estimate_min, r.loo_estimate_max], [i, i], color=c, lw=2.6, alpha=0.4, solid_capstyle='round')
        ax.plot(r.interaction, i, 'o', color=c, ms=4.6, mec='white', mew=0.6)
        ax.text(0.98, i, f'{r.p_permutation:.3f}', transform=ax.get_yaxis_transform(), ha='right', va='center',
                fontsize=5.6, color=INK if sig else MUTED)
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels([NAME[f] for f in x.feature], fontsize=6)
    ax.set(xlabel='T2D × male interaction (log OCR)', xlim=(-0.75, 1.15))
    ax.set_title('Sex interaction, LOO range', loc='left')


def pH(ax, d):
    for s, c in [('Female', BLUE), ('Male', ORANGE)]:
        x = d[(d.sex == s) & d.suppression.notna()]
        ax.scatter(x.spare_capacity, x.suppression, s=14, color=c, alpha=0.78, lw=0.3, edgecolors='white', label=s)
        if len(x) > 4:
            b = np.polyfit(x.spare_capacity, x.suppression, 1)
            xx = np.linspace(x.spare_capacity.min(), x.spare_capacity.max(), 20)
            ax.plot(xx, np.polyval(b, xx), color=c, lw=1.1)
    ax.legend(loc='upper right', handletextpad=0.3, borderpad=0.2)
    ax.set(xlabel='Spare capacity (log OCR)', ylabel='Glucagon suppression (log)')
    ax.set_title('Oxygen does not predict glucagon', loc='left')
    ax.text(0.97, 0.04, f'n = {int(d.suppression.notna().sum())}', transform=ax.transAxes, fontsize=5.8,
            color=INK2, ha='right')


def pI(ax, A):
    a = A[A.added != 'nothing'].copy()
    a['lab'] = ['Insulin response\n(same run)' if 'insulin' in s else NAME.get(s, s) for s in a.added]
    a = a.sort_values('change_pct')
    y = np.arange(len(a))
    cols = [PURPLE if 'insulin' in s else BLUE for s in a.added]
    ax.barh(y, a.change_pct, color=cols, height=0.68, lw=0)
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels(a.lab, fontsize=5.8)
    ax.set(xlabel='Change in HbA1c coefficient (%)')
    ax.set_title('Attenuation after adjustment', loc='left')


def main():
    style()
    ALL, N, Q = norm_traces()
    O = pd.read_csv(T / 'oxygen_onsets.tsv', sep='\t')
    X = pd.read_csv(T / 'oxygen_sex_checks.tsv', sep='\t')
    A = pd.read_csv(T / 'oxygen_covariate_attenuation.tsv', sep='\t')
    d = pd.read_csv(P / 'donor_oxygen.tsv', sep='\t', index_col=0)

    fig, ax = plt.subplots(3, 3, figsize=(7.4, 7.6))
    pA(ax[0, 0], N, O); pB(ax[0, 1], O); pC(ax[0, 2], ALL)
    pD(ax[1, 0], Q)
    strip(ax[1, 1], d, 'spare_capacity', 'Spare capacity (log OCR)', 'Spare capacity by group and sex')
    strip(ax[1, 2], d, 'glucose_16_response', '16.7 mM G response (log)', 'Glucose response by group and sex')
    pG(ax[2, 0], X); pH(ax[2, 1], d); pI(ax[2, 2], A)
    for a_, l in zip(ax.ravel(), 'ABCDEFGHI'): panel_label(a_, l)
    fig.tight_layout(h_pad=1.9, w_pad=1.5)
    save_multi(fig, FIG / 'Figure2_oxygen')
    print('wrote', FIG / 'Figure2_oxygen.svg')


if __name__ == '__main__':
    main()
