"""Response windows and data matrices for the window model.

Windows are stimulus periods shifted by the flow lag (2 fractions at Penn, 1 fraction at Vanderbilt). At Penn the first
fraction after each lagged switch is dropped (switching transient) and the last fraction (t = 160 min) is dropped.
Missing values are interpolated only within a response window, never across a stimulus boundary, so an interpolated
point cannot smooth away the transition the analysis is trying to measure.
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]

WINDOWS = {
    'Penn': [('basal', 1, 12), ('AA', 14, 43), ('AA+G3', 45, 63), ('AA+G16.7', 65, 83), ('IBMX', 85, 103), ('wash', 105, 123), ('KCl', 125, 140)],
    'Vanderbilt': [('basal', 1, 4), ('G16.7', 5, 14), ('rec1', 15, 21), ('IBMX', 22, 28), ('rec2', 29, 31), ('Epi', 32, 38),
                   ('rec3', 39, 41), ('KCl', 42, 47), ('rec4', 48, 50)],
}
# stimulus-specific contrasts on window effects (window name -> weight). Responses are read relative to the preceding
# reference state, which removes the donor's capacity.
CONTRASTS = {
    'Penn': {'AA response': {'AA': 1, 'basal': -1}, 'Glucose suppression (3 to 16.7 mM)': {'AA+G16.7': 1, 'AA+G3': -1},
             'Low-glucose response (AA to AA+G3)': {'AA+G3': 1, 'AA': -1}, 'cAMP (IBMX) response': {'IBMX': 1, 'AA+G16.7': -1},
             'KCl response': {'KCl': 1, 'wash': -1}},
    'Vanderbilt': {'Glucose suppression (5.6 to 16.7 mM)': {'G16.7': 1, 'basal': -0.5, 'rec1': -0.5}, 'cAMP (IBMX) response': {'IBMX': 1, 'rec1': -1},
                   'Adrenaline response': {'Epi': 1, 'rec2': -1}, 'KCl response': {'KCl': 1, 'rec3': -1}},
}


def contrast_vector(lab, name):
    names = [w[0] for w in WINDOWS[lab]]; a = np.zeros(len(names))
    for k, v in CONTRASTS[lab][name].items(): a[names.index(k)] = v
    return a


def load_matrix(lab, hormone, scale='value', donors=None, min_frac_valid=0.9):
    L = pd.read_csv(ROOT / 'data/processed/perifusion_long.tsv.gz', sep='\t')
    R = pd.read_csv(ROOT / 'data/processed/perifusion_runs.tsv', sep='\t')
    ok = set(R[(R.lab == lab) & (R.hormone == hormone) & R.usable].donor)
    if donors is not None: ok &= set(donors)
    x = L[(L.lab == lab) & (L.hormone == hormone) & L.donor.isin(ok)]
    Y = x.pivot_table(index='donor', columns='fraction', values=scale, aggfunc='first')
    C = x.pivot_table(index='donor', columns='fraction', values='censored', aggfunc='first').reindex_like(Y).fillna(False).astype(bool)
    wins = WINDOWS[lab]; frac, win = [], []
    for k, (_, a, b) in enumerate(wins):
        f = [i for i in range(a, b + 1) if i in Y.columns]; frac += f; win += [k] * len(f)
    Y = Y[frac].astype(float); C = C[frac]
    pos = Y[Y > 0].stack(); off = float(pos.quantile(0.01)) if len(pos) else 1e-6
    Y = Y.mask(Y <= 0, off)   # zeros and negatives to the offset. Missing values stay missing (interpolated below)
    # interpolate isolated missing values (<= 3 per run) WITHIN each response window, never across a stimulus
    # boundary, then drop runs that still have a missing value or had more than 3 in total
    nmiss = Y.isna().sum(1); Y = Y[nmiss <= 3]; C = C.loc[Y.index]
    win = np.array(win)
    parts = [Y.loc[:, np.array(frac)[win == k]].interpolate(axis=1, limit_direction='both') for k in range(len(wins))]
    Y = pd.concat(parts, axis=1)
    keep = Y.notna().all(axis=1); Y = Y[keep]; C = C.loc[Y.index]
    LY = np.log(Y + off)
    mu = LY.mean(0)
    # a censored cell is recorded at the assay floor, so its log value is the upper bound on the latent value
    return {'donors': list(LY.index), 'Y': LY.values, 'mu': mu.values, 'R': (LY - mu).values, 'frac': np.array(frac), 'win': win,
            'windows': [w[0] for w in wins], 'censored': C.values, 'threshold': LY.values, 'offset': off}
