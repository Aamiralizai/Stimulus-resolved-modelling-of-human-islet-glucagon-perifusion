"""Conventional perifusion summary traits (AUC, stimulation and inhibition indices) computed from time courses.

Vanderbilt protocol (3-min fractions, 50 fractions). Definitions reproduce the PanKbase GCG-IEQ traits for the HPAP
donors (Spearman rho 0.83 to 1.00, see step 04): basal = mean of fractions 1-3; G16.7 AUC = area below basal over
fractions 4-20; G16.7 II = basal / minimum of fractions 5-13; IBMX AUC = area above the mean of fractions 19-20 over
22-30, IBMX SI = peak of 22-26 / mean of 18-20; Epi AUC = area above the mean of 29-30 over 32-36, Epi SI = peak of 32-36
/ fraction 30; KCl AUC = area above fraction 40 over 41-46, KCl SI = peak of 42-50 / fraction 40.
Penn protocol (1-min fractions, 141): analogous traits with the preceding period as baseline.
"""
import numpy as np, pandas as pd


def _w(Y, a, b): return Y[:, a - 1:b]


def vanderbilt_traits(Y):
    """Y: N x 50 linear-scale secretion (fraction 1..50)."""
    b = _w(Y, 1, 3).mean(1); m = lambda a, c: _w(Y, a, c).mean(1)
    T = {'Basal': b,
         'G16.7 AUC': np.clip(b[:, None] - _w(Y, 4, 20), 0, None).sum(1) * 3,
         'G16.7 II': b / np.maximum(_w(Y, 5, 13).min(1), 1e-9),
         'IBMX AUC': np.clip(_w(Y, 22, 30) - m(19, 20)[:, None], 0, None).sum(1) * 3,
         'IBMX SI': _w(Y, 22, 26).max(1) / m(18, 20),
         'Epi AUC': (_w(Y, 32, 36) - m(29, 30)[:, None]).sum(1) * 3,
         'Epi SI': _w(Y, 32, 36).max(1) / Y[:, 29],
         'KCl AUC': np.clip(_w(Y, 41, 46) - Y[:, 39][:, None], 0, None).sum(1) * 3,
         'KCl SI': _w(Y, 42, 50).max(1) / Y[:, 39]}
    return pd.DataFrame(T)


def penn_traits(Y):
    """Y: N x 141 linear-scale secretion (1-min fractions)."""
    m = lambda a, c: _w(Y, a, c).mean(1)
    b = m(1, 10)
    T = {'Basal': b,
         'AA AUC': np.clip(_w(Y, 13, 43) - b[:, None], 0, None).sum(1), 'AA SI': _w(Y, 13, 43).max(1) / b,
         'AA+G3 AUC': _w(Y, 44, 63).sum(1), 'G16.7 II': m(54, 63) / np.maximum(m(74, 83), 1e-9),
         'IBMX AUC': np.clip(_w(Y, 84, 103) - m(74, 83)[:, None], 0, None).sum(1), 'IBMX SI': _w(Y, 84, 103).max(1) / m(74, 83),
         'KCl AUC': np.clip(_w(Y, 124, 140) - m(114, 123)[:, None], 0, None).sum(1), 'KCl SI': _w(Y, 124, 140).max(1) / m(114, 123)}
    return pd.DataFrame(T)


def inverse_normal(x):
    from scipy.stats import norm, rankdata
    x = np.asarray(x, float); r = rankdata(x); return norm.ppf((r - 0.5) / len(x))


def parallel_analysis(X, n_iter=200, seed=0):
    """Number of components whose eigenvalue exceeds the 95th percentile of eigenvalues from column-permuted data."""
    rng = np.random.default_rng(seed); Z = (X - X.mean(0)) / X.std(0)
    ev = np.sort(np.linalg.eigvalsh(np.corrcoef(Z.T)))[::-1]; null = []
    for _ in range(n_iter):
        P = np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])])
        null.append(np.sort(np.linalg.eigvalsh(np.corrcoef(P.T)))[::-1])
    thr = np.quantile(np.array(null), 0.95, axis=0); k = 0
    while k < len(ev) and ev[k] > thr[k]: k += 1
    return k, ev


def varimax(L, gamma=1.0, q=50, tol=1e-6):
    p, k = L.shape; R = np.eye(k); d = 0
    for _ in range(q):
        Lr = L @ R; u, s, vh = np.linalg.svd(L.T @ (Lr ** 3 - (gamma / p) * Lr @ np.diag((Lr ** 2).sum(0))))
        R = u @ vh; d_old, d = d, s.sum()
        if d_old != 0 and d / d_old < 1 + tol: break
    return L @ R


def axes(T, k):
    Z = np.column_stack([inverse_normal(T[c]) for c in T.columns]); C = np.corrcoef(Z.T)
    w, U = np.linalg.eigh(C); o = np.argsort(w)[::-1][:k]; L = U[:, o] * np.sqrt(w[o])
    return varimax(L) if k > 1 else L


def congruence(L1, L2):
    """Best-match Tucker congruence of each column of L1 with any column of L2."""
    out = []
    for i in range(L1.shape[1]):
        c = [abs(L1[:, i] @ L2[:, j]) / np.sqrt((L1[:, i] @ L1[:, i]) * (L2[:, j] @ L2[:, j])) for j in range(L2.shape[1])]
        out.append(max(c))
    return out
