"""Tests for the window model and the censored fit. Discoverable by pytest, and runnable directly with
`python tests/test_model.py`.

test_recovery        simulate from a known Sigma with AR(1) plus nugget noise on the Penn grid and check that the fit
                     recovers the capacity variance, the noise parameters and the per-window specific variances.
test_null            simulate with capacity only and check that the estimated stimulus-specific share stays near zero
                     and the likelihood-ratio statistic stays near its degrees of freedom.
test_gradient        compare the analytic gradient with a central difference for every covariance structure.
test_factor_nested   the factor-analytic and capacity structures are nested inside the unstructured one, so their
                     log-likelihoods must be ordered.
test_truncated_moments  the Gibbs sampler reproduces the closed-form moments of a univariate truncated normal.
test_censored_recovery  simulate with an assay floor and check that ignoring the floor biases the window means upward
                     while the censored fit removes most of that bias.
"""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from glucv2.model import WindowModel, decompose
from glucv2.data import WINDOWS
from glucv2.censored import CensoredFit, _gibbs, _trunc1


def grid(lab='Penn'):
    frac, win = [], []
    for k, (_, a, b) in enumerate(WINDOWS[lab]):
        f = list(range(a, b + 1)); frac += f; win += [k] * len(f)
    return np.array(frac, float), np.array(win)


@pytest.fixture(scope='module')
def truth():
    frac, win = grid(); W = int(win.max() + 1); rng = np.random.default_rng(1)
    A = rng.normal(size=(W, 2)) * 0.25
    Sig = 0.4 * np.ones((W, W)) + A @ A.T + np.diag(np.full(W, 0.02))
    return frac, win, {'Sigma': Sig, 's2': np.full(W, 0.05), 'rho': 0.7, 'eta': 0.3}


def test_recovery(truth):
    frac, win, t = truth
    m0 = WindowModel(np.zeros((134, len(frac))), frac, win)
    R, _ = m0.simulate(t, rng=2); m = WindowModel(R - R.mean(0), frac, win); f = m.fit('full')
    assert f['converged']
    dt, df = decompose(t['Sigma']), decompose(f['Sigma'])
    assert abs(df['capacity_var'] / dt['capacity_var'] - 1) < 0.35
    assert abs(f['rho'] - t['rho']) < 0.1 and abs(f['eta'] - t['eta']) < 0.1
    assert np.corrcoef(dt['specific_var_per_window'], df['specific_var_per_window'])[0, 1] > 0.8


def test_null(truth):
    frac, win, t = truth; W = int(win.max() + 1)
    m0 = WindowModel(np.zeros((134, len(frac))), frac, win)
    R, _ = m0.simulate(dict(t, Sigma=0.4 * np.ones((W, W))), rng=3)
    m = WindowModel(R - R.mean(0), frac, win); fn, fc = m.fit('full'), m.fit('capacity')
    assert decompose(fn['Sigma'])['specific_share'] < 0.03
    assert 2 * (fn['loglik'] - fc['loglik']) < 2 * (fn['n_par'] - fc['n_par'])


@pytest.mark.parametrize('kind', ['full', 'capacity', 'capacity_diag', 'factor1', 'factor2'])
def test_gradient(truth, kind):
    frac, win, t = truth
    m0 = WindowModel(np.zeros((40, len(frac))), frac, win)
    R, _ = m0.simulate(t, rng=5); m = WindowModel(R - R.mean(0), frac, win)
    th = m.start(kind); _, g = m.nll_grad(th, kind)
    num = np.array([(m.nll(th + e, kind) - m.nll(th - e, kind)) / 2e-6 for e in 1e-6 * np.eye(len(th))])
    assert np.abs(g - num).max() / max(np.abs(num).max(), 1e-8) < 1e-5


def test_factor_nested(truth):
    frac, win, t = truth
    m0 = WindowModel(np.zeros((120, len(frac))), frac, win)
    R, _ = m0.simulate(t, rng=7); m = WindowModel(R - R.mean(0), frac, win)
    ll = {k: m.fit(k)['loglik'] for k in ['capacity', 'capacity_diag', 'factor1', 'factor2', 'full']}
    assert ll['capacity'] <= ll['capacity_diag'] + 1e-6
    assert ll['factor1'] <= ll['factor2'] + 1e-6
    assert ll['factor2'] <= ll['full'] + 1e-6


def test_truncated_moments():
    rng = np.random.default_rng(0)
    m, sd, c = np.array([0.4]), 1.3, np.array([-0.2])
    exact_m, exact_v = _trunc1(m[0], sd, c[0])
    gm, gc, _ = _gibbs(m, np.array([[sd ** 2]]), c, rng, n=4000, burn=200)
    assert abs(gm[0] - exact_m) < 0.05
    assert abs(np.sqrt(max(gc[0, 0], 0)) - np.sqrt(exact_v)) < 0.08


def test_censored_recovery(truth):
    frac, win, t = truth
    m0 = WindowModel(np.zeros((90, len(frac))), frac, win)
    R, _ = m0.simulate(t, rng=11); L = R + 2.0
    floor = np.quantile(L, 0.12); cens = L <= floor
    Lc = np.where(cens, floor, L); thr = np.full_like(L, floor)
    naive_bias = abs(Lc.mean() - L.mean())
    cf = CensoredFit(Lc, cens, thr, frac, win, seed=2); f = cf.fit('full', iters=8, tail=4)
    cens_bias = abs(f['mu'].mean() - L.mean())
    assert naive_bias > 0.01
    assert cens_bias < naive_bias


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
