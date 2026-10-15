"""Stimulus-resolved hierarchical model for perifusion time courses.

For donor i and fraction t (log scale, population mean curve mu_t removed):
    r_it = delta_{i,w(t)} + e_it
    delta_i = (delta_i1..delta_iW) ~ N(0, Sigma)                window-level donor effects
    e_i ~ N(0, D^1/2 [(1 - eta) * rho^|f_s - f_t| + eta * 1(s = t)] D^1/2)   within-run noise, AR(1) plus nugget,
    D = diag(s2_w(t)), a separate noise variance for each window (hetero=True, default) or one common s2.
w(t) maps each analysed fraction to one response window (stimulus period shifted by the flow lag).
Capacity and stimulus-specific responsiveness are read from Sigma:
    capacity c_i = mean_w delta_iw,  stimulus-specific r_iw = delta_iw - c_i.
Because replicate fractions inform the within-run noise, Sigma is the covariance of the true donor effects. Sampling
noise, including the noise that conventional stimulation indices share through a common basal denominator, is
modelled in e and does not enter Sigma.
Models: 'full' (unstructured Sigma), 'capacity' (Sigma = tau2 * 11'), 'capacity_diag' (tau2 * 11' + diag(psi)),
'factorQ' (Sigma = Lambda Lambda' + diag(psi) with Q factors, a shrinkage alternative to the unstructured fit).
Fitting is maximum likelihood (all donors share the fraction grid, so Omega is common to all donors).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve


def ar_matrix(frac, rho, eta):
    d = np.abs(np.subtract.outer(frac, frac))
    return (1 - eta) * rho ** d + eta * np.eye(len(frac))


class WindowModel:
    def __init__(self, R, frac, win, W=None, hetero=True):
        """R: N x T residual matrix (log scale, mean curve removed), frac: fraction numbers (T), win: window index (T)."""
        self.R = np.asarray(R, float); self.frac = np.asarray(frac, float); self.win = np.asarray(win, int); self.hetero = hetero
        self.N, self.T = self.R.shape; self.W = W or int(self.win.max() + 1)
        self.Z = np.zeros((self.T, self.W)); self.Z[np.arange(self.T), self.win] = 1
        self.S = self.R.T @ self.R  # T x T scatter

    # ---- parameterisation ----
    def unpack(self, th, kind):
        W = self.W; k = 0
        if kind == 'full':
            Lm = np.zeros((W, W)); idx = np.tril_indices(W)
            Lm[idx] = th[:len(idx[0])]; k = len(idx[0])
            Lm[np.diag_indices(W)] = np.exp(Lm[np.diag_indices(W)])
            Sig = Lm @ Lm.T
        elif kind == 'capacity':
            Sig = np.exp(2 * th[0]) * np.ones((W, W)); k = 1
        elif kind == 'capacity_diag':
            Sig = np.exp(2 * th[0]) * np.ones((W, W)) + np.diag(np.exp(2 * th[1:1 + W])); k = 1 + W
        elif kind.startswith('factor'):
            q = int(kind[6:]); Lam = th[:W * q].reshape(W, q)
            Sig = Lam @ Lam.T + np.diag(np.exp(2 * th[W * q:W * q + W])); k = W * q + W
        nv = self.W if self.hetero else 1
        s2 = np.exp(2 * th[k:k + nv]); rho = 1 / (1 + np.exp(-th[k + nv])); eta = 1 / (1 + np.exp(-th[k + nv + 1]))
        return Sig, s2, rho, eta

    def noise_cov(self, s2, rho, eta):
        s2 = np.atleast_1d(s2); sd = np.sqrt(s2[self.win] if len(s2) > 1 else np.full(self.T, s2[0]))
        return np.outer(sd, sd) * ar_matrix(self.frac, rho, eta)

    def nll(self, th, kind):
        return self.nll_grad(th, kind)[0]

    def nll_grad(self, th, kind):
        """Negative log-likelihood and its analytic gradient. With G = (Om^-1 S Om^-1 - N Om^-1) / 2,
        d nll / d theta = -sum(G * d Om / d theta)."""
        Sig, s2, rho, eta = self.unpack(th, kind); W = self.W
        s2 = np.atleast_1d(s2); sd = np.sqrt(s2[self.win] if len(s2) > 1 else np.full(self.T, s2[0]))
        d = np.abs(np.subtract.outer(self.frac, self.frac)); P = rho ** d; A = (1 - eta) * P + eta * np.eye(self.T)
        SS = np.outer(sd, sd); Vn = SS * A
        Om = self.Z @ Sig @ self.Z.T + Vn
        try:
            c = cho_factor(Om, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return 1e12, np.zeros_like(th)
        logdet = 2 * np.log(np.diag(c[0])).sum(); Oi = cho_solve(c, np.eye(self.T), check_finite=False); OiS = Oi @ self.S
        f = 0.5 * (self.N * logdet + np.trace(OiS) + self.N * self.T * np.log(2 * np.pi))
        G = 0.5 * (OiS @ Oi - self.N * Oi); G = (G + G.T) / 2
        GS = self.Z.T @ G @ self.Z                     # d f / d Sigma = -GS
        g = []
        if kind == 'full':
            Lm = np.zeros((W, W)); idx = np.tril_indices(W); Lm[idx] = th[:len(idx[0])]; Lm[np.diag_indices(W)] = np.exp(Lm[np.diag_indices(W)])
            gL = -2 * GS @ Lm; gL[np.diag_indices(W)] *= np.diag(Lm); g += list(gL[idx]); k = len(idx[0])
        elif kind == 'capacity':
            g.append(-np.sum(GS) * 2 * np.exp(2 * th[0])); k = 1
        elif kind.startswith('factor'):
            q = int(kind[6:]); Lam = th[:W * q].reshape(W, q)
            g += list((-2 * GS @ Lam).ravel()); g += list(-np.diag(GS) * 2 * np.exp(2 * th[W * q:W * q + W])); k = W * q + W
        else:
            g.append(-np.sum(GS) * 2 * np.exp(2 * th[0])); g += list(-np.diag(GS) * 2 * np.exp(2 * th[1:1 + W])); k = 1 + W
        GV = G * Vn
        if len(s2) > 1:
            rs = GV.sum(1); g += [-2 * rs[self.win == w].sum() for w in range(W)]
        else:
            g.append(-2 * GV.sum())
        dA_rho = (1 - eta) * d * np.where(d > 0, rho ** np.maximum(d - 1, 0), 0.0)
        g.append(-np.sum(G * SS * dA_rho) * rho * (1 - rho))
        g.append(-np.sum(G * SS * (np.eye(self.T) - P)) * eta * (1 - eta))
        return f, np.array(g)

    def start(self, kind):
        # moment start: donor window means
        M = self.R @ self.Z / self.Z.sum(0); C = np.cov(M.T) + 1e-4 * np.eye(self.W)
        resid = self.R - M @ self.Z.T
        s = [np.log(resid[:, self.win == w].std()) for w in range(self.W)] if self.hetero else [np.log(resid.std())]
        if kind == 'full':
            Lm = np.linalg.cholesky(C); Lm[np.diag_indices(self.W)] = np.log(np.diag(Lm)); th = list(Lm[np.tril_indices(self.W)])
        elif kind.startswith('factor'):
            q = int(kind[6:]); w_, U = np.linalg.eigh(C); o = np.argsort(w_)[::-1][:q]
            Lam = U[:, o] * np.sqrt(np.maximum(w_[o], 1e-6))
            th = list(Lam.ravel()) + list(0.5 * np.log(np.maximum(np.diag(C) - np.sum(Lam ** 2, 1), 1e-4)))
        elif kind == 'capacity':
            th = [0.5 * np.log(max(np.mean(C), 1e-4))]
        else:
            th = [0.5 * np.log(max(np.mean(C[np.triu_indices(self.W, 1)]), 1e-4))] + list(0.5 * np.log(np.maximum(np.diag(C) * 0.3, 1e-4)))
        return np.array(th + list(s) + [1.0, -1.0], float)

    def fit(self, kind='full', th0=None, maxiter=3000):
        th0 = self.start(kind) if th0 is None else th0
        r = minimize(self.nll_grad, th0, args=(kind,), jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'maxfun': 200000})
        Sig, s2, rho, eta = self.unpack(r.x, kind)
        return {'kind': kind, 'th': r.x, 'loglik': -r.fun, 'converged': r.success, 'Sigma': Sig, 's2': s2, 'rho': rho, 'eta': eta,
                'n_par': len(r.x)}

    # ---- donor-level estimates ----
    def donor_estimates(self, fit, R=None):
        R = self.R if R is None else np.asarray(R, float)
        V = self.noise_cov(fit['s2'], fit['rho'], fit['eta']); Vi = np.linalg.inv(V)
        A = self.Z.T @ Vi @ self.Z; Ainv = np.linalg.inv(A)
        gls = R @ Vi @ self.Z @ Ainv                         # N x W, unbiased, sampling cov = Ainv
        Om = self.Z @ fit['Sigma'] @ self.Z.T + V
        blup = R @ np.linalg.solve(Om, self.Z @ fit['Sigma'])  # N x W, shrunk
        return gls, blup, Ainv

    def simulate(self, fit, N=None, rng=None):
        rng = np.random.default_rng(rng); N = N or self.N
        V = self.noise_cov(fit['s2'], fit['rho'], fit['eta'])
        d = rng.multivariate_normal(np.zeros(self.W), fit['Sigma'] + 1e-12 * np.eye(self.W), size=N)
        e = rng.multivariate_normal(np.zeros(self.T), V, size=N)
        return d @ self.Z.T + e, d


def decompose(Sig):
    """Capacity versus stimulus-specific variance from a window covariance matrix."""
    W = Sig.shape[0]; one = np.ones(W) / W; P = np.eye(W) - np.outer(np.ones(W), one)
    cap = float(one @ Sig @ one); spec = P @ Sig @ P.T
    return {'capacity_var': cap, 'specific_var_per_window': np.diag(spec), 'specific_share': float(np.trace(spec) / np.trace(Sig)),
            'specific_cov': spec}


def contrast_var(Sig, a):
    a = np.asarray(a, float); return float(a @ Sig @ a)
