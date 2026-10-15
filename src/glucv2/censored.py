"""Left-censored fitting of the window model (assay-floor values).

Values at or below the assay floor are known only to be at most the floor. On the log scale donor i contributes
    l_it = mu_t + delta_{i,w(t)} + e_it,     observed exactly when not censored, else known to satisfy l_it <= c_it.
The likelihood is maximised by EM. The E-step completes each run with the moments of the latent values given the
observed values and the censoring constraints, which is a multivariate normal truncated from above on the censored
coordinates. The M-step is the ordinary complete-data fit of WindowModel on the completed scatter matrix, so the
same analytic gradient and optimiser are used.
    E-step   a_i = E[l_i | observed, censored <= c],  Cov_i = Var[l_i | ...]   (zero outside the censored block)
             mu = mean_i a_i,   S = sum_i (a_i - mu)(a_i - mu)' + sum_i Cov_i
    M-step   maximise the complete-data likelihood with that S
One censored value per run has closed-form truncated-normal moments. Runs with several censored values use a Gibbs
sampler over the censored block, each coordinate being univariate truncated normal given the others.
Uncertainty from censoring is carried into donor-level analyses by multiple imputation: draws of the censored values
from the same truncated conditional distribution give complete data sets, which are analysed separately and combined
with Rubin's rules.
"""
import numpy as np
from scipy import stats
from scipy.special import ndtr, ndtri
from scipy.linalg import cho_factor, cho_solve
from .model import WindowModel


def _trunc1(m, s, c):
    """Mean and variance of X ~ N(m, s^2) given X <= c."""
    a = (c - m) / s; lp = stats.norm.logcdf(a); r = np.exp(stats.norm.logpdf(a) - lp)
    mean = m - s * r; var = s * s * np.maximum(1 - a * r - r * r, 1e-10)
    return mean, var


def _gibbs(m, Sig, c, rng, n=200, burn=40):
    """Mean, covariance and draws of N(m, Sig) truncated to x <= c, by Gibbs sampling one coordinate at a time.
    Uses the ndtr and ndtri ufuncs rather than scipy.stats objects, which are much slower on scalars."""
    k = len(m); P = np.linalg.inv(Sig + 1e-10 * np.eye(k)); d = np.diag(P).copy()
    sd = 1 / np.sqrt(d); Pn = P / d[:, None]          # row j scaled so its diagonal entry is 1
    z = np.minimum(m, c) - 1e-3 - m; keep = np.empty((n, k)); U = rng.random((burn + n, k))
    for it in range(burn + n):
        for j in range(k):
            mj = m[j] - (Pn[j] @ z - z[j])
            a = ndtr((c[j] - mj) / sd[j])
            z[j] = mj + sd[j] * ndtri(min(max(U[it, j] * a, 1e-12), 1 - 1e-12)) - m[j]
        if it >= burn: keep[it - burn] = z + m
    return keep.mean(0), np.cov(keep.T).reshape(k, k), keep


class CensoredFit:
    """EM fit of the window model with left-censored values. L: N x T log-scale values, M: N x T censoring mask."""

    def __init__(self, L, M, thr, frac, win, seed=0):
        self.L = np.asarray(L, float); self.M = np.asarray(M, bool); self.thr = np.asarray(thr, float)
        self.frac = np.asarray(frac, float); self.win = np.asarray(win, int)
        self.N, self.T = self.L.shape; self.rng = np.random.default_rng(seed)
        self.rows = [np.flatnonzero(self.M[i]) for i in range(self.N)]

    def _conditional(self, Om, mu, i):
        """Conditional mean and covariance of the censored coordinates of run i given its observed values."""
        C = self.rows[i]; O = np.flatnonzero(~self.M[i])
        c = cho_factor(Om[np.ix_(O, O)], lower=True, check_finite=False)
        A = Om[np.ix_(C, O)]; sol = cho_solve(c, (self.L[i, O] - mu[O]), check_finite=False)
        m = mu[C] + A @ sol
        S = Om[np.ix_(C, C)] - A @ cho_solve(c, A.T, check_finite=False)
        return C, m, (S + S.T) / 2

    def estep(self, Om, mu, draws=0):
        """Completed first moments, summed conditional covariance, and optional imputation draws."""
        A = self.L.copy(); Csum = np.zeros((self.T, self.T)); imp = [] if draws else None
        if draws: imp = [self.L.copy() for _ in range(draws)]
        for i in range(self.N):
            if not len(self.rows[i]): continue
            C, m, S = self._conditional(Om, mu, i); c = self.thr[i, C]
            if len(C) == 1:
                mean, var = _trunc1(m[0], np.sqrt(S[0, 0]), c[0]); A[i, C] = mean; Csum[C[0], C[0]] += var
                if draws:
                    u = self.rng.random(draws); a = (c[0] - m[0]) / np.sqrt(S[0, 0])
                    d = m[0] + np.sqrt(S[0, 0]) * stats.norm.ppf(np.clip(u * stats.norm.cdf(a), 1e-12, 1 - 1e-12))
                    for k in range(draws): imp[k][i, C] = d[k]
            else:
                mean, cov, keep = _gibbs(m, S, c, self.rng)
                A[i, C] = mean; Csum[np.ix_(C, C)] += cov
                if draws:
                    idx = self.rng.choice(len(keep), draws, replace=False)
                    for k in range(draws): imp[k][i, C] = keep[idx[k]]
        return A, Csum, imp

    def fit(self, kind='full', iters=20, tail=8, verbose=False):
        """Monte Carlo EM. The Gibbs E-step makes the iteration sequence noisy rather than monotone, so the estimate is
        the average of the parameter vector over the last `tail` iterations, as is usual for Monte Carlo EM."""
        mu = self.L.mean(0); R = self.L - mu
        m = WindowModel(R, self.frac, self.win); f = m.fit(kind)
        th, mus, lls = [], [], []
        for it in range(iters):
            Om = m.Z @ f['Sigma'] @ m.Z.T + m.noise_cov(f['s2'], f['rho'], f['eta'])
            A, Csum, _ = self.estep(Om, mu)
            mu = A.mean(0); Rc = A - mu
            m = WindowModel(Rc, self.frac, self.win); m.S = Rc.T @ Rc + Csum
            f = m.fit(kind, th0=f['th'])
            th.append(f['th']); mus.append(mu); lls.append(f['loglik'])
            if verbose: print(f'  EM {it + 1} loglik {f["loglik"]:.2f}', flush=True)
        thbar = np.mean(th[-tail:], axis=0); mu = np.mean(mus[-tail:], axis=0)
        Sig, s2, rho, eta = m.unpack(thbar, kind)
        f = {'kind': kind, 'th': thbar, 'Sigma': Sig, 's2': s2, 'rho': rho, 'eta': eta, 'n_par': len(thbar),
             'loglik': float(np.mean(lls[-tail:])), 'loglik_sd_tail': float(np.std(lls[-tail:])), 'em_iters': iters,
             'converged': True}
        self.mu, self.model, self.fit_ = mu, m, f
        f['mu'] = mu
        return f

    def donor_estimates(self, kind='full'):
        """GLS donor window effects on the completed data (point estimate)."""
        f = self.fit_; Om = self.model.Z @ f['Sigma'] @ self.model.Z.T + self.model.noise_cov(f['s2'], f['rho'], f['eta'])
        A, _, _ = self.estep(Om, self.mu)
        gls, blup, Ainv = self.model.donor_estimates(f, R=A - self.mu)
        return gls, blup, Ainv

    def imputations(self, n=20):
        """n complete data sets with the censored values drawn from their truncated conditional distribution."""
        f = self.fit_; Om = self.model.Z @ f['Sigma'] @ self.model.Z.T + self.model.noise_cov(f['s2'], f['rho'], f['eta'])
        _, _, imp = self.estep(Om, self.mu, draws=n)
        out = []
        for Li in imp:
            mu = Li.mean(0); mm = WindowModel(Li - mu, self.frac, self.win)
            ff = mm.fit('full', th0=f['th'])
            gls, _, Ainv = mm.donor_estimates(ff)
            out.append({'gls': gls, 'Ainv': Ainv, 'Sigma': ff['Sigma']})
        return out


def rubin(est, var):
    """Combine multiple-imputation estimates. Returns the pooled estimate, standard error, t statistic, df and P."""
    est = np.asarray(est, float); var = np.asarray(var, float); m = len(est)
    qbar = est.mean(); ubar = var.mean(); b = est.var(ddof=1) if m > 1 else 0.0
    tvar = ubar + (1 + 1 / m) * b; se = np.sqrt(tvar)
    lam = (1 + 1 / m) * b / tvar if tvar > 0 else 0.0
    df = (m - 1) / max(lam ** 2, 1e-12) if lam > 0 else np.inf
    t = qbar / se if se > 0 else np.nan
    return {'estimate': qbar, 'se': se, 't': t, 'df': df, 'p': 2 * stats.t.sf(abs(t), df), 'fmi': lam}
