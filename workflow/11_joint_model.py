"""Step 11 - joint two-laboratory latent model, covariance-structure sensitivity, and honest cross-laboratory
uncertainty (Gate 1.5).

1. Covariance sensitivity. The unstructured Sigma has W(W+1)/2 free parameters, which is generous at these sample
   sizes. Each laboratory and hormone is refitted with Sigma = Lambda Lambda' + diag(psi) using two factors and one
   factor, and with capacity plus a diagonal, and the quantities the conclusions rest on are compared.
2. Joint model. Instead of averaging z scores across laboratories, donor i's latent suppression phenotype u_i is
   modelled directly from whichever laboratories measured it:
       y_ij = alpha_j + lambda_j u_i + e_ij,     e_ij ~ N(0, s_ij^2 + tau_j^2)
       u_i  = x_i' beta + zeta_i,                zeta_i ~ N(0, 1)
   where s_ij is the known sampling standard error of the donor estimate from the window model and tau_j is
   laboratory-specific true variation. Donors measured at one laboratory contribute one term. The scale is fixed by
   var(zeta) = 1 and lambda_Penn > 0, so beta is in units of the residual SD of the latent phenotype.
3. Cross-laboratory correlation. The disattenuated correlation is bootstrapped by resampling donors and refitting both
   laboratory models inside each replicate, so the uncertainty in the true variances is carried through.
Outputs: results/tables/covariance_sensitivity.tsv, joint_latent_model.tsv, crosslab_bootstrap.tsv
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from glucv2.model import WindowModel, decompose
from glucv2.data import load_matrix, contrast_vector, CONTRASTS

OUT = ROOT / 'results/tables'
GS = {'Penn': 'Glucose suppression (3 to 16.7 mM)', 'Vanderbilt': 'Glucose suppression (5.6 to 16.7 mM)'}
KINDS = ['full', 'factor2', 'factor1', 'capacity_diag']


def covariance_sensitivity():
    rows = []
    for lab in ['Penn', 'Vanderbilt']:
        for hormone in ['glucagon', 'insulin']:
            D = load_matrix(lab, hormone); m = WindowModel(D['R'], D['frac'], D['win'])
            for kind in KINDS:
                f = m.fit(kind); dec = decompose(f['Sigma']); _, _, Ainv = m.donor_estimates(f)
                r = {'lab': lab, 'hormone': hormone, 'N': m.N, 'structure': kind, 'n_par': f['n_par'],
                     'loglik': f['loglik'], 'specific_share': dec['specific_share']}
                for n in CONTRASTS[lab]:
                    a = contrast_vector(lab, n); tv = float(a @ f['Sigma'] @ a); sv = float(a @ Ainv @ a)
                    r[f'{n}: true SD'] = np.sqrt(max(tv, 0)); r[f'{n}: reliability'] = tv / (tv + sv)
                rows.append(r); print(lab, hormone, kind, round(f['loglik'], 1), flush=True)
    T = pd.DataFrame(rows); T.to_csv(OUT / 'covariance_sensitivity.tsv', sep='\t', index=False)
    pd.set_option('display.width', 250)
    print(T[['lab', 'hormone', 'structure', 'n_par', 'loglik', 'specific_share']].round(4).to_string(index=False))
    return T


# ---------------------------------------------------------------- joint latent model
def joint_nll(th, Y, S, X, has):
    """Negative log-likelihood of the two-laboratory latent model. Y and S are N x 2 with zeros where absent.
    The scale is fixed by lambda_Penn = 1, so the latent phenotype and beta are in Penn log units."""
    p = X.shape[1]
    al = th[:2]; lam = np.array([1.0, th[2]]); su2 = np.exp(2 * th[3]); tau2 = np.exp(2 * th[4:6]); beta = th[6:6 + p]
    mu_u = X @ beta; nll = 0.0
    for i in range(len(Y)):
        j = has[i]
        if len(j) == 0: continue
        m = al[j] + lam[j] * mu_u[i]
        V = su2 * np.outer(lam[j], lam[j]) + np.diag(S[i, j] ** 2 + tau2[j])
        r = Y[i, j] - m
        _, ld = np.linalg.slogdet(V)
        nll += 0.5 * (ld + r @ np.linalg.solve(V, r))
    return nll


def joint_model():
    D0 = pd.read_csv(ROOT / 'data/processed/donors.tsv', sep='\t').set_index('donor')
    E = {lab: pd.read_csv(ROOT / f'data/processed/donor_effects_{lab}_glucagon.tsv', sep='\t', index_col=0) for lab in GS}
    don = sorted(set(E['Penn'].index) | set(E['Vanderbilt'].index))
    d = D0.loc[[x for x in don if x in D0.index]].copy(); d = d[d.group.isin(['control', 'T2D'])].copy()
    d['male'] = (d.sex == 'Male').astype(int); d['isolation_year'] = d.isolation_year.fillna(d.isolation_year.median())
    d = d.dropna(subset=['hba1c', 'age', 'bmi'])
    Y = np.full((len(d), 2), np.nan); S = np.full((len(d), 2), np.nan)
    for k, lab in enumerate(['Penn', 'Vanderbilt']):
        Y[:, k] = E[lab][f'gls_{GS[lab]}'].reindex(d.index).values
        S[:, k] = E[lab][f'se_{GS[lab]}'].reindex(d.index).values
    has = [np.flatnonzero(~np.isnan(Y[i])) for i in range(len(d))]
    cols = ['hba1c', 'age', 'male', 'bmi', 'isolation_year']
    Xdf = d[cols].astype(float); X = (Xdf - Xdf.mean()).values; p = X.shape[1]
    Yf = np.nan_to_num(Y); Sf = np.nan_to_num(S, nan=1.0)
    th0 = np.r_[np.nanmean(Y, 0), 0.5, np.log(max(np.nanstd(Y[:, 0]), 1e-2)), np.log([0.05, 0.05]), np.zeros(p)]
    r = minimize(joint_nll, th0, args=(Yf, Sf, X, has), method='L-BFGS-B', options={'maxiter': 20000, 'maxfun': 200000})
    H = np.zeros((len(r.x), len(r.x))); e = 1e-4
    for i in range(len(r.x)):
        for j in range(i, len(r.x)):
            ei = np.zeros(len(r.x)); ej = np.zeros(len(r.x)); ei[i] = e; ej[j] = e
            H[i, j] = H[j, i] = (joint_nll(r.x + ei + ej, Yf, Sf, X, has) - joint_nll(r.x + ei - ej, Yf, Sf, X, has)
                                 - joint_nll(r.x - ei + ej, Yf, Sf, X, has) + joint_nll(r.x - ei - ej, Yf, Sf, X, has)) / (4 * e * e)
    cov = np.linalg.pinv(H); se = np.sqrt(np.maximum(np.diag(cov), 0))
    names = ['alpha_Penn', 'alpha_Vanderbilt', 'lambda_Vanderbilt', 'log_sd_latent', 'log_tau_Penn', 'log_tau_Vanderbilt'] + cols
    rows = [{'parameter': n, 'estimate': v, 'se': s_, 'z': v / s_ if s_ > 0 else np.nan,
             'p': 2 * stats.norm.sf(abs(v / s_)) if s_ > 0 else np.nan} for n, v, s_ in zip(names, r.x, se)]
    sd_u = float(np.exp(r.x[3]))
    rows.append({'parameter': 'hba1c, in SD of the latent phenotype', 'estimate': r.x[6] / sd_u, 'se': se[6] / sd_u,
                 'z': r.x[6] / se[6] if se[6] > 0 else np.nan, 'p': 2 * stats.norm.sf(abs(r.x[6] / se[6])) if se[6] > 0 else np.nan})
    both = int(sum(len(h) == 2 for h in has)); one = int(sum(len(h) == 1 for h in has))
    T = pd.DataFrame(rows); T['n_donors'] = len(d); T['n_both_labs'] = both; T['n_one_lab'] = one
    T.to_csv(OUT / 'joint_latent_model.tsv', sep='\t', index=False)
    print(f'joint latent model: {len(d)} donors, {both} at both laboratories, {one} at one, latent SD {sd_u:.3f}')
    print(T.round(4).to_string(index=False))
    return T


# ---------------------------------------------------------------- bootstrap of the disattenuated correlation
def crosslab_bootstrap(B=200, seed=5):
    Dp, Dv = load_matrix('Penn', 'glucagon'), load_matrix('Vanderbilt', 'glucagon')
    ip, iv = pd.Index(Dp['donors']), pd.Index(Dv['donors']); union = sorted(set(ip) | set(iv))
    ap = contrast_vector('Penn', GS['Penn']); av = contrast_vector('Vanderbilt', GS['Vanderbilt'])
    rng = np.random.default_rng(seed); out = []

    def one(dp_idx, dv_idx, common_p, common_v):
        mp = WindowModel(Dp['R'][dp_idx], Dp['frac'], Dp['win']); fp = mp.fit('full')
        mv = WindowModel(Dv['R'][dv_idx], Dv['frac'], Dv['win']); fv = mv.fit('full')
        gp, _, _ = mp.donor_estimates(fp); gv, _, _ = mv.donor_estimates(fv)
        yp = (Dp['R'][common_p] @ np.linalg.solve(mp.noise_cov(fp['s2'], fp['rho'], fp['eta']), mp.Z)
              @ np.linalg.inv(mp.Z.T @ np.linalg.solve(mp.noise_cov(fp['s2'], fp['rho'], fp['eta']), mp.Z))) @ ap
        yv = (Dv['R'][common_v] @ np.linalg.solve(mv.noise_cov(fv['s2'], fv['rho'], fv['eta']), mv.Z)
              @ np.linalg.inv(mv.Z.T @ np.linalg.solve(mv.noise_cov(fv['s2'], fv['rho'], fv['eta']), mv.Z))) @ av
        tv = np.sqrt(max(ap @ fp['Sigma'] @ ap, 1e-12) * max(av @ fv['Sigma'] @ av, 1e-12))
        return np.corrcoef(yp, yv)[0, 1], np.cov(yp, yv)[0, 1] / tv

    pos_p = {d: k for k, d in enumerate(Dp['donors'])}; pos_v = {d: k for k, d in enumerate(Dv['donors'])}
    com = [d for d in union if d in pos_p and d in pos_v]
    r_obs, rt_obs = one(np.arange(len(ip)), np.arange(len(iv)), [pos_p[d] for d in com], [pos_v[d] for d in com])
    for b in range(B):
        s = rng.choice(union, len(union), replace=True)
        dp = [pos_p[d] for d in s if d in pos_p]; dv = [pos_v[d] for d in s if d in pos_v]
        cm = [d for d in s if d in pos_p and d in pos_v]
        if len(dp) < 30 or len(dv) < 30 or len(cm) < 20: continue
        try:
            r, rt = one(np.array(dp), np.array(dv), [pos_p[d] for d in cm], [pos_v[d] for d in cm])
            out.append({'rep': b, 'r_observed': r, 'r_true_effects': rt, 'n_common': len(cm)})
        except np.linalg.LinAlgError:
            continue
        if (b + 1) % 25 == 0: print('bootstrap', b + 1, flush=True)
    T = pd.DataFrame(out); T.to_csv(OUT / 'crosslab_bootstrap.tsv', sep='\t', index=False)
    q = lambda c: np.nanquantile(T[c], [0.025, 0.975])
    S = pd.DataFrame([{'quantity': 'observed correlation', 'estimate': r_obs, 'ci_low': q('r_observed')[0], 'ci_high': q('r_observed')[1]},
                      {'quantity': 'true-effect correlation', 'estimate': rt_obs, 'ci_low': q('r_true_effects')[0], 'ci_high': q('r_true_effects')[1]}])
    S['replicates'] = len(T); S.to_csv(OUT / 'crosslab_bootstrap_summary.tsv', sep='\t', index=False)
    print(S.round(3).to_string(index=False))


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('cov', 'all'): covariance_sensitivity()
    if what in ('joint', 'all'): joint_model()
    if what in ('boot', 'all'): crosslab_bootstrap(int(sys.argv[2]) if len(sys.argv) > 2 else 200)
