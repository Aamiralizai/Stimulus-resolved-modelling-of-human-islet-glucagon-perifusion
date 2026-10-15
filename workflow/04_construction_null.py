"""Step 04 - construction-null validation: conventional summary traits versus the stimulus-resolved model.

Time courses are simulated on the Vanderbilt protocol (50 x 3-min fractions) from models fitted to the real HPAP
Vanderbilt glucagon data (population mean curve, window-specific AR(1) + nugget noise):
  A  capacity only            Sigma = tau2 * 11'            (no stimulus-specific biology)
  B  capacity + independent   tau2 * 11' + diag(psi)        (stimulus-specific variation, no shared stimulus axes)
  C  fitted full model        unstructured Sigma            (the structure estimated from real data)
  D  positive control         A + a glucose-suppression effect (true SD 0.20 on the 16.7 mM glucose window)
For each scenario, R replicates of N = 400 donors. Per replicate:
  conventional  9 HIPP-style traits (traits.vanderbilt_traits), inverse-normal transformed, parallel analysis,
                varimax loadings and their congruence with the axes found in the real PanKbase GCG-IEQ traits (v1.1)
  model         ML fits (capacity, capacity_diag, full), stimulus-specific share of variance, likelihood-ratio
                statistics, estimated SD of the glucose-suppression contrast.
Also checks that traits computed from the HPAP curves reproduce the PanKbase traits (Spearman rho per trait).
Outputs: results/tables/null_*.tsv
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from glucv2.model import WindowModel, decompose
from glucv2.data import load_matrix, contrast_vector
from glucv2.traits import vanderbilt_traits, parallel_analysis, axes, congruence, inverse_normal

OUT = ROOT / 'results/tables'
OLD = ROOT / 'data/raw/pankbase'   # PanKbase perifusion traits copied from the v1.1 package
PK_TRAITS = ['GCG-IEQ Basal Secretion', 'GCG-IEQ G 16.7 AUC', 'GCG-IEQ G 16.7 II', 'GCG-IEQ G 16.7+IMBX AUC', 'GCG-IEQ G 16.7+IMBX SI',
             'GCG-IEQ G 1.7+Epi AUC', 'GCG-IEQ G 1.7+Epi SI', 'GCG-IEQ KCl 20 AUC', 'GCG-IEQ KCl 20 SI']


def pankbase_axes():
    pk = pd.read_csv(OLD / 'perifusion_traits.tsv', sep='\t')[PK_TRAITS].apply(pd.to_numeric, errors='coerce').dropna()
    pk.columns = ['Basal', 'G16.7 AUC', 'G16.7 II', 'IBMX AUC', 'IBMX SI', 'Epi AUC', 'Epi SI', 'KCl AUC', 'KCl SI']
    Z = np.column_stack([inverse_normal(pk[c]) for c in pk.columns]); k, ev = parallel_analysis(Z)
    return axes(pk, k), k, len(pk), pk


def reproduce_check(pk_all):
    du = json.load(open(ROOT / 'data/raw/pankbase_donors_used.json'))
    amap = {x['accession']: a.split(':', 1)[1] for x in du for a in x.get('aliases', []) if 'HPAP' in a}
    raw = pd.read_csv(OLD / 'perifusion_traits.tsv', sep='\t'); raw['hpap'] = raw.donor_id.map(amap); raw = raw.dropna(subset=['hpap']).set_index('hpap')
    L = pd.read_csv(ROOT / 'data/processed/perifusion_long.tsv.gz', sep='\t'); v = L[(L.lab == 'Vanderbilt') & (L.hormone == 'glucagon')]
    Y = v.pivot_table(index='donor', columns='fraction', values='value'); Y = Y[Y.notna().all(1)]
    com = Y.index.intersection(raw.index); T = vanderbilt_traits(Y.loc[com].values); T.index = com
    rows = [{'trait': t, 'pankbase_column': p, 'n': len(com), 'spearman_rho': stats.spearmanr(T[t], pd.to_numeric(raw.loc[com, p], errors='coerce'), nan_policy='omit')[0]}
            for t, p in zip(T.columns, PK_TRAITS)]
    return pd.DataFrame(rows), T


def to_linear(m, mu, off, R):
    return np.maximum(np.exp(mu + R) - off, 1e-6)


def main(R=20, N=400, seed=11):
    rng = np.random.default_rng(seed)
    Lpk, k_pk, n_pk, _ = pankbase_axes()
    rep, Treal = reproduce_check(None); rep.to_csv(OUT / 'null_trait_reproduction.tsv', sep='\t', index=False); print(rep.round(3).to_string(index=False))
    D = load_matrix('Vanderbilt', 'glucagon'); m = WindowModel(D['R'], D['frac'], D['win'])
    fits = {k: m.fit(k) for k in ['capacity', 'capacity_diag', 'full']}
    gs = list(D['windows']).index('G16.7'); a_g = contrast_vector('Vanderbilt', 'Glucose suppression (5.6 to 16.7 mM)')
    posc = dict(fits['capacity']); S = posc['Sigma'].copy(); S[gs, gs] += 0.04; posc['Sigma'] = S
    scen = {'A capacity only': fits['capacity'], 'B capacity + independent window effects': fits['capacity_diag'],
            'C fitted full model': fits['full'], 'D positive control (glucose suppression SD 0.20)': posc}
    msim = WindowModel(np.zeros((N, m.T)), D['frac'], D['win'])
    # real-data conventional axes (HPAP Vanderbilt curves)
    Zr = np.column_stack([inverse_normal(Treal[c]) for c in Treal.columns]); k_real, _ = parallel_analysis(Zr)
    rows = []
    for name, f in scen.items():
        true_g = float(a_g @ f['Sigma'] @ a_g); true_share = decompose(f['Sigma'])['specific_share']
        for r in range(R):
            Rs, _ = msim.simulate(f, N=N, rng=rng.integers(1 << 31))
            Ylin = to_linear(msim, D['mu'], D['offset'], Rs)
            T = vanderbilt_traits(Ylin); T = T.replace([np.inf, -np.inf], np.nan).fillna(T.median())
            Z = np.column_stack([inverse_normal(T[c]) for c in T.columns]); k, ev = parallel_analysis(Z, n_iter=100, seed=r)
            Ls = axes(T, max(k, 1)); cong = congruence(Lpk, Ls) if k >= 1 else [np.nan] * Lpk.shape[1]
            LY = np.log(Ylin + D['offset']); Rc = LY - LY.mean(0); mm = WindowModel(Rc, D['frac'], D['win'])
            ff = {kk: mm.fit(kk) for kk in ['capacity', 'capacity_diag', 'full']}
            dec = decompose(ff['full']['Sigma'])
            rows.append({'scenario': name, 'rep': r, 'N': N, 'conv_axes_retained': k, 'conv_first_eigenvalue': ev[0],
                         **{f'congruence_pankbase_axis{i + 1}': c for i, c in enumerate(cong)},
                         'model_specific_share': dec['specific_share'], 'true_specific_share': true_share,
                         'LR_diag_vs_capacity': 2 * (ff['capacity_diag']['loglik'] - ff['capacity']['loglik']),
                         'LR_full_vs_diag': 2 * (ff['full']['loglik'] - ff['capacity_diag']['loglik']),
                         'glucose_suppression_SD_est': np.sqrt(max(a_g @ ff['full']['Sigma'] @ a_g, 0)), 'glucose_suppression_SD_true': np.sqrt(true_g)})
            print(name, r, k, np.round(cong, 2), round(dec['specific_share'], 3), flush=True)
    Rdf = pd.DataFrame(rows); Rdf.to_csv(OUT / 'null_simulations.tsv', sep='\t', index=False)
    summ = Rdf.groupby('scenario').agg(conv_axes_median=('conv_axes_retained', 'median'), conv_axes_min=('conv_axes_retained', 'min'),
                                       conv_axes_max=('conv_axes_retained', 'max'),
                                       **{f'congruence_axis{i + 1}_median': (f'congruence_pankbase_axis{i + 1}', 'median') for i in range(Lpk.shape[1])},
                                       model_specific_share_median=('model_specific_share', 'median'), true_specific_share=('true_specific_share', 'first'),
                                       LR_diag_vs_capacity_median=('LR_diag_vs_capacity', 'median'), LR_full_vs_diag_median=('LR_full_vs_diag', 'median'),
                                       gs_SD_est_median=('glucose_suppression_SD_est', 'median'), gs_SD_true=('glucose_suppression_SD_true', 'first')).reset_index()
    summ.to_csv(OUT / 'null_summary.tsv', sep='\t', index=False)
    pd.DataFrame(Lpk, index=['Basal', 'G16.7 AUC', 'G16.7 II', 'IBMX AUC', 'IBMX SI', 'Epi AUC', 'Epi SI', 'KCl AUC', 'KCl SI'],
                 columns=[f'axis{i + 1}' for i in range(Lpk.shape[1])]).to_csv(OUT / 'null_pankbase_axes.tsv', sep='\t')
    pd.DataFrame([{'pankbase_donors': n_pk, 'pankbase_axes': k_pk, 'hpap_vanderbilt_donors': len(Treal), 'hpap_axes': k_real}]).to_csv(OUT / 'null_real_axes.tsv', sep='\t', index=False)
    print(summ.round(3).to_string(index=False))


if __name__ == '__main__':
    main(*[int(x) for x in sys.argv[1:]])
