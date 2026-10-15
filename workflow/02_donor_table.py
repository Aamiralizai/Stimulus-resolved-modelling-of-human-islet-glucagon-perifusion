"""Step 02 - donor table: HPAP clinical and islet-preparation covariates, PanKbase identifiers, single-cell availability.

Group from PANC-DB clinical_diagnosis: 'control' (T1D control, T2D control), 'T2D' (any T2DM entry), 'T1D' (any T1DM entry).
Composition: endocrine fractions from the HPAP alpha, beta and delta cell counts.
Single-cell: untreated alpha, beta and delta cell numbers per donor in PanKbase scRNA v3.3 (from the earlier package).
Output: data/processed/donors.tsv
"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PK = ROOT / 'data/raw/pankbase'   # PanKbase tables copied from the v1.1 package


def hours(s):
    m = re.match(r'(\d+) days (\d+):(\d+):(\d+)', str(s))
    return int(m.group(1)) * 24 + int(m.group(2)) + int(m.group(3)) / 60 if m else np.nan


def main():
    d = pd.read_excel(ROOT / 'data/raw/hpap/Donor_Summary_216.xlsx')
    diag = d.clinical_diagnosis.astype(str)
    grp = np.where(diag.str.contains('control', case=False), 'control', np.where(diag.str.startswith('T2D'), 'T2D',
                   np.where(diag.str.contains('T1D', case=False), 'T1D', 'other')))
    T = pd.DataFrame({'donor': d.donor_ID, 'rrid': d.rrid, 'group': grp, 'diagnosis': diag,
                      'age': pd.to_numeric(d.age_years, errors='coerce'), 'sex': d.sex, 'bmi': pd.to_numeric(d.bmi, errors='coerce'),
                      'hba1c': pd.to_numeric(d.hba1c, errors='coerce'), 'race': d.race,
                      'viability': pd.to_numeric(d.islet_viability_percent, errors='coerce'),
                      'cold_ischemia_h': d.cold_ischemia_time_DDHHMMSS.map(hours),
                      'isolation_year': pd.to_datetime(d.biopsy_islet_isolation_datetime_est, errors='coerce').dt.year + pd.to_datetime(d.biopsy_islet_isolation_datetime_est, errors='coerce').dt.dayofyear / 365.25,
                      'c_peptide': pd.to_numeric(d.c_peptide_ng_ml, errors='coerce'), 'gsir_si': pd.to_numeric(d.gsir_si, errors='coerce')})
    # islet autoantibodies (GADA, IA-2, IAA, ZnT8) and availability of the donor-summary polygenic score (prs_score, disease not stated, likely type 1 diabetes)
    aab = d[['aab_gada', 'aab_ia_2', 'aab_iaa', 'aab_znt8']].apply(pd.to_numeric, errors='coerce')
    T['aab_n'] = aab.sum(1, min_count=1); T['aab_any'] = (T.aab_n > 0).astype(float).where(T.aab_n.notna())
    T['prs_available'] = pd.to_numeric(d.prs_score, errors='coerce').notna()
    cells = d[['alpha_cell_count', 'beta_cell_count', 'delta_cell_count']].apply(pd.to_numeric, errors='coerce')
    tot = cells.sum(1).replace(0, np.nan)
    T['alpha_frac_hist'] = cells.alpha_cell_count / tot; T['beta_frac_hist'] = cells.beta_cell_count / tot; T['delta_frac_hist'] = cells.delta_cell_count / tot
    # PanKbase identifiers and single-cell availability
    duf = ROOT / 'data/raw/pankbase_donors_used.json'
    if duf.exists():
        du = json.load(open(duf))
        amap = {a.split(':', 1)[1]: x['accession'] for x in du for a in x.get('aliases', []) if 'HPAP' in a}
        T['pankbase_id'] = T.donor.map(amap)
    else:
        # PanKbase tables ship in the separate data package.  Their columns are
        # annotation only and enter no association reported here.
        print('note: PanKbase donor file absent, pankbase_id and single-cell counts left empty')
        T['pankbase_id'] = np.nan
    sc = PK / 'scrna_samples.tsv'
    if sc.exists():
        S = pd.read_csv(sc, sep='\t'); u = S[S.untreated].groupby('donor_accession')[['Alpha', 'Beta', 'Delta']].sum()
        for c in ['Alpha', 'Beta', 'Delta']: T[f'sc_{c.lower()}_cells'] = T.pankbase_id.map(u[c]).fillna(0).astype(int)
    R = pd.read_csv(ROOT / 'data/processed/perifusion_runs.tsv', sep='\t')
    for lab in ['Penn', 'Vanderbilt']:
        for h in ['glucagon', 'insulin']:
            ok = set(R[(R.lab == lab) & (R.hormone == h) & R.usable].donor); T[f'{lab.lower()}_{h}'] = T.donor.isin(ok)
    T.to_csv(ROOT / 'data/processed/donors.tsv', sep='\t', index=False)
    for lab in ['penn', 'vanderbilt']:
        t = T[T[f'{lab}_glucagon']]
        nsc = int((t.sc_alpha_cells >= 30).sum()) if 'sc_alpha_cells' in t else -1
        print(lab, len(t), t.group.value_counts().to_dict(), 'HbA1c', int(t.hba1c.notna().sum()), 'sc alpha>=30', nsc)
    both = T[T.penn_glucagon & T.vanderbilt_glucagon]; print('both labs', len(both), both.group.value_counts().to_dict())


if __name__ == '__main__':
    main()
