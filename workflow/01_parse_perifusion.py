"""Step 01 - parse HPAP islet perifusion time courses (PANC-DB) into long tables.

Inputs (data/raw/hpap/hpapdata/HPAP-xxx/Islet Studies/Islet physiology studies/...):
  Penn       Perifusions/HPAP-xxx_Perifusion_data.csv. 1-min fractions, 141 points (t = 20-160 min).
             Basal 10 min, amino acid mixture 4 mM 31 min, + glucose 3 mM 20 min, + glucose 16.7 mM 20 min,
             + IBMX 0.1 mM 20 min, no stimulus 20 min, KCl 30 mM 20 min. Insulin ng/100 islets/min,
             glucagon pg/100 islets/min, content per islet in the first row.
  Vanderbilt Islets to Vanderbilt/Islet perifusion at Vanderbilt/HPAP-xxx_Perifusion[-_]data.xlsx. 3-min fractions,
             50 fractions. 5.6 mM glucose basal, 16.7 mM glucose, washes in 5.6 mM glucose, 16.7 mM glucose + IBMX
             100 uM, 1.7 mM glucose + adrenaline 1 uM, 5.6 mM glucose + KCl 20 mM. Values per 100 IEQ/min and % content/min.
Censoring. The most frequent value of a run is treated as an assay floor when it occurs at least 5 times and at most 15%
of the run lies below it (e.g. 20 pg/mL for the Millipore glucagon RIA). Values at or below it are flagged as censored.
Penn values are reported on a coarse grid at low levels, which this rule does not mistake for a floor.
Outputs: data/processed/perifusion_long.tsv.gz, perifusion_runs.tsv (one row per donor x lab x hormone).
"""
import re, sys, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; RAW = ROOT / 'data/raw/hpap/hpapdata'; OUT = ROOT / 'data/processed'

PENN_EPOCHS = [('basal', 'No Stimuli'), ('AA', 'Amino acid Mixure, 4 mM'), ('AA+G3', 'Amino acid Mixure, 4 mM + Glucose, 3 mM'),
               ('AA+G16.7', 'Amino acid Mixure, 4 mM + Glucose, 16.7 mM'), ('AA+G16.7+IBMX', 'Amino acid Mixure, 4 mM + Glucose, 16.7 mM + IBMX, 0.1 mM'),
               ('wash', 'No Stimuli'), ('KCl', 'KCl, 30 mM')]
# Vanderbilt: epoch by fraction number (1-based), robust to label typos
VAN_EPOCHS = [('basal', 1, 3), ('G16.7', 4, 13), ('wash1', 14, 20), ('IBMX', 21, 23), ('wash2', 24, 30),
              ('Epi', 31, 33), ('wash3', 34, 40), ('KCl', 41, 43), ('wash4', 44, 50)]


def num(x):
    x = pd.Series(x).astype(str).str.strip().str.replace(',', '')
    lt = x.str.startswith('<')
    return pd.to_numeric(x.str.lstrip('<>'), errors='coerce'), lt


def flag_floor(conc, lt):
    """Assay floor: the most frequent value is treated as a floor when it occurs >= 5 times and <= 15% of the run lies
    below it (values reported at the lowest standard). Values at or below it are flagged as censored."""
    c = pd.Series(conc).astype(float); fl = pd.Series(lt).astype(bool).copy()
    x = c.dropna().round(6)
    if len(x):
        vc = x.value_counts(); m, k = vc.index[0], vc.iloc[0]
        if k >= 5 and (x < m - 1e-9).mean() <= 0.15:
            fl = fl | (c.round(6) <= m + 1e-9)
    return fl.values


def parse_penn(f):
    donor = Path(f).name.split('_')[0]
    d = pd.read_csv(f); d.columns = [c.strip() for c in d.columns]; d = d.dropna(subset=['Time (min)']).reset_index(drop=True)
    d['Stimulus'] = d['Stimulus'].astype(str).str.strip()
    blocks = [g.Stimulus.iloc[0] for _, g in d.groupby((d.Stimulus != d.Stimulus.shift()).cumsum())]
    canonical = blocks == [s for _, s in PENN_EPOCHS]
    ep = []
    if canonical:
        for k, (_, g) in enumerate(d.groupby((d.Stimulus != d.Stimulus.shift()).cumsum(), sort=True)):
            ep += [PENN_EPOCHS[k][0]] * len(g)
    else:
        ep = [None] * len(d)
    cont = {k: pd.to_numeric(d.iloc[0][[c for c in d.columns if c.startswith(k)][0]], errors='coerce') if any(c.startswith(k) for c in d.columns) else np.nan
            for k in ['DNA content', 'Insulin Content', 'Glucagon Content']}
    rows = []
    for hormone, pref in [('glucagon', 'Glucagon Release'), ('insulin', 'Insulin Release')]:
        v, lt = num(d[[c for c in d.columns if c.startswith(pref)][0]])
        rows.append(pd.DataFrame({'donor': donor, 'lab': 'Penn', 'hormone': hormone, 'fraction': np.arange(1, len(d) + 1),
                                  'time_min': d['Time (min)'].values, 'stimulus': d.Stimulus.values, 'epoch': ep,
                                  'value': v.values, 'value_pct_content': np.nan, 'conc': np.nan, 'censored': flag_floor(v, lt),
                                  'protocol_canonical': canonical, 'content': cont['Glucagon Content' if hormone == 'glucagon' else 'Insulin Content'],
                                  'dna_content': cont['DNA content'], 'unit': 'per 100 islets per min'}))
    return pd.concat(rows)


def read_van_sheet(x, sheet):
    d = x.parse(sheet, header=None)
    h = [i for i in range(min(12, len(d))) if d.iloc[i].astype(str).str.contains('Stimulus').any()][0]
    cols = [re.sub(r'\s+', ' ', str(c)).strip() for c in d.iloc[h]]; d = d.iloc[h + 1:].copy(); d.columns = cols
    d = d[pd.to_numeric(d['Fraction'], errors='coerce').notna()].copy()
    d['Fraction'] = d['Fraction'].astype(float).astype(int)
    return d


def parse_van(f):
    donor = Path(f).name.split('_')[0]
    x = pd.ExcelFile(f); rows = []
    for hormone in ['glucagon', 'insulin']:
        sh = [s for s in x.sheet_names if s.lower().startswith(hormone)]
        if not sh: continue
        d = read_van_sheet(x, sh[0])
        c_conc = [c for c in d.columns if '/mL' in c][0]
        c_ieq = [c for c in d.columns if 'IEQ' in c or '/100' in c][0]
        c_pct = [c for c in d.columns if '%' in c]
        conc, lt = num(d[c_conc]); v, _ = num(d[c_ieq]); pc = num(d[c_pct[0]])[0] if c_pct else pd.Series(np.nan, index=d.index)
        fr = d['Fraction'].values
        ep = np.array([next((e for e, a, b in VAN_EPOCHS if a <= k <= b), None) for k in fr], dtype=object)
        rows.append(pd.DataFrame({'donor': donor, 'lab': 'Vanderbilt', 'hormone': hormone, 'fraction': fr,
                                  'time_min': pd.to_numeric(d['Time (min)'], errors='coerce').values, 'stimulus': d['Stimulus'].astype(str).str.strip().values,
                                  'epoch': ep, 'value': v.values, 'value_pct_content': pc.values, 'conc': conc.values,
                                  'censored': flag_floor(conc, lt), 'protocol_canonical': len(d) == 50 and fr.max() == 50,
                                  'content': np.nan, 'dna_content': np.nan, 'unit': 'per 100 IEQ per min'}))
    return pd.concat(rows) if rows else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pf = sorted(glob.glob(str(RAW / '*/Islet Studies/Islet physiology studies/Perifusions/*_Perifusion_data.csv')))
    vf = sorted(glob.glob(str(RAW / '*/Islet Studies/Islet physiology studies/Islets to Vanderbilt/*/*.xlsx')))
    L = pd.concat([parse_penn(f) for f in pf] + [r for r in (parse_van(f) for f in vf) if r is not None], ignore_index=True)
    L.to_csv(OUT / 'perifusion_long.tsv.gz', sep='\t', index=False)
    runs = L.groupby(['donor', 'lab', 'hormone']).agg(n=('value', 'size'), n_missing=('value', lambda s: int(s.isna().sum())),
                                                     n_censored=('censored', 'sum'), canonical=('protocol_canonical', 'first'),
                                                     median=('value', 'median'), content=('content', 'first')).reset_index()
    runs['usable'] = runs.canonical & (runs.n_missing <= 0.1 * runs.n) & ((runs.n_censored / runs.n) <= 0.25) & (runs['median'] > 0)
    runs.to_csv(OUT / 'perifusion_runs.tsv', sep='\t', index=False)
    print(runs.groupby(['lab', 'hormone']).agg(runs=('donor', 'size'), canonical=('canonical', 'sum'), usable=('usable', 'sum'),
                                              median_censored=('n_censored', 'median')).to_string())
    both = set(runs[(runs.lab == 'Penn') & runs.usable & (runs.hormone == 'glucagon')].donor) & \
        set(runs[(runs.lab == 'Vanderbilt') & runs.usable & (runs.hormone == 'glucagon')].donor)
    print('donors with usable glucagon runs at both labs:', len(both))


if __name__ == '__main__':
    main()
