"""Input resolution for the figure scripts.

Two sources are supported.  A full working copy holds the donor-level
perifusion file and the full donor table, which are not redistributed under the
PANC-DB terms.  A public checkout holds aggregate equivalents that carry
everything the figures take from them.  Each loader prefers the full file and
falls back to the aggregate, so the same script runs in both places.
"""
from pathlib import Path
import pandas as pd

__all__ = ['find', 'have_long', 'donor_table', 'curve_summary', 'censoring_by_window',
           'censoring_per_donor', 'long_table']


def _p(root):
    return Path(root) / 'data/processed', Path(root) / 'data/derived'


def find(root, name):
    """Locate a data file in a working copy or in a public checkout.

    data/processed holds files rebuilt locally from the PANC-DB download.
    data/derived holds the files shipped with the repository.  The local copy
    wins when both are present.
    """
    full, agg = _p(root)
    for d in (full, agg):
        f = d / name
        if f.exists():
            return f
    raise FileNotFoundError(
        f'{name} was found in neither data/processed nor data/derived. '
        'See data/README.md for how to rebuild the local files from PANC-DB.')


def have_long(root):
    full, _ = _p(root)
    return (full / 'perifusion_long.tsv.gz').exists()


def long_table(root):
    """The donor-level perifusion file. Raises if this is a public checkout."""
    full, _ = _p(root)
    f = full / 'perifusion_long.tsv.gz'
    if not f.exists():
        raise FileNotFoundError(
            'data/processed/perifusion_long.tsv.gz is not in this checkout. It holds '
            'donor-level PANC-DB measurements, which are not redistributed. Rebuild it with '
            'workflow/01_parse_perifusion.py after downloading the files from PANC-DB, or use '
            'the aggregate loaders in glucv2.inputs.')
    return pd.read_csv(f, sep='\t')


def donor_table(root):
    """Donor covariates, indexed by donor."""
    full, agg = _p(root)
    f = full / 'donors.tsv'
    if f.exists():
        return pd.read_csv(f, sep='\t').set_index('donor')
    return pd.read_csv(find(root, 'donor_covariates.tsv'), sep='\t').set_index('donor')


def curve_summary(root, lab, hormone='glucagon'):
    """Group mean and SEM at each fraction. Columns group, fraction, time_min, mean, sem."""
    full, agg = _p(root)
    if have_long(root):
        L = long_table(root)
        D = donor_table(root)
        x = L[(L.lab == lab) & (L.hormone == hormone) & L.protocol_canonical.astype(bool)]
        tm = x.groupby('fraction').time_min.first()
        out = []
        for g in ['control', 'T2D', 'T1D']:
            s = x[x.donor.isin(D.index[D.group == g])]
            if s.empty:
                continue
            m, se = s.groupby('fraction').value.mean(), s.groupby('fraction').value.sem()
            out.append(pd.DataFrame({'group': g, 'fraction': m.index,
                                     'time_min': tm.reindex(m.index).values,
                                     'mean': m.values, 'sem': se.values}))
        return pd.concat(out, ignore_index=True)
    C = pd.read_csv(find(root, 'curve_summary.tsv'), sep='\t')
    return C[(C.lab == lab) & (C.hormone == hormone)].reset_index(drop=True)


def censoring_by_window(root, lab, hormone='glucagon'):
    """Percentage of donors with any censored fraction in each response window."""
    full, agg = _p(root)
    if have_long(root):
        from glucv2.data import WINDOWS
        L = long_table(root)
        x = L[(L.lab == lab) & (L.hormone == hormone) & L.protocol_canonical.astype(bool)]
        rows = []
        for nm, a, b in WINDOWS[lab]:
            s = x[(x.fraction >= a) & (x.fraction <= b)]
            if len(s):
                per = s.groupby('donor').censored.apply(lambda v: bool(v.astype(bool).any()))
                rows.append((lab, nm, 100 * per.mean()))
        return pd.DataFrame(rows, columns=['lab', 'window', 'pct_donors_any_censored'])
    C = pd.read_csv(find(root, 'censoring_by_window.tsv'), sep='\t')
    return C[C.lab == lab][['lab', 'window', 'pct_donors_any_censored']].reset_index(drop=True)


def censoring_per_donor(root, lab, hormone='glucagon'):
    """Percentage of fractions at the assay floor, one row per donor."""
    full, agg = _p(root)
    if have_long(root):
        L = long_table(root)
        x = L[(L.lab == lab) & (L.hormone == hormone) & L.protocol_canonical.astype(bool)]
        fr = x.groupby('donor').censored.mean() * 100
        return fr.rename('pct_fractions_censored').reset_index()
    C = pd.read_csv(find(root, 'censoring_per_donor.tsv'), sep='\t')
    return C[C.lab == lab][['donor', 'pct_fractions_censored']].reset_index(drop=True)
