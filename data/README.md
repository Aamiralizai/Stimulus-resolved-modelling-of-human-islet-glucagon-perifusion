# Data

## What ships with this repository

`data/derived/` holds everything the figures and the verification script need.

| File | What it is |
|---|---|
| `donor_effects_{lab}_{hormone}.tsv` | donor-specific deviations in every response window, from the stimulus-resolved model |
| `donor_effects_censored_{lab}_{hormone}.tsv` | the same, from the left-censored fit |
| `model_{lab}_{hormone}.json` | fitted covariance, window means and fit diagnostics |
| `model_censored_{lab}_{hormone}.json` | the same, from the left-censored fit |
| `donor_phenotypes.tsv` | one column per laboratory, hormone and contrast |
| `donor_bihormonal.tsv` | the two reciprocal coordination measures per donor |
| `donor_covariates.tsv` | donor age, sex, BMI, HbA1c in both units, diabetes group, isolation year, RRID |
| `curve_summary.tsv` | group mean and SEM at each perifusion fraction |
| `censoring_by_window.tsv` | percentage of donors with any censored fraction in each response window |
| `censoring_per_donor.tsv` | percentage of fractions at the assay floor, one row per donor |

`results/tables/` holds every table the manuscript and the supplementary
material draw on. `results/figures/` holds the rendered figures.

These are summary statistics and model estimates. They are released under
CC BY 4.0, see `LICENSE-DATA`.

## What does not ship, and why

The source measurements come from the Human Pancreas Analysis Program and are
used under the PANC-DB data use terms, which do not permit redistribution.
Three files are therefore absent from this repository.

| Absent file | What it holds |
|---|---|
| `data/raw/hpap/` | the PANC-DB downloads |
| `data/processed/perifusion_long.tsv.gz` | every perifusion fraction for every donor |
| `data/processed/perifusion_runs.tsv` | one row per run, with the quality control flags |
| `data/processed/donors.tsv` | the full PANC-DB donor table |

Every figure in the paper is still reproducible without them, because
`data/derived/` carries the aggregates the figure scripts take from these
files. `src/glucv2/inputs.py` prefers a local `data/processed` copy when one is
present and falls back to the shipped aggregate otherwise, so the same script
runs in both situations.

## Rebuilding the absent files

1. Register with PANC-DB at https://hpap.pmacs.upenn.edu and accept the data
   use terms.
2. Download the perifusion and oxygen-consumption data and the donor metadata.
3. Lay them out as follows.

```
data/raw/hpap/
  Donor_Summary_216.xlsx
  metadata/            PancDB_*.xlsx
  hpapdata/
    HPAP-001/          one folder per donor, both assays in the same tree
    HPAP-002/
    ...
```

4. Run `make all-from-raw`, or steps 01 and 02 alone to regenerate just the
   three absent files.

The donor identifiers used throughout are the PANC-DB identifiers, so every
derived row can be traced back to its source preparation.
