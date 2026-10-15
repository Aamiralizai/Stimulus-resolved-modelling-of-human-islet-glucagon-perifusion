# Data provenance

## Source

All primary measurements come from the Human Pancreas Analysis Program, served
by PANC-DB. Two assays are used.

- Dynamic perifusion of isolated islets, glucagon and insulin, at two
  laboratories with different stimulus protocols.
- Oxygen consumption of islets from the same preparations.

Donor metadata comes from the PANC-DB donor summary and the PANC-DB metadata
workbooks. Donor identifiers used here are the PANC-DB identifiers, so every
derived row traces back to its source preparation.

## From source to shipped files

```
PANC-DB download
   |  01_parse_perifusion.py   quality control, window assignment, censoring flags
   v
perifusion_long.tsv.gz, perifusion_runs.tsv          not redistributed
   |  02_donor_table.py        donor metadata assembly
   v
donors.tsv                                           not redistributed
   |  03_fit_models.py, 09_censored_model.py
   v
donor_effects_*.tsv, model_*.json                    shipped
   |  05, 06, 10, 11, 12, 13, 19
   v
results/tables/*.tsv                                 shipped
```

The three aggregates in `data/derived/` named `curve_summary.tsv`,
`censoring_by_window.tsv` and `censoring_per_donor.tsv` are computed directly
from `perifusion_long.tsv.gz` and hold exactly what the figure scripts take
from it. Nothing else in the figures touches the measurements.

## Quality control

A run is usable when all four hold.

1. It follows the standard stimulus schedule for its laboratory.
2. No more than one tenth of its fractions are missing.
3. No more than one quarter of its fractions sit at the assay floor.
4. Its median is positive.

Isolated missing fractions are interpolated within a response window and never
across a window boundary. A run with more than three missing fractions is
dropped. The attrition is as follows.

| | deposited | standard schedule | pass quality control | enter the model |
|---|---|---|---|---|
| Laboratory 1 glucagon | 161 | 158 | 134 | 130 |
| Laboratory 1 insulin | 161 | 158 | 153 | 153 |
| Laboratory 2 glucagon | 97 | 97 | 92 | 91 |
| Laboratory 2 insulin | 95 | 95 | 92 | 91 |

Respirometry: 96 deposited traces, 7 excluded at quality control, 89 usable.
Per-trace detail is in `results/tables/oxygen_qc.tsv`.

## Donor-level fields released here

`data/derived/donor_covariates.tsv` carries donor identifier and RRID, age,
sex, BMI, HbA1c in percent and in mmol/mol, diabetes group and clinical
diagnosis, isolation year, and which assays each donor contributed. These are
the fields the Diabetologia human islets checklist requires a paper to report,
so they are published with the article in any case.

Cause of death, ischaemia times, viability and the glucose-stimulated insulin
release index are reported in the human islets checklist submitted with the
article rather than here, because they play no part in the analysis.

## Terms

PANC-DB data are used under the Human Pancreas Analysis Program data use
agreement. Source files are not redistributed. Derived estimates and summary
statistics are released under CC BY 4.0.
