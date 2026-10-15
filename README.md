# Stimulus-resolved modelling of human islet glucagon perifusion

Analysis code and derived data for the article

> Stimulus-resolved modelling reveals weaker glucose suppression of glucagon
> with higher HbA1c in human islets. *Diabetologia*, 2026.

[![reproduce](https://github.com/REPLACE-WITH-OWNER/t2d-glucagon-perifusion/actions/workflows/ci.yml/badge.svg)](https://github.com/REPLACE-WITH-OWNER/t2d-glucagon-perifusion/actions/workflows/ci.yml)

Conventional perifusion summaries confound how much hormone an islet
preparation can release with how it responds to a particular stimulus. This
repository holds a hierarchical model that separates the two, fitted to human
islet perifusion curves from the Human Pancreas Analysis Program at two
laboratories, with a left-censored likelihood for the assay floor and matched
islet respirometry.

Headline result. Each 10 mmol/mol higher donor HbA1c is associated with
0.26 SD weaker glucose suppression of glucagon, 95% CI 0.17 to 0.34,
p = 2.9 × 10⁻⁹, in 112 donors at laboratory 1. The association replicates at a
second laboratory on islets from the same donors and survives every robustness
check in `results/tables/robustness_checks.tsv`. Whole-islet respiration does
not detectably account for it.

## Quick start

```bash
git clone https://github.com/REPLACE-WITH-OWNER/t2d-glucagon-perifusion.git
cd t2d-glucagon-perifusion
pip install -r requirements.txt

make test        # 10 unit tests on the model and the censored likelihood
make verify      # 30 checks of the reported numbers against the result tables
make figures     # redraw every figure from the shipped data
```

All three run from what the repository ships. No PANC-DB account is needed.

`make verify` is the quickest way to see that the numbers in the paper are the
numbers in the tables. It prints the manuscript value and the stored value side
by side for 30 quantities and exits non-zero if any disagree.

## What is here

```
src/glucv2/        model.py        hierarchical model, analytic gradients
                   censored.py     left-censored likelihood, MCEM, imputation
                   data.py         response windows, contrasts, loaders
                   traits.py       conventional perifusion traits, factor tools
                   inputs.py       resolves local against shipped data
                   plotting.py     figure style, editable SVG output
workflow/          numbered pipeline steps, run in order
tests/             pytest suite, 10 tests
scripts/           verify_results.py
data/derived/      donor-level estimates and the aggregates the figures need
data/README.md     what ships, what does not, and how to rebuild it
results/tables/    37 result tables
results/figures/   the figures, with a README mapping each one to its script
manuscript/        the cited reference library, RIS and BibTeX
```

Every SVG keeps axis labels and tick text as real text, so the figures stay
editable in Illustrator or Inkscape.

## The pipeline

Steps 01 and 02 need the PANC-DB download. Everything from 03 onwards runs from
what the repository ships.

| Step | What it does |
|---|---|
| `01_parse_perifusion.py` | read the PANC-DB files, apply the quality control rule, write the long table |
| `02_donor_table.py` | assemble the donor table from the PANC-DB metadata |
| `03_fit_models.py` | fit the stimulus-resolved model, one laboratory and hormone at a time |
| `04_construction_null.py` | simulate capacity-only curves and run the conventional factor analysis on them |
| `04b_null_lrt_check.py` | calibrate the likelihood ratio test against the null |
| `05_cross_lab.py` | matched, mismatched and cross-stimulus agreement between laboratories |
| `06_clinical.py` | associations with HbA1c and diabetes status |
| `07_figures.py` | the model and construction-null figure |
| `08_followup_checks.py` | confound checks |
| `09_censored_model.py` | the left-censored fit, and its validation on simulated data |
| `10_robustness.py` | specification, influence and subsample analyses |
| `11_joint_model.py` | covariance sensitivity, the joint latent model, the donor bootstrap |
| `12_bihormonal.py` | reciprocal coordination of glucagon and insulin |
| `13_oxygen.py` | respirometry feature extraction and quality control |
| `14_oxygen_figure.py` | the respirometry figure |
| `19_t1d_sensitivity.py` | refit without donors with type 1 diabetes |
| `22` to `26` | the published figures, the graphical abstract and the study flow diagram |

Steps 03 and 04 take about 15 and 5 minutes on one core. Step 09 at 100
validation replicates is the long one, a few hours. Steps 22 to 26 read only
the tables and the derived estimates, so they finish in seconds.

Set `OMP_NUM_THREADS=1` before the fitting steps. The optimiser is
single-threaded and BLAS threading makes it slower, not faster.

## Method, in short

Each perifusion run is described on the log scale by a donor-specific deviation
in every pre-specified response window. Overall capacity is the mean of those
deviations. Each stimulus response is a different contrast of the same random
effect, which is why a conventional factor solution recovers one dominant axis
whether or not stimulus-specific structure exists. A capacity-only simulation
reproduces the conventional first axis at a Tucker congruence of 0.91 and fails
to reproduce the second, which is the construction null the paper is built
against.

Values at the assay floor are not measurements. They are statements that the
concentration lies below a threshold, so they enter a left-censored likelihood
fitted by Monte Carlo expectation maximisation, with 20 multiple imputations
pooled by Rubin's rules.

Full detail is in the supplementary methods of the article.

## Data

Summary statistics and model estimates ship here under CC BY 4.0. The source
measurements come from PANC-DB and are not redistributed. `data/README.md` sets
out exactly what is absent and how to rebuild it.

## Citing

See `CITATION.cff`. Please cite both the article and the Human Pancreas
Analysis Program, which produced the underlying data.

## Licence

Code under MIT, see `LICENSE`. Derived data, tables and figures under
CC BY 4.0, see `LICENSE-DATA`.
