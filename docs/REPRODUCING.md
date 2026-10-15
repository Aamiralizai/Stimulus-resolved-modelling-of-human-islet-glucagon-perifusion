# Reproducing the results

Three levels, from quickest to most complete.

## 1. Check the numbers, seconds

```bash
pip install -r requirements.txt
PYTHONPATH=src python scripts/verify_results.py
```

Thirty quantities from the abstract, results and supplementary tables are read
from `results/tables/` and compared against the value printed in the paper. The
script prints both and exits non-zero on any disagreement. This catches a table
that has drifted from the manuscript. It does not rerun the analysis.

## 2. Redraw the figures, under a minute

```bash
make figures
```

Figures 2, 3 and 4, the graphical abstract and the study flow diagram are
redrawn from `results/tables/` and `data/derived/`. Output goes to
`results/figures/` in SVG, PDF and PNG.

Figures 1 and 5 are produced by `07_figures.py` and `14_oxygen_figure.py`,
which need the perifusion long table, so they belong to level 3.

## 3. Rerun everything, several hours

Needs a PANC-DB account and the download laid out as `data/README.md`
describes.

```bash
make all-from-raw
```

Expect the following, on one core.

| Step | Time |
|---|---|
| 01, 02 parsing | 2 to 3 min |
| 03 model fitting, each laboratory | about 15 min |
| 04 construction null | about 5 min |
| 09 validate at 100 replicates | 3 to 4 h |
| 11 bootstrap at 1000 replicates | about 40 min |
| everything else | under 10 min |

## Determinism

Every stochastic step takes an explicit seed, so a rerun on the same input
reproduces the published tables exactly. The numbers that are stochastic are
the construction null, the censored-likelihood validation, the likelihood ratio
calibration, the donor bootstrap and the bihormonal parametric bootstrap. If
you change a replicate count you will get different Monte Carlo error, which is
expected, and `scripts/verify_results.py` allows a tolerance on those
quantities.

The fitted model itself is deterministic given the input. Log likelihoods,
parameter counts and the clinical coefficients should match to the digits
printed in the verification output.

## Environment

The reported results were produced with Python 3.13 and the versions pinned in
`requirements.txt`. The continuous integration workflow runs the tests, the
verification and the figure scripts on every push, so the badge in the README
reflects whether this still holds on a clean machine.
