# Targets that need only what ships with the repository are marked "no raw data".
PY ?= python3
export PYTHONPATH := src

.PHONY: help setup test figures verify clean all-from-raw

help:
	@echo "make setup    install the pinned dependencies"
	@echo "make test     run the unit tests                        (no raw data)"
	@echo "make verify   check the headline numbers in the tables  (no raw data)"
	@echo "make figures  redraw every figure from the shipped data (no raw data)"
	@echo "make all-from-raw   rerun the whole pipeline, needs the PANC-DB download"

setup:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests/ -q

verify:
	$(PY) scripts/verify_results.py

figures:
	$(PY) workflow/22_figure_phenotype.py
	$(PY) workflow/23_figure_censoring.py
	$(PY) workflow/24_figure_crosslab.py
	$(PY) workflow/25_graphical_abstract_simple.py
	$(PY) workflow/26_figure_workflow_simple.py

all-from-raw:
	OMP_NUM_THREADS=1 $(PY) workflow/01_parse_perifusion.py
	OMP_NUM_THREADS=1 $(PY) workflow/02_donor_table.py
	OMP_NUM_THREADS=1 $(PY) workflow/03_fit_models.py Penn 100 200
	OMP_NUM_THREADS=1 $(PY) workflow/03_fit_models.py Vanderbilt 100 200
	OMP_NUM_THREADS=1 $(PY) workflow/04_construction_null.py 20 400
	OMP_NUM_THREADS=1 $(PY) workflow/04b_null_lrt_check.py 1000
	OMP_NUM_THREADS=1 $(PY) workflow/05_cross_lab.py
	OMP_NUM_THREADS=1 $(PY) workflow/06_clinical.py
	OMP_NUM_THREADS=1 $(PY) workflow/07_figures.py
	OMP_NUM_THREADS=1 $(PY) workflow/08_followup_checks.py
	OMP_NUM_THREADS=1 $(PY) workflow/09_censored_model.py validate 100
	OMP_NUM_THREADS=1 $(PY) workflow/09_censored_model.py real
	OMP_NUM_THREADS=1 $(PY) workflow/10_robustness.py
	OMP_NUM_THREADS=1 $(PY) workflow/11_joint_model.py cov
	OMP_NUM_THREADS=1 $(PY) workflow/11_joint_model.py joint
	OMP_NUM_THREADS=1 $(PY) workflow/11_joint_model.py boot 1000
	OMP_NUM_THREADS=1 $(PY) workflow/12_bihormonal.py 200
	OMP_NUM_THREADS=1 $(PY) workflow/13_oxygen.py
	OMP_NUM_THREADS=1 $(PY) workflow/14_oxygen_figure.py
	OMP_NUM_THREADS=1 $(PY) workflow/19_t1d_sensitivity.py
	$(MAKE) figures

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
