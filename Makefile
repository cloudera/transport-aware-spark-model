# Makefile (submission-friendly, aligned entrypoints)
SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help prepare-data calibrate predict-stage predict-job repro all clean \
        lint format check

PYTHON ?= python
PACKAGE ?= transport_aware_model
LINT_PATHS ?= $(PACKAGE)

help:
	@echo "Targets:"
	@echo "  extract-data   Extract data from data.zip"
	@echo "  prepare-data   Build modeling dataset (feature engineering)"
	@echo "  calibrate      Calibrate transport coefficients (NNLS)"
	@echo "  predict-stage  Run stage-level predictions"
	@echo "  predict-job    Run job-level predictions"
	@echo "  repro          Full pipeline (prepare-data -> calibrate -> predict-stage -> predict-job)"
	@echo "  clean          Remove caches and build artifacts"

extract-data:
	unzip -o data.zip

prepare-data:
	@echo "Preparing data / features..."
	$(PYTHON) -m $(PACKAGE).data_management.prepare_data_job
	$(PYTHON) -m $(PACKAGE).data_management.prepare_data_stage

calibrate:
	@echo "Running system identification / calibration..."
	$(PYTHON) -m $(PACKAGE).calibrate

predict-stage:
	@echo "Running stage-level predictions..."
	$(PYTHON) -m $(PACKAGE).predict_stage

predict-job:
	@echo "Running job-level predictions..."
	$(PYTHON) -m $(PACKAGE).predict_job

repro: prepare-data calibrate predict-stage predict-job
all: repro

clean:
	@echo "Cleaning caches..."
	rm -rf .ruff_cache .pytest_cache
	find . -type d -name "__pycache__" -print0 | xargs -0 rm -rf
	rm -rf build dist *.egg-info

# Internal dev targets (hidden from help)
format:
	@echo "Formatting..."
	ruff check $(LINT_PATHS) --select I --fix
	ruff format $(LINT_PATHS)

lint:
	@echo "Linting..."
	ruff check $(LINT_PATHS)

check: format lint