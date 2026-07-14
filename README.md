# Transport-Aware Performance Model and Dataset

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21353271.svg)](https://doi.org/10.5281/zenodo.21353271)

Reference implementation of a transport-aware submission-time performance predictor for big data workloads in hybrid cloud environments.

Accompanies the IEEE Access paper: [Transport-Aware Performance Modeling for Big Data Workloads in Network-Constrained Hybrid Cloud Systems](https://ieeexplore.ieee.org/document/11569730). If you use this model or dataset, please [cite the paper](#how-to-cite).

The framework:
- Builds stage-level modeling features from execution telemetry
- Calibrates transport coefficients using NNLS, with workload-level cross-validation and a collinearity (VIF) diagnostic
- Runs one-at-a-time (OAT) sensitivity analysis of prediction quality to individual coefficient errors
- Predicts stage-level execution time
- Aggregates stage predictions into job-level runtime estimates

## Installation
Clone the repository and create an isolated Python environment. The project has no external runtime dependencies beyond the packages listed in requirements.txt.

```bash
git clone https://github.com/cloudera/transport-aware-spark-model.git
cd transport-aware-spark-model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Data Setup

The experimental dataset is provided as a compressed archive: [data.zip](data.zip)

### Extract the dataset
To reproduce the results, the archive must be extracted into the project root directory so that the `data/` directory is created.

```bash
make extract-data
```
After extraction, the directory structure should look like:

```bash
transport-aware-spark-model/
│
├── data/
│   ├── graph/
│   ├── raw/
│   ├── system_identification/
│   ├── enriched_measurement_summary.json
│   └── enriched_stage_measurement_summary.json
│
├── transport_aware_model/
├── Makefile
└── ...
```

See [DATA.md](DATA.md) for a field-by-field description of the dataset JSON files, including how each field maps to the paper's terminology.

## Pipeline

The modeling workflow follows the same sequence described in the paper: feature preparation, coefficient calibration, sensitivity analysis, stage-level prediction, and job-level aggregation.

Run the complete pipeline:

``` bash
make repro
```

Or execute individual steps:

``` bash
make prepare-data
make calibrate
make sensitivity
make predict-stage
make predict-job
```

### Step details

| Step | Description |
|---|---|
| `prepare-data` | Builds stage-level features from raw execution telemetry |
| `calibrate` | Fits NNLS transport coefficients; reports bootstrap confidence intervals, workload-level 80/20 cross-validation, and a VIF collinearity diagnostic |
| `sensitivity` | OAT sensitivity analysis: perturbs each coefficient across its 95% CI bound while holding the others fixed; reports MAE / RMSE / WMAPE / R² per network configuration |
| `predict-stage` | Applies calibrated coefficients to produce stage-level runtime predictions |
| `predict-job` | Aggregates stage predictions into job-level runtime estimates |

## How to cite

If you use this model or the accompanying dataset in your research, please cite the paper:

> A. Kanto and A. C. Marosi, "Transport-Aware Performance Modeling for Big Data Workloads in Network-Constrained Hybrid Cloud Systems," *IEEE Access*, vol. 14, pp. 92831–92855, 2026, doi: [10.1109/ACCESS.2026.3704628](https://doi.org/10.1109/ACCESS.2026.3704628).

```bibtex
@article{Kanto2026TransportAware,
  author  = {Kanto, Attila and Marosi, Attila Csaba},
  title   = {Transport-Aware Performance Modeling for Big Data Workloads in Network-Constrained Hybrid Cloud Systems},
  journal = {IEEE Access},
  year    = {2026},
  month   = jun,
  volume  = {14},
  pages   = {92831--92855},
  doi     = {10.1109/ACCESS.2026.3704628},
  issn    = {2169-3536},
  url     = {https://ieeexplore.ieee.org/document/11569730},
}
```
