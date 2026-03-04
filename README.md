# Transport-Aware Performance Model

Reference implementation of a transport-aware submission-time performance predictor for big data workloads in hybrid cloud environments.

The framework:
- Builds stage-level modeling features from execution telemetry
- Calibrates transport coefficients using NNLS
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

## Pipeline

The modeling workflow follows the same sequence described in the paper: feature preparation, coefficient calibration, stage-level prediction, and job-level aggregation.

Run the complete pipeline:

``` bash
make repro
```

Or execute individual steps:

``` bash
make prepare-data
make calibrate
make predict-stage
make predict-job
```
