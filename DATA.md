# Dataset

This document describes the four JSON datasets that drive the transport-aware
performance model. After running `make extract-data`, they are found under `data/`:

```
data/
├── raw/measurement_summary.json          # Job-level measurements (raw)
├── raw/stage_measurement_summary.json    # Stage-level measurements (raw)
├── enriched_measurement_summary.json     # Job-level measurements + baseline/derived features
└── enriched_stage_measurement_summary.json  # Stage-level measurements + baseline/derived features
```

The datasets record executions of the TPC-DS benchmark at a scale factor of 1,000, which
corresponds to roughly 1 TB of raw generated data. That figure refers to the logical
input volume; the physical size actually stored and read in each run varies with
configuration, because the columnar storage format (ORC or Parquet) and the compression
codec (uncompressed, Snappy, or Gzip) change the on-disk footprint. The physical dataset
size for each configuration is recorded in the `data_size_gb` field (about 240 to 326 GB
across the five configurations here), while the per-run and per-stage volumes actually
read are recorded in the `spark_total_input_read_gb` and `input_read_gb` fields.
Executions are captured at two levels of granularity: the job (a complete query,
corresponding to a job dependency graph) and the execution stage (a set of homogeneous
parallel tasks separated by shuffle boundaries).

Each query template is executed across two columnar storage formats (ORC and Parquet),
three compression strategies (uncompressed, Snappy, and Gzip, the last for Parquet only),
and a range of hybrid interconnect conditions imposed by the Traffic Shaping Gateway
(TSG), which enforces a bandwidth cap and a round-trip time (RTT) on all inter-site
traffic.

Every network-constrained run is paired with an on-premises baseline run of the same
query template and storage configuration. The baseline is the transport-unconstrained
reference execution and is identified by the marker `tsg_max_bw_gbs` equal to `999.0`.
The model predicts
hybrid-environment execution time by scaling these on-premises baseline profiles
according to the modelled transport effects, so the baseline runs are the reference from
which cross-environment performance degradation is estimated.

Storage format and compression are encoded as mutually exclusive one-hot flags
(`file_format_parquet` / `file_format_orc`; `compression_codec_uncompressed` /
`compression_codec_snappy` / `compression_codec_gzip`).

---

## 1. raw/measurement_summary.json (job-level measurements)

A JSON array of one object per query execution (6,435 records). Each record aggregates
an entire job across all of its stages.

| Field | Description |
|---|---|
| `query_id` | TPC-DS query template identifier (e.g. `query6`). |
| `data_size_gb` | Physical size of the input dataset for this storage format and compression configuration, in GB. Derived from the TPC-DS scale factor 1,000 (about 1 TB raw), it varies across configurations (about 240 to 326 GB in this dataset). |
| `file_format_parquet`, `file_format_orc` | One-hot flags for the columnar storage format. |
| `compression_codec_uncompressed`, `compression_codec_snappy`, `compression_codec_gzip` | One-hot flags for the compression strategy. |
| `tsg_max_bw_gbs` | Bandwidth cap enforced by the Traffic Shaping Gateway, i.e. the physical bandwidth capacity of the hybrid interconnect, in Gb/s. `999.0` marks the transport-unconstrained on-premises baseline run. |
| `tsg_avg_bw_gbs` | Observed average throughput sustained over the hybrid interconnect during the run, in Gb/s. |
| `tsg_transfer_gb` | Total data volume transferred across the hybrid interconnect, in GB. |
| `network_latency_ms` | Round-trip time (RTT) of the emulated hybrid interconnect, in ms (observed values 0.301, 10.492, and 25.456). |
| `mtu` | Network maximum transmission unit, in bytes. |
| `exec_time_s` | Measured end-to-end job execution time, in seconds. |
| `num_spark_stages` | Number of execution stages in the job. |
| `spark_total_executor_runtime_s` | Aggregate executor runtime across the job, in seconds. |
| `spark_total_executor_cpu_time_s` | Aggregate executor CPU time across the job, in seconds. |
| `spark_total_input_read_gb` | Total input data read by the job, in GB. |
| `spark_total_shuffle_read_gb` | Total shuffle data read by the job, in GB. |
| `worker_avg_cpu` | Average worker-node CPU utilization during the job. |

---

## 2. raw/stage_measurement_summary.json (stage-level measurements)

A JSON array of one object per execution stage (70,134 records). Each record carries the
workload and hybrid-interconnect configuration together with per-stage telemetry and the
cluster's execution structure.

| Field | Paper notation | Description |
|---|---|---|
| `query_id` | | TPC-DS query template identifier of the parent job. |
| `stage_id`, `unique_stage_id` | `S_i` | Execution stage identifiers. |
| `total_stages` | | Number of stages in the parent job. |
| `match_id`, `match_category`, `match_quality` | | Keys and quality indicators used to pair a network-constrained stage with its on-premises baseline stage. |
| `debug_id`, `experiment_type` | | Experiment provenance / bookkeeping. |
| `stage_duration_s` | `y` | Measured stage execution time, in seconds. This is the observed hybrid stage time used as the calibration target, and the on-premises baseline stage time on baseline rows. |
| `num_tasks` | `N_tasks` | Number of tasks in the stage. |
| `avg_task_duration_s` | | Mean task duration in the stage, in seconds. |
| `executor_runtime_s` | | Executor runtime accumulated by the stage, in seconds. |
| `executor_cpu_time_s` | | Executor CPU time accumulated by the stage, in seconds. |
| `input_read_gb` | | Input data read by the stage, in GB. |
| `number_of_files_read` | | Number of files read by the stage. |
| `static_number_of_files_read` | | Number of files read as resolved during physical planning. |
| `number_of_partitions_read` | `N_partitions` | Number of input partitions processed by the stage. |
| `worker_avg_cpu` | | Average worker-node CPU utilization during the stage. |
| `spark_num_executors` | `N_executors` | Number of executors available to the application. |
| `spark_executor_cores` | `N_cores` | Number of cores per executor. |
| `spark_executor_memory_gb` | | Memory per executor, in GB. |
| `num_workers` | | Number of cluster worker nodes. |
| `worker_disk_bw_gbps` | | Per-worker disk bandwidth, in Gb/s. |
| `worker_net_bw_gbps` | | Per-worker network bandwidth, in Gb/s. |
| `total_worker_disk_bw_gbps` | | Aggregate cluster disk bandwidth, in Gb/s. |
| `total_worker_net_bw_gbps` | | Aggregate cluster network bandwidth, in Gb/s. |
| `window_max_bytes` | `W_max` | Maximum receiver-advertised / congestion window size, in bytes (the model derives `W_max` from this value). |
| `tsg_max_bw_gbs` | `B_link` | Bandwidth capacity of the hybrid interconnect enforced by the TSG, in Gb/s (`999.0` marks the on-premises baseline). |
| `network_latency_ms` | `RTT` | Round-trip time (RTT) of the hybrid interconnect, in ms. |
| `mtu` | `W_init` | Network maximum transmission unit, in bytes (the model derives the initial window `W_init` from this value). |
| `data_size_gb` | | Physical size of the input dataset for this storage format and compression configuration, in GB (varies across configurations; see Section 1). |
| `file_format_parquet`, `file_format_orc` | | One-hot storage-format flags. |
| `compression_codec_uncompressed`, `compression_codec_snappy`, `compression_codec_gzip` | | One-hot compression-strategy flags. |

---

## 3. enriched_measurement_summary.json (job-level, enriched)

Produced by `prepare_data_job.py` (6,435 records). This is the raw job dataset with each
run matched to its on-premises baseline run and augmented with baseline and derived
fields.

Baseline fields follow a `*_baseline` naming pattern. For a set of job metrics, the value
from the matched on-premises baseline run is copied onto the record under the same name
with a `_baseline` suffix, so `field_baseline` always means "the on-premises baseline
value of `field`." These are technical, denormalized fields rather than additional
measurements: each network-constrained run and its matched on-premises baseline are
flattened into a single record, so downstream code can compute cross-environment
degradation directly without a separate join back to the baseline run. The copied metrics
are: `exec_time_s`, `worker_avg_cpu`, `tsg_avg_bw_gbs`, `tsg_transfer_gb`,
`num_spark_stages`, `spark_total_executor_runtime_s`, `spark_total_executor_cpu_time_s`,
`spark_total_input_read_gb`, `spark_total_shuffle_read_gb`, and the derived fields listed
below.

Derived fields new to the enriched dataset:

| Field | Description |
|---|---|
| `estimated_query_avg_bw_gbps` | Average data demand rate of the job, computed as total input read divided by execution time (expressed in Gb/s). |
| `estimated_query_avg_bw_gbps_baseline` | The same demand rate computed for the on-premises baseline run. |
| `executor_cpu_utilization_ratio_baseline` | Baseline ratio of executor CPU time to executor runtime. |
| `executor_cpu_time_to_exec_time_ratio_baseline` | Baseline ratio of executor CPU time to job execution time. |
| `executor_cpu_runtime_to_exec_time_ratio_baseline` | Baseline ratio of executor runtime to job execution time. |

Runs are matched to their baseline by query template together with the storage
configuration and cluster sizing (storage-format and compression flags, worker count,
and the per-worker and aggregate disk and network bandwidths).

---

## 4. enriched_stage_measurement_summary.json (stage-level, enriched)

Produced by `prepare_data_stage.py`. Only network-constrained stages are retained
(64,733 records); the on-premises baseline stages are consumed during matching. Each
record contains the raw stage fields, baseline copies, engineered features, and the
stage model's prediction.

Baseline fields follow the same `*_baseline` naming pattern, so each `field_baseline`
holds the on-premises baseline stage's value of `field`. As in the job-level dataset,
these are technical convenience fields, not additional measurements: the
network-constrained stage and its matched on-premises baseline stage are stored in one
flat row so that degradation features can be computed without a separate lookup back to
the baseline stage. The copied metrics include `stage_id`, `stage_duration_s`,
`num_tasks`, `avg_task_duration_s`, `executor_runtime_s`, `executor_cpu_time_s`,
`input_read_gb`, `shuffle_read_gb`, `shuffle_write_gb`, `estimated_shuffle_read_gbps`,
`estimated_shuffle_write_gbps`, `worker_avg_cpu`, `number_of_files_read`,
`static_number_of_files_read`, and `number_of_partitions_read`.

Engineered features new to the enriched dataset:

| Field | Paper notation | Description |
|---|---|---|
| `estimated_stage_avg_bw_gbps` | | Average data demand rate of the network-constrained stage itself, computed as its own input read divided by its own stage duration (expressed in Gb/s). |
| `estimated_stage_avg_bw_gbps_baseline` | `b_base` | Baseline average data demand rate, computed from the on-premises baseline stage's input read divided by its duration (the paper's baseline demand rate). |
| `estimated_stage_avg_bw_to_max_bw_ratio_baseline` | `b_base / B_link` | Ratio of the baseline demand rate to the hybrid interconnect's bandwidth capacity. |
| `network_latency_s` | `RTT` | Round-trip time (RTT) of the hybrid interconnect, in seconds. |
| `task_waves` | `N_waves` | Number of sequential task execution waves in the stage, i.e. the task count divided by the available concurrency (executors times cores per executor), floored at 1. |
| `partition_access_per_task` | `ρ` (partition density) | The average number of input partitions processed per task. |
| `fold_id` | | Grouping key (`{query_id}_{stage_id}`) used to form cross-validation folds. |

Stage-model output new to the enriched dataset:

| Field | Paper notation | Description |
|---|---|---|
| `T_pred_total` | `T_hybrid` (predicted) | Predicted hybrid stage execution time from the transport-aware stage model. For stages whose baseline reads no remote input, this equals the on-premises baseline duration. |
| `T_bw` | `T_bw` | Predicted bandwidth-limited data transfer component of stage time, in seconds. |
| `T_storage` | `T_storage` | Predicted storage-access latency overhead component of stage time, in seconds. |
| `T_dynamic` | `T_dynamic` | Predicted transient TCP slow-start (ramp-up) overhead component of stage time, in seconds. |
| `T_bw_base`, `T_storage_base`, `T_dynamic_base` | `T_bw*`, `T_storage*`, `T_dynamic*` | The corresponding coefficient-free transport terms: the component values before scaling by the calibrated bandwidth, storage, and dynamic coefficients. These are the terms assembled into the predictor matrix during coefficient calibration. |

The three transport-component fields (`T_bw`, `T_storage`, `T_dynamic`) and their
coefficient-free counterparts are present only for stages whose on-premises baseline
reads remote input (`input_read_gb_baseline` greater than zero); for stages with no
remote data transfer the model returns the baseline duration and these fields are
omitted.
