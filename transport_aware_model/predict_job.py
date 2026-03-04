import json
from math import sqrt
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from transport_aware_model.model import predict_graph_groups

RAW_JOBS_PATH = Path("data/raw/measurement_summary.json")
RAW_STAGES_PATH = Path("data/raw/stage_measurement_summary.json")

MTU_FILTER = 9001
ONPREM_BW_GBPS = 999.0


# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    err = y_pred_arr - y_true_arr

    mae = float(np.mean(np.abs(err)))
    rmse = float(sqrt(np.mean(err**2)))

    nonzero = y_true_arr != 0
    mape = (
        float(np.mean(np.abs(err[nonzero] / y_true_arr[nonzero])) * 100.0)
        if np.any(nonzero)
        else float("nan")
    )

    denom = float(np.sum(np.abs(y_true_arr)))
    wmape = float(np.sum(np.abs(err)) / denom * 100.0) if denom > 0 else float("nan")

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true_arr - float(np.mean(y_true_arr))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape, "WMAPE%": wmape, "R2": r2}


def print_summary_metrics(
    y_observed: list[float],
    y_transport_aware_grouping: list[float],
    y_transport_aware_cumulative: list[float],
    y_transport_agnostic_cumulative: list[float],
    label: str,
) -> None:
    m_grouping = compute_metrics(y_observed, y_transport_aware_grouping)
    m_ta_cumulative = compute_metrics(y_observed, y_transport_aware_cumulative)
    m_baseline = compute_metrics(y_observed, y_transport_agnostic_cumulative)

    print(f"\n=== Summary: {label} ===")
    print(f"Number of jobs: {len(y_observed)}")
    print(
        f"{'Metric':<8} "
        f"{'Transport-aware + dependency-aware grouping':>45} "
        f"{'Transport-aware + cumulative sum':>35} "
        f"{'Transport-agnostic + cumulative sum':>38} "
        f"{'Best':>12}"
    )

    for k in ["MAE", "RMSE", "MAPE%", "WMAPE%", "R2"]:
        a = m_grouping[k]
        b = m_ta_cumulative[k]
        c = m_baseline[k]

        # For R2: higher is better. For all others: lower is better.
        if k == "R2":
            best_val = max(a, b, c)
            if best_val == a:
                best = "Transport-aware + dependency-aware grouping"
            elif best_val == b:
                best = "Transport-aware + cumulative sum"
            else:
                best = "Transport-agnostic + cumulative sum"
        else:
            best_val = min(a, b, c)
            if best_val == a:
                best = "Transport-aware + dependency-aware grouping"
            elif best_val == b:
                best = "Transport-aware + cumulative sum"
            else:
                best = "Transport-agnostic + cumulative sum"

        print(
            f"{k:<8} "
            f"{a:45.4f} "
            f"{b:35.4f} "
            f"{c:38.4f} "
            f"{best:>12}"
        )


# =============================================================================
# Plotting
# =============================================================================
def plot_global_parity(y_observed: list[float], y_pred: list[float]) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    y_obs = np.asarray(y_observed, dtype=float)
    y_hat = np.asarray(y_pred, dtype=float)

    # log-scale safety
    mask = (y_obs > 0) & (y_hat > 0)
    y_obs = y_obs[mask]
    y_hat = y_hat[mask]

    if len(y_obs) == 0:
        print("No positive values available for parity plot.")
        return

    print(f"Observed values: min={float(np.min(y_obs)):.4f}, max={float(np.max(y_obs)):.4f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.scatter(y_obs, y_hat, s=12, alpha=0.25)

    lo = float(min(y_obs.min(), y_hat.min()))
    hi = float(max(y_obs.max(), y_hat.max()))

    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Observed job execution time [s]")
    ax.set_ylabel("Predicted job execution time [s]")
    ax.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    plt.show()


# =============================================================================
# Data helpers
# =============================================================================
def _get_one_hot_key(row: dict[str, Any], prefix: str) -> str | None:
    for k, v in row.items():
        if k.startswith(prefix) and v is not None and int(v) == 1:
            return k
    return None


def filter_on_prem_stages_for_job(
    job: dict[str, Any],
    all_onprem_stages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    query_id = job.get("query_id")
    job_comp_key = _get_one_hot_key(job, "compression_codec_")
    job_format_key = _get_one_hot_key(job, "file_format_")

    if query_id is None or job_comp_key is None or job_format_key is None:
        raise ValueError(
            "Job is missing query_id, or compression/file format one-hot fields."
        )

    return [
        s
        for s in all_onprem_stages
        if s.get("query_id") == query_id
        and _get_one_hot_key(s, "compression_codec_") == job_comp_key
        and _get_one_hot_key(s, "file_format_") == job_format_key
    ]


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    print("=== Verifying job-level predictions on all configurations ===")

    with open(RAW_JOBS_PATH, "r") as f:
        jobs: list[dict[str, Any]] = json.load(f)

    with open(RAW_STAGES_PATH, "r") as f:
        raw_stages: list[dict[str, Any]] = json.load(f)

    # Evaluation set: hybrid runs only (exclude on-prem baseline), fixed MTU
    eval_jobs = [
        j
        for j in jobs
        if int(j.get("mtu", -1)) == MTU_FILTER
        and float(j.get("tsg_max_bw_gbs", 0.0)) != ONPREM_BW_GBPS
    ]

    # Stage baseline: on-prem (used as historical execution signature)
    onprem_stages = [
        s for s in raw_stages if float(s.get("tsg_max_bw_gbs", 0.0)) == ONPREM_BW_GBPS
    ]

    if not eval_jobs or not onprem_stages:
        print(
            f"No data matched filters: mtu={MTU_FILTER}, exclude bw={ONPREM_BW_GBPS} for jobs, "
            f"and require bw={ONPREM_BW_GBPS} for baseline stages."
        )
        return

    # Grouped (by network regime)
    grouped: dict[tuple[float, float], dict[str, list[float]]] = {}

    # Global accumulators
    global_observed: list[float] = []
    global_ta_grouping: list[float] = []
    global_ta_cumulative: list[float] = []
    global_baseline_cumulative: list[float] = []

    for job in eval_jobs:
        baseline_stages_for_job = filter_on_prem_stages_for_job(job, onprem_stages)
        if not baseline_stages_for_job:
            # If there is no baseline signature for this job configuration, skip gracefully.
            continue

        # Returns scalars for a single job:
        # - observed job runtime
        # - transport-aware stage prediction + dependency-aware aggregation (grouping)
        # - transport-aware stage prediction + cumulative aggregation
        # - transport-agnostic baseline + cumulative aggregation
        (
            observed,
            ta_grouping,
            ta_cumulative,
            baseline_cumulative,
        ) = predict_graph_groups.predict_single_job(job, baseline_stages_for_job)

        bw = float(job.get("tsg_max_bw_gbs"))
        lat = float(job.get("network_latency_ms"))
        key = (bw, lat)

        if key not in grouped:
            grouped[key] = {
                "observed": [],
                "ta_grouping": [],
                "ta_cumulative": [],
                "baseline_cumulative": [],
            }

        grouped[key]["observed"].append(float(observed))
        grouped[key]["ta_grouping"].append(float(ta_grouping))
        grouped[key]["ta_cumulative"].append(float(ta_cumulative))
        grouped[key]["baseline_cumulative"].append(float(baseline_cumulative))

        global_observed.append(float(observed))
        global_ta_grouping.append(float(ta_grouping))
        global_ta_cumulative.append(float(ta_cumulative))
        global_baseline_cumulative.append(float(baseline_cumulative))

    if not global_observed:
        print("No jobs could be evaluated (missing baseline stage signatures).")
        return

    # Per-regime metrics (consistent with paper-style regime reporting)
    for (bw, latency), r in sorted(grouped.items(), key=lambda x: (-x[0][0], x[0][1])):
        label = f"bw={bw:g} Gbps, latency={latency:g} ms"
        print_summary_metrics(
            r["observed"],
            r["ta_grouping"],
            r["ta_cumulative"],
            r["baseline_cumulative"],
            label=label,
        )

    # Parity plot (paper figure style): transport-aware + dependency-aware grouping
    plot_global_parity(global_observed, global_ta_grouping)


if __name__ == "__main__":
    main()