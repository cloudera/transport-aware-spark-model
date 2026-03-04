# prepare_data_stage.py
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import transport_aware_model.model.stage_model as stage_model


# ----------------------------------------------------------------------
# IO / Configuration
# ----------------------------------------------------------------------

BASELINE_BANDWIDTH_GBPS = 999.0  # on-prem marker
BASELINE_LATENCY_MS = 1          # on-prem marker (as used in your dataset)

INPUT_FILE = Path("data/raw/stage_measurement_summary.json")
OUTPUT_FILE = Path("data/enriched_stage_measurement_summary.json")


# Which baseline fields to copy from the baseline row
BASELINE_FIELDS: Dict[str, str] = {
    "stage_id": "stage_id_baseline",
    "stage_duration_s": "stage_duration_s_baseline",
    "num_tasks": "num_tasks_baseline",
    "avg_task_duration_s": "avg_task_duration_s_baseline",
    "executor_runtime_s": "executor_runtime_s_baseline",
    "executor_cpu_time_s": "executor_cpu_time_s_baseline",
    "input_read_gb": "input_read_gb_baseline",
    "estimated_shuffle_read_gbps": "estimated_shuffle_read_gbps_baseline",
    "shuffle_read_gb": "shuffle_read_gb_baseline",
    "estimated_shuffle_write_gbps": "estimated_shuffle_write_gbps_baseline",
    "shuffle_write_gb": "shuffle_write_gb_baseline",
    "worker_avg_cpu": "worker_avg_cpu_baseline",
    "number_of_files_read": "number_of_files_read_baseline",
    "static_number_of_files_read": "static_number_of_files_read_baseline",
    "number_of_partitions_read": "number_of_partitions_read_baseline",
}


# In your current flow you enrich by match_id (best choice if it is stable).
MATCH_ID_KEYS: List[str] = ["match_id"]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if isinstance(b, float) and math.isclose(b, 0.0):
        return None
    if b == 0:
        return None
    return a / b


def make_key(row: Mapping[str, Any], keys: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(row.get(k) for k in keys)


def build_baseline_lookup(rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    """
    Build baseline lookup by (match_id) -> baseline row.
    Baseline rows are those marked as (tsg_max_bw_gbs == 999) and (network_latency_ms == 1).
    """
    lookup: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    duplicates = 0

    for r in rows:
        if r.get("tsg_max_bw_gbs") != BASELINE_BANDWIDTH_GBPS:
            continue
        if r.get("network_latency_ms") != BASELINE_LATENCY_MS:
            continue

        k = make_key(r, MATCH_ID_KEYS)
        if k in lookup:
            duplicates += 1
            # keep first occurrence, but fail hard to avoid silent skew
            raise ValueError(f"Duplicate baseline match_id key: {k}")
        lookup[k] = r

    print(f"Identified {len(lookup)} baseline stage rows (duplicates={duplicates})")
    return lookup


def engineer_features(rows: List[Dict[str, Any]]) -> None:
    """
    Adds derived stage-level features in-place.
    """
    for r in rows:
        r["estimated_stage_avg_bw_gbps"] = None

        input_read_gb = r.get("input_read_gb")
        dur_s = r.get("stage_duration_s")
        if input_read_gb is None or dur_s is None:
            continue

        if dur_s > 0:
            r["estimated_stage_avg_bw_gbps"] = (input_read_gb * 8.0) / dur_s
        else:
            r["estimated_stage_avg_bw_gbps"] = 0.0


def enrich_row(hybrid_row: Dict[str, Any], baseline_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a new enriched row (does not mutate input rows).
    """
    out = hybrid_row.copy()

    # Copy baseline metrics
    for src, dst in BASELINE_FIELDS.items():
        out[dst] = baseline_row.get(src)

    # Engineer baseline feature(s) that you want to reference later
    baseline_tmp = baseline_row.copy()
    # minimal baseline feature computation
    input_read_gb = baseline_tmp.get("input_read_gb")
    dur_s = baseline_tmp.get("stage_duration_s")
    baseline_tmp["estimated_stage_avg_bw_gbps"] = (
        (input_read_gb * 8.0) / dur_s if (input_read_gb is not None and dur_s and dur_s > 0) else None
    )

    out["estimated_stage_avg_bw_gbps_baseline"] = baseline_tmp.get("estimated_stage_avg_bw_gbps")
    out["estimated_stage_avg_bw_to_max_bw_ratio_baseline"] = safe_div(
        out.get("estimated_stage_avg_bw_gbps_baseline"),
        out.get("tsg_max_bw_gbs"),
    )

    # Common convenience fields
    latency_ms = out.get("network_latency_ms")
    out["network_latency_s"] = (latency_ms / 1000.0) if latency_ms is not None else None

    # Stage-model inputs derived from baseline row (per your paper / approach)
    t_base = out.get("stage_duration_s_baseline")
    num_tasks_base = out.get("num_tasks_baseline")

    spark_execs = out.get("spark_num_executors")
    spark_cores = out.get("spark_executor_cores")
    parallelism = spark_execs * spark_cores
    out["task_waves"] = safe_div(float(num_tasks_base), float(parallelism)) if (num_tasks_base and parallelism) else 1.0
    out["task_waves"] = max(1.0, out["task_waves"] or 1.0)

    partitions = out.get("number_of_partitions_read_baseline")
    out["partition_access_per_task"] = safe_div(
        float(partitions) if partitions is not None else None,
        float(num_tasks_base) if num_tasks_base is not None else None,
    )

    # Call stage model
    pred, components = stage_model.predict_stage_duration(
        T_base=float(t_base) if t_base is not None else 0.0,
        N_tasks=int(num_tasks_base) if num_tasks_base is not None else 0,
        N_partitions=int(partitions) if partitions is not None else 0,
        input_read_gb=float(out.get("input_read_gb_baseline") or 0.0),
        window_max_bytes=int(out.get("window_max_bytes") or 0),
        mtu_bytes=int(out.get("mtu") or 1500),
        spark_num_executors=int(spark_execs or 0),
        cores_per_executor=int(spark_cores or 0),
        network_latency_ms=float(out.get("network_latency_ms") or 0.0),
        B_link=float(out.get("tsg_max_bw_gbs") or 0.0),
    )

    out.update(components)
    out["T_pred_total"] = pred

    qid = out.get("query_id")
    sid = out.get("stage_id")
    out["fold_id"] = f"{qid}_{sid}"

    return out


def enrich_by_match_id(
    hybrid_rows: List[Dict[str, Any]],
    baseline_lookup: Dict[Tuple[Any, ...], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Enrich all hybrid stage rows using baseline lookup keyed by match_id.
    Missing baseline rows are skipped (with a print), not fatal.
    """
    engineer_features(hybrid_rows)

    enriched: List[Dict[str, Any]] = []
    missing = 0

    for r in hybrid_rows:
        k = make_key(r, MATCH_ID_KEYS)
        base = baseline_lookup.get(k)

        if base is None:
            missing += 1
            print(f"Missing baseline for match_id={r.get('match_id')} (debug_id={r.get('debug_id')})")
            continue

        enriched.append(enrich_row(r, base))

    print(f"Enriched {len(enriched)} rows (missing_baseline={missing})")
    return enriched


def main() -> None:
    print(f"Loading stage measurements from {INPUT_FILE}")
    with INPUT_FILE.open() as f:
        full: List[Dict[str, Any]] = json.load(f)

    baseline_lookup = build_baseline_lookup(full)

    hybrid = [r for r in full if r.get("tsg_max_bw_gbs") != BASELINE_BANDWIDTH_GBPS]
    print(f"Found {len(hybrid)} hybrid stage rows")

    enriched = enrich_by_match_id(hybrid, baseline_lookup)

    print(f"Writing enriched stage dataset to {OUTPUT_FILE}")
    with OUTPUT_FILE.open("w") as f:
        json.dump(enriched, f, indent=2)


if __name__ == "__main__":
    main()