import json
from pathlib import Path
from typing import Dict, List, Tuple, Any


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

BASELINE_BANDWIDTH_GBPS = 999.0  # On-prem baseline bandwidth


BASELINE_FIELDS: Dict[str, str] = {
    "exec_time_s": "exec_time_s_baseline",
    "worker_avg_cpu": "worker_avg_cpu_baseline",
    "tsg_avg_bw_gbs": "tsg_avg_bw_gbs_baseline",
    "tsg_transfer_gb": "tsg_transfer_gb_baseline",
    "num_spark_stages": "num_spark_stages_baseline",
    "spark_total_executor_runtime_s": "spark_total_executor_runtime_s_baseline",
    "spark_total_executor_cpu_time_s": "spark_total_executor_cpu_time_s_baseline",
    "spark_total_input_read_gb": "spark_total_input_read_gb_baseline",
    "spark_total_shuffle_read_gb": "spark_total_shuffle_read_gb_baseline",
    "estimated_query_avg_bw_gbps": "estimated_query_avg_bw_gbps_baseline",
    "executor_cpu_utilization_ratio": "executor_cpu_utilization_ratio_baseline",
    "executor_cpu_time_to_exec_time_ratio": "executor_cpu_time_to_exec_time_ratio_baseline",
    "executor_cpu_runtime_to_exec_time_ratio": "executor_cpu_runtime_to_exec_time_ratio_baseline",
}


MATCH_KEYS: List[str] = [
    "query_id",
    "compression_codec_uncompressed",
    "compression_codec_snappy",
    "compression_codec_gzip",
    "file_format_parquet",
    "file_format_orc",
    "num_workers",
    "worker_disk_bw_gbps",
    "worker_net_bw_gbps",
    "total_worker_disk_bw_gbps",
    "total_worker_net_bw_gbps",
]


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def make_match_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    """Create a tuple-based key uniquely identifying a query configuration."""
    return tuple(row.get(key) for key in MATCH_KEYS)


def make_baseline_lookup_table(data: List[Dict[str, Any]]) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    """Build lookup table mapping match key → baseline row."""
    lookup: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    for row in data:
        if row.get("tsg_max_bw_gbs") == BASELINE_BANDWIDTH_GBPS:
            key = make_match_key(row)

            if key in lookup:
                raise ValueError(f"Duplicate baseline entry detected for key: {key}")

            lookup[key] = row

    print(f"Identified {len(lookup)} baseline rows")
    return lookup


def engineer_features(data: List[Dict[str, Any]]) -> None:
    """
    Compute derived modeling features in-place.
    """
    for row in data:
        exec_time = row.get("exec_time_s", 0.0)
        input_read = row.get("spark_total_input_read_gb", 0.0)

        if exec_time > 0:
            row["estimated_query_avg_bw_gbps"] = (input_read / exec_time) * 8.0
        else:
            row["estimated_query_avg_bw_gbps"] = 0.0


def enrich_rows(
    data: List[Dict[str, Any]],
    baseline_lookup: Dict[Tuple[Any, ...], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Enrich rows with baseline fields and derived degradation metrics.
    """
    enriched: List[Dict[str, Any]] = []

    for row in data:
        key = make_match_key(row)
        baseline_row = baseline_lookup.get(key)

        if baseline_row is None:
            raise KeyError(f"No baseline match found for query {row.get('query_id')}")

        enriched_row = row.copy()

        # Copy baseline fields
        for src, target in BASELINE_FIELDS.items():
            enriched_row[target] = baseline_row.get(src)


        enriched.append(enriched_row)

    return enriched


# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------

def main() -> None:
    input_path = Path("data/raw/measurement_summary.json")
    output_path = Path("data/enriched_measurement_summary.json")

    with input_path.open() as f:
        data = json.load(f)

    engineer_features(data)
    baseline_lookup = make_baseline_lookup_table(data)
    enriched_data = enrich_rows(data, baseline_lookup)

    print(f"Writing enriched dataset to {output_path}")

    with output_path.open("w") as f:
        json.dump(enriched_data, f, indent=2)


if __name__ == "__main__":
    main()