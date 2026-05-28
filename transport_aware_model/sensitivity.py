"""One-at-a-time (OAT) sensitivity analysis of stage-level prediction
quality to individual coefficient errors.

Perturbation bounds are the 95% confidence interval of the restricted
single-configuration calibration (10 Gb/s, 10.49 ms RTT), which
represents the realistic scenario where a practitioner calibrates on
the network conditions available and later operates under a different
configuration without recalibration.

The "calibrated" baseline uses the restricted-mean coefficients
(beta_bw=1.0130, beta_dynamic=1.5703, beta_storage=13.8017); each
perturbation changes one coefficient to its CI bound while holding the
other two at the restricted-mean values.

Metrics (MAE / RMSE / WMAPE / R^2) are reported on two configurations:
  - 2 Gb/s, 25.46 ms RTT  (bandwidth-constrained; T_bw* dominates)
  - 25 Gb/s, 25.46 ms RTT (highest bandwidth at max latency; RTT-driven
    components most prominent)
Both configurations are filtered to MTU = 9001.
"""

from __future__ import annotations

import numpy as np

from transport_aware_model.data_management.load_data import (
    load_enriched_stage_measurement_summary,
)
from transport_aware_model.predict_job import compute_metrics


# Restricted single-configuration calibration (10 Gb/s, 10.49 ms RTT)
# mean coefficients and 95% CI bounds.
R_BW  = 1.0130;  BW_LO  = 0.9986;  BW_HI  = 1.0287
R_DYN = 1.5703;  DYN_LO = 0.5900;  DYN_HI = 2.4672
R_STO = 13.8017; STO_LO = 13.0380; STO_HI = 14.5763

CONFIGS = [
    ("2 Gb/s,  25.46 ms RTT",  2.0,  25.46),
    ("25 Gb/s, 25.46 ms RTT", 25.0,  25.46),
]

CASES = [
    ("Transport-agnostic baseline",   None,   None,   None,   True),
    ("Calibrated (restricted mean)",  R_BW,   R_DYN,  R_STO,  False),
    ("beta_bw  = 0.9986 (-0.4%)",     BW_LO,  R_DYN,  R_STO,  False),
    ("beta_bw  = 1.0287 (+2.6%)",     BW_HI,  R_DYN,  R_STO,  False),
    ("beta_dyn = 0.5900 (-62.4%)",    R_BW,   DYN_LO, R_STO,  False),
    ("beta_dyn = 2.4672 (+57.1%)",    R_BW,   DYN_HI, R_STO,  False),
    ("beta_sto = 13.0380 (-3.4%)",    R_BW,   R_DYN,  STO_LO, False),
    ("beta_sto = 14.5763 (+8.0%)",    R_BW,   R_DYN,  STO_HI, False),
]


def _print_metrics_table(
    title: str, rows: list[tuple[str, str, dict]]
) -> None:
    print("=" * 90)
    print(title)
    print("=" * 90)
    print(f"{'Case':<36}{'Config':<22}{'MAE':>8}{'RMSE':>8}{'WMAPE%':>9}{'R2':>8}")
    print("-" * 90)
    prev_label = None
    for label, cfg, m in rows:
        lbl = label if label != prev_label else ""
        prev_label = label
        print(
            f"{lbl:<36}{cfg:<22}"
            f"{m['MAE']:>8.4f}"
            f"{m['RMSE']:>8.4f}"
            f"{m['WMAPE%']:>9.4f}"
            f"{m['R2']:>8.4f}"
        )
    print()


def main() -> None:
    df = load_enriched_stage_measurement_summary()
    df = df[df["match_category"] == "good"].copy()
    df = df[df["input_read_gb_baseline"] > 0.0].copy()
    df = df[df["mtu"] == 9001].copy()

    print(f"Stage-level OAT sensitivity: {len(df)} stages (MTU=9001).")
    print(
        f"Restricted-mean baseline: beta_bw={R_BW}, "
        f"beta_dynamic={R_DYN}, beta_storage={R_STO}"
    )
    print()

    T_bw  = df["T_bw_base"].values.astype(float)
    T_dyn = df["T_dynamic_base"].values.astype(float)
    T_sto = df["T_storage_base"].values.astype(float)
    y_true = df["stage_duration_s"].values.astype(float)
    y_base = df["stage_duration_s_baseline"].values.astype(float)
    bw_col = df["tsg_max_bw_gbs"].astype(float).values
    lat_col = df["network_latency_ms"].astype(float).round(2).values

    rows: list[tuple[str, str, dict]] = []
    for label, b_bw, b_dyn, b_sto, is_baseline in CASES:
        pred_full = (
            y_base if is_baseline
            else b_bw * T_bw + b_dyn * T_dyn + b_sto * T_sto
        )
        for cfg_name, bwv, latv in CONFIGS:
            mask = (bw_col == bwv) & (lat_col == round(latv, 2))
            rows.append((label, cfg_name, compute_metrics(y_true[mask], pred_full[mask])))

    _print_metrics_table(
        "Stage-level OAT sensitivity — restricted calibration CI bounds",
        rows,
    )


if __name__ == "__main__":
    main()
