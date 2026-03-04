from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------- data loading --------------------
def load_to_df(filepath: str | Path) -> pd.DataFrame:
    with open(filepath, "r") as f:
        raw = json.load(f)
    return pd.DataFrame(raw)


def load_enriched_stage_measurement_summary() -> pd.DataFrame:
    return load_to_df("data/enriched_stage_measurement_summary.json")


# -------------------- metrics (paper-aligned) --------------------
def mape_pct(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def wmape_pct(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.maximum(np.sum(np.abs(y_true)), eps)
    return float((numerator / denominator) * 100.0)


def metrics_dict(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    y_true = np.asarray(list(y_true), float)
    y_pred = np.asarray(list(y_pred), float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mape = mape_pct(y_true, y_pred)
    wmape = wmape_pct(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))

    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape, "WMAPE%": wmape, "R2": r2}


# -------------------- evaluation --------------------
def evaluate_by_bw_latency(
    df: pd.DataFrame,
    y_col: str = "stage_duration_s",
    yhat_col: str = "y_hat",
    bw_col: str = "tsg_max_bw_gbs",
    lat_col: str = "network_latency_ms",
    mtu_col: str = "mtu",
) -> pd.DataFrame:
    """
    Report metrics overall and grouped by (bandwidth, latency, mtu).
    Output columns match paper tables: MAE, RMSE, (MAPE/WMAPE), R2.
    """
    cols = [y_col, yhat_col, bw_col, lat_col, mtu_col]
    df = df.dropna(subset=cols).copy()

    # coerce to numeric (errors -> NaN -> drop)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols)

    # overall metrics
    overall = metrics_dict(df[y_col].values, df[yhat_col].values)
    overall_row = {
        bw_col: "ALL",
        lat_col: "ALL",
        mtu_col: "ALL",
        "count": int(len(df)),
        **overall,
    }

    # per (bw, latency, mtu) metrics
    rows = []
    for (bw, lat, mtu), g in df.groupby([bw_col, lat_col, mtu_col], sort=False):
        core = metrics_dict(g[y_col].values, g[yhat_col].values)
        rows.append(
            {bw_col: bw, lat_col: lat, mtu_col: mtu, "count": int(len(g)), **core}
        )

    detail = pd.DataFrame(rows)

    if not detail.empty:
        detail[bw_col] = pd.to_numeric(detail[bw_col], errors="coerce")
        detail[lat_col] = pd.to_numeric(detail[lat_col], errors="coerce")
        detail[mtu_col] = pd.to_numeric(detail[mtu_col], errors="coerce")
        detail = detail.sort_values(
            [bw_col, lat_col, mtu_col], ascending=[False, True, True]
        ).reset_index(drop=True)

    return pd.concat([pd.DataFrame([overall_row]), detail], ignore_index=True)


def identify_outliers(
    df: pd.DataFrame,
    bw: float,
    lat: float,
    y_col: str = "stage_duration_s",
    yhat_col: str = "y_hat",
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the top-N rows contributing most to squared error for a given (bw, latency) pair."""
    subset = df[
        (df["tsg_max_bw_gbs"] == bw) & (df["network_latency_ms"].sub(lat).abs() <= 1)
    ].copy()
    if subset.empty:
        print(f"No rows for {bw} Gbps, {lat} ms")
        return subset

    subset["residual"] = subset[y_col] - subset[yhat_col]
    subset["sq_residual"] = subset["residual"] ** 2

    total_se = subset["sq_residual"].sum()
    subset["contrib_pct"] = 100.0 * subset["sq_residual"] / max(total_se, 1e-9)

    top = subset.sort_values("sq_residual", ascending=False).head(top_n)
    cols = [
        "query_id",
        "debug_id",
        "match_id",
        "match_category",
        "stage_duration_s_baseline",
        y_col,
        yhat_col,
        "residual",
        "sq_residual",
        "contrib_pct",
        "shuffle_read_gb",
        "input_read_gb",
        "shuffle_write_gb",
        "num_tasks",
        "task_waves",
    ]
    return top[cols]


# -------------------- evaluation modes (paper sections) --------------------
def no_network_transfer(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = df[df["input_read_gb_baseline"] == 0.0].copy()
    filtered["y_hat"] = filtered["stage_duration_s_baseline"]
    results = evaluate_by_bw_latency(filtered)
    print("\n=== Stages Without Remote Data Transfer (Baseline) ===")
    print(results.to_string(index=False))
    return filtered, results


def uninformed_baseline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = df[df["input_read_gb_baseline"] > 0.0].copy()
    filtered["y_hat"] = filtered["stage_duration_s_baseline"]
    results = evaluate_by_bw_latency(filtered)
    print("\n=== Stages With Remote Data Transfer (Transport-Agnostic Baseline) ===")
    print(results.to_string(index=False))
    return filtered, results


def bw_only(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = df[df["input_read_gb_baseline"] > 0.0].copy()
    filtered["y_hat"] = filtered["T_bw"]
    results = evaluate_by_bw_latency(filtered)
    print("\n=== Bandwidth-Only Component (T_bw) ===")
    print(results.to_string(index=False))
    return filtered, results


def full_model(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = df[df["input_read_gb_baseline"] > 0.0].copy()
    filtered["y_hat"] = filtered["T_pred_total"]
    results = evaluate_by_bw_latency(filtered)
    print("\n=== Full Transport-Aware Stage Model (T_bw + T_storage + T_dynamic) ===")
    print(results.to_string(index=False))
    return filtered, results


# -------------------- main --------------------
def main() -> None:
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    df = load_enriched_stage_measurement_summary()
    df = df[df["match_category"] == "good"].copy()

    required = {
        "stage_duration_s",
        "stage_duration_s_baseline",
        "T_bw",
        "T_pred_total",
        "tsg_max_bw_gbs",
        "network_latency_ms",
        "mtu",
        "input_read_gb_baseline",
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    no_network_transfer(df)
    uninformed_baseline(df)
    bw_only(df)
    full_model(df)


if __name__ == "__main__":
    main()