# calibrate.py
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

from transport_aware_model.data_management.load_data import (
    load_enriched_stage_measurement_summary,
)


INPUT_FILE = Path("data/enriched_stage_measurement_summary.json")
OUTPUT_DIR = Path("data/system_identification")
OUTPUT_STATS_FILE = OUTPUT_DIR / "system_params_ci.json"
OUTPUT_COEFFS_FILE = OUTPUT_DIR / "system_params_bootstrap.csv"
OUTPUT_PLOT_FILE = OUTPUT_DIR / "system_params_bootstrap.png"

N_ITERATIONS = 1000

FEATURE_COLS = [
    "T_bw_base",
    "T_dynamic_base",
    "T_storage_base",
]


def main() -> None:
    # Ensure input exists (also keeps your artifact self-contained)
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}. "
            "Run the data preparation step first (prepare-data)."
        )

    # Load dataset via your loader (single source of truth)
    df = load_enriched_stage_measurement_summary()

    # Filter: keep only stages with remote input read
    df = df[df["input_read_gb_baseline"] > 0.0].copy()
    print(f"DataFrame shape after filtering: {df.shape}")

    # Feature matrix / target
    X = df[FEATURE_COLS]
    y = df["stage_duration_s"]

    # Stratified bins by baseline input read volume (small vs large stages)
    df["stratify_bin"] = pd.qcut(df["input_read_gb_baseline"], q=5, labels=False)

    coeffs: List[List[float]] = []
    print(f"Starting stratified bootstrapping ({N_ITERATIONS} iterations)...")

    for i in range(N_ITERATIONS):
        X_sample, y_sample = resample(X, y, stratify=df["stratify_bin"], random_state=i)

        # NNLS via LinearRegression with positivity constraint
        model = LinearRegression(positive=True, fit_intercept=False)
        model.fit(X_sample, y_sample)

        coeffs.append(model.coef_.tolist())

    # Results dataframe
    coeff_df = pd.DataFrame(coeffs, columns=["beta_bw", "beta_dynamic", "beta_storage"])

    # Statistics
    stats = coeff_df.describe(percentiles=[0.025, 0.975]).T
    stats_out = stats[["mean", "std", "2.5%", "97.5%"]]
    print("\n=== System Parameter Confidence Intervals (95%) ===")
    print(stats_out)

    # Persist results (reproducibility)
    OUTPUT_COEFFS_FILE.parent.mkdir(parents=True, exist_ok=True)
    coeff_df.to_csv(OUTPUT_COEFFS_FILE, index=False)

    with OUTPUT_STATS_FILE.open("w") as f:
        json.dump(stats_out.to_dict(orient="index"), f, indent=2)

    plt.figure(figsize=(9, 5))

    bp = plt.boxplot(
        [coeff_df["beta_bw"], coeff_df["beta_dynamic"], coeff_df["beta_storage"]],
        tick_labels=[r"$\beta_{bw}$", r"$\beta_{dynamic}$", r"$\beta_{storage}$"],
        showfliers=True,
        patch_artist=True,
    )

    # Set full light blue styling
    for box in bp["boxes"]:
        box.set(facecolor="lightblue", edgecolor="black", linewidth=1.2)

    for median in bp["medians"]:
        median.set(color="black", linewidth=1.5)

    for whisker in bp["whiskers"]:
        whisker.set(color="black", linewidth=1.0)

    for cap in bp["caps"]:
        cap.set(color="black", linewidth=1.0)

    for flier in bp["fliers"]:
        flier.set(marker="o", markerfacecolor="black", markersize=3, alpha=0.6)

    plt.ylabel("Coefficient value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_FILE, dpi=200)
    plt.close()

    print(f"\nSaved bootstrap coefficients to: {OUTPUT_COEFFS_FILE}")
    print(f"Saved confidence intervals to:   {OUTPUT_STATS_FILE}")
    print(f"Saved plot to:                  {OUTPUT_PLOT_FILE}")

    # Workload-level cross-validation (80/20 split of query templates).
    # Complements the stage-level stratified bootstrap above by assessing
    # coefficient stability when the calibration workload changes.
    workload_level_cv(df)

    # Predictor collinearity diagnostic (variance inflation factor) for the
    # three model components. Quantifies why beta_dynamic and beta_storage
    # exhibit larger dispersion in the workload-level cross-validation.
    print_vif(df)

    print("Done.")


def print_vif(df: pd.DataFrame) -> None:
    """Compute and print the Variance Inflation Factor for each predictor.

    VIF_i = (R^-1)_{ii}, where R is the correlation matrix of the
    predictors. VIF = 1 indicates no correlation with the other
    predictors; values above ~5 indicate moderate collinearity.
    """
    X = df[FEATURE_COLS].values
    # Standardize columns then form the correlation matrix.
    Xs = (X - X.mean(axis=0)) / X.std(axis=0)
    R = np.corrcoef(Xs.T)
    vifs = np.diag(np.linalg.inv(R))

    print("\n=== Variance Inflation Factor (predictor collinearity) ===")
    for col, v in zip(FEATURE_COLS, vifs):
        print(f"  {col:18s}  VIF = {v:.3f}")


def workload_level_cv(df: pd.DataFrame) -> None:
    """1,000-iteration 80/20 workload-level cross-validation.

    Partitions the TPC-DS query templates 80/20 (train/held-out), refits
    NNLS on the training set, and reports Mean +/- Std.Dev. of the
    structural coefficients across iterations.
    """
    queries = sorted(df["query_id"].unique())
    n_queries = len(queries)
    n_hold = int(n_queries * 0.2)

    print(
        f"\nStarting workload-level 80/20 cross-validation "
        f"({N_ITERATIONS} iterations, {n_queries} query templates)..."
    )

    coeffs: List[List[float]] = []
    for seed in range(N_ITERATIONS):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(queries)
        train_queries = set(perm[n_hold:])
        sub = df[df["query_id"].isin(train_queries)]

        model = LinearRegression(positive=True, fit_intercept=False)
        model.fit(sub[FEATURE_COLS], sub["stage_duration_s"])
        coeffs.append(model.coef_.tolist())

    cv_df = pd.DataFrame(
        coeffs, columns=["beta_bw", "beta_dynamic", "beta_storage"]
    )

    stats = cv_df.describe(percentiles=[0.025, 0.975]).T
    stats_out = stats[["mean", "std", "2.5%", "97.5%"]]

    print("\n=== Workload-Level 80/20 Cross-Validation (95%) ===")
    print(stats_out)


if __name__ == "__main__":
    main()