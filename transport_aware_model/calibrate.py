# calibrate.py
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
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
    print("Done.")


if __name__ == "__main__":
    main()