"""Visualize and compare real vs synthetic PV power data distributions."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def compare_distributions(
    real_csv: str,
    synthetic_csv: str,
    output_plot: str = "data/distribution_comparison.png",
):
    """
    Create comprehensive visualization plots comparing real and synthetic data.

    Parameters
    ----------
    real_csv:
        Path to real data CSV file.
    synthetic_csv:
        Path to synthetic data CSV file.
    output_plot:
        Path where the comparison plot will be saved.
    """
    print("=" * 70)
    print("Loading data for comparison...")
    
    # Load data
    real_df = pd.read_csv(real_csv)
    real_df["Datetime"] = pd.to_datetime(real_df["Datetime"])
    real_df = real_df.dropna(subset=["P"]).copy()
    real_df = real_df.sort_values("Datetime").reset_index(drop=True)
    
    synthetic_df = pd.read_csv(synthetic_csv)
    synthetic_df["Datetime"] = pd.to_datetime(synthetic_df["Datetime"])
    synthetic_df = synthetic_df.dropna(subset=["P"]).copy()
    synthetic_df = synthetic_df.sort_values("Datetime").reset_index(drop=True)

    print(f"Real data: {len(real_df)} rows")
    print(f"Synthetic data: {len(synthetic_df)} rows")

    # Create comprehensive comparison plots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Time series plot - P_exp (full)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(real_df["Datetime"], real_df["P_exp"], alpha=0.5, label="Real", linewidth=0.8, color="blue")
    ax1.plot(synthetic_df["Datetime"], synthetic_df["P_exp"], alpha=0.5, label="Synthetic", linewidth=0.8, color="orange")
    ax1.set_xlabel("Datetime", fontsize=10)
    ax1.set_ylabel("P_exp (W)", fontsize=10)
    ax1.set_title("Time Series: Expected Power (Full)", fontsize=11, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 2. Time series plot - P (full)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(real_df["Datetime"], real_df["P"], alpha=0.5, label="Real", linewidth=0.8, color="blue")
    ax2.plot(synthetic_df["Datetime"], synthetic_df["P"], alpha=0.5, label="Synthetic", linewidth=0.8, color="orange")
    ax2.set_xlabel("Datetime", fontsize=10)
    ax2.set_ylabel("P (W)", fontsize=10)
    ax2.set_title("Time Series: Actual Power (Full)", fontsize=11, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 3. Time series plot - Zoomed (first week)
    ax3 = fig.add_subplot(gs[0, 2])
    week_end = min(real_df["Datetime"].min() + pd.Timedelta(days=7), real_df["Datetime"].max())
    real_week = real_df[real_df["Datetime"] <= week_end]
    synth_week = synthetic_df[synthetic_df["Datetime"] <= week_end]
    ax3.plot(real_week["Datetime"], real_week["P_exp"], alpha=0.7, label="Real", linewidth=1.5, color="blue")
    ax3.plot(synth_week["Datetime"], synth_week["P_exp"], alpha=0.7, label="Synthetic", linewidth=1.5, color="orange")
    ax3.set_xlabel("Datetime", fontsize=10)
    ax3.set_ylabel("P_exp (W)", fontsize=10)
    ax3.set_title("Time Series: Expected Power (First Week)", fontsize=11, fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 4. Distribution histogram - P_exp
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(real_df["P_exp"], bins=60, alpha=0.6, label="Real", density=True, color="blue", edgecolor="black", linewidth=0.5)
    ax4.hist(synthetic_df["P_exp"], bins=60, alpha=0.6, label="Synthetic", density=True, color="orange", edgecolor="black", linewidth=0.5)
    ax4.set_xlabel("P_exp (W)", fontsize=10)
    ax4.set_ylabel("Density", fontsize=10)
    ax4.set_title("Distribution: Expected Power", fontsize=11, fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    # 5. Distribution histogram - P
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(real_df["P"], bins=60, alpha=0.6, label="Real", density=True, color="blue", edgecolor="black", linewidth=0.5)
    ax5.hist(synthetic_df["P"], bins=60, alpha=0.6, label="Synthetic", density=True, color="orange", edgecolor="black", linewidth=0.5)
    ax5.set_xlabel("P (W)", fontsize=10)
    ax5.set_ylabel("Density", fontsize=10)
    ax5.set_title("Distribution: Actual Power", fontsize=11, fontweight="bold")
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis="y")

    # 6. KDE comparison - P_exp
    ax6 = fig.add_subplot(gs[1, 2])
    sns.kdeplot(data=real_df["P_exp"], label="Real", ax=ax6, color="blue", linewidth=2)
    sns.kdeplot(data=synthetic_df["P_exp"], label="Synthetic", ax=ax6, color="orange", linewidth=2)
    ax6.set_xlabel("P_exp (W)", fontsize=10)
    ax6.set_ylabel("Density", fontsize=10)
    ax6.set_title("KDE: Expected Power", fontsize=11, fontweight="bold")
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis="y")

    # 7. Scatter plot - P_exp vs P (Real)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.scatter(real_df["P_exp"], real_df["P"], alpha=0.2, s=2, color="blue", label="Real")
    # Add diagonal line
    max_val = max(real_df["P_exp"].max(), real_df["P"].max())
    ax7.plot([0, max_val], [0, max_val], "r--", alpha=0.5, linewidth=1, label="y=x")
    ax7.set_xlabel("P_exp (W)", fontsize=10)
    ax7.set_ylabel("P (W)", fontsize=10)
    ax7.set_title("Scatter: Real Data (P_exp vs P)", fontsize=11, fontweight="bold")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Scatter plot - P_exp vs P (Synthetic)
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.scatter(synthetic_df["P_exp"], synthetic_df["P"], alpha=0.2, s=2, color="orange", label="Synthetic")
    # Add diagonal line
    max_val = max(synthetic_df["P_exp"].max(), synthetic_df["P"].max())
    ax8.plot([0, max_val], [0, max_val], "r--", alpha=0.5, linewidth=1, label="y=x")
    ax8.set_xlabel("P_exp (W)", fontsize=10)
    ax8.set_ylabel("P (W)", fontsize=10)
    ax8.set_title("Scatter: Synthetic Data (P_exp vs P)", fontsize=11, fontweight="bold")
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Daily pattern comparison (hourly average)
    ax9 = fig.add_subplot(gs[2, 2])
    real_df["Hour"] = real_df["Datetime"].dt.hour
    synthetic_df["Hour"] = synthetic_df["Datetime"].dt.hour
    real_hourly = real_df.groupby("Hour")["P_exp"].mean()
    synth_hourly = synthetic_df.groupby("Hour")["P_exp"].mean()
    ax9.plot(real_hourly.index, real_hourly.values, marker="o", label="Real", linewidth=2, color="blue", markersize=6)
    ax9.plot(synth_hourly.index, synth_hourly.values, marker="s", label="Synthetic", linewidth=2, color="orange", markersize=6)
    ax9.set_xlabel("Hour of Day", fontsize=10)
    ax9.set_ylabel("Average P_exp (W)", fontsize=10)
    ax9.set_title("Daily Pattern: Hourly Average", fontsize=11, fontweight="bold")
    ax9.set_xticks(range(0, 24, 2))
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    fig.suptitle("Real vs Synthetic PV Power Data - Comprehensive Comparison", fontsize=16, fontweight="bold", y=0.995)
    plt.savefig(output_plot, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to: {output_plot}")
    plt.close()

    # Print detailed statistical comparison
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'Real':<20} {'Synthetic':<20} {'Diff %':<15}")
    print("-" * 70)
    
    metrics = {
        "P_exp - Mean": (real_df["P_exp"].mean(), synthetic_df["P_exp"].mean()),
        "P_exp - Std Dev": (real_df["P_exp"].std(), synthetic_df["P_exp"].std()),
        "P_exp - Min": (real_df["P_exp"].min(), synthetic_df["P_exp"].min()),
        "P_exp - Max": (real_df["P_exp"].max(), synthetic_df["P_exp"].max()),
        "P_exp - Median": (real_df["P_exp"].median(), synthetic_df["P_exp"].median()),
        "P_exp - Q25": (real_df["P_exp"].quantile(0.25), synthetic_df["P_exp"].quantile(0.25)),
        "P_exp - Q75": (real_df["P_exp"].quantile(0.75), synthetic_df["P_exp"].quantile(0.75)),
        "P - Mean": (real_df["P"].mean(), synthetic_df["P"].mean()),
        "P - Std Dev": (real_df["P"].std(), synthetic_df["P"].std()),
        "P - Min": (real_df["P"].min(), synthetic_df["P"].min()),
        "P - Max": (real_df["P"].max(), synthetic_df["P"].max()),
        "P - Median": (real_df["P"].median(), synthetic_df["P"].median()),
        "P - Q25": (real_df["P"].quantile(0.25), synthetic_df["P"].quantile(0.25)),
        "P - Q75": (real_df["P"].quantile(0.75), synthetic_df["P"].quantile(0.75)),
    }
    
    for metric, (real_val, synth_val) in metrics.items():
        diff = abs(real_val - synth_val)
        diff_pct = (diff / abs(real_val) * 100) if real_val != 0 else 0
        print(f"{metric:<30} {real_val:<20.2f} {synth_val:<20.2f} {diff_pct:<15.2f}%")

    # Correlation comparison
    print("\n" + "-" * 70)
    print("CORRELATION ANALYSIS")
    print("-" * 70)
    real_corr = real_df["P_exp"].corr(real_df["P"])
    synth_corr = synthetic_df["P_exp"].corr(synthetic_df["P"])
    print(f"P_exp vs P correlation - Real: {real_corr:.4f}")
    print(f"P_exp vs P correlation - Synthetic: {synth_corr:.4f}")
    print(f"Correlation difference: {abs(real_corr - synth_corr):.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys

    real_file = "data/SampleData.csv"
    synthetic_file = "data/SyntheticData.csv"
    output_file = "data/distribution_comparison.png"

    if len(sys.argv) > 1:
        real_file = sys.argv[1]
    if len(sys.argv) > 2:
        synthetic_file = sys.argv[2]
    if len(sys.argv) > 3:
        output_file = sys.argv[3]

    compare_distributions(real_file, synthetic_file, output_file)

