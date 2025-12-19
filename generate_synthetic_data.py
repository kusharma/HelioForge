"""Generate synthetic PV power data using SDV synthesizers.

This script uses GaussianCopulaSynthesizer for tabular data synthesis.
For time series data, ensure SDV is properly installed: pip install sdv
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata


def generate_synthetic_pv_data(
    input_csv: str,
    output_csv: str,
    num_rows: int | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic PV power time series data similar to the input dataset.

    Parameters
    ----------
    input_csv:
        Path to the input CSV file with real data.
    output_csv:
        Path where the synthetic CSV will be saved.
    num_rows:
        Number of synthetic rows to generate. If None, uses the same count as input.

    Returns
    -------
    DataFrame with synthetic data.
    """
    # Load real data
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Remove rows with missing P values for training
    df_clean = df.dropna(subset=["P"]).copy()

    print(f"Loaded {len(df_clean)} rows with complete data")
    print(f"Date range: {df_clean['Datetime'].min()} to {df_clean['Datetime'].max()}")
    print(f"P_exp range: {df_clean['P_exp'].min():.2f} to {df_clean['P_exp'].max():.2f} W")
    print(f"P range: {df_clean['P'].min():.2f} to {df_clean['P'].max():.2f} W")

    # Create metadata
    print("\nTraining synthesizer...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_clean)
    
    # Ensure Datetime is treated as datetime type
    if "Datetime" in metadata.columns:
        metadata.update_column("Datetime", sdtype="datetime")

    # Initialize and train synthesizer
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df_clean)

    # Generate synthetic data
    if num_rows is None:
        num_rows = len(df_clean)

    print(f"\nGenerating {num_rows} synthetic rows...")
    synthetic_df = synthesizer.sample(num_rows=num_rows)

    # Ensure Datetime is datetime type
    synthetic_df["Datetime"] = pd.to_datetime(synthetic_df["Datetime"])

    # Sort by datetime
    synthetic_df = synthetic_df.sort_values("Datetime").reset_index(drop=True)

    # Ensure P_exp and P are non-negative (physical constraint)
    synthetic_df["P_exp"] = synthetic_df["P_exp"].clip(lower=0)
    synthetic_df["P"] = synthetic_df["P"].clip(lower=0)

    # Round to reasonable precision
    synthetic_df["P_exp"] = synthetic_df["P_exp"].round(2)
    synthetic_df["P"] = synthetic_df["P"].round(2)

    # Save to CSV
    print(f"\nSaving synthetic data to {output_csv}...")
    synthetic_df.to_csv(output_csv, index=False)

    print(f"\nSynthetic data summary:")
    print(f"  Rows: {len(synthetic_df)}")
    print(f"  Date range: {synthetic_df['Datetime'].min()} to {synthetic_df['Datetime'].max()}")
    print(f"  P_exp range: {synthetic_df['P_exp'].min():.2f} to {synthetic_df['P_exp'].max():.2f} W")
    print(f"  P range: {synthetic_df['P'].min():.2f} to {synthetic_df['P'].max():.2f} W")
    print(f"  Mean P_exp: {synthetic_df['P_exp'].mean():.2f} W")
    print(f"  Mean P: {synthetic_df['P'].mean():.2f} W")

    return synthetic_df




if __name__ == "__main__":
    input_file = "data/SampleData.csv"
    output_file = "data/SyntheticData.csv"
    create_plots = True

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    if len(sys.argv) > 3:
        create_plots = sys.argv[3].lower() == "true"

    # Generate synthetic data
    synthetic_df = generate_synthetic_pv_data(input_file, output_file)

    # Create comparison plots by calling visualize script as subprocess
    if create_plots:
        script_dir = Path(__file__).parent
        visualize_script = script_dir / "visualize_data_comparison.py"
        output_plot = "data/distribution_comparison.png"
        
        print("\n" + "=" * 60)
        print("Generating comparison plots...")
        print("=" * 60)
        
        result = subprocess.run(
            [
                sys.executable,
                str(visualize_script),
                input_file,
                output_file,
                output_plot,
            ],
            cwd=str(script_dir),
        )
        
        if result.returncode != 0:
            print(f"\nWarning: Visualization script exited with code {result.returncode}")
        else:
            print(f"\n✓ Comparison plots generated successfully")
