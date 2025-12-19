<p align="center">
  <img src="https://raw.githubusercontent.com/kusharma/HelioForge/main/DALLe_HelioForge.png" width="900">
</p>


# PV Toolkit

An open-source library for solar geometry calculations, shadow projection analysis, and PV time-series anomaly detection.

## Overview

This toolkit provides three complementary workflows:

- **Solar Geometry**: Compute sun direction vectors and roof coordinate frames from astronomical principles
- **Shadow Projection**: Project 3D geometries onto tilted roofs and find optimal shadow configurations
- **Anomaly Detection**: Analyze PV power time series and flag anomalies using statistical and ML methods

## Installation

Install the library in editable mode:

```bash
pip install -e .
```

Or using uv:

```bash
uv pip install -e .
```

## Quick Start

### Shadow Calculations

```python
from datetime import datetime
from pvtoolkit.workflows import run_shadow_example, run_shadow_minimization

# Calculate shadow for a specific time
example = run_shadow_example(
    timestamp=datetime(2024, 6, 21, 11, 0),
    latitude=35.0,
    longitude=-105.0,
    tilt_deg=30.0,
    azimuth_deg=135.0,
    height=1.0,
    base_size=1.0,
)

# Find minimum shadow over a year
minimum = run_shadow_minimization(
    latitude=35.0,
    longitude=-105.0,
    tilt_deg=30.0,
    azimuth_deg=135.0,
    height=1.0,
    base_size=1.0,
    year=2024,
)
```

### Anomaly Detection

```python
from pvtoolkit.workflows import assemble_anomaly_workflow
import pandas as pd

# Load PV data
df = pd.read_csv("data/SyntheticData.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])

# Run anomaly detection workflow
results = assemble_anomaly_workflow(df)
```

## Generating Synthetic Data

The repository includes scripts to generate synthetic PV power data using SDV (Synthetic Data Vault):

### Generate Synthetic Data

```bash
python generate_synthetic_data.py
```

This will:
1. Load your SampleData as `data/SampleData.csv` 
2. Train a Gaussian Copula synthesizer
3. Generate `data/SyntheticData.csv` with similar statistical properties
4. Automatically create comparison visualizations

### Custom Parameters

```bash
python generate_synthetic_data.py data/SampleData.csv data/SyntheticData.csv
```

### Generate Visualizations Only

To create comparison plots from existing data:

```bash
python visualize_data_comparison.py data/SampleData.csv data/SyntheticData.csv
```

Or with custom output path:

```bash
python visualize_data_comparison.py data/SampleData.csv data/SyntheticData.csv data/my_comparison.png
```

## Mathematical Foundations

### Solar Geometry

We compute the sun direction vector \( \mathbf{s} = [s_e, s_n, s_u] \) from the apparent zenith \( \theta_z \) and azimuth \( \phi \) angles:

\[
s_e = \sin(\theta_z) \sin(\phi), \qquad
s_n = \sin(\theta_z) \cos(\phi), \qquad
s_u = \cos(\theta_z).
\]

The roof frame assembles three orthonormal vectors:

* \( \mathbf{n} \) — the roof normal defined by tilt \( \alpha \) and azimuth \( \psi \):
\[
\mathbf{n} = [\sin(\alpha)\sin(\psi), \; \sin(\alpha)\cos(\psi), \; \cos(\alpha)].
\]

* \( \mathbf{e}_1 = \mathbf{n} \times \mathbf{k} \) — the down-slope direction (cross product with vertical \( \mathbf{k} = [0, 0, 1] \))
* \( \mathbf{e}_2 = \mathbf{n} \times \mathbf{e}_1 \) — the cross-slope direction completing the right-hand basis

### Shadow Calculation

To compute the roof-projected shadow of a rectangular prism:

1. **Vertex enumeration**: Construct top vertices at height \( h \) above the roof plane with coordinates \( (x, y, h) \) where \( x, y \in \{\pm \tfrac{w}{2}\} \)

2. **Ray-plane intersection**: For each vertex, project along shadow direction \( -\mathbf{s} \). The roof plane is \( z = 0 \), so:

\[
z + t(-s_u) = 0 \quad \Rightarrow \quad t = \frac{z}{s_u}.
\]

The projected point is \( (x - s_e t, \; y - s_n t) \) on the roof plane.

3. **Polygon assembly**: Combine projected points to form a polygon. Area calculated via Shapely.

4. **Validation**: If \( s_u \leq 0 \) (sun below horizon) or \( \mathbf{s} \cdot \mathbf{n} \leq 0 \) (surface not illuminated), shadow area is zero.

### Anomaly Detection

#### Metrics

- **Residual**: \( r = P_\text{exp} - P \) quantifies deviations from expected power
- **Efficiency**: \( \eta = P / P_\text{exp} \) describes system performance

#### Statistical Thresholds

Flag anomalies when residual leaves \( \mu \pm k \sigma \), where \( \mu \) is mean, \( \sigma \) is standard deviation, and \( k = 3 \) for 3-sigma bounds.

#### Machine Learning Detectors

**Isolation Forest**: Partitions feature space using random splits. Observations requiring fewer splits to isolate score as anomalies.

**CBLOF**: Clustering-Based Local Outlier Factor groups data into clusters and flags points in small clusters or far from centroids.

**Autoencoders**: Learn compressed representations of normal observations. Reconstruction error:

\[
\mathrm{error} = \frac{1}{n}\sum_{i=1}^n (x_i - \hat{x}_i)^2
\]

Anomalies are points whose reconstruction error exceeds a high percentile (typically 99%).

## Project Structure

```
shadow-calc/
├── data/
│   ├── SampleData.csv          ← Original data (not committed)
│   └── SyntheticData.csv        ← Generated synthetic data
├── src/
│   └── pvtoolkit/
│       ├── anomaly_detection.py
│       ├── shadow_analysis.py
│       ├── shadow_projection.py
│       ├── solar_geometry.py
│       ├── time_series.py
│       └── workflows/
│           ├── anomaly_workflow.py
│           └── shadow_workflow.py
├── generate_synthetic_data.py   ← Generate synthetic PV data
├── visualize_data_comparison.py ← Create comparison plots
├── README.md
├── pyproject.toml
└── uv.lock
```

## Dependencies

- `numpy>=1.21,<2.0`
- `pandas>=1.3`
- `pvlib>=0.9`
- `shapely>=1.8`
- `matplotlib>=3.5.0`
- `plotly>=5.0.0`
- `seaborn>=0.11.0`
- `scikit-learn>=1.0.0`
- `tensorflow>=2.8.0`
- `pyod>=1.0.0`
- `sdv>=1.0.0`

## Data Files

- **SampleData.csv**: Original PV power time series (not committed to repository)
- **SyntheticData.csv**: Generated synthetic data with similar statistical properties (committed)

To generate synthetic data, run `python generate_synthetic_data.py`. The script will create comparison visualizations automatically.

## Validation

- Shadow projection uses ray-plane intersection formulas and maintains polygonal fidelity through Shapely
- Residual and efficiency metrics follow linear relationships (expected − actual) and ratios
- Anomaly thresholds combine μ ± 3σ bounds, ensemble unsupervised detectors, and autoencoder reconstruction percentiles

## License

Open-source toolkit for research and educational purposes.
