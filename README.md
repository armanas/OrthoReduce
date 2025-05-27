# OrthoReduce: Dimensionality Reduction with Orthogonal Projections

OrthoReduce is a Python library for dimensionality reduction using orthogonal projections and geometric embeddings. It provides implementations of various dimensionality reduction techniques, including:

- Johnson-Lindenstrauss (JLL) random projections
- Principal Component Analysis (PCA)
- Poincaré (hyperbolic) embeddings
- Spherical embeddings
- UMAP (optional)

The library focuses on preserving pairwise distances and geometric structure during dimensionality reduction, with a particular emphasis on numerical stability.

## Installation

```bash
# Clone the repository
git clone https://github.com/armanas/OrthoReduce.git
cd OrthoReduce

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
import numpy as np
from orthogonal_projection.projection import generate_orthogonal_basis, project_data, jll_dimension
from orthogonal_projection.evaluation import compute_distortion, nearest_neighbor_overlap

# Generate synthetic data
n, d = 1000, 100  # n points in d dimensions
X = np.random.randn(n, d)
X = X / np.linalg.norm(X, axis=1, keepdims=True)  # Normalize to unit sphere

# Calculate target dimension using Johnson-Lindenstrauss lemma
epsilon = 0.2  # Desired maximum distortion
k = min(jll_dimension(n, epsilon), d)

# Generate orthogonal basis and project data
basis = generate_orthogonal_basis(d, k, seed=42)
Y = project_data(X, basis)

# Evaluate distortion
mean_distortion, max_distortion, _, _ = compute_distortion(X, Y)
print(f"Mean distortion: {mean_distortion:.4f}")
print(f"Max distortion: {max_distortion:.4f}")

# Evaluate nearest neighbor preservation
nn_overlap = nearest_neighbor_overlap(X, Y, k=10)
print(f"Nearest neighbor overlap: {nn_overlap:.4f}")
```

### Running Experiments

The library includes a command-line interface for running dimensionality reduction experiments:

```bash
python main_dimensionality_reduction.py --n 1000 --d 100 --sample_size 500 --use_poincare --use_spherical
```

Parameters:
- `--n`: Number of data points (default: 15000)
- `--d`: Original dimensionality (default: 1200)
- `--epsilon`: Desired maximum distortion (default: 0.2)
- `--seed`: Random seed (default: 42)
- `--sample_size`: Sample size for distortion computation (default: 5000)
- `--use_poincare`: Use Poincaré embedding
- `--use_spherical`: Use Spherical embedding
- `--use_elliptic`: Use Elliptic embedding (not implemented yet)

## Available Methods

### Johnson-Lindenstrauss (JLL) Random Projections

JLL projections use random orthogonal matrices to project high-dimensional data into lower-dimensional spaces while approximately preserving pairwise distances.

### Principal Component Analysis (PCA)

PCA finds the directions of maximum variance in the data and projects onto those directions.

### Poincaré (Hyperbolic) Embeddings

Poincaré embeddings map data to the Poincaré disk (a model of hyperbolic space), which can better preserve hierarchical structures.

### Spherical Embeddings

Spherical embeddings map data to the unit sphere, which can be useful for directional data or when angular distances are important.

### UMAP (Optional)

Uniform Manifold Approximation and Projection (UMAP) is a dimensionality reduction technique that preserves both local and global structure. This method is optional and requires the `umap-learn` package.

## Evaluation Metrics

The library provides several metrics to evaluate the quality of dimensionality reduction:

- **Mean/Max Distortion**: Measures how well pairwise distances are preserved
- **Rank Correlation**: Spearman correlation between original and reduced pairwise distances
- **Nearest Neighbor Overlap**: Measures how well nearest neighbors are preserved
- **KL Divergence**: Measures the difference between softmax distributions in original and reduced spaces

## License

This project is licensed under the terms of the LICENSE file included in the repository.