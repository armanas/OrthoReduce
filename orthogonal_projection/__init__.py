"""
OrthoReduce: Dimensionality Reduction with Orthogonal Projections

This package provides various dimensionality reduction techniques including:
- Johnson-Lindenstrauss random projections
- Principal Component Analysis (PCA)
- UMAP (optional)
- Geometric embeddings (Poincaré, Spherical)
- Convex hull projections

Main Interface:
- For full functionality, import from dimensionality_reduction
- For simplified interface, import from pipeline
- For basic projections, import from projection
- For evaluation metrics, import from evaluation
"""

# Core projection functionality
from .projection import jll_dimension, generate_orthogonal_basis, project_data

# Full interface with all methods and configurations
from .dimensionality_reduction import (
    # Main experiment function
    run_experiment as run_full_experiment,
    # Individual method functions
    run_pca, run_jll as run_jll_full, run_umap, run_poincare, run_spherical, run_convex,
    # Data generation
    generate_mixture_gaussians,
    # Evaluation functions  
    compute_distortion_sample as compute_distortion,
    rank_corr as evaluate_rank_correlation,
    dist_stats as distribution_stats,
    # Utility functions
    softmax, project_onto_convex_hull, jll_dimension as jll_dim,
    # Geometric embedding classes
    HyperbolicEmbedding, SphericalEmbedding,
)

# Simplified pipeline interface (backward compatibility)
from .pipeline import (
    run_jll,
    run_poincare_pipeline,
    run_spherical_pipeline,
    run_experiment,
)

# Evaluation metrics
from .evaluation import (
    compute_distortion as compute_distortion_exact,
    nearest_neighbor_overlap,
)
