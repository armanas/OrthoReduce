"""
OrthoReduce: Dimensionality Reduction with Orthogonal Projections

This package provides a clean, unified interface for dimensionality reduction
techniques including:
- Johnson-Lindenstrauss random projections
- Principal Component Analysis (PCA) 
- UMAP (optional)
- Gaussian random projections
- Evaluation metrics and MS data processing

Simplified Interface:
- Import core functions directly from this module
- For advanced usage, import from specific submodules
"""

# Core projection functionality
from .projection import jll_dimension, generate_orthogonal_basis, project_data

# Main dimensionality reduction interface
from .dimensionality_reduction import (
    # Main experiment function
    run_experiment,
    # Individual method functions
    run_pca, run_jll, run_gaussian_projection, run_umap,
    # Simple interface functions
    run_pca_simple, run_jll_simple, run_umap_simple,
    # Data generation
    generate_mixture_gaussians,
    # Evaluation function
    evaluate_projection,
)

# Evaluation metrics
from .evaluation import (
    compute_distortion,
    rank_correlation,
    nearest_neighbor_overlap,
)

# MS pipeline components (optional imports)
try:
    from .ms_data import parse_mzml_build_matrix
except ImportError:
    pass

try:
    from .fingerprinting import (
        compute_fingerprints,
        cosine_similarity_matrix,
        match_frames_to_peptides,
    )
except ImportError:
    pass

try:
    from .convex_optimized import project_onto_convex_hull_qp
except ImportError:
    pass

# For backward compatibility - simple aliases
run_jll_full = run_jll
compute_distortion_exact = compute_distortion
evaluate_rank_correlation = rank_correlation
