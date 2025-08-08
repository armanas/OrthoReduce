"""
OrthoReduce: Dimensionality Reduction with Orthogonal Projections

This package provides a clean, unified interface for dimensionality reduction
techniques including:
- Johnson-Lindenstrauss random projections
- Principal Component Analysis (PCA) 
- UMAP (optional)
- Gaussian random projections
- Post-processing calibration utilities for improved correlation
- Data preprocessing and whitening utilities
- Evaluation metrics and MS data processing

New Calibration Features:
- Isotonic regression calibration for monotonic distance mapping
- Procrustes alignment for removing rigid transformations
- Local linear correction for per-point neighborhood rescaling
- Combined calibration methods for maximum improvement

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
    run_poincare, run_spherical,
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
    # Enhanced calibration functions
    evaluate_projection_calibrated,
    compute_calibrated_correlation,
    benchmark_calibration_methods,
)

# Post-processing calibration utilities
try:
    from .calibration import (
        isotonic_regression_calibration,
        procrustes_alignment,
        local_linear_correction,
        combined_calibration,
    )
except ImportError:
    pass

# Data preprocessing utilities
try:
    from .preprocessing import (
        # Standardization functions
        standardize_features,
        l2_normalize_rows,
        whiten_data,
        # Denoising functions
        pca_denoise,
        jl_denoise,
        # Metric utilities
        compute_cosine_distances,
        spearman_loss,
        compute_distance_matrix,
        # Adaptive pipeline
        adaptive_preprocessing_pipeline,
    )
except ImportError:
    pass

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
    from .convex_optimized import project_onto_convex_hull_qp, project_onto_convex_hull_enhanced
except ImportError:
    pass

# Advanced visualization capabilities
try:
    from .advanced_plotting import (
        # Main plotting classes
        AdvancedPlotter,
        InteractivePlotter,
        # Convenience functions
        plot_embedding_comparison,
        create_evaluation_report,
        quick_embedding_plot,
        plot_specialized_embedding,
        # Styling utilities
        setup_enhanced_plotting,
        METHOD_COLORS_EXTENDED,
        QUALITY_COLORS_EXTENDED,
    )
except ImportError:
    pass

# Enhanced experiment function with visualization
try:
    from .dimensionality_reduction import run_experiment_with_visualization
except ImportError:
    pass

# Professional visualization system
try:
    from .visualization import OrthoReduceVisualizer
except ImportError:
    pass

# Hyperbolic geometry operations (optional)
try:
    from .hyperbolic import (
        PoincareBall,
        RiemannianOptimizer, 
        HyperbolicEmbedding,
        run_poincare_optimized
    )
except ImportError:
    pass

# For backward compatibility - simple aliases
run_jll_full = run_jll
compute_distortion_exact = compute_distortion
evaluate_rank_correlation = rank_correlation
