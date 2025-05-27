import numpy as np
from .projection import jll_dimension, generate_orthogonal_basis, project_data
from .dimensionality_reduction import (
    softmax, 
    compute_distortion_sample as compute_distortion,
    rank_corr as evaluate_rank_correlation,
    dist_stats as distribution_stats,
    HyperbolicEmbedding,
    SphericalEmbedding
)


def run_jll(X, k, seed=42):
    """
    Run Johnson-Lindenstrauss random projection.

    This is a simplified version that uses the projection module directly.
    For a more robust implementation with error handling, use the version in
    dimensionality_reduction.py.

    Parameters:
    -----------
    X : ndarray
        Input data of shape (n_samples, n_features)
    k : int
        Target dimension
    seed : int
        Random seed

    Returns:
    --------
    ndarray
        Projected data of shape (n_samples, k)
    """
    basis = generate_orthogonal_basis(X.shape[1], k, seed=seed)
    Y = project_data(X, basis)
    return Y


def run_poincare_pipeline(X, k, c=1.0, seed=42):
    """
    Run Poincaré (hyperbolic) embedding pipeline.

    This pipeline maps data to the Poincaré disk, projects in the tangent space,
    and maps back to the original space.

    Parameters:
    -----------
    X : ndarray
        Input data of shape (n_samples, n_features)
    k : int
        Target dimension for the projection in tangent space
    c : float
        Curvature parameter for the hyperbolic space
    seed : int
        Random seed

    Returns:
    --------
    ndarray
        Reconstructed data in the original space
    """
    original_dim = X.shape[1]
    emb = HyperbolicEmbedding(c=c)
    Y_disk = emb.euclidean_to_disk(X)
    Y_tangent = emb.log_map_zero(Y_disk)
    Y_proj = run_jll(Y_tangent, k, seed=seed)
    Y_tangent_recon = emb.exp_map_zero(Y_proj)
    Y_recon = emb.disk_to_euclidean(Y_tangent_recon)

    # Restore original dimensionality if needed (simple padding/truncating)
    if Y_recon.shape[1] != original_dim:
        result = np.zeros((Y_recon.shape[0], original_dim))
        # Copy values, handling both truncation and padding
        copy_dims = min(Y_recon.shape[1], original_dim)
        result[:, :copy_dims] = Y_recon[:, :copy_dims]
        return result

    return Y_recon


def run_spherical_pipeline(X, k, seed=42):
    """
    Run Spherical embedding pipeline.

    This pipeline maps data to the unit sphere, projects in the tangent space
    at the north pole, and maps back to the original space.

    Parameters:
    -----------
    X : ndarray
        Input data of shape (n_samples, n_features)
    k : int
        Target dimension for the projection in tangent space
    seed : int
        Random seed

    Returns:
    --------
    ndarray
        Reconstructed data in the original space
    """
    original_dim = X.shape[1]
    emb = SphericalEmbedding()
    Y_sphere = emb.euclidean_to_sphere(X)
    Y_tangent = emb.log_map_north_pole(Y_sphere)
    Y_proj = run_jll(Y_tangent, k, seed=seed)
    Y_recon = emb.exp_map_north_pole(Y_proj)

    # Restore original dimensionality if needed (simple padding/truncating)
    if Y_recon.shape[1] != original_dim:
        result = np.zeros((Y_recon.shape[0], original_dim))
        # Copy values, handling both truncation and padding
        copy_dims = min(Y_recon.shape[1], original_dim)
        result[:, :copy_dims] = Y_recon[:, :copy_dims]
        return result

    return Y_recon


def run_experiment(n=15000, d=1200, epsilon=0.2, seed=42, sample_size=5000):
    """
    Run dimensionality reduction experiment with various methods.

    This is a wrapper around the more comprehensive implementation in 
    dimensionality_reduction.py, maintained for backward compatibility.

    Parameters:
    -----------
    n : int
        Number of data points
    d : int
        Original dimensionality
    epsilon : float
        Desired maximum distortion
    seed : int
        Random seed
    sample_size : int
        Sample size for distortion computation

    Returns:
    --------
    dict
        Results for each method
    """
    from .dimensionality_reduction import run_experiment as run_experiment_full

    # Call the more comprehensive implementation with default geometric embeddings
    return run_experiment_full(
        n=n, 
        d=d, 
        epsilon=epsilon, 
        seed=seed, 
        sample_size=sample_size,
        use_poincare=True,
        use_spherical=True,
        use_elliptic=False
    )
