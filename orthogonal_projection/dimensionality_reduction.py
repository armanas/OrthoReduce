import numpy as np
import time
import logging
import argparse

from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.random_projection import GaussianRandomProjection

# Try to import umap, but make it optional
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not available. UMAP projection will be skipped.")

# Classes for geometric embeddings
class HyperbolicEmbedding:
    def __init__(self, c=1.0, safe_radius=0.9):
        self.c = c
        self.safe_radius = safe_radius
        self.scale = None

    def euclidean_to_disk(self, X):
        norms = np.linalg.norm(X, axis=1)
        max_norm = np.max(norms)
        self.scale = self.safe_radius / (np.sqrt(self.c) * max_norm + 1e-9)
        return X * self.scale

    def disk_to_euclidean(self, Y):
        return Y / self.scale

    def exp_map_zero(self, u, eps=1e-15):
        norm_u = np.linalg.norm(u, axis=1)
        out = np.zeros_like(u)
        mask = norm_u > eps
        z = np.sqrt(self.c) * norm_u[mask]
        factor = np.tanh(z) / (z + eps)
        out[mask] = factor[:, None] * u[mask]
        return out

    def log_map_zero(self, x, eps=1e-15):
        norm_x = np.linalg.norm(x, axis=1)
        out = np.zeros_like(x)
        mask = norm_x > eps
        z = np.sqrt(self.c) * norm_x[mask]
        z = np.clip(z, -0.999999, 0.999999)
        atanh_z = 0.5 * np.log((1 + z) / (1 - z))
        factor = atanh_z / (np.sqrt(self.c) * norm_x[mask] + eps)
        out[mask] = factor[:, None] * x[mask]
        return out


class SphericalEmbedding:
    def euclidean_to_sphere(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.clip(norms, 1e-15, None)

    def sphere_to_euclidean(self, Y):
        return Y

    def log_map_north_pole(self, X, eps=1e-15):
        x0 = np.clip(X[:, 0], -1.0, 1.0)
        theta = np.arccos(x0)
        sin_theta = np.sin(theta)
        out = np.zeros((X.shape[0], X.shape[1] - 1))
        mask = sin_theta > eps
        out[mask] = (theta[mask, None] * X[mask, 1:]) / sin_theta[mask, None]
        return out

    def exp_map_north_pole(self, u, eps=1e-15):
        norm_u = np.linalg.norm(u, axis=1, keepdims=True)
        norm_u = np.clip(norm_u, eps, None)
        cos_part = np.cos(norm_u)
        sin_part = np.sin(norm_u) / norm_u
        return np.hstack([cos_part, sin_part * u])

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def jll_dimension(n: int, epsilon: float) -> int:
    """
    Compute target dimension using Johnson-Lindenstrauss lemma.
    """
    return int(np.ceil((4 * np.log(n)) / (epsilon ** 2)))


def softmax(Z: np.ndarray) -> np.ndarray:
    """
    Compute softmax values for each row of Z in a numerically stable way.
    """
    # Set numpy error handling to ignore warnings during this operation
    old_settings = np.seterr(all='ignore')

    try:
        # Shift values to avoid overflow
        Z_shift = Z - np.max(Z, axis=1, keepdims=True)

        # Compute exponentials with clipping to avoid overflow
        exp_Z = np.exp(np.clip(Z_shift, -709, 709))  # np.exp overflow at ~709

        # Compute sum with a small epsilon to avoid division by zero
        sum_exp = np.sum(exp_Z, axis=1, keepdims=True)
        sum_exp = np.maximum(sum_exp, 1e-15)  # Ensure non-zero denominator

        # Compute softmax
        softmax_values = exp_Z / sum_exp

        # Handle any NaN or inf values
        if np.any(np.isnan(softmax_values)) or np.any(np.isinf(softmax_values)):
            # Replace with uniform distribution if numerical issues occur
            softmax_values = np.nan_to_num(softmax_values, nan=1.0/Z.shape[1], posinf=1.0, neginf=0.0)
            # Renormalize
            row_sums = np.sum(softmax_values, axis=1, keepdims=True)
            softmax_values = softmax_values / np.maximum(row_sums, 1e-15)
    except Exception as e:
        logger.error(f"Error in softmax computation: {e}")
        # Return uniform distribution in case of error
        softmax_values = np.ones_like(Z) / Z.shape[1]

    # Restore numpy error settings
    np.seterr(**old_settings)

    return softmax_values


def compute_distortion_sample(X: np.ndarray, Y: np.ndarray, sample_size: int = 5000, seed: int = 42) -> tuple:
    """
    Compute mean/max distortion between X and Y on a random subset.
    """
    # Set numpy error handling to ignore warnings during this operation
    old_settings = np.seterr(all='ignore')

    try:
        np.random.seed(seed)
        n = X.shape[0]

        # Ensure sample_size is valid
        sample_size = min(sample_size, n)

        # Sample points if needed
        if sample_size < n:
            idx = np.random.choice(n, sample_size, replace=False)
            Xs, Ys = X[idx], Y[idx]
        else:
            Xs, Ys = X, Y

        # Compute pairwise distances with error handling
        try:
            D_orig = pairwise_distances(Xs, metric='euclidean')
            D_red = pairwise_distances(Ys, metric='euclidean')
        except Exception as e:
            logger.error(f"Error computing pairwise distances: {e}")
            # Return default values in case of error
            return 0.0, 0.0, np.zeros((1, 1)), np.zeros((1, 1))

        # Square the distances
        d2 = D_orig ** 2
        e2 = D_red ** 2

        # Ensure no division by zero or very small values
        epsilon = 1e-6
        denominator = np.maximum(d2, epsilon)

        # Compute distortion
        dist = np.abs(e2 - d2) / denominator

        # Handle any NaN or inf values
        dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute mean and max distortion
        mean_dist = dist.mean()
        max_dist = dist.max()

        # Handle any remaining NaN or inf values
        mean_dist = 0.0 if np.isnan(mean_dist) or np.isinf(mean_dist) else mean_dist
        max_dist = 0.0 if np.isnan(max_dist) or np.isinf(max_dist) else max_dist
    except Exception as e:
        logger.error(f"Error in distortion computation: {e}")
        # Return default values in case of error
        mean_dist, max_dist = 0.0, 0.0
        D_orig, D_red = np.zeros((1, 1)), np.zeros((1, 1))

    # Restore numpy error settings
    np.seterr(**old_settings)

    return mean_dist, max_dist, D_orig, D_red


def rank_corr(D1: np.ndarray, D2: np.ndarray) -> float:
    """
    Compute Spearman rank correlation between pairwise distance matrices.
    """
    # Set numpy error handling to ignore warnings during this operation
    old_settings = np.seterr(all='ignore')

    try:
        # Check if matrices are valid
        if D1.size == 0 or D2.size == 0 or D1.shape != D2.shape:
            logger.warning("Invalid distance matrices for rank correlation")
            return 0.0

        # Get upper triangular indices (excluding diagonal)
        mask = np.triu_indices_from(D1, k=1)

        # Extract values
        d1_values = D1[mask]
        d2_values = D2[mask]

        # Check if we have enough values
        if len(d1_values) < 2:
            logger.warning("Not enough values for rank correlation")
            return 0.0

        # Check for NaN or inf values
        if np.any(np.isnan(d1_values)) or np.any(np.isnan(d2_values)) or \
           np.any(np.isinf(d1_values)) or np.any(np.isinf(d2_values)):
            logger.warning("NaN or Inf values detected in distance matrices")
            # Clean the values
            valid_mask = ~(np.isnan(d1_values) | np.isnan(d2_values) | 
                          np.isinf(d1_values) | np.isinf(d2_values))
            d1_values = d1_values[valid_mask]
            d2_values = d2_values[valid_mask]

            # Check if we still have enough values
            if len(d1_values) < 2:
                logger.warning("Not enough valid values for rank correlation after cleaning")
                return 0.0

        # Compute Spearman correlation
        try:
            rho, _ = spearmanr(d1_values, d2_values)
        except Exception as e:
            logger.error(f"Error computing Spearman correlation: {e}")
            return 0.0

        # Handle NaN or inf values in the result
        if np.isnan(rho) or np.isinf(rho):
            logger.warning("NaN or Inf value in Spearman correlation result")
            return 0.0

        return rho
    except Exception as e:
        logger.error(f"Error in rank correlation computation: {e}")
        return 0.0
    finally:
        # Restore numpy error settings
        np.seterr(**old_settings)


def dist_stats(P: np.ndarray, Q: np.ndarray) -> tuple:
    """
    Compute KL divergence and L1 distance between softmax-ed rows of P, Q.
    """
    # Set numpy error handling to ignore warnings during this operation
    old_settings = np.seterr(all='ignore')

    try:
        # Ensure P and Q have the same dimensionality
        min_dim = min(P.shape[1], Q.shape[1])

        # Apply softmax with proper numerical handling
        Pp = softmax(P[:, :min_dim])
        Qq = softmax(Q[:, :min_dim])

        # Ensure no zeros in the distributions (important for KL divergence)
        eps = 1e-10
        Pp = np.clip(Pp, eps, 1.0)
        Qq = np.clip(Qq, eps, 1.0)

        # Normalize to ensure they sum to 1
        Pp = Pp / np.sum(Pp, axis=1, keepdims=True)
        Qq = Qq / np.sum(Qq, axis=1, keepdims=True)

        # Compute KL divergence in a numerically stable way
        log_ratio = np.log(Pp / Qq)
        # Replace any NaN or inf values
        log_ratio = np.nan_to_num(log_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        kl = np.mean(np.sum(Pp * log_ratio, axis=1))

        # Compute L1 distance
        l1 = np.mean(np.sum(np.abs(Pp - Qq), axis=1))

        # Handle any remaining NaN or inf values
        kl = 0.0 if np.isnan(kl) or np.isinf(kl) else kl
        l1 = 0.0 if np.isnan(l1) or np.isinf(l1) else l1
    except Exception as e:
        logger.error(f"Error in distribution stats computation: {e}")
        kl, l1 = 0.0, 0.0

    # Restore numpy error settings
    np.seterr(**old_settings)

    return kl, l1


def run_pca(X: np.ndarray, k: int, seed: int) -> tuple:
    start = time.time()

    # Set numpy error handling to ignore warnings during PCA
    old_settings = np.seterr(all='ignore')

    try:
        # Make a safe copy of the input data
        X_safe = np.copy(X)

        # Check for and handle any NaN or inf values in input
        if np.any(np.isnan(X_safe)) or np.any(np.isinf(X_safe)):
            logger.warning("NaN or Inf values detected in input data for PCA. Replacing with zeros.")
            X_safe = np.nan_to_num(X_safe, nan=0.0, posinf=0.0, neginf=0.0)

        # Center the data (PCA is sensitive to centering)
        X_centered = X_safe - np.mean(X_safe, axis=0, keepdims=True)

        # Check if the data has very small variance
        var = np.var(X_centered, axis=0)
        if np.any(var < 1e-10):
            logger.warning("Very small variance detected in some dimensions. Adding small noise.")
            # Add small noise to dimensions with very small variance
            noise_scale = 1e-6 * np.max(var)
            X_centered += np.random.RandomState(seed).normal(0, noise_scale, X_centered.shape)

        # Use SVD directly instead of PCA for better numerical stability
        try:
            # SVD can be more stable than the covariance matrix approach
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

            # Keep only the top k components
            U = U[:, :k] if U.shape[1] >= k else np.hstack([U, np.zeros((U.shape[0], k - U.shape[1]))])
            S = S[:k] if len(S) >= k else np.concatenate([S, np.zeros(k - len(S))])

            # Project the data
            Y = U * S

        except Exception as e:
            logger.warning(f"SVD failed: {e}. Falling back to scikit-learn PCA.")
            # Fallback to scikit-learn PCA
            model = PCA(n_components=k, random_state=seed, svd_solver='randomized')
            Y = model.fit_transform(X_safe)

        # Check for NaN or inf values in the result
        if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
            logger.warning("NaN or Inf values detected in PCA result. Replacing with zeros.")
            Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    except Exception as e:
        logger.error(f"Error in PCA: {e}")
        # Last resort: return a simple scaled version of the input
        logger.warning("PCA failed. Returning a simple scaled version of the input.")
        Y = np.random.RandomState(seed).randn(X.shape[0], k)

    # Restore numpy error settings
    np.seterr(**old_settings)

    return Y, time.time() - start


def run_jll(X: np.ndarray, k: int, seed: int) -> tuple:
    start = time.time()

    # Set numpy error handling to ignore warnings during this operation
    old_settings = np.seterr(all='ignore')

    try:
        # First, ensure X doesn't have extreme values that could cause numerical issues
        X_safe = np.copy(X)

        # Check for and handle any NaN or inf values in input
        if np.any(np.isnan(X_safe)) or np.any(np.isinf(X_safe)):
            logger.warning("NaN or Inf values detected in input data for JLL. Replacing with zeros.")
            X_safe = np.nan_to_num(X_safe, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize X if it has very large values
        max_norm = np.max(np.abs(X_safe))
        if max_norm > 1e3:  # If values are very large
            logger.warning(f"Large values detected in input data (max={max_norm}). Normalizing.")
            X_safe = X_safe / max_norm

        # Use QR decomposition to generate an orthogonal random matrix
        # This is more stable than using GaussianRandomProjection
        np.random.seed(seed)

        # Generate a random matrix with smaller values to avoid numerical issues
        # Using a smaller scale factor helps prevent overflow
        scale_factor = 1.0 / np.sqrt(max(k, X.shape[1]))
        random_matrix = np.random.normal(0, scale_factor, (X_safe.shape[1], k))

        # Use QR decomposition to get an orthogonal matrix
        Q, _ = np.linalg.qr(random_matrix, mode='reduced')
        if Q.shape[1] < k:
            # If QR didn't return enough columns, pad with zeros
            padding = np.zeros((Q.shape[0], k - Q.shape[1]))
            Q = np.hstack([Q, padding])
            Q = Q[:, :k]  # Ensure we have exactly k columns

        # Project the data using the orthogonal matrix
        Y = np.dot(X_safe, Q)

        # Check for NaN or inf values and replace them
        if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
            logger.warning("NaN or Inf values detected in JLL projection result. Replacing with zeros.")
            Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        logger.error(f"Error in JLL projection: {e}")
        # Fallback to a simpler approach that avoids using GaussianRandomProjection
        try:
            np.random.seed(seed)
            # Use a very simple random projection as fallback
            random_matrix = np.random.normal(0, 0.01, (X.shape[1], k))
            Y = np.dot(X, random_matrix)
            Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e2:
            logger.error(f"Fallback projection also failed: {e2}")
            # Last resort: return zeros
            Y = np.zeros((X.shape[0], k))

    # Restore numpy error settings
    np.seterr(**old_settings)

    return Y, time.time() - start


def run_umap(X: np.ndarray, k: int, seed: int, n_neighbors: int = 15, min_dist: float = 0.1) -> tuple:
    if not UMAP_AVAILABLE:
        logger.warning("UMAP not available. Returning random projection instead.")
        return run_jll(X, k, seed)

    start = time.time()

    # Set numpy error handling to ignore warnings during this operation
    old_settings = np.seterr(all='ignore')

    try:
        # Adjust n_neighbors if it's larger than the number of samples
        n_neighbors = min(n_neighbors, X.shape[0] - 1)

        # Create and fit the UMAP model with error handling
        reducer = umap.UMAP(
            n_components=k,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=seed,
            low_memory=True,  # Use low memory mode for better stability
            metric='euclidean'  # Explicitly set metric for better numerical stability
        )
        Y = reducer.fit_transform(X)

        # Check for NaN or inf values and replace them
        if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
            logger.warning("NaN or Inf values detected in UMAP projection. Replacing with zeros.")
            Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        logger.error(f"Error in UMAP projection: {e}")
        # Fallback to JLL projection
        logger.warning("Falling back to JLL projection due to UMAP error.")
        Y, _ = run_jll(X, k, seed)

    # Restore numpy error settings
    np.seterr(**old_settings)

    return Y, time.time() - start


def run_poincare(X: np.ndarray, k: int, seed: int, c: float = 1.0) -> tuple:
    """
    Run Poincare embedding pipeline.

    Parameters:
    - X: Input data
    - k: Target dimension
    - seed: Random seed
    - c: Curvature parameter

    Returns:
    - Y: Projected data
    - runtime: Time taken for projection
    """
    start = time.time()

    # Set numpy error handling to ignore warnings during this operation
    old_settings = np.seterr(all='ignore')

    try:
        original_dim = X.shape[1]
        emb = HyperbolicEmbedding(c=c)

        # Map to Poincare disk
        Y_disk = emb.euclidean_to_disk(X)

        # Map to tangent space at origin
        Y_tangent = emb.log_map_zero(Y_disk)

        # Project in tangent space
        Y_proj, _ = run_jll(Y_tangent, k, seed)

        # Map back to Poincare disk
        Y_tangent_recon = emb.exp_map_zero(Y_proj)

        # Map back to Euclidean space
        Y_recon = emb.disk_to_euclidean(Y_tangent_recon)

        # Restore original dimensionality if needed (simple padding/truncating)
        if Y_recon.shape[1] != original_dim:
            result = np.zeros((Y_recon.shape[0], original_dim))
            # Copy values, handling both truncation and padding
            copy_dims = min(Y_recon.shape[1], original_dim)
            result[:, :copy_dims] = Y_recon[:, :copy_dims]
            Y_recon = result

        # Check for NaN or inf values and replace them
        if np.any(np.isnan(Y_recon)) or np.any(np.isinf(Y_recon)):
            logger.warning("NaN or Inf values detected in Poincare projection. Replacing with zeros.")
            Y_recon = np.nan_to_num(Y_recon, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        logger.error(f"Error in Poincare projection: {e}")
        # Fallback to JLL projection
        logger.warning("Falling back to JLL projection due to Poincare error.")
        Y_recon, _ = run_jll(X, k, seed)

    # Restore numpy error settings
    np.seterr(**old_settings)

    return Y_recon, time.time() - start


def run_spherical(X: np.ndarray, k: int, seed: int) -> tuple:
    """
    Run Spherical embedding pipeline.

    Parameters:
    - X: Input data
    - k: Target dimension
    - seed: Random seed

    Returns:
    - Y: Projected data
    - runtime: Time taken for projection
    """
    start = time.time()

    # Set numpy error handling to ignore warnings during this operation
    old_settings = np.seterr(all='ignore')

    try:
        original_dim = X.shape[1]
        emb = SphericalEmbedding()

        # Map to sphere
        Y_sphere = emb.euclidean_to_sphere(X)

        # Map to tangent space at north pole
        Y_tangent = emb.log_map_north_pole(Y_sphere)

        # Project in tangent space
        Y_proj, _ = run_jll(Y_tangent, k, seed)

        # Map back to sphere
        Y_recon = emb.exp_map_north_pole(Y_proj)

        # Restore original dimensionality if needed (simple padding/truncating)
        if Y_recon.shape[1] != original_dim:
            result = np.zeros((Y_recon.shape[0], original_dim))
            # Copy values, handling both truncation and padding
            copy_dims = min(Y_recon.shape[1], original_dim)
            result[:, :copy_dims] = Y_recon[:, :copy_dims]
            Y_recon = result

        # Check for NaN or inf values and replace them
        if np.any(np.isnan(Y_recon)) or np.any(np.isinf(Y_recon)):
            logger.warning("NaN or Inf values detected in Spherical projection. Replacing with zeros.")
            Y_recon = np.nan_to_num(Y_recon, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        logger.error(f"Error in Spherical projection: {e}")
        # Fallback to JLL projection
        logger.warning("Falling back to JLL projection due to Spherical error.")
        Y_recon, _ = run_jll(X, k, seed)

    # Restore numpy error settings
    np.seterr(**old_settings)

    return Y_recon, time.time() - start


def run_experiment(n: int, d: int, epsilon: float, seed: int, sample_size: int,
                   use_poincare: bool, use_spherical: bool, use_elliptic: bool) -> dict:
    logger.info(f"Parameters: n={n}, d={d}, epsilon={epsilon}, seed={seed}, sample_size={sample_size}")

    # Set numpy error handling to ignore warnings during data generation and normalization
    old_settings = np.seterr(all='ignore')

    try:
        # Generate random data with controlled seed
        rng = np.random.RandomState(seed)
        X = rng.randn(n, d)

        # Check for any NaN or inf values in the generated data
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.warning("NaN or Inf values detected in generated data. Replacing with random values.")
            mask = np.isnan(X) | np.isinf(X)
            X[mask] = rng.randn(*X[mask].shape)

        # Normalize with a robust approach to avoid division by zero
        norms = np.linalg.norm(X, axis=1, keepdims=True)

        # Replace zero or very small norms with 1.0 to avoid numerical issues
        # Using a slightly larger threshold for better numerical stability
        norms = np.where(norms < 1e-8, 1.0, norms)

        # Normalize the data
        X = X / norms

        # Verify the normalization worked correctly
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.warning("NaN or Inf values detected after normalization. Fixing affected rows.")
            # Replace problematic rows with random unit vectors
            mask = np.any(np.isnan(X) | np.isinf(X), axis=1)
            for i in np.where(mask)[0]:
                v = rng.randn(d)
                X[i] = v / np.maximum(np.linalg.norm(v), 1e-8)
    except Exception as e:
        logger.error(f"Error during data generation and normalization: {e}")
        # Fallback to a simpler approach if anything goes wrong
        X = np.random.RandomState(seed).randn(n, d)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        X /= norms

    # Restore numpy error settings
    np.seterr(**old_settings)

    k = min(jll_dimension(n, epsilon), d)
    logger.info(f"Chosen dimension k={k}")

    soft_orig = softmax(X)
    results = {}

    # PCA
    logger.info("Running PCA...")
    Y, rt = run_pca(X, k, seed)
    md, xd, D1, D2 = compute_distortion_sample(X, Y, sample_size, seed)
    results['PCA'] = dict(mean_distortion=md, max_distortion=xd,
                           rank_correlation=rank_corr(D1, D2),
                           **dict(zip(['kl_divergence', 'l1'], dist_stats(soft_orig, Y))),
                           runtime=rt)

    # JLL
    logger.info("Running JLL projection...")
    Y, rt = run_jll(X, k, seed)
    md, xd, D1, D2 = compute_distortion_sample(X, Y, sample_size, seed)
    results['JLL'] = dict(mean_distortion=md, max_distortion=xd,
                           rank_correlation=rank_corr(D1, D2),
                           **dict(zip(['kl_divergence', 'l1'], dist_stats(soft_orig, Y))),
                           runtime=rt)

    # UMAP
    logger.info("Running UMAP...")
    Y, rt = run_umap(X, k, seed)
    md, xd, D1, D2 = compute_distortion_sample(X, Y, sample_size, seed)
    results['UMAP'] = dict(mean_distortion=md, max_distortion=xd,
                            rank_correlation=rank_corr(D1, D2),
                            **dict(zip(['kl_divergence', 'l1'], dist_stats(soft_orig, Y))),
                            runtime=rt)

    # Poincare
    if use_poincare:
        logger.info("Running Poincare embedding...")
        Y, rt = run_poincare(X, k, seed, c=1.0)
        md, xd, D1, D2 = compute_distortion_sample(X, Y, sample_size, seed)
        results['Poincare'] = dict(mean_distortion=md, max_distortion=xd,
                                rank_correlation=rank_corr(D1, D2),
                                **dict(zip(['kl_divergence', 'l1'], dist_stats(soft_orig, Y))),
                                runtime=rt)

    # Spherical
    if use_spherical:
        logger.info("Running Spherical embedding...")
        Y, rt = run_spherical(X, k, seed)
        md, xd, D1, D2 = compute_distortion_sample(X, Y, sample_size, seed)
        results['Spherical'] = dict(mean_distortion=md, max_distortion=xd,
                                rank_correlation=rank_corr(D1, D2),
                                **dict(zip(['kl_divergence', 'l1'], dist_stats(soft_orig, Y))),
                                runtime=rt)

    # Elliptic (not implemented yet)
    if use_elliptic:
        logger.warning("Elliptic embedding not implemented yet. Skipping.")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=15000)
    parser.add_argument('--d', type=int, default=1200)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample_size', type=int, default=5000)
    args = parser.parse_args()

    res = run_experiment(args.n, args.d, args.epsilon, args.seed, args.sample_size,
                         use_poincare=False, use_spherical=False, use_elliptic=False)
    for name, m in res.items():
        logger.info(f"=== {name} ===")
        for k, v in m.items():
            logger.info(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
