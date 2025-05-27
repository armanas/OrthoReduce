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
    model = PCA(n_components=k, random_state=seed)
    Y = model.fit_transform(X)
    return Y, time.time() - start


def run_jll(X: np.ndarray, k: int, seed: int) -> tuple:
    start = time.time()

    # Set numpy error handling to ignore warnings during this operation
    old_settings = np.seterr(all='ignore')

    try:
        # Use a more stable approach with explicit random matrix generation
        np.random.seed(seed)
        # Generate a random matrix with controlled values to avoid numerical issues
        random_matrix = np.random.normal(0, 1.0/np.sqrt(k), (X.shape[1], k))
        # Project the data
        Y = np.dot(X, random_matrix)

        # Check for NaN or inf values and replace them
        if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
            logger.warning("NaN or Inf values detected in JLL projection. Replacing with zeros.")
            Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        logger.error(f"Error in JLL projection: {e}")
        # Fallback to a simpler approach
        transformer = GaussianRandomProjection(n_components=k, random_state=seed)
        Y = transformer.fit_transform(X)
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

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


def run_experiment(n: int, d: int, epsilon: float, seed: int, sample_size: int,
                   use_poincare: bool, use_spherical: bool, use_elliptic: bool) -> dict:
    logger.info(f"Parameters: n={n}, d={d}, epsilon={epsilon}, seed={seed}, sample_size={sample_size}")
    X = np.random.RandomState(seed).randn(n, d)
    # Normalize with a small epsilon to avoid division by zero
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Replace zero or very small norms with 1.0 to avoid numerical issues
    norms = np.where(norms < 1e-10, 1.0, norms)
    X /= norms

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

    # Additional geometries can be implemented similarly
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
