import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances

from .projection import jll_dimension, generate_orthogonal_basis, project_data


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def compute_distortion(X, Y, epsilon=1e-9):
    D_original = pairwise_distances(X)
    D_reduced = pairwise_distances(Y)
    D_orig_sq = D_original ** 2
    D_red_sq = D_reduced ** 2
    distortion = np.abs(D_red_sq - D_orig_sq) / (D_orig_sq + epsilon)
    return distortion.mean(), distortion.max(), D_original, D_reduced


def evaluate_rank_correlation(D_orig, D_reduced):
    mask = np.triu_indices_from(D_orig, k=1)
    rho, _ = spearmanr(D_orig[mask], D_reduced[mask])
    return 0.0 if np.isnan(rho) else rho


def kl_divergence(p, q, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q), axis=1)


def distribution_stats(p, q):
    kl = np.mean(kl_divergence(p, q))
    l1 = np.mean(np.sum(np.abs(p - q), axis=1))
    return kl, l1


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


def run_jll(X, k, seed=42):
    basis = generate_orthogonal_basis(X.shape[1], k, seed=seed)
    Y = project_data(X, basis)
    return Y


def run_poincare_pipeline(X, k, c=1.0, seed=42):
    emb = HyperbolicEmbedding(c=c)
    Y_disk = emb.euclidean_to_disk(X)
    Y_tangent = emb.log_map_zero(Y_disk)
    Y_proj = run_jll(Y_tangent, k, seed=seed)
    Y_tangent_recon = emb.exp_map_zero(Y_proj)
    Y_recon = emb.disk_to_euclidean(Y_tangent_recon)
    return Y_recon


def run_spherical_pipeline(X, k, seed=42):
    emb = SphericalEmbedding()
    Y_sphere = emb.euclidean_to_sphere(X)
    Y_tangent = emb.log_map_north_pole(Y_sphere)
    Y_proj = run_jll(Y_tangent, k, seed=seed)
    Y_recon = emb.exp_map_north_pole(Y_proj)
    return Y_recon


def run_experiment(n=15000, d=1200, epsilon=0.2, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, d)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    k = jll_dimension(n, epsilon)
    k = min(k, d)
    softmax_original = softmax(X)
    results = {}

    # PCA baseline
    from sklearn.decomposition import PCA
    pca = PCA(n_components=k, random_state=seed)
    Y_pca = pca.fit_transform(X)
    md, mx, D_o, D_r = compute_distortion(X, Y_pca)
    results['PCA'] = {
        'mean_distortion': md,
        'max_distortion': mx,
        'rank_correlation': evaluate_rank_correlation(D_o, D_r),
        'kl_divergence': distribution_stats(softmax_original, softmax(Y_pca))[0],
    }

    # JLL baseline
    Y_jll = run_jll(X, k, seed=seed)
    md, mx, D_o, D_r = compute_distortion(X, Y_jll)
    results['JLL'] = {
        'mean_distortion': md,
        'max_distortion': mx,
        'rank_correlation': evaluate_rank_correlation(D_o, D_r),
        'kl_divergence': distribution_stats(softmax_original, softmax(Y_jll))[0],
    }

    # Poincare pipeline
    Y_poincare = run_poincare_pipeline(X, k, c=1.0, seed=seed)
    md, mx, D_o, D_r = compute_distortion(X, Y_poincare)
    results['Poincare'] = {
        'mean_distortion': md,
        'max_distortion': mx,
        'rank_correlation': evaluate_rank_correlation(D_o, D_r),
        'kl_divergence': distribution_stats(softmax_original, softmax(Y_poincare))[0],
    }

    # Spherical pipeline
    Y_spherical = run_spherical_pipeline(X, k, seed=seed)
    md, mx, D_o, D_r = compute_distortion(X, Y_spherical)
    results['Spherical'] = {
        'mean_distortion': md,
        'max_distortion': mx,
        'rank_correlation': evaluate_rank_correlation(D_o, D_r),
        'kl_divergence': distribution_stats(softmax_original, softmax(Y_spherical))[0],
    }

    return results
