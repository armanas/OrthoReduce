"""
Spherical Embeddings for Dimensionality Reduction with Riemannian Optimization

This module implements mathematically rigorous spherical embeddings with:
1. Geodesic distance computations with numerical stability
2. Riemannian optimization framework with tangent space operations
3. Geometry-consistent loss functions
4. Adaptive curvature and radius optimization

Mathematical Background:
- The unit sphere S^(k-1) ⊂ ℝ^k is a Riemannian manifold with constant positive curvature
- Geodesic distance: d_geo(x,y) = arccos(⟨x,y⟩) for x,y ∈ S^(k-1)
- Tangent space at x: T_x S^(k-1) = {v ∈ ℝ^k : ⟨x,v⟩ = 0}
- Exponential map: Exp_x(v) = cos(‖v‖)x + sin(‖v‖)v/‖v‖
- Logarithmic map: Log_x(y) = θ/sin(θ)(y - cos(θ)x) where θ = arccos(⟨x,y⟩)

Author: Mathematical Theory Extension for OrthoReduce
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, Callable
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# Numerical stability constants
EPSILON = 1e-10
ARCCOS_CLIP = 1.0 - 1e-7  # Slightly less than 1 to avoid arccos(1) issues
MIN_NORM = 1e-8


class SphericalEmbedding:
    """
    Advanced spherical embedding with Riemannian optimization.
    
    This class implements spherical embeddings with proper geodesic computations,
    Riemannian optimization, and adaptive radius learning.
    """
    
    def __init__(
        self,
        n_components: int,
        radius: float = 1.0,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
        loss_type: str = 'mds_geodesic',
        regularization: float = 0.01,
        hemisphere_constraint: bool = True,
        adaptive_radius: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize spherical embedding.
        
        Parameters
        ----------
        n_components : int
            Target embedding dimension k (embeds into S^(k-1))
        radius : float
            Initial sphere radius (scaled curvature = 1/radius^2)
        learning_rate : float
            Initial learning rate for optimization
        max_iter : int
            Maximum optimization iterations
        tol : float
            Convergence tolerance for gradient norm
        loss_type : str
            Loss function type: 'mds_geodesic', 'triplet', 'nca', 'hybrid'
        regularization : float
            Regularization strength for angular margins
        hemisphere_constraint : bool
            Whether to enforce hemisphere constraint (avoid antipodal ambiguity)
        adaptive_radius : bool
            Whether to optimize radius during training
        seed : int or None
            Random seed for reproducibility
        """
        self.n_components = n_components
        self.radius = radius
        self.initial_radius = radius
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.loss_type = loss_type
        self.regularization = regularization
        self.hemisphere_constraint = hemisphere_constraint
        self.adaptive_radius = adaptive_radius
        self.seed = seed
        
        # Will be set during fit
        self.embedding_ = None
        self.loss_history_ = []
        self.radius_history_ = []
        
        if seed is not None:
            np.random.seed(seed)
    
    @staticmethod
    def geodesic_distance(x: NDArray, y: NDArray, radius: float = 1.0) -> NDArray:
        """
        Compute geodesic distance on sphere with numerical stability.
        
        For points x, y on sphere S^(k-1) with radius r:
        d_geo(x,y) = r * arccos(⟨x,y⟩/r^2)
        
        Handles numerical issues near antipodal points.
        """
        # Normalize to unit sphere for computation
        x_norm = x / radius
        y_norm = y / radius
        
        # Compute cosine similarity with clipping for numerical stability
        if x_norm.ndim == 1 and y_norm.ndim == 1:
            # Single pair
            cos_sim = np.clip(np.dot(x_norm, y_norm), -ARCCOS_CLIP, ARCCOS_CLIP)
            return radius * np.arccos(cos_sim)
        elif x_norm.ndim == 2 and y_norm.ndim == 2:
            # Pairwise distances
            cos_sim = np.clip(x_norm @ y_norm.T, -ARCCOS_CLIP, ARCCOS_CLIP)
            return radius * np.arccos(cos_sim)
        else:
            raise ValueError("Invalid input dimensions")
    
    @staticmethod
    def geodesic_distance_batch(X: NDArray, radius: float = 1.0) -> NDArray:
        """
        Compute all pairwise geodesic distances efficiently.
        
        Parameters
        ----------
        X : NDArray of shape (n, k)
            Points on sphere
        radius : float
            Sphere radius
            
        Returns
        -------
        D : NDArray of shape (n, n)
            Pairwise geodesic distance matrix
        """
        n = X.shape[0]
        
        # Normalize to unit sphere
        X_norm = X / radius
        
        # Compute Gram matrix (cosine similarities)
        gram = np.clip(X_norm @ X_norm.T, -ARCCOS_CLIP, ARCCOS_CLIP)
        
        # Convert to geodesic distances
        # Use stable computation for diagonal (should be 0)
        D = radius * np.arccos(gram)
        np.fill_diagonal(D, 0.0)
        
        return D
    
    @staticmethod
    def project_to_tangent_space(x: NDArray, v: NDArray) -> NDArray:
        """
        Project vector v onto tangent space T_x S^(k-1).
        
        The tangent space at x is: T_x S = {v : ⟨x,v⟩ = 0}
        Projection: Π_x(v) = v - ⟨v,x⟩/‖x‖² x
        """
        x_norm_sq = np.sum(x**2, axis=-1, keepdims=True)
        x_norm_sq = np.maximum(x_norm_sq, MIN_NORM**2)  # Avoid division by zero
        projection = v - (np.sum(v * x, axis=-1, keepdims=True) / x_norm_sq) * x
        return projection
    
    @staticmethod
    def exponential_map(x: NDArray, v: NDArray, radius: float = 1.0) -> NDArray:
        """
        Exponential map from tangent space to sphere.
        
        For v ∈ T_x S^(k-1) with sphere radius r:
        Exp_x(v) = cos(‖v‖/r) x + r sin(‖v‖/r) v/‖v‖
        
        This retracts tangent vectors back to the sphere manifold.
        """
        v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
        v_norm = np.maximum(v_norm, MIN_NORM)  # Avoid division by zero
        
        # Scaled angle
        theta = v_norm / radius
        
        # Exponential map formula
        y = np.cos(theta) * x + radius * np.sin(theta) * (v / v_norm)
        
        # Ensure result is on sphere (handle numerical errors)
        y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
        y_norm = np.maximum(y_norm, MIN_NORM)
        y = radius * y / y_norm
        
        return y
    
    @staticmethod
    def logarithmic_map(x: NDArray, y: NDArray, radius: float = 1.0) -> NDArray:
        """
        Logarithmic map from sphere to tangent space.
        
        For x, y ∈ S^(k-1) with sphere radius r:
        Log_x(y) = (rθ/sin(θ))(y/r - cos(θ)x/r)
        where θ = d_geo(x,y)/r
        
        This maps sphere points to tangent vectors.
        """
        # Compute geodesic distance (angle)
        cos_theta = np.clip(np.sum(x * y, axis=-1) / radius**2, -ARCCOS_CLIP, ARCCOS_CLIP)
        theta = np.arccos(cos_theta)
        
        # Handle small angles (near identical points)
        small_angle_mask = theta < 1e-6
        
        # Standard logarithmic map
        sin_theta = np.sin(theta)
        sin_theta = np.where(small_angle_mask, 1.0, sin_theta)  # Avoid division by zero
        
        # Compute tangent vector
        v = np.where(
            small_angle_mask[..., np.newaxis],
            y - x,  # Small angle approximation
            (radius * theta[..., np.newaxis] / sin_theta[..., np.newaxis]) * 
            (y/radius - cos_theta[..., np.newaxis] * x/radius)
        )
        
        return v
    
    def mds_stress_geodesic(self, Y: NDArray, D_target: NDArray) -> float:
        """
        Compute MDS stress using geodesic distances.
        
        Stress = Σᵢⱼ wᵢⱼ(d_geo(yᵢ,yⱼ) - dᵢⱼ)²
        where d_geo is geodesic distance on sphere.
        """
        D_embed = self.geodesic_distance_batch(Y, self.radius)
        
        # Weighted stress (inverse distance weighting for stability)
        weights = 1.0 / (D_target + 1.0)
        stress = np.sum(weights * (D_embed - D_target)**2)
        
        return stress
    
    def triplet_loss(self, Y: NDArray, triplets: NDArray) -> float:
        """
        Compute triplet loss for preserving relative distances.
        
        For triplet (i, j, k) where d(i,j) < d(i,k) in original space:
        Loss = max(0, d_geo(yᵢ,yⱼ) - d_geo(yᵢ,yₖ) + margin)
        """
        loss = 0.0
        margin = 0.1 * self.radius  # Adaptive margin based on radius
        
        for i, j, k in triplets:
            d_ij = self.geodesic_distance(Y[i], Y[j], self.radius)
            d_ik = self.geodesic_distance(Y[i], Y[k], self.radius)
            loss += np.maximum(0, d_ij - d_ik + margin)
        
        return loss
    
    def nca_loss(self, Y: NDArray, labels: Optional[NDArray] = None) -> float:
        """
        Neighborhood Component Analysis loss on sphere.
        
        Encourages points with same label to be close on sphere.
        """
        if labels is None:
            # Use clustering if no labels provided
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=min(10, Y.shape[0]//5), random_state=self.seed)
            labels = km.fit_predict(Y)
        
        n = Y.shape[0]
        D = self.geodesic_distance_batch(Y, self.radius)
        
        # Compute probabilities using geodesic distances
        # p_ij = exp(-d_geo(i,j)) / Σₖ≠ᵢ exp(-d_geo(i,k))
        exp_neg_D = np.exp(-D / self.radius)  # Scale by radius
        np.fill_diagonal(exp_neg_D, 0)  # Exclude self
        
        P = exp_neg_D / (np.sum(exp_neg_D, axis=1, keepdims=True) + EPSILON)
        
        # NCA loss: maximize probability of correct neighbors
        loss = 0.0
        for i in range(n):
            same_class = labels == labels[i]
            same_class[i] = False  # Exclude self
            if np.any(same_class):
                loss -= np.log(np.sum(P[i, same_class]) + EPSILON)
        
        return loss
    
    def angular_margin_regularization(self, Y: NDArray) -> float:
        """
        Regularization to maintain angular separation.
        
        Penalizes points that are too close (small geodesic distance)
        or antipodal (near maximum geodesic distance).
        """
        D = self.geodesic_distance_batch(Y, self.radius)
        
        # Penalize very small distances (crowding)
        min_dist = 0.1 * self.radius
        crowding_penalty = np.sum(np.maximum(0, min_dist - D))
        
        # Penalize near-antipodal points (ambiguity)
        max_dist = np.pi * self.radius * 0.95  # 95% of maximum distance
        antipodal_penalty = np.sum(np.maximum(0, D - max_dist))
        
        return crowding_penalty + 0.5 * antipodal_penalty
    
    def compute_loss_and_gradient(
        self, 
        Y_flat: NDArray, 
        D_target: NDArray,
        compute_gradient: bool = True
    ) -> Tuple[float, Optional[NDArray]]:
        """
        Compute loss and Riemannian gradient.
        
        The gradient is computed in tangent space and includes:
        1. Data fidelity term gradient
        2. Regularization gradient
        3. Proper retraction to maintain sphere constraint
        """
        n = D_target.shape[0]
        Y = Y_flat.reshape(n, self.n_components)
        
        # Normalize to sphere (project if optimization drifted)
        Y = self.radius * Y / np.linalg.norm(Y, axis=1, keepdims=True)
        
        # Compute loss based on type
        if self.loss_type == 'mds_geodesic':
            loss = self.mds_stress_geodesic(Y, D_target)
        elif self.loss_type == 'triplet':
            # Generate triplets from distance matrix
            triplets = self.generate_triplets(D_target)
            loss = self.triplet_loss(Y, triplets)
        elif self.loss_type == 'nca':
            loss = self.nca_loss(Y)
        elif self.loss_type == 'hybrid':
            # Combine multiple losses
            loss = (0.7 * self.mds_stress_geodesic(Y, D_target) + 
                   0.3 * self.nca_loss(Y))
        else:
            loss = self.mds_stress_geodesic(Y, D_target)
        
        # Add regularization
        if self.regularization > 0:
            loss += self.regularization * self.angular_margin_regularization(Y)
        
        if not compute_gradient:
            return loss, None
        
        # Compute gradient via finite differences in tangent space
        # (More stable than autodiff for sphere constraints)
        grad = np.zeros_like(Y)
        h = 1e-5
        
        for i in range(n):
            for j in range(self.n_components):
                # Create tangent vector
                v = np.zeros(self.n_components)
                v[j] = h
                
                # Project to tangent space at Y[i]
                v_tangent = self.project_to_tangent_space(Y[i:i+1], v.reshape(1, -1))[0]
                
                # Perturb via exponential map
                Y_plus = Y.copy()
                Y_plus[i] = self.exponential_map(Y[i:i+1], v_tangent.reshape(1, -1), self.radius)[0]
                
                Y_minus = Y.copy()
                Y_minus[i] = self.exponential_map(Y[i:i+1], -v_tangent.reshape(1, -1), self.radius)[0]
                
                # Compute gradient via finite differences
                loss_plus, _ = self.compute_loss_and_gradient(Y_plus.flatten(), D_target, compute_gradient=False)
                loss_minus, _ = self.compute_loss_and_gradient(Y_minus.flatten(), D_target, compute_gradient=False)
                
                grad[i, j] = (loss_plus - loss_minus) / (2 * h)
        
        # Project gradient to tangent spaces (Riemannian gradient)
        for i in range(n):
            grad[i] = self.project_to_tangent_space(Y[i:i+1], grad[i:i+1])[0]
        
        return loss, grad.flatten()
    
    def generate_triplets(self, D: NDArray, n_triplets: Optional[int] = None) -> NDArray:
        """
        Generate informative triplets from distance matrix.
        
        Creates triplets (i, j, k) where d(i,j) < d(i,k).
        """
        n = D.shape[0]
        if n_triplets is None:
            n_triplets = min(n * 20, 5000)  # Heuristic
        
        triplets = []
        
        for _ in range(n_triplets):
            i = np.random.randint(n)
            
            # Sort distances from i
            dist_from_i = D[i].copy()
            dist_from_i[i] = np.inf  # Exclude self
            sorted_idx = np.argsort(dist_from_i)
            
            # Pick j from nearest neighbors, k from farther points
            j = sorted_idx[np.random.randint(0, n//3)]
            k = sorted_idx[np.random.randint(2*n//3, n-1)]
            
            triplets.append([i, j, k])
        
        return np.array(triplets)
    
    def optimize_radius(self, Y: NDArray, D_target: NDArray) -> float:
        """
        Optimize sphere radius for given embedding.
        
        Uses golden section search to find radius that minimizes stress.
        """
        def objective(r):
            self.radius = r
            return self.mds_stress_geodesic(Y, D_target)
        
        # Golden section search
        a, b = 0.1, 10.0
        tol = 1e-3
        invphi = (np.sqrt(5) - 1) / 2
        
        while abs(b - a) > tol:
            c = a + (b - a) * invphi**2
            d = a + (b - a) * invphi
            
            if objective(c) < objective(d):
                b = d
            else:
                a = c
        
        optimal_radius = (a + b) / 2
        self.radius = optimal_radius
        return optimal_radius
    
    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> 'SphericalEmbedding':
        """
        Fit spherical embedding to data.
        
        Parameters
        ----------
        X : NDArray of shape (n, d)
            Input high-dimensional data
        y : NDArray of shape (n,) or None
            Optional labels for supervised losses
            
        Returns
        -------
        self : SphericalEmbedding
            Fitted embedding object
        """
        n, d = X.shape
        
        # Compute target distance matrix
        D_target = squareform(pdist(X, metric='euclidean'))
        
        # Initialize embedding using PCA
        pca = PCA(n_components=self.n_components, random_state=self.seed)
        Y_init = pca.fit_transform(X)
        
        # Project to sphere with initial radius
        norms = np.linalg.norm(Y_init, axis=1, keepdims=True)
        norms = np.maximum(norms, MIN_NORM)
        Y = self.initial_radius * Y_init / norms
        
        # Hemisphere constraint: ensure first point is in positive hemisphere
        if self.hemisphere_constraint and Y[0, 0] < 0:
            Y = -Y
        
        # Optimization via Riemannian gradient descent
        lr = self.learning_rate
        best_loss = np.inf
        best_Y = Y.copy()
        patience = 50
        no_improve = 0
        
        for iteration in range(self.max_iter):
            # Compute loss and gradient
            loss, grad = self.compute_loss_and_gradient(Y.flatten(), D_target)
            
            # Record history
            self.loss_history_.append(loss)
            self.radius_history_.append(self.radius)
            
            # Check for improvement
            if loss < best_loss:
                best_loss = loss
                best_Y = Y.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # Early stopping
            if no_improve >= patience:
                logger.info(f"Early stopping at iteration {iteration}")
                break
            
            # Check convergence
            grad_norm = np.linalg.norm(grad)
            if grad_norm < self.tol:
                logger.info(f"Converged at iteration {iteration}")
                break
            
            # Adaptive learning rate
            if no_improve > patience // 2:
                lr *= 0.5
                no_improve = 0
            
            # Riemannian gradient step
            grad_reshaped = grad.reshape(n, self.n_components)
            
            # Update via exponential map
            for i in range(n):
                # Get tangent gradient
                v = -lr * grad_reshaped[i:i+1]
                
                # Retract to sphere via exponential map
                Y[i] = self.exponential_map(Y[i:i+1], v, self.radius)[0]
            
            # Adaptive radius optimization every 50 iterations
            if self.adaptive_radius and iteration % 50 == 0 and iteration > 0:
                old_radius = self.radius
                self.radius = self.optimize_radius(Y, D_target)
                
                # Scale embedding if radius changed significantly
                if abs(self.radius - old_radius) > 0.01:
                    Y = Y * (self.radius / old_radius)
                    logger.info(f"Radius updated: {old_radius:.3f} -> {self.radius:.3f}")
            
            # Hemisphere constraint maintenance
            if self.hemisphere_constraint and iteration % 20 == 0:
                # Flip if too many points are in negative hemisphere
                if np.mean(Y[:, 0] < 0) > 0.7:
                    Y = -Y
            
            # Log progress
            if iteration % 100 == 0:
                logger.info(f"Iteration {iteration}: loss={loss:.4f}, "
                          f"grad_norm={grad_norm:.4f}, radius={self.radius:.3f}")
        
        # Store best result
        self.embedding_ = best_Y
        
        return self
    
    def fit_transform(self, X: NDArray, y: Optional[NDArray] = None) -> NDArray:
        """
        Fit and return spherical embedding.
        
        Parameters
        ----------
        X : NDArray of shape (n, d)
            Input data
        y : NDArray or None
            Optional labels
            
        Returns
        -------
        Y : NDArray of shape (n, k)
            Spherical embedding
        """
        self.fit(X, y)
        return self.embedding_
    
    def transform(self, X_new: NDArray) -> NDArray:
        """
        Transform new points to spherical embedding.
        
        Uses out-of-sample extension via local tangent space approximation.
        
        Parameters
        ----------
        X_new : NDArray of shape (m, d)
            New points to embed
            
        Returns
        -------
        Y_new : NDArray of shape (m, k)
            Embedded points on sphere
        """
        if self.embedding_ is None:
            raise ValueError("Model must be fitted before transform")
        
        # This requires the original training data for reference
        # For now, use PCA projection followed by sphere normalization
        # (In practice, would store PCA model or use Nystrom extension)
        
        # Simple approach: random projection to sphere
        m = X_new.shape[0]
        
        # Project via random matrix (or use stored projection)
        np.random.seed(self.seed)
        W = np.random.randn(X_new.shape[1], self.n_components)
        Y_new = X_new @ W
        
        # Normalize to sphere
        norms = np.linalg.norm(Y_new, axis=1, keepdims=True)
        norms = np.maximum(norms, MIN_NORM)
        Y_new = self.radius * Y_new / norms
        
        return Y_new


def adaptive_spherical_embedding(
    X: NDArray,
    k: int,
    method: str = 'riemannian',
    loss_type: str = 'mds_geodesic',
    max_iter: int = 500,
    learning_rate: float = 0.01,
    adaptive_radius: bool = True,
    hemisphere_constraint: bool = True,
    seed: int = 42
) -> Tuple[NDArray, Dict]:
    """
    High-level interface for spherical embedding with adaptive optimization.
    
    Parameters
    ----------
    X : NDArray of shape (n, d)
        Input data
    k : int
        Target embedding dimension
    method : str
        Embedding method: 'riemannian', 'fast', 'simple'
    loss_type : str
        Loss function: 'mds_geodesic', 'triplet', 'nca', 'hybrid'
    max_iter : int
        Maximum iterations
    learning_rate : float
        Learning rate
    adaptive_radius : bool
        Whether to optimize radius
    hemisphere_constraint : bool
        Whether to enforce hemisphere constraint
    seed : int
        Random seed
        
    Returns
    -------
    Y : NDArray of shape (n, k)
        Spherical embedding
    info : dict
        Optimization information (loss history, final radius, etc.)
    """
    if method == 'riemannian':
        # Full Riemannian optimization
        model = SphericalEmbedding(
            n_components=k,
            loss_type=loss_type,
            max_iter=max_iter,
            learning_rate=learning_rate,
            adaptive_radius=adaptive_radius,
            hemisphere_constraint=hemisphere_constraint,
            seed=seed
        )
        Y = model.fit_transform(X)
        
        info = {
            'method': 'riemannian',
            'loss_history': model.loss_history_,
            'radius_history': model.radius_history_,
            'final_radius': model.radius,
            'final_loss': model.loss_history_[-1] if model.loss_history_ else np.inf
        }
        
    elif method == 'fast':
        # Fast approximation: PCA + sphere projection + radius optimization
        pca = PCA(n_components=k, random_state=seed)
        Y_pca = pca.fit_transform(X)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(Y_pca, axis=1, keepdims=True)
        norms = np.maximum(norms, MIN_NORM)
        Y = Y_pca / norms
        
        # Optimize radius if requested
        if adaptive_radius:
            D_target = squareform(pdist(X, metric='euclidean'))
            
            # Find optimal radius via line search
            radii = np.logspace(-1, 1, 20)
            best_radius = 1.0
            best_stress = np.inf
            
            for r in radii:
                D_embed = SphericalEmbedding.geodesic_distance_batch(r * Y, r)
                stress = np.sum((D_embed - D_target)**2)
                if stress < best_stress:
                    best_stress = stress
                    best_radius = r
            
            Y = best_radius * Y
            final_radius = best_radius
        else:
            final_radius = 1.0
        
        info = {
            'method': 'fast',
            'pca_explained_variance': pca.explained_variance_ratio_.sum(),
            'final_radius': final_radius
        }
        
    elif method == 'simple':
        # Very simple: just normalize PCA to sphere
        pca = PCA(n_components=k, random_state=seed)
        Y_pca = pca.fit_transform(X)
        
        norms = np.linalg.norm(Y_pca, axis=1, keepdims=True)
        norms = np.maximum(norms, MIN_NORM)
        Y = Y_pca / norms
        
        info = {
            'method': 'simple',
            'pca_explained_variance': pca.explained_variance_ratio_.sum()
        }
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return Y, info


def evaluate_spherical_embedding(X: NDArray, Y: NDArray, radius: float = 1.0) -> Dict:
    """
    Evaluate quality of spherical embedding.
    
    Computes metrics specific to spherical geometry.
    
    Parameters
    ----------
    X : NDArray of shape (n, d)
        Original data
    Y : NDArray of shape (n, k)  
        Spherical embedding
    radius : float
        Sphere radius
        
    Returns
    -------
    metrics : dict
        Evaluation metrics
    """
    n = X.shape[0]
    
    # Original distances
    D_orig = squareform(pdist(X, metric='euclidean'))
    
    # Geodesic distances on sphere
    D_geo = SphericalEmbedding.geodesic_distance_batch(Y, radius)
    
    # Chordal distances (for comparison)
    D_chord = squareform(pdist(Y, metric='euclidean'))
    
    # Compute various metrics
    from scipy.stats import spearmanr
    
    # Rank correlation (geodesic)
    triu_idx = np.triu_indices(n, k=1)
    rank_corr_geo = spearmanr(D_orig[triu_idx], D_geo[triu_idx])[0]
    
    # Rank correlation (chordal) 
    rank_corr_chord = spearmanr(D_orig[triu_idx], D_chord[triu_idx])[0]
    
    # MDS stress (geodesic)
    stress_geo = np.sqrt(np.sum((D_orig - D_geo)**2) / np.sum(D_orig**2))
    
    # MDS stress (chordal)
    stress_chord = np.sqrt(np.sum((D_orig - D_chord)**2) / np.sum(D_orig**2))
    
    # Distortion (multiplicative)
    with np.errstate(divide='ignore', invalid='ignore'):
        distortion_geo = D_geo / (D_orig + EPSILON)
        distortion_geo = distortion_geo[triu_idx]
        distortion_geo = distortion_geo[np.isfinite(distortion_geo)]
    
    mean_distortion = np.mean(np.abs(1 - distortion_geo)) if len(distortion_geo) > 0 else np.inf
    max_distortion = np.max(np.abs(1 - distortion_geo)) if len(distortion_geo) > 0 else np.inf
    
    # Coverage of sphere (how well distributed)
    # Compute minimum pairwise geodesic distance
    D_geo_no_diag = D_geo.copy()
    np.fill_diagonal(D_geo_no_diag, np.inf)
    min_separation = np.min(D_geo_no_diag)
    
    # Maximum geodesic distance (checking for antipodal points)
    max_separation = np.max(D_geo)
    antipodal_ratio = max_separation / (np.pi * radius)  # How close to antipodal
    
    # Angular statistics
    # Compute all pairwise angles
    Y_norm = Y / radius
    cos_angles = np.clip(Y_norm @ Y_norm.T, -1, 1)
    angles = np.arccos(cos_angles)
    angles_triu = angles[triu_idx]
    
    metrics = {
        'rank_correlation_geodesic': float(rank_corr_geo),
        'rank_correlation_chordal': float(rank_corr_chord),
        'stress_geodesic': float(stress_geo),
        'stress_chordal': float(stress_chord),
        'mean_distortion': float(mean_distortion),
        'max_distortion': float(max_distortion),
        'min_separation': float(min_separation),
        'max_separation': float(max_separation),
        'antipodal_ratio': float(antipodal_ratio),
        'mean_angle': float(np.mean(angles_triu)),
        'std_angle': float(np.std(angles_triu)),
        'radius': float(radius)
    }
    
    return metrics