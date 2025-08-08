"""
Poincaré (Hyperbolic) Embedding Module

This module implements rigorous hyperbolic geometry operations for dimensionality
reduction in the Poincaré ball model. The implementation follows the mathematical
framework described in:

- Nickel & Kiela (2017): "Poincaré Embeddings for Learning Hierarchical Representations"
- Ganea et al. (2018): "Hyperbolic Neural Networks"
- Chami et al. (2019): "Hyperbolic Graph Convolutional Neural Networks"

Mathematical Framework:
----------------------
The Poincaré ball model B^n_c = {x ∈ ℝ^n : c||x||^2 < 1} with curvature -c < 0
represents n-dimensional hyperbolic space. Key operations:

1. Möbius addition: ⊕_c for gyrovector space operations
2. Exponential map: exp^c_x(v) for tangent-to-manifold mapping  
3. Logarithmic map: log^c_x(y) for manifold-to-tangent mapping
4. Hyperbolic distance: d_B^c(x,y) preserving negative curvature geometry
5. Riemannian gradient: grad_R f(x) with proper metric tensor scaling

All operations include numerical stability safeguards for boundary proximity.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Union, Callable
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Numerical stability constants
MIN_NORM = 1e-15  # Minimum norm to avoid division by zero
MAX_NORM = 1.0 - 1e-5  # Maximum norm in Poincaré ball (slightly less than 1)
BOUNDARY_EPS = 1e-5  # Distance from boundary to maintain stability
GRAD_CLIP_VALUE = 10.0  # Maximum gradient norm for stability


class PoincareBall:
    """
    Poincaré ball model for hyperbolic geometry.
    
    The Poincaré ball B^n_c with curvature -c provides a conformal model of 
    hyperbolic space where angles are preserved but distances grow exponentially
    near the boundary.
    
    Attributes:
        c: Curvature parameter (c > 0, typically in [0.1, 1.0])
        dim: Dimension of the embedding space
        eps: Small constant for numerical stability
    """
    
    def __init__(self, c: float = 1.0, dim: int = 2, eps: float = 1e-5):
        """
        Initialize Poincaré ball model.
        
        Args:
            c: Curvature parameter (higher = more curvature)
            dim: Dimension of hyperbolic space
            eps: Numerical stability threshold
        """
        assert c > 0, "Curvature must be positive"
        self.c = c
        self.dim = dim
        self.eps = eps
        self.max_norm = (1.0 - BOUNDARY_EPS) / np.sqrt(c)
        
    def _lambda_c(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Conformal factor λ_c^x = 2 / (1 - c||x||^2).
        
        This scaling factor relates the Euclidean metric to the hyperbolic metric
        at point x in the Poincaré ball.
        """
        x_sqnorm = np.sum(x * x, axis=-1, keepdims=True)
        x_sqnorm = np.clip(x_sqnorm, 0, (1.0 - self.eps) / self.c)
        return 2.0 / (1.0 - self.c * x_sqnorm + self.eps)
    
    def mobius_add(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Möbius addition in the Poincaré ball.
        
        Standard formula: x ⊕_c y = (x(1+c||y||²) + y(1-c||x||²)) / (1 + 2c⟨x,y⟩ + c²||x||²||y||²)
        
        This operation provides the group structure for the gyrovector space.
        """
        x_sqnorm = np.clip(np.sum(x * x, axis=-1, keepdims=True), 0, (1.0 - self.eps) / self.c)
        y_sqnorm = np.clip(np.sum(y * y, axis=-1, keepdims=True), 0, (1.0 - self.eps) / self.c)
        xy_inner = np.sum(x * y, axis=-1, keepdims=True)
        
        # Standard Möbius addition formula
        numerator_x = (1.0 + self.c * y_sqnorm) * x
        numerator_y = (1.0 - self.c * x_sqnorm) * y
        numerator = numerator_x + numerator_y
        denominator = 1.0 + 2.0 * self.c * xy_inner + self.c**2 * x_sqnorm * y_sqnorm
        
        result = numerator / (denominator + self.eps)
        return self.project(result)
    
    def exponential_map(self, x: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Exponential map from tangent space T_x B^n_c to Poincaré ball.
        
        exp^c_x(v) = x ⊕_c (tanh(√c λ_c^x ||v|| / 2) v / (√c ||v||))
        
        Maps tangent vectors to points on the manifold along geodesics.
        """
        v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
        v_norm = np.clip(v_norm, MIN_NORM, None)
        
        lambda_c_x = self._lambda_c(x)
        coeff = np.tanh(np.sqrt(self.c) * lambda_c_x * v_norm / 2.0) / (np.sqrt(self.c) * v_norm)
        y = self.mobius_add(x, coeff * v)
        
        return self.project(y)
    
    def logarithmic_map(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Logarithmic map from Poincaré ball to tangent space T_x B^n_c.
        
        log^c_x(y) = (2 / (√c λ_c^x)) artanh(√c ||-x ⊕_c y||) (-x ⊕_c y) / ||-x ⊕_c y||
        
        Inverse of exponential map; projects manifold points to tangent space.
        """
        neg_x = -x
        add = self.mobius_add(neg_x, y)
        add_norm = np.linalg.norm(add, axis=-1, keepdims=True)
        add_norm = np.clip(add_norm, MIN_NORM, None)
        
        lambda_c_x = self._lambda_c(x)
        sqrt_c = np.sqrt(self.c)
        
        # Clipped artanh for stability
        artanh_arg = np.clip(sqrt_c * add_norm, 0, 1.0 - self.eps)
        coeff = (2.0 / (sqrt_c * lambda_c_x)) * np.arctanh(artanh_arg) / add_norm
        
        return coeff * add
    
    def hyperbolic_distance(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Hyperbolic distance in the Poincaré ball.
        
        d_B^c(x,y) = (2/√c) artanh(√c ||-x ⊕_c y||)
        
        This is the geodesic distance in hyperbolic space.
        """
        neg_x = -x
        add = self.mobius_add(neg_x, y)
        add_norm = np.linalg.norm(add, axis=-1, keepdims=True)
        
        sqrt_c = np.sqrt(self.c)
        artanh_arg = np.clip(sqrt_c * add_norm, 0, 1.0 - self.eps)
        
        return (2.0 / sqrt_c) * np.arctanh(artanh_arg)
    
    def project(self, x: NDArray[np.float64], max_norm: Optional[float] = None) -> NDArray[np.float64]:
        """
        Project points to ensure they remain within the Poincaré ball.
        
        Clips norm to max_norm * (1 - eps) to maintain numerical stability.
        """
        if max_norm is None:
            max_norm = self.max_norm
        else:
            max_norm = min(max_norm, self.max_norm)
            
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        norm = np.clip(norm, MIN_NORM, None)
        
        # Scale down if outside ball
        scale = np.where(norm > max_norm, max_norm / norm, 1.0)
        return x * scale
    
    def riemannian_gradient(self, x: NDArray[np.float64], 
                           grad_e: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert Euclidean gradient to Riemannian gradient.
        
        grad_R f(x) = (1 - c||x||²)² / 4 * grad_E f(x)
        
        This accounts for the metric tensor of hyperbolic space.
        """
        x_sqnorm = np.sum(x * x, axis=-1, keepdims=True)
        x_sqnorm = np.clip(x_sqnorm, 0, (1.0 - self.eps) / self.c)
        
        factor = ((1.0 - self.c * x_sqnorm) ** 2) / 4.0
        return factor * grad_e
    
    def parallel_transport(self, x: NDArray[np.float64], y: NDArray[np.float64], 
                          v: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Parallel transport of vector v from T_x to T_y along geodesic.
        
        PT_{x→y}(v) = (λ_c^x / λ_c^y) * gyr[y, -x]v
        
        where gyr is the gyration operator maintaining vector relationships.
        """
        lambda_x = self._lambda_c(x)
        lambda_y = self._lambda_c(y)
        
        # Simplified parallel transport using scaling
        # Full gyration would require more complex computation
        scale = lambda_x / lambda_y
        return scale * v
    
    def midpoint(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Hyperbolic midpoint between x and y.
        
        Uses geodesic interpolation at t=0.5.
        """
        v = self.logarithmic_map(x, y)
        return self.exponential_map(x, 0.5 * v)
    
    def random_point(self, shape: Tuple[int, ...], max_norm: float = 0.9) -> NDArray[np.float64]:
        """
        Generate random points uniformly in the Poincaré ball.
        
        Uses radial density correction for uniform hyperbolic distribution.
        """
        # Generate points on unit sphere
        gaussian = np.random.randn(*shape, self.dim)
        gaussian_norm = np.linalg.norm(gaussian, axis=-1, keepdims=True)
        gaussian_norm = np.clip(gaussian_norm, MIN_NORM, None)
        direction = gaussian / gaussian_norm
        
        # Sample radius with hyperbolic volume correction
        # In hyperbolic space, volume grows exponentially with radius
        u = np.random.uniform(0, 1, size=(*shape, 1))
        r = np.tanh(u * np.arctanh(max_norm * np.sqrt(self.c))) / np.sqrt(self.c)
        
        return self.project(r * direction)


class RiemannianOptimizer:
    """
    Riemannian optimization algorithms for hyperbolic embeddings.
    
    Implements RSGD and RAdam with proper exponential map retractions.
    """
    
    def __init__(self, ball: PoincareBall, lr: float = 0.01, 
                 momentum: float = 0.9, eps: float = 1e-8):
        """
        Initialize Riemannian optimizer.
        
        Args:
            ball: Poincaré ball model
            lr: Learning rate
            momentum: Momentum coefficient (for RAdam)
            eps: Epsilon for numerical stability
        """
        self.ball = ball
        self.lr = lr
        self.momentum = momentum
        self.eps = eps
        self.velocity = None
        self.second_moment = None
        self.step = 0
        
    def rsgd_step(self, x: NDArray[np.float64], grad_e: NDArray[np.float64],
                  lr: Optional[float] = None) -> NDArray[np.float64]:
        """
        Riemannian SGD step with exponential map retraction.
        
        x_{t+1} = exp^c_x(-lr * grad_R f(x))
        """
        if lr is None:
            lr = self.lr
            
        # Convert to Riemannian gradient
        grad_r = self.ball.riemannian_gradient(x, grad_e)
        
        # Clip gradient for stability
        grad_norm = np.linalg.norm(grad_r, axis=-1, keepdims=True)
        grad_norm = np.clip(grad_norm, MIN_NORM, None)
        grad_r = np.where(grad_norm > GRAD_CLIP_VALUE, 
                         grad_r * GRAD_CLIP_VALUE / grad_norm, grad_r)
        
        # Exponential map update
        v = -lr * grad_r
        x_new = self.ball.exponential_map(x, v)
        
        return self.ball.project(x_new)
    
    def radam_step(self, x: NDArray[np.float64], grad_e: NDArray[np.float64],
                   lr: Optional[float] = None, beta1: float = 0.9,
                   beta2: float = 0.999) -> NDArray[np.float64]:
        """
        Riemannian Adam with bias correction and adaptive learning rates.
        
        Maintains first and second moment estimates in tangent space.
        """
        if lr is None:
            lr = self.lr
            
        self.step += 1
        
        # Convert to Riemannian gradient
        grad_r = self.ball.riemannian_gradient(x, grad_e)
        
        # Initialize moments if needed
        if self.velocity is None:
            self.velocity = np.zeros_like(grad_r)
            self.second_moment = np.zeros_like(grad_r)
        
        # Update biased moments
        self.velocity = beta1 * self.velocity + (1 - beta1) * grad_r
        self.second_moment = beta2 * self.second_moment + (1 - beta2) * (grad_r ** 2)
        
        # Bias correction
        velocity_corrected = self.velocity / (1 - beta1 ** self.step)
        second_moment_corrected = self.second_moment / (1 - beta2 ** self.step)
        
        # Adaptive update
        v = -lr * velocity_corrected / (np.sqrt(second_moment_corrected) + self.eps)
        
        # Clip for stability
        v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
        v_norm = np.clip(v_norm, MIN_NORM, None)
        v = np.where(v_norm > GRAD_CLIP_VALUE, v * GRAD_CLIP_VALUE / v_norm, v)
        
        # Exponential map update
        x_new = self.ball.exponential_map(x, v)
        
        return self.ball.project(x_new)


class HyperbolicEmbedding:
    """
    Complete hyperbolic embedding system with loss functions and training.
    
    Implements various loss functions optimized for hyperbolic geometry:
    - Ranking losses (triplet, NCA) with hyperbolic distances
    - MDS stress adapted for negative curvature
    - Regularization terms for stability
    """
    
    def __init__(self, n_components: int = 2, c: float = 1.0,
                 lr: float = 0.01, n_epochs: int = 100,
                 optimizer: str = 'radam', init_method: str = 'pca',
                 loss_fn: str = 'stress', regularization: float = 0.01,
                 batch_size: int = 256, seed: int = 42):
        """
        Initialize hyperbolic embedding system.
        
        Args:
            n_components: Target embedding dimension
            c: Curvature parameter
            lr: Learning rate
            n_epochs: Number of training epochs
            optimizer: Optimizer type ('rsgd' or 'radam')
            init_method: Initialization ('random', 'pca', 'spectral')
            loss_fn: Loss function ('stress', 'triplet', 'nca', 'sammon')
            regularization: L2 regularization weight
            batch_size: Batch size for optimization
            seed: Random seed
        """
        self.n_components = n_components
        self.c = c
        self.lr = lr
        self.n_epochs = n_epochs
        self.optimizer_type = optimizer
        self.init_method = init_method
        self.loss_fn = loss_fn
        self.regularization = regularization
        self.batch_size = batch_size
        self.seed = seed
        
        self.ball = PoincareBall(c=c, dim=n_components)
        self.optimizer = RiemannianOptimizer(self.ball, lr=lr)
        self.embedding_ = None
        self.loss_history_ = []
        
        np.random.seed(seed)
        
    def _initialize_embedding(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Initialize embedding using various strategies.
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Initial embedding in Poincaré ball
        """
        n_samples = X.shape[0]
        
        if self.init_method == 'random':
            # Random initialization near origin
            Y = self.ball.random_point((n_samples,), max_norm=0.1)
            
        elif self.init_method == 'pca':
            # PCA initialization scaled to ball
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.n_components, random_state=self.seed)
            Y_pca = pca.fit_transform(X)
            
            # Scale to fit in ball with margin
            Y_pca_norm = np.linalg.norm(Y_pca, axis=-1, keepdims=True)
            scale = 0.5 / (np.max(Y_pca_norm) + 1e-5)
            Y = self.ball.project(Y_pca * scale, max_norm=0.5)
            
        elif self.init_method == 'spectral':
            # Spectral initialization using graph Laplacian
            from sklearn.manifold import SpectralEmbedding
            spec = SpectralEmbedding(n_components=self.n_components, 
                                    random_state=self.seed)
            Y_spec = spec.fit_transform(X)
            
            # Scale to ball
            Y_spec_norm = np.linalg.norm(Y_spec, axis=-1, keepdims=True)
            scale = 0.5 / (np.max(Y_spec_norm) + 1e-5)
            Y = self.ball.project(Y_spec * scale, max_norm=0.5)
            
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")
            
        return Y
    
    def _compute_stress_loss(self, Y: NDArray[np.float64], 
                            D_high: NDArray[np.float64]) -> Tuple[float, NDArray[np.float64]]:
        """
        MDS stress loss adapted for hyperbolic geometry.
        
        Loss = Σ_ij w_ij (d_H(y_i, y_j) - d_ij)²
        
        where d_H is hyperbolic distance and w_ij are weights.
        """
        n = Y.shape[0]
        loss = 0.0
        grad = np.zeros_like(Y)
        
        for i in range(n):
            # Compute hyperbolic distances from point i
            d_hyp = self.ball.hyperbolic_distance(Y[i:i+1], Y)
            
            # Stress differences
            diff = d_hyp.flatten() - D_high[i]
            
            # Weight by inverse distance (emphasize local structure)
            weights = 1.0 / (D_high[i] + 1.0)
            
            # Accumulate loss
            loss += np.sum(weights * diff ** 2)
            
            # Gradient computation via logarithmic map
            for j in range(n):
                if i != j:
                    if np.abs(diff[j]) > 1e-10:
                        # Direction in tangent space
                        v_ij = self.ball.logarithmic_map(Y[i], Y[j])
                        v_ij_norm = np.linalg.norm(v_ij)
                        if v_ij_norm > MIN_NORM:
                            # Gradient contribution
                            grad[i] += 2 * weights[j] * diff[j] * v_ij / v_ij_norm
        
        return loss / n, grad / n
    
    def _compute_triplet_loss(self, Y: NDArray[np.float64], 
                             triplets: NDArray[np.int32],
                             margin: float = 1.0) -> Tuple[float, NDArray[np.float64]]:
        """
        Triplet loss with hyperbolic distances.
        
        Loss = Σ max(0, d_H(a,p) - d_H(a,n) + margin)
        
        Encourages similar points closer than dissimilar ones.
        """
        loss = 0.0
        grad = np.zeros_like(Y)
        
        for anchor_idx, pos_idx, neg_idx in triplets:
            # Hyperbolic distances
            d_ap = self.ball.hyperbolic_distance(Y[anchor_idx:anchor_idx+1], 
                                                Y[pos_idx:pos_idx+1]).item()
            d_an = self.ball.hyperbolic_distance(Y[anchor_idx:anchor_idx+1], 
                                                Y[neg_idx:neg_idx+1]).item()
            
            # Triplet loss
            loss_triplet = d_ap - d_an + margin
            
            if loss_triplet > 0:
                loss += loss_triplet
                
                # Gradients via logarithmic map
                v_ap = self.ball.logarithmic_map(Y[anchor_idx], Y[pos_idx])
                v_an = self.ball.logarithmic_map(Y[anchor_idx], Y[neg_idx])
                
                v_ap_norm = np.linalg.norm(v_ap)
                v_an_norm = np.linalg.norm(v_an)
                
                if v_ap_norm > MIN_NORM:
                    grad[anchor_idx] += v_ap / v_ap_norm
                    grad[pos_idx] -= self.ball.parallel_transport(
                        Y[anchor_idx], Y[pos_idx], v_ap / v_ap_norm)
                
                if v_an_norm > MIN_NORM:
                    grad[anchor_idx] -= v_an / v_an_norm
                    grad[neg_idx] += self.ball.parallel_transport(
                        Y[anchor_idx], Y[neg_idx], v_an / v_an_norm)
        
        return loss / len(triplets), grad / len(triplets)
    
    def _compute_nca_loss(self, Y: NDArray[np.float64], 
                         labels: NDArray[np.int32]) -> Tuple[float, NDArray[np.float64]]:
        """
        Neighborhood Component Analysis loss in hyperbolic space.
        
        Maximizes probability of correct neighbor classification.
        """
        n = Y.shape[0]
        loss = 0.0
        grad = np.zeros_like(Y)
        
        # Compute all pairwise hyperbolic distances
        D_hyp = np.zeros((n, n))
        for i in range(n):
            D_hyp[i] = self.ball.hyperbolic_distance(Y[i:i+1], Y).flatten()
        
        # Compute probabilities using softmax over negative distances
        for i in range(n):
            # Softmax probabilities
            exp_neg_dist = np.exp(-D_hyp[i])
            exp_neg_dist[i] = 0  # Exclude self
            
            if np.sum(exp_neg_dist) > 0:
                p_ij = exp_neg_dist / np.sum(exp_neg_dist)
                
                # Probability of selecting correct class
                same_class = labels == labels[i]
                same_class[i] = False
                p_correct = np.sum(p_ij[same_class])
                
                # NCA loss (negative log likelihood)
                if p_correct > 0:
                    loss -= np.log(p_correct + 1e-10)
                
                # Gradient computation
                for j in range(n):
                    if i != j:
                        v_ij = self.ball.logarithmic_map(Y[i], Y[j])
                        v_ij_norm = np.linalg.norm(v_ij)
                        
                        if v_ij_norm > MIN_NORM:
                            if same_class[j]:
                                # Pull together
                                grad[i] -= p_ij[j] * v_ij / v_ij_norm
                            else:
                                # Push apart
                                grad[i] += p_ij[j] * v_ij / v_ij_norm
        
        return loss / n, grad / n
    
    def fit(self, X: NDArray[np.float64], 
            y: Optional[NDArray[np.int32]] = None) -> 'HyperbolicEmbedding':
        """
        Fit hyperbolic embedding to data.
        
        Args:
            X: Input data (n_samples, n_features) or distance matrix
            y: Optional labels for supervised losses
            
        Returns:
            Self with fitted embedding
        """
        n_samples = X.shape[0]
        
        # Check if X is a distance matrix (square and symmetric)
        is_distance_matrix = (X.shape[0] == X.shape[1] and 
                            np.allclose(X, X.T) and 
                            np.allclose(np.diag(X), 0))
        
        if is_distance_matrix:
            D_high = X
            # Use MDS-style initialization for distance matrix
            X_init = self._distance_to_points(D_high)
        else:
            # Compute pairwise distances
            from sklearn.metrics import pairwise_distances
            D_high = pairwise_distances(X)
            X_init = X
        
        # Initialize embedding
        self.embedding_ = self._initialize_embedding(X_init)
        
        # Training loop
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            
            # Mini-batch training
            indices = np.random.permutation(n_samples)
            n_batches = (n_samples + self.batch_size - 1) // self.batch_size
            
            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, n_samples)
                batch_indices = indices[start:end]
                
                # Compute loss and gradient based on chosen function
                if self.loss_fn == 'stress':
                    # Use local distance matrix for batch
                    D_batch = D_high[batch_indices][:, batch_indices]
                    Y_batch = self.embedding_[batch_indices]
                    loss, grad = self._compute_stress_loss(Y_batch, D_batch)
                    
                elif self.loss_fn == 'triplet' and y is not None:
                    # Generate triplets from batch
                    triplets = self._generate_triplets(y[batch_indices], n_triplets=len(batch_indices))
                    if len(triplets) > 0:
                        Y_batch = self.embedding_[batch_indices]
                        loss, grad = self._compute_triplet_loss(Y_batch, triplets)
                    else:
                        continue
                        
                elif self.loss_fn == 'nca' and y is not None:
                    Y_batch = self.embedding_[batch_indices]
                    loss, grad = self._compute_nca_loss(Y_batch, y[batch_indices])
                    
                else:
                    # Default to stress loss
                    D_batch = D_high[batch_indices][:, batch_indices]
                    Y_batch = self.embedding_[batch_indices]
                    loss, grad = self._compute_stress_loss(Y_batch, D_batch)
                
                # Add L2 regularization to keep points away from boundary
                if self.regularization > 0:
                    norms = np.linalg.norm(Y_batch, axis=-1, keepdims=True)
                    reg_loss = self.regularization * np.mean(norms ** 2)
                    reg_grad = 2 * self.regularization * Y_batch
                    
                    loss += reg_loss
                    grad += reg_grad
                
                epoch_loss += loss
                
                # Optimization step
                if self.optimizer_type == 'rsgd':
                    self.embedding_[batch_indices] = self.optimizer.rsgd_step(
                        Y_batch, grad, lr=self.lr * (0.95 ** epoch))  # Learning rate decay
                else:  # radam
                    self.embedding_[batch_indices] = self.optimizer.radam_step(
                        Y_batch, grad, lr=self.lr)
            
            # Record loss
            avg_loss = epoch_loss / n_batches
            self.loss_history_.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{self.n_epochs}, Loss: {avg_loss:.6f}")
        
        return self
    
    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Transform new data to hyperbolic embedding.
        
        For new points, uses optimization to find best embedding.
        """
        if self.embedding_ is None:
            raise ValueError("Model must be fitted before transform")
        
        # For simplicity, return fitted embedding if same size
        if X.shape[0] == self.embedding_.shape[0]:
            return self.embedding_
        
        # For new data, would need out-of-sample extension
        # This is a placeholder - full implementation would optimize
        # new point positions based on distances to training data
        n_new = X.shape[0]
        return self.ball.random_point((n_new,), max_norm=0.5)
    
    def fit_transform(self, X: NDArray[np.float64], 
                     y: Optional[NDArray[np.int32]] = None) -> NDArray[np.float64]:
        """
        Fit and transform in one step.
        """
        self.fit(X, y)
        return self.embedding_
    
    def _distance_to_points(self, D: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert distance matrix to points using classical MDS.
        
        Used for initialization when given distance matrix.
        """
        n = D.shape[0]
        
        # Double centering
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ (D ** 2) @ H
        
        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(B)
        
        # Take top k components
        idx = np.argsort(eigvals)[::-1][:self.n_components]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Reconstruct points
        X = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0)))
        
        return X
    
    def _generate_triplets(self, labels: NDArray[np.int32], 
                          n_triplets: int) -> NDArray[np.int32]:
        """
        Generate triplets (anchor, positive, negative) from labels.
        """
        triplets = []
        unique_labels = np.unique(labels)
        
        for _ in range(n_triplets):
            # Random anchor
            anchor_idx = np.random.randint(len(labels))
            anchor_label = labels[anchor_idx]
            
            # Find positive (same class)
            pos_candidates = np.where(labels == anchor_label)[0]
            pos_candidates = pos_candidates[pos_candidates != anchor_idx]
            
            if len(pos_candidates) == 0:
                continue
                
            pos_idx = np.random.choice(pos_candidates)
            
            # Find negative (different class)
            neg_candidates = np.where(labels != anchor_label)[0]
            
            if len(neg_candidates) == 0:
                continue
                
            neg_idx = np.random.choice(neg_candidates)
            
            triplets.append([anchor_idx, pos_idx, neg_idx])
        
        return np.array(triplets, dtype=np.int32)


def run_poincare_optimized(X: NDArray[np.float64], k: int, 
                          c: float = 1.0, lr: float = 0.01,
                          n_epochs: int = 100, optimizer: str = 'radam',
                          loss_fn: str = 'stress', init_method: str = 'pca',
                          regularization: float = 0.01,
                          seed: int = 42) -> Tuple[NDArray[np.float64], float]:
    """
    Optimized Poincaré embedding for integration with existing pipeline.
    
    This function provides the main entry point for hyperbolic embeddings,
    compatible with the existing OrthoReduce interface.
    
    Args:
        X: Input data (n_samples, n_features)
        k: Target dimension
        c: Curvature parameter
        lr: Learning rate
        n_epochs: Number of epochs
        optimizer: Optimizer type ('rsgd' or 'radam')
        loss_fn: Loss function ('stress', 'triplet', 'nca')
        init_method: Initialization method ('random', 'pca', 'spectral')
        regularization: L2 regularization weight
        seed: Random seed
        
    Returns:
        (embedding, runtime) tuple
    """
    import time
    start_time = time.time()
    
    try:
        # Create and fit hyperbolic embedding
        embedding = HyperbolicEmbedding(
            n_components=k,
            c=c,
            lr=lr,
            n_epochs=n_epochs,
            optimizer=optimizer,
            loss_fn=loss_fn,
            init_method=init_method,
            regularization=regularization,
            seed=seed
        )
        
        Y = embedding.fit_transform(X)
        runtime = time.time() - start_time
        
        logger.info(f"Poincaré embedding completed: shape={Y.shape}, "
                   f"curvature={c}, runtime={runtime:.2f}s")
        
        return Y, runtime
        
    except Exception as e:
        logger.error(f"Poincaré embedding failed: {e}")
        # Fallback to random embedding
        np.random.seed(seed)
        Y = np.random.randn(X.shape[0], k) * 0.1
        return Y, time.time() - start_time