"""
convex_optimized.py - Enhanced convex hull projection with numerical stability.

For projected data Y, compute convex hull vertices and project each point
onto the convex hull by solving min ||V a - y||^2 s.t. sum(a) = 1, a >= 0.

Enhanced Implementation Features:
- Ridge regularization: G = VV^T + λI for numerical stability
- Configurable solver tolerances for accuracy vs. speed trade-offs
- Float64 support for high-precision requirements
- Robust objective variants (Huber loss, epsilon-insensitive)
- Improved candidate selection and warm-start mechanisms
- Normalized vertex handling for geometric stability
- Enhanced convergence monitoring and fallback strategies

Implementation notes:
- We use scipy.spatial.ConvexHull to compute vertices when feasible.
- For numerical stability, ridge regularization prevents ill-conditioning.
- Optimization uses SLSQP with enhanced constraint handling.
- Multiple solver tolerance modes support both screening and refinement.
"""
from __future__ import annotations
from typing import Tuple, Optional, Literal, Callable
import warnings

import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, LinearConstraint, Bounds


def _huber_loss(residual: np.ndarray, delta: float = 1.0) -> float:
    """Huber loss function for robust optimization."""
    abs_r = np.abs(residual)
    quad_mask = abs_r <= delta
    return np.where(quad_mask, 0.5 * residual**2, delta * (abs_r - 0.5 * delta)).sum()


def _epsilon_insensitive_loss(residual: np.ndarray, epsilon: float = 1e-3) -> float:
    """Epsilon-insensitive loss function."""
    abs_r = np.abs(residual)
    return np.maximum(0, abs_r - epsilon).sum()


def _normalize_candidates(V: np.ndarray, mode: Literal['none', 'l2', 'unit_sphere'] = 'none') -> np.ndarray:
    """Normalize candidate vertices according to specified mode."""
    if mode == 'none':
        return V
    elif mode == 'l2':
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms = np.maximum(norms, np.finfo(V.dtype).eps)
        return V / norms
    elif mode == 'unit_sphere':
        # Project onto unit sphere
        norms = np.linalg.norm(V, axis=1)
        norms = np.maximum(norms, np.finfo(V.dtype).eps)
        return V / norms[:, None]
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def _hull_vertices(Y: np.ndarray) -> np.ndarray:
    """Compute or approximate convex hull vertices for Y.

    Falls back to per-dimension extremes if exact hull construction fails.
    """
    n, d = Y.shape
    if n <= d + 1:
        return Y.copy()
    try:
        hull = ConvexHull(Y)
        V = Y[hull.vertices]
        return V
    except Exception:
        # Approximate via extremes per dimension
        idxs = set()
        for j in range(d):
            idxs.add(int(np.argmin(Y[:, j])))
            idxs.add(int(np.argmax(Y[:, j])))
        V = Y[sorted(list(idxs))]
        return V


def _project_single(V: np.ndarray, y: np.ndarray, tol: float = 1e-6, maxiter: int = 200) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Project single point y onto conv(V) via constrained least squares.

    Returns projected point y_hat, alpha weights, and success flag.
    """
    m = V.shape[0]

    # Objective: f(a) = 0.5 * ||V a - y||^2
    def fun(a: np.ndarray) -> float:
        r = V.T @ a - y  # (d,)
        return 0.5 * float(r @ r)

    def jac(a: np.ndarray) -> np.ndarray:
        # grad = V (V a - y)
        r = V.T @ a - y
        return (V @ r).astype(float)

    # Constraints: sum(a) = 1, a >= 0
    A = np.ones((1, m))
    lc = LinearConstraint(A, lb=np.array([1.0]), ub=np.array([1.0]))
    bounds = Bounds(lb=np.zeros(m), ub=np.ones(m))

    a0 = np.full(m, 1.0 / m, dtype=float)
    res = minimize(fun, a0, jac=jac, method='SLSQP', constraints=[lc], bounds=bounds,
                   options={'maxiter': maxiter, 'ftol': tol, 'disp': False})

    if not res.success:
        # Fallback: project to nearest vertex
        idx = int(np.argmin(np.sum((V - y) ** 2, axis=1)))
        a = np.zeros(m)
        a[idx] = 1.0
        return V[idx].copy(), a, False

    a = np.clip(res.x, 0.0, 1.0)
    # Renormalize to ensure sum=1
    s = float(a.sum())
    if s <= 0:
        a = np.full(m, 1.0 / m)
    else:
        a = a / s
    y_hat = V.T @ a
    return y_hat.astype(float), a.astype(float), True


def _project_single_precomputed(G: np.ndarray, b: np.ndarray, y_norm2: float,
                                tol: float = 1e-6, maxiter: int = 200,
                                a0: Optional[np.ndarray] = None,
                                ridge_lambda: float = 0.0,
                                objective_type: Literal['quadratic', 'huber', 'epsilon_insensitive'] = 'quadratic',
                                huber_delta: float = 1.0,
                                epsilon: float = 1e-3) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Enhanced projection using precomputed Gram matrix with ridge regularization.

    Minimizes 0.5 a^T (G + λI) a - b^T a + 0.5 ||y||^2 s.t. sum(a)=1, a>=0.
    
    Parameters
    ----------
    G : np.ndarray
        Gram matrix V V^T
    b : np.ndarray  
        V y vector
    y_norm2 : float
        Squared norm of y
    tol : float
        Solver tolerance
    maxiter : int
        Maximum iterations
    a0 : Optional[np.ndarray]
        Initial point
    ridge_lambda : float
        Ridge regularization parameter λ for G + λI
    objective_type : str
        Type of objective function ('quadratic', 'huber', 'epsilon_insensitive')
    huber_delta : float
        Delta parameter for Huber loss
    epsilon : float
        Epsilon parameter for epsilon-insensitive loss
        
    Returns
    -------
    y_hat : None (placeholder)
    a : np.ndarray
        Barycentric coordinates
    success : bool
        Whether optimization succeeded
    """
    m = G.shape[0]
    
    # Add ridge regularization to Gram matrix
    if ridge_lambda > 0:
        G_reg = G + ridge_lambda * np.eye(m, dtype=G.dtype)
    else:
        G_reg = G.copy()

    # Define objective function based on type
    if objective_type == 'quadratic':
        def fun(a: np.ndarray) -> float:
            return 0.5 * float(a @ (G_reg @ a) - 2.0 * (b @ a) + y_norm2)
        
        def jac(a: np.ndarray) -> np.ndarray:
            return (G_reg @ a - b).astype(float)
            
    elif objective_type == 'huber':
        def fun(a: np.ndarray) -> float:
            residual = G_reg @ a - b
            return _huber_loss(residual, huber_delta) + 0.5 * y_norm2
            
        def jac(a: np.ndarray) -> np.ndarray:
            residual = G_reg @ a - b
            abs_r = np.abs(residual)
            grad_factor = np.where(abs_r <= huber_delta, residual, huber_delta * np.sign(residual))
            return (G_reg.T @ grad_factor).astype(float)
            
    elif objective_type == 'epsilon_insensitive':
        def fun(a: np.ndarray) -> float:
            residual = G_reg @ a - b
            return _epsilon_insensitive_loss(residual, epsilon) + 0.5 * y_norm2
            
        def jac(a: np.ndarray) -> np.ndarray:
            residual = G_reg @ a - b
            abs_r = np.abs(residual)
            grad_factor = np.where(abs_r > epsilon, np.sign(residual), 0.0)
            return (G_reg.T @ grad_factor).astype(float)
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")

    A = np.ones((1, m))
    lc = LinearConstraint(A, lb=np.array([1.0]), ub=np.array([1.0]))
    bounds = Bounds(lb=np.zeros(m), ub=np.ones(m))

    if a0 is None:
        # Enhanced initialization: start at vertex with highest correlation
        if objective_type == 'quadratic':
            # For quadratic, use maximum dot product with b
            idx = int(np.argmax(b))
        else:
            # For robust losses, use a more balanced initialization
            idx = int(np.argmax(np.abs(b)))
        a0 = np.zeros(m, dtype=float)
        a0[idx] = 1.0

    # Enhanced solver with better options for numerical stability
    solver_options = {
        'maxiter': maxiter,
        'ftol': tol,
        'disp': False,
        'eps': np.sqrt(np.finfo(float).eps)  # Better finite difference step
    }
    
    res = minimize(fun, a0, jac=jac, method='SLSQP', constraints=[lc], bounds=bounds,
                   options=solver_options)

    # Enhanced fallback strategy
    if not res.success:
        # Try with looser tolerance if original failed
        if tol < 1e-4:
            looser_options = solver_options.copy()
            looser_options['ftol'] = 1e-4
            res = minimize(fun, a0, jac=jac, method='SLSQP', constraints=[lc], bounds=bounds,
                          options=looser_options)
        
        if not res.success:
            # Final fallback to best vertex
            if objective_type == 'quadratic':
                idx = int(np.argmax(b))
            else:
                # For robust objectives, choose vertex minimizing the robust loss
                vertex_losses = np.zeros(m)
                for i in range(m):
                    a_vertex = np.zeros(m)
                    a_vertex[i] = 1.0
                    vertex_losses[i] = fun(a_vertex)
                idx = int(np.argmin(vertex_losses))
            
            a = np.zeros(m, dtype=float)
            a[idx] = 1.0
            return None, a, False

    # Enhanced post-processing with numerical stability checks
    a = np.clip(res.x.astype(float), 0.0, 1.0)
    s = float(a.sum())
    
    # Check for degenerate solutions
    if s <= np.finfo(float).eps * m:
        # Very small sum - use uniform weights
        a = np.full(m, 1.0 / m, dtype=float)
    else:
        a = a / s
        
    # Final validation: ensure numerical constraints are satisfied
    constraint_violation = abs(a.sum() - 1.0)
    if constraint_violation > 10 * tol:
        warnings.warn(f"Constraint violation {constraint_violation:.2e} exceeds tolerance {tol:.2e}")
    
    return None, a, True


def project_onto_convex_hull_qp(
    Y: np.ndarray,
    tol: float = 1e-6,
    maxiter: int = 200,
    k_candidates: int = 64,
    use_float32: bool = True,
    warm_start_alphas: Optional[np.ndarray] = None,
    batch_size: int = 1024,
    ridge_lambda: float = 1e-6,
    objective_type: Literal['quadratic', 'huber', 'epsilon_insensitive'] = 'quadratic',
    huber_delta: float = 1.0,
    epsilon: float = 1e-3,
    candidate_normalization: Literal['none', 'l2', 'unit_sphere'] = 'none',
    use_float64: bool = False,
    solver_tolerance_mode: Literal['strict', 'balanced', 'loose'] = 'balanced',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enhanced convex hull projection with numerical stability and robust objectives.

    Parameters
    ----------
    Y : (n, d)
        Input data points to project
    tol : float
        Solver tolerance for SLSQP optimization
    maxiter : int
        Maximum iterations for SLSQP solver
    k_candidates : int
        Number of candidate hull vertices to consider per point
    use_float32 : bool
        If True, use float32 for speed and memory (deprecated, use use_float64)
    warm_start_alphas : Optional[np.ndarray]
        Optional warm-start weights of shape (n, m)
    batch_size : int
        Batch size for computing projections
    ridge_lambda : float
        Ridge regularization parameter λ for G + λI (default: 1e-6)
    objective_type : str
        Objective function type: 'quadratic', 'huber', 'epsilon_insensitive'
    huber_delta : float
        Delta parameter for Huber loss (default: 1.0)
    epsilon : float
        Epsilon parameter for epsilon-insensitive loss (default: 1e-3)
    candidate_normalization : str
        Vertex normalization: 'none', 'l2', 'unit_sphere'
    use_float64 : bool
        If True, use float64 for high precision (overrides use_float32)
    solver_tolerance_mode : str
        Tolerance mode: 'strict' (1e-8), 'balanced' (1e-6), 'loose' (1e-4)

    Returns
    -------
    Y_proj : (n, d)
        Projected points onto convex hull
    alphas : (n, m)
        Barycentric coordinates where m = number of hull vertices
    V : (m, d)
        Hull vertex matrix
    """
    # Enhanced precision handling
    if use_float64:
        dtype = np.float64
    else:
        dtype = np.float32 if use_float32 else np.float64
    
    # Enhanced solver tolerance selection
    tolerance_map = {
        'strict': 1e-8,
        'balanced': 1e-6,
        'loose': 1e-4
    }
    if solver_tolerance_mode in tolerance_map:
        effective_tol = tolerance_map[solver_tolerance_mode]
    else:
        effective_tol = tol
    
    Y = np.asarray(Y, dtype=dtype)
    V_raw = _hull_vertices(Y.astype(np.float64))  # hull on float64 for robustness
    
    # Apply candidate normalization
    V = _normalize_candidates(V_raw.astype(dtype), candidate_normalization)
    
    m = V.shape[0]
    n, d = Y.shape
    Y_proj = np.zeros((n, d), dtype=dtype)
    alphas = np.zeros((n, m), dtype=dtype)

    # Enhanced candidate selection preparation with numerical stability
    if candidate_normalization == 'unit_sphere':
        V_unit = V  # Already normalized
    else:
        V_norms = np.linalg.norm(V, axis=1)
        # Avoid division by zero with safer threshold
        safe_norms = np.maximum(V_norms, np.sqrt(np.finfo(dtype).eps))
        V_unit = V / safe_norms[:, None]
        
        # Check for and handle degenerate vertices
        invalid_mask = V_norms < np.sqrt(np.finfo(dtype).eps)
        if invalid_mask.any():
            warnings.warn(f"Found {invalid_mask.sum()} near-zero vertices, using ridge regularization helps")

    # Global Gram matrix with optional ridge regularization
    G = (V @ V.T).astype(dtype)
    if ridge_lambda > 0:
        G += ridge_lambda * np.eye(m, dtype=dtype)

    # Process in batches to compute b = V y efficiently
    for start in range(0, n, max(1, batch_size)):
        end = min(n, start + batch_size)
        Yb = Y[start:end]  # (B, d)
        # Candidate selection via top-|dot| with unit vertices
        sims = (Yb @ V_unit.T)  # (B, m)
        # pick top-k by absolute similarity
        k = min(k_candidates, m)
        idx_part = np.argpartition(np.abs(sims), -k, axis=1)[:, -k:]
        # For reproducible order, sort by descending abs sim per row
        row_indices = np.arange(end - start)[:, None]
        top_sorted = idx_part[row_indices, np.argsort(-np.abs(sims[row_indices, idx_part]))]

        for r in range(end - start):
            y = Yb[r]
            cand_idx = top_sorted[r]
            # Build G_sub and b_sub
            G_sub = G[np.ix_(cand_idx, cand_idx)]
            b_sub = V[cand_idx] @ y
            y_norm2 = float(y @ y)

            a0 = None
            if warm_start_alphas is not None and warm_start_alphas.shape[1] == m:
                a0_full = warm_start_alphas[start + r]
                a0 = a0_full[cand_idx].astype(float)
                # Ensure feasible start: project onto simplex if needed
                a0 = np.clip(a0, 0.0, 1.0)
                s = float(a0.sum())
                if s <= 0:
                    a0 = None
                else:
                    a0 = (a0 / s).astype(float)

            _, a_sub, ok = _project_single_precomputed(
                G_sub.astype(float), b_sub.astype(float), y_norm2,
                tol=effective_tol, maxiter=maxiter, a0=a0,
                ridge_lambda=0.0,  # Ridge already applied to G
                objective_type=objective_type,
                huber_delta=huber_delta,
                epsilon=epsilon
            )
            # Scatter back to full alpha
            a_full = np.zeros(m, dtype=float)
            a_full[cand_idx] = a_sub
            # Compute y_hat = V^T a
            y_hat = (V.T @ a_full).astype(dtype)

            Y_proj[start + r] = y_hat
            alphas[start + r] = a_full.astype(dtype)

    return Y_proj.astype(float), alphas.astype(float), V.astype(float)


def project_onto_convex_hull_enhanced(
    Y: np.ndarray,
    ridge_lambda: float = 1e-6,
    solver_tol: float = 1e-6,
    maxiter: int = 200,
    objective_type: Literal['quadratic', 'huber', 'epsilon_insensitive'] = 'quadratic',
    huber_delta: float = 1.0,
    epsilon: float = 1e-3,
    use_float64: bool = False,
    candidate_normalization: Literal['none', 'l2', 'unit_sphere'] = 'none',
    k_candidates: int = 64,
    warm_start_alphas: Optional[np.ndarray] = None,
    batch_size: int = 1024,
    solver_mode: Literal['strict', 'balanced', 'loose'] = 'balanced',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enhanced convex hull projection with advanced numerical stability features.
    
    This is the recommended interface for new applications requiring enhanced
    stability and robust optimization objectives.
    
    Parameters
    ----------
    Y : np.ndarray, shape (n, d)
        Input data points to project onto convex hull
    ridge_lambda : float, default=1e-6
        Ridge regularization parameter λ for G + λI to prevent ill-conditioning.
        Typical range: [1e-8, 1e-3]. Larger values improve stability but may
        introduce bias.
    solver_tol : float, default=1e-6
        Solver tolerance for SLSQP optimization. Use 1e-8 for high accuracy,
        1e-4 for faster screening.
    maxiter : int, default=200
        Maximum iterations for SLSQP solver
    objective_type : {'quadratic', 'huber', 'epsilon_insensitive'}, default='quadratic'
        Objective function type:
        - 'quadratic': Standard L2 loss (fastest)
        - 'huber': Robust loss, reduces outlier influence
        - 'epsilon_insensitive': Sparse solutions with epsilon tube
    huber_delta : float, default=1.0
        Delta parameter for Huber loss. Transition point between L2 and L1.
    epsilon : float, default=1e-3
        Epsilon parameter for epsilon-insensitive loss. Tolerance for sparsity.
    use_float64 : bool, default=False
        Use float64 for high-precision computation. Slower but more accurate.
    candidate_normalization : {'none', 'l2', 'unit_sphere'}, default='none'
        Normalization applied to hull vertices:
        - 'none': No normalization
        - 'l2': L2 normalization per vertex
        - 'unit_sphere': Project to unit sphere
    k_candidates : int, default=64
        Number of candidate vertices per point. Larger = more accurate but slower.
    warm_start_alphas : np.ndarray, optional
        Warm start weights from previous optimization
    batch_size : int, default=1024
        Batch size for processing points
    solver_mode : {'strict', 'balanced', 'loose'}, default='balanced'
        Predefined solver tolerance modes:
        - 'strict': 1e-8 tolerance, high accuracy
        - 'balanced': 1e-6 tolerance, good trade-off
        - 'loose': 1e-4 tolerance, faster computation
        
    Returns
    -------
    Y_proj : np.ndarray, shape (n, d)
        Points projected onto convex hull
    alphas : np.ndarray, shape (n, m)
        Barycentric coordinates where m = number of hull vertices
    V : np.ndarray, shape (m, d)
        Hull vertex matrix
        
    Examples
    --------
    >>> import numpy as np
    >>> # Standard usage
    >>> Y = np.random.randn(100, 10)
    >>> Y_proj, alphas, V = project_onto_convex_hull_enhanced(Y)
    
    >>> # High-precision mode with ridge regularization
    >>> Y_proj, alphas, V = project_onto_convex_hull_enhanced(
    ...     Y, ridge_lambda=1e-4, use_float64=True, solver_mode='strict')
    
    >>> # Robust optimization with Huber loss
    >>> Y_proj, alphas, V = project_onto_convex_hull_enhanced(
    ...     Y, objective_type='huber', huber_delta=0.5)
    """
    # Map solver mode to tolerance if not overridden
    if solver_mode == 'strict':
        effective_tol = min(solver_tol, 1e-8)
    elif solver_mode == 'balanced':
        effective_tol = min(solver_tol, 1e-6)
    elif solver_mode == 'loose':
        effective_tol = max(solver_tol, 1e-4)
    else:
        effective_tol = solver_tol
        
    return project_onto_convex_hull_qp(
        Y=Y,
        tol=effective_tol,
        maxiter=maxiter,
        k_candidates=k_candidates,
        use_float32=not use_float64,
        warm_start_alphas=warm_start_alphas,
        batch_size=batch_size,
        ridge_lambda=ridge_lambda,
        objective_type=objective_type,
        huber_delta=huber_delta,
        epsilon=epsilon,
        candidate_normalization=candidate_normalization,
        use_float64=use_float64,
        solver_tolerance_mode=solver_mode,
    )
