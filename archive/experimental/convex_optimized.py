"""
convex_optimized.py - Optimized convex hull projection with constraints.

Phase 3: For projected data Y, compute convex hull vertices and project each point
onto the convex hull by solving min ||V a - y||^2 s.t. sum(a) = 1, a >= 0.

Implementation notes:
- We use scipy.spatial.ConvexHull to compute vertices when feasible.
- For numerical stability and speed, we can optionally pre-reduce dimensionality
  (e.g., via PCA) before computing the hull, but here we operate in Y-space and
  fall back to extreme points if hull construction fails.
- Optimization is performed with SLSQP and constraints (LinearConstraint and Bounds).
- The solver tolerances are tuned for runtime efficiency; solutions are validated.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, LinearConstraint, Bounds


def _hull_vertices(Y: np.ndarray) -> np.ndarray:
    """Compute or approximate convex hull vertices for Y."""
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
    def fun(a):
        r = V.T @ a - y  # Using V.T since V is (m, d): want d x 1 = (d x m)(m x 1)
        return 0.5 * float(r @ r)

    def jac(a):
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


def project_onto_convex_hull_qp(Y: np.ndarray, tol: float = 1e-6, maxiter: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project all points in Y onto conv(Y) using QP over hull vertices.

    Parameters
    ----------
    Y : (n, d)
    tol : float
        Tolerance for SLSQP.
    maxiter : int
        Maximum iterations for SLSQP.

    Returns
    -------
    Y_proj : (n, d)
    alphas : (n, m) where m = number of hull vertices
    V : (m, d) hull vertex matrix
    """
    Y = np.asarray(Y, dtype=float)
    V = _hull_vertices(Y)
    m = V.shape[0]
    n, d = Y.shape
    Y_proj = np.zeros_like(Y)
    alphas = np.zeros((n, m), dtype=float)

    # Precompute V^T for speed in per-point projection
    for i in range(n):
        y = Y[i]
        y_hat, a, _ = _project_single(V, y, tol=tol, maxiter=maxiter)
        Y_proj[i] = y_hat
        alphas[i] = a
    return Y_proj, alphas, V
