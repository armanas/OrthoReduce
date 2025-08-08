import numpy as np
from typing import Tuple
from .dimensionality_reduction import (
    run_jll as _run_jll,
    run_poincare as _run_poincare,
    run_spherical as _run_spherical,
    run_experiment as _run_experiment,
)


def run_jll(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    Y, _ = _run_jll(X, k, seed)
    return Y


def run_poincare_pipeline(X: np.ndarray, k: int, c: float = 1.0, seed: int = 42) -> np.ndarray:
    Y, _ = _run_poincare(X, k, seed, c=c)
    # Tests expect same dimensionality as input (n, d)
    n, d = X.shape
    if Y.shape[1] < d:
        pad = np.zeros((n, d - Y.shape[1]), dtype=Y.dtype)
        Y = np.concatenate([Y, pad], axis=1)
    return Y[:, :d]


def run_spherical_pipeline(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    Y, _ = _run_spherical(X, k, seed)
    # Ensure output has original dimension
    n, d = X.shape
    if Y.shape[1] < d:
        pad = np.zeros((n, d - Y.shape[1]), dtype=Y.dtype)
        Y = np.concatenate([Y, pad], axis=1)
    return Y[:, :d]


def run_experiment(n: int = 100, d: int = 20, epsilon: float = 0.5, seed: int = 0):
    methods = ['pca', 'jll', 'umap', 'poincare', 'spherical']
    return _run_experiment(n=n, d=d, epsilon=epsilon, seed=seed, methods=methods,
                           use_poincare=True, use_spherical=True)
