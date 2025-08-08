#!/usr/bin/env python3
"""
Hyperbolic Embedding Demonstration

This script demonstrates the comprehensive Poincaré (hyperbolic) embedding
system implemented in OrthoReduce. It showcases:

1. Rigorous mathematical foundations with proper hyperbolic geometry
2. Riemannian optimization algorithms (RSGD and RAdam)  
3. Multiple loss functions optimized for hyperbolic space
4. Numerical stability and convergence analysis
5. Comparison with Euclidean methods on hierarchical data

The implementation follows established mathematical literature:
- Nickel & Kiela (2017): "Poincaré Embeddings for Learning Hierarchical Representations"
- Ganea et al. (2018): "Hyperbolic Neural Networks"  
- Chami et al. (2019): "Hyperbolic Graph Convolutional Neural Networks"
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import OrthoReduce functionality
from orthogonal_projection import (
    run_poincare, run_jll, run_pca, run_umap,
    generate_mixture_gaussians, compute_distortion, rank_correlation
)

# Try to import hyperbolic-specific functionality
try:
    from orthogonal_projection.hyperbolic import (
        PoincareBall, HyperbolicEmbedding, run_poincare_optimized
    )
    HYPERBOLIC_AVAILABLE = True
    logger.info("✓ Advanced hyperbolic functionality available")
except ImportError:
    HYPERBOLIC_AVAILABLE = False
    logger.warning("⚠ Advanced hyperbolic functionality not available, using fallback")


def generate_hierarchical_data(n_samples: int = 300, n_features: int = 50, 
                              seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic hierarchical data that should benefit from hyperbolic embedding.
    
    Creates a tree-like structure with multiple levels of hierarchy,
    simulating data that naturally lives in hyperbolic space.
    """
    np.random.seed(seed)
    
    # Root node
    root = np.random.randn(1, n_features) * 0.1
    
    # Level 1: Main branches
    n_main_branches = 4
    branch_data = []
    labels = []
    
    for branch_id in range(n_main_branches):
        # Branch center
        branch_center = root + np.random.randn(1, n_features) * 1.0
        
        # Level 2: Sub-branches
        n_sub_branches = 3
        for sub_id in range(n_sub_branches):
            sub_center = branch_center + np.random.randn(1, n_features) * 0.5
            
            # Level 3: Leaf nodes
            n_leaves = n_samples // (n_main_branches * n_sub_branches)
            leaves = sub_center + np.random.randn(n_leaves, n_features) * 0.2
            
            branch_data.append(leaves)
            labels.extend([branch_id * n_sub_branches + sub_id] * n_leaves)
    
    X = np.vstack(branch_data)
    y = np.array(labels)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    logger.info(f"Generated hierarchical data: {X.shape}, {len(np.unique(y))} clusters")
    return X, y


def demonstrate_hyperbolic_geometry():
    """Demonstrate core hyperbolic geometry operations."""
    if not HYPERBOLIC_AVAILABLE:
        logger.warning("Skipping hyperbolic geometry demo - module not available")
        return
        
    logger.info("=== Hyperbolic Geometry Operations Demo ===")
    
    # Create Poincaré ball
    ball = PoincareBall(c=1.0, dim=2)
    
    # Generate some points
    x = np.array([[0.1, 0.2]])
    y = np.array([[0.3, -0.1]])
    
    print(f"Point x: {x.flatten()}")
    print(f"Point y: {y.flatten()}")
    
    # Möbius addition
    xy = ball.mobius_add(x, y)
    print(f"x ⊕ y: {xy.flatten()}")
    
    # Hyperbolic distance
    d_hyp = ball.hyperbolic_distance(x, y).item()
    d_eucl = np.linalg.norm(x - y)
    print(f"Hyperbolic distance: {d_hyp:.4f}")
    print(f"Euclidean distance: {d_eucl:.4f}")
    
    # Exponential and logarithmic maps
    v = ball.logarithmic_map(x, y)
    y_reconstructed = ball.exponential_map(x, v)
    
    print(f"Tangent vector: {v.flatten()}")
    print(f"Reconstruction error: {np.linalg.norm(y - y_reconstructed):.2e}")
    
    # Conformal factor
    lambda_x = ball._lambda_c(x).item()
    print(f"Conformal factor λ_c^x: {lambda_x:.4f}")


def run_embedding_comparison(X: np.ndarray, y: np.ndarray, 
                           target_dim: int = 2) -> Dict[str, Dict]:
    """
    Compare different embedding methods on the same data.
    
    Args:
        X: Input data
        y: Labels (for visualization)
        target_dim: Target embedding dimension
        
    Returns:
        Dictionary with results for each method
    """
    logger.info("=== Embedding Method Comparison ===")
    
    methods = {
        'PCA': lambda X, k: run_pca(X, k, seed=42),
        'JLL': lambda X, k: run_jll(X, k, seed=42),
        'UMAP': lambda X, k: run_umap(X, k, seed=42),
        'Poincaré': lambda X, k: run_poincare(X, k, seed=42, n_epochs=30)
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        logger.info(f"Running {method_name}...")
        
        start_time = time.time()
        try:
            Y, runtime = method_func(X, target_dim)
            
            # Compute quality metrics
            mean_dist, max_dist, _, _ = compute_distortion(X, Y, sample_size=100)
            rank_corr = rank_correlation(X, Y, sample_size=100)
            
            results[method_name] = {
                'embedding': Y,
                'runtime': runtime,
                'mean_distortion': mean_dist,
                'max_distortion': max_dist,
                'rank_correlation': rank_corr,
                'total_time': time.time() - start_time
            }
            
            logger.info(f"{method_name}: corr={rank_corr:.3f}, "
                       f"dist={mean_dist:.3f}, time={runtime:.2f}s")
            
        except Exception as e:
            logger.error(f"{method_name} failed: {e}")
            results[method_name] = None
    
    return results


def demonstrate_loss_functions():
    """Demonstrate different loss functions for hyperbolic embedding."""
    if not HYPERBOLIC_AVAILABLE:
        logger.warning("Skipping loss function demo - module not available")
        return
        
    logger.info("=== Loss Function Comparison ===")
    
    # Generate labeled data
    X, y = generate_hierarchical_data(n_samples=150, n_features=20)
    
    loss_functions = ['stress', 'nca', 'triplet']
    results = {}
    
    for loss_fn in loss_functions:
        logger.info(f"Testing {loss_fn} loss...")
        
        embedding = HyperbolicEmbedding(
            n_components=3,
            c=1.0,
            loss_fn=loss_fn,
            n_epochs=15,
            lr=0.02,
            seed=42
        )
        
        try:
            if loss_fn in ['nca', 'triplet']:
                Y = embedding.fit_transform(X, y)
            else:
                Y = embedding.fit_transform(X)
            
            # Compute metrics
            rank_corr = rank_correlation(X, Y, sample_size=75)
            
            results[loss_fn] = {
                'embedding': Y,
                'rank_correlation': rank_corr,
                'loss_history': embedding.loss_history_
            }
            
            logger.info(f"{loss_fn}: final correlation = {rank_corr:.3f}")
            
        except Exception as e:
            logger.error(f"{loss_fn} failed: {e}")
            results[loss_fn] = None
    
    return results


def demonstrate_parameter_sensitivity():
    """Demonstrate sensitivity to key hyperparameters."""
    logger.info("=== Parameter Sensitivity Analysis ===")
    
    # Generate test data
    X = generate_mixture_gaussians(100, 15, n_clusters=4, seed=42)
    
    # Test curvature parameter
    curvatures = [0.1, 0.5, 1.0, 2.0]
    curvature_results = {}
    
    for c in curvatures:
        logger.info(f"Testing curvature c={c}...")
        Y, _ = run_poincare(X, 3, c=c, n_epochs=15, seed=42)
        rank_corr = rank_correlation(X, Y, sample_size=50)
        curvature_results[c] = rank_corr
        logger.info(f"c={c}: correlation = {rank_corr:.3f}")
    
    # Test learning rates
    learning_rates = [0.001, 0.01, 0.05, 0.1]
    lr_results = {}
    
    for lr in learning_rates:
        logger.info(f"Testing learning rate lr={lr}...")
        Y, _ = run_poincare(X, 3, lr=lr, n_epochs=15, seed=42)
        rank_corr = rank_correlation(X, Y, sample_size=50)
        lr_results[lr] = rank_corr
        logger.info(f"lr={lr}: correlation = {rank_corr:.3f}")
    
    # Find best parameters
    best_c = max(curvature_results, key=curvature_results.get)
    best_lr = max(lr_results, key=lr_results.get)
    
    logger.info(f"Best curvature: {best_c} (corr={curvature_results[best_c]:.3f})")
    logger.info(f"Best learning rate: {best_lr} (corr={lr_results[best_lr]:.3f})")
    
    return curvature_results, lr_results


def visualize_embeddings(results: Dict[str, Dict], y: np.ndarray = None):
    """
    Visualize 2D embeddings for comparison.
    
    Args:
        results: Results from run_embedding_comparison
        y: Optional labels for coloring
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Matplotlib not available, skipping visualization")
        return
    
    # Filter successful results and 2D embeddings
    valid_results = {name: res for name, res in results.items() 
                    if res is not None and res['embedding'].shape[1] == 2}
    
    if not valid_results:
        logger.warning("No 2D embeddings to visualize")
        return
    
    n_methods = len(valid_results)
    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4))
    
    if n_methods == 1:
        axes = [axes]
    
    for idx, (method_name, result) in enumerate(valid_results.items()):
        ax = axes[idx]
        Y = result['embedding']
        
        if y is not None:
            scatter = ax.scatter(Y[:, 0], Y[:, 1], c=y, cmap='tab10', s=30, alpha=0.7)
        else:
            scatter = ax.scatter(Y[:, 0], Y[:, 1], s=30, alpha=0.7)
        
        ax.set_title(f"{method_name}\nCorr: {result['rank_correlation']:.3f}")
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.grid(True, alpha=0.3)
        
        # For Poincaré, draw unit circle to show boundary
        if method_name == 'Poincaré' and HYPERBOLIC_AVAILABLE:
            circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--', alpha=0.5)
            ax.add_patch(circle)
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('hyperbolic_embedding_comparison.png', dpi=150, bbox_inches='tight')
    logger.info("Visualization saved as 'hyperbolic_embedding_comparison.png'")
    plt.show()


def benchmark_performance():
    """Benchmark performance on different data sizes."""
    logger.info("=== Performance Benchmark ===")
    
    sizes = [100, 200, 500] if HYPERBOLIC_AVAILABLE else [100, 200]
    dimensions = [10, 20, 50]
    target_dim = 3
    
    benchmark_results = {}
    
    for n in sizes:
        for d in dimensions:
            logger.info(f"Benchmarking n={n}, d={d}...")
            
            # Generate data
            X = generate_mixture_gaussians(n, d, n_clusters=min(5, n//20), seed=42)
            
            # Test methods
            methods_to_test = ['JLL', 'Poincaré']
            
            for method in methods_to_test:
                start_time = time.time()
                
                try:
                    if method == 'JLL':
                        Y, runtime = run_jll(X, target_dim, seed=42)
                    elif method == 'Poincaré':
                        Y, runtime = run_poincare(X, target_dim, n_epochs=10, seed=42)
                    
                    total_time = time.time() - start_time
                    
                    # Compute quality
                    sample_size = min(50, n)
                    rank_corr = rank_correlation(X, Y, sample_size=sample_size)
                    
                    key = f"{method}_{n}_{d}"
                    benchmark_results[key] = {
                        'method': method,
                        'n': n, 'd': d,
                        'runtime': runtime,
                        'total_time': total_time,
                        'rank_correlation': rank_corr
                    }
                    
                    logger.info(f"{method}: {total_time:.2f}s, corr={rank_corr:.3f}")
                    
                except Exception as e:
                    logger.error(f"{method} failed on n={n}, d={d}: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Method':<10} {'n':<6} {'d':<6} {'Time(s)':<10} {'Correlation':<12}")
    print("-"*60)
    
    for key, result in benchmark_results.items():
        print(f"{result['method']:<10} {result['n']:<6} {result['d']:<6} "
              f"{result['total_time']:<10.2f} {result['rank_correlation']:<12.3f}")
    
    return benchmark_results


def main():
    """Main demonstration function."""
    print("🌀 Poincaré (Hyperbolic) Embedding Demonstration")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Demonstrate hyperbolic geometry basics
    demonstrate_hyperbolic_geometry()
    print()
    
    # 2. Generate hierarchical test data
    logger.info("Generating hierarchical test data...")
    X, y = generate_hierarchical_data(n_samples=200, n_features=30)
    
    # 3. Compare embedding methods
    results = run_embedding_comparison(X, y, target_dim=2)
    print()
    
    # 4. Visualize results (if matplotlib available)
    visualize_embeddings(results, y)
    print()
    
    # 5. Demonstrate loss functions (if hyperbolic module available)
    if HYPERBOLIC_AVAILABLE:
        loss_results = demonstrate_loss_functions()
        print()
    
    # 6. Parameter sensitivity analysis
    curvature_results, lr_results = demonstrate_parameter_sensitivity()
    print()
    
    # 7. Performance benchmark
    benchmark_results = benchmark_performance()
    print()
    
    # 8. Summary and recommendations
    print("="*60)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*60)
    
    if HYPERBOLIC_AVAILABLE:
        print("✓ Full hyperbolic functionality is available")
        
        # Find best performing method
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_method = max(valid_results, key=lambda k: valid_results[k]['rank_correlation'])
            best_corr = valid_results[best_method]['rank_correlation']
            print(f"✓ Best method: {best_method} (correlation: {best_corr:.3f})")
        
        # Optimal parameters
        best_c = max(curvature_results, key=curvature_results.get)
        best_lr = max(lr_results, key=lr_results.get)
        print(f"✓ Recommended curvature: {best_c}")
        print(f"✓ Recommended learning rate: {best_lr}")
        
        print("\nFor hierarchical data:")
        print("• Use Poincaré embeddings with curvature c=1.0")
        print("• Start with 'stress' loss for unsupervised tasks")
        print("• Use 'nca' or 'triplet' loss for supervised tasks")
        print("• RAdam optimizer generally converges faster than RSGD")
        print("• Use regularization λ=0.01 to prevent boundary issues")
        
    else:
        print("⚠ Using fallback implementation")
        print("• Install full hyperbolic module for optimal performance")
        print("• Current implementation provides basic hyperbolic mapping")
        
    print("\nGeneral recommendations:")
    print("• Poincaré embeddings excel on hierarchical/tree-like data")
    print("• For general data, compare with PCA/JLL/UMAP")
    print("• Monitor rank correlation as primary quality metric")
    print("• Increase epochs for better convergence on complex data")


if __name__ == '__main__':
    main()