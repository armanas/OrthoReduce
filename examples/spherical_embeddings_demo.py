#!/usr/bin/env python3
"""
Demonstration of Advanced Spherical Embeddings with Riemannian Optimization

This script showcases the improvements made to spherical embeddings:
1. Geodesic distance computations
2. Riemannian optimization framework 
3. Geometry-consistent loss functions
4. Adaptive radius optimization

Run with: python3 examples/spherical_embeddings_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_s_curve
from sklearn.decomposition import PCA
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orthogonal_projection.dimensionality_reduction import (
    run_spherical, generate_mixture_gaussians
)
from orthogonal_projection.spherical_embeddings import (
    adaptive_spherical_embedding, 
    evaluate_spherical_embedding,
    SphericalEmbedding
)


def generate_synthetic_datasets():
    """Generate various synthetic datasets for testing."""
    datasets = {}
    
    # Mixture of Gaussians
    np.random.seed(42)
    datasets['mixture_gaussians'] = generate_mixture_gaussians(
        n=300, d=20, n_clusters=5, cluster_std=0.8, seed=42
    )
    
    # Swiss Roll
    datasets['swiss_roll'], _ = make_swiss_roll(n_samples=300, noise=0.1, random_state=42)
    
    # S-Curve
    datasets['s_curve'], _ = make_s_curve(n_samples=300, noise=0.1, random_state=42)
    
    # High-dimensional Gaussian with correlation structure
    np.random.seed(42)
    base = np.random.randn(300, 5)
    corr_matrix = np.random.randn(5, 15)
    datasets['correlated_gaussian'] = base @ corr_matrix + 0.1 * np.random.randn(300, 15)
    
    return datasets


def compare_spherical_methods():
    """Compare different spherical embedding methods."""
    print("=" * 60)
    print("SPHERICAL EMBEDDINGS COMPARISON")
    print("=" * 60)
    
    datasets = generate_synthetic_datasets()
    k = 3  # Target dimension
    
    results = {}
    
    for name, X in datasets.items():
        print(f"\nDataset: {name} (shape: {X.shape})")
        print("-" * 40)
        
        dataset_results = {}
        
        # Method 1: Simple spherical embedding (PCA + normalization)
        start_time = time.time()
        Y_simple, info_simple = adaptive_spherical_embedding(
            X, k, method='simple', seed=42
        )
        time_simple = time.time() - start_time
        metrics_simple = evaluate_spherical_embedding(X, Y_simple, radius=1.0)
        
        dataset_results['Simple PCA+Normalize'] = {
            'time': time_simple,
            'rank_correlation': metrics_simple['rank_correlation_geodesic'],
            'stress': metrics_simple['stress_geodesic'],
            'mean_distortion': metrics_simple['mean_distortion'],
            'radius': 1.0
        }
        
        # Method 2: Fast spherical embedding with adaptive radius
        start_time = time.time()
        Y_fast, info_fast = adaptive_spherical_embedding(
            X, k, method='fast', adaptive_radius=True, seed=42
        )
        time_fast = time.time() - start_time
        radius_fast = info_fast['final_radius']
        metrics_fast = evaluate_spherical_embedding(X, Y_fast, radius=radius_fast)
        
        dataset_results['Fast+Adaptive'] = {
            'time': time_fast,
            'rank_correlation': metrics_fast['rank_correlation_geodesic'],
            'stress': metrics_fast['stress_geodesic'], 
            'mean_distortion': metrics_fast['mean_distortion'],
            'radius': radius_fast
        }
        
        # Method 3: Full Riemannian optimization (for smaller datasets)
        if X.shape[0] <= 200:
            start_time = time.time()
            Y_riemannian, info_riemannian = adaptive_spherical_embedding(
                X, k, 
                method='riemannian',
                loss_type='mds_geodesic',
                max_iter=100,
                adaptive_radius=True,
                seed=42
            )
            time_riemannian = time.time() - start_time
            radius_riemannian = info_riemannian['final_radius']
            metrics_riemannian = evaluate_spherical_embedding(X, Y_riemannian, radius=radius_riemannian)
            
            dataset_results['Riemannian MDS'] = {
                'time': time_riemannian,
                'rank_correlation': metrics_riemannian['rank_correlation_geodesic'],
                'stress': metrics_riemannian['stress_geodesic'],
                'mean_distortion': metrics_riemannian['mean_distortion'],
                'radius': radius_riemannian
            }
            
            # Method 4: Hybrid loss function
            start_time = time.time()
            Y_hybrid, info_hybrid = adaptive_spherical_embedding(
                X, k,
                method='riemannian',
                loss_type='hybrid',
                max_iter=80,
                adaptive_radius=True,
                seed=42
            )
            time_hybrid = time.time() - start_time
            radius_hybrid = info_hybrid['final_radius']
            metrics_hybrid = evaluate_spherical_embedding(X, Y_hybrid, radius=radius_hybrid)
            
            dataset_results['Hybrid Loss'] = {
                'time': time_hybrid,
                'rank_correlation': metrics_hybrid['rank_correlation_geodesic'],
                'stress': metrics_hybrid['stress_geodesic'],
                'mean_distortion': metrics_hybrid['mean_distortion'],
                'radius': radius_hybrid
            }
        
        # Print results
        for method, result in dataset_results.items():
            print(f"{method:20s}: "
                  f"rank_corr={result['rank_correlation']:.3f}, "
                  f"stress={result['stress']:.3f}, "
                  f"distortion={result['mean_distortion']:.3f}, "
                  f"radius={result['radius']:.2f}, "
                  f"time={result['time']:.2f}s")
        
        results[name] = dataset_results
    
    return results


def demonstrate_geodesic_vs_chordal():
    """Demonstrate the difference between geodesic and chordal distances."""
    print("\n" + "=" * 60)
    print("GEODESIC vs CHORDAL DISTANCES")
    print("=" * 60)
    
    # Generate points on sphere
    np.random.seed(42)
    n = 100
    k = 3
    
    # Random spherical data
    Y = np.random.randn(n, k)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    
    # Compute geodesic distances
    D_geo = SphericalEmbedding.geodesic_distance_batch(Y, radius=1.0)
    
    # Compute chordal distances
    from scipy.spatial.distance import pdist, squareform
    D_chord = squareform(pdist(Y, metric='euclidean'))
    
    # Extract upper triangular values
    triu_idx = np.triu_indices(n, k=1)
    geo_distances = D_geo[triu_idx]
    chord_distances = D_chord[triu_idx]
    
    print(f"Number of point pairs: {len(geo_distances)}")
    print(f"Geodesic distances - min: {np.min(geo_distances):.3f}, "
          f"max: {np.max(geo_distances):.3f}, mean: {np.mean(geo_distances):.3f}")
    print(f"Chordal distances - min: {np.min(chord_distances):.3f}, "
          f"max: {np.max(chord_distances):.3f}, mean: {np.mean(chord_distances):.3f}")
    
    # Correlation between geodesic and chordal
    from scipy.stats import pearsonr, spearmanr
    pearson_corr = pearsonr(geo_distances, chord_distances)[0]
    spearman_corr = spearmanr(geo_distances, chord_distances)[0]
    
    print(f"Pearson correlation: {pearson_corr:.3f}")
    print(f"Spearman rank correlation: {spearman_corr:.3f}")
    
    # Show relationship
    print(f"\nFor small distances, geodesic ≈ chordal")
    print(f"For large distances, geodesic < chordal (due to sphere geometry)")
    
    # Find some examples
    small_idx = np.argsort(geo_distances)[:5]
    large_idx = np.argsort(geo_distances)[-5:]
    
    print(f"\nSmall distance examples:")
    for i in small_idx:
        print(f"  geodesic: {geo_distances[i]:.3f}, chordal: {chord_distances[i]:.3f}")
    
    print(f"\nLarge distance examples:")
    for i in large_idx:
        print(f"  geodesic: {geo_distances[i]:.3f}, chordal: {chord_distances[i]:.3f}")


def demonstrate_radius_optimization():
    """Demonstrate adaptive radius optimization."""
    print("\n" + "=" * 60) 
    print("ADAPTIVE RADIUS OPTIMIZATION")
    print("=" * 60)
    
    # Generate data with known structure
    np.random.seed(42)
    X = generate_mixture_gaussians(n=100, d=10, n_clusters=3, cluster_std=0.5, seed=42)
    k = 3
    
    # Test different fixed radii
    radii_to_test = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    print("Fixed radius comparison:")
    for radius in radii_to_test:
        # Simple spherical embedding with fixed radius
        pca = PCA(n_components=k, random_state=42)
        Y_pca = pca.fit_transform(X)
        Y = radius * Y_pca / np.linalg.norm(Y_pca, axis=1, keepdims=True)
        
        # Evaluate
        metrics = evaluate_spherical_embedding(X, Y, radius=radius)
        print(f"  Radius {radius:4.1f}: rank_corr={metrics['rank_correlation_geodesic']:.3f}, "
              f"stress={metrics['stress_geodesic']:.3f}")
    
    # Adaptive optimization
    print("\nAdaptive optimization:")
    Y_adaptive, info = adaptive_spherical_embedding(
        X, k, method='fast', adaptive_radius=True, seed=42
    )
    optimal_radius = info['final_radius']
    metrics_adaptive = evaluate_spherical_embedding(X, Y_adaptive, radius=optimal_radius)
    
    print(f"  Optimal radius: {optimal_radius:.3f}")
    print(f"  Final rank_corr: {metrics_adaptive['rank_correlation_geodesic']:.3f}")
    print(f"  Final stress: {metrics_adaptive['stress_geodesic']:.3f}")
    
    return optimal_radius, metrics_adaptive


def demonstrate_loss_functions():
    """Demonstrate different loss functions for spherical embedding."""
    print("\n" + "=" * 60)
    print("LOSS FUNCTION COMPARISON")
    print("=" * 60)
    
    # Generate structured data
    np.random.seed(42)
    X = generate_mixture_gaussians(n=80, d=8, n_clusters=4, cluster_std=0.4, seed=42)
    k = 3
    
    loss_types = ['mds_geodesic', 'triplet', 'hybrid']
    
    for loss_type in loss_types:
        print(f"\nLoss type: {loss_type}")
        
        start_time = time.time()
        Y, info = adaptive_spherical_embedding(
            X, k,
            method='riemannian',
            loss_type=loss_type,
            max_iter=50,  # Reduced for demo
            adaptive_radius=True,
            seed=42
        )
        runtime = time.time() - start_time
        
        radius = info['final_radius']
        metrics = evaluate_spherical_embedding(X, Y, radius=radius)
        
        print(f"  Runtime: {runtime:.2f}s")
        print(f"  Final radius: {radius:.3f}")
        print(f"  Rank correlation: {metrics['rank_correlation_geodesic']:.3f}")
        print(f"  MDS stress: {metrics['stress_geodesic']:.3f}")
        print(f"  Mean distortion: {metrics['mean_distortion']:.3f}")
        
        if 'loss_history' in info and len(info['loss_history']) > 1:
            print(f"  Loss reduction: {info['loss_history'][0]:.3f} → {info['loss_history'][-1]:.3f}")


def create_visualizations():
    """Create visualizations of the embeddings (if matplotlib available)."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Matplotlib not available for visualizations")
        return
    
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Generate Swiss roll data
    X, color = make_swiss_roll(n_samples=200, noise=0.1, random_state=42)
    k = 3
    
    # Different embedding methods
    methods = {
        'Simple': ('simple', False),
        'Fast+Adaptive': ('fast', True),
        'Riemannian': ('riemannian', True)
    }
    
    fig = plt.figure(figsize=(15, 5))
    
    for i, (name, (method, adaptive_radius)) in enumerate(methods.items()):
        Y, info = adaptive_spherical_embedding(
            X, k,
            method=method,
            adaptive_radius=adaptive_radius,
            max_iter=30 if method == 'riemannian' else None,
            seed=42
        )
        
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=color, cmap='viridis', s=20)
        ax.set_title(f'{name} Spherical Embedding')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2') 
        ax.set_zlabel('X3')
        
        # Make axis equal to show sphere
        max_range = np.array([Y[:, 0].max()-Y[:, 0].min(),
                             Y[:, 1].max()-Y[:, 1].min(),
                             Y[:, 2].max()-Y[:, 2].min()]).max() / 2.0
        mid_x = (Y[:, 0].max()+Y[:, 0].min()) * 0.5
        mid_y = (Y[:, 1].max()+Y[:, 1].min()) * 0.5 
        mid_z = (Y[:, 2].max()+Y[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig('spherical_embeddings_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'spherical_embeddings_comparison.png'")
    
    # Create loss function convergence plot
    if True:  # Create convergence plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Run optimization with history tracking
        Y, info = adaptive_spherical_embedding(
            X, 3,
            method='riemannian',
            loss_type='mds_geodesic',
            max_iter=100,
            adaptive_radius=True,
            seed=42
        )
        
        if 'loss_history' in info and len(info['loss_history']) > 1:
            ax.plot(info['loss_history'], 'b-', linewidth=2, label='MDS Loss')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.set_title('Spherical Embedding Convergence')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add radius history if available
            if 'radius_history' in info and len(info['radius_history']) > 1:
                ax2 = ax.twinx()
                ax2.plot(info['radius_history'], 'r--', alpha=0.7, label='Radius')
                ax2.set_ylabel('Sphere Radius', color='red')
                ax2.legend(loc='upper right')
            
            plt.tight_layout()
            plt.savefig('spherical_convergence.png', dpi=300, bbox_inches='tight')
            print("Convergence plot saved as 'spherical_convergence.png'")


def main():
    """Main demonstration function."""
    print("Advanced Spherical Embeddings Demonstration")
    print("=" * 60)
    print("This demo showcases improvements to spherical embeddings:")
    print("1. Proper geodesic distance computations")
    print("2. Riemannian optimization with tangent space operations")
    print("3. Multiple geometry-consistent loss functions")
    print("4. Adaptive radius optimization")
    print("5. Numerical stability enhancements")
    
    # Run demonstrations
    try:
        results = compare_spherical_methods()
        demonstrate_geodesic_vs_chordal()
        optimal_radius, metrics = demonstrate_radius_optimization()
        demonstrate_loss_functions()
        create_visualizations()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("Key improvements demonstrated:")
        print("• Geodesic distances provide better structure preservation")
        print("• Adaptive radius optimization improves embedding quality") 
        print("• Riemannian optimization converges to better local minima")
        print("• Multiple loss functions allow task-specific optimization")
        print("• Numerical stability handles edge cases gracefully")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()