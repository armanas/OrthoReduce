#!/usr/bin/env python3
"""
Test script for enhanced spherical and hyperbolic plotting functionality.

This script demonstrates the new mathematically rigorous visualization features
for non-Euclidean embeddings in the OrthoReduce library.

Usage:
    python3 test_enhanced_plotting.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_test_data():
    """Generate sample data for testing visualizations."""
    np.random.seed(42)
    
    # Create synthetic 3D data with cluster structure
    n_points = 50
    n_clusters = 3
    
    data = []
    labels = []
    
    for i in range(n_clusters):
        # Generate cluster centers
        center = np.random.randn(3) * 2
        
        # Generate points around center
        cluster_points = center + np.random.randn(n_points // n_clusters, 3) * 0.5
        
        data.append(cluster_points)
        labels.extend([i] * (n_points // n_clusters))
    
    data = np.vstack(data)
    labels = np.array(labels)
    
    return data, labels

def test_spherical_visualization():
    """Test enhanced spherical embedding visualization."""
    logger.info("Testing enhanced spherical visualization...")
    
    from orthogonal_projection.advanced_plotting import AdvancedPlotter
    
    # Generate test data
    data, labels = generate_test_data()
    
    # Create spherical embedding (normalize to unit sphere)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    spherical_embedding = data / np.maximum(norms, 1e-10)
    
    # Create plotter
    plotter = AdvancedPlotter(output_dir="test_plots")
    
    # Test basic spherical plot
    fig1 = plotter.plot_spherical_embedding(
        spherical_embedding,
        labels=labels,
        title="Basic Spherical Embedding",
        save_path="test_plots/spherical_basic.png"
    )
    plt.close(fig1)
    
    # Test enhanced spherical plot with all features
    fig2 = plotter.plot_spherical_embedding(
        spherical_embedding,
        labels=labels,
        title="Enhanced Spherical Embedding",
        show_geodesics=True,
        show_great_circles=True,
        show_stereographic=True,
        mesh_quality=30,
        save_path="test_plots/spherical_enhanced.png"
    )
    plt.close(fig2)
    
    logger.info("✓ Spherical visualization tests completed")

def test_hyperbolic_visualization():
    """Test enhanced hyperbolic embedding visualization."""
    logger.info("Testing enhanced hyperbolic visualization...")
    
    from orthogonal_projection.advanced_plotting import AdvancedPlotter
    
    # Generate test data
    data, labels = generate_test_data()
    
    # Create Poincaré disk embedding (project to 2D and scale to unit disk)
    data_2d = data[:, :2]
    norms = np.linalg.norm(data_2d, axis=1, keepdims=True)
    max_norm = np.max(norms)
    hyperbolic_embedding = data_2d / max_norm * 0.8  # Scale to stay within unit disk
    
    # Create plotter
    plotter = AdvancedPlotter(output_dir="test_plots")
    
    # Test basic hyperbolic plot
    fig1 = plotter.plot_poincare_disk(
        hyperbolic_embedding,
        labels=labels,
        title="Basic Poincaré Disk",
        save_path="test_plots/hyperbolic_basic.png"
    )
    plt.close(fig1)
    
    # Test enhanced hyperbolic plot with all features
    fig2 = plotter.plot_poincare_disk(
        hyperbolic_embedding,
        labels=labels,
        title="Enhanced Poincaré Disk Embedding",
        curvature=1.0,
        show_geodesics=True,
        show_horocycles=True,
        show_klein_model=True,
        show_curvature_grid=True,
        save_path="test_plots/hyperbolic_enhanced.png"
    )
    plt.close(fig2)
    
    logger.info("✓ Hyperbolic visualization tests completed")

def test_geometric_comparison():
    """Test comprehensive geometric comparison visualization."""
    logger.info("Testing geometric comparison visualization...")
    
    from orthogonal_projection.advanced_plotting import AdvancedPlotter
    
    # Generate test data
    data, labels = generate_test_data()
    
    # Create different embedding types
    # Euclidean (PCA-like)
    euclidean_embedding = data[:, :2]
    
    # Spherical (normalized 3D)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    spherical_embedding = data / np.maximum(norms, 1e-10)
    
    # Hyperbolic (scaled 2D)
    data_2d = data[:, :2] 
    norms_2d = np.linalg.norm(data_2d, axis=1, keepdims=True)
    max_norm = np.max(norms_2d)
    hyperbolic_embedding = data_2d / max_norm * 0.7
    
    # Prepare embeddings and types
    embeddings = {
        'PCA': euclidean_embedding,
        'Spherical': spherical_embedding, 
        'Poincaré': hyperbolic_embedding
    }
    
    embedding_types = {
        'PCA': 'euclidean',
        'Spherical': 'spherical',
        'Poincaré': 'hyperbolic'
    }
    
    # Create plotter
    plotter = AdvancedPlotter(output_dir="test_plots")
    
    # Test geometric comparison
    fig = plotter.plot_geometric_comparison(
        embeddings,
        embedding_types,
        data,
        title="Geometric Embedding Comparison with Mathematical Analysis",
        save_path="test_plots/geometric_comparison.png"
    )
    plt.close(fig)
    
    logger.info("✓ Geometric comparison tests completed")

def test_curvature_effects():
    """Test curvature effects analysis visualization."""
    logger.info("Testing curvature effects visualization...")
    
    from orthogonal_projection.advanced_plotting import AdvancedPlotter
    
    # Generate smaller test data for curvature analysis
    np.random.seed(42)
    sample_data = np.random.randn(15, 3) * 0.5
    
    # Create plotter
    plotter = AdvancedPlotter(output_dir="test_plots")
    
    # Test curvature effects comparison
    fig = plotter.plot_curvature_effects_comparison(
        sample_data,
        curvatures=[0.0, 1.0, -1.0],
        title="Mathematical Effects of Curvature on Geometric Properties",
        save_path="test_plots/curvature_effects.png"
    )
    plt.close(fig)
    
    logger.info("✓ Curvature effects tests completed")

def test_convenience_functions():
    """Test convenience functions for specialized plotting."""
    logger.info("Testing convenience functions...")
    
    from orthogonal_projection.advanced_plotting import (
        plot_specialized_embedding,
        create_geometric_analysis_report,
        plot_educational_geometry_comparison
    )
    
    # Generate test data
    data, labels = generate_test_data()
    
    # Test specialized embedding plots
    
    # Spherical
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    spherical_embedding = data / np.maximum(norms, 1e-10)
    
    fig1 = plot_specialized_embedding(
        spherical_embedding,
        'spherical',
        labels=labels,
        show_geodesics=True,
        show_great_circles=True,
        save_path="test_plots/convenience_spherical.png"
    )
    plt.close(fig1)
    
    # Hyperbolic
    data_2d = data[:, :2]
    norms_2d = np.linalg.norm(data_2d, axis=1, keepdims=True)
    max_norm = np.max(norms_2d)
    hyperbolic_embedding = data_2d / max_norm * 0.8
    
    fig2 = plot_specialized_embedding(
        hyperbolic_embedding,
        'hyperbolic',
        labels=labels,
        show_geodesics=True,
        show_horocycles=True,
        save_path="test_plots/convenience_hyperbolic.png"
    )
    plt.close(fig2)
    
    # Test educational comparison
    sample_data = np.random.randn(10, 3) * 0.5
    fig3 = plot_educational_geometry_comparison(
        sample_data,
        title="Understanding Non-Euclidean Geometries",
        save_path="test_plots/educational_comparison.png"
    )
    plt.close(fig3)
    
    logger.info("✓ Convenience function tests completed")

def main():
    """Run all visualization tests."""
    logger.info("Starting enhanced plotting functionality tests...")
    
    # Create output directory
    Path("test_plots").mkdir(exist_ok=True)
    
    try:
        # Test individual visualization components
        test_spherical_visualization()
        test_hyperbolic_visualization()
        test_geometric_comparison()
        test_curvature_effects()
        test_convenience_functions()
        
        logger.info("🎉 All enhanced plotting tests completed successfully!")
        logger.info("Check the 'test_plots/' directory for generated visualizations.")
        
        # Print summary of created files
        plot_files = list(Path("test_plots").glob("*.png"))
        logger.info(f"Generated {len(plot_files)} visualization files:")
        for file in sorted(plot_files):
            logger.info(f"  - {file.name}")
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        raise

if __name__ == "__main__":
    main()