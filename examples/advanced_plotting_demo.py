#!/usr/bin/env python3
"""
Advanced Plotting Demonstration for OrthoReduce

This script demonstrates the comprehensive plotting capabilities of the OrthoReduce
dimensionality reduction library, showcasing various visualization types including:

- Enhanced 2D/3D scatter plots with professional styling
- Method comparison visualizations (grid and overlay)
- Evaluation metric plots (trustworthiness, stress, correlation)
- Specialized embedding plots (spherical, hyperbolic)
- Interactive plotting capabilities with plotly
- Multi-metric dashboard views

Usage:
    python examples/advanced_plotting_demo.py
"""

import numpy as np
import logging
from pathlib import Path
import sys

# Add the parent directory to the path to import orthogonal_projection
sys.path.append(str(Path(__file__).parent.parent))

from orthogonal_projection.advanced_plotting import (
    AdvancedPlotter, InteractivePlotter, 
    plot_embedding_comparison, create_evaluation_report,
    quick_embedding_plot, plot_specialized_embedding
)
from orthogonal_projection.dimensionality_reduction import (
    run_experiment_with_visualization, generate_mixture_gaussians,
    run_pca, run_jll, run_umap, run_spherical, run_poincare
)
from orthogonal_projection.evaluation import comprehensive_evaluation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def demo_basic_plotting():
    """Demonstrate basic enhanced plotting capabilities."""
    logger.info("=" * 60)
    logger.info("DEMO 1: Basic Enhanced Plotting")
    logger.info("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n, d = 500, 50
    X = generate_mixture_gaussians(n, d, n_clusters=5, cluster_std=0.8)
    
    # Create embeddings using different methods
    k = 2  # 2D for visualization
    
    logger.info("Generating embeddings with multiple methods...")
    embeddings = {}
    
    # PCA
    Y_pca, _ = run_pca(X, k, seed=42)
    embeddings['PCA'] = Y_pca
    
    # JLL
    Y_jll, _ = run_jll(X, k, seed=42)
    embeddings['JLL'] = Y_jll
    
    # UMAP (if available)
    try:
        Y_umap, _ = run_umap(X, k, seed=42)
        embeddings['UMAP'] = Y_umap
    except Exception as e:
        logger.warning(f"UMAP not available: {e}")
    
    # Generate cluster labels for coloring
    labels = np.repeat(range(5), n // 5)[:n]
    
    # Create output directory
    output_dir = Path("demo_plots/basic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plotter = AdvancedPlotter(output_dir=str(output_dir), style="publication")
    
    # Individual embedding plots
    for method_name, embedding in embeddings.items():
        logger.info(f"Creating enhanced plot for {method_name}...")
        fig = plotter.plot_embedding_2d(
            embedding, labels, 
            title=f"Enhanced {method_name} Embedding",
            method_name=method_name,
            show_density=True,
            show_hull=True,
            save_path=str(output_dir / f"{method_name.lower()}_enhanced.png")
        )
        fig.show()
    
    # Method comparison grid
    logger.info("Creating method comparison grid...")
    fig_grid = plotter.plot_method_comparison_grid(
        embeddings, labels,
        title="Method Comparison - Enhanced Styling",
        save_path=str(output_dir / "method_comparison_grid.png")
    )
    fig_grid.show()
    
    # Method overlay comparison
    logger.info("Creating method overlay comparison...")
    fig_overlay = plotter.plot_method_overlay(
        embeddings,
        title="Method Overlay Comparison",
        save_path=str(output_dir / "method_overlay.png")
    )
    fig_overlay.show()
    
    logger.info(f"✅ Basic plotting demo completed! Files saved to: {output_dir}")

def demo_specialized_embeddings():
    """Demonstrate specialized embedding visualizations."""
    logger.info("=" * 60)
    logger.info("DEMO 2: Specialized Embedding Plots")
    logger.info("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n, d = 300, 20
    X = generate_mixture_gaussians(n, d, n_clusters=4, cluster_std=0.6)
    k = 3  # 3D for specialized embeddings
    
    # Create output directory
    output_dir = Path("demo_plots/specialized")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plotter = AdvancedPlotter(output_dir=str(output_dir), style="presentation")
    labels = np.repeat(range(4), n // 4)[:n]
    
    # 1. Spherical embedding
    logger.info("Creating spherical embedding...")
    try:
        Y_spherical, _ = run_spherical(X, k, seed=42)
        fig_sphere = plotter.plot_spherical_embedding(
            Y_spherical, labels,
            title="Spherical Embedding on Unit Sphere",
            radius=1.0,
            show_wireframe=True,
            hemisphere_only=False,
            save_path=str(output_dir / "spherical_embedding.png")
        )
        fig_sphere.show()
    except Exception as e:
        logger.warning(f"Spherical embedding failed: {e}")
    
    # 2. Poincaré disk embedding (2D)
    logger.info("Creating Poincaré disk embedding...")
    try:
        Y_poincare, _ = run_poincare(X, 2, seed=42)  # 2D for Poincaré disk
        fig_poincare = plotter.plot_poincare_disk(
            Y_poincare, labels,
            title="Poincaré Disk Hyperbolic Embedding",
            show_boundary=True,
            show_geodesics=True,
            save_path=str(output_dir / "poincare_disk.png")
        )
        fig_poincare.show()
    except Exception as e:
        logger.warning(f"Poincaré embedding failed: {e}")
    
    # 3. 3D embedding with convex hull
    logger.info("Creating 3D embedding with convex hull...")
    Y_3d, _ = run_pca(X, k, seed=42)
    fig_3d = plotter.plot_embedding_3d(
        Y_3d, labels,
        title="3D Embedding with Convex Hull",
        method_name="PCA",
        show_hull=True,
        save_path=str(output_dir / "3d_with_hull.png")
    )
    fig_3d.show()
    
    logger.info(f"✅ Specialized embeddings demo completed! Files saved to: {output_dir}")

def demo_evaluation_metrics():
    """Demonstrate evaluation metric visualizations."""
    logger.info("=" * 60)
    logger.info("DEMO 3: Evaluation Metrics Visualization")
    logger.info("=" * 60)
    
    # Run comprehensive experiment
    logger.info("Running comprehensive experiment...")
    results = {}
    embeddings = {}
    
    # Generate data
    np.random.seed(42)
    n, d = 400, 30
    X = generate_mixture_gaussians(n, d, n_clusters=3, cluster_std=0.7)
    k = 2
    
    # Run multiple methods and evaluate comprehensively
    methods = ['pca', 'jll', 'umap']
    
    for method in methods:
        logger.info(f"Evaluating {method}...")
        try:
            if method == 'pca':
                Y, runtime = run_pca(X, k, seed=42)
            elif method == 'jll':
                Y, runtime = run_jll(X, k, seed=42)
            elif method == 'umap':
                Y, runtime = run_umap(X, k, seed=42)
            
            # Comprehensive evaluation
            eval_results = comprehensive_evaluation(
                X, Y, 
                k_values=[5, 10, 20],
                sample_size=min(300, n),
                include_advanced=True
            )
            
            # Store results
            method_key = method.upper()
            results[method_key] = {
                'runtime': runtime,
                'compression_ratio': d / k,
                'mean_distortion': eval_results['distortion']['mean_distortion'],
                'max_distortion': eval_results['distortion']['max_distortion'],
                'rank_correlation': eval_results['basic_correlation'],
                'trustworthiness': eval_results['trustworthiness'],
                'continuity': eval_results['continuity'],
                'advanced_correlation': eval_results.get('advanced_correlation', {}),
                'sammon_stress': eval_results.get('sammon_stress', {}),
                'weighted_stress': eval_results.get('weighted_stress', {})
            }
            embeddings[method_key] = Y
            
        except Exception as e:
            logger.warning(f"Failed to evaluate {method}: {e}")\n    \n    if not results:\n        logger.error(\"No methods successfully evaluated!\")\n        return\n    \n    # Create output directory\n    output_dir = Path(\"demo_plots/evaluation\")\n    output_dir.mkdir(parents=True, exist_ok=True)\n    \n    plotter = AdvancedPlotter(output_dir=str(output_dir))\n    \n    # 1. Trustworthiness and continuity analysis\n    logger.info(\"Creating trustworthiness and continuity plots...\")\n    trust_cont_data = {}\n    for method, method_results in results.items():\n        trust_cont_data[method] = {\n            'trustworthiness': method_results.get('trustworthiness', {}),\n            'continuity': method_results.get('continuity', {})\n        }\n    \n    fig_trust = plotter.plot_trustworthiness_continuity(\n        trust_cont_data,\n        title=\"Neighborhood Preservation Analysis\",\n        save_path=str(output_dir / \"trustworthiness_continuity.png\")\n    )\n    fig_trust.show()\n    \n    # 2. Stress decomposition\n    logger.info(\"Creating stress decomposition plots...\")\n    stress_data = {}\n    for method, method_results in results.items():\n        if 'sammon_stress' in method_results:\n            stress_data[method] = method_results['sammon_stress']\n    \n    if stress_data:\n        fig_stress = plotter.plot_stress_decomposition(\n            stress_data,\n            title=\"Stress Analysis: Local vs Global\",\n            save_path=str(output_dir / \"stress_decomposition.png\")\n        )\n        fig_stress.show()\n    \n    # 3. Correlation heatmap\n    logger.info(\"Creating correlation heatmap...\")\n    correlation_data = {}\n    for method, method_results in results.items():\n        corr_dict = {\n            'spearman': method_results['rank_correlation']\n        }\n        if 'advanced_correlation' in method_results:\n            corr_dict.update(method_results['advanced_correlation'])\n        correlation_data[method] = corr_dict\n    \n    fig_corr = plotter.plot_correlation_heatmap(\n        correlation_data,\n        title=\"Method Correlation Analysis\",\n        save_path=str(output_dir / \"correlation_heatmap.png\")\n    )\n    fig_corr.show()\n    \n    # 4. Multi-metric dashboard\n    logger.info(\"Creating multi-metric dashboard...\")\n    fig_dashboard = plotter.create_multi_metric_dashboard(\n        results, embeddings,\n        title=\"Comprehensive Evaluation Dashboard\",\n        save_path=str(output_dir / \"multi_metric_dashboard.png\")\n    )\n    fig_dashboard.show()\n    \n    logger.info(f\"✅ Evaluation metrics demo completed! Files saved to: {output_dir}\")\n\ndef demo_interactive_plots():\n    \"\"\"Demonstrate interactive plotting capabilities.\"\"\"\n    logger.info(\"=\" * 60)\n    logger.info(\"DEMO 4: Interactive Plotting (Plotly)\")\n    logger.info(\"=\" * 60)\n    \n    try:\n        # Generate data\n        np.random.seed(42)\n        n, d = 300, 25\n        X = generate_mixture_gaussians(n, d, n_clusters=4, cluster_std=0.8)\n        k = 3\n        labels = np.repeat(range(4), n // 4)[:n]\n        \n        # Create embeddings\n        logger.info(\"Generating embeddings for interactive visualization...\")\n        embeddings = {}\n        \n        Y_pca, _ = run_pca(X, k, seed=42)\n        embeddings['PCA'] = Y_pca\n        \n        Y_jll, _ = run_jll(X, k, seed=42)\n        embeddings['JLL'] = Y_jll\n        \n        # Create output directory\n        output_dir = Path(\"demo_plots/interactive\")\n        output_dir.mkdir(parents=True, exist_ok=True)\n        \n        interactive_plotter = InteractivePlotter(output_dir=str(output_dir))\n        \n        # 1. Interactive 3D embedding\n        logger.info(\"Creating interactive 3D embedding...\")\n        fig_3d = interactive_plotter.plot_embedding_interactive(\n            embeddings['PCA'], labels,\n            title=\"Interactive 3D PCA Embedding\",\n            method_name=\"PCA\",\n            save_path=str(output_dir / \"interactive_3d_embedding.html\")\n        )\n        \n        # 2. Interactive method comparison with dropdown\n        logger.info(\"Creating interactive method comparison...\")\n        # Use 2D for comparison\n        embeddings_2d = {k: v[:, :2] for k, v in embeddings.items()}\n        fig_comparison = interactive_plotter.plot_method_comparison_interactive(\n            embeddings_2d, labels,\n            title=\"Interactive Method Comparison\",\n            save_path=str(output_dir / \"interactive_method_comparison.html\")\n        )\n        \n        # Note: Interactive plots are saved as HTML files\n        logger.info(f\"✅ Interactive plotting demo completed!\")\n        logger.info(f\"📁 HTML files saved to: {output_dir}\")\n        logger.info(\"📝 Open the HTML files in a web browser to view interactive plots\")\n        \n    except ImportError:\n        logger.warning(\"Plotly not available - skipping interactive demo\")\n    except Exception as e:\n        logger.error(f\"Interactive plotting demo failed: {e}\")\n\ndef demo_comprehensive_report():\n    \"\"\"Demonstrate comprehensive evaluation report generation.\"\"\"\n    logger.info(\"=\" * 60)\n    logger.info(\"DEMO 5: Comprehensive Evaluation Report\")\n    logger.info(\"=\" * 60)\n    \n    # Use the integrated experiment function with visualization\n    logger.info(\"Running experiment with comprehensive visualization...\")\n    \n    results, plot_files = run_experiment_with_visualization(\n        n=500, d=40, epsilon=0.2,\n        methods=['pca', 'jll', 'gaussian'],\n        use_mixture_gaussians=True,\n        n_clusters=4,\n        cluster_std=0.7,\n        output_dir=\"demo_plots/comprehensive_report\",\n        include_advanced_plots=True,\n        include_interactive=True\n    )\n    \n    if plot_files:\n        logger.info(\"✅ Comprehensive report generated!\")\n        logger.info(f\"📊 Total visualizations created: {len(plot_files)}\")\n        logger.info(\"📁 Generated files:\")\n        for plot_type, file_path in plot_files.items():\n            logger.info(f\"   {plot_type}: {file_path}\")\n    else:\n        logger.warning(\"Failed to generate comprehensive report\")\n\ndef main():\n    \"\"\"Run all plotting demonstrations.\"\"\"\n    logger.info(\"🎨 OrthoReduce Advanced Plotting Demonstration\")\n    logger.info(\"\" * 80)\n    \n    try:\n        # Run all demos\n        demo_basic_plotting()\n        demo_specialized_embeddings()\n        demo_evaluation_metrics()\n        demo_interactive_plots()\n        demo_comprehensive_report()\n        \n        logger.info(\"\" * 80)\n        logger.info(\"🎉 All plotting demonstrations completed successfully!\")\n        logger.info(\"📁 Check the 'demo_plots/' directory for generated visualizations\")\n        logger.info(\"📝 Interactive plots are saved as HTML files - open in web browser\")\n        \n    except KeyboardInterrupt:\n        logger.info(\"\\n⚠️ Demo interrupted by user\")\n    except Exception as e:\n        logger.error(f\"❌ Demo failed: {e}\")\n        raise\n\nif __name__ == \"__main__\":\n    main()\n"