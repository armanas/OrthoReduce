#!/usr/bin/env python3
"""
OrthoReduce Enhanced - Professional Dimensionality Reduction with Advanced Features

This enhanced version integrates all the latest OrthoReduce improvements including:
- Advanced visualization system with publication-ready plots
- Comprehensive evaluation metrics (trustworthiness, continuity)
- Post-processing calibration (isotonic regression, Procrustes)
- Enhanced convex hull projection with ridge regularization
- Specialized spherical and hyperbolic embedding visualizations
- Interactive dashboard capabilities
- Experiment orchestration and staged pipelines

Author: OrthoReduce Team
Version: 2.0.0 (Enhanced)
License: See LICENSE file
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import pandas as pd
import numpy as np

# Import enhanced OrthoReduce modules
from orthogonal_projection.dimensionality_reduction import run_experiment_with_visualization
from orthogonal_projection.advanced_plotting import AdvancedPlotter, create_evaluation_report
from orthogonal_projection.monitoring import (
    experiment_monitoring, 
    format_performance_report,
    print_system_info,
    get_system_info
)

# Optional import for experiment orchestration
try:
    from orthogonal_projection.experiment_orchestration import run_synthetic_data_experiment
    ORCHESTRATION_AVAILABLE = True
except ImportError:
    ORCHESTRATION_AVAILABLE = False
    print("⚠️  Experiment orchestration unavailable (missing dependencies)")
    print("   Staged pipeline mode will be disabled")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Available methods and their descriptions
AVAILABLE_METHODS = {
    'jll': 'Johnson-Lindenstrauss Lemma based projection',
    'pca': 'Principal Component Analysis', 
    'gaussian': 'Random Gaussian projection',
    'pocs': 'Projection Onto Convex Sets (enhanced with ridge regularization)',
    'poincare': 'Poincaré disk model projection (hyperbolic geometry)',
    'spherical': 'Spherical projection (Riemannian optimization)',
    'umap': 'Uniform Manifold Approximation and Projection'
}

# Default enhanced configuration
DEFAULT_CONFIG = {
    'dataset_size': 500,
    'dimensions': 50, 
    'epsilon': 0.2,
    'methods': ['jll', 'pca', 'gaussian', 'pocs'],
    'output_dir': 'enhanced_experiment_results',
    'use_advanced_plots': True,
    'use_interactive': False,
    'use_comprehensive_eval': True,
    'use_calibration': True,
    'use_staged_pipeline': False,
    'launch_dashboard': False,
    'enable_monitoring': True,
    'show_system_stats': True,
    'monitoring_interval': 0.5
}


class OrthoReduceEnhancedError(Exception):
    """Custom exception for OrthoReduce Enhanced-specific errors."""
    pass


class EnhancedExperimentRunner:
    """
    Enhanced experiment runner with advanced visualization and evaluation capabilities.
    
    This class provides access to all the latest OrthoReduce improvements including
    publication-ready visualizations, comprehensive evaluation metrics, and advanced
    geometric embeddings.
    """
    
    def __init__(
        self,
        dataset_size: int,
        dimensions: int, 
        epsilon: float,
        methods: List[str],
        output_dir: str,
        use_advanced_plots: bool = True,
        use_interactive: bool = False,
        use_comprehensive_eval: bool = True,
        use_calibration: bool = True,
        use_staged_pipeline: bool = False,
        launch_dashboard: bool = False,
        enable_monitoring: bool = True,
        show_system_stats: bool = True
    ) -> None:
        """
        Initialize the enhanced experiment runner.
        
        Args:
            dataset_size: Number of data points to generate
            dimensions: Original dimensionality of the data
            epsilon: Johnson-Lindenstrauss distortion parameter  
            methods: List of dimensionality reduction methods to test
            output_dir: Directory to save results
            use_advanced_plots: Use enhanced visualization system
            use_interactive: Create interactive HTML plots
            use_comprehensive_eval: Include trustworthiness/continuity metrics
            use_calibration: Apply post-processing calibration
            use_staged_pipeline: Use orchestrated multi-stage pipeline
            launch_dashboard: Launch web dashboard after experiment
            enable_monitoring: Enable comprehensive progress monitoring
            show_system_stats: Show real-time system resource usage
            
        Raises:
            OrthoReduceEnhancedError: If parameters are invalid
        """
        self._validate_parameters(dataset_size, dimensions, epsilon, methods)
        
        self.dataset_size = dataset_size
        self.dimensions = dimensions
        self.epsilon = epsilon
        self.methods = methods
        self.output_dir = Path(output_dir)
        self.use_advanced_plots = use_advanced_plots
        self.use_interactive = use_interactive
        self.use_comprehensive_eval = use_comprehensive_eval
        self.use_calibration = use_calibration
        self.use_staged_pipeline = use_staged_pipeline
        self.launch_dashboard = launch_dashboard
        self.enable_monitoring = enable_monitoring
        self.show_system_stats = show_system_stats
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("🚀 OrthoReduce Enhanced Initialized")
        print(f"   Advanced Features: {self._get_features_summary()}")
        
    def _validate_parameters(
        self, 
        dataset_size: int,
        dimensions: int, 
        epsilon: float,
        methods: List[str]
    ) -> None:
        """Validate experiment parameters."""
        if dataset_size < 10:
            raise OrthoReduceEnhancedError(f"Dataset size must be >= 10, got {dataset_size}")
        
        if dimensions < 2:
            raise OrthoReduceEnhancedError(f"Dimensions must be >= 2, got {dimensions}")
        
        if not 0.01 <= epsilon <= 1.0:
            raise OrthoReduceEnhancedError(f"Epsilon must be between 0.01 and 1.0, got {epsilon}")
        
        if not methods:
            raise OrthoReduceEnhancedError("At least one method must be specified")
        
        invalid_methods = [m for m in methods if m not in AVAILABLE_METHODS]
        if invalid_methods:
            raise OrthoReduceEnhancedError(
                f"Invalid methods: {invalid_methods}. "
                f"Available methods: {list(AVAILABLE_METHODS.keys())}"
            )
    
    def _get_features_summary(self) -> str:
        """Get summary of enabled enhanced features."""
        features = []
        if self.use_advanced_plots:
            features.append("Advanced Plots")
        if self.use_interactive:
            features.append("Interactive Viz")
        if self.use_comprehensive_eval:
            features.append("Comprehensive Eval")
        if self.use_calibration:
            features.append("Calibration")
        if self.use_staged_pipeline:
            features.append("Staged Pipeline")
        if self.launch_dashboard:
            features.append("Dashboard")
        if self.enable_monitoring:
            features.append("Progress Monitoring")
        return ", ".join(features) if features else "Basic Mode"
    
    def run_experiment(self) -> Dict[str, Any]:
        """
        Execute the enhanced dimensionality reduction experiment.
        
        Returns:
            Dictionary containing experimental results, plot files, and metadata
            
        Raises:
            OrthoReduceEnhancedError: If experiment execution fails
        """
        print("🔬 Starting Enhanced OrthoReduce Experiment...")
        print(f"📊 Configuration: n={self.dataset_size}, d={self.dimensions}, ε={self.epsilon}")
        print(f"🔧 Methods: {', '.join(self.methods)}")
        print(f"📁 Output directory: {self.output_dir}")
        # Show system information if monitoring enabled
        if self.enable_monitoring and self.show_system_stats:
            print()
            print_system_info()
        
        print()
        
        # Initialize monitoring
        performance_summary = {}
        
        try:
            # Use comprehensive monitoring for experiment execution
            with experiment_monitoring(
                methods=self.methods,
                show_system_stats=self.enable_monitoring and self.show_system_stats,
                show_progress_bars=self.enable_monitoring
            ) as monitor:
                
                if self.use_staged_pipeline and ORCHESTRATION_AVAILABLE:
                    # Use advanced orchestration system with monitoring
                    print("🎯 Using staged optimization pipeline...")
                    
                    # Note: For orchestration, we'll add a simple overall progress indicator
                    monitor.start_method("Orchestrated Pipeline", 
                                       data_points=self.dataset_size, 
                                       dimensions=self.dimensions)
                    
                    results = run_synthetic_data_experiment(
                        experiment_name="enhanced_experiment",
                        n_samples=self.dataset_size,
                        n_features=self.dimensions,
                        template="comprehensive" if self.use_comprehensive_eval else "fast",
                        methods=self.methods,
                        output_dir=str(self.output_dir)
                    )
                    
                    monitor.complete_method({'stage': 'orchestration_complete'})
                    
                    # Extract results in compatible format
                    experiment_results = results.get('experiment_results', {})
                    plot_files = results.get('visualization_files', {})
                    
                elif self.use_staged_pipeline and not ORCHESTRATION_AVAILABLE:
                    print("⚠️  Staged pipeline requested but unavailable, using enhanced visualization mode...")
                    
                    experiment_results, plot_files = self._run_monitored_experiment(
                        monitor, use_visualization=True
                    )
                    
                else:
                    # Use enhanced visualization system with detailed monitoring
                    print("📈 Using enhanced visualization system...")
                    
                    experiment_results, plot_files = self._run_monitored_experiment(
                        monitor, use_visualization=True
                    )
                
                # Get performance summary from monitoring
                performance_summary = monitor.get_performance_summary()
            
            # Combine results with monitoring data
            combined_results = {
                'experiment_results': experiment_results,
                'plot_files': plot_files or {},
                'performance_summary': performance_summary,
                '_metadata': {
                    'dataset_size': self.dataset_size,
                    'dimensions': self.dimensions, 
                    'epsilon': self.epsilon,
                    'methods': self.methods,
                    'enhanced_features': self._get_features_summary(),
                    'output_directory': str(self.output_dir),
                    'system_info': get_system_info() if self.enable_monitoring else {},
                    'monitoring_enabled': self.enable_monitoring
                }
            }
            
            # Print performance summary if monitoring was enabled
            if self.enable_monitoring and performance_summary:
                print("\n" + "="*60)
                print(format_performance_report(performance_summary))
                print("="*60 + "\n")
            
            logger.info("Enhanced experiment completed successfully")
            return combined_results
            
        except Exception as e:
            error_msg = f"Enhanced experiment failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise OrthoReduceEnhancedError(error_msg) from e
    
    def _run_monitored_experiment(self, monitor, use_visualization: bool = True):
        """
        Run experiment with detailed method-by-method monitoring.
        
        Args:
            monitor: ProgressTracker instance for monitoring
            use_visualization: Whether to use visualization system
            
        Returns:
            Tuple of (experiment_results, plot_files)
        """
        if use_visualization:
            # Import here to avoid circular imports and provide better error messages
            from orthogonal_projection.dimensionality_reduction import (
                generate_mixture_gaussians, jll_dimension, run_pca, run_jll, 
                run_gaussian_projection, run_umap, run_pocs, run_poincare, 
                run_spherical, evaluate_projection_comprehensive
            )
            
            # Generate data (replicating what run_experiment_with_visualization does)
            monitor.start_method("Data Generation", data_points=self.dataset_size, dimensions=self.dimensions)
            monitor.update_method_progress(10, "Generating synthetic data...")
            
            X = generate_mixture_gaussians(self.dataset_size, self.dimensions, seed=42)
            
            # Normalize data
            monitor.update_method_progress(30, "Normalizing data...")
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            safe_norms = np.where(norms > 1e-10, norms, 1.0)
            X = X / safe_norms
            
            monitor.update_method_progress(50, "Calculating target dimension...")
            k = min(jll_dimension(self.dataset_size, self.epsilon), self.dimensions)
            
            monitor.update_method_progress(100, "Data preparation complete")
            monitor.complete_method({'stage': 'data_generation', 'target_dimension': k})
            
            # Run each method with detailed monitoring
            experiment_results = {}
            
            for method_name in self.methods:
                monitor.start_method(method_name.upper(), data_points=self.dataset_size, dimensions=k)
                
                try:
                    monitor.update_method_progress(10, f"Initializing {method_name}...")
                    
                    # Run the method
                    if method_name == 'pca':
                        monitor.update_method_progress(30, "Running PCA decomposition...")
                        Y, runtime = run_pca(X, k, seed=42)
                    elif method_name == 'jll':
                        monitor.update_method_progress(30, "Generating orthogonal basis...")
                        monitor.update_method_progress(60, "Projecting data...")
                        Y, runtime = run_jll(X, k, seed=42, method='auto')
                    elif method_name == 'gaussian':
                        monitor.update_method_progress(30, "Creating Gaussian projection...")
                        Y, runtime = run_gaussian_projection(X, k, seed=42)
                    elif method_name == 'umap':
                        monitor.update_method_progress(30, "Building neighborhood graph...")
                        monitor.update_method_progress(60, "Optimizing embedding...")
                        Y, runtime = run_umap(X, k, seed=42)
                    elif method_name == 'pocs':
                        monitor.update_method_progress(30, "JLL projection...")
                        monitor.update_method_progress(60, "Convex hull projection...")
                        Y, runtime = run_pocs(X, k, seed=42)
                    elif method_name == 'poincare':
                        monitor.update_method_progress(30, "Hyperbolic embedding...")
                        Y, runtime = run_poincare(X, k, seed=42)
                    elif method_name == 'spherical':
                        monitor.update_method_progress(30, "Spherical projection...")
                        Y, runtime = run_spherical(X, k, seed=42)
                    else:
                        logger.warning(f"Unknown method: {method_name}")
                        continue
                    
                    monitor.update_method_progress(80, "Evaluating projection quality...")
                    
                    # Evaluate projection
                    if self.use_comprehensive_eval:
                        metrics = evaluate_projection_comprehensive(X, Y, sample_size=2000)
                    else:
                        from orthogonal_projection.dimensionality_reduction import evaluate_projection
                        metrics = evaluate_projection(X, Y, sample_size=2000)
                    
                    monitor.update_method_progress(95, "Finalizing results...")
                    
                    # Add runtime and compression info
                    result = {**metrics, 'runtime': runtime, 'compression_ratio': self.dimensions / k}
                    experiment_results[method_name.upper()] = result
                    
                    monitor.update_method_progress(100, "Complete")
                    monitor.complete_method(result)
                    
                except Exception as e:
                    logger.error(f"Method {method_name} failed: {e}")
                    monitor.complete_method({'error': str(e)})
            
            # Create visualizations if requested
            plot_files = {}
            if self.use_advanced_plots:
                monitor.start_method("Visualization", data_points=len(experiment_results))
                monitor.update_method_progress(20, "Creating advanced plots...")
                
                try:
                    from orthogonal_projection.visualization import OrthoReduceVisualizer
                    visualizer = OrthoReduceVisualizer(output_dir=str(self.output_dir))
                    
                    monitor.update_method_progress(60, "Generating plot suite...")
                    plot_files = visualizer.create_complete_visualization(experiment_results)
                    
                    monitor.update_method_progress(100, "Visualizations complete")
                    monitor.complete_method({'plots_generated': len(plot_files)})
                    
                except Exception as e:
                    logger.warning(f"Visualization failed: {e}")
                    monitor.complete_method({'error': str(e)})
            
            return experiment_results, plot_files
        
        else:
            # Fallback to original method without detailed monitoring
            return run_experiment_with_visualization(
                n=self.dataset_size,
                d=self.dimensions,
                epsilon=self.epsilon,
                methods=self.methods,
                output_dir=str(self.output_dir),
                include_advanced_plots=self.use_advanced_plots,
                include_interactive=self.use_interactive
            )
    
    def save_results(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Save enhanced experiment results to multiple formats.
        
        Args:
            results: Enhanced experiment results dictionary
            
        Returns:
            Dictionary mapping file types to saved file paths
        """
        saved_files = {}
        experiment_results = results.get('experiment_results', {})
        plot_files = results.get('plot_files', {})
        metadata = results.get('_metadata', {})
        
        try:
            # Save JSON results
            json_file = self.output_dir / 'enhanced_results.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            saved_files['json'] = str(json_file)
            print(f"✅ Enhanced results saved to: {json_file}")
            
            # Save CSV results (compatible format)
            if experiment_results:
                csv_file = self.output_dir / 'results.csv'
                df_data = []
                for method, metrics in experiment_results.items():
                    if method != '_metadata' and isinstance(metrics, dict):
                        row = {'method': method}
                        # Flatten nested metrics
                        for key, value in metrics.items():
                            if isinstance(value, (int, float, str, bool)):
                                row[key] = value
                            elif isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    if isinstance(subvalue, (int, float, str, bool)):
                                        row[f"{key}_{subkey}"] = subvalue
                        df_data.append(row)
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    df.to_csv(csv_file, index=False)
                    saved_files['csv'] = str(csv_file)
                    print(f"✅ CSV results saved to: {csv_file}")
            
            # Save enhanced summary report
            summary_file = self.output_dir / 'enhanced_summary.txt'
            self._create_enhanced_summary_report(results, summary_file)
            saved_files['summary'] = str(summary_file)
            print(f"✅ Enhanced summary saved to: {summary_file}")
            
            # Save performance monitoring report if available
            performance_summary = results.get('performance_summary', {})
            if performance_summary:
                performance_file = self.output_dir / 'performance_report.txt'
                with open(performance_file, 'w', encoding='utf-8') as f:
                    f.write(format_performance_report(performance_summary))
                saved_files['performance'] = str(performance_file)
                print(f"📈 Performance report saved to: {performance_file}")
                
                # Save performance data as JSON for programmatic access
                performance_json_file = self.output_dir / 'performance_data.json'
                with open(performance_json_file, 'w', encoding='utf-8') as f:
                    json.dump(performance_summary, f, indent=2, default=str)
                saved_files['performance_json'] = str(performance_json_file)
                print(f"📊 Performance data saved to: {performance_json_file}")
            
            # List plot files
            if plot_files:
                print(f"📊 Generated {len(plot_files)} visualization files:")
                for plot_type, file_path in plot_files.items():
                    print(f"   • {plot_type}: {file_path}")
                    saved_files[f"plot_{plot_type}"] = str(file_path)
            
            return saved_files
            
        except Exception as e:
            error_msg = f"Failed to save enhanced results: {str(e)}"
            logger.error(error_msg)
            raise OrthoReduceEnhancedError(error_msg) from e
    
    def _create_enhanced_summary_report(self, results: Dict[str, Any], filepath: Path) -> None:
        """Create an enhanced human-readable summary report."""
        experiment_results = results.get('experiment_results', {})
        plot_files = results.get('plot_files', {})
        metadata = results.get('_metadata', {})
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("OrthoReduce Enhanced Experiment Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Write enhanced metadata
            f.write("Experiment Configuration:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Dataset Size: {metadata.get('dataset_size', 'Unknown')}\n")
            f.write(f"Dimensions: {metadata.get('dimensions', 'Unknown')}\n")
            f.write(f"Epsilon: {metadata.get('epsilon', 'Unknown')}\n")
            f.write(f"Methods: {', '.join(metadata.get('methods', []))}\n")
            f.write(f"Enhanced Features: {metadata.get('enhanced_features', 'None')}\n\n")
            
            # Write method results
            f.write("Method Performance:\n")
            f.write("-" * 20 + "\n")
            for method, metrics in experiment_results.items():
                if method != '_metadata' and isinstance(metrics, dict):
                    f.write(f"\n{method.upper()}:\n")
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            f.write(f"  {key}: {value:.6f}\n")
                        elif isinstance(value, str):
                            f.write(f"  {key}: {value}\n")
                        elif isinstance(value, dict):
                            f.write(f"  {key}:\n")
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, (int, float)):
                                    f.write(f"    {subkey}: {subvalue:.6f}\n")
            
            # Write visualization summary
            if plot_files:
                f.write(f"\n\nGenerated Visualizations ({len(plot_files)} files):\n")
                f.write("-" * 30 + "\n")
                for plot_type, file_path in plot_files.items():
                    f.write(f"  • {plot_type}: {file_path}\n")
    
    def launch_interactive_dashboard(self) -> None:
        """Launch the interactive dashboard for exploring results."""
        if not self.launch_dashboard:
            return
            
        try:
            print("🌐 Launching interactive dashboard...")
            import subprocess
            import time
            
            # Launch dashboard in background
            dashboard_cmd = [
                "python3", "launch_dashboard.py",
                "--results-dir", str(self.output_dir),
                "--auto-refresh"
            ]
            
            process = subprocess.Popen(dashboard_cmd)
            time.sleep(2)  # Give it time to start
            
            print("✅ Dashboard launched! Access at: http://localhost:8501")
            print("   (Dashboard will open automatically in your browser)")
            
        except Exception as e:
            print(f"⚠️  Failed to launch dashboard: {e}")
            print("   You can manually launch with: python3 launch_dashboard.py")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the enhanced runner."""
    parser = argparse.ArgumentParser(
        description="OrthoReduce Enhanced - Professional dimensionality reduction with advanced features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick-test
  %(prog)s --dataset-size 1000 --methods jll,pca,pocs --advanced-plots
  %(prog)s --full-benchmark --interactive --dashboard
  %(prog)s --staged-pipeline --comprehensive-eval --calibration
        """
    )
    
    # Core experiment parameters
    parser.add_argument(
        '--dataset-size', type=int, default=DEFAULT_CONFIG['dataset_size'],
        help=f'Number of data points (default: {DEFAULT_CONFIG["dataset_size"]})'
    )
    parser.add_argument(
        '--dimensions', type=int, default=DEFAULT_CONFIG['dimensions'],
        help=f'Original dimensions (default: {DEFAULT_CONFIG["dimensions"]})'
    )
    parser.add_argument(
        '--epsilon', type=float, default=DEFAULT_CONFIG['epsilon'], 
        help=f'JL distortion parameter (default: {DEFAULT_CONFIG["epsilon"]})'
    )
    parser.add_argument(
        '--methods', type=str, default=','.join(DEFAULT_CONFIG['methods']),
        help=f'Comma-separated methods (default: {",".join(DEFAULT_CONFIG["methods"])})'
    )
    parser.add_argument(
        '--output-dir', type=str, default=DEFAULT_CONFIG['output_dir'],
        help=f'Output directory (default: {DEFAULT_CONFIG["output_dir"]})'
    )
    
    # Enhanced features
    parser.add_argument(
        '--advanced-plots', action='store_true', default=DEFAULT_CONFIG['use_advanced_plots'],
        help='Use enhanced visualization system (default)'
    )
    parser.add_argument(
        '--no-advanced-plots', dest='advanced_plots', action='store_false',
        help='Disable advanced plotting'
    )
    parser.add_argument(
        '--interactive', action='store_true', default=DEFAULT_CONFIG['use_interactive'],
        help='Create interactive HTML plots'
    )
    parser.add_argument(
        '--comprehensive-eval', action='store_true', default=DEFAULT_CONFIG['use_comprehensive_eval'],
        help='Include trustworthiness/continuity metrics (default)'
    )
    parser.add_argument(
        '--no-comprehensive-eval', dest='comprehensive_eval', action='store_false',
        help='Disable comprehensive evaluation'
    )
    parser.add_argument(
        '--calibration', action='store_true', default=DEFAULT_CONFIG['use_calibration'],
        help='Apply post-processing calibration (default)'
    )
    parser.add_argument(
        '--no-calibration', dest='calibration', action='store_false',
        help='Disable calibration'
    )
    parser.add_argument(
        '--staged-pipeline', action='store_true', default=DEFAULT_CONFIG['use_staged_pipeline'],
        help='Use orchestrated multi-stage pipeline'
    )
    parser.add_argument(
        '--dashboard', action='store_true', default=DEFAULT_CONFIG['launch_dashboard'],
        help='Launch interactive dashboard after experiment'
    )
    
    # Monitoring options
    parser.add_argument(
        '--enable-monitoring', action='store_true', default=DEFAULT_CONFIG['enable_monitoring'],
        help='Enable comprehensive progress monitoring (default)'
    )
    parser.add_argument(
        '--no-monitoring', dest='enable_monitoring', action='store_false',
        help='Disable progress monitoring'
    )
    parser.add_argument(
        '--show-system-stats', action='store_true', default=DEFAULT_CONFIG['show_system_stats'],
        help='Show real-time system resource usage (default)'
    )
    parser.add_argument(
        '--no-system-stats', dest='show_system_stats', action='store_false',
        help='Hide system resource monitoring'
    )
    
    # Convenience presets
    parser.add_argument(
        '--quick-test', action='store_true',
        help='Quick test: n=200, d=30, basic methods'
    )
    parser.add_argument(
        '--full-benchmark', action='store_true', 
        help='Full benchmark: n=1000, d=100, all methods, all features'
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for the enhanced experiment runner."""
    try:
        args = parse_arguments()
        
        # Handle presets
        if args.quick_test:
            args.dataset_size = 200
            args.dimensions = 30
            args.methods = "jll,pca"
            args.output_dir = "quick_test_enhanced"
            print("🚀 Quick test mode enabled")
        
        if args.full_benchmark:
            args.dataset_size = 1000
            args.dimensions = 100
            args.methods = "jll,pca,gaussian,pocs,poincare,spherical"
            args.advanced_plots = True
            args.interactive = True
            args.comprehensive_eval = True
            args.calibration = True
            args.dashboard = True
            args.output_dir = "full_benchmark_enhanced"
            print("🏆 Full benchmark mode enabled")
        
        # Parse methods
        methods = [m.strip() for m in args.methods.split(',')]
        
        # Create and run enhanced experiment
        runner = EnhancedExperimentRunner(
            dataset_size=args.dataset_size,
            dimensions=args.dimensions,
            epsilon=args.epsilon,
            methods=methods,
            output_dir=args.output_dir,
            use_advanced_plots=args.advanced_plots,
            use_interactive=args.interactive,
            use_comprehensive_eval=args.comprehensive_eval,
            use_calibration=args.calibration,
            use_staged_pipeline=args.staged_pipeline,
            launch_dashboard=args.dashboard,
            enable_monitoring=args.enable_monitoring,
            show_system_stats=args.show_system_stats
        )
        
        # Run experiment
        results = runner.run_experiment()
        
        # Save results
        saved_files = runner.save_results(results)
        
        # Launch dashboard if requested
        runner.launch_interactive_dashboard()
        
        print("\n🎉 Enhanced experiment completed successfully!")
        print(f"📁 Results saved in: {args.output_dir}/")
        
        # Show enhanced summary
        plot_files = results.get('plot_files', {})
        if plot_files:
            print(f"📊 Generated {len(plot_files)} advanced visualizations")
        
        if args.dashboard:
            print("🌐 Interactive dashboard available at: http://localhost:8501")
            
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Enhanced experiment failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()