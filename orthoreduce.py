#!/usr/bin/env python3
"""
OrthoReduce - Professional Dimensionality Reduction Experiment Runner

This is the main entry point for the OrthoReduce library, providing a comprehensive
command-line interface for running dimensionality reduction experiments with various
methods and configurations.

The script supports multiple dimensionality reduction methods including Johnson-
Lindenstrauss Lemma based projection, PCA, Gaussian random projection, POCS,
Poincaré disk model projection, and spherical projection.

Author: OrthoReduce Team
Version: 1.0.0
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

from orthogonal_projection.dimensionality_reduction import run_experiment
from orthogonal_projection.visualization import OrthoReduceVisualizer

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
    'pocs': 'Projection Onto Convex Sets (enhanced)',
    'poincare': 'Poincaré disk model projection',
    'spherical': 'Spherical projection'
}

# Default configuration
DEFAULT_CONFIG = {
    'dataset_size': 500,
    'dimensions': 50,
    'epsilon': 0.2,
    'methods': list(AVAILABLE_METHODS.keys()),
    'output_dir': 'experiment_results',
    'use_adaptive': True,
    'use_optimized_eval': True
}


class OrthoReduceError(Exception):
    """Custom exception for OrthoReduce-specific errors."""
    pass


class ExperimentRunner:
    """
    Main class for running OrthoReduce experiments.
    
    This class handles experiment configuration, execution, and result generation
    with comprehensive error handling and user feedback.
    """
    
    def __init__(
        self,
        dataset_size: int,
        dimensions: int,
        epsilon: float,
        methods: List[str],
        output_dir: str,
        use_adaptive: bool = True,
        use_optimized_eval: bool = True
    ) -> None:
        """
        Initialize the experiment runner.
        
        Args:
            dataset_size: Number of data points to generate
            dimensions: Original dimensionality of the data
            epsilon: Johnson-Lindenstrauss distortion parameter
            methods: List of dimensionality reduction methods to test
            output_dir: Directory to save results
            use_adaptive: Whether to use adaptive algorithms
            use_optimized_eval: Whether to use optimized evaluation functions
            
        Raises:
            OrthoReduceError: If parameters are invalid
        """
        self._validate_parameters(dataset_size, dimensions, epsilon, methods)
        
        self.dataset_size = dataset_size
        self.dimensions = dimensions
        self.epsilon = epsilon
        self.methods = methods
        self.output_dir = Path(output_dir)
        self.use_adaptive = use_adaptive
        self.use_optimized_eval = use_optimized_eval
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _validate_parameters(
        self, 
        dataset_size: int, 
        dimensions: int, 
        epsilon: float, 
        methods: List[str]
    ) -> None:
        """Validate experiment parameters."""
        if dataset_size < 10:
            raise OrthoReduceError(f"Dataset size must be >= 10, got {dataset_size}")
        
        if dimensions < 2:
            raise OrthoReduceError(f"Dimensions must be >= 2, got {dimensions}")
        
        if not 0.01 <= epsilon <= 1.0:
            raise OrthoReduceError(f"Epsilon must be between 0.01 and 1.0, got {epsilon}")
        
        if not methods:
            raise OrthoReduceError("At least one method must be specified")
        
        invalid_methods = [m for m in methods if m not in AVAILABLE_METHODS]
        if invalid_methods:
            raise OrthoReduceError(
                f"Invalid methods: {invalid_methods}. "
                f"Available methods: {list(AVAILABLE_METHODS.keys())}"
            )
    
    def run_experiment(self) -> Dict[str, Any]:
        """
        Execute the dimensionality reduction experiment.
        
        Returns:
            Dictionary containing experimental results and metadata
            
        Raises:
            OrthoReduceError: If experiment execution fails
        """
        print("🔬 Starting OrthoReduce Experiment...")
        print(f"📊 Configuration: n={self.dataset_size}, d={self.dimensions}, ε={self.epsilon}")
        print(f"🔧 Methods: {', '.join(self.methods)}")
        print(f"📁 Output directory: {self.output_dir}")
        print()
        
        try:
            results = run_experiment(
                n=self.dataset_size,
                d=self.dimensions,
                epsilon=self.epsilon,
                methods=self.methods,
                use_adaptive=self.use_adaptive,
                use_optimized_eval=self.use_optimized_eval
            )
            
            logger.info("Experiment completed successfully")
            return results
            
        except Exception as e:
            error_msg = f"Experiment failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise OrthoReduceError(error_msg) from e
    
    def save_results(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Save experiment results to multiple formats.
        
        Args:
            results: Experiment results dictionary
            
        Returns:
            Dictionary mapping file types to saved file paths
            
        Raises:
            OrthoReduceError: If saving fails
        """
        saved_files = {}
        
        try:
            # Save JSON results
            json_file = self.output_dir / 'results.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            saved_files['json'] = str(json_file)
            print(f"✅ JSON results saved to: {json_file}")
            
            # Save CSV results
            csv_file = self.output_dir / 'results.csv'
            df_data = []
            for method, metrics in results.items():
                if method != '_metadata':
                    row = {'method': method, **metrics}
                    df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(csv_file, index=False)
            saved_files['csv'] = str(csv_file)
            print(f"✅ CSV results saved to: {csv_file}")
            
            # Save metadata
            metadata_file = self.output_dir / 'metadata.json'
            metadata = results.get('_metadata', {})
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            saved_files['metadata'] = str(metadata_file)
            print(f"✅ Metadata saved to: {metadata_file}")
            
            # Save summary report
            summary_file = self.output_dir / 'summary.txt'
            self._create_summary_report(results, summary_file)
            saved_files['summary'] = str(summary_file)
            print(f"✅ Summary report saved to: {summary_file}")
            
            return saved_files
            
        except Exception as e:
            error_msg = f"Failed to save results: {str(e)}"
            logger.error(error_msg)
            raise OrthoReduceError(error_msg) from e
    
    def _create_summary_report(self, results: Dict[str, Any], filepath: Path) -> None:
        """Create a human-readable summary report."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("OrthoReduce Experiment Summary\n")
            f.write("=" * 40 + "\n\n")
            
            # Write metadata
            if '_metadata' in results:
                meta = results['_metadata']
                f.write(f"Dataset: {meta['n']} points, {meta['d']} dimensions\n")
                f.write(f"Target dimension: {meta['k']}\n")
                f.write(f"Compression ratio: {meta['d']/meta['k']:.2f}x\n")
                f.write(f"Epsilon: {meta['epsilon']}\n")
                f.write(f"Intrinsic dimension: {meta['intrinsic_dimension']}\n\n")
            
            # Write results
            f.write("Method Results:\n")
            f.write("-" * 20 + "\n")
            for method, metrics in results.items():
                if method != '_metadata':
                    f.write(f"\n{method}:\n")
                    f.write(f"  Mean distortion: {metrics['mean_distortion']:.4f}\n")
                    f.write(f"  Max distortion: {metrics['max_distortion']:.4f}\n")
                    f.write(f"  Rank correlation: {metrics['rank_correlation']:.4f}\n")
                    f.write(f"  Runtime: {metrics['runtime']:.4f}s\n")
                    f.write(f"  Compression ratio: {metrics['compression_ratio']:.2f}x\n")
    
    def create_visualizations(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create professional visualizations of the results.
        
        Args:
            results: Experiment results dictionary
            
        Returns:
            Dictionary of visualization file paths or None if failed
        """
        print("\n📊 Generating Professional Visualizations...")
        
        try:
            visualizer = OrthoReduceVisualizer(str(self.output_dir))
            viz_files = visualizer.create_complete_visualization(results)
            
            print("✅ Visualizations created successfully!")
            print(f"📄 Comprehensive PDF report: {viz_files['comprehensive_report']}")
            print(f"📊 Main dashboard: {viz_files['main_dashboard']}")
            print(f"🖼️  Individual plots: {len(viz_files['individual_plots'])} files")
            
            return viz_files
            
        except Exception as e:
            print(f"⚠️  Visualization generation failed: {e}")
            logger.warning(f"Visualization failed: {str(e)}")
            print("Results data files were still saved successfully.")
            return None
    
    def print_results_summary(self, results: Dict[str, Any]) -> None:
        """Print a quick summary of the results."""
        print("\n📊 Quick Results Preview:")
        print("=" * 50)
        
        for method, metrics in results.items():
            if method != '_metadata':
                print(f"{method:>10}: distortion={metrics['mean_distortion']:.4f}, "
                      f"runtime={metrics['runtime']:.4f}s, "
                      f"compression={metrics['compression_ratio']:.2f}x")
        
        print(f"\n📁 All results saved in: {self.output_dir}/")
        self._print_file_structure()
    
    def _print_file_structure(self) -> None:
        """Print the output file structure."""
        print("Files created:")
        print("  • results.json       - Full results in JSON format")
        print("  • results.csv        - Results in CSV format for Excel")
        print("  • metadata.json      - Experiment configuration")
        print("  • summary.txt        - Human-readable summary")
        print("  • plots/             - Professional visualization directory")
        print("    ├── comprehensive_report.pdf - Multi-page publication report")
        print("    ├── main_dashboard.png       - Performance overview")
        print("    └── individual/              - Separate high-res plots")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="""
        OrthoReduce - Professional Dimensionality Reduction Experiment Runner
        
        A comprehensive command-line interface for running OrthoReduce dimensionality
        reduction experiments with various methods and configurations.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with default parameters
  %(prog)s --quick-test

  # Custom experiment with specific methods  
  %(prog)s --dataset-size 1000 --dimensions 100 --methods jll,pca,pocs

  # Full benchmark with custom output directory
  %(prog)s --full-benchmark --output-dir my_results

  # Advanced experiment with custom epsilon
  %(prog)s -n 800 -d 75 -e 0.15 -o advanced_experiment

Available Methods:
""" + '\n'.join(f"  {name:10} - {desc}" for name, desc in AVAILABLE_METHODS.items()) + """

For more information, visit: https://github.com/AlitheaBio/OrthoReduce
        """
    )
    
    # Core parameters
    parser.add_argument(
        '-n', '--dataset-size',
        type=int,
        default=DEFAULT_CONFIG['dataset_size'],
        help=f'Number of data points (default: {DEFAULT_CONFIG["dataset_size"]})'
    )
    
    parser.add_argument(
        '-d', '--dimensions',
        type=int,
        default=DEFAULT_CONFIG['dimensions'],
        help=f'Original dimensions (default: {DEFAULT_CONFIG["dimensions"]})'
    )
    
    parser.add_argument(
        '-e', '--epsilon',
        type=float,
        default=DEFAULT_CONFIG['epsilon'],
        help=f'JL distortion parameter (default: {DEFAULT_CONFIG["epsilon"]})'
    )
    
    parser.add_argument(
        '-m', '--methods',
        type=str,
        default=','.join(DEFAULT_CONFIG['methods']),
        help=f'Comma-separated list of methods. Available: {", ".join(AVAILABLE_METHODS.keys())} (default: all methods)'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=DEFAULT_CONFIG['output_dir'],
        help=f'Output directory (default: {DEFAULT_CONFIG["output_dir"]})'
    )
    
    # Preset options
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run small fast test (n=200, d=30)'
    )
    
    parser.add_argument(
        '--full-benchmark',
        action='store_true',
        help='Run comprehensive benchmark with all methods'
    )
    
    # Advanced options
    parser.add_argument(
        '--no-adaptive',
        action='store_true',
        help='Disable adaptive algorithms'
    )
    
    parser.add_argument(
        '--no-optimized-eval',
        action='store_true',
        help='Disable optimized evaluation functions'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def validate_environment() -> None:
    """Validate the runtime environment."""
    try:
        import numpy
        import pandas
        import matplotlib
        import scipy
        import sklearn
    except ImportError as e:
        raise OrthoReduceError(
            f"Missing required dependency: {e}. "
            "Please install with: pip install -r requirements.txt"
        ) from e


def main() -> int:
    """
    Main entry point for the OrthoReduce experiment runner.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Configure logging
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Validate environment
        validate_environment()
        
        # Handle presets
        if args.quick_test:
            args.dataset_size = 200
            args.dimensions = 30
            args.epsilon = 0.2
            args.output_dir = 'quick_test_results'
            print("🚀 Quick test mode: n=200, d=30")
        
        if args.full_benchmark:
            args.dataset_size = 1000
            args.dimensions = 100
            args.epsilon = 0.15
            args.methods = ','.join(AVAILABLE_METHODS.keys())
            args.output_dir = 'full_benchmark_results'
            print("🏆 Full benchmark mode: n=1000, d=100, all methods")
        
        # Parse methods list
        methods = [method.strip() for method in args.methods.split(',')]
        
        # Create experiment runner
        runner = ExperimentRunner(
            dataset_size=args.dataset_size,
            dimensions=args.dimensions,
            epsilon=args.epsilon,
            methods=methods,
            output_dir=args.output_dir,
            use_adaptive=not args.no_adaptive,
            use_optimized_eval=not args.no_optimized_eval
        )
        
        # Run experiment
        results = runner.run_experiment()
        
        # Save results
        saved_files = runner.save_results(results)
        
        # Create visualizations
        viz_files = runner.create_visualizations(results)
        
        # Print summary
        runner.print_results_summary(results)
        
        print("\n🎉 Experiment completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        logger.info("Experiment interrupted by user")
        return 1
        
    except OrthoReduceError as e:
        print(f"\n❌ Error: {e}")
        logger.error(str(e))
        return 1
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())