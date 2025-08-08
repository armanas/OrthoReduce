"""
Command Line Interface for Staged Optimization Experiments

This module provides a comprehensive CLI for running staged optimization
experiments with modular stage execution, configuration management,
and interactive experiment design.
"""

import argparse
import sys
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
from datetime import datetime

# Import experiment components
try:
    from .experiment_config import (
        ExperimentConfig, create_default_config, create_fast_config,
        create_comprehensive_config, create_ms_data_config
    )
    from .pipeline_orchestrator import PipelineOrchestrator
    from .results_aggregator import ResultsAggregator, ExperimentResult
    from .hyperparameter_search import HyperparameterSearchEngine, ParameterGridGenerator
    from .dimensionality_reduction import generate_mixture_gaussians
    from .ms_data import parse_mzml_build_matrix
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    sys.exit(1)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Set up logging configuration."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Configure logging format
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Set up handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format=format_str,
        handlers=handlers
    )


def load_or_generate_data(data_config: Dict[str, Any]) -> tuple:
    """Load or generate data based on configuration."""
    data_source = data_config.get("data_source", "synthetic")
    data_params = data_config.get("data_params", {})
    
    if data_source == "synthetic":
        # Generate synthetic data
        n_samples = data_params.get("n_samples", 1000)
        n_features = data_params.get("n_features", 500)
        n_components = data_params.get("n_components", 50)
        noise_level = data_params.get("noise_level", 0.1)
        
        print(f"Generating synthetic data: {n_samples} samples, {n_features} features")
        X = generate_mixture_gaussians(n_samples, n_features, n_components, noise_level)
        y = None
        
    elif data_source == "ms_data":
        # Load mass spectrometry data
        mzml_path = data_params.get("mzml_path")
        if not mzml_path or not Path(mzml_path).exists():
            raise FileNotFoundError(f"mzML file not found: {mzml_path}")
        
        print(f"Loading MS data from {mzml_path}")
        try:
            X = parse_mzml_build_matrix(
                mzml_path,
                max_spectra=data_params.get("max_spectra", 1000),
                min_intensity=data_params.get("min_intensity", 1e4),
                mz_range=data_params.get("mz_range", (100, 2000))
            )
            y = None
        except Exception as e:
            print(f"Failed to load MS data: {e}")
            raise
            
    elif data_source == "custom":
        # Load custom data
        data_path = data_params.get("data_path")
        if not data_path or not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        print(f"Loading custom data from {data_path}")
        # Simple numpy/CSV loading - could be extended
        if str(data_path).endswith('.npy'):
            X = np.load(data_path)
        elif str(data_path).endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(data_path)
            X = df.values
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
        
        y = None
        
    else:
        raise ValueError(f"Unknown data source: {data_source}")
    
    print(f"Loaded data shape: {X.shape}")
    return X, y


def run_single_experiment(config: ExperimentConfig, X: np.ndarray, y: Optional[np.ndarray] = None, 
                         run_id: int = 0) -> ExperimentResult:
    """Run a single experiment with the given configuration."""
    print(f"\nRunning experiment: {config.name} (Run {run_id + 1})")
    
    # Set random seed for reproducibility
    np.random.seed(config.random_seed + run_id)
    
    # Create orchestrator and run pipeline
    orchestrator = PipelineOrchestrator(config)
    start_time = time.time()
    
    try:
        pipeline_results = orchestrator.run_pipeline(X, y)
        runtime = time.time() - start_time
        
        # Extract evaluation metrics
        evaluation_metrics = pipeline_results.get("evaluation_results", {})
        best_embedding = pipeline_results.get("best_embedding", "")
        
        # Get resource usage
        resource_usage = pipeline_results.get("resource_usage", {})
        memory_usage = resource_usage.get("peak_memory_gb", 0.0)
        
        # Create experiment result
        experiment_result = ExperimentResult(
            config=config,
            pipeline_results=pipeline_results,
            evaluation_metrics=evaluation_metrics,
            best_embedding=best_embedding,
            runtime=runtime,
            memory_usage=memory_usage,
            timestamp=datetime.now().isoformat(),
            run_id=run_id
        )
        
        print(f"Experiment completed in {runtime:.2f}s")
        print(f"Best embedding: {best_embedding}")
        
        return experiment_result
        
    except Exception as e:
        print(f"Experiment failed: {str(e)}")
        # Create a failed experiment result
        return ExperimentResult(
            config=config,
            pipeline_results={"error": str(e)},
            evaluation_metrics={},
            best_embedding="",
            runtime=time.time() - start_time,
            memory_usage=0.0,
            timestamp=datetime.now().isoformat(),
            run_id=run_id,
            metadata={"failed": True, "error": str(e)}
        )


def run_experiments_command(args):
    """Run experiments command handler."""
    print("Starting staged optimization experiments...")
    
    # Load configuration
    if args.config:
        config = ExperimentConfig.load(args.config)
        print(f"Loaded configuration from {args.config}")
    else:
        # Create default configuration
        if args.template == "fast":
            config = create_fast_config(args.name, args.data_source)
        elif args.template == "comprehensive":
            config = create_comprehensive_config(args.name, args.data_source)
        elif args.template == "ms":
            config = create_ms_data_config(args.name, args.data_path)
        else:
            config = create_default_config(args.name, args.data_source)
        
        print(f"Created {args.template} configuration template")
    
    # Override configuration with command line arguments
    if args.n_runs:
        config.n_runs = args.n_runs
    if args.random_seed:
        config.random_seed = args.random_seed
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.n_jobs:
        config.n_jobs = args.n_jobs
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print("Configuration validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        if not args.ignore_validation:
            print("Aborting due to validation issues. Use --ignore-validation to proceed anyway.")
            return 1
    
    # Load or generate data
    data_config = {
        "data_source": args.data_source,
        "data_params": getattr(config, 'data_params', {})
    }
    
    if args.data_path:
        data_config["data_params"]["data_path"] = args.data_path
        data_config["data_params"]["mzml_path"] = args.data_path
    
    try:
        X, y = load_or_generate_data(data_config)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return 1
    
    # Run experiments
    experiment_results = []
    
    for run_id in range(config.n_runs):
        try:
            result = run_single_experiment(config, X, y, run_id)
            experiment_results.append(result)
            
            # Save intermediate results if requested
            if args.save_intermediate:
                output_dir = Path(config.output_dir) / config.name
                output_dir.mkdir(parents=True, exist_ok=True)
                
                import pickle
                with open(output_dir / f"result_run_{run_id}.pkl", "wb") as f:
                    pickle.dump(result, f)
                
        except KeyboardInterrupt:
            print("\nExperiment interrupted by user")
            break
        except Exception as e:
            print(f"Experiment run {run_id} failed: {e}")
            continue
    
    if not experiment_results:
        print("No experiments completed successfully")
        return 1
    
    # Aggregate results
    print(f"\nAggregating results from {len(experiment_results)} experiments...")
    
    aggregator = ResultsAggregator(output_dir=config.output_dir)
    aggregated_results = aggregator.aggregate_experiment_results(experiment_results)
    
    # Save results
    results_path = aggregator.save_aggregated_results(aggregated_results, config.name)
    print(f"Results saved to {results_path}")
    
    # Generate report
    if args.generate_report:
        report = aggregator.generate_comprehensive_report(aggregated_results)
        report_path = Path(config.output_dir) / f"{config.name}_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Report saved to {report_path}")
        
        if args.print_report:
            print("\n" + report)
    
    print(f"\nBest overall method: {aggregated_results.best_overall_method}")
    
    return 0


def create_config_command(args):
    """Create configuration command handler."""
    print(f"Creating {args.template} configuration template...")
    
    if args.template == "fast":
        config = create_fast_config(args.name, args.data_source)
    elif args.template == "comprehensive":
        config = create_comprehensive_config(args.name, args.data_source)
    elif args.template == "ms":
        config = create_ms_data_config(args.name, args.data_path or "data/sample.mzML")
    else:
        config = create_default_config(args.name, args.data_source)
    
    # Override with command line arguments
    if args.description:
        config.description = args.description
    if args.random_seed:
        config.random_seed = args.random_seed
    if args.n_runs:
        config.n_runs = args.n_runs
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Save configuration
    output_path = Path(args.output) if args.output else Path(f"{args.name}_config.yaml")
    config.save(output_path)
    
    print(f"Configuration saved to {output_path}")
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print("\nConfiguration validation issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration is valid!")
    
    return 0


def validate_config_command(args):
    """Validate configuration command handler."""
    print(f"Validating configuration: {args.config}")
    
    try:
        config = ExperimentConfig.load(args.config)
        issues = config.validate()
        
        if issues:
            print(f"Found {len(issues)} validation issues:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
            return 1
        else:
            print("Configuration is valid!")
            
            # Print summary
            print(f"\nConfiguration Summary:")
            print(f"  Name: {config.name}")
            print(f"  Description: {config.description}")
            print(f"  Random seed: {config.random_seed}")
            print(f"  Number of runs: {config.n_runs}")
            print(f"  Data source: {config.data_source}")
            
            # List enabled stages
            enabled_stages = [stage.name for stage in config.get_stages() if stage.enabled]
            print(f"  Enabled stages: {', '.join(enabled_stages)}")
            
            return 0
            
    except Exception as e:
        print(f"Failed to load or validate configuration: {e}")
        return 1


def list_stages_command(args):
    """List available pipeline stages command handler."""
    print("Available Pipeline Stages:")
    print("=" * 50)
    
    stages_info = [
        {
            "name": "preprocessing",
            "description": "Data preprocessing with PCA, standardization, and denoising",
            "methods": ["adaptive_pipeline", "manual_steps"],
            "key_params": ["pca_components", "standardization", "denoising_method"]
        },
        {
            "name": "convex_optimization", 
            "description": "Convex hull projection with hyperparameter grid search",
            "methods": ["project_onto_convex_hull_enhanced"],
            "key_params": ["k_candidates_grid", "lambda_grid", "objective_types"]
        },
        {
            "name": "geometric_embeddings",
            "description": "Spherical and Poincaré embeddings with Riemannian optimization",
            "methods": ["adaptive_spherical_embedding", "run_poincare_optimized"],
            "key_params": ["spherical_config", "poincare_config", "riemannian_optimization"]
        },
        {
            "name": "calibration",
            "description": "Post-processing calibration with isotonic regression and Procrustes alignment",
            "methods": ["combined_calibration"],
            "key_params": ["methods", "isotonic_params", "procrustes_params", "local_correction_params"]
        }
    ]
    
    for stage in stages_info:
        print(f"\n{stage['name'].upper()}")
        print("-" * len(stage['name']))
        print(f"Description: {stage['description']}")
        print(f"Methods: {', '.join(stage['methods'])}")
        print(f"Key Parameters: {', '.join(stage['key_params'])}")
    
    if args.detailed:
        print("\n" + "="*50)
        print("STAGE DEPENDENCIES:")
        print("="*50)
        print("preprocessing → convex_optimization → geometric_embeddings → calibration")
        print("\nNOTE: Each stage depends on the successful completion of all previous stages.")
    
    return 0


def analyze_results_command(args):
    """Analyze existing results command handler."""
    print(f"Analyzing results from: {args.results_path}")
    
    try:
        import pickle
        with open(args.results_path, "rb") as f:
            aggregated_results = pickle.load(f)
        
        # Generate analysis report
        aggregator = ResultsAggregator()
        report = aggregator.generate_comprehensive_report(aggregated_results)
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"Analysis report saved to {args.output}")
        else:
            print("\n" + report)
        
        return 0
        
    except Exception as e:
        print(f"Failed to analyze results: {e}")
        return 1


def interactive_config_command(args):
    """Interactive configuration creation command handler."""
    print("Interactive Configuration Creator")
    print("=" * 40)
    
    # Basic experiment info
    name = input("Experiment name: ").strip() or "interactive_experiment"
    description = input("Description (optional): ").strip()
    
    # Data source selection
    print("\nData Source Options:")
    print("1. Synthetic data (mixture of Gaussians)")
    print("2. Mass spectrometry data (mzML)")
    print("3. Custom data (CSV/NPY)")
    
    data_choice = input("Select data source (1-3): ").strip()
    
    if data_choice == "1":
        data_source = "synthetic"
        n_samples = int(input("Number of samples (default 1000): ") or "1000")
        n_features = int(input("Number of features (default 500): ") or "500")
        n_components = int(input("Number of components (default 50): ") or "50")
    elif data_choice == "2":
        data_source = "ms_data"
        mzml_path = input("Path to mzML file: ").strip()
        if not mzml_path:
            print("mzML path is required for MS data")
            return 1
    elif data_choice == "3":
        data_source = "custom"
        data_path = input("Path to data file: ").strip()
        if not data_path:
            print("Data path is required for custom data")
            return 1
    else:
        print("Invalid choice, using synthetic data")
        data_source = "synthetic"
        n_samples, n_features, n_components = 1000, 500, 50
    
    # Experiment parameters
    n_runs = int(input("Number of runs (default 3): ") or "3")
    random_seed = int(input("Random seed (default 42): ") or "42")
    
    # Stage selection
    print("\nPipeline Stages (y/n):")
    enable_preprocessing = input("Enable preprocessing? (Y/n): ").strip().lower() != 'n'
    enable_convex = input("Enable convex optimization? (Y/n): ").strip().lower() != 'n'
    enable_geometric = input("Enable geometric embeddings? (Y/n): ").strip().lower() != 'n'
    enable_calibration = input("Enable calibration? (Y/n): ").strip().lower() != 'n'
    
    # Create configuration
    if data_source == "ms_data":
        config = create_ms_data_config(name, mzml_path)
    else:
        config = create_default_config(name, data_source)
    
    # Update configuration
    config.description = description
    config.n_runs = n_runs
    config.random_seed = random_seed
    
    if data_source == "synthetic":
        config.data_params.update({
            "n_samples": n_samples,
            "n_features": n_features,
            "n_components": n_components
        })
    elif data_source == "custom":
        config.data_params["data_path"] = data_path
    
    # Update stage enablement
    config.preprocessing.enabled = enable_preprocessing
    config.convex_optimization.enabled = enable_convex
    config.geometric_embeddings.enabled = enable_geometric
    config.calibration.enabled = enable_calibration
    
    # Save configuration
    output_path = input(f"Output file (default {name}_config.yaml): ").strip() or f"{name}_config.yaml"
    config.save(output_path)
    
    print(f"\nConfiguration created and saved to {output_path}")
    
    # Ask if user wants to run immediately
    run_now = input("Run experiment now? (y/N): ").strip().lower() == 'y'
    
    if run_now:
        # Set up args object for run_experiments_command
        class Args:
            def __init__(self):
                self.config = output_path
                self.name = name
                self.template = "default"
                self.data_source = data_source
                self.data_path = locals().get('data_path') or locals().get('mzml_path')
                self.n_runs = None  # Use config value
                self.random_seed = None  # Use config value
                self.output_dir = None  # Use config value
                self.n_jobs = None  # Use config value
                self.ignore_validation = False
                self.save_intermediate = False
                self.generate_report = True
                self.print_report = True
        
        args_obj = Args()
        return run_experiments_command(args_obj)
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OrthoReduce Staged Optimization Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiments with default configuration
  python -m orthogonal_projection.cli run --name my_experiment
  
  # Create and run fast experiment template
  python -m orthogonal_projection.cli run --template fast --name fast_exp
  
  # Run with custom configuration
  python -m orthogonal_projection.cli run --config my_config.yaml
  
  # Create configuration template
  python -m orthogonal_projection.cli create-config --name my_exp --template comprehensive
  
  # Interactive configuration creation
  python -m orthogonal_projection.cli interactive
  
  # Validate existing configuration
  python -m orthogonal_projection.cli validate --config my_config.yaml
  
  # List available pipeline stages
  python -m orthogonal_projection.cli list-stages --detailed
        """
    )
    
    # Global arguments
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-file", help="Log to file")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run experiments command
    run_parser = subparsers.add_parser("run", help="Run staged optimization experiments")
    run_parser.add_argument("--config", help="Configuration file path")
    run_parser.add_argument("--name", default="experiment", help="Experiment name")
    run_parser.add_argument("--template", default="default", 
                           choices=["default", "fast", "comprehensive", "ms"],
                           help="Configuration template")
    run_parser.add_argument("--data-source", default="synthetic", 
                           choices=["synthetic", "ms_data", "custom"],
                           help="Data source type")
    run_parser.add_argument("--data-path", help="Path to data file")
    run_parser.add_argument("--n-runs", type=int, help="Number of experimental runs")
    run_parser.add_argument("--random-seed", type=int, help="Random seed")
    run_parser.add_argument("--output-dir", default="experiments", help="Output directory")
    run_parser.add_argument("--n-jobs", type=int, help="Number of parallel jobs")
    run_parser.add_argument("--ignore-validation", action="store_true",
                           help="Ignore configuration validation issues")
    run_parser.add_argument("--save-intermediate", action="store_true",
                           help="Save intermediate results for each run")
    run_parser.add_argument("--generate-report", action="store_true", default=True,
                           help="Generate analysis report")
    run_parser.add_argument("--print-report", action="store_true",
                           help="Print report to console")
    
    # Create configuration command
    create_parser = subparsers.add_parser("create-config", help="Create configuration file")
    create_parser.add_argument("--name", required=True, help="Experiment name")
    create_parser.add_argument("--template", default="default",
                              choices=["default", "fast", "comprehensive", "ms"],
                              help="Configuration template")
    create_parser.add_argument("--data-source", default="synthetic",
                              choices=["synthetic", "ms_data", "custom"],
                              help="Data source type")
    create_parser.add_argument("--data-path", help="Path to data file")
    create_parser.add_argument("--description", help="Experiment description")
    create_parser.add_argument("--n-runs", type=int, help="Number of runs")
    create_parser.add_argument("--random-seed", type=int, help="Random seed")
    create_parser.add_argument("--output-dir", help="Output directory")
    create_parser.add_argument("--output", help="Output configuration file path")
    
    # Validate configuration command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration file")
    validate_parser.add_argument("--config", required=True, help="Configuration file path")
    
    # List stages command
    stages_parser = subparsers.add_parser("list-stages", help="List available pipeline stages")
    stages_parser.add_argument("--detailed", action="store_true",
                              help="Show detailed stage information")
    
    # Analyze results command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze existing results")
    analyze_parser.add_argument("--results-path", required=True,
                               help="Path to pickled aggregated results")
    analyze_parser.add_argument("--output", help="Output file for analysis report")
    
    # Interactive configuration command
    interactive_parser = subparsers.add_parser("interactive", 
                                              help="Interactive configuration creation")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    
    # Dispatch to command handlers
    try:
        if args.command == "run":
            return run_experiments_command(args)
        elif args.command == "create-config":
            return create_config_command(args)
        elif args.command == "validate":
            return validate_config_command(args)
        elif args.command == "list-stages":
            return list_stages_command(args)
        elif args.command == "analyze":
            return analyze_results_command(args)
        elif args.command == "interactive":
            return interactive_config_command(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())