#!/usr/bin/env python3
"""
Example script demonstrating the complete staged optimization system.

This script shows how to use the comprehensive experiment orchestration
system for dimensionality reduction with staged optimization.
"""

import sys
import numpy as np
from pathlib import Path

# Add the parent directory to path so we can import the modules
sys.path.append(str(Path(__file__).parent.parent))

from orthogonal_projection.experiment_orchestration import (
    StagedOptimizationExperiment, run_synthetic_data_experiment
)
from orthogonal_projection.experiment_config import (
    ExperimentConfig, create_default_config, create_fast_config, create_comprehensive_config
)
from orthogonal_projection.dimensionality_reduction import generate_mixture_gaussians


def example_1_simple_synthetic_experiment():
    """Example 1: Simple synthetic data experiment using convenience function."""
    print("=" * 80)
    print("EXAMPLE 1: Simple Synthetic Data Experiment")
    print("=" * 80)
    
    try:
        # Run a fast experiment on synthetic data
        results = run_synthetic_data_experiment(
            experiment_name="simple_synthetic",
            n_samples=300,  # Small for quick demo
            n_features=100,
            n_components=10,
            template="fast"
        )
        
        print(f"✓ Experiment completed successfully!")
        print(f"  Best method: {results['best_overall_method']}")
        print(f"  Number of successful runs: {len(results['individual_results'])}")
        print(f"  Results saved to: {results['results_path']}")
        print(f"  Report available at: {results['report_path']}")
        
        # Show some key metrics
        aggregated = results['aggregated_results']
        if aggregated.method_comparison and aggregated.method_comparison.method_rankings:
            print(f"\n  Method Rankings:")
            for i, (method, score) in enumerate(aggregated.method_comparison.method_rankings[:3]):
                print(f"    {i+1}. {method}: {score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Example 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def example_2_custom_configuration_experiment():
    """Example 2: Custom configuration with manual setup."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Custom Configuration Experiment")
    print("=" * 80)
    
    try:
        # Create custom configuration
        config = create_default_config("custom_experiment", "synthetic")
        
        # Customize the configuration
        config.description = "Custom experiment with specific parameters"
        config.n_runs = 2  # Reduced for demo
        config.random_seed = 123
        
        # Customize preprocessing
        config.preprocessing.standardization = "robust"
        config.preprocessing.denoising_method = "pca"
        
        # Customize convex optimization
        config.convex_optimization.k_candidates_grid = [32, 64]
        config.convex_optimization.lambda_grid = [1e-6, 1e-4]
        config.convex_optimization.objective_types = ["quadratic", "huber"]
        
        # Customize geometric embeddings - disable Poincare for speed
        config.geometric_embeddings.poincare_config["curvatures"] = [1.0]  # Single curvature
        config.geometric_embeddings.spherical_config["methods"] = ["fast"]  # Fast method only
        
        # Generate synthetic data
        print("Generating synthetic data...")
        X = generate_mixture_gaussians(
            n_samples=400,
            n_features=150, 
            n_components=15,
            noise_level=0.1,
            random_state=config.random_seed
        )
        print(f"Generated data shape: {X.shape}")
        
        # Create and run experiment
        experiment = StagedOptimizationExperiment(config)
        results = experiment.run_complete_experiment(X)
        
        print(f"✓ Custom experiment completed!")
        print(f"  Best method: {results['best_overall_method']}")
        
        # Show stage timing information
        if results['individual_results']:
            first_result = results['individual_results'][0]
            stage_timing = first_result.pipeline_results.get('stage_timing', {})
            print(f"\n  Stage Timing (Run 1):")
            for stage, time_taken in stage_timing.items():
                print(f"    {stage}: {time_taken:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Example 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def example_3_configuration_from_file():
    """Example 3: Loading configuration from YAML file."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Configuration from YAML File")
    print("=" * 80)
    
    try:
        # Path to example configuration
        config_path = Path(__file__).parent / "configs" / "fast_experiment.yaml"
        
        if not config_path.exists():
            print(f"Configuration file not found: {config_path}")
            print("Creating a sample configuration file...")
            
            # Create the configs directory
            config_path.parent.mkdir(exist_ok=True)
            
            # Create and save a sample config
            sample_config = create_fast_config("yaml_experiment", "synthetic")
            sample_config.save(config_path)
            print(f"Sample configuration saved to: {config_path}")
        
        # Load configuration from file
        config = ExperimentConfig.load(config_path)
        print(f"Loaded configuration: {config.name}")
        
        # Validate configuration
        issues = config.validate()
        if issues:
            print("Configuration validation issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        print("Configuration is valid!")
        
        # Generate data according to config
        data_params = config.data_params
        X = generate_mixture_gaussians(
            n_samples=data_params.get("n_samples", 500),
            n_features=data_params.get("n_features", 200),
            n_components=data_params.get("n_components", 20),
            noise_level=data_params.get("noise_level", 0.1),
            random_state=config.random_seed
        )
        
        # Run experiment
        experiment = StagedOptimizationExperiment(config)
        results = experiment.run_complete_experiment(X)
        
        print(f"✓ YAML-based experiment completed!")
        print(f"  Configuration: {config.name}")
        print(f"  Best method: {results['best_overall_method']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Example 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def example_4_hyperparameter_search():
    """Example 4: Hyperparameter search experiment."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Hyperparameter Search")
    print("=" * 80)
    
    try:
        # Create configuration for hyperparameter search
        config = create_fast_config("hyperparam_search", "synthetic")
        config.n_runs = 1  # Single run for hyperparameter search
        
        # Generate small dataset for quick search
        X = generate_mixture_gaussians(200, 100, 10, random_state=42)
        
        # Define parameter grid for search
        param_grid = {
            "convex_k_candidates": {
                "type": "categorical",
                "values": [32, 64, 128]
            },
            "convex_lambda": {
                "type": "logrange",
                "start": 1e-6,
                "stop": 1e-2,
                "num": 4
            },
            "spherical_learning_rate": {
                "type": "categorical",
                "values": [0.01, 0.02, 0.05]
            }
        }
        
        print("Parameter grid defined with ranges:")
        print("  - Convex k_candidates: [32, 64, 128]")
        print("  - Convex lambda: log-range from 1e-6 to 1e-2")
        print("  - Spherical learning rate: [0.01, 0.02, 0.05]")
        
        # Create experiment and run hyperparameter search
        experiment = StagedOptimizationExperiment(config)
        
        # Note: This is a simplified example - full hyperparameter search
        # would require more sophisticated parameter mapping
        print("\nNote: This example demonstrates the hyperparameter search framework.")
        print("In a full implementation, the parameter grid would be mapped to")
        print("the specific configuration sections and the search would be executed.")
        
        # For demonstration, we'll run the basic experiment instead
        results = experiment.run_complete_experiment(X)
        
        print(f"✓ Hyperparameter search framework demonstrated!")
        print(f"  Best method from standard pipeline: {results['best_overall_method']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Example 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def example_5_method_comparison():
    """Example 5: Method comparison with baseline methods."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Method Comparison")
    print("=" * 80)
    
    try:
        # Create configuration
        config = create_fast_config("method_comparison", "synthetic")
        config.n_runs = 2  # Multiple runs for statistical comparison
        
        # Generate synthetic data
        X = generate_mixture_gaussians(300, 80, 8, random_state=42)
        
        # Define baseline methods for comparison
        from orthogonal_projection.dimensionality_reduction import (
            run_pca_simple, run_jll_simple, run_umap_simple
        )
        
        baseline_methods = {
            "pca": lambda X_input, y_input=None: run_pca_simple(X_input, k=20),
            "jll": lambda X_input, y_input=None: run_jll_simple(X_input, k=20),
        }
        
        # Try to add UMAP if available
        try:
            baseline_methods["umap"] = lambda X_input, y_input=None: run_umap_simple(X_input, k=20)
        except Exception:
            print("UMAP not available, skipping...")
        
        print(f"Comparing staged optimization against {len(baseline_methods)} baseline methods")
        
        # Create experiment and run method comparison
        experiment = StagedOptimizationExperiment(config)
        
        # Note: Method comparison requires careful implementation to ensure
        # fair comparison between methods. This is a simplified demonstration.
        print("\nRunning basic experiment (method comparison framework available)...")
        results = experiment.run_complete_experiment(X)
        
        print(f"✓ Method comparison framework demonstrated!")
        print(f"  Staged optimization result: {results['best_overall_method']}")
        print("  Full method comparison would rank all methods statistically")
        
        return True
        
    except Exception as e:
        print(f"✗ Example 5 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all examples."""
    print("STAGED OPTIMIZATION EXPERIMENT SYSTEM EXAMPLES")
    print("=" * 80)
    print("This script demonstrates the comprehensive experiment orchestration")
    print("system for dimensionality reduction with staged optimization.")
    print()
    
    examples = [
        ("Simple Synthetic Experiment", example_1_simple_synthetic_experiment),
        ("Custom Configuration", example_2_custom_configuration_experiment),
        ("Configuration from YAML", example_3_configuration_from_file),
        ("Hyperparameter Search", example_4_hyperparameter_search),
        ("Method Comparison", example_5_method_comparison)
    ]
    
    results = []
    for name, example_func in examples:
        print(f"\nRunning {name}...")
        success = example_func()
        results.append((name, success))
        
        if not success:
            print(f"Stopping after failure in {name}")
            break
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"Completed {successful}/{total} examples successfully")
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    if successful == total:
        print("\n🎉 All examples completed successfully!")
        print("\nThe staged optimization system is working correctly.")
        print("You can now:")
        print("  - Run experiments using the CLI: python -m orthogonal_projection.cli")
        print("  - Create custom configurations for your data")
        print("  - Use the programmatic API for advanced experiments")
    else:
        print(f"\n⚠️  {total - successful} examples failed.")
        print("Check the error messages above for troubleshooting.")
    
    return successful == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)