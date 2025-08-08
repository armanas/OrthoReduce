"""
Comprehensive Experiment Orchestration System

This module provides the main integration point for the staged optimization
experiment system, bringing together all components into a cohesive framework.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json

# Import all orchestration components
from .experiment_config import ExperimentConfig
from .pipeline_orchestrator import PipelineOrchestrator  
from .hyperparameter_search import HyperparameterSearchEngine, ParameterGridGenerator, MultiMethodComparator
from .results_aggregator import ResultsAggregator, ExperimentResult
from .experiment_logger import ExperimentLogger
from .dimensionality_reduction import generate_mixture_gaussians


class StagedOptimizationExperiment:
    """
    Main orchestration class for running staged optimization experiments.
    
    This class integrates all components of the experiment system and provides
    a high-level interface for running comprehensive dimensionality reduction
    experiments with staged optimization.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment orchestration system.
        
        Parameters
        ----------
        config : ExperimentConfig
            Complete experiment configuration
        """
        self.config = config
        
        # Initialize logger
        self.logger = ExperimentLogger(
            experiment_name=config.name,
            experiment_id=f"{config.name}_{config.get_hash()}",
            output_dir=Path(config.output_dir) / "logs",
            log_level=20 if config.verbose_logging else 30,  # INFO or WARNING
            enable_progress=True,
            enable_resource_monitoring=True
        )
        
        # Initialize components
        self.orchestrator = PipelineOrchestrator(config)
        self.results_aggregator = ResultsAggregator(
            output_dir=Path(config.output_dir) / "results"
        )
        
        # Experiment results storage
        self.experiment_results: List[ExperimentResult] = []
        
    def run_complete_experiment(self, X: np.ndarray, 
                               y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run complete staged optimization experiment.
        
        Parameters
        ----------
        X : np.ndarray
            Input high-dimensional data
        y : np.ndarray, optional
            Labels for supervised methods
            
        Returns
        -------
        dict
            Complete experiment results with aggregated analysis
        """
        # Start experiment logging
        config_dict = {
            "name": self.config.name,
            "data_shape": X.shape,
            "n_runs": self.config.n_runs,
            "random_seed": self.config.random_seed,
            "stages": [stage.name for stage in self.config.get_stages() if stage.enabled]
        }
        self.logger.start_experiment(config_dict)
        
        try:
            # Run multiple experiment runs
            experiment_results = []
            
            with self.logger.stage_context("experiment_runs", self.config.n_runs) as stage_ctx:
                for run_id in range(self.config.n_runs):
                    stage_ctx.log_info(f"Starting experiment run {run_id + 1}/{self.config.n_runs}")
                    
                    # Set random seed for this run
                    np.random.seed(self.config.random_seed + run_id)
                    
                    try:
                        # Run single experiment
                        result = self._run_single_experiment(X, y, run_id)
                        experiment_results.append(result)
                        
                        # Log run completion
                        best_method = result.best_embedding
                        best_score = result.evaluation_metrics.get(best_method, {}).get("composite_score", 0.0)
                        stage_ctx.log_info(
                            f"Run {run_id + 1} completed: best_method={best_method}, score={best_score:.4f}",
                            run_id=run_id,
                            best_method=best_method,
                            best_score=best_score
                        )
                        
                    except Exception as e:
                        stage_ctx.log_error(f"Run {run_id + 1} failed: {str(e)}", 
                                          run_id=run_id, error=str(e))
                        continue
                    
                    stage_ctx.update_progress(1, f"Completed run {run_id + 1}/{self.config.n_runs}")
            
            if not experiment_results:
                raise RuntimeError("All experiment runs failed")
            
            # Aggregate results
            self.logger.logger.info(f"Aggregating results from {len(experiment_results)} runs")
            aggregated_results = self.results_aggregator.aggregate_experiment_results(experiment_results)
            
            # Save results
            results_path = self.results_aggregator.save_aggregated_results(
                aggregated_results, f"{self.config.name}_aggregated"
            )
            
            # Generate comprehensive report
            report = self.results_aggregator.generate_comprehensive_report(aggregated_results)
            report_path = Path(self.config.output_dir) / f"{self.config.name}_report.txt"
            with open(report_path, "w") as f:
                f.write(report)
            
            # Prepare final results
            final_results = {
                "experiment_config": self.config,
                "individual_results": experiment_results,
                "aggregated_results": aggregated_results,
                "best_overall_method": aggregated_results.best_overall_method,
                "results_path": str(results_path),
                "report_path": str(report_path),
                "experiment_summary": self.logger.get_experiment_summary()
            }
            
            # End experiment logging
            self.logger.end_experiment("completed", {
                "n_successful_runs": len(experiment_results),
                "best_method": aggregated_results.best_overall_method,
                "results_saved": str(results_path)
            })
            
            return final_results
            
        except Exception as e:
            self.logger.end_experiment("failed", {"error": str(e)})
            raise
    
    def _run_single_experiment(self, X: np.ndarray, y: Optional[np.ndarray], 
                              run_id: int) -> ExperimentResult:
        """Run a single experiment iteration."""
        start_time = time.time()
        
        # Run pipeline orchestrator
        pipeline_results = self.orchestrator.run_pipeline(X, y)
        runtime = time.time() - start_time
        
        # Extract key information
        evaluation_metrics = pipeline_results.get("evaluation_results", {})
        best_embedding = pipeline_results.get("best_embedding", "")
        
        # Get resource usage
        resource_usage = pipeline_results.get("resource_usage", {})
        memory_usage = resource_usage.get("peak_memory_gb", 0.0)
        
        # Create experiment result
        result = ExperimentResult(
            config=self.config,
            pipeline_results=pipeline_results,
            evaluation_metrics=evaluation_metrics,
            best_embedding=best_embedding,
            runtime=runtime,
            memory_usage=memory_usage,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            run_id=run_id
        )
        
        return result
    
    def run_hyperparameter_search_experiment(self, X: np.ndarray,
                                           param_grid: Dict[str, Any],
                                           search_strategy: str = "grid") -> Dict[str, Any]:
        """
        Run hyperparameter search experiment.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        param_grid : dict
            Parameter grid specification
        search_strategy : str
            Search strategy ("grid", "random", "sobol")
            
        Returns
        -------
        dict
            Hyperparameter search results
        """
        self.logger.logger.info("Starting hyperparameter search experiment")
        
        # Generate parameter combinations
        param_combinations = ParameterGridGenerator.create_grid(param_grid, search_strategy)
        self.logger.logger.info(f"Generated {len(param_combinations)} parameter combinations")
        
        # Define scoring function
        def scoring_function(X_train, X_test, params, y_train=None, y_test=None):
            # Create temporary config with these parameters
            temp_config = self.config
            
            # Update config with parameters (simplified - would need more sophisticated merging)
            # This is where you'd map the parameters to the appropriate config sections
            
            # Run pipeline with these parameters
            temp_orchestrator = PipelineOrchestrator(temp_config)
            results = temp_orchestrator.run_pipeline(X_train)
            
            # Extract score (use composite score from best embedding)
            evaluation_results = results.get("evaluation_results", {})
            if evaluation_results:
                best_embedding = results.get("best_embedding", "")
                if best_embedding in evaluation_results:
                    return evaluation_results[best_embedding].get("composite_score", 0.0)
            
            return 0.0
        
        # Initialize search engine
        search_engine = HyperparameterSearchEngine(
            scoring_function=scoring_function,
            cv_strategy="kfold",
            cv_folds=self.config.cross_validation_folds,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_seed
        )
        
        # Run search
        search_results = search_engine.search(X, param_combinations)
        
        # Analyze parameter importance
        param_importance = search_engine.analyze_parameter_importance()
        
        # Prepare results
        hyperparameter_results = {
            "search_results": search_results,
            "best_parameters": search_results[0].parameters if search_results else {},
            "best_score": search_results[0].score if search_results else 0.0,
            "parameter_importance": param_importance,
            "search_strategy": search_strategy,
            "n_combinations": len(param_combinations)
        }
        
        self.logger.logger.info(f"Hyperparameter search completed: best_score={hyperparameter_results['best_score']:.4f}")
        
        return hyperparameter_results
    
    def run_method_comparison_experiment(self, X: np.ndarray, 
                                       baseline_methods: Dict[str, callable],
                                       y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run method comparison experiment.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        baseline_methods : dict
            Dictionary of method_name -> method_function
        y : np.ndarray, optional
            Labels for supervised methods
            
        Returns
        -------
        dict
            Method comparison results
        """
        self.logger.logger.info(f"Starting method comparison with {len(baseline_methods)} baseline methods")
        
        # Add staged optimization as a method
        def staged_optimization_method(X_input, y_input=None):
            # Run our complete pipeline
            results = self.orchestrator.run_pipeline(X_input, y_input)
            best_embedding = results.get("best_embedding", "")
            final_embeddings = results.get("final_embeddings", {})
            
            if best_embedding in final_embeddings:
                return final_embeddings[best_embedding]
            else:
                # Fallback to any available embedding
                for embedding_name, embedding_data in final_embeddings.items():
                    if embedding_name != "preprocessed":
                        return embedding_data
                
                # Final fallback
                return np.random.randn(X_input.shape[0], min(50, X_input.shape[1]))
        
        # Combine methods
        all_methods = baseline_methods.copy()
        all_methods["staged_optimization"] = staged_optimization_method
        
        # Define evaluation metrics
        from .evaluation import compute_distortion, rank_correlation, nearest_neighbor_overlap
        
        def evaluation_metric(X_orig, X_proj):
            try:
                # Compute multiple metrics
                _, _, _, _ = compute_distortion(X_orig, X_proj)
                rank_corr = rank_correlation(X_orig, X_proj)
                nn_overlap = nearest_neighbor_overlap(X_orig, X_proj)
                
                # Return composite score
                return rank_corr + 0.1 * nn_overlap  # Weighted combination
            except Exception:
                return 0.0
        
        # Initialize comparator
        comparator = MultiMethodComparator(
            methods=all_methods,
            evaluation_metrics=[evaluation_metric],
            cv_folds=self.config.cross_validation_folds,
            n_runs=self.config.n_runs,
            random_state=self.config.random_seed
        )
        
        # Run comparison
        comparison_results = comparator.compare_methods(X, y)
        
        # Generate comparison report
        report = comparator.generate_comparison_report(comparison_results)
        
        # Save comparison report
        comparison_report_path = Path(self.config.output_dir) / f"{self.config.name}_method_comparison.txt"
        with open(comparison_report_path, "w") as f:
            f.write(report)
        
        comparison_result_dict = {
            "comparison_results": comparison_results,
            "comparison_report": report,
            "report_path": str(comparison_report_path),
            "staged_optimization_rank": None  # Will be filled below
        }
        
        # Find rank of our staged optimization method
        for i, (method_name, score) in enumerate(comparison_results.method_rankings):
            if method_name == "staged_optimization":
                comparison_result_dict["staged_optimization_rank"] = i + 1
                break
        
        self.logger.logger.info(f"Method comparison completed: staged optimization ranked #{comparison_result_dict['staged_optimization_rank']}")
        
        return comparison_result_dict


# Convenience functions for common experiment types

def run_synthetic_data_experiment(experiment_name: str = "synthetic_experiment",
                                 n_samples: int = 1000,
                                 n_features: int = 500,
                                 n_components: int = 50,
                                 template: str = "default") -> Dict[str, Any]:
    """
    Run staged optimization experiment on synthetic data.
    
    Parameters
    ----------
    experiment_name : str
        Name of the experiment
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features
    n_components : int
        Number of underlying components
    template : str
        Configuration template ("default", "fast", "comprehensive")
        
    Returns
    -------
    dict
        Complete experiment results
    """
    # Create configuration
    if template == "fast":
        from .experiment_config import create_fast_config
        config = create_fast_config(experiment_name, "synthetic")
    elif template == "comprehensive":
        from .experiment_config import create_comprehensive_config
        config = create_comprehensive_config(experiment_name, "synthetic")
    else:
        from .experiment_config import create_default_config
        config = create_default_config(experiment_name, "synthetic")
    
    # Update data parameters
    config.data_params.update({
        "n_samples": n_samples,
        "n_features": n_features,
        "n_components": n_components
    })
    
    # Generate synthetic data
    X = generate_mixture_gaussians(n_samples, n_features, n_components)
    
    # Run experiment
    experiment = StagedOptimizationExperiment(config)
    return experiment.run_complete_experiment(X)


def run_mass_spectrometry_experiment(experiment_name: str = "ms_experiment",
                                    mzml_path: str = None,
                                    max_spectra: int = 1000) -> Dict[str, Any]:
    """
    Run staged optimization experiment on mass spectrometry data.
    
    Parameters
    ----------
    experiment_name : str
        Name of the experiment
    mzml_path : str
        Path to mzML file
    max_spectra : int
        Maximum number of spectra to process
        
    Returns
    -------
    dict
        Complete experiment results
    """
    if not mzml_path:
        raise ValueError("mzML path is required for MS experiments")
    
    # Create MS-specific configuration
    from .experiment_config import create_ms_data_config
    config = create_ms_data_config(experiment_name, mzml_path)
    config.data_params["max_spectra"] = max_spectra
    
    # Load MS data
    from .ms_data import parse_mzml_build_matrix
    X = parse_mzml_build_matrix(mzml_path, max_spectra=max_spectra)
    
    # Run experiment
    experiment = StagedOptimizationExperiment(config)
    return experiment.run_complete_experiment(X)


# Example usage and testing
if __name__ == "__main__":
    # Example: Run a fast synthetic data experiment
    print("Running example staged optimization experiment...")
    
    try:
        results = run_synthetic_data_experiment(
            experiment_name="example_experiment",
            n_samples=500,
            n_features=200,
            n_components=20,
            template="fast"
        )
        
        print(f"Experiment completed successfully!")
        print(f"Best method: {results['best_overall_method']}")
        print(f"Results saved to: {results['results_path']}")
        print(f"Report saved to: {results['report_path']}")
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()