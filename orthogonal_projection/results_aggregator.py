"""
Results Aggregation and Best Method Selection System

This module provides comprehensive results aggregation, analysis, and 
best method selection capabilities for staged optimization experiments.
It includes advanced statistical analysis, visualization, and reporting.
"""

import numpy as np
import pandas as pd
import logging
import json
import pickle
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import auc
import warnings

# Import custom modules
try:
    from .experiment_config import ExperimentConfig
    from .hyperparameter_search import SearchResult, MethodComparisonResult
except ImportError:
    from experiment_config import ExperimentConfig
    from hyperparameter_search import SearchResult, MethodComparisonResult

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Single experiment result with metadata."""
    config: ExperimentConfig
    pipeline_results: Dict[str, Any]
    evaluation_metrics: Dict[str, Dict[str, float]]
    best_embedding: str
    runtime: float
    memory_usage: float
    timestamp: str
    run_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedResults:
    """Aggregated results across multiple experiments."""
    experiment_results: List[ExperimentResult]
    statistical_summary: Dict[str, Any]
    method_comparison: MethodComparisonResult
    best_overall_method: str
    performance_trends: Dict[str, Any]
    resource_usage_stats: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceAnalyzer:
    """Analyze performance across different dimensions."""
    
    @staticmethod
    def analyze_convergence(results: List[ExperimentResult]) -> Dict[str, Any]:
        """Analyze convergence patterns across experiments."""
        convergence_data = {}
        
        for result in results:
            # Extract loss histories from different stages
            pipeline_results = result.pipeline_results
            
            # Geometric embeddings convergence
            if "geometric_embeddings" in pipeline_results.get("stage_metadata", {}):
                geometric_meta = pipeline_results["stage_metadata"]["geometric_embeddings"]
                
                if "spherical_metadata" in geometric_meta:
                    spherical_info = geometric_meta["spherical_metadata"]["info"]
                    if "loss_history" in spherical_info:
                        convergence_data.setdefault("spherical_loss_history", []).append(
                            spherical_info["loss_history"]
                        )
                
                if "poincare_metadata" in geometric_meta:
                    # Poincaré embeddings might have different convergence metrics
                    poincare_info = geometric_meta["poincare_metadata"]
                    if "runtime" in poincare_info:
                        convergence_data.setdefault("poincare_runtime", []).append(
                            poincare_info["runtime"]
                        )
        
        # Analyze convergence statistics
        analysis = {}
        
        for method, histories in convergence_data.items():
            if histories:
                if "loss_history" in method:
                    # Analyze loss convergence
                    final_losses = [hist[-1] if hist else np.inf for hist in histories]
                    convergence_rates = []
                    
                    for hist in histories:
                        if len(hist) > 10:
                            # Calculate convergence rate (improvement over last 10% of iterations)
                            n_tail = len(hist) // 10
                            if n_tail > 0:
                                initial_loss = np.mean(hist[:n_tail])
                                final_loss = np.mean(hist[-n_tail:])
                                if initial_loss > 0:
                                    convergence_rate = (initial_loss - final_loss) / initial_loss
                                    convergence_rates.append(convergence_rate)
                    
                    analysis[method] = {
                        "mean_final_loss": np.mean(final_losses) if final_losses else np.inf,
                        "std_final_loss": np.std(final_losses) if final_losses else 0.0,
                        "mean_convergence_rate": np.mean(convergence_rates) if convergence_rates else 0.0,
                        "n_experiments": len(histories)
                    }
                else:
                    # For runtime or other scalar metrics
                    values = [v for v in histories if np.isfinite(v)]
                    analysis[method] = {
                        "mean": np.mean(values) if values else 0.0,
                        "std": np.std(values) if values else 0.0,
                        "min": np.min(values) if values else 0.0,
                        "max": np.max(values) if values else 0.0,
                        "n_experiments": len(values)
                    }
        
        return analysis
    
    @staticmethod
    def analyze_hyperparameter_sensitivity(results: List[ExperimentResult]) -> Dict[str, Any]:
        """Analyze sensitivity to hyperparameter choices."""
        sensitivity_data = {}
        
        for result in results:
            pipeline_results = result.pipeline_results
            
            # Extract hyperparameter choices and corresponding performance
            if "convex_optimization" in pipeline_results.get("stage_results", {}):
                convex_results = pipeline_results["stage_results"]["convex_optimization"]
                if "all_results" in convex_results:
                    for param_result in convex_results["all_results"]:
                        hyperparams = param_result.get("hyperparameters", {})
                        score = param_result.get("score", 0.0)
                        
                        for param_name, param_value in hyperparams.items():
                            key = f"convex_{param_name}"
                            if key not in sensitivity_data:
                                sensitivity_data[key] = {}
                            
                            param_str = str(param_value)
                            if param_str not in sensitivity_data[key]:
                                sensitivity_data[key][param_str] = []
                            
                            sensitivity_data[key][param_str].append(score)
        
        # Analyze sensitivity
        sensitivity_analysis = {}
        
        for param_name, value_scores in sensitivity_data.items():
            if len(value_scores) > 1:  # Need multiple values to analyze sensitivity
                # Calculate variance explained by this parameter
                all_scores = [score for scores in value_scores.values() for score in scores]
                total_variance = np.var(all_scores) if len(all_scores) > 1 else 0.0
                
                if total_variance > 0:
                    # Within-group variance
                    within_variance = 0.0
                    total_count = 0
                    
                    for scores in value_scores.values():
                        if len(scores) > 1:
                            within_variance += np.var(scores) * len(scores)
                            total_count += len(scores)
                    
                    if total_count > 0:
                        within_variance /= total_count
                        variance_explained = (total_variance - within_variance) / total_variance
                        
                        sensitivity_analysis[param_name] = {
                            "variance_explained": max(0.0, variance_explained),
                            "total_variance": total_variance,
                            "within_variance": within_variance,
                            "value_means": {val: np.mean(scores) for val, scores in value_scores.items()},
                            "value_stds": {val: np.std(scores) for val, scores in value_scores.items()},
                            "n_values": len(value_scores),
                            "n_experiments": len(all_scores)
                        }
        
        return sensitivity_analysis
    
    @staticmethod
    def analyze_scaling_behavior(results: List[ExperimentResult]) -> Dict[str, Any]:
        """Analyze how methods scale with data size and dimensionality."""
        scaling_data = {
            "data_sizes": [],
            "dimensionalities": [],
            "runtimes": {},
            "memory_usage": {},
            "performance": {}
        }
        
        for result in results:
            config = result.config
            pipeline_results = result.pipeline_results
            
            # Extract data characteristics
            if "preprocessing" in pipeline_results.get("stage_results", {}):
                preprocessing_result = pipeline_results["stage_results"]["preprocessing"]
                original_shape = preprocessing_result.get("original_shape", (0, 0))
                processed_shape = preprocessing_result.get("processed_shape", (0, 0))
                
                data_size = original_shape[0]
                dimensionality = original_shape[1]
                
                scaling_data["data_sizes"].append(data_size)
                scaling_data["dimensionalities"].append(dimensionality)
                
                # Extract timing and memory usage for each method
                stage_timing = pipeline_results.get("stage_timing", {})
                stage_memory = pipeline_results.get("stage_memory_usage", {})
                evaluation_results = pipeline_results.get("evaluation_results", {})
                
                for stage_name, runtime in stage_timing.items():
                    if stage_name not in scaling_data["runtimes"]:
                        scaling_data["runtimes"][stage_name] = []
                    scaling_data["runtimes"][stage_name].append({
                        "data_size": data_size,
                        "dimensionality": dimensionality,
                        "runtime": runtime
                    })
                
                for stage_name, memory in stage_memory.items():
                    if stage_name not in scaling_data["memory_usage"]:
                        scaling_data["memory_usage"][stage_name] = []
                    scaling_data["memory_usage"][stage_name].append({
                        "data_size": data_size,
                        "dimensionality": dimensionality,
                        "memory": memory
                    })
                
                # Performance scaling
                for method_name, metrics in evaluation_results.items():
                    if method_name not in scaling_data["performance"]:
                        scaling_data["performance"][method_name] = []
                    scaling_data["performance"][method_name].append({
                        "data_size": data_size,
                        "dimensionality": dimensionality,
                        "composite_score": metrics.get("composite_score", 0.0),
                        "rank_correlation": metrics.get("rank_correlation", 0.0)
                    })
        
        # Analyze scaling patterns
        scaling_analysis = {}
        
        # Runtime scaling analysis
        for stage_name, runtime_data in scaling_data["runtimes"].items():
            if len(runtime_data) > 3:  # Need sufficient data points
                data_sizes = np.array([d["data_size"] for d in runtime_data])
                runtimes = np.array([d["runtime"] for d in runtime_data])
                
                # Fit different scaling models
                scaling_models = {}
                
                try:
                    # Linear scaling: O(n)
                    linear_coeff = np.polyfit(data_sizes, runtimes, 1)
                    linear_pred = np.polyval(linear_coeff, data_sizes)
                    linear_r2 = 1 - np.sum((runtimes - linear_pred)**2) / np.sum((runtimes - np.mean(runtimes))**2)
                    scaling_models["linear"] = {"r2": linear_r2, "coefficients": linear_coeff.tolist()}
                    
                    # Quadratic scaling: O(n²)
                    if np.all(data_sizes > 0):
                        quad_coeff = np.polyfit(data_sizes, runtimes, 2)
                        quad_pred = np.polyval(quad_coeff, data_sizes)
                        quad_r2 = 1 - np.sum((runtimes - quad_pred)**2) / np.sum((runtimes - np.mean(runtimes))**2)
                        scaling_models["quadratic"] = {"r2": quad_r2, "coefficients": quad_coeff.tolist()}
                        
                        # Log scaling: O(n log n)
                        log_data_sizes = data_sizes * np.log(data_sizes)
                        log_coeff = np.polyfit(log_data_sizes, runtimes, 1)
                        log_pred = np.polyval(log_coeff, log_data_sizes)
                        log_r2 = 1 - np.sum((runtimes - log_pred)**2) / np.sum((runtimes - np.mean(runtimes))**2)
                        scaling_models["log_linear"] = {"r2": log_r2, "coefficients": log_coeff.tolist()}
                    
                    scaling_analysis[f"{stage_name}_runtime"] = {
                        "scaling_models": scaling_models,
                        "best_model": max(scaling_models.keys(), key=lambda x: scaling_models[x]["r2"]),
                        "n_data_points": len(runtime_data),
                        "data_size_range": (np.min(data_sizes), np.max(data_sizes)),
                        "runtime_range": (np.min(runtimes), np.max(runtimes))
                    }
                    
                except Exception as e:
                    logger.warning(f"Scaling analysis failed for {stage_name}: {str(e)}")
                    continue
        
        return scaling_analysis


class BestMethodSelector:
    """Advanced best method selection with multiple criteria."""
    
    def __init__(self, selection_criteria: Dict[str, float] = None):
        """
        Initialize best method selector.
        
        Parameters
        ----------
        selection_criteria : dict
            Weights for different selection criteria:
            - "rank_correlation": Spearman rank correlation weight
            - "distortion": Distortion penalty weight  
            - "nn_overlap": Nearest neighbor overlap weight
            - "runtime": Runtime efficiency weight
            - "memory": Memory efficiency weight
            - "robustness": Robustness across experiments weight
            - "statistical_significance": Statistical significance weight
        """
        self.selection_criteria = selection_criteria or {
            "rank_correlation": 0.4,
            "distortion": -0.2,  # Negative because lower is better
            "nn_overlap": 0.2,
            "runtime": -0.1,     # Negative because lower is better
            "memory": -0.05,     # Negative because lower is better
            "robustness": 0.15,
            "statistical_significance": 0.1
        }
        
        # Normalize weights
        total_weight = sum(abs(w) for w in self.selection_criteria.values())
        if total_weight > 0:
            self.selection_criteria = {k: v/total_weight for k, v in self.selection_criteria.items()}
    
    def select_best_method(self, aggregated_results: AggregatedResults) -> Dict[str, Any]:
        """
        Select the best method based on multiple criteria.
        
        Parameters
        ----------
        aggregated_results : AggregatedResults
            Aggregated experimental results
            
        Returns
        -------
        dict
            Best method selection results with detailed analysis
        """
        logger.info("Selecting best method based on comprehensive criteria")
        
        # Collect method performance data
        method_performance = self._collect_method_performance(aggregated_results)
        
        # Calculate criterion scores for each method
        criterion_scores = {}
        for method_name in method_performance.keys():
            criterion_scores[method_name] = self._calculate_criterion_scores(
                method_name, method_performance, aggregated_results
            )
        
        # Calculate composite scores
        composite_scores = {}
        for method_name, scores in criterion_scores.items():
            composite_score = sum(
                self.selection_criteria.get(criterion, 0.0) * score
                for criterion, score in scores.items()
            )
            composite_scores[method_name] = composite_score
        
        # Rank methods
        method_rankings = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        best_method = method_rankings[0][0]
        
        # Generate detailed analysis
        selection_analysis = {
            "best_method": best_method,
            "best_score": method_rankings[0][1],
            "method_rankings": method_rankings,
            "criterion_scores": criterion_scores,
            "composite_scores": composite_scores,
            "selection_criteria": self.selection_criteria,
            "detailed_analysis": self._generate_selection_analysis(
                method_performance, criterion_scores, composite_scores, aggregated_results
            )
        }
        
        logger.info(f"Best method selected: {best_method} (score: {method_rankings[0][1]:.4f})")
        
        return selection_analysis
    
    def _collect_method_performance(self, aggregated_results: AggregatedResults) -> Dict[str, Dict[str, List[float]]]:
        """Collect performance metrics for all methods across experiments."""
        method_performance = {}
        
        for experiment_result in aggregated_results.experiment_results:
            evaluation_metrics = experiment_result.evaluation_metrics
            
            for method_name, metrics in evaluation_metrics.items():
                if method_name not in method_performance:
                    method_performance[method_name] = {
                        "rank_correlation": [],
                        "mean_distortion": [],
                        "max_distortion": [],
                        "nn_overlap": [],
                        "composite_score": []
                    }
                
                # Collect metrics
                for metric_name in method_performance[method_name].keys():
                    value = metrics.get(metric_name, 0.0)
                    if np.isfinite(value):
                        method_performance[method_name][metric_name].append(value)
        
        return method_performance
    
    def _calculate_criterion_scores(self, method_name: str, method_performance: Dict, 
                                   aggregated_results: AggregatedResults) -> Dict[str, float]:
        """Calculate normalized criterion scores for a method."""
        scores = {}
        
        # Performance-based criteria
        performance_data = method_performance[method_name]
        
        # Rank correlation score (0-1, higher is better)
        rank_correlations = performance_data.get("rank_correlation", [])
        scores["rank_correlation"] = np.mean(rank_correlations) if rank_correlations else 0.0
        
        # Distortion score (normalized, lower is better, so we invert)
        mean_distortions = performance_data.get("mean_distortion", [])
        if mean_distortions:
            # Normalize distortion relative to all methods
            all_distortions = []
            for other_method_data in method_performance.values():
                all_distortions.extend(other_method_data.get("mean_distortion", []))
            
            if all_distortions:
                max_distortion = max(all_distortions)
                min_distortion = min(all_distortions)
                if max_distortion > min_distortion:
                    normalized_distortion = (np.mean(mean_distortions) - min_distortion) / (max_distortion - min_distortion)
                    scores["distortion"] = 1.0 - normalized_distortion  # Invert so higher is better
                else:
                    scores["distortion"] = 0.5  # All methods have same distortion
            else:
                scores["distortion"] = 0.5
        else:
            scores["distortion"] = 0.0
        
        # Nearest neighbor overlap score
        nn_overlaps = performance_data.get("nn_overlap", [])
        scores["nn_overlap"] = np.mean(nn_overlaps) if nn_overlaps else 0.0
        
        # Runtime efficiency score
        runtime_scores = []
        for experiment_result in aggregated_results.experiment_results:
            if method_name in experiment_result.evaluation_metrics:
                # Use overall runtime for this experiment
                runtime_scores.append(experiment_result.runtime)
        
        if runtime_scores:
            # Normalize runtime (lower is better, so invert)
            all_runtimes = [exp.runtime for exp in aggregated_results.experiment_results]
            if all_runtimes:
                max_runtime = max(all_runtimes)
                min_runtime = min(all_runtimes)
                if max_runtime > min_runtime:
                    normalized_runtime = (np.mean(runtime_scores) - min_runtime) / (max_runtime - min_runtime)
                    scores["runtime"] = 1.0 - normalized_runtime
                else:
                    scores["runtime"] = 0.5
            else:
                scores["runtime"] = 0.5
        else:
            scores["runtime"] = 0.0
        
        # Memory efficiency score
        memory_scores = []
        for experiment_result in aggregated_results.experiment_results:
            if method_name in experiment_result.evaluation_metrics:
                memory_scores.append(experiment_result.memory_usage)
        
        if memory_scores:
            # Normalize memory usage (lower is better, so invert)
            all_memory = [exp.memory_usage for exp in aggregated_results.experiment_results]
            if all_memory:
                max_memory = max(all_memory)
                min_memory = min(all_memory)
                if max_memory > min_memory:
                    normalized_memory = (np.mean(memory_scores) - min_memory) / (max_memory - min_memory)
                    scores["memory"] = 1.0 - normalized_memory
                else:
                    scores["memory"] = 0.5
            else:
                scores["memory"] = 0.5
        else:
            scores["memory"] = 0.0
        
        # Robustness score (consistency across experiments)
        composite_scores = performance_data.get("composite_score", [])
        if len(composite_scores) > 1:
            # Use coefficient of variation (lower is better, so invert)
            mean_composite = np.mean(composite_scores)
            std_composite = np.std(composite_scores)
            if mean_composite > 0:
                cv = std_composite / mean_composite
                # Normalize CV and invert (lower CV is better robustness)
                scores["robustness"] = max(0.0, 1.0 - cv)  # Assuming CV rarely exceeds 1
            else:
                scores["robustness"] = 0.0
        else:
            scores["robustness"] = 0.0
        
        # Statistical significance score
        method_comparison = aggregated_results.method_comparison
        if method_comparison and hasattr(method_comparison, 'statistical_tests'):
            # Check if this method is significantly better than others
            significance_count = 0
            total_comparisons = 0
            
            paired_t_results = method_comparison.statistical_tests.get("paired_t_test", {})
            for comparison, p_value in paired_t_results.items():
                if method_name in comparison:
                    total_comparisons += 1
                    if p_value < 0.05:  # Statistically significant
                        # Check if this method is the better one
                        methods = comparison.split("_vs_")
                        if len(methods) == 2:
                            method1, method2 = methods
                            # Need to check which method performed better
                            # For simplicity, assume this method is better if it's first in comparison
                            if method1 == method_name:
                                significance_count += 1
            
            if total_comparisons > 0:
                scores["statistical_significance"] = significance_count / total_comparisons
            else:
                scores["statistical_significance"] = 0.0
        else:
            scores["statistical_significance"] = 0.0
        
        # Ensure all scores are in [0, 1] range
        for criterion, score in scores.items():
            scores[criterion] = max(0.0, min(1.0, score))
        
        return scores
    
    def _generate_selection_analysis(self, method_performance: Dict, criterion_scores: Dict, 
                                   composite_scores: Dict, aggregated_results: AggregatedResults) -> Dict[str, Any]:
        """Generate detailed analysis of method selection."""
        analysis = {}
        
        # Performance summary for each method
        method_summaries = {}
        for method_name, performance_data in method_performance.items():
            summary = {}
            for metric_name, values in performance_data.items():
                if values:
                    summary[f"{metric_name}_mean"] = np.mean(values)
                    summary[f"{metric_name}_std"] = np.std(values)
                    summary[f"{metric_name}_min"] = np.min(values)
                    summary[f"{metric_name}_max"] = np.max(values)
                else:
                    summary[f"{metric_name}_mean"] = 0.0
                    summary[f"{metric_name}_std"] = 0.0
                    summary[f"{metric_name}_min"] = 0.0
                    summary[f"{metric_name}_max"] = 0.0
            
            method_summaries[method_name] = summary
        
        analysis["method_summaries"] = method_summaries
        
        # Criterion importance analysis
        criterion_importance = {}
        best_method = max(composite_scores, key=composite_scores.get)
        best_scores = criterion_scores[best_method]
        
        for criterion, weight in self.selection_criteria.items():
            if criterion in best_scores:
                contribution = weight * best_scores[criterion]
                criterion_importance[criterion] = {
                    "weight": weight,
                    "score": best_scores[criterion],
                    "contribution": contribution,
                    "relative_importance": abs(contribution) / sum(abs(w * best_scores.get(c, 0)) 
                                                                  for c, w in self.selection_criteria.items())
                }
        
        analysis["criterion_importance"] = criterion_importance
        
        # Method comparison matrix
        comparison_matrix = {}
        method_names = list(composite_scores.keys())
        
        for i, method1 in enumerate(method_names):
            comparison_matrix[method1] = {}
            for j, method2 in enumerate(method_names):
                if i != j:
                    score_diff = composite_scores[method1] - composite_scores[method2]
                    comparison_matrix[method1][method2] = {
                        "score_difference": score_diff,
                        "better": score_diff > 0
                    }
        
        analysis["comparison_matrix"] = comparison_matrix
        
        # Sensitivity analysis
        sensitivity_analysis = {}
        for criterion, weight in self.selection_criteria.items():
            # Calculate how rankings would change with different weights
            modified_criteria = self.selection_criteria.copy()
            modified_criteria[criterion] = 0.0  # Remove this criterion
            
            # Recalculate composite scores without this criterion
            modified_scores = {}
            for method_name, scores in criterion_scores.items():
                modified_score = sum(
                    modified_criteria.get(crit, 0.0) * score
                    for crit, score in scores.items()
                )
                modified_scores[method_name] = modified_score
            
            modified_best = max(modified_scores, key=modified_scores.get)
            
            sensitivity_analysis[criterion] = {
                "original_weight": weight,
                "best_method_without": modified_best,
                "ranking_stable": modified_best == best_method
            }
        
        analysis["sensitivity_analysis"] = sensitivity_analysis
        
        return analysis


class ResultsAggregator:
    """Main results aggregator that combines all analysis components."""
    
    def __init__(self, output_dir: Union[str, Path] = "aggregated_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.performance_analyzer = PerformanceAnalyzer()
        self.best_method_selector = BestMethodSelector()
        
    def aggregate_experiment_results(self, experiment_results: List[ExperimentResult]) -> AggregatedResults:
        """
        Aggregate results from multiple experiments.
        
        Parameters
        ----------
        experiment_results : list
            List of individual experiment results
            
        Returns
        -------
        AggregatedResults
            Comprehensive aggregated analysis
        """
        logger.info(f"Aggregating results from {len(experiment_results)} experiments")
        
        # Statistical summary
        statistical_summary = self._compute_statistical_summary(experiment_results)
        
        # Method comparison (if multiple methods were compared)
        method_comparison = self._perform_method_comparison(experiment_results)
        
        # Performance trends analysis
        performance_trends = {
            "convergence_analysis": self.performance_analyzer.analyze_convergence(experiment_results),
            "hyperparameter_sensitivity": self.performance_analyzer.analyze_hyperparameter_sensitivity(experiment_results),
            "scaling_behavior": self.performance_analyzer.analyze_scaling_behavior(experiment_results)
        }
        
        # Resource usage statistics
        resource_usage_stats = self._analyze_resource_usage(experiment_results)
        
        # Create aggregated results
        aggregated = AggregatedResults(
            experiment_results=experiment_results,
            statistical_summary=statistical_summary,
            method_comparison=method_comparison,
            best_overall_method="",  # Will be determined by best method selector
            performance_trends=performance_trends,
            resource_usage_stats=resource_usage_stats,
            metadata={
                "n_experiments": len(experiment_results),
                "aggregation_timestamp": datetime.now().isoformat(),
                "experiment_date_range": self._get_experiment_date_range(experiment_results)
            }
        )
        
        # Select best method
        if method_comparison:
            best_method_analysis = self.best_method_selector.select_best_method(aggregated)
            aggregated.best_overall_method = best_method_analysis["best_method"]
            aggregated.metadata["best_method_analysis"] = best_method_analysis
        
        return aggregated
    
    def _compute_statistical_summary(self, experiment_results: List[ExperimentResult]) -> Dict[str, Any]:
        """Compute comprehensive statistical summary."""
        summary = {}
        
        # Collect all evaluation metrics
        all_metrics = {}
        for result in experiment_results:
            for method_name, metrics in result.evaluation_metrics.items():
                if method_name not in all_metrics:
                    all_metrics[method_name] = {metric: [] for metric in metrics.keys()}
                
                for metric_name, value in metrics.items():
                    if np.isfinite(value):
                        all_metrics[method_name][metric_name].append(value)
        
        # Compute statistics for each method and metric
        for method_name, method_metrics in all_metrics.items():
            summary[method_name] = {}
            
            for metric_name, values in method_metrics.items():
                if values:
                    summary[method_name][metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "median": np.median(values),
                        "q25": np.percentile(values, 25),
                        "q75": np.percentile(values, 75),
                        "n_samples": len(values)
                    }
                    
                    # Add confidence intervals
                    if len(values) > 1:
                        sem = stats.sem(values)
                        ci_lower, ci_upper = stats.t.interval(
                            0.95, len(values)-1, loc=np.mean(values), scale=sem
                        )
                        summary[method_name][metric_name]["ci_lower"] = ci_lower
                        summary[method_name][metric_name]["ci_upper"] = ci_upper
        
        return summary
    
    def _perform_method_comparison(self, experiment_results: List[ExperimentResult]) -> Optional[MethodComparisonResult]:
        """Perform statistical comparison between methods."""
        # This is a simplified version - in practice, you'd want more sophisticated comparison
        method_scores = {}
        
        for result in experiment_results:
            for method_name, metrics in result.evaluation_metrics.items():
                if method_name not in method_scores:
                    method_scores[method_name] = []
                
                # Use composite score as overall performance measure
                composite_score = metrics.get("composite_score", 0.0)
                if np.isfinite(composite_score):
                    method_scores[method_name].append(composite_score)
        
        if len(method_scores) < 2:
            return None  # Need at least 2 methods to compare
        
        # Perform pairwise statistical tests
        statistical_tests = {"paired_t_test": {}}
        method_names = list(method_scores.keys())
        
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names[i+1:], i+1):
                scores1 = method_scores[method1]
                scores2 = method_scores[method2]
                
                if len(scores1) >= 3 and len(scores2) >= 3:
                    try:
                        _, p_value = stats.ttest_ind(scores1, scores2)
                        pair_key = f"{method1}_vs_{method2}"
                        statistical_tests["paired_t_test"][pair_key] = p_value
                    except Exception:
                        pass
        
        # Calculate method rankings
        method_rankings = []
        for method_name, scores in method_scores.items():
            if scores:
                mean_score = np.mean(scores)
                method_rankings.append((method_name, mean_score))
        
        method_rankings.sort(key=lambda x: x[1], reverse=True)
        best_method = method_rankings[0][0] if method_rankings else ""
        
        # Create significance matrix (simplified)
        n_methods = len(method_names)
        significance_matrix = np.ones((n_methods, n_methods))
        
        # Calculate effect sizes (simplified)
        effect_sizes = {}
        if method_rankings:
            worst_score = method_rankings[-1][1]
            for method_name, mean_score in method_rankings:
                effect_sizes[method_name] = mean_score - worst_score
        
        return MethodComparisonResult(
            method_scores=method_scores,
            statistical_tests=statistical_tests,
            best_method=best_method,
            method_rankings=method_rankings,
            significance_matrix=significance_matrix,
            effect_sizes=effect_sizes
        )
    
    def _analyze_resource_usage(self, experiment_results: List[ExperimentResult]) -> Dict[str, Any]:
        """Analyze computational resource usage."""
        runtimes = [result.runtime for result in experiment_results if np.isfinite(result.runtime)]
        memory_usage = [result.memory_usage for result in experiment_results if np.isfinite(result.memory_usage)]
        
        stats_dict = {}
        
        if runtimes:
            stats_dict["runtime"] = {
                "mean": np.mean(runtimes),
                "std": np.std(runtimes),
                "min": np.min(runtimes),
                "max": np.max(runtimes),
                "total": np.sum(runtimes)
            }
        
        if memory_usage:
            stats_dict["memory"] = {
                "mean": np.mean(memory_usage),
                "std": np.std(memory_usage),
                "min": np.min(memory_usage),
                "max": np.max(memory_usage),
                "peak": np.max(memory_usage)
            }
        
        return stats_dict
    
    def _get_experiment_date_range(self, experiment_results: List[ExperimentResult]) -> Dict[str, str]:
        """Get date range of experiments."""
        timestamps = [result.timestamp for result in experiment_results if result.timestamp]
        
        if timestamps:
            return {
                "earliest": min(timestamps),
                "latest": max(timestamps)
            }
        else:
            return {"earliest": "", "latest": ""}
    
    def save_aggregated_results(self, aggregated_results: AggregatedResults, filename: str = "aggregated_results") -> Path:
        """Save aggregated results to disk."""
        # Save as pickle for complete data
        pickle_path = self.output_dir / f"{filename}.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(aggregated_results, f)
        
        # Save summary as JSON
        json_path = self.output_dir / f"{filename}_summary.json"
        summary_dict = {
            "best_overall_method": aggregated_results.best_overall_method,
            "n_experiments": aggregated_results.metadata.get("n_experiments", 0),
            "statistical_summary": aggregated_results.statistical_summary,
            "resource_usage_stats": aggregated_results.resource_usage_stats,
            "metadata": aggregated_results.metadata
        }
        
        with open(json_path, "w") as f:
            json.dump(summary_dict, f, indent=2, default=str)
        
        logger.info(f"Aggregated results saved to {pickle_path} and {json_path}")
        return pickle_path
    
    def generate_comprehensive_report(self, aggregated_results: AggregatedResults) -> str:
        """Generate a comprehensive text report."""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE EXPERIMENT ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Experiment overview
        metadata = aggregated_results.metadata
        report.append("EXPERIMENT OVERVIEW")
        report.append("-" * 40)
        report.append(f"Number of experiments: {metadata.get('n_experiments', 0)}")
        report.append(f"Date range: {metadata.get('experiment_date_range', {}).get('earliest', 'N/A')} to {metadata.get('experiment_date_range', {}).get('latest', 'N/A')}")
        report.append(f"Best overall method: {aggregated_results.best_overall_method}")
        report.append("")
        
        # Method performance summary
        report.append("METHOD PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        if aggregated_results.method_comparison:
            for method_name, score in aggregated_results.method_comparison.method_rankings:
                report.append(f"{method_name}: {score:.4f}")
        report.append("")
        
        # Statistical summary
        report.append("STATISTICAL SUMMARY")
        report.append("-" * 40)
        
        for method_name, method_stats in aggregated_results.statistical_summary.items():
            report.append(f"\n{method_name}:")
            
            for metric_name, stats in method_stats.items():
                if isinstance(stats, dict) and "mean" in stats:
                    report.append(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        report.append("")
        
        # Resource usage
        report.append("RESOURCE USAGE")
        report.append("-" * 40)
        
        resource_stats = aggregated_results.resource_usage_stats
        if "runtime" in resource_stats:
            runtime_stats = resource_stats["runtime"]
            report.append(f"Runtime - Mean: {runtime_stats['mean']:.2f}s, Total: {runtime_stats['total']:.2f}s")
        
        if "memory" in resource_stats:
            memory_stats = resource_stats["memory"]
            report.append(f"Memory - Mean: {memory_stats['mean']:.2f}GB, Peak: {memory_stats['peak']:.2f}GB")
        
        report.append("")
        
        # Best method analysis
        if "best_method_analysis" in metadata:
            best_analysis = metadata["best_method_analysis"]
            report.append("BEST METHOD ANALYSIS")
            report.append("-" * 40)
            report.append(f"Selected method: {best_analysis['best_method']}")
            report.append(f"Composite score: {best_analysis['best_score']:.4f}")
            
            # Top criteria contributions
            if "detailed_analysis" in best_analysis and "criterion_importance" in best_analysis["detailed_analysis"]:
                criterion_importance = best_analysis["detailed_analysis"]["criterion_importance"]
                report.append("\nTop contributing criteria:")
                
                sorted_criteria = sorted(criterion_importance.items(), 
                                       key=lambda x: abs(x[1]["contribution"]), reverse=True)
                
                for criterion, info in sorted_criteria[:5]:
                    report.append(f"  {criterion}: {info['contribution']:.4f} "
                                f"(weight: {info['weight']:.2f}, score: {info['score']:.4f})")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)