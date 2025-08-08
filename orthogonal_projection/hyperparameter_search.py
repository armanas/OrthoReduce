"""
Hyperparameter Search and Multi-Method Comparison Framework

This module provides comprehensive hyperparameter search capabilities with
statistical significance testing, reproducible seed management, and 
sophisticated parameter grid specification.
"""

import itertools
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from pathlib import Path
import json
from scipy import stats
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.metrics import make_scorer
import warnings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Individual hyperparameter search result."""
    parameters: Dict[str, Any]
    score: float
    scores: List[float]  # Cross-validation scores
    metadata: Dict[str, Any] = field(default_factory=dict)
    runtime: float = 0.0
    memory_usage: float = 0.0
    error: Optional[str] = None


@dataclass
class MethodComparisonResult:
    """Result of comparing multiple methods."""
    method_scores: Dict[str, List[float]]
    statistical_tests: Dict[str, Dict[str, float]]
    best_method: str
    method_rankings: List[Tuple[str, float]]
    significance_matrix: np.ndarray
    effect_sizes: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParameterGridGenerator:
    """Advanced parameter grid generation with various strategies."""
    
    @staticmethod
    def create_grid(param_spec: Dict[str, Any], strategy: str = "grid") -> List[Dict[str, Any]]:
        """
        Create parameter grid using different strategies.
        
        Parameters
        ----------
        param_spec : dict
            Parameter specifications. Can include:
            - Lists of values: {"param": [1, 2, 3]}
            - Ranges: {"param": {"type": "range", "start": 1, "stop": 10, "num": 5}}
            - Log ranges: {"param": {"type": "logrange", "start": 1e-6, "stop": 1e-2, "num": 5}}
            - Categorical: {"param": {"type": "categorical", "values": ["a", "b", "c"]}}
        strategy : str
            Grid generation strategy: "grid", "random", "sobol", "latin_hypercube"
            
        Returns
        -------
        list
            List of parameter combinations
        """
        if strategy == "grid":
            return ParameterGridGenerator._create_grid_search(param_spec)
        elif strategy == "random":
            return ParameterGridGenerator._create_random_search(param_spec)
        elif strategy == "sobol":
            return ParameterGridGenerator._create_sobol_search(param_spec)
        elif strategy == "latin_hypercube":
            return ParameterGridGenerator._create_lhs_search(param_spec)
        else:
            raise ValueError(f"Unknown grid strategy: {strategy}")
    
    @staticmethod
    def _create_grid_search(param_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create full grid search parameter combinations."""
        processed_params = {}
        
        for param_name, param_config in param_spec.items():
            if isinstance(param_config, list):
                processed_params[param_name] = param_config
            elif isinstance(param_config, dict):
                param_type = param_config.get("type", "values")
                
                if param_type == "range":
                    values = np.linspace(
                        param_config["start"],
                        param_config["stop"],
                        param_config.get("num", 10)
                    ).tolist()
                elif param_type == "logrange":
                    values = np.logspace(
                        np.log10(param_config["start"]),
                        np.log10(param_config["stop"]),
                        param_config.get("num", 10)
                    ).tolist()
                elif param_type == "categorical":
                    values = param_config["values"]
                else:
                    values = param_config.get("values", [param_config])
                
                processed_params[param_name] = values
            else:
                processed_params[param_name] = [param_config]
        
        return list(ParameterGrid(processed_params))
    
    @staticmethod
    def _create_random_search(param_spec: Dict[str, Any], n_samples: int = 100) -> List[Dict[str, Any]]:
        """Create random search parameter combinations."""
        # Implementation would use random sampling from parameter distributions
        # For now, use a subset of grid search
        grid = ParameterGridGenerator._create_grid_search(param_spec)
        if len(grid) <= n_samples:
            return grid
        
        np.random.seed(42)  # Reproducible
        indices = np.random.choice(len(grid), n_samples, replace=False)
        return [grid[i] for i in indices]
    
    @staticmethod
    def _create_sobol_search(param_spec: Dict[str, Any], n_samples: int = 100) -> List[Dict[str, Any]]:
        """Create Sobol sequence parameter combinations for quasi-random search."""
        try:
            from scipy.stats import qmc
            
            # Get parameter bounds and types
            param_names = list(param_spec.keys())
            bounds = []
            param_types = []
            
            for param_config in param_spec.values():
                if isinstance(param_config, list):
                    bounds.append([0, len(param_config) - 1])
                    param_types.append("discrete")
                elif isinstance(param_config, dict):
                    param_type = param_config.get("type", "values")
                    if param_type in ["range", "logrange"]:
                        bounds.append([param_config["start"], param_config["stop"]])
                        param_types.append("continuous" if param_type == "range" else "log")
                    else:
                        values = param_config.get("values", [param_config])
                        bounds.append([0, len(values) - 1])
                        param_types.append("discrete")
                else:
                    bounds.append([0, 0])  # Single value
                    param_types.append("discrete")
            
            # Generate Sobol sequence
            sampler = qmc.Sobol(d=len(param_names), seed=42)
            samples = sampler.random(n_samples)
            
            # Convert samples to parameter values
            param_combinations = []
            for sample in samples:
                param_dict = {}
                for i, param_name in enumerate(param_names):
                    param_config = param_spec[param_name]
                    bound = bounds[i]
                    param_type = param_types[i]
                    
                    if param_type == "discrete":
                        if isinstance(param_config, list):
                            idx = int(sample[i] * len(param_config))
                            idx = min(idx, len(param_config) - 1)
                            param_dict[param_name] = param_config[idx]
                        else:
                            values = param_config.get("values", [param_config])
                            idx = int(sample[i] * len(values))
                            idx = min(idx, len(values) - 1)
                            param_dict[param_name] = values[idx]
                    elif param_type == "continuous":
                        value = bound[0] + sample[i] * (bound[1] - bound[0])
                        param_dict[param_name] = value
                    elif param_type == "log":
                        log_value = np.log10(bound[0]) + sample[i] * (np.log10(bound[1]) - np.log10(bound[0]))
                        param_dict[param_name] = 10 ** log_value
                
                param_combinations.append(param_dict)
            
            return param_combinations
            
        except ImportError:
            logger.warning("scipy.stats.qmc not available, falling back to random search")
            return ParameterGridGenerator._create_random_search(param_spec, n_samples)
    
    @staticmethod
    def _create_lhs_search(param_spec: Dict[str, Any], n_samples: int = 100) -> List[Dict[str, Any]]:
        """Create Latin Hypercube Sampling parameter combinations."""
        try:
            from scipy.stats import qmc
            
            # Similar to Sobol but using LHS
            param_names = list(param_spec.keys())
            sampler = qmc.LatinHypercube(d=len(param_names), seed=42)
            # Implementation would be similar to Sobol but with LHS sampling
            return ParameterGridGenerator._create_random_search(param_spec, n_samples)
            
        except ImportError:
            logger.warning("scipy.stats.qmc not available, falling back to random search")
            return ParameterGridGenerator._create_random_search(param_spec, n_samples)


class CrossValidationStrategy:
    """Cross-validation strategies for hyperparameter search."""
    
    @staticmethod
    def create_folds(X: np.ndarray, strategy: str = "kfold", n_splits: int = 5, 
                    test_size: float = 0.2, random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cross-validation folds.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        strategy : str
            CV strategy: "kfold", "stratified", "timeseries", "bootstrap"
        n_splits : int
            Number of splits
        test_size : float
            Test size for bootstrap
        random_state : int
            Random seed
            
        Returns
        -------
        list
            List of (train_indices, test_indices) tuples
        """
        from sklearn.model_selection import (
            KFold, StratifiedKFold, TimeSeriesSplit, ShuffleSplit
        )
        
        n_samples = X.shape[0]
        
        if strategy == "kfold":
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            return list(cv.split(X))
        elif strategy == "stratified":
            # For unsupervised learning, create pseudo-labels based on data distribution
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(n_splits, n_samples // 10), random_state=random_state)
            pseudo_labels = kmeans.fit_predict(X)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            return list(cv.split(X, pseudo_labels))
        elif strategy == "timeseries":
            cv = TimeSeriesSplit(n_splits=n_splits)
            return list(cv.split(X))
        elif strategy == "bootstrap":
            cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
            return list(cv.split(X))
        else:
            raise ValueError(f"Unknown CV strategy: {strategy}")


class HyperparameterSearchEngine:
    """
    Comprehensive hyperparameter search engine with multiple optimization strategies.
    """
    
    def __init__(self, 
                 scoring_function: Callable,
                 cv_strategy: str = "kfold",
                 cv_folds: int = 5,
                 n_jobs: int = -1,
                 random_state: int = 42,
                 early_stopping: bool = False,
                 early_stopping_patience: int = 10,
                 early_stopping_min_delta: float = 0.001):
        """
        Initialize hyperparameter search engine.
        
        Parameters
        ----------
        scoring_function : callable
            Function to evaluate parameter combinations
        cv_strategy : str
            Cross-validation strategy
        cv_folds : int
            Number of CV folds
        n_jobs : int
            Number of parallel jobs
        random_state : int
            Random seed
        early_stopping : bool
            Whether to use early stopping
        early_stopping_patience : int
            Early stopping patience
        early_stopping_min_delta : float
            Minimum improvement for early stopping
        """
        self.scoring_function = scoring_function
        self.cv_strategy = cv_strategy
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        
        # Results tracking
        self.search_results: List[SearchResult] = []
        self.best_result: Optional[SearchResult] = None
        
    def search(self, X: np.ndarray, param_grid: List[Dict[str, Any]], 
              y: Optional[np.ndarray] = None) -> List[SearchResult]:
        """
        Perform hyperparameter search.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        param_grid : list
            Parameter combinations to search
        y : np.ndarray, optional
            Target labels (for supervised methods)
            
        Returns
        -------
        list
            Search results sorted by score
        """
        logger.info(f"Starting hyperparameter search with {len(param_grid)} combinations")
        
        # Create CV folds
        cv_folds = CrossValidationStrategy.create_folds(
            X, self.cv_strategy, self.cv_folds, random_state=self.random_state
        )
        
        # Reset results
        self.search_results = []
        self.best_result = None
        best_score = -np.inf
        patience_counter = 0
        
        # Evaluate parameter combinations
        if self.n_jobs == 1:
            # Sequential execution
            for i, params in enumerate(param_grid):
                result = self._evaluate_parameters(X, params, cv_folds, y)
                self.search_results.append(result)
                
                # Update best result
                if result.score > best_score:
                    best_score = result.score
                    self.best_result = result
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping check
                if (self.early_stopping and 
                    patience_counter >= self.early_stopping_patience and 
                    i >= self.early_stopping_patience):
                    logger.info(f"Early stopping at iteration {i}")
                    break
                
                if i % 10 == 0:
                    logger.info(f"Completed {i}/{len(param_grid)} combinations, "
                              f"best score: {best_score:.4f}")
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.n_jobs if self.n_jobs > 0 else None) as executor:
                # Submit all jobs
                future_to_params = {
                    executor.submit(self._evaluate_parameters, X, params, cv_folds, y): params
                    for params in param_grid
                }
                
                # Collect results
                for i, future in enumerate(as_completed(future_to_params)):
                    result = future.result()
                    self.search_results.append(result)
                    
                    if result.score > best_score:
                        best_score = result.score
                        self.best_result = result
                    
                    if i % 10 == 0:
                        logger.info(f"Completed {i}/{len(param_grid)} combinations, "
                                  f"best score: {best_score:.4f}")
        
        # Sort results by score
        self.search_results.sort(key=lambda x: x.score, reverse=True)
        
        if self.best_result:
            logger.info(f"Best parameters: {self.best_result.parameters}")
            logger.info(f"Best score: {self.best_result.score:.4f} ± "
                       f"{np.std(self.best_result.scores):.4f}")
        
        return self.search_results
    
    def _evaluate_parameters(self, X: np.ndarray, params: Dict[str, Any], 
                           cv_folds: List[Tuple[np.ndarray, np.ndarray]], 
                           y: Optional[np.ndarray] = None) -> SearchResult:
        """Evaluate a single parameter combination."""
        start_time = time.time()
        
        try:
            scores = []
            
            # Perform cross-validation
            for train_idx, test_idx in cv_folds:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train = y[train_idx] if y is not None else None
                y_test = y[test_idx] if y is not None else None
                
                # Evaluate with current parameters
                score = self.scoring_function(X_train, X_test, params, y_train, y_test)
                scores.append(score)
            
            runtime = time.time() - start_time
            mean_score = np.mean(scores)
            
            return SearchResult(
                parameters=params,
                score=mean_score,
                scores=scores,
                runtime=runtime,
                metadata={
                    "score_std": np.std(scores),
                    "score_sem": stats.sem(scores) if len(scores) > 1 else 0.0,
                    "n_folds": len(scores)
                }
            )
            
        except Exception as e:
            logger.warning(f"Parameter evaluation failed: {params}, error: {str(e)}")
            return SearchResult(
                parameters=params,
                score=-np.inf,
                scores=[],
                runtime=time.time() - start_time,
                error=str(e)
            )
    
    def get_top_k_results(self, k: int = 10) -> List[SearchResult]:
        """Get top k results."""
        return self.search_results[:k]
    
    def analyze_parameter_importance(self) -> Dict[str, float]:
        """Analyze parameter importance based on score variance."""
        if not self.search_results:
            return {}
        
        # Get all parameter names
        param_names = set()
        for result in self.search_results:
            param_names.update(result.parameters.keys())
        
        importance_scores = {}
        
        for param_name in param_names:
            # Group results by parameter value
            param_groups = {}
            for result in self.search_results:
                param_value = result.parameters.get(param_name)
                if param_value is not None:
                    if param_value not in param_groups:
                        param_groups[param_value] = []
                    param_groups[param_value].append(result.score)
            
            # Calculate importance as variance explained
            if len(param_groups) > 1:
                all_scores = [score for scores in param_groups.values() for score in scores]
                total_variance = np.var(all_scores)
                
                if total_variance > 0:
                    # Calculate within-group variance
                    within_group_variance = 0
                    total_count = 0
                    for scores in param_groups.values():
                        if len(scores) > 1:
                            within_group_variance += np.var(scores) * len(scores)
                            total_count += len(scores)
                    
                    if total_count > 0:
                        within_group_variance /= total_count
                        # Importance = variance explained
                        importance = (total_variance - within_group_variance) / total_variance
                        importance_scores[param_name] = max(0, importance)
                    else:
                        importance_scores[param_name] = 0.0
                else:
                    importance_scores[param_name] = 0.0
            else:
                importance_scores[param_name] = 0.0
        
        return importance_scores


class MultiMethodComparator:
    """
    Compare multiple dimensionality reduction methods with statistical significance testing.
    """
    
    def __init__(self, methods: Dict[str, Callable], 
                 evaluation_metrics: List[Callable],
                 cv_folds: int = 5,
                 n_runs: int = 10,
                 random_state: int = 42):
        """
        Initialize multi-method comparator.
        
        Parameters
        ----------
        methods : dict
            Dictionary of method_name -> method_function
        evaluation_metrics : list
            List of evaluation functions
        cv_folds : int
            Number of CV folds
        n_runs : int
            Number of independent runs
        random_state : int
            Random seed base
        """
        self.methods = methods
        self.evaluation_metrics = evaluation_metrics
        self.cv_folds = cv_folds
        self.n_runs = n_runs
        self.random_state = random_state
        
    def compare_methods(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> MethodComparisonResult:
        """
        Compare all methods with statistical significance testing.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        y : np.ndarray, optional
            Target labels
            
        Returns
        -------
        MethodComparisonResult
            Comprehensive comparison results
        """
        logger.info(f"Comparing {len(self.methods)} methods with {len(self.evaluation_metrics)} metrics")
        
        # Collect scores for each method
        method_scores = {method_name: [] for method_name in self.methods.keys()}
        
        # Run multiple independent experiments
        for run in range(self.n_runs):
            run_seed = self.random_state + run
            
            # Create CV folds for this run
            cv_folds = CrossValidationStrategy.create_folds(
                X, "kfold", self.cv_folds, random_state=run_seed
            )
            
            for method_name, method_func in self.methods.items():
                run_scores = []
                
                for train_idx, test_idx in cv_folds:
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train = y[train_idx] if y is not None else None
                    y_test = y[test_idx] if y is not None else None
                    
                    try:
                        # Apply method
                        Y_train = method_func(X_train, y=y_train)
                        Y_test = method_func(X_test, y=y_test)  # This might need refitting
                        
                        # Evaluate with all metrics
                        fold_scores = []
                        for metric_func in self.evaluation_metrics:
                            score = metric_func(X_test, Y_test)
                            fold_scores.append(score)
                        
                        # Use composite score (average of all metrics)
                        composite_score = np.mean(fold_scores)
                        run_scores.append(composite_score)
                        
                    except Exception as e:
                        logger.warning(f"Method {method_name} failed on fold: {str(e)}")
                        run_scores.append(-np.inf)
                
                # Average score for this run
                if run_scores and all(np.isfinite(run_scores)):
                    method_scores[method_name].append(np.mean(run_scores))
                else:
                    method_scores[method_name].append(-np.inf)
            
            if run % 5 == 0:
                logger.info(f"Completed run {run}/{self.n_runs}")
        
        # Perform statistical tests
        statistical_tests = self._perform_statistical_tests(method_scores)
        
        # Calculate rankings
        method_rankings = self._calculate_rankings(method_scores)
        
        # Create significance matrix
        significance_matrix = self._create_significance_matrix(method_scores)
        
        # Calculate effect sizes
        effect_sizes = self._calculate_effect_sizes(method_scores)
        
        # Find best method
        best_method = method_rankings[0][0]  # First in ranking
        
        return MethodComparisonResult(
            method_scores=method_scores,
            statistical_tests=statistical_tests,
            best_method=best_method,
            method_rankings=method_rankings,
            significance_matrix=significance_matrix,
            effect_sizes=effect_sizes,
            metadata={
                "n_runs": self.n_runs,
                "cv_folds": self.cv_folds,
                "n_methods": len(self.methods),
                "n_metrics": len(self.evaluation_metrics)
            }
        )
    
    def _perform_statistical_tests(self, method_scores: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Perform pairwise statistical tests."""
        tests = {}
        method_names = list(method_scores.keys())
        
        for test_name in ["paired_t_test", "wilcoxon", "mann_whitney"]:
            tests[test_name] = {}
            
            for i, method1 in enumerate(method_names):
                for j, method2 in enumerate(method_names[i+1:], i+1):
                    scores1 = method_scores[method1]
                    scores2 = method_scores[method2]
                    
                    # Filter out infinite scores
                    valid_pairs = [(s1, s2) for s1, s2 in zip(scores1, scores2) 
                                  if np.isfinite(s1) and np.isfinite(s2)]
                    
                    if len(valid_pairs) < 3:  # Need at least 3 pairs for meaningful test
                        p_value = 1.0
                    else:
                        scores1_valid = [p[0] for p in valid_pairs]
                        scores2_valid = [p[1] for p in valid_pairs]
                        
                        try:
                            if test_name == "paired_t_test":
                                _, p_value = stats.ttest_rel(scores1_valid, scores2_valid)
                            elif test_name == "wilcoxon":
                                _, p_value = stats.wilcoxon(scores1_valid, scores2_valid)
                            elif test_name == "mann_whitney":
                                _, p_value = stats.mannwhitneyu(scores1_valid, scores2_valid, 
                                                               alternative='two-sided')
                        except Exception:
                            p_value = 1.0
                    
                    pair_key = f"{method1}_vs_{method2}"
                    tests[test_name][pair_key] = p_value
        
        return tests
    
    def _calculate_rankings(self, method_scores: Dict[str, List[float]]) -> List[Tuple[str, float]]:
        """Calculate method rankings based on mean scores."""
        rankings = []
        
        for method_name, scores in method_scores.items():
            valid_scores = [s for s in scores if np.isfinite(s)]
            if valid_scores:
                mean_score = np.mean(valid_scores)
                rankings.append((method_name, mean_score))
            else:
                rankings.append((method_name, -np.inf))
        
        # Sort by score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def _create_significance_matrix(self, method_scores: Dict[str, List[float]]) -> np.ndarray:
        """Create pairwise significance matrix (p-values)."""
        method_names = list(method_scores.keys())
        n_methods = len(method_names)
        significance_matrix = np.ones((n_methods, n_methods))
        
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names):
                if i != j:
                    scores1 = [s for s in method_scores[method1] if np.isfinite(s)]
                    scores2 = [s for s in method_scores[method2] if np.isfinite(s)]
                    
                    if len(scores1) >= 3 and len(scores2) >= 3:
                        try:
                            _, p_value = stats.mannwhitneyu(scores1, scores2, 
                                                           alternative='two-sided')
                            significance_matrix[i, j] = p_value
                        except Exception:
                            significance_matrix[i, j] = 1.0
        
        return significance_matrix
    
    def _calculate_effect_sizes(self, method_scores: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate Cohen's d effect sizes relative to the worst method."""
        # Find the worst performing method
        method_means = {}
        for method_name, scores in method_scores.items():
            valid_scores = [s for s in scores if np.isfinite(s)]
            method_means[method_name] = np.mean(valid_scores) if valid_scores else -np.inf
        
        worst_method = min(method_means, key=method_means.get)
        worst_scores = [s for s in method_scores[worst_method] if np.isfinite(s)]
        
        effect_sizes = {}
        for method_name, scores in method_scores.items():
            if method_name == worst_method:
                effect_sizes[method_name] = 0.0
                continue
            
            valid_scores = [s for s in scores if np.isfinite(s)]
            
            if len(valid_scores) >= 3 and len(worst_scores) >= 3:
                # Cohen's d
                mean_diff = np.mean(valid_scores) - np.mean(worst_scores)
                pooled_std = np.sqrt((np.var(valid_scores) + np.var(worst_scores)) / 2)
                
                if pooled_std > 0:
                    cohens_d = mean_diff / pooled_std
                    effect_sizes[method_name] = cohens_d
                else:
                    effect_sizes[method_name] = 0.0
            else:
                effect_sizes[method_name] = 0.0
        
        return effect_sizes
    
    def generate_comparison_report(self, result: MethodComparisonResult) -> str:
        """Generate a comprehensive comparison report."""
        report = []
        report.append("=== Multi-Method Comparison Report ===\n")
        
        # Method rankings
        report.append("Method Rankings (by mean score):")
        for i, (method, score) in enumerate(result.method_rankings, 1):
            valid_scores = [s for s in result.method_scores[method] if np.isfinite(s)]
            std_score = np.std(valid_scores) if valid_scores else 0.0
            report.append(f"{i}. {method}: {score:.4f} ± {std_score:.4f}")
        report.append("")
        
        # Best method
        report.append(f"Best Method: {result.best_method}")
        report.append("")
        
        # Effect sizes
        report.append("Effect Sizes (Cohen's d relative to worst method):")
        for method, effect_size in result.effect_sizes.items():
            if effect_size >= 0.8:
                magnitude = "Large"
            elif effect_size >= 0.5:
                magnitude = "Medium" 
            elif effect_size >= 0.2:
                magnitude = "Small"
            else:
                magnitude = "Negligible"
            report.append(f"{method}: {effect_size:.3f} ({magnitude})")
        report.append("")
        
        # Statistical significance summary
        report.append("Statistical Significance (p < 0.05):")
        paired_t_results = result.statistical_tests.get("paired_t_test", {})
        significant_pairs = [pair for pair, p_val in paired_t_results.items() if p_val < 0.05]
        
        if significant_pairs:
            for pair in significant_pairs:
                report.append(f"  {pair}: p = {paired_t_results[pair]:.4f}")
        else:
            report.append("  No statistically significant differences found")
        
        report.append("")
        report.append(f"Total experiments: {result.metadata['n_runs']} runs × {result.metadata['cv_folds']} folds")
        
        return "\n".join(report)