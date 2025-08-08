"""
Pipeline Orchestrator for Staged Optimization

This module implements the core orchestration framework for running staged
optimization pipelines with dependency management, data flow tracking, and
comprehensive experiment management.
"""

import time
import logging
import traceback
import psutil
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import pickle
import json
import numpy as np
from datetime import datetime

# Import configuration system
try:
    from .experiment_config import ExperimentConfig, StageConfig
except ImportError:
    from experiment_config import ExperimentConfig, StageConfig

# Import pipeline components
try:
    from .preprocessing import adaptive_preprocessing_pipeline
    from .convex_optimized import project_onto_convex_hull_enhanced
    from .spherical_embeddings import adaptive_spherical_embedding
    from .hyperbolic import run_poincare_optimized
    from .calibration import combined_calibration
    from .evaluation import compute_distortion, rank_correlation, nearest_neighbor_overlap
    from .dimensionality_reduction import generate_mixture_gaussians, run_pca_simple, run_jll_simple, run_umap_simple
except ImportError as e:
    logging.warning(f"Some pipeline components not available: {e}")

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Tracks the state of pipeline execution."""
    stage_results: Dict[str, Any] = field(default_factory=dict)
    stage_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stage_timing: Dict[str, float] = field(default_factory=dict)
    stage_memory_usage: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    completed_stages: List[str] = field(default_factory=list)
    failed_stages: List[str] = field(default_factory=list)
    current_stage: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    
    def get_stage_output(self, stage_name: str, key: str = None) -> Any:
        """Get output from a completed stage."""
        if stage_name not in self.stage_results:
            raise ValueError(f"Stage {stage_name} has not completed successfully")
        
        if key is None:
            return self.stage_results[stage_name]
        else:
            return self.stage_results[stage_name].get(key)
    
    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if a stage has completed successfully."""
        return stage_name in self.completed_stages
    
    def can_run_stage(self, stage: StageConfig) -> bool:
        """Check if all dependencies for a stage are satisfied."""
        return all(self.is_stage_completed(dep) for dep in stage.dependencies)


class ResourceMonitor:
    """Monitor computational resources during pipeline execution."""
    
    def __init__(self, memory_limit_gb: Optional[float] = None):
        self.memory_limit_gb = memory_limit_gb
        self.monitoring = False
        self.peak_memory_gb = 0.0
        self.memory_history = []
        
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        self.monitoring = True
        self.peak_memory_gb = 0.0
        self.memory_history = []
        
        def monitor():
            while self.monitoring:
                try:
                    process = psutil.Process()
                    memory_gb = process.memory_info().rss / (1024**3)
                    self.peak_memory_gb = max(self.peak_memory_gb, memory_gb)
                    self.memory_history.append((time.time(), memory_gb))
                    
                    # Check memory limit
                    if self.memory_limit_gb and memory_gb > self.memory_limit_gb:
                        logger.warning(f"Memory usage {memory_gb:.2f}GB exceeds limit {self.memory_limit_gb}GB")
                    
                    time.sleep(1)  # Check every second
                except Exception:
                    pass
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return resource usage summary."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
        
        return {
            "peak_memory_gb": self.peak_memory_gb,
            "memory_history": self.memory_history[-100:],  # Keep last 100 measurements
            "final_memory_gb": self.memory_history[-1][1] if self.memory_history else 0.0
        }


class PipelineOrchestrator:
    """
    Orchestrates the execution of staged optimization pipelines.
    
    Handles stage dependencies, data flow, resource management, and error handling.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.state = PipelineState()
        self.resource_monitor = ResourceMonitor(config.memory_limit_gb)
        self.stage_executors = self._initialize_stage_executors()
        
        # Set up logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up experiment-specific logging."""
        logger = logging.getLogger(f"orchestrator.{self.config.name}")
        
        if self.config.verbose_logging:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
            
        if self.config.debug_mode:
            logger.setLevel(logging.DEBUG)
            
        return logger
    
    def _initialize_stage_executors(self) -> Dict[str, Callable]:
        """Initialize stage executor functions."""
        return {
            "preprocessing": self._execute_preprocessing_stage,
            "convex_optimization": self._execute_convex_optimization_stage,
            "geometric_embeddings": self._execute_geometric_embeddings_stage,
            "calibration": self._execute_calibration_stage
        }
    
    def run_pipeline(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run the complete staged optimization pipeline.
        
        Parameters
        ----------
        X : np.ndarray
            Input high-dimensional data
        y : np.ndarray, optional
            Labels for supervised methods
            
        Returns
        -------
        dict
            Complete pipeline results including all stage outputs and metadata
        """
        self.logger.info(f"Starting pipeline execution: {self.config.name}")
        self.logger.info(f"Input data shape: {X.shape}")
        
        # Initialize pipeline state with input data
        self.state.stage_results["input"] = {"X": X, "y": y}
        self.state.completed_stages.append("input")
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        try:
            # Get enabled stages in dependency order
            stages = self._get_execution_order()
            self.logger.info(f"Executing {len(stages)} stages: {[s.name for s in stages]}")
            
            # Execute stages sequentially (respecting dependencies)
            for stage in stages:
                if self._should_skip_stage(stage):
                    self.logger.info(f"Skipping disabled stage: {stage.name}")
                    continue
                
                try:
                    self._execute_stage(stage)
                except Exception as e:
                    self.logger.error(f"Stage {stage.name} failed: {str(e)}")
                    self.state.failed_stages.append(stage.name)
                    self.state.errors.append(f"Stage {stage.name}: {str(e)}")
                    
                    if not self._can_continue_after_failure(stage):
                        self.logger.error("Pipeline execution halted due to critical stage failure")
                        break
            
            # Generate final results
            results = self._compile_results()
            
            self.logger.info(f"Pipeline completed in {time.time() - self.state.start_time:.2f} seconds")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            self.state.errors.append(f"Pipeline: {str(e)}")
            return self._compile_results()
            
        finally:
            # Stop resource monitoring
            resource_stats = self.resource_monitor.stop_monitoring()
            self.state.stage_metadata["resource_usage"] = resource_stats
    
    def _get_execution_order(self) -> List[StageConfig]:
        """Get stages in topological order respecting dependencies."""
        stages = self.config.get_stages()
        
        # Simple topological sort
        completed = set()
        ordered_stages = []
        
        while len(ordered_stages) < len(stages):
            added_any = False
            
            for stage in stages:
                if stage.name not in completed and all(dep in completed for dep in stage.dependencies):
                    ordered_stages.append(stage)
                    completed.add(stage.name)
                    added_any = True
            
            if not added_any:
                remaining = [s.name for s in stages if s.name not in completed]
                raise ValueError(f"Circular dependency or missing dependency in stages: {remaining}")
        
        return ordered_stages
    
    def _should_skip_stage(self, stage: StageConfig) -> bool:
        """Check if stage should be skipped."""
        return not stage.enabled or not self.state.can_run_stage(stage)
    
    def _can_continue_after_failure(self, failed_stage: StageConfig) -> bool:
        """Determine if pipeline can continue after stage failure."""
        # For now, treat all stage failures as critical
        return False
    
    @contextmanager
    def _stage_execution_context(self, stage: StageConfig):
        """Context manager for stage execution with timing and error handling."""
        self.state.current_stage = stage.name
        start_time = time.time()
        start_memory = self.resource_monitor.peak_memory_gb
        
        self.logger.info(f"Starting stage: {stage.name}")
        
        try:
            yield
            
            # Mark stage as completed
            execution_time = time.time() - start_time
            memory_used = self.resource_monitor.peak_memory_gb - start_memory
            
            self.state.completed_stages.append(stage.name)
            self.state.stage_timing[stage.name] = execution_time
            self.state.stage_memory_usage[stage.name] = memory_used
            
            self.logger.info(f"Stage {stage.name} completed in {execution_time:.2f}s, "
                           f"memory: {memory_used:.2f}GB")
                           
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Stage {stage.name} failed after {execution_time:.2f}s: {str(e)}")
            
            if self.config.debug_mode:
                self.logger.debug(f"Stack trace:\n{traceback.format_exc()}")
            
            raise
        finally:
            self.state.current_stage = None
    
    def _execute_stage(self, stage: StageConfig):
        """Execute a single pipeline stage."""
        with self._stage_execution_context(stage):
            # Get stage executor
            executor = self.stage_executors.get(stage.method)
            if not executor:
                raise ValueError(f"No executor found for stage method: {stage.method}")
            
            # Execute stage
            results = executor(stage)
            
            # Store results
            self.state.stage_results[stage.name] = results
            
            # Store metadata if provided
            if isinstance(results, dict) and 'metadata' in results:
                self.state.stage_metadata[stage.name] = results['metadata']
    
    # Stage-specific execution methods
    
    def _execute_preprocessing_stage(self, stage: StageConfig) -> Dict[str, Any]:
        """Execute preprocessing stage."""
        X = self.state.get_stage_output("input", "X")
        params = stage.parameters
        
        if params.get("adaptive_pipeline", True):
            # Use adaptive preprocessing pipeline
            X_processed, metadata = adaptive_preprocessing_pipeline(
                X,
                target_method=params.get("target_method", "jll"),
                quality_threshold=params.get("quality_threshold", 0.9),
                max_components=params.get("max_components", 300)
            )
        else:
            # Manual preprocessing steps
            X_processed = X.copy()
            metadata = {"method": "manual"}
            
            # Apply individual preprocessing steps based on configuration
            if params.get("standardization", "zscore") != "none":
                from .preprocessing import standardize_features
                X_processed, std_meta = standardize_features(
                    X_processed, method=params["standardization"]
                )
                metadata["standardization"] = std_meta
            
            if params.get("denoising_method", "none") != "none":
                if params["denoising_method"] == "pca":
                    from .preprocessing import pca_denoise
                    X_processed, denoise_meta = pca_denoise(
                        X_processed,
                        n_components=params.get("pca_components"),
                        adaptive_components=True
                    )
                    metadata["denoising"] = denoise_meta
                elif params["denoising_method"] == "jl":
                    from .preprocessing import jl_denoise
                    X_processed, denoise_meta = jl_denoise(
                        X_processed,
                        adaptive_components=True
                    )
                    metadata["denoising"] = denoise_meta
            
            if params.get("l2_normalize", False):
                from .preprocessing import l2_normalize_rows
                X_processed, norm_meta = l2_normalize_rows(X_processed)
                metadata["normalization"] = norm_meta
        
        return {
            "preprocessed_data": X_processed,
            "preprocessing_metadata": metadata,
            "original_shape": X.shape,
            "processed_shape": X_processed.shape
        }
    
    def _execute_convex_optimization_stage(self, stage: StageConfig) -> Dict[str, Any]:
        """Execute convex optimization stage."""
        X_processed = self.state.get_stage_output("preprocessing", "preprocessed_data")
        params = stage.parameters
        
        # Grid search over hyperparameters
        best_result = None
        best_score = -np.inf
        all_results = []
        
        k_candidates_grid = params.get("k_candidates_grid", [64])
        lambda_grid = params.get("lambda_grid", [1e-6])
        tolerance_grid = params.get("tolerance_grid", ["balanced"])
        objective_types = params.get("objective_types", ["quadratic"])
        
        total_combinations = len(k_candidates_grid) * len(lambda_grid) * len(tolerance_grid) * len(objective_types)
        self.logger.info(f"Testing {total_combinations} hyperparameter combinations")
        
        combination_count = 0
        for k_candidates in k_candidates_grid:
            for ridge_lambda in lambda_grid:
                for tolerance_mode in tolerance_grid:
                    for objective_type in objective_types:
                        combination_count += 1
                        
                        try:
                            # Run convex optimization
                            Y_proj, alphas, V = project_onto_convex_hull_enhanced(
                                X_processed,
                                ridge_lambda=ridge_lambda,
                                k_candidates=k_candidates,
                                solver_mode=tolerance_mode,
                                objective_type=objective_type,
                                use_float64=params.get("use_float64", False),
                                batch_size=params.get("batch_size", 1024),
                                candidate_normalization=params.get("candidate_normalization", "none")
                            )
                            
                            # Evaluate result
                            score = self._evaluate_convex_result(X_processed, Y_proj)
                            
                            result = {
                                "Y_proj": Y_proj,
                                "alphas": alphas,
                                "vertices": V,
                                "score": score,
                                "hyperparameters": {
                                    "k_candidates": k_candidates,
                                    "ridge_lambda": ridge_lambda,
                                    "tolerance_mode": tolerance_mode,
                                    "objective_type": objective_type
                                }
                            }
                            all_results.append(result)
                            
                            if score > best_score:
                                best_score = score
                                best_result = result
                            
                            self.logger.debug(f"Combination {combination_count}/{total_combinations}: "
                                            f"score={score:.4f}")
                            
                        except Exception as e:
                            self.logger.warning(f"Convex optimization failed for combination "
                                              f"{combination_count}: {str(e)}")
                            continue
        
        if best_result is None:
            raise RuntimeError("All convex optimization combinations failed")
        
        self.logger.info(f"Best convex optimization score: {best_score:.4f}")
        self.logger.info(f"Best hyperparameters: {best_result['hyperparameters']}")
        
        return {
            "convex_optimized_data": best_result["Y_proj"],
            "alphas": best_result["alphas"],
            "vertices": best_result["vertices"],
            "best_hyperparameters": best_result["hyperparameters"],
            "best_score": best_score,
            "all_results": all_results,
            "metadata": {
                "n_combinations_tested": len(all_results),
                "n_combinations_total": total_combinations
            }
        }
    
    def _execute_geometric_embeddings_stage(self, stage: StageConfig) -> Dict[str, Any]:
        """Execute geometric embeddings stage."""
        X_convex = self.state.get_stage_output("convex_optimization", "convex_optimized_data")
        params = stage.parameters
        
        results = {}
        
        # Spherical embeddings
        if "spherical_config" in params:
            spherical_config = params["spherical_config"]
            spherical_results = []
            
            for method in spherical_config.get("methods", ["riemannian"]):
                for loss_type in spherical_config.get("loss_types", ["mds_geodesic"]):
                    for lr in spherical_config.get("learning_rates", [0.01]):
                        try:
                            Y_spherical, info = adaptive_spherical_embedding(
                                X_convex,
                                k=X_convex.shape[1],  # Keep same dimensionality
                                method=method,
                                loss_type=loss_type,
                                max_iter=spherical_config.get("max_iter", 500),
                                learning_rate=lr,
                                adaptive_radius=spherical_config.get("adaptive_radius", True),
                                hemisphere_constraint=spherical_config.get("hemisphere_constraint", True),
                                seed=self.config.random_seed
                            )
                            
                            spherical_results.append({
                                "embedding": Y_spherical,
                                "info": info,
                                "method": method,
                                "loss_type": loss_type,
                                "learning_rate": lr
                            })
                            
                        except Exception as e:
                            self.logger.warning(f"Spherical embedding failed: {str(e)}")
                            continue
            
            if spherical_results:
                # Select best spherical embedding based on final loss
                best_spherical = min(spherical_results, 
                                   key=lambda x: x["info"].get("final_loss", np.inf))
                results["spherical_embedding"] = best_spherical["embedding"]
                results["spherical_metadata"] = best_spherical
        
        # Poincaré embeddings
        if "poincare_config" in params:
            poincare_config = params["poincare_config"]
            poincare_results = []
            
            for c in poincare_config.get("curvatures", [1.0]):
                for lr in poincare_config.get("learning_rates", [0.01]):
                    for optimizer in poincare_config.get("optimizers", ["radam"]):
                        for loss_fn in poincare_config.get("loss_functions", ["stress"]):
                            for init_method in poincare_config.get("init_methods", ["pca"]):
                                try:
                                    Y_poincare, runtime = run_poincare_optimized(
                                        X_convex,
                                        k=X_convex.shape[1],
                                        c=c,
                                        lr=lr,
                                        n_epochs=poincare_config.get("n_epochs", 100),
                                        optimizer=optimizer,
                                        loss_fn=loss_fn,
                                        init_method=init_method,
                                        regularization=poincare_config.get("regularization", 0.01),
                                        seed=self.config.random_seed
                                    )
                                    
                                    poincare_results.append({
                                        "embedding": Y_poincare,
                                        "runtime": runtime,
                                        "curvature": c,
                                        "learning_rate": lr,
                                        "optimizer": optimizer,
                                        "loss_function": loss_fn,
                                        "init_method": init_method
                                    })
                                    
                                except Exception as e:
                                    self.logger.warning(f"Poincaré embedding failed: {str(e)}")
                                    continue
            
            if poincare_results:
                # Select best Poincaré embedding (could use more sophisticated selection)
                best_poincare = min(poincare_results, key=lambda x: x["runtime"])  # Arbitrary selection
                results["poincare_embedding"] = best_poincare["embedding"]
                results["poincare_metadata"] = best_poincare
        
        if not results:
            raise RuntimeError("All geometric embedding methods failed")
        
        return results
    
    def _execute_calibration_stage(self, stage: StageConfig) -> Dict[str, Any]:
        """Execute calibration stage."""
        X_original = self.state.get_stage_output("preprocessing", "preprocessed_data")
        params = stage.parameters
        
        # Get embeddings to calibrate
        embeddings_to_calibrate = {}
        
        # Add convex optimized embedding
        if self.state.is_stage_completed("convex_optimization"):
            embeddings_to_calibrate["convex"] = self.state.get_stage_output(
                "convex_optimization", "convex_optimized_data"
            )
        
        # Add geometric embeddings
        if self.state.is_stage_completed("geometric_embeddings"):
            geometric_results = self.state.get_stage_output("geometric_embeddings")
            if "spherical_embedding" in geometric_results:
                embeddings_to_calibrate["spherical"] = geometric_results["spherical_embedding"]
            if "poincare_embedding" in geometric_results:
                embeddings_to_calibrate["poincare"] = geometric_results["poincare_embedding"]
        
        calibrated_embeddings = {}
        calibration_metadata = {}
        
        for embedding_name, Y in embeddings_to_calibrate.items():
            try:
                # Apply combined calibration
                Y_calibrated, calib_info = combined_calibration(
                    X_original,
                    Y,
                    methods=params.get("methods", ["procrustes", "isotonic", "local"]),
                    **params.get("procrustes_params", {}),
                    **params.get("isotonic_params", {}),
                    **params.get("local_correction_params", {})
                )
                
                calibrated_embeddings[f"{embedding_name}_calibrated"] = Y_calibrated
                calibration_metadata[embedding_name] = calib_info
                
                self.logger.info(f"Calibrated {embedding_name} embedding: "
                               f"correlation improvement = {calib_info.get('total_improvement', 0):.4f}")
                
            except Exception as e:
                self.logger.warning(f"Calibration failed for {embedding_name}: {str(e)}")
                continue
        
        return {
            "calibrated_embeddings": calibrated_embeddings,
            "calibration_metadata": calibration_metadata
        }
    
    def _evaluate_convex_result(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Evaluate convex optimization result for hyperparameter selection."""
        try:
            # Use rank correlation as primary metric
            correlation = rank_correlation(X, Y, sample_size=min(1000, X.shape[0]))
            return correlation
        except Exception:
            return 0.0
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile final pipeline results."""
        total_time = time.time() - self.state.start_time
        
        # Collect all embeddings
        final_embeddings = {}
        
        # Add intermediate embeddings
        if "preprocessing" in self.state.stage_results:
            final_embeddings["preprocessed"] = self.state.get_stage_output("preprocessing", "preprocessed_data")
        
        if "convex_optimization" in self.state.stage_results:
            final_embeddings["convex_optimized"] = self.state.get_stage_output("convex_optimization", "convex_optimized_data")
        
        # Add geometric embeddings
        if "geometric_embeddings" in self.state.stage_results:
            geometric_results = self.state.get_stage_output("geometric_embeddings")
            for key in ["spherical_embedding", "poincare_embedding"]:
                if key in geometric_results:
                    final_embeddings[key] = geometric_results[key]
        
        # Add calibrated embeddings
        if "calibration" in self.state.stage_results:
            calibrated_embeddings = self.state.get_stage_output("calibration", "calibrated_embeddings")
            final_embeddings.update(calibrated_embeddings)
        
        # Evaluate all final embeddings
        X_original = self.state.get_stage_output("input", "X")
        evaluation_results = {}
        
        for embedding_name, Y in final_embeddings.items():
            if embedding_name == "preprocessed":  # Skip preprocessing stage for evaluation
                continue
                
            try:
                eval_result = self._evaluate_embedding(X_original, Y)
                evaluation_results[embedding_name] = eval_result
            except Exception as e:
                self.logger.warning(f"Evaluation failed for {embedding_name}: {str(e)}")
                continue
        
        return {
            "config": self.config,
            "pipeline_state": {
                "completed_stages": self.state.completed_stages,
                "failed_stages": self.state.failed_stages,
                "errors": self.state.errors,
                "warnings": self.state.warnings,
                "total_time": total_time
            },
            "stage_results": self.state.stage_results,
            "stage_metadata": self.state.stage_metadata,
            "stage_timing": self.state.stage_timing,
            "stage_memory_usage": self.state.stage_memory_usage,
            "final_embeddings": final_embeddings,
            "evaluation_results": evaluation_results,
            "best_embedding": self._select_best_embedding(evaluation_results),
            "resource_usage": self.state.stage_metadata.get("resource_usage", {}),
            "timestamp": datetime.now().isoformat()
        }
    
    def _evaluate_embedding(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
        """Evaluate a single embedding."""
        try:
            # Compute evaluation metrics
            mean_dist, max_dist, _, _ = compute_distortion(X, Y, sample_size=min(1000, X.shape[0]))
            rank_corr = rank_correlation(X, Y, sample_size=min(1000, X.shape[0]))
            nn_overlap = nearest_neighbor_overlap(X, Y, k=min(10, X.shape[0] - 1))
            
            return {
                "mean_distortion": float(mean_dist),
                "max_distortion": float(max_dist),
                "rank_correlation": float(rank_corr),
                "nn_overlap": float(nn_overlap),
                "composite_score": float(rank_corr - 0.1 * mean_dist + 0.1 * nn_overlap)  # Weighted combination
            }
            
        except Exception as e:
            self.logger.warning(f"Embedding evaluation failed: {str(e)}")
            return {
                "mean_distortion": np.inf,
                "max_distortion": np.inf,
                "rank_correlation": 0.0,
                "nn_overlap": 0.0,
                "composite_score": -np.inf
            }
    
    def _select_best_embedding(self, evaluation_results: Dict[str, Dict[str, float]]) -> Optional[str]:
        """Select the best embedding based on evaluation metrics."""
        if not evaluation_results:
            return None
        
        # Find embedding with highest composite score
        best_embedding = max(evaluation_results.keys(), 
                           key=lambda x: evaluation_results[x].get("composite_score", -np.inf))
        
        self.logger.info(f"Best embedding: {best_embedding} "
                        f"(score: {evaluation_results[best_embedding].get('composite_score', 0):.4f})")
        
        return best_embedding
    
    def save_results(self, results: Dict[str, Any], output_dir: Union[str, Path]) -> Path:
        """Save pipeline results to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment-specific subdirectory
        exp_dir = output_dir / f"{self.config.name}_{self.config.get_hash()}"
        exp_dir.mkdir(exist_ok=True)
        
        # Save configuration
        self.config.save(exp_dir / "config.yaml")
        
        # Save results (without large embedding arrays for JSON)
        results_summary = {
            k: v for k, v in results.items() 
            if k not in ["final_embeddings", "stage_results"]
        }
        
        with open(exp_dir / "results_summary.json", "w") as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        # Save embeddings separately
        if self.config.save_embeddings and "final_embeddings" in results:
            embeddings_file = exp_dir / "embeddings.pkl"
            with open(embeddings_file, "wb") as f:
                pickle.dump(results["final_embeddings"], f)
        
        # Save full results as pickle
        with open(exp_dir / "full_results.pkl", "wb") as f:
            pickle.dump(results, f)
        
        self.logger.info(f"Results saved to {exp_dir}")
        return exp_dir