"""
Experiment Configuration System for OrthoReduce

This module provides a comprehensive configuration system for running complex
dimensionality reduction experiments with staged optimization pipelines.
It supports YAML/JSON configuration files, parameter grid specification,
and reproducible experiment tracking.
"""

import json
import yaml
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    """Configuration for a single pipeline stage."""
    name: str
    enabled: bool = True
    method: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)
    timeout_minutes: Optional[int] = None
    memory_limit_gb: Optional[float] = None


@dataclass
class PreprocessingConfig:
    """Stage 1: Preprocessing configuration."""
    name: str = "preprocessing"
    enabled: bool = True
    pca_components: Union[int, str] = "auto"  # Can be int or "auto" for 100-200
    standardization: str = "zscore"  # "zscore", "unit_variance", "robust"
    whitening: bool = False
    denoising_method: str = "none"  # "none", "pca", "jl"
    l2_normalize: bool = False
    cosine_distance_screening: bool = True
    adaptive_pipeline: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_stage_config(self) -> StageConfig:
        return StageConfig(
            name=self.name,
            enabled=self.enabled,
            method="preprocessing",
            parameters={
                "pca_components": self.pca_components,
                "standardization": self.standardization,
                "whitening": self.whitening,
                "denoising_method": self.denoising_method,
                "l2_normalize": self.l2_normalize,
                "cosine_distance_screening": self.cosine_distance_screening,
                "adaptive_pipeline": self.adaptive_pipeline,
                **self.parameters
            },
            dependencies=[],
            output_keys=["preprocessed_data", "preprocessing_metadata"]
        )


@dataclass
class ConvexOptimizationConfig:
    """Stage 2: Convex optimization configuration."""
    name: str = "convex_optimization"
    enabled: bool = True
    k_candidates_grid: List[int] = field(default_factory=lambda: [32, 64, 128])
    lambda_grid: List[float] = field(default_factory=lambda: [1e-6, 1e-4, 1e-2])
    tolerance_grid: List[str] = field(default_factory=lambda: ["strict", "balanced", "loose"])
    objective_types: List[str] = field(default_factory=lambda: ["quadratic", "huber"])
    use_float64: bool = False
    batch_size: int = 1024
    candidate_normalization: str = "none"
    grid_search_cv: int = 3
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_stage_config(self) -> StageConfig:
        return StageConfig(
            name=self.name,
            enabled=self.enabled,
            method="convex_optimization",
            parameters={
                "k_candidates_grid": self.k_candidates_grid,
                "lambda_grid": self.lambda_grid,
                "tolerance_grid": self.tolerance_grid,
                "objective_types": self.objective_types,
                "use_float64": self.use_float64,
                "batch_size": self.batch_size,
                "candidate_normalization": self.candidate_normalization,
                "grid_search_cv": self.grid_search_cv,
                **self.parameters
            },
            dependencies=["preprocessing"],
            output_keys=["convex_optimized_data", "convex_optimization_metadata"]
        )


@dataclass
class GeometricEmbeddingsConfig:
    """Stage 3: Spherical and Poincaré embeddings configuration."""
    name: str = "geometric_embeddings"
    enabled: bool = True
    spherical_config: Dict[str, Any] = field(default_factory=lambda: {
        "methods": ["riemannian", "fast"],
        "loss_types": ["mds_geodesic", "hybrid"],
        "max_iter": 500,
        "learning_rates": [0.01, 0.05],
        "adaptive_radius": True,
        "hemisphere_constraint": True
    })
    poincare_config: Dict[str, Any] = field(default_factory=lambda: {
        "curvatures": [0.5, 1.0, 2.0],
        "learning_rates": [0.01, 0.02],
        "n_epochs": 100,
        "optimizers": ["radam"],
        "loss_functions": ["stress"],
        "init_methods": ["pca"],
        "regularization": 0.01
    })
    riemannian_optimization: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_stage_config(self) -> StageConfig:
        return StageConfig(
            name=self.name,
            enabled=self.enabled,
            method="geometric_embeddings",
            parameters={
                "spherical_config": self.spherical_config,
                "poincare_config": self.poincare_config,
                "riemannian_optimization": self.riemannian_optimization,
                **self.parameters
            },
            dependencies=["convex_optimization"],
            output_keys=["spherical_embedding", "poincare_embedding", "geometric_metadata"]
        )


@dataclass
class CalibrationConfig:
    """Stage 4: Post-calibration configuration."""
    name: str = "calibration"
    enabled: bool = True
    methods: List[str] = field(default_factory=lambda: ["procrustes", "isotonic", "local"])
    isotonic_params: Dict[str, Any] = field(default_factory=lambda: {
        "sample_sizes": [None, 5000, 10000],
        "increasing": True,
        "out_of_bounds": "clip"
    })
    procrustes_params: Dict[str, Any] = field(default_factory=lambda: {
        "scaling": True,
        "reflection": False
    })
    local_correction_params: Dict[str, Any] = field(default_factory=lambda: {
        "k_neighbors_grid": [5, 10, 20],
        "adaptive_k": True,
        "reg_strengths": [1e-6, 1e-4, 1e-2]
    })
    apply_to_all_embeddings: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_stage_config(self) -> StageConfig:
        return StageConfig(
            name=self.name,
            enabled=self.enabled,
            method="calibration",
            parameters={
                "methods": self.methods,
                "isotonic_params": self.isotonic_params,
                "procrustes_params": self.procrustes_params,
                "local_correction_params": self.local_correction_params,
                "apply_to_all_embeddings": self.apply_to_all_embeddings,
                **self.parameters
            },
            dependencies=["geometric_embeddings"],
            output_keys=["calibrated_embeddings", "calibration_metadata"]
        )


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    description: str = ""
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Data configuration
    data_source: str = "synthetic"  # "synthetic", "ms_data", "custom"
    data_params: Dict[str, Any] = field(default_factory=dict)
    
    # Global experiment settings
    random_seed: int = 42
    n_runs: int = 3  # Multiple runs for statistical significance
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    
    # Pipeline stages
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    convex_optimization: ConvexOptimizationConfig = field(default_factory=ConvexOptimizationConfig)
    geometric_embeddings: GeometricEmbeddingsConfig = field(default_factory=GeometricEmbeddingsConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    
    # Comparison methods (baseline methods to compare against)
    baseline_methods: List[str] = field(default_factory=lambda: ["pca", "jll", "umap"])
    baseline_params: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation configuration
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "rank_correlation", "distortion", "nn_overlap", "stress"
    ])
    statistical_tests: List[str] = field(default_factory=lambda: [
        "paired_t_test", "wilcoxon", "mann_whitney"
    ])
    
    # Output configuration
    output_dir: str = "experiments"
    save_intermediate_results: bool = True
    save_embeddings: bool = True
    generate_plots: bool = True
    plot_formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    
    # Computational resources
    n_jobs: int = -1  # Parallel processing
    memory_limit_gb: Optional[float] = None
    timeout_hours: Optional[float] = None
    use_gpu: bool = False
    
    # Advanced options
    debug_mode: bool = False
    verbose_logging: bool = True
    save_intermediate_data: bool = False

    def get_stages(self) -> List[StageConfig]:
        """Get all pipeline stages as StageConfig objects."""
        stages = []
        if self.preprocessing.enabled:
            stages.append(self.preprocessing.to_stage_config())
        if self.convex_optimization.enabled:
            stages.append(self.convex_optimization.to_stage_config())
        if self.geometric_embeddings.enabled:
            stages.append(self.geometric_embeddings.to_stage_config())
        if self.calibration.enabled:
            stages.append(self.calibration.to_stage_config())
        return stages

    def get_hash(self) -> str:
        """Generate a unique hash for this configuration."""
        config_dict = asdict(self)
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to file."""
        filepath = Path(filepath)
        config_dict = asdict(self)
        
        if filepath.suffix.lower() == '.yaml' or filepath.suffix.lower() == '.yml':
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            if filepath.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        # Handle nested dataclass construction
        if 'preprocessing' in config_dict:
            config_dict['preprocessing'] = PreprocessingConfig(**config_dict['preprocessing'])
        if 'convex_optimization' in config_dict:
            config_dict['convex_optimization'] = ConvexOptimizationConfig(**config_dict['convex_optimization'])
        if 'geometric_embeddings' in config_dict:
            config_dict['geometric_embeddings'] = GeometricEmbeddingsConfig(**config_dict['geometric_embeddings'])
        if 'calibration' in config_dict:
            config_dict['calibration'] = CalibrationConfig(**config_dict['calibration'])
        
        logger.info(f"Configuration loaded from {filepath}")
        return cls(**config_dict)

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Basic validation
        if not self.name:
            issues.append("Experiment name is required")
        
        if self.random_seed < 0:
            issues.append("Random seed must be non-negative")
        
        if self.n_runs < 1:
            issues.append("Number of runs must be at least 1")
        
        if not (0 < self.validation_split < 1):
            issues.append("Validation split must be between 0 and 1")
        
        # Stage dependency validation
        stages = self.get_stages()
        stage_names = {stage.name for stage in stages}
        
        for stage in stages:
            for dep in stage.dependencies:
                if dep not in stage_names:
                    issues.append(f"Stage '{stage.name}' depends on unknown stage '{dep}'")
        
        # Check for circular dependencies
        if self._has_circular_dependencies(stages):
            issues.append("Circular dependency detected in pipeline stages")
        
        # Computational resource validation
        if self.memory_limit_gb is not None and self.memory_limit_gb <= 0:
            issues.append("Memory limit must be positive")
        
        if self.timeout_hours is not None and self.timeout_hours <= 0:
            issues.append("Timeout must be positive")
        
        return issues

    def _has_circular_dependencies(self, stages: List[StageConfig]) -> bool:
        """Check for circular dependencies in stages."""
        def has_cycle(graph, node, visited, rec_stack):
            visited[node] = True
            rec_stack[node] = True
            
            for neighbor in graph.get(node, []):
                if not visited.get(neighbor, False):
                    if has_cycle(graph, neighbor, visited, rec_stack):
                        return True
                elif rec_stack.get(neighbor, False):
                    return True
            
            rec_stack[node] = False
            return False
        
        # Build dependency graph
        graph = {}
        for stage in stages:
            graph[stage.name] = stage.dependencies
        
        visited = {}
        rec_stack = {}
        
        for stage_name in graph:
            if not visited.get(stage_name, False):
                if has_cycle(graph, stage_name, visited, rec_stack):
                    return True
        
        return False


# Predefined configuration templates

def create_default_config(
    name: str = "default_experiment",
    data_source: str = "synthetic"
) -> ExperimentConfig:
    """Create a default experiment configuration."""
    return ExperimentConfig(
        name=name,
        description="Default OrthoReduce experiment with staged optimization",
        data_source=data_source,
        data_params={
            "n_samples": 1000,
            "n_features": 500,
            "n_components": 50,
            "noise_level": 0.1
        }
    )


def create_fast_config(
    name: str = "fast_experiment",
    data_source: str = "synthetic"
) -> ExperimentConfig:
    """Create a configuration optimized for speed."""
    config = create_default_config(name, data_source)
    
    # Reduce computational complexity
    config.n_runs = 1
    config.cross_validation_folds = 3
    
    # Simplify preprocessing
    config.preprocessing.adaptive_pipeline = False
    config.preprocessing.denoising_method = "none"
    
    # Reduce convex optimization search space
    config.convex_optimization.k_candidates_grid = [64]
    config.convex_optimization.lambda_grid = [1e-4]
    config.convex_optimization.tolerance_grid = ["balanced"]
    config.convex_optimization.objective_types = ["quadratic"]
    
    # Simplify geometric embeddings
    config.geometric_embeddings.spherical_config["methods"] = ["fast"]
    config.geometric_embeddings.spherical_config["max_iter"] = 100
    config.geometric_embeddings.poincare_config["n_epochs"] = 50
    
    # Reduce calibration complexity
    config.calibration.methods = ["procrustes", "isotonic"]
    config.calibration.local_correction_params["k_neighbors_grid"] = [10]
    
    return config


def create_comprehensive_config(
    name: str = "comprehensive_experiment",
    data_source: str = "synthetic"
) -> ExperimentConfig:
    """Create a comprehensive configuration for thorough evaluation."""
    config = create_default_config(name, data_source)
    
    # Increase statistical power
    config.n_runs = 5
    config.cross_validation_folds = 10
    
    # Enable comprehensive preprocessing
    config.preprocessing.adaptive_pipeline = True
    config.preprocessing.denoising_method = "pca"
    config.preprocessing.whitening = True
    
    # Extensive hyperparameter search
    config.convex_optimization.k_candidates_grid = [32, 64, 128, 256]
    config.convex_optimization.lambda_grid = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
    config.convex_optimization.tolerance_grid = ["strict", "balanced", "loose"]
    config.convex_optimization.objective_types = ["quadratic", "huber", "epsilon_insensitive"]
    
    # Comprehensive geometric embeddings
    config.geometric_embeddings.spherical_config.update({
        "methods": ["riemannian", "fast"],
        "loss_types": ["mds_geodesic", "triplet", "nca", "hybrid"],
        "max_iter": 1000,
        "learning_rates": [0.005, 0.01, 0.02, 0.05]
    })
    config.geometric_embeddings.poincare_config.update({
        "curvatures": [0.1, 0.5, 1.0, 2.0, 5.0],
        "learning_rates": [0.005, 0.01, 0.02],
        "n_epochs": 200,
        "optimizers": ["rsgd", "radam"],
        "loss_functions": ["stress", "triplet", "nca"],
        "init_methods": ["pca", "spectral"]
    })
    
    # Full calibration suite
    config.calibration.methods = ["procrustes", "isotonic", "local"]
    config.calibration.isotonic_params["sample_sizes"] = [None, 1000, 5000, 10000]
    config.calibration.local_correction_params.update({
        "k_neighbors_grid": [5, 10, 15, 20, 30],
        "reg_strengths": [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
    })
    
    # Extended baseline comparison
    config.baseline_methods = ["pca", "jll", "umap", "tsne", "isomap"]
    
    # Comprehensive evaluation
    config.evaluation_metrics = [
        "rank_correlation", "distortion", "nn_overlap", "stress", "geodesic_stress"
    ]
    config.statistical_tests = [
        "paired_t_test", "wilcoxon", "mann_whitney", "kruskal_wallis", "friedman"
    ]
    
    return config


def create_ms_data_config(
    name: str = "ms_experiment",
    mzml_path: str = "data/sample.mzML"
) -> ExperimentConfig:
    """Create a configuration specialized for mass spectrometry data."""
    config = create_default_config(name, "ms_data")
    
    config.data_params = {
        "mzml_path": mzml_path,
        "max_spectra": 1000,
        "min_intensity": 1e4,
        "mz_range": (100, 2000),
        "rt_range": None,
        "use_fingerprinting": True
    }
    
    # MS-specific preprocessing
    config.preprocessing.standardization = "robust"  # Better for MS data
    config.preprocessing.l2_normalize = True  # Common for MS spectra
    config.preprocessing.cosine_distance_screening = True
    config.preprocessing.adaptive_pipeline = True
    
    # Focus on methods that work well with sparse, high-dimensional data
    config.convex_optimization.k_candidates_grid = [64, 128, 256]
    config.convex_optimization.objective_types = ["huber"]  # Robust to outliers
    
    # Geometric embeddings may need different parameters for MS data
    config.geometric_embeddings.spherical_config["loss_types"] = ["mds_geodesic", "hybrid"]
    config.geometric_embeddings.poincare_config["curvatures"] = [1.0, 2.0, 5.0]
    
    return config


# Configuration validation and utilities

def validate_config_file(filepath: Union[str, Path]) -> Tuple[bool, List[str]]:
    """Validate a configuration file without loading it fully."""
    try:
        config = ExperimentConfig.load(filepath)
        issues = config.validate()
        return len(issues) == 0, issues
    except Exception as e:
        return False, [f"Failed to load configuration: {str(e)}"]


def merge_configs(base_config: ExperimentConfig, override_config: ExperimentConfig) -> ExperimentConfig:
    """Merge two configurations, with override_config taking precedence."""
    # This is a simplified merge - in practice, you'd want more sophisticated merging
    base_dict = asdict(base_config)
    override_dict = asdict(override_config)
    
    def merge_dicts(base, override):
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                base[key] = merge_dicts(base[key], value)
            else:
                base[key] = value
        return base
    
    merged_dict = merge_dicts(base_dict, override_dict)
    
    # Reconstruct nested dataclasses
    if 'preprocessing' in merged_dict:
        merged_dict['preprocessing'] = PreprocessingConfig(**merged_dict['preprocessing'])
    if 'convex_optimization' in merged_dict:
        merged_dict['convex_optimization'] = ConvexOptimizationConfig(**merged_dict['convex_optimization'])
    if 'geometric_embeddings' in merged_dict:
        merged_dict['geometric_embeddings'] = GeometricEmbeddingsConfig(**merged_dict['geometric_embeddings'])
    if 'calibration' in merged_dict:
        merged_dict['calibration'] = CalibrationConfig(**merged_dict['calibration'])
    
    return ExperimentConfig(**merged_dict)