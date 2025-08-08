# Staged Optimization Experiment Orchestration System

## Overview

This document describes the comprehensive experiment orchestration system for staged optimization in dimensionality reduction. The system implements a sophisticated pipeline that combines:

1. **Stage 1**: Preprocessing with PCA (100-200 dims), cosine distance screening
2. **Stage 2**: Convex optimization with grid search over k_candidates, λ, tolerance  
3. **Stage 3**: Spherical and Poincaré embeddings with Riemannian optimization
4. **Stage 4**: Post-calibration with isotonic regression and Procrustes alignment

## Key Features

- **Comprehensive Configuration System**: YAML/JSON configuration files with parameter grids
- **Staged Pipeline Orchestration**: Dependency management and data flow tracking
- **Hyperparameter Search**: Grid, random, and Sobol sequence parameter exploration
- **Statistical Method Comparison**: Multi-method comparison with significance testing
- **Results Aggregation**: Best method selection with multiple criteria
- **CLI Interface**: Command-line tools for experiment management
- **Comprehensive Logging**: Structured logging, progress tracking, resource monitoring

## System Architecture

### Core Components

```
orthogonal_projection/
├── experiment_config.py          # Configuration system and templates
├── pipeline_orchestrator.py      # Pipeline execution and stage management
├── hyperparameter_search.py      # Parameter search and method comparison
├── results_aggregator.py         # Results analysis and best method selection
├── experiment_logger.py          # Comprehensive logging system
├── experiment_orchestration.py   # Main integration and convenience functions
└── cli.py                        # Command-line interface
```

### Configuration Schema

The system uses a hierarchical configuration system with these main sections:

```yaml
name: experiment_name
description: "Experiment description"

# Data Configuration
data_source: synthetic | ms_data | custom
data_params:
  n_samples: 1000
  n_features: 500
  n_components: 50

# Global Settings
random_seed: 42
n_runs: 3
validation_split: 0.2
cross_validation_folds: 5

# Pipeline Stages
preprocessing:
  enabled: true
  pca_components: auto
  standardization: zscore
  # ... stage-specific parameters

convex_optimization:
  enabled: true
  k_candidates_grid: [32, 64, 128]
  lambda_grid: [1e-6, 1e-4, 1e-2]
  # ... hyperparameter grids

geometric_embeddings:
  enabled: true
  spherical_config:
    methods: [riemannian, fast]
    loss_types: [mds_geodesic, hybrid]
  poincare_config:
    curvatures: [0.5, 1.0, 2.0]
  # ... embedding parameters

calibration:
  enabled: true
  methods: [procrustes, isotonic, local]
  # ... calibration parameters
```

## Usage Examples

### 1. Command-Line Interface

#### Basic Experiment
```bash
# Run experiment with default configuration
python -m orthogonal_projection.cli run --name my_experiment

# Run with custom configuration
python -m orthogonal_projection.cli run --config my_config.yaml

# Create configuration template
python -m orthogonal_projection.cli create-config --name exp1 --template comprehensive
```

#### Interactive Configuration
```bash
# Interactive configuration creation
python -m orthogonal_projection.cli interactive
```

#### Experiment Management
```bash
# Validate configuration
python -m orthogonal_projection.cli validate --config config.yaml

# List available pipeline stages
python -m orthogonal_projection.cli list-stages --detailed

# Analyze existing results
python -m orthogonal_projection.cli analyze --results-path results.pkl
```

### 2. Programmatic API

#### Simple Experiment
```python
from orthogonal_projection.experiment_orchestration import run_synthetic_data_experiment

# Run quick experiment
results = run_synthetic_data_experiment(
    experiment_name="test_experiment",
    n_samples=1000,
    n_features=500,
    template="fast"
)

print(f"Best method: {results['best_overall_method']}")
```

#### Custom Configuration
```python
from orthogonal_projection.experiment_config import create_default_config
from orthogonal_projection.experiment_orchestration import StagedOptimizationExperiment
from orthogonal_projection.dimensionality_reduction import generate_mixture_gaussians

# Create and customize configuration
config = create_default_config("custom_experiment", "synthetic")
config.n_runs = 5
config.convex_optimization.k_candidates_grid = [64, 128, 256]

# Generate data
X = generate_mixture_gaussians(1000, 500, 50)

# Run experiment
experiment = StagedOptimizationExperiment(config)
results = experiment.run_complete_experiment(X)
```

#### Advanced Usage
```python
# Load configuration from file
config = ExperimentConfig.load("config.yaml")

# Run hyperparameter search
param_grid = {
    "convex_lambda": {"type": "logrange", "start": 1e-6, "stop": 1e-2, "num": 5},
    "spherical_lr": {"type": "categorical", "values": [0.01, 0.02, 0.05]}
}
search_results = experiment.run_hyperparameter_search_experiment(X, param_grid)

# Compare with baseline methods
baseline_methods = {
    "pca": lambda X: run_pca_simple(X, k=50),
    "jll": lambda X: run_jll_simple(X, k=50)
}
comparison_results = experiment.run_method_comparison_experiment(X, baseline_methods)
```

## Configuration Templates

### Fast Template
Optimized for speed with minimal hyperparameter search:
- Single run
- Limited parameter grids
- Fast geometric embedding methods
- Basic calibration

### Comprehensive Template  
Thorough evaluation with extensive search:
- Multiple runs (5+)
- Large hyperparameter grids
- All geometric embedding variants
- Full calibration suite
- Extended baseline comparisons

### Mass Spectrometry Template
Specialized for MS data:
- Robust preprocessing
- Cosine distance metrics
- Outlier-resistant methods
- MS-specific evaluation metrics

## Experiment Workflow

### 1. Configuration Phase
- Define experiment parameters
- Specify hyperparameter grids
- Configure pipeline stages
- Set evaluation criteria

### 2. Execution Phase
- Data loading/generation
- Multi-run pipeline execution
- Stage-wise dependency management
- Progress tracking and logging

### 3. Analysis Phase
- Results aggregation across runs
- Statistical significance testing
- Best method selection
- Comprehensive reporting

### 4. Output Phase
- Structured results storage
- Performance visualizations
- Detailed analysis reports
- Method comparison summaries

## Best Method Selection

The system uses a multi-criteria approach for selecting the best method:

### Selection Criteria
- **Performance Metrics** (40%): Spearman rank correlation
- **Distortion Penalty** (-20%): Lower distortion preferred
- **Nearest Neighbor Overlap** (20%): Local structure preservation
- **Runtime Efficiency** (-10%): Lower runtime preferred  
- **Memory Efficiency** (-5%): Lower memory usage preferred
- **Robustness** (15%): Consistency across runs
- **Statistical Significance** (10%): Significant improvements

### Composite Scoring
Each method receives a composite score based on weighted criteria. The system performs:
- Cross-validation evaluation
- Statistical significance testing
- Effect size calculations
- Sensitivity analysis

## Logging and Monitoring

### Structured Logging
- JSON-formatted log entries
- Stage-specific context tracking
- Performance metrics logging
- Error and warning capture

### Progress Tracking
- Real-time progress bars
- Stage completion monitoring
- ETA calculations
- Resource usage tracking

### Resource Monitoring
- CPU and memory usage
- GPU utilization (if available)
- Peak resource consumption
- Resource efficiency metrics

## Output Structure

```
experiments/
├── experiment_name_hash/
│   ├── config.yaml                    # Experiment configuration
│   ├── results_summary.json           # High-level results
│   ├── full_results.pkl              # Complete results object
│   ├── embeddings.pkl                # Final embeddings
│   ├── experiment_report.txt         # Comprehensive report
│   └── method_comparison.txt         # Method comparison analysis
├── logs/
│   ├── experiment_id.log             # Text logs
│   └── experiment_id.jsonl           # Structured JSON logs
└── results/
    ├── aggregated_results.pkl        # Aggregated analysis
    └── aggregated_results_summary.json
```

## Integration with Existing Codebase

The orchestration system integrates seamlessly with existing OrthoReduce components:

- **preprocessing.py**: Adaptive preprocessing pipeline
- **convex_optimized.py**: Enhanced convex hull projection  
- **spherical_embeddings.py**: Riemannian spherical embeddings
- **hyperbolic.py**: Poincaré ball embeddings
- **calibration.py**: Post-processing calibration methods
- **evaluation.py**: Comprehensive evaluation metrics

## Performance Considerations

### Computational Efficiency
- Parallel hyperparameter search
- Batch processing for large datasets
- Memory-efficient data handling
- GPU acceleration support (where applicable)

### Scalability
- Configurable resource limits
- Adaptive batch sizing
- Distributed execution support
- Progress checkpointing

### Reproducibility
- Deterministic random seeding
- Configuration versioning
- Complete audit trails
- Environment capturing

## Error Handling and Robustness

### Fault Tolerance
- Graceful stage failure handling
- Automatic fallback mechanisms
- Partial result preservation
- Comprehensive error logging

### Validation
- Configuration validation
- Input data validation
- Parameter range checking
- Dependency verification

## Extension Points

The system is designed for extensibility:

### Adding New Stages
1. Create stage-specific configuration class
2. Implement stage executor function
3. Define stage dependencies
4. Add to pipeline orchestrator

### Custom Evaluation Metrics
1. Define metric calculation function
2. Add to evaluation configuration
3. Include in results aggregation
4. Update best method selection

### New Hyperparameter Search Strategies
1. Implement search strategy in ParameterGridGenerator
2. Add strategy option to configuration
3. Test with existing pipeline

## Troubleshooting

### Common Issues
1. **Configuration Validation Errors**: Check parameter types and ranges
2. **Stage Dependencies**: Ensure prerequisite stages are enabled
3. **Memory Issues**: Reduce batch sizes or enable memory limits
4. **Runtime Errors**: Check log files for detailed error messages

### Performance Optimization
1. Use "fast" template for initial testing
2. Reduce hyperparameter grid sizes
3. Enable parallel processing
4. Use appropriate data sampling

### Debugging
1. Enable debug mode in configuration
2. Use verbose logging
3. Save intermediate results
4. Run with single experiment run

## Future Enhancements

### Planned Features
- Automated hyperparameter optimization (Bayesian optimization)
- Distributed computing support
- Interactive result exploration
- Advanced visualization dashboards
- Integration with experiment tracking systems

### Extension Opportunities
- Support for additional data types
- Custom loss function frameworks
- Online learning capabilities
- Federated experiment coordination

---

This orchestration system provides a comprehensive framework for conducting sophisticated dimensionality reduction experiments with rigorous evaluation and comparison capabilities. The modular design ensures extensibility while maintaining ease of use for both researchers and practitioners.