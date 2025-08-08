
# Dashboard Integration Guide

This guide shows how to integrate the OrthoReduce Dashboard with your existing experiments and workflows.

## 🔗 Quick Integration

### 1. Run Experiments and Launch Dashboard

```bash
# Run your OrthoReduce experiments
python main.py  # or your experiment script

# Launch dashboard to view results
python launch_dashboard.py
```

### 2. Programmatic Integration

```python
from orthogonal_projection.dashboard_utils import ExperimentDatabase
from orthogonal_projection.experiment_orchestration import StagedOptimizationExperiment
from orthogonal_projection.experiment_config import create_default_config

# Run experiment
config = create_default_config("my_experiment", "synthetic")
experiment = StagedOptimizationExperiment(config)
results = experiment.run_complete_experiment(X)

# Store results in dashboard database
db = ExperimentDatabase()
experiment_id = db.store_experiment(
    name="my_experiment",
    config=config,
    results=results['evaluation_results'],
    embeddings=results['final_embeddings']
)

print(f"Experiment stored with ID: {experiment_id}")
print("Launch dashboard to view results: python launch_dashboard.py")
```

## 📊 Integration Examples

### Example 1: Synthetic Data Experiment with Dashboard

```python
#!/usr/bin/env python3
"""
Complete example showing experiment execution and dashboard integration.
"""

import numpy as np
from orthogonal_projection.experiment_orchestration import run_synthetic_data_experiment
from orthogonal_projection.dashboard_utils import ExperimentDatabase
import subprocess
import sys

def main():
    print("🧪 Running synthetic data experiment...")
    
    # Run experiment
    results = run_synthetic_data_experiment(
        experiment_name="dashboard_demo",
        n_samples=1000,
        n_features=200,
        n_components=50,
        template="comprehensive"
    )
    
    print(f"✅ Experiment completed: {results['best_overall_method']}")
    print(f"📁 Results saved to: {results['results_path']}")
    
    # Store in dashboard database
    db = ExperimentDatabase()
    experiment_id = db.store_experiment(
        name="dashboard_demo",
        config=results['experiment_config'],
        results=results['aggregated_results'].statistical_summary,
        embeddings=results.get('final_embeddings', {})
    )
    
    print(f"💾 Stored in dashboard database with ID: {experiment_id}")
    
    # Launch dashboard
    print("🚀 Launching dashboard...")
    try:
        subprocess.run([sys.executable, "launch_dashboard.py"], check=True)
    except KeyboardInterrupt:
        print("Dashboard stopped by user")

if __name__ == "__main__":
    main()
```

### Example 2: Mass Spectrometry Data with Dashboard

```python
#!/usr/bin/env python3
"""
Mass spectrometry experiment with dashboard integration.
"""

from orthogonal_projection.experiment_orchestration import run_mass_spectrometry_experiment
from orthogonal_projection.dashboard_utils import ExperimentDatabase
import sys

def main():
    mzml_path = "archive/mass_spectrometry/data/20150731_QEp7_MiBa_SA_SKOV3-1.mzML"
    
    print(f"🧬 Running MS experiment on: {mzml_path}")
    
    try:
        results = run_mass_spectrometry_experiment(
            experiment_name="ms_dashboard_demo",
            mzml_path=mzml_path,
            max_spectra=500
        )
        
        print(f"✅ MS experiment completed: {results['best_overall_method']}")
        
        # Store results
        db = ExperimentDatabase()
        experiment_id = db.store_experiment(
            name="ms_dashboard_demo",
            config=results['experiment_config'],
            results=results['aggregated_results'].statistical_summary
        )
        
        print(f"💾 Stored in dashboard with ID: {experiment_id}")
        print("🚀 Launch dashboard: python launch_dashboard.py")
        
    except FileNotFoundError:
        print(f"❌ mzML file not found: {mzml_path}")
        print("Please ensure the file exists or use a different path")

if __name__ == "__main__":
    main()
```

### Example 3: Custom Analysis Integration

```python
#!/usr/bin/env python3
"""
Custom analysis with dashboard visualization.
"""

import numpy as np
from orthogonal_projection import run_jll_simple, run_pca_simple, run_umap_simple
from orthogonal_projection.evaluation import compute_distortion, rank_correlation
from orthogonal_projection.dashboard_utils import ExperimentDatabase, DataProcessor
import json

def run_custom_analysis():
    """Run custom dimensionality reduction analysis."""
    
    # Generate test data
    np.random.seed(42)
    n, d, k = 1000, 100, 20
    X = np.random.randn(n, d)
    
    print(f"🔬 Running custom analysis on {n}×{d} data...")
    
    # Run multiple methods
    methods = {}
    
    # JLL projection
    Y_jll, runtime_jll = run_jll_simple(X, k)
    methods['JLL'] = {
        'embedding': Y_jll,
        'runtime': runtime_jll,
        'rank_correlation': rank_correlation(X, Y_jll),
        'mean_distortion': compute_distortion(X, Y_jll)[0],
        'compression_ratio': d / k
    }
    
    # PCA
    Y_pca, runtime_pca = run_pca_simple(X, k)
    methods['PCA'] = {
        'embedding': Y_pca,
        'runtime': runtime_pca,
        'rank_correlation': rank_correlation(X, Y_pca),
        'mean_distortion': compute_distortion(X, Y_pca)[0],
        'compression_ratio': d / k
    }
    
    # UMAP (if available)
    try:
        Y_umap, runtime_umap = run_umap_simple(X, k)
        methods['UMAP'] = {
            'embedding': Y_umap,
            'runtime': runtime_umap,
            'rank_correlation': rank_correlation(X, Y_umap),
            'mean_distortion': compute_distortion(X, Y_umap)[0],
            'compression_ratio': d / k
        }
    except ImportError:
        print("⚠️  UMAP not available, skipping...")
    
    # Prepare results for dashboard
    dashboard_results = {}
    embeddings = {}
    
    for method_name, method_data in methods.items():
        dashboard_results[method_name] = {
            k: v for k, v in method_data.items() 
            if k != 'embedding'
        }
        embeddings[method_name] = method_data['embedding']
    
    # Store in dashboard
    db = ExperimentDatabase()
    experiment_id = db.store_experiment(
        name="custom_analysis",
        config=None,
        results=dashboard_results,
        embeddings=embeddings
    )
    
    print(f"✅ Custom analysis completed")
    print(f"💾 Stored with ID: {experiment_id}")
    
    # Save results to JSON for manual inspection
    with open("custom_analysis_results.json", "w") as f:
        json.dump(dashboard_results, f, indent=2, default=str)
    
    print("📄 Results also saved to: custom_analysis_results.json")
    print("🚀 Launch dashboard: python launch_dashboard.py")
    
    return dashboard_results, embeddings

if __name__ == "__main__":
    run_custom_analysis()
```

## 🔧 Configuration Integration

### Environment-based Configuration

```bash
# Set environment variables for dashboard
export ORTHOREDUCE_RESULTS_DIR="./my_experiment_results"
export ORTHOREDUCE_CACHE_DIR="./my_dashboard_cache"
export ORTHOREDUCE_DEBUG="true"

# Launch with environment settings
python launch_dashboard.py
```

### Config File Integration

```python
# Create custom config
import yaml

config = {
    'host': '0.0.0.0',  # Allow external access
    'port': 8502,
    'results_dir': './experiment_outputs',
    'auto_refresh': True,
    'enable_monitoring': True,
    'dashboard_settings': {
        'default_view': 'method_comparison',
        'show_advanced_metrics': True,
        'cache_embeddings': True
    }
}

with open('my_dashboard_config.yaml', 'w') as f:
    yaml.dump(config, f)

# Launch with custom config
# python launch_dashboard.py --config my_dashboard_config.yaml
```

## 🚀 Workflow Integration

### Jupyter Notebook Integration

```python
# In a Jupyter notebook cell
import subprocess
import threading
import time

def launch_dashboard_background():
    """Launch dashboard in background thread."""
    subprocess.run([
        'python', 'launch_dashboard.py', 
        '--port', '8502',  # Use different port to avoid conflicts
        '--results-dir', './notebook_results'
    ])

# Start dashboard in background
dashboard_thread = threading.Thread(target=launch_dashboard_background)
dashboard_thread.daemon = True
dashboard_thread.start()

print("Dashboard starting in background...")
time.sleep(3)
print("Access dashboard at: http://localhost:8502")
```

### Batch Processing Integration

```python
#!/usr/bin/env python3
"""
Batch processing with dashboard updates.
"""

from pathlib import Path
from orthogonal_projection.dashboard_utils import ExperimentDatabase
import json
import time

def process_experiments_batch(experiment_configs):
    """Process multiple experiments and update dashboard."""
    
    db = ExperimentDatabase()
    
    for i, config in enumerate(experiment_configs):
        print(f"📊 Processing experiment {i+1}/{len(experiment_configs)}: {config['name']}")
        
        try:
            # Run experiment (pseudo-code)
            results = run_experiment(config)
            
            # Store in dashboard
            experiment_id = db.store_experiment(
                name=config['name'],
                config=config,
                results=results['metrics'],
                embeddings=results.get('embeddings', {})
            )
            
            print(f"✅ Completed: {config['name']} (ID: {experiment_id})")
            
        except Exception as e:
            print(f"❌ Failed: {config['name']} - {e}")
            continue
        
        time.sleep(1)  # Brief pause between experiments
    
    print(f"🎉 Batch processing complete!")
    print("🚀 Launch dashboard: python launch_dashboard.py")

# Example usage
experiment_configs = [
    {'name': 'exp_small', 'n_samples': 500, 'n_features': 100},
    {'name': 'exp_medium', 'n_samples': 1000, 'n_features': 200},
    {'name': 'exp_large', 'n_samples': 2000, 'n_features': 500},
]

process_experiments_batch(experiment_configs)
```

## 🔍 Monitoring Integration

### Real-time Experiment Monitoring

```python
#!/usr/bin/env python3
"""
Monitor running experiments with dashboard integration.
"""

import time
import json
from pathlib import Path
from datetime import datetime

class ExperimentMonitor:
    """Monitor running experiments for dashboard."""
    
    def __init__(self, experiments_dir="./experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.monitoring = True
    
    def create_experiment_lock(self, experiment_name):
        """Create lock file for running experiment."""
        lock_file = self.experiments_dir / f"{experiment_name}.lock"
        with open(lock_file, 'w') as f:
            json.dump({
                'status': 'running',
                'started': datetime.now().isoformat(),
                'pid': os.getpid()
            }, f)
        return lock_file
    
    def update_experiment_progress(self, experiment_name, stage, progress):
        """Update experiment progress for dashboard."""
        progress_file = self.experiments_dir / f"{experiment_name}_progress.json"
        with open(progress_file, 'w') as f:
            json.dump({
                'experiment': experiment_name,
                'current_stage': stage,
                'progress_percent': progress,
                'last_update': datetime.now().isoformat()
            }, f)
    
    def complete_experiment(self, experiment_name, results):
        """Mark experiment as complete and cleanup."""
        # Remove lock file
        lock_file = self.experiments_dir / f"{experiment_name}.lock"
        if lock_file.exists():
            lock_file.unlink()
        
        # Remove progress file
        progress_file = self.experiments_dir / f"{experiment_name}_progress.json"
        if progress_file.exists():
            progress_file.unlink()
        
        # Save final results
        results_file = self.experiments_dir / f"{experiment_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

# Example usage in experiment
def run_monitored_experiment():
    monitor = ExperimentMonitor()
    experiment_name = "monitored_experiment"
    
    # Start monitoring
    lock_file = monitor.create_experiment_lock(experiment_name)
    
    try:
        # Preprocessing stage
        monitor.update_experiment_progress(experiment_name, "preprocessing", 10)
        time.sleep(2)  # Simulate work
        
        # Dimensionality reduction stage
        monitor.update_experiment_progress(experiment_name, "dimensionality_reduction", 50)
        time.sleep(3)  # Simulate work
        
        # Evaluation stage
        monitor.update_experiment_progress(experiment_name, "evaluation", 90)
        time.sleep(1)  # Simulate work
        
        # Complete
        results = {"status": "completed", "best_method": "PCA"}
        monitor.complete_experiment(experiment_name, results)
        print("✅ Monitored experiment completed")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        if lock_file.exists():
            lock_file.unlink()

if __name__ == "__main__":
    run_monitored_experiment()
```

## 📱 Mobile/Remote Access

### Secure Remote Access

```bash
# Launch dashboard with external access
python launch_dashboard.py --host 0.0.0.0 --port 8501

# With authentication (recommended for remote access)
# Set up reverse proxy with nginx or similar for production
```

### Cloud Deployment

```yaml
# Docker deployment configuration
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["python", "launch_dashboard.py", "--host", "0.0.0.0", "--port", "8501"]
```

## 🧪 Testing Integration

```python
#!/usr/bin/env python3
"""
Test dashboard integration.
"""

import tempfile
import json
from pathlib import Path
from orthogonal_projection.dashboard_utils import ExperimentDatabase

def test_dashboard_integration():
    """Test basic dashboard integration functionality."""
    
    # Create temporary results
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Mock experiment results
        mock_results = {
            "PCA": {
                "rank_correlation": 0.95,
                "mean_distortion": 0.05,
                "runtime": 0.001
            },
            "JLL": {
                "rank_correlation": 0.88,
                "mean_distortion": 0.12,
                "runtime": 0.0005
            }
        }
        
        # Save results file
        results_file = temp_path / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(mock_results, f)
        
        # Test database storage
        db = ExperimentDatabase(":memory:")  # In-memory database for testing
        experiment_id = db.store_experiment(
            name="test_experiment",
            config=None,
            results=mock_results
        )
        
        # Verify storage
        stored_results = db.get_experiment_results(experiment_id)
        assert stored_results is not None
        assert "PCA" in stored_results
        assert "JLL" in stored_results
        
        print("✅ Dashboard integration test passed!")

if __name__ == "__main__":
    test_dashboard_integration()
```

## 🎯 Best Practices

### 1. Experiment Organization
- Use consistent naming conventions for experiments
- Store results in structured directories
- Include metadata with each experiment

### 2. Performance Optimization
- Enable caching for frequently accessed data
- Use sampling for large datasets in visualizations
- Monitor memory usage for large embeddings

### 3. Data Management
- Regularly cleanup old cache files
- Archive completed experiments
- Backup important experiment results

### 4. Security Considerations
- Use authentication for remote deployments
- Limit file upload sizes
- Validate all input data

### 5. Monitoring Best Practices
- Use structured logging for debugging
- Monitor resource usage during experiments
- Set up alerts for failed experiments

This integration guide provides comprehensive examples for incorporating the OrthoReduce Dashboard into various workflows and use cases. The dashboard system is designed to be flexible and can be adapted to different research needs and deployment scenarios.