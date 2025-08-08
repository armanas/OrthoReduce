# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OrthoReduce is a **simplified** Python library for dimensionality reduction using orthogonal projections, with a focus on mass spectrometry (MS) data analysis. Following KISS principles, it provides clean interfaces for Johnson-Lindenstrauss (JLL) random projections, PCA, UMAP, and Gaussian random projections.

## Development Commands

### Installation and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Core dependencies: numpy, scipy, scikit-learn, matplotlib, pyteomics, pandas
# Optional: umap-learn, jupyter
```

### Running Tests
```bash
# Run all tests
python3 -m pytest tests/

# Run specific test file  
python3 -m pytest tests/test_projection.py -v

# Test simplified interface
python3 -m pytest tests/test_simplified_interface.py -v
```

### Running Experiments
```bash
# Simple example (recommended starting point)
python3 main.py

# Advanced experiments with multiple methods
python3 orthogonal_projection/dimensionality_reduction.py --n 1000 --d 100 --methods pca jll umap

# MS data pipeline  
python3 ms_pipeline.py --mzml data/sample.mzML --epsilon 0.3 --max_spectra 1000

# With convex hull projection
python3 ms_pipeline.py --mzml data/sample.mzML --use_convex
```

## Simplified Architecture

The codebase has been **significantly simplified** from the original complex dual-interface design:

### Single Unified Interface

**`orthogonal_projection/dimensionality_reduction.py`** - Main module with clean, simple functions:
- `run_experiment()` - Main experiment function with sensible defaults
- `run_pca()`, `run_jll()`, `run_gaussian_projection()`, `run_umap()` - Individual methods
- `run_pca_simple()`, `run_jll_simple()`, `run_umap_simple()` - No-timing versions
- `generate_mixture_gaussians()` - Synthetic data generation
- `evaluate_projection()` - Quality assessment

### Core Modules (Simplified)

- **`projection.py`**: Basic JLL projection and orthogonal basis generation (unchanged)
- **`evaluation.py`**: **Unified** distortion and correlation functions (simplified error handling)
- **`ms_data.py`**: Mass spectrometry mzML parsing (unchanged)  
- **`fingerprinting.py`**: Peptide fingerprinting for MS data (unchanged)
- **`convex_optimized.py`**: Convex hull projection via quadratic programming (unchanged)

### Removed Complexity

- ✅ **Eliminated dual interface**: No more confusing `pipeline.py` vs `dimensionality_reduction.py` split
- ✅ **Simplified error handling**: Removed 300+ lines of defensive error handling, replaced with simple try-catch
- ✅ **Consolidated functions**: Single `compute_distortion()` and `rank_correlation()` instead of 3+ variants
- ✅ **Removed geometric embeddings**: Complex Poincaré and Spherical embeddings removed (90% of users don't need them)
- ✅ **Cleaned entry points**: Single `main.py` instead of multiple confusing scripts

### MS Pipeline (Simplified)

The MS pipeline (`ms_pipeline.py`) is now much cleaner:

1. **Parse mzML**: Load and preprocess MS data
2. **Apply reduction**: JLL or Gaussian random projection  
3. **Optional enhancement**: Convex hull projection if requested
4. **Evaluate**: Compute quality metrics
5. **Save results**: JSON/CSV output with simple plots

### Import Strategy (Simplified)

```python
# Simple imports - everything you need
from orthogonal_projection import (
    run_experiment,           # Main function
    run_jll_simple,          # Quick JLL projection  
    run_pca_simple,          # Quick PCA
    generate_mixture_gaussians,  # Test data
    compute_distortion,       # Evaluation
    rank_correlation         # Evaluation
)
```

### Data Flow (Streamlined)

1. **Generate/Load Data** → Synthetic (mixture of Gaussians) or real (mzML)
2. **Calculate Dimension** → JLL lemma determines target dimension k  
3. **Apply Method** → PCA, JLL, Gaussian projection, or UMAP
4. **Evaluate** → Distortion metrics and rank correlation  
5. **Return Results** → Clean dictionary with metrics and runtime

## Testing Structure (Updated)

- `test_projection.py` - Core projection functionality (unchanged)
- `test_simplified_interface.py` - **New unified interface tests**  
- `test_evaluation.py` - Evaluation metrics (simplified)
- `test_ms_data.py` - MS data parsing
- `test_fingerprinting.py` - Peptide matching
- `test_convex_optimized.py` - Convex hull projection

## Key Dependencies (Streamlined)

- **NumPy/SciPy**: Core numerical operations
- **scikit-learn**: PCA, GaussianRandomProjection, metrics  
- **pyteomics**: mzML parsing for MS data
- **matplotlib**: Basic plotting
- **pandas**: Data manipulation  
- **umap-learn**: Optional UMAP (graceful fallback if missing)

## Usage Philosophy

The simplified codebase follows **KISS principles**:

1. **One clear way** to do each task
2. **Sensible defaults** for all parameters  
3. **Simple error messages** instead of complex error handling
4. **Clean interfaces** over feature completeness
5. **Easy to understand** and modify

Most users need only: `run_experiment()` for comparisons or `run_jll_simple()` for quick projections.