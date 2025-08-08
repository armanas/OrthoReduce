#!/bin/bash
###############################################################################
#
# OrthoReduce - Professional Dimensionality Reduction Experiment Runner
# 
# This script provides a user-friendly command-line interface for running
# OrthoReduce dimensionality reduction experiments with various methods and
# configurations.
#
# Author: OrthoReduce Team
# Version: 1.0.0
# License: See LICENSE file
#
###############################################################################

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Color definitions for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly WHITE='\033[1;37m'
readonly NC='\033[0m' # No Color

# Default values
DEFAULT_DATASET_SIZE=500
DEFAULT_DIMENSIONS=50
DEFAULT_EPSILON=0.2
DEFAULT_OUTPUT_DIR="experiment_results"
DEFAULT_METHODS="jll,pca,gaussian,pocs,poincare,spherical"

# Convex projection defaults
DEFAULT_CONVEX_K=64
DEFAULT_CONVEX_BATCH=1024
DEFAULT_CONVEX_FLOAT32=true

# Available methods
AVAILABLE_METHODS=("jll" "pca" "gaussian" "pocs" "poincare" "spherical")

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/orthoreduce.py"

# Function to print colored output
print_color() {
    local color=$1
    shift
    printf "${color}%s${NC}\n" "$*"
}

# Function to run only the optimized convex hull projection
run_convex_only() {
    local dataset_size="$1"
    local dimensions="$2"
    local output_dir="$3"
    local convex_k="$4"
    local convex_batch="$5"
    local convex_float32="$6" # true/false

    print_header
    print_info "Starting optimized convex hull projection..."
    echo

    print_color "$WHITE" "📊 Convex Projection Configuration:"
    printf "   Points (n):      %s\n" "$dataset_size"
    printf "   Dimensions (d):  %s\n" "$dimensions"
    printf "   k_candidates:    %s\n" "$convex_k"
    printf "   batch_size:      %s\n" "$convex_batch"
    printf "   float32:         %s\n" "$convex_float32"
    echo

    # Export args for Python step
    OR_N="$dataset_size" OR_D="$dimensions" OR_K="$convex_k" OR_BATCH="$convex_batch" OR_F32="$convex_float32" OR_OUTPUT_DIR="$output_dir" \
    python3 - <<'PY'
import json, os, numpy as np
from orthogonal_projection.convex_optimized import project_onto_convex_hull_qp

n = int(os.environ.get('OR_N', '1000'))
d = int(os.environ.get('OR_D', '64'))
k = int(os.environ.get('OR_K', '64'))
batch = int(os.environ.get('OR_BATCH', '1024'))
use_f32 = os.environ.get('OR_F32', 'true').lower() == 'true'
out_dir = os.environ['OR_OUTPUT_DIR']
os.makedirs(out_dir, exist_ok=True)

Y = np.random.randn(n, d).astype(np.float32 if use_f32 else np.float64)
Y_proj, alphas, V = project_onto_convex_hull_qp(
    Y,
    k_candidates=k,
    batch_size=batch,
    use_float32=use_f32,
)

summary = {
    'n': n,
    'd': d,
    'k_candidates': k,
    'batch_size': batch,
    'use_float32': use_f32,
    'Y_proj_shape': tuple(Y_proj.shape),
    'alphas_shape': tuple(alphas.shape),
    'V_shape': tuple(V.shape),
}
with open(os.path.join(out_dir, 'convex_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print('Saved:', os.path.join(out_dir, 'convex_summary.json'))
PY

    print_success "🎉 Convex projection completed successfully!"
    print_info "📁 Results saved in: $output_dir/convex_summary.json"
}
# Function to print info messages
print_info() {
    print_color "$BLUE" "ℹ️  $*"
}

# Function to print success messages
print_success() {
    print_color "$GREEN" "✅ $*"
}

# Function to print warning messages
print_warning() {
    print_color "$YELLOW" "⚠️  $*"
}

# Function to print error messages
print_error() {
    print_color "$RED" "❌ $*"
}

# Function to print header
print_header() {
    print_color "$CYAN" "================================================"
    print_color "$CYAN" "🧬 OrthoReduce Experiment Runner"
    print_color "$CYAN" "================================================"
}

# Function to show usage information
show_usage() {
    cat << EOF

Usage: $0 [OPTIONS]

A professional command-line interface for running OrthoReduce dimensionality
reduction experiments with comprehensive analysis and visualization.

OPTIONS:
    -n, --dataset-size N    Number of data points (default: $DEFAULT_DATASET_SIZE)
    -d, --dimensions D      Original dimensions (default: $DEFAULT_DIMENSIONS)
    -e, --epsilon E         JL distortion parameter (default: $DEFAULT_EPSILON)
    -m, --methods METHODS   Comma-separated list of methods
                           Available: ${AVAILABLE_METHODS[*]}
                           (default: all methods)
    -o, --output-dir DIR    Output directory (default: $DEFAULT_OUTPUT_DIR)
    --quick-test           Run small fast test (n=200, d=30)
    --full-benchmark       Run comprehensive benchmark with all methods
    
    # Optimized convex hull projection (optional fast path)
    --convex-only          Run only the optimized convex hull projection on synthetic data
    --convex-k K           Candidate vertices per point (default: $DEFAULT_CONVEX_K)
    --convex-batch B       Batch size for b = V y (default: $DEFAULT_CONVEX_BATCH)
    --convex-float32       Use float32 for speed (default)
    --no-convex-float32    Use float64 for higher precision
    -h, --help             Show this help message and exit

EXAMPLES:
    # Quick test with default parameters
    $0 --quick-test

    # Custom experiment with specific methods
    $0 --dataset-size 1000 --dimensions 100 --methods "jll,pca,pocs"

    # Full benchmark with custom output directory
    $0 --full-benchmark --output-dir my_results

    # Advanced experiment with custom epsilon
    $0 -n 800 -d 75 -e 0.15 -o advanced_experiment

METHODS:
    jll        - Johnson-Lindenstrauss Lemma based projection
    pca        - Principal Component Analysis
    gaussian   - Random Gaussian projection
    pocs       - Projection Onto Convex Sets (enhanced)
    poincare   - Poincaré disk model projection
    spherical  - Spherical projection

OUTPUT:
    The script generates comprehensive results including:
    • results.json - Full experiment data
    • results.csv - Tabular results for analysis
    • summary.txt - Human-readable summary
    • comprehensive_report.pdf - Publication-quality report
    • Interactive plots and visualizations

For more information, visit: https://github.com/AlitheaBio/OrthoReduce

EOF
}

# Function to validate methods
validate_methods() {
    local methods_str="$1"
    IFS=',' read -ra methods_array <<< "$methods_str"
    
    for method in "${methods_array[@]}"; do
        method=$(echo "$method" | xargs)  # Trim whitespace
        if [[ ! " ${AVAILABLE_METHODS[*]} " =~ " ${method} " ]]; then
            print_error "Invalid method: '$method'"
            print_info "Available methods: ${AVAILABLE_METHODS[*]}"
            exit 1
        fi
    done
}

# Function to validate numeric parameters
validate_numeric() {
    local value="$1"
    local name="$2"
    local min_val="${3:-1}"
    
    if ! [[ "$value" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        print_error "$name must be a positive number, got: '$value'"
        exit 1
    fi
    
    if (( $(echo "$value < $min_val" | bc -l) )); then
        print_error "$name must be >= $min_val, got: '$value'"
        exit 1
    fi
}

# Function to check environment
check_environment() {
    print_info "Checking environment..."
    
    # Check Python 3
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    local python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_success "Python $python_version found"
    
    # Check if we're in the right directory
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        print_error "Cannot find orthoreduce.py script at: $PYTHON_SCRIPT"
        print_info "Please run this script from the OrthoReduce root directory"
        exit 1
    fi
    
    # Check required Python packages
    print_info "Checking required packages..."
    local missing_packages=()
    
    for package in "numpy" "pandas" "matplotlib" "scipy" "sklearn"; do
        if ! python3 -c "import $package" &> /dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        print_error "Missing required packages: ${missing_packages[*]}"
        print_info "Please install with: pip install -r requirements.txt"
        exit 1
    fi
    
    print_success "All dependencies satisfied"
}

# Function to create output directory
create_output_dir() {
    local output_dir="$1"
    
    if [[ -d "$output_dir" ]]; then
        print_warning "Output directory '$output_dir' already exists"
        read -p "Do you want to continue and potentially overwrite results? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Experiment cancelled by user"
            exit 0
        fi
    else
        mkdir -p "$output_dir"
        print_success "Created output directory: $output_dir"
    fi
}

# Function to show progress indicator
show_progress() {
    local pid=$1
    local message="$2"
    
    local spinner='|/-\'
    local i=0
    
    printf "%s " "$message"
    
    while kill -0 "$pid" 2>/dev/null; do
        printf "\b${spinner:$i:1}"
        i=$(( (i+1) % 4 ))
        sleep 0.2
    done
    
    printf "\b✓\n"
}

# Function to run the experiment
run_experiment() {
    local dataset_size="$1"
    local dimensions="$2"
    local epsilon="$3"
    local methods="$4"
    local output_dir="$5"
    
    print_header
    print_info "Starting OrthoReduce experiment..."
    echo
    
    print_color "$WHITE" "📊 Experiment Configuration:"
    printf "   Dataset size:    %s points\n" "$dataset_size"
    printf "   Dimensions:      %s → auto-calculated target\n" "$dimensions"
    printf "   Epsilon:         %s (JL distortion parameter)\n" "$epsilon"
    printf "   Methods:         %s\n" "$methods"
    printf "   Output dir:      %s\n" "$output_dir"
    echo
    
    # Build command arguments for the Python script
    local python_args=(
        "--dataset-size" "$dataset_size"
        "--dimensions" "$dimensions" 
        "--epsilon" "$epsilon"
        "--methods" "$methods"
        "--output-dir" "$output_dir"
    )
    
    # Run the experiment
    print_info "🔄 Running experiment (this may take several minutes)..."
    echo
    
    # Run Python script and capture output in real-time
    if python3 "$PYTHON_SCRIPT" "${python_args[@]}"; then
        echo
        print_success "🎉 Experiment completed successfully!"
        print_info "📁 Results saved in: $output_dir/"
        
        # Show final summary if available
        if [[ -f "$output_dir/summary.txt" ]]; then
            echo
            print_color "$CYAN" "📋 Final Summary:"
            echo
            tail -n +3 "$output_dir/summary.txt" | head -n 20
            
            if [[ -f "$output_dir/plots/comprehensive_report.pdf" ]]; then
                print_success "📄 PDF Report: $output_dir/plots/comprehensive_report.pdf"
            fi
        fi
        
    else
        print_error "Experiment failed! Check the output above for details."
        exit 1
    fi
}

# Main function
main() {
    # Initialize variables
    local dataset_size="$DEFAULT_DATASET_SIZE"
    local dimensions="$DEFAULT_DIMENSIONS"
    local epsilon="$DEFAULT_EPSILON"
    local methods="$DEFAULT_METHODS"
    local output_dir="$DEFAULT_OUTPUT_DIR"
    local quick_test=false
    local full_benchmark=false
    local convex_only=false
    local convex_k="$DEFAULT_CONVEX_K"
    local convex_batch="$DEFAULT_CONVEX_BATCH"
    local convex_float32="$DEFAULT_CONVEX_FLOAT32"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--dataset-size)
                dataset_size="$2"
                validate_numeric "$dataset_size" "Dataset size" 10
                shift 2
                ;;
            -d|--dimensions)
                dimensions="$2"
                validate_numeric "$dimensions" "Dimensions" 2
                shift 2
                ;;
            -e|--epsilon)
                epsilon="$2"
                validate_numeric "$epsilon" "Epsilon" 0.01
                shift 2
                ;;
            -m|--methods)
                methods="$2"
                validate_methods "$methods"
                shift 2
                ;;
            -o|--output-dir)
                output_dir="$2"
                shift 2
                ;;
            --quick-test)
                quick_test=true
                shift
                ;;
            --full-benchmark)
                full_benchmark=true
                shift
                ;;
            --convex-only)
                convex_only=true
                shift
                ;;
            --convex-k)
                convex_k="$2"
                validate_numeric "$convex_k" "convex-k" 1
                shift 2
                ;;
            --convex-batch)
                convex_batch="$2"
                validate_numeric "$convex_batch" "convex-batch" 1
                shift 2
                ;;
            --convex-float32)
                convex_float32=true
                shift
                ;;
            --no-convex-float32)
                convex_float32=false
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Handle presets
    if [[ "$quick_test" == true ]]; then
        dataset_size=200
        dimensions=30
        epsilon=0.2
        output_dir="quick_test_results"
        print_info "🚀 Quick test mode: n=$dataset_size, d=$dimensions"
    fi
    
    if [[ "$full_benchmark" == true ]]; then
        dataset_size=1000
        dimensions=100
        epsilon=0.15
        methods="$DEFAULT_METHODS"
        output_dir="full_benchmark_results"
        print_info "🏆 Full benchmark mode: n=$dataset_size, d=$dimensions, all methods"
    fi
    
    # Validate environment
    check_environment
    
    # Create output directory
    create_output_dir "$output_dir"
    
    # Run selected mode (env and output dir already prepared above)
    if [[ "$convex_only" == true ]]; then
        run_convex_only "$dataset_size" "$dimensions" "$output_dir" "$convex_k" "$convex_batch" "$convex_float32"
    else
        run_experiment "$dataset_size" "$dimensions" "$epsilon" "$methods" "$output_dir"
    fi
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi