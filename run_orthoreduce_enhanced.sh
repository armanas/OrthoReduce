#!/bin/bash
###############################################################################
#
# OrthoReduce Enhanced - Professional Dimensionality Reduction with Advanced Features
# 
# This enhanced script provides access to all the latest OrthoReduce improvements:
# • Advanced visualization system with publication-ready plots
# • Comprehensive evaluation metrics (trustworthiness, continuity)  
# • Post-processing calibration (isotonic regression, Procrustes)
# • Enhanced convex hull projection with ridge regularization
# • Specialized spherical and hyperbolic embedding visualizations
# • Interactive dashboard capabilities
# • Experiment orchestration and staged pipelines
#
# Author: OrthoReduce Team
# Version: 2.0.0 (Enhanced)
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

# Enhanced default values
DEFAULT_DATASET_SIZE=500
DEFAULT_DIMENSIONS=50
DEFAULT_EPSILON=0.2
DEFAULT_OUTPUT_DIR="enhanced_experiment_results"
DEFAULT_METHODS="jll,pca,gaussian,pocs"
DEFAULT_ENHANCED_METHODS="jll,pca,gaussian,pocs,poincare,spherical"

# Available methods (expanded)
AVAILABLE_METHODS=("jll" "pca" "gaussian" "pocs" "poincare" "spherical" "umap")

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/orthoreduce_enhanced.py"

# Function to print colored output
print_color() {
    local color=$1
    shift
    printf "${color}%s${NC}\n" "$*"
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
    print_color "$CYAN" "🚀 OrthoReduce Enhanced Experiment Runner"
    print_color "$CYAN" "================================================"
}

# Function to show usage information
show_usage() {
    cat << EOF

Usage: $0 [OPTIONS]

An enhanced command-line interface for running OrthoReduce dimensionality
reduction experiments with advanced features and comprehensive analysis.

🎯 CORE OPTIONS:
    -n, --dataset-size N    Number of data points (default: $DEFAULT_DATASET_SIZE)
    -d, --dimensions D      Original dimensions (default: $DEFAULT_DIMENSIONS)
    -e, --epsilon E         JL distortion parameter (default: $DEFAULT_EPSILON)
    -m, --methods METHODS   Comma-separated list of methods
                           Available: ${AVAILABLE_METHODS[*]}
                           (default: $DEFAULT_METHODS)
    -o, --output-dir DIR    Output directory (default: $DEFAULT_OUTPUT_DIR)

🎨 ENHANCED FEATURES:
    --advanced-plots       Use publication-ready visualization system (default)
    --no-advanced-plots    Disable advanced plotting
    --interactive          Create interactive HTML plots with Plotly
    --comprehensive-eval   Include trustworthiness/continuity metrics (default)
    --no-comprehensive-eval Disable comprehensive evaluation
    --calibration          Apply post-processing calibration (default)
    --no-calibration       Disable calibration
    --staged-pipeline      Use orchestrated multi-stage optimization
    --dashboard            Launch interactive web dashboard after experiment

🚀 CONVENIENCE PRESETS:
    --quick-test           Fast test: n=200, d=30, basic methods
    --full-benchmark       Comprehensive: n=1000, d=100, all methods, all features
    --demo-advanced        Demo advanced features: spherical, hyperbolic, interactive
    --publication-ready    High-quality plots for papers: 300dpi, calibration
    
⚡ SPECIAL MODES:
    --dashboard-only       Launch dashboard without running experiment
    --convex-enhanced      Test enhanced convex hull with ridge regularization
    --geometric-analysis   Focus on spherical/hyperbolic geometry visualizations
    
    -h, --help             Show this help message and exit

🌟 EXAMPLES:

    # Quick test with enhanced features
    $0 --quick-test --interactive

    # Full benchmark with all advanced features  
    $0 --full-benchmark

    # Custom experiment with specific enhancements
    $0 --dataset-size 800 --methods "jll,pca,pocs,poincare" --advanced-plots --dashboard

    # Publication-ready experiment
    $0 --publication-ready --methods "jll,pca,gaussian" --output-dir paper_results

    # Interactive geometric analysis
    $0 --geometric-analysis --interactive --dashboard

    # Just launch the dashboard
    $0 --dashboard-only

📊 METHODS AVAILABLE:
    jll        - Johnson-Lindenstrauss Lemma projection
    pca        - Principal Component Analysis
    gaussian   - Random Gaussian projection
    pocs       - Enhanced Projection Onto Convex Sets (with ridge regularization)
    poincare   - Poincaré disk hyperbolic embedding (with curvature tuning)
    spherical  - Spherical embedding (with Riemannian optimization)
    umap       - Uniform Manifold Approximation and Projection

📈 ENHANCED OUTPUT:
    The script generates comprehensive results including:
    • enhanced_results.json - Complete experiment data with advanced metrics
    • results.csv - Tabular results for analysis
    • enhanced_summary.txt - Detailed human-readable summary
    • Publication-ready plots (PNG, SVG, PDF formats)
    • Interactive HTML visualizations (with --interactive)
    • Web dashboard for exploration (with --dashboard)

🔗 For more information: https://github.com/AlitheaBio/OrthoReduce

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

# Function to check enhanced environment
check_enhanced_environment() {
    print_info "Checking enhanced environment..."
    
    # Check Python 3
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    local python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_success "Python $python_version found"
    
    # Check if we're in the right directory
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        print_error "Cannot find enhanced script at: $PYTHON_SCRIPT"
        print_info "Please run this script from the OrthoReduce root directory"
        exit 1
    fi
    
    # Check required Python packages (enhanced list)
    print_info "Checking enhanced dependencies..."
    local missing_packages=()
    
    for package in "numpy" "pandas" "matplotlib" "scipy" "sklearn"; do
        if ! python3 -c "import $package" &> /dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    # Check optional enhanced packages
    local optional_packages=()
    for package in "plotly" "streamlit" "umap"; do
        if ! python3 -c "import $package" &> /dev/null; then
            optional_packages+=("$package")
        fi
    done
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        print_error "Missing required packages: ${missing_packages[*]}"
        print_info "Please install with: pip install -r requirements.txt"
        exit 1
    fi
    
    if [[ ${#optional_packages[@]} -gt 0 ]]; then
        print_warning "Optional packages missing: ${optional_packages[*]}"
        print_info "Some enhanced features may be unavailable"
        print_info "Install with: pip install ${optional_packages[*]}"
    fi
    
    print_success "Enhanced dependencies satisfied"
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

# Function to launch dashboard only
launch_dashboard_only() {
    local results_dir="${1:-enhanced_experiment_results}"
    
    print_header
    print_info "Launching interactive dashboard..."
    
    if [[ ! -f "launch_dashboard.py" ]]; then
        print_error "Dashboard launcher not found: launch_dashboard.py"
        print_info "Please ensure you have the complete OrthoReduce Enhanced installation"
        exit 1
    fi
    
    print_info "🌐 Starting dashboard server..."
    print_info "📁 Results directory: $results_dir"
    
    python3 launch_dashboard.py --results-dir "$results_dir" --auto-refresh
}

# Function to run enhanced experiment
run_enhanced_experiment() {
    local dataset_size="$1"
    local dimensions="$2"
    local epsilon="$3"
    local methods="$4"
    local output_dir="$5"
    shift 5  # Remove first 5 arguments, rest are feature flags
    local feature_flags=("$@")
    
    print_header
    print_info "Starting OrthoReduce Enhanced experiment..."
    echo
    
    print_color "$WHITE" "📊 Enhanced Experiment Configuration:"
    printf "   Dataset size:      %s points\n" "$dataset_size"
    printf "   Dimensions:        %s → auto-calculated target\n" "$dimensions"
    printf "   Epsilon:           %s (JL distortion parameter)\n" "$epsilon"
    printf "   Methods:           %s\n" "$methods"
    printf "   Output dir:        %s\n" "$output_dir"
    printf "   Enhanced features: %s\n" "${feature_flags[*]}"
    echo
    
    # Build command arguments for the enhanced Python script
    local python_args=(
        "--dataset-size" "$dataset_size"
        "--dimensions" "$dimensions" 
        "--epsilon" "$epsilon"
        "--methods" "$methods"
        "--output-dir" "$output_dir"
    )
    
    # Add feature flags
    python_args+=("${feature_flags[@]}")
    
    # Run the enhanced experiment
    print_info "🔄 Running enhanced experiment (this may take several minutes)..."
    echo
    
    # Run enhanced Python script with real-time output
    if python3 "$PYTHON_SCRIPT" "${python_args[@]}"; then
        echo
        print_success "🎉 Enhanced experiment completed successfully!"
        print_info "📁 Results saved in: $output_dir/"
        
        # Show enhanced summary if available
        if [[ -f "$output_dir/enhanced_summary.txt" ]]; then
            echo
            print_color "$CYAN" "📋 Enhanced Summary:"
            echo
            head -n 25 "$output_dir/enhanced_summary.txt" | tail -n +3
        fi
        
        # List generated visualizations
        local plot_count=$(find "$output_dir" -name "*.png" -o -name "*.html" -o -name "*.svg" 2>/dev/null | wc -l)
        if [[ $plot_count -gt 0 ]]; then
            print_success "📊 Generated $plot_count advanced visualizations"
        fi
        
    else
        print_error "Enhanced experiment failed! Check the output above for details."
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
    local feature_flags=()
    
    # Feature flags (defaults match enhanced Python script)
    local advanced_plots=true
    local interactive=false
    local comprehensive_eval=true
    local calibration=true
    local staged_pipeline=false
    local dashboard=false
    local dashboard_only=false
    
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
            --advanced-plots)
                advanced_plots=true
                shift
                ;;
            --no-advanced-plots)
                advanced_plots=false
                shift
                ;;
            --interactive)
                interactive=true
                shift
                ;;
            --comprehensive-eval)
                comprehensive_eval=true
                shift
                ;;
            --no-comprehensive-eval)
                comprehensive_eval=false
                shift
                ;;
            --calibration)
                calibration=true
                shift
                ;;
            --no-calibration)
                calibration=false
                shift
                ;;
            --staged-pipeline)
                staged_pipeline=true
                shift
                ;;
            --dashboard)
                dashboard=true
                shift
                ;;
            --dashboard-only)
                dashboard_only=true
                shift
                ;;
            --quick-test)
                dataset_size=200
                dimensions=30
                methods="jll,pca"
                output_dir="quick_test_enhanced"
                print_info "🚀 Quick test mode enabled"
                shift
                ;;
            --full-benchmark)
                dataset_size=1000
                dimensions=100
                methods="$DEFAULT_ENHANCED_METHODS"
                advanced_plots=true
                interactive=true
                comprehensive_eval=true
                calibration=true
                dashboard=true
                output_dir="full_benchmark_enhanced"
                print_info "🏆 Full benchmark mode enabled"
                shift
                ;;
            --demo-advanced)
                dataset_size=400
                dimensions=40
                methods="pca,poincare,spherical"
                advanced_plots=true
                interactive=true
                comprehensive_eval=true
                output_dir="demo_advanced_enhanced"
                print_info "🎨 Advanced demo mode enabled"
                shift
                ;;
            --publication-ready)
                advanced_plots=true
                calibration=true
                comprehensive_eval=true
                output_dir="publication_results_enhanced"
                print_info "📄 Publication-ready mode enabled"
                shift
                ;;
            --convex-enhanced)
                methods="pocs"
                dataset_size=500
                dimensions=64
                advanced_plots=true
                output_dir="convex_enhanced_test"
                print_info "⚡ Enhanced convex hull test mode"
                shift
                ;;
            --geometric-analysis)
                methods="pca,spherical,poincare"
                advanced_plots=true
                interactive=true
                output_dir="geometric_analysis_enhanced"
                print_info "🔮 Geometric analysis mode enabled"
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
    
    # Build feature flags for Python script
    if [[ "$advanced_plots" == true ]]; then
        feature_flags+=("--advanced-plots")
    else
        feature_flags+=("--no-advanced-plots")
    fi
    
    if [[ "$interactive" == true ]]; then
        feature_flags+=("--interactive")
    fi
    
    if [[ "$comprehensive_eval" == true ]]; then
        feature_flags+=("--comprehensive-eval")
    else
        feature_flags+=("--no-comprehensive-eval")
    fi
    
    if [[ "$calibration" == true ]]; then
        feature_flags+=("--calibration")
    else
        feature_flags+=("--no-calibration")
    fi
    
    if [[ "$staged_pipeline" == true ]]; then
        feature_flags+=("--staged-pipeline")
    fi
    
    if [[ "$dashboard" == true ]]; then
        feature_flags+=("--dashboard")
    fi
    
    # Handle dashboard-only mode
    if [[ "$dashboard_only" == true ]]; then
        launch_dashboard_only "$output_dir"
        return
    fi
    
    # Validate environment
    check_enhanced_environment
    
    # Create output directory
    create_output_dir "$output_dir"
    
    # Run enhanced experiment
    run_enhanced_experiment "$dataset_size" "$dimensions" "$epsilon" "$methods" "$output_dir" "${feature_flags[@]}"
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi