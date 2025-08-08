# OrthoReduce Comprehensive Monitoring System

This document describes the professional-grade progress monitoring and system utilization tracking system for OrthoReduce.

## Overview

The OrthoReduce monitoring system provides comprehensive real-time progress tracking, system resource monitoring, and performance analysis for dimensionality reduction experiments. It's designed to give users clear visibility into long-running experiments and help optimize performance.

## Features

### 🎯 Multi-Level Progress Tracking
- **Overall Experiment Progress**: Shows completion across all methods
- **Method-Specific Progress**: Detailed progress for each dimensionality reduction technique
- **Stage-by-Stage Tracking**: Data generation, method execution, evaluation, visualization
- **Real-Time ETA Calculations**: Dynamic time estimates based on current throughput

### 💻 System Resource Monitoring
- **CPU Usage**: Real-time system and process-specific CPU utilization
- **Memory Tracking**: System memory usage and process memory consumption with peak measurements
- **Disk I/O Monitoring**: Read/write throughput tracking during file operations
- **Cross-Platform Support**: Works on Windows, macOS, and Linux

### 📊 Performance Analytics
- **Throughput Calculations**: Samples processed per second for each method
- **Resource Peak Tracking**: Maximum CPU and memory usage during execution
- **Compression Metrics**: Dimensionality reduction ratios
- **Runtime Analysis**: Detailed timing for each processing stage

### 🎨 Beautiful Terminal Display
- **Professional Progress Bars**: Using tqdm for clean, informative displays
- **Color-Coded Status**: Visual indicators for different stages and completion states
- **System Stats Integration**: Real-time resource usage shown alongside progress
- **Clean Terminal Output**: Non-interfering progress tracking that doesn't clutter logs

## Usage

### Enhanced Experiment Runner

The easiest way to use monitoring is through the enhanced experiment runner:

```bash
# Enable comprehensive monitoring with system stats
python3 orthoreduce_enhanced.py --quick-test --enable-monitoring --show-system-stats

# Run full benchmark with monitoring
python3 orthoreduce_enhanced.py --full-benchmark --enable-monitoring

# Disable monitoring for minimal output
python3 orthoreduce_enhanced.py --quick-test --no-monitoring
```

### Core Function Integration

You can also enable monitoring in core dimensionality reduction functions:

```python
from orthogonal_projection.dimensionality_reduction import (
    run_experiment, run_experiment_with_visualization
)

# Enable basic monitoring
results = run_experiment(
    n=1000, d=100, 
    methods=['pca', 'jll', 'umap'],
    enable_monitoring=True,
    show_method_progress=True
)

# With visualization and monitoring
results, plots = run_experiment_with_visualization(
    n=1000, d=100,
    methods=['pca', 'jll', 'umap'],
    enable_monitoring=True,
    show_method_progress=True
)
```

### Programmatic Monitoring

For custom experiments, use the monitoring context manager:

```python
from orthogonal_projection.monitoring import experiment_monitoring

methods = ['pca', 'jll', 'umap']

with experiment_monitoring(methods, show_system_stats=True) as monitor:
    for method in methods:
        monitor.start_method(method.upper(), data_points=1000, dimensions=50)
        
        # Your method implementation here
        # monitor.update_method_progress(50, "Processing...")
        
        monitor.complete_method({'runtime': 2.5, 'accuracy': 0.95})

# Get comprehensive performance report
performance_summary = monitor.get_performance_summary()
report = format_performance_report(performance_summary)
print(report)
```

## Output Files

When monitoring is enabled, several output files are generated:

### Performance Reports
- **`performance_report.txt`**: Human-readable performance summary
- **`performance_data.json`**: Machine-readable performance metrics
- **`enhanced_results.json`**: Complete experiment results with monitoring data

### Enhanced Summaries
- **`enhanced_summary.txt`**: Comprehensive experiment summary including system info
- **`results.csv`**: Tabular results with performance metrics

## Configuration Options

### Enhanced Experiment Runner Options

| Option | Default | Description |
|--------|---------|-------------|
| `--enable-monitoring` | `True` | Enable comprehensive progress monitoring |
| `--no-monitoring` | - | Disable progress monitoring |
| `--show-system-stats` | `True` | Show real-time system resource usage |
| `--no-system-stats` | - | Hide system resource monitoring |

### Monitoring System Configuration

```python
# System monitor settings
SystemMonitor(update_interval=1.0)  # Update every second

# Progress tracker settings
ProgressTracker(
    total_methods=len(methods),
    method_names=methods,
    show_system_stats=True,
    update_interval=0.5,  # Progress bar update frequency
    ncols=100  # Terminal width
)
```

## Performance Report Format

The monitoring system generates comprehensive performance reports:

```
🎯 OrthoReduce Performance Report
==================================================

📊 Overall Statistics:
   Total Runtime: 15.42 seconds
   Methods Executed: 4
   Average Time per Method: 3.86 seconds

💻 System Resource Usage:
   Peak Memory: 8.45 GB
   Peak CPU: 87.3%
   Peak Process Memory: 892.1 MB
   Avg Disk Read: 12.3 MB/s
   Avg Disk Write: 8.7 MB/s

🚀 Method Performance:
------------------------------
   PCA:
     Runtime: 2.45s
     Throughput: 408 samples/sec
     Peak Memory: 256.3 MB
     Compression Ratio: 5.0x

   JLL:
     Runtime: 1.87s
     Throughput: 535 samples/sec
     Peak Memory: 189.7 MB
     Compression Ratio: 5.0x

   UMAP:
     Runtime: 8.93s
     Throughput: 112 samples/sec
     Peak Memory: 892.1 MB
     Compression Ratio: 5.0x
```

## Architecture

### Core Components

1. **`SystemMonitor`**: Background thread monitoring system resources
2. **`ProgressTracker`**: Multi-level progress tracking with tqdm integration
3. **`PerformanceStats`**: Data containers for method performance metrics
4. **`experiment_monitoring`**: Context manager for easy monitoring setup

### Integration Points

- **`orthoreduce_enhanced.py`**: Full integration with enhanced experiment runner
- **`dimensionality_reduction.py`**: Core function monitoring support
- **Method Functions**: Individual progress tracking for PCA, JLL, UMAP, etc.
- **Visualization System**: Monitoring for plot generation stages

### Dependencies

- **`tqdm`**: Professional progress bars and terminal output
- **`psutil`**: Cross-platform system and process monitoring
- **`threading`**: Background resource monitoring
- **`pathlib`**: File system operations

## Error Handling

The monitoring system is designed to be robust:

- **Graceful Fallbacks**: If monitoring dependencies aren't available, experiments continue without monitoring
- **Thread Safety**: All monitoring operations are thread-safe
- **Exception Handling**: Monitor failures don't crash experiments
- **Resource Cleanup**: Proper cleanup of system resources and threads

## Performance Impact

The monitoring system is designed to have minimal performance impact:

- **Background Monitoring**: System resource tracking runs in separate thread
- **Efficient Updates**: Progress bars update at reasonable intervals (0.5-1.0 seconds)
- **Optional Features**: All monitoring can be disabled for maximum performance
- **Memory Efficient**: Monitoring overhead typically <10MB

## Troubleshooting

### Common Issues

**Monitoring Not Showing**:
```bash
# Install required dependencies
pip install tqdm psutil

# Verify imports work
python3 -c "from orthogonal_projection.monitoring import SystemMonitor; print('OK')"
```

**Progress Bars Not Displaying Properly**:
- Ensure terminal supports ANSI escape codes
- Try running with `--no-monitoring` if terminal compatibility issues occur
- Use `TERM=xterm-256color` environment variable if needed

**System Stats Not Available**:
- Some systems may not support all monitoring features
- Disk I/O monitoring may not be available on some platforms
- The system gracefully handles unavailable features

## Future Enhancements

Planned improvements for the monitoring system:

- **Web-Based Dashboard**: Real-time monitoring via web interface
- **Historical Analysis**: Trend tracking across multiple experiment runs  
- **Resource Optimization**: Automatic suggestions for performance improvements
- **Custom Metrics**: User-defined performance indicators
- **Distributed Monitoring**: Support for multi-machine experiments

## Contributing

To contribute to the monitoring system:

1. **Add New Metrics**: Extend `PerformanceStats` and `SystemMetrics` classes
2. **Improve Visualization**: Enhance progress bar displays and reports
3. **Platform Support**: Add support for additional operating systems
4. **Integration**: Connect monitoring to new experiment types

See the main README for general contribution guidelines.