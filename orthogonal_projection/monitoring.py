"""
monitoring.py - Comprehensive Progress and System Monitoring for OrthoReduce

This module provides professional-grade monitoring capabilities including:
- Multi-level progress bars with ETA calculations
- Real-time system resource monitoring (CPU, memory, disk I/O)
- Stage-by-stage progress reporting with performance metrics
- Beautiful terminal-based displays using tqdm
- Integration with existing OrthoReduce functionality

Author: OrthoReduce Team
Version: 1.0.0
"""

from __future__ import annotations

import gc
import logging
import os
import platform
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Event
from typing import Dict, List, Optional, Union, Any, Callable

import numpy as np
import psutil
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Container for system resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_used_gb: float
    memory_percent: float
    memory_available_gb: float
    disk_read_mb_s: float = 0.0
    disk_write_mb_s: float = 0.0
    process_memory_mb: float = 0.0
    process_cpu_percent: float = 0.0


@dataclass
class PerformanceStats:
    """Container for method performance statistics."""
    method: str
    start_time: float
    end_time: Optional[float] = None
    peak_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    samples_processed: int = 0
    data_points: int = 0
    dimensions: int = 0
    compression_ratio: float = 0.0
    
    @property
    def runtime(self) -> float:
        """Calculate runtime in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def throughput(self) -> float:
        """Calculate throughput in samples/second."""
        runtime = self.runtime
        if runtime > 0:
            return self.samples_processed / runtime
        return 0.0


class SystemMonitor:
    """
    Real-time system resource monitoring with thread-safe data collection.
    
    This class continuously monitors CPU usage, memory consumption, and disk I/O
    in a background thread, providing real-time system statistics.
    """
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize system monitor.
        
        Args:
            update_interval: Time between metric updates in seconds
        """
        self.update_interval = update_interval
        self._running = False
        self._thread = None
        self._metrics_lock = Lock()
        self._stop_event = Event()
        
        # Metrics storage
        self._current_metrics: Optional[SystemMetrics] = None
        self._metrics_history: List[SystemMetrics] = []
        self._max_history = 1000  # Keep last 1000 measurements
        
        # Process monitoring
        self._process = psutil.Process()
        self._initial_io_counters = None
        
        # Initialize baseline I/O counters
        try:
            self._initial_io_counters = psutil.disk_io_counters()
            self._last_io_counters = self._initial_io_counters
            self._last_io_time = time.time()
        except (AttributeError, OSError):
            logger.warning("Disk I/O monitoring not available on this system")
            self._initial_io_counters = None
    
    def start(self) -> None:
        """Start monitoring in background thread."""
        if self._running:
            return
            
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.debug("System monitor started")
    
    def stop(self) -> None:
        """Stop monitoring."""
        if not self._running:
            return
            
        self._running = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        
        logger.debug("System monitor stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        while self._running and not self._stop_event.is_set():
            try:
                metrics = self._collect_metrics()
                
                with self._metrics_lock:
                    self._current_metrics = metrics
                    self._metrics_history.append(metrics)
                    
                    # Maintain history size limit
                    if len(self._metrics_history) > self._max_history:
                        self._metrics_history = self._metrics_history[-self._max_history:]
                
            except Exception as e:
                logger.warning(f"Error collecting system metrics: {e}")
            
            self._stop_event.wait(self.update_interval)
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        timestamp = time.time()
        
        # System-wide metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # Disk I/O metrics (if available)
        disk_read_mb_s = 0.0
        disk_write_mb_s = 0.0
        
        if self._initial_io_counters is not None:
            try:
                current_io = psutil.disk_io_counters()
                if current_io and self._last_io_counters:
                    time_delta = timestamp - self._last_io_time
                    if time_delta > 0:
                        read_delta = current_io.read_bytes - self._last_io_counters.read_bytes
                        write_delta = current_io.write_bytes - self._last_io_counters.write_bytes
                        
                        disk_read_mb_s = (read_delta / time_delta) / (1024 * 1024)
                        disk_write_mb_s = (write_delta / time_delta) / (1024 * 1024)
                    
                    self._last_io_counters = current_io
                    self._last_io_time = timestamp
            except (AttributeError, OSError):
                pass  # I/O monitoring not available
        
        # Process-specific metrics
        process_memory_mb = 0.0
        process_cpu_percent = 0.0
        
        try:
            process_memory_mb = self._process.memory_info().rss / (1024 * 1024)
            process_cpu_percent = self._process.cpu_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass  # Process might have ended or access denied
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_used_gb=(memory.total - memory.available) / (1024**3),
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            disk_read_mb_s=disk_read_mb_s,
            disk_write_mb_s=disk_write_mb_s,
            process_memory_mb=process_memory_mb,
            process_cpu_percent=process_cpu_percent
        )
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        with self._metrics_lock:
            return self._current_metrics
    
    def get_peak_metrics(self) -> Dict[str, float]:
        """Get peak resource usage during monitoring period."""
        with self._metrics_lock:
            if not self._metrics_history:
                return {}
            
            peak_memory = max(m.memory_used_gb for m in self._metrics_history)
            peak_cpu = max(m.cpu_percent for m in self._metrics_history)
            peak_process_memory = max(m.process_memory_mb for m in self._metrics_history)
            peak_process_cpu = max(m.process_cpu_percent for m in self._metrics_history)
            avg_disk_read = np.mean([m.disk_read_mb_s for m in self._metrics_history])
            avg_disk_write = np.mean([m.disk_write_mb_s for m in self._metrics_history])
            
            return {
                'peak_memory_gb': peak_memory,
                'peak_cpu_percent': peak_cpu,
                'peak_process_memory_mb': peak_process_memory,
                'peak_process_cpu_percent': peak_process_cpu,
                'avg_disk_read_mb_s': avg_disk_read,
                'avg_disk_write_mb_s': avg_disk_write,
                'num_measurements': len(self._metrics_history)
            }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class ProgressTracker:
    """
    Multi-level progress tracking with beautiful terminal displays.
    
    This class provides hierarchical progress tracking with:
    - Overall experiment progress
    - Method-specific progress bars
    - Real-time ETA calculations
    - Integration with system monitoring
    """
    
    def __init__(self, 
                 total_methods: int,
                 method_names: List[str] = None,
                 show_system_stats: bool = True,
                 update_interval: float = 0.5,
                 ncols: int = 100):
        """
        Initialize progress tracker.
        
        Args:
            total_methods: Total number of methods to process
            method_names: Names of methods (for display)
            show_system_stats: Whether to show system resource usage
            update_interval: Progress bar update interval
            ncols: Terminal width for progress bars
        """
        self.total_methods = total_methods
        self.method_names = method_names or []
        self.show_system_stats = show_system_stats
        self.update_interval = update_interval
        self.ncols = ncols
        
        # Progress tracking
        self.completed_methods = 0
        self.current_method = ""
        self.overall_start_time = time.time()
        
        # Progress bars
        self._overall_pbar: Optional[tqdm] = None
        self._method_pbar: Optional[tqdm] = None
        self._stats_pbar: Optional[tqdm] = None
        
        # Performance tracking
        self.performance_stats: List[PerformanceStats] = []
        self.current_stats: Optional[PerformanceStats] = None
        
        # System monitoring
        self.system_monitor: Optional[SystemMonitor] = None
        if show_system_stats:
            self.system_monitor = SystemMonitor(update_interval=1.0)
        
        # Thread safety
        self._pbar_lock = Lock()
    
    def start(self) -> None:
        """Start progress tracking with system monitoring."""
        # Start system monitoring
        if self.system_monitor:
            self.system_monitor.start()
        
        # Initialize overall progress bar
        with self._pbar_lock:
            self._overall_pbar = tqdm(
                total=self.total_methods,
                desc="🔬 OrthoReduce Experiment",
                unit="method",
                ncols=self.ncols,
                position=0,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]"
            )
            
            # System stats bar (if enabled)
            if self.show_system_stats:
                self._stats_pbar = tqdm(
                    total=100,
                    desc="💻 System Resources",
                    unit="%",
                    ncols=self.ncols,
                    position=2,
                    bar_format="{desc}: CPU {postfix}",
                    leave=False
                )
        
        logger.info("Progress tracking started")
    
    def start_method(self, method_name: str, data_points: int = 0, dimensions: int = 0) -> None:
        """Start tracking a new method."""
        self.current_method = method_name
        
        # Finalize previous method if exists
        if self.current_stats is not None:
            self.current_stats.end_time = time.time()
            self.performance_stats.append(self.current_stats)
        
        # Start new performance tracking
        self.current_stats = PerformanceStats(
            method=method_name,
            start_time=time.time(),
            data_points=data_points,
            dimensions=dimensions
        )
        
        with self._pbar_lock:
            # Update method progress bar
            if self._method_pbar is not None:
                self._method_pbar.close()
            
            self._method_pbar = tqdm(
                total=100,
                desc=f"🚀 {method_name}",
                unit="%",
                ncols=self.ncols,
                position=1,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]",
                leave=False
            )
        
        logger.debug(f"Started method: {method_name}")
    
    def update_method_progress(self, progress: float, status: str = "") -> None:
        """Update current method progress."""
        if self._method_pbar is None:
            return
        
        with self._pbar_lock:
            # Update progress bar
            target_progress = min(max(progress, 0), 100)
            current_progress = self._method_pbar.n
            
            if target_progress > current_progress:
                self._method_pbar.update(target_progress - current_progress)
            
            # Update description with status
            if status:
                desc = f"🚀 {self.current_method} - {status}"
                self._method_pbar.set_description(desc)
        
        # Update performance stats
        if self.current_stats:
            self.current_stats.samples_processed = int(progress)
    
    def complete_method(self, metrics: Dict[str, Any] = None) -> None:
        """Complete current method and update overall progress."""
        # Finalize current method
        if self.current_stats:
            self.current_stats.end_time = time.time()
            
            # Update with final metrics
            if metrics:
                if 'compression_ratio' in metrics:
                    self.current_stats.compression_ratio = metrics['compression_ratio']
            
            # Get peak resource usage if monitoring enabled
            if self.system_monitor:
                peak_metrics = self.system_monitor.get_peak_metrics()
                if peak_metrics:
                    self.current_stats.peak_memory_mb = peak_metrics.get('peak_process_memory_mb', 0)
                    self.current_stats.peak_cpu_percent = peak_metrics.get('peak_process_cpu_percent', 0)
            
            self.performance_stats.append(self.current_stats)
            self.current_stats = None
        
        # Update overall progress
        self.completed_methods += 1
        
        with self._pbar_lock:
            if self._overall_pbar:
                self._overall_pbar.update(1)
                
                # Update description with current method completion
                total_elapsed = time.time() - self.overall_start_time
                methods_per_second = self.completed_methods / total_elapsed if total_elapsed > 0 else 0
                
                if self.completed_methods < self.total_methods:
                    remaining_methods = self.total_methods - self.completed_methods
                    eta_seconds = remaining_methods / methods_per_second if methods_per_second > 0 else 0
                    self._overall_pbar.set_postfix_str(f"ETA: {eta_seconds:.0f}s")
            
            # Close method progress bar
            if self._method_pbar:
                self._method_pbar.close()
                self._method_pbar = None
        
        logger.debug(f"Completed method: {self.current_method}")
    
    def update_system_stats(self) -> None:
        """Update system resource statistics display."""
        if not self.show_system_stats or not self.system_monitor or not self._stats_pbar:
            return
        
        metrics = self.system_monitor.get_current_metrics()
        if not metrics:
            return
        
        with self._pbar_lock:
            # Update CPU usage as progress
            cpu_progress = min(max(metrics.cpu_percent, 0), 100)
            current_cpu = self._stats_pbar.n
            
            if cpu_progress != current_cpu:
                self._stats_pbar.n = cpu_progress
                self._stats_pbar.refresh()
            
            # Update postfix with detailed stats
            postfix = (
                f"{metrics.cpu_percent:.1f}% | "
                f"MEM: {metrics.memory_used_gb:.1f}GB ({metrics.memory_percent:.1f}%) | "
                f"PROC: {metrics.process_memory_mb:.0f}MB"
            )
            
            if metrics.disk_read_mb_s > 0.1 or metrics.disk_write_mb_s > 0.1:
                postfix += f" | I/O: R{metrics.disk_read_mb_s:.1f}/W{metrics.disk_write_mb_s:.1f} MB/s"
            
            self._stats_pbar.set_postfix_str(postfix)
    
    def finish(self) -> None:
        """Finish progress tracking and cleanup."""
        # Finalize any remaining method
        if self.current_stats:
            self.current_stats.end_time = time.time()
            self.performance_stats.append(self.current_stats)
            self.current_stats = None
        
        # Close progress bars
        with self._pbar_lock:
            if self._method_pbar:
                self._method_pbar.close()
                self._method_pbar = None
            
            if self._stats_pbar:
                self._stats_pbar.close()
                self._stats_pbar = None
            
            if self._overall_pbar:
                self._overall_pbar.close()
                self._overall_pbar = None
        
        # Stop system monitoring
        if self.system_monitor:
            self.system_monitor.stop()
        
        logger.info("Progress tracking completed")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.performance_stats:
            return {}
        
        total_runtime = sum(stat.runtime for stat in self.performance_stats)
        
        # Method-specific stats
        method_stats = {}
        for stat in self.performance_stats:
            method_stats[stat.method] = {
                'runtime_seconds': stat.runtime,
                'throughput_samples_per_sec': stat.throughput,
                'peak_memory_mb': stat.peak_memory_mb,
                'peak_cpu_percent': stat.peak_cpu_percent,
                'compression_ratio': stat.compression_ratio,
                'data_points': stat.data_points,
                'dimensions': stat.dimensions
            }
        
        # Overall system stats
        overall_stats = {
            'total_runtime_seconds': total_runtime,
            'total_methods': len(self.performance_stats),
            'avg_runtime_per_method': total_runtime / len(self.performance_stats),
            'method_performance': method_stats
        }
        
        # Add peak system metrics if available
        if self.system_monitor:
            peak_metrics = self.system_monitor.get_peak_metrics()
            overall_stats['system_peak_metrics'] = peak_metrics
        
        return overall_stats
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


@contextmanager
def experiment_monitoring(methods: List[str], 
                         show_system_stats: bool = True,
                         show_progress_bars: bool = True):
    """
    Context manager for comprehensive experiment monitoring.
    
    Usage:
        with experiment_monitoring(['pca', 'jll', 'umap']) as monitor:
            monitor.start_method('pca', data_points=1000, dimensions=100)
            # ... run PCA ...
            monitor.complete_method({'compression_ratio': 2.0})
    
    Args:
        methods: List of method names to track
        show_system_stats: Whether to show system resource monitoring
        show_progress_bars: Whether to show progress bars
    
    Yields:
        ProgressTracker: Configured progress tracker
    """
    if not show_progress_bars:
        # Create a minimal dummy tracker
        class DummyTracker:
            def start_method(self, *args, **kwargs): pass
            def update_method_progress(self, *args, **kwargs): pass
            def complete_method(self, *args, **kwargs): pass
            def update_system_stats(self): pass
            def get_performance_summary(self): return {}
        
        yield DummyTracker()
        return
    
    tracker = ProgressTracker(
        total_methods=len(methods),
        method_names=methods,
        show_system_stats=show_system_stats
    )
    
    try:
        tracker.start()
        yield tracker
    finally:
        tracker.finish()


def format_performance_report(performance_summary: Dict[str, Any]) -> str:
    """
    Format performance summary into a human-readable report.
    
    Args:
        performance_summary: Performance data from ProgressTracker
        
    Returns:
        Formatted performance report string
    """
    if not performance_summary:
        return "No performance data available."
    
    lines = []
    lines.append("🎯 OrthoReduce Performance Report")
    lines.append("=" * 50)
    lines.append("")
    
    # Overall statistics
    total_runtime = performance_summary.get('total_runtime_seconds', 0)
    total_methods = performance_summary.get('total_methods', 0)
    avg_runtime = performance_summary.get('avg_runtime_per_method', 0)
    
    lines.append(f"📊 Overall Statistics:")
    lines.append(f"   Total Runtime: {total_runtime:.2f} seconds")
    lines.append(f"   Methods Executed: {total_methods}")
    lines.append(f"   Average Time per Method: {avg_runtime:.2f} seconds")
    lines.append("")
    
    # System peak metrics
    if 'system_peak_metrics' in performance_summary:
        peak = performance_summary['system_peak_metrics']
        lines.append(f"💻 System Resource Usage:")
        lines.append(f"   Peak Memory: {peak.get('peak_memory_gb', 0):.2f} GB")
        lines.append(f"   Peak CPU: {peak.get('peak_cpu_percent', 0):.1f}%")
        lines.append(f"   Peak Process Memory: {peak.get('peak_process_memory_mb', 0):.1f} MB")
        
        if peak.get('avg_disk_read_mb_s', 0) > 0.1:
            lines.append(f"   Avg Disk Read: {peak.get('avg_disk_read_mb_s', 0):.1f} MB/s")
            lines.append(f"   Avg Disk Write: {peak.get('avg_disk_write_mb_s', 0):.1f} MB/s")
        lines.append("")
    
    # Method-specific performance
    method_perf = performance_summary.get('method_performance', {})
    if method_perf:
        lines.append(f"🚀 Method Performance:")
        lines.append("-" * 30)
        
        for method, stats in method_perf.items():
            runtime = stats.get('runtime_seconds', 0)
            throughput = stats.get('throughput_samples_per_sec', 0)
            peak_mem = stats.get('peak_memory_mb', 0)
            compression = stats.get('compression_ratio', 0)
            
            lines.append(f"   {method.upper()}:")
            lines.append(f"     Runtime: {runtime:.2f}s")
            if throughput > 0:
                lines.append(f"     Throughput: {throughput:.0f} samples/sec")
            if peak_mem > 0:
                lines.append(f"     Peak Memory: {peak_mem:.1f} MB")
            if compression > 0:
                lines.append(f"     Compression Ratio: {compression:.1f}x")
            lines.append("")
    
    return "\n".join(lines)


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information for debugging and optimization."""
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'cpu_count_physical': psutil.cpu_count(logical=False),
    }
    
    # Memory information
    memory = psutil.virtual_memory()
    info.update({
        'total_memory_gb': memory.total / (1024**3),
        'available_memory_gb': memory.available / (1024**3),
        'memory_percent_used': memory.percent
    })
    
    # Disk information
    try:
        disk = psutil.disk_usage('/')
        info.update({
            'disk_total_gb': disk.total / (1024**3),
            'disk_free_gb': disk.free / (1024**3),
            'disk_percent_used': (disk.used / disk.total) * 100
        })
    except (OSError, AttributeError):
        pass  # Disk info not available
    
    return info


def print_system_info() -> None:
    """Print formatted system information."""
    info = get_system_info()
    
    print("🖥️  System Information")
    print("=" * 30)
    print(f"Platform: {info.get('platform', 'Unknown')}")
    print(f"Processor: {info.get('processor', 'Unknown')}")
    print(f"Python: {info.get('python_version', 'Unknown')}")
    print(f"CPU Cores: {info.get('cpu_count_physical', '?')} physical, {info.get('cpu_count_logical', '?')} logical")
    print(f"Memory: {info.get('total_memory_gb', 0):.1f} GB total, {info.get('available_memory_gb', 0):.1f} GB available")
    
    if 'disk_total_gb' in info:
        print(f"Disk: {info['disk_total_gb']:.1f} GB total, {info['disk_free_gb']:.1f} GB free")
    
    print()


# Memory optimization utilities
def optimize_memory() -> None:
    """Force garbage collection to optimize memory usage."""
    gc.collect()


def get_memory_usage() -> float:
    """Get current process memory usage in MB."""
    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


# Convenience function for quick monitoring setup
def create_simple_monitor(method_name: str, show_stats: bool = True) -> tqdm:
    """
    Create a simple progress bar for a single method.
    
    Args:
        method_name: Name of the method being monitored
        show_stats: Whether to show system stats
        
    Returns:
        tqdm progress bar instance
    """
    desc = f"🚀 {method_name}"
    if show_stats:
        current_memory = get_memory_usage()
        desc += f" | MEM: {current_memory:.0f}MB"
    
    return tqdm(
        total=100,
        desc=desc,
        unit="%",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]"
    )