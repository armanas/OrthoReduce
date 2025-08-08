"""
Comprehensive Logging and Progress Tracking System

This module provides advanced logging, progress tracking, and experiment
monitoring capabilities for the staged optimization pipeline. It includes
structured logging, real-time progress bars, experiment telemetry, and
comprehensive audit trails.
"""

import logging
import json
import time
import threading
import queue
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from contextlib import contextmanager
import sys
import os
from collections import deque, defaultdict

# Optional dependencies for enhanced features
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class LogEntry:
    """Structured log entry with metadata."""
    timestamp: str
    level: str
    logger_name: str
    message: str
    stage: Optional[str] = None
    experiment_id: Optional[str] = None
    run_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ProgressState:
    """Progress tracking state."""
    current: int = 0
    total: int = 100
    description: str = ""
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    rate: float = 0.0
    eta: Optional[float] = None
    completed: bool = False
    
    def update(self, increment: int = 1, description: str = None):
        """Update progress state."""
        now = time.time()
        self.current += increment
        if description:
            self.description = description
        
        # Calculate rate and ETA
        elapsed = now - self.start_time
        if elapsed > 0:
            self.rate = self.current / elapsed
            if self.rate > 0 and self.current < self.total:
                remaining = self.total - self.current
                self.eta = remaining / self.rate
        
        self.last_update = now
        
        if self.current >= self.total:
            self.completed = True
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get progress information."""
        elapsed = time.time() - self.start_time
        percentage = (self.current / self.total * 100) if self.total > 0 else 0
        
        return {
            "current": self.current,
            "total": self.total,
            "percentage": percentage,
            "description": self.description,
            "elapsed": elapsed,
            "rate": self.rate,
            "eta": self.eta,
            "completed": self.completed
        }


class StructuredLogger:
    """Enhanced logger with structured logging capabilities."""
    
    def __init__(self, name: str, level: int = logging.INFO,
                 log_file: Optional[Path] = None,
                 json_log_file: Optional[Path] = None,
                 experiment_id: Optional[str] = None):
        """
        Initialize structured logger.
        
        Parameters
        ----------
        name : str
            Logger name
        level : int
            Logging level
        log_file : Path, optional
            Text log file path
        json_log_file : Path, optional
            JSON log file path for structured logs
        experiment_id : str, optional
            Experiment identifier for tracking
        """
        self.name = name
        self.experiment_id = experiment_id
        self.current_stage = None
        self.current_run_id = None
        
        # Create standard logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(stage)s:%(run_id)s] - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # JSON logging setup
        self.json_log_file = json_log_file
        self.json_log_entries = deque(maxlen=10000)  # Keep recent entries in memory
        self.json_log_lock = threading.Lock()
        
        # Start background JSON logging thread if file specified
        if json_log_file:
            self._start_json_logging_thread()
    
    def set_context(self, stage: str = None, run_id: int = None, experiment_id: str = None):
        """Set logging context."""
        if stage is not None:
            self.current_stage = stage
        if run_id is not None:
            self.current_run_id = run_id
        if experiment_id is not None:
            self.experiment_id = experiment_id
    
    def _log_structured(self, level: str, message: str, **kwargs):
        """Log structured entry."""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            logger_name=self.name,
            message=message,
            stage=self.current_stage,
            experiment_id=self.experiment_id,
            run_id=self.current_run_id,
            metadata=kwargs
        )
        
        # Add to memory buffer
        with self.json_log_lock:
            self.json_log_entries.append(entry)
        
        # Standard logging
        extra = {"stage": self.current_stage or "", "run_id": self.current_run_id or ""}
        getattr(self.logger, level.lower())(message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log_structured("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log_structured("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log_structured("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log_structured("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log_structured("CRITICAL", message, **kwargs)
    
    def log_experiment_start(self, config: Dict[str, Any]):
        """Log experiment start with configuration."""
        self.info("Experiment started", config=config, event_type="experiment_start")
    
    def log_experiment_end(self, duration: float, status: str, results: Dict[str, Any] = None):
        """Log experiment completion."""
        self.info(f"Experiment completed: {status}", 
                 duration=duration, status=status, results=results, event_type="experiment_end")
    
    def log_stage_start(self, stage_name: str, parameters: Dict[str, Any] = None):
        """Log stage start."""
        self.set_context(stage=stage_name)
        self.info(f"Stage '{stage_name}' started", 
                 parameters=parameters, event_type="stage_start")
    
    def log_stage_end(self, stage_name: str, duration: float, results: Dict[str, Any] = None):
        """Log stage completion."""
        self.info(f"Stage '{stage_name}' completed", 
                 duration=duration, results=results, event_type="stage_end")
        self.set_context(stage=None)
    
    def log_hyperparameter_result(self, parameters: Dict[str, Any], score: float, 
                                 metadata: Dict[str, Any] = None):
        """Log hyperparameter search result."""
        self.info("Hyperparameter evaluation completed",
                 parameters=parameters, score=score, metadata=metadata,
                 event_type="hyperparameter_result")
    
    def log_performance_metrics(self, metrics: Dict[str, float]):
        """Log performance metrics."""
        self.info("Performance metrics computed", metrics=metrics, event_type="performance_metrics")
    
    def log_resource_usage(self, cpu_percent: float = None, memory_gb: float = None,
                          gpu_utilization: float = None):
        """Log resource usage."""
        resource_info = {}
        if cpu_percent is not None:
            resource_info["cpu_percent"] = cpu_percent
        if memory_gb is not None:
            resource_info["memory_gb"] = memory_gb
        if gpu_utilization is not None:
            resource_info["gpu_utilization"] = gpu_utilization
        
        self.debug("Resource usage", **resource_info, event_type="resource_usage")
    
    def get_log_entries(self, level: str = None, stage: str = None, 
                       event_type: str = None, limit: int = None) -> List[LogEntry]:
        """Get log entries with optional filtering."""
        with self.json_log_lock:
            entries = list(self.json_log_entries)
        
        # Apply filters
        if level:
            entries = [e for e in entries if e.level == level.upper()]
        if stage:
            entries = [e for e in entries if e.stage == stage]
        if event_type:
            entries = [e for e in entries if e.metadata.get("event_type") == event_type]
        
        if limit:
            entries = entries[-limit:]
        
        return entries
    
    def _start_json_logging_thread(self):
        """Start background thread for JSON logging."""
        def json_logger():
            while True:
                try:
                    time.sleep(1)  # Write every second
                    with self.json_log_lock:
                        if self.json_log_entries and self.json_log_file:
                            # Write new entries to file
                            entries_to_write = list(self.json_log_entries)
                            self.json_log_entries.clear()
                    
                    if entries_to_write and self.json_log_file:
                        with open(self.json_log_file, "a") as f:
                            for entry in entries_to_write:
                                json.dump(entry.to_dict(), f)
                                f.write("\n")
                
                except Exception as e:
                    # Avoid logging errors in the logger itself
                    print(f"JSON logging error: {e}", file=sys.stderr)
        
        thread = threading.Thread(target=json_logger, daemon=True)
        thread.start()


class ProgressTracker:
    """Advanced progress tracking with multiple progress bars."""
    
    def __init__(self, use_tqdm: bool = True):
        """
        Initialize progress tracker.
        
        Parameters
        ----------
        use_tqdm : bool
            Whether to use tqdm for progress bars (if available)
        """
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE
        self.progress_states = {}
        self.progress_bars = {}
        self.lock = threading.Lock()
        
    def create_progress(self, name: str, total: int, description: str = "") -> str:
        """Create a new progress tracker."""
        with self.lock:
            progress_state = ProgressState(
                total=total,
                description=description
            )
            self.progress_states[name] = progress_state
            
            if self.use_tqdm:
                pbar = tqdm(
                    total=total,
                    desc=description,
                    unit="it",
                    position=len(self.progress_bars),
                    leave=True
                )
                self.progress_bars[name] = pbar
        
        return name
    
    def update_progress(self, name: str, increment: int = 1, description: str = None):
        """Update progress."""
        with self.lock:
            if name in self.progress_states:
                self.progress_states[name].update(increment, description)
                
                if self.use_tqdm and name in self.progress_bars:
                    pbar = self.progress_bars[name]
                    pbar.update(increment)
                    if description:
                        pbar.set_description(description)
    
    def set_progress(self, name: str, current: int, description: str = None):
        """Set absolute progress."""
        with self.lock:
            if name in self.progress_states:
                state = self.progress_states[name]
                increment = current - state.current
                if increment != 0:
                    self.update_progress(name, increment, description)
    
    def finish_progress(self, name: str):
        """Finish and close progress tracker."""
        with self.lock:
            if name in self.progress_states:
                self.progress_states[name].completed = True
                
                if self.use_tqdm and name in self.progress_bars:
                    self.progress_bars[name].close()
                    del self.progress_bars[name]
    
    def get_progress_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get progress information."""
        with self.lock:
            if name in self.progress_states:
                return self.progress_states[name].get_progress_info()
        return None
    
    def get_all_progress(self) -> Dict[str, Dict[str, Any]]:
        """Get all progress information."""
        with self.lock:
            return {
                name: state.get_progress_info() 
                for name, state in self.progress_states.items()
            }
    
    @contextmanager
    def progress_context(self, name: str, total: int, description: str = ""):
        """Context manager for progress tracking."""
        progress_id = self.create_progress(name, total, description)
        try:
            yield progress_id
        finally:
            self.finish_progress(progress_id)


class ResourceMonitor:
    """Monitor computational resources during experiments."""
    
    def __init__(self, monitoring_interval: float = 1.0, 
                 logger: Optional[StructuredLogger] = None):
        """
        Initialize resource monitor.
        
        Parameters
        ----------
        monitoring_interval : float
            Monitoring interval in seconds
        logger : StructuredLogger, optional
            Logger for resource usage
        """
        self.monitoring_interval = monitoring_interval
        self.logger = logger
        self.monitoring = False
        self.resource_history = deque(maxlen=1000)
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring."""
        if not PSUTIL_AVAILABLE:
            if self.logger:
                self.logger.warning("psutil not available, resource monitoring disabled")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        if self.logger:
            self.logger.info("Resource monitoring started", 
                           interval=self.monitoring_interval)
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return summary statistics."""
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.monitoring_interval * 2)
        
        # Compute summary statistics
        summary = self._compute_summary_stats()
        
        if self.logger:
            self.logger.info("Resource monitoring stopped", summary=summary)
        
        return summary
    
    def _monitor_loop(self):
        """Resource monitoring loop."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # Get system-wide metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                
                # Get process-specific metrics
                process_memory = process.memory_info().rss / (1024**3)  # GB
                process_cpu = process.cpu_percent()
                
                # Record resource usage
                resource_data = {
                    "timestamp": time.time(),
                    "system_cpu_percent": cpu_percent,
                    "system_memory_percent": memory_info.percent,
                    "system_memory_available_gb": memory_info.available / (1024**3),
                    "process_memory_gb": process_memory,
                    "process_cpu_percent": process_cpu
                }
                
                # Try to get GPU info if available
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        resource_data.update({
                            "gpu_utilization": gpu.load * 100,
                            "gpu_memory_used": gpu.memoryUsed,
                            "gpu_memory_total": gpu.memoryTotal,
                            "gpu_temperature": gpu.temperature
                        })
                except ImportError:
                    pass
                
                self.resource_history.append(resource_data)
                
                # Log resource usage periodically
                if self.logger and len(self.resource_history) % 60 == 0:  # Every minute
                    self.logger.log_resource_usage(
                        cpu_percent=cpu_percent,
                        memory_gb=process_memory
                    )
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Resource monitoring error: {e}")
                break
    
    def _compute_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics from resource history."""
        if not self.resource_history:
            return {}
        
        # Convert to arrays for easier computation
        timestamps = [r["timestamp"] for r in self.resource_history]
        cpu_percents = [r["system_cpu_percent"] for r in self.resource_history]
        memory_percents = [r["system_memory_percent"] for r in self.resource_history]
        process_memory = [r["process_memory_gb"] for r in self.resource_history]
        
        summary = {
            "monitoring_duration": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
            "samples_collected": len(self.resource_history),
            "cpu": {
                "mean": np.mean(cpu_percents),
                "max": np.max(cpu_percents),
                "min": np.min(cpu_percents),
                "std": np.std(cpu_percents)
            },
            "system_memory": {
                "mean": np.mean(memory_percents),
                "max": np.max(memory_percents),
                "min": np.min(memory_percents)
            },
            "process_memory": {
                "mean": np.mean(process_memory),
                "max": np.max(process_memory),
                "min": np.min(process_memory),
                "peak": np.max(process_memory)
            }
        }
        
        # Add GPU stats if available
        gpu_utils = [r.get("gpu_utilization") for r in self.resource_history]
        gpu_utils = [u for u in gpu_utils if u is not None]
        
        if gpu_utils:
            summary["gpu"] = {
                "mean_utilization": np.mean(gpu_utils),
                "max_utilization": np.max(gpu_utils),
                "min_utilization": np.min(gpu_utils)
            }
        
        return summary
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        if not PSUTIL_AVAILABLE:
            return {}
        
        try:
            process = psutil.Process()
            
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "process_memory_gb": process.memory_info().rss / (1024**3),
                "process_cpu_percent": process.cpu_percent()
            }
        except Exception:
            return {}


class ExperimentLogger:
    """
    Comprehensive experiment logging system that integrates all logging components.
    """
    
    def __init__(self, experiment_name: str, experiment_id: str = None,
                 output_dir: Union[str, Path] = "logs",
                 log_level: int = logging.INFO,
                 enable_progress: bool = True,
                 enable_resource_monitoring: bool = True,
                 resource_monitoring_interval: float = 1.0):
        """
        Initialize experiment logger.
        
        Parameters
        ----------
        experiment_name : str
            Name of the experiment
        experiment_id : str, optional
            Unique experiment identifier
        output_dir : str or Path
            Directory for log files
        log_level : int
            Logging level
        enable_progress : bool
            Whether to enable progress tracking
        enable_resource_monitoring : bool
            Whether to monitor resource usage
        resource_monitoring_interval : float
            Resource monitoring interval in seconds
        """
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id or f"{experiment_name}_{int(time.time())}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log files
        log_file = self.output_dir / f"{self.experiment_id}.log"
        json_log_file = self.output_dir / f"{self.experiment_id}.jsonl"
        
        # Initialize structured logger
        self.logger = StructuredLogger(
            name=f"experiment.{self.experiment_name}",
            level=log_level,
            log_file=log_file,
            json_log_file=json_log_file,
            experiment_id=self.experiment_id
        )
        
        # Initialize progress tracker
        self.progress_tracker = ProgressTracker() if enable_progress else None
        
        # Initialize resource monitor
        self.resource_monitor = (ResourceMonitor(
            monitoring_interval=resource_monitoring_interval,
            logger=self.logger
        ) if enable_resource_monitoring else None)
        
        # Experiment state
        self.experiment_start_time = None
        self.stage_start_times = {}
        self.current_stage = None
        
    def start_experiment(self, config: Dict[str, Any]):
        """Start experiment logging."""
        self.experiment_start_time = time.time()
        self.logger.set_context(experiment_id=self.experiment_id)
        self.logger.log_experiment_start(config)
        
        if self.resource_monitor:
            self.resource_monitor.start_monitoring()
        
        print(f"Experiment '{self.experiment_name}' started (ID: {self.experiment_id})")
    
    def end_experiment(self, status: str = "completed", results: Dict[str, Any] = None):
        """End experiment logging."""
        if self.experiment_start_time:
            duration = time.time() - self.experiment_start_time
            self.logger.log_experiment_end(duration, status, results)
        
        # Stop resource monitoring
        resource_summary = {}
        if self.resource_monitor:
            resource_summary = self.resource_monitor.stop_monitoring()
        
        # Close all progress bars
        if self.progress_tracker:
            for name in list(self.progress_tracker.progress_states.keys()):
                self.progress_tracker.finish_progress(name)
        
        print(f"Experiment '{self.experiment_name}' {status}")
        if resource_summary:
            peak_memory = resource_summary.get("process_memory", {}).get("peak", 0)
            print(f"Peak memory usage: {peak_memory:.2f} GB")
    
    @contextmanager
    def stage_context(self, stage_name: str, total_steps: int = None, 
                     stage_params: Dict[str, Any] = None):
        """Context manager for stage logging."""
        # Start stage
        self.current_stage = stage_name
        self.stage_start_times[stage_name] = time.time()
        self.logger.log_stage_start(stage_name, stage_params)
        
        # Create progress tracker for stage
        progress_name = None
        if self.progress_tracker and total_steps:
            progress_name = f"{stage_name}_progress"
            self.progress_tracker.create_progress(
                progress_name, total_steps, f"Stage: {stage_name}"
            )
        
        try:
            yield StageContext(self, stage_name, progress_name)
        finally:
            # End stage
            duration = time.time() - self.stage_start_times[stage_name]
            self.logger.log_stage_end(stage_name, duration)
            
            if progress_name:
                self.progress_tracker.finish_progress(progress_name)
            
            self.current_stage = None
    
    def log_hyperparameter_result(self, parameters: Dict[str, Any], score: float,
                                 metadata: Dict[str, Any] = None):
        """Log hyperparameter search result."""
        self.logger.log_hyperparameter_result(parameters, score, metadata)
    
    def log_performance_metrics(self, metrics: Dict[str, float]):
        """Log performance metrics."""
        self.logger.log_performance_metrics(metrics)
    
    def create_progress(self, name: str, total: int, description: str = "") -> Optional[str]:
        """Create progress tracker."""
        if self.progress_tracker:
            return self.progress_tracker.create_progress(name, total, description)
        return None
    
    def update_progress(self, name: str, increment: int = 1, description: str = None):
        """Update progress."""
        if self.progress_tracker:
            self.progress_tracker.update_progress(name, increment, description)
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        if self.resource_monitor:
            return self.resource_monitor.get_current_usage()
        return {}
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        summary = {
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "start_time": self.experiment_start_time,
            "current_stage": self.current_stage,
            "log_file": str(self.output_dir / f"{self.experiment_id}.log"),
            "json_log_file": str(self.output_dir / f"{self.experiment_id}.jsonl")
        }
        
        # Add progress information
        if self.progress_tracker:
            summary["progress"] = self.progress_tracker.get_all_progress()
        
        # Add resource information
        if self.resource_monitor:
            summary["current_resource_usage"] = self.resource_monitor.get_current_usage()
        
        return summary


class StageContext:
    """Context object for stage execution with logging helpers."""
    
    def __init__(self, experiment_logger: ExperimentLogger, stage_name: str, 
                 progress_name: Optional[str]):
        self.experiment_logger = experiment_logger
        self.stage_name = stage_name
        self.progress_name = progress_name
        
    def log_info(self, message: str, **kwargs):
        """Log info message for current stage."""
        self.experiment_logger.logger.info(message, **kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """Log warning message for current stage."""
        self.experiment_logger.logger.warning(message, **kwargs)
    
    def log_error(self, message: str, **kwargs):
        """Log error message for current stage."""
        self.experiment_logger.logger.error(message, **kwargs)
    
    def update_progress(self, increment: int = 1, description: str = None):
        """Update stage progress."""
        if self.progress_name:
            self.experiment_logger.progress_tracker.update_progress(
                self.progress_name, increment, description
            )
    
    def log_hyperparameter_result(self, parameters: Dict[str, Any], score: float,
                                 metadata: Dict[str, Any] = None):
        """Log hyperparameter result for current stage."""
        self.experiment_logger.log_hyperparameter_result(parameters, score, metadata)
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics for current stage."""
        self.experiment_logger.log_performance_metrics(metrics)


# Convenience function for creating experiment loggers
def create_experiment_logger(experiment_name: str, **kwargs) -> ExperimentLogger:
    """Create an experiment logger with sensible defaults."""
    return ExperimentLogger(experiment_name, **kwargs)