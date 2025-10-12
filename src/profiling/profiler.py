"""
Comprehensive profiling system for RGB training pipeline.

This module provides detailed profiling capabilities to identify bottlenecks
in the RGB training pipeline, including environment rendering, model forward
passes, memory usage, and training steps.
"""

import time
import psutil
import torch
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import os


@dataclass
class ProfilerStats:
    """Statistics for a profiled operation."""
    name: str
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    memory_peak: float = 0.0
    memory_avg: float = 0.0
    gpu_memory_peak: float = 0.0
    gpu_memory_avg: float = 0.0
    
    def update(self, duration: float, memory_used: float = 0.0, gpu_memory_used: float = 0.0):
        """Update statistics with a new measurement."""
        self.total_time += duration
        self.call_count += 1
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.avg_time = self.total_time / self.call_count
        self.memory_peak = max(self.memory_peak, memory_used)
        self.memory_avg = (self.memory_avg * (self.call_count - 1) + memory_used) / self.call_count
        self.gpu_memory_peak = max(self.gpu_memory_peak, gpu_memory_used)
        self.gpu_memory_avg = (self.gpu_memory_avg * (self.call_count - 1) + gpu_memory_used) / self.call_count


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: float
    cpu_memory_mb: float
    gpu_memory_mb: float
    process_memory_mb: float
    available_memory_mb: float


class Profiler:
    """Comprehensive profiler for RGB training pipeline."""
    
    def __init__(self, enable_gpu_profiling: bool = True, max_history: int = 1000):
        # Support CUDA and Apple MPS as GPU-like devices
        self._has_cuda = torch.cuda.is_available()
        self._has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        self.enable_gpu_profiling = enable_gpu_profiling and (self._has_cuda or self._has_mps)
        self.max_history = max_history
        self.stats: Dict[str, ProfilerStats] = {}
        self.memory_snapshots: List[MemorySnapshot] = []
        self._lock = threading.Lock()
        self._enabled = True
        
        # Initialize GPU memory tracking
        if self.enable_gpu_profiling:
            if self._has_cuda:
                torch.cuda.empty_cache()
                self._initial_gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            else:
                # MPS memory APIs are limited; set baseline to 0 and only sync
                self._initial_gpu_memory = 0.0
        else:
            self._initial_gpu_memory = 0.0
    
    def _get_memory_usage(self) -> tuple[float, float, float]:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info().rss / 1024**2  # MB
        system_memory = psutil.virtual_memory()
        available_memory = system_memory.available / 1024**2  # MB
        
        gpu_memory = 0.0
        if self.enable_gpu_profiling:
            if self._has_cuda:
                gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            else:
                # MPS: PyTorch does not expose allocated bytes; approximate via reserved if available
                try:
                    # Newer PyTorch may expose torch.mps.current_allocated_memory()
                    mps_mod = getattr(torch, "mps", None)
                    if mps_mod is not None and hasattr(mps_mod, "current_allocated_memory"):
                        gpu_memory = float(torch.mps.current_allocated_memory()) / 1024**2
                    else:
                        gpu_memory = 0.0
                except Exception:
                    gpu_memory = 0.0
        
        return process_memory, gpu_memory, available_memory
    
    def _take_memory_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        process_memory, gpu_memory, available_memory = self._get_memory_usage()
        return MemorySnapshot(
            timestamp=time.time(),
            cpu_memory_mb=psutil.virtual_memory().used / 1024**2,
            gpu_memory_mb=gpu_memory,
            process_memory_mb=process_memory,
            available_memory_mb=available_memory
        )
    
    @contextmanager
    def profile(self, name: str, track_memory: bool = True):
        """Context manager for profiling operations."""
        if not self._enabled:
            yield
            return
        
        # Take initial memory snapshot
        if track_memory:
            initial_snapshot = self._take_memory_snapshot()
            if self.enable_gpu_profiling:
                try:
                    if self._has_cuda:
                        torch.cuda.synchronize()
                    elif self._has_mps:
                        # Ensure pending kernels are flushed on MPS
                        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                            torch.mps.synchronize()
                except Exception:
                    pass
        
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            # Take final memory snapshot
            if track_memory:
                final_snapshot = self._take_memory_snapshot()
                memory_used = final_snapshot.process_memory_mb - initial_snapshot.process_memory_mb
                gpu_memory_used = final_snapshot.gpu_memory_mb - initial_snapshot.gpu_memory_mb
            else:
                memory_used = 0.0
                gpu_memory_used = 0.0
            
            # Update statistics
            with self._lock:
                if name not in self.stats:
                    self.stats[name] = ProfilerStats(name)
                self.stats[name].update(duration, memory_used, gpu_memory_used)
    
    def profile_function(self, name: str, track_memory: bool = True):
        """Decorator for profiling functions."""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                with self.profile(name, track_memory):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_stats(self, name: Optional[str] = None) -> Dict[str, ProfilerStats]:
        """Get profiling statistics."""
        with self._lock:
            if name is None:
                return dict(self.stats)
            elif name in self.stats:
                return {name: self.stats[name]}
            else:
                return {}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all profiling statistics."""
        with self._lock:
            summary = {
                'total_operations': len(self.stats),
                'operations': {}
            }
            
            for name, stats in self.stats.items():
                summary['operations'][name] = {
                    'total_time': stats.total_time,
                    'call_count': stats.call_count,
                    'avg_time': stats.avg_time,
                    'min_time': stats.min_time,
                    'max_time': stats.max_time,
                    'memory_peak_mb': stats.memory_peak,
                    'memory_avg_mb': stats.memory_avg,
                    'gpu_memory_peak_mb': stats.gpu_memory_peak,
                    'gpu_memory_avg_mb': stats.gpu_memory_avg,
                }
            
            return summary
    
    def reset(self):
        """Reset all profiling statistics."""
        with self._lock:
            self.stats.clear()
            self.memory_snapshots.clear()
            if self.enable_gpu_profiling and self._has_cuda:
                torch.cuda.empty_cache()
    
    def enable(self):
        """Enable profiling."""
        self._enabled = True
    
    def disable(self):
        """Disable profiling."""
        self._enabled = False
    
    def print_summary(self, sort_by: str = 'total_time'):
        """Print a formatted summary of profiling statistics."""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("PROFILING SUMMARY")
        print("="*80)
        
        if not summary['operations']:
            print("No profiling data available.")
            return
        
        # Sort operations by specified metric
        operations = summary['operations']
        sorted_ops = sorted(operations.items(), 
                          key=lambda x: x[1].get(sort_by, 0), 
                          reverse=True)
        
        print(f"{'Operation':<30} {'Calls':<8} {'Total(s)':<10} {'Avg(s)':<10} {'Min(s)':<10} {'Max(s)':<10} {'Mem Peak(MB)':<12} {'GPU Peak(MB)':<12}")
        print("-"*120)
        
        for name, stats in sorted_ops:
            print(f"{name:<30} {stats['call_count']:<8} {stats['total_time']:<10.4f} {stats['avg_time']:<10.4f} "
                  f"{stats['min_time']:<10.4f} {stats['max_time']:<10.4f} {stats['memory_peak_mb']:<12.2f} {stats['gpu_memory_peak_mb']:<12.2f}")
        
        print("="*80)
    
    def save_report(self, filename: str):
        """Save profiling report to file."""
        import json
        
        report = {
            'timestamp': time.time(),
            'summary': self.get_summary(),
            'memory_snapshots': [
                {
                    'timestamp': snap.timestamp,
                    'cpu_memory_mb': snap.cpu_memory_mb,
                    'gpu_memory_mb': snap.gpu_memory_mb,
                    'process_memory_mb': snap.process_memory_mb,
                    'available_memory_mb': snap.available_memory_mb
                }
                for snap in self.memory_snapshots
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"Profiling report saved to {filename}")


# Global profiler instance
_global_profiler = Profiler()


def get_profiler() -> Profiler:
    """Get the global profiler instance."""
    return _global_profiler


def profile(name: str, track_memory: bool = True):
    """Convenience function for profiling operations."""
    return _global_profiler.profile(name, track_memory)


def profile_function(name: str, track_memory: bool = True):
    """Convenience decorator for profiling functions."""
    return _global_profiler.profile_function(name, track_memory)


def print_profiling_summary(sort_by: str = 'total_time'):
    """Print profiling summary."""
    _global_profiler.print_summary(sort_by)


def save_profiling_report(filename: str):
    """Save profiling report."""
    _global_profiler.save_report(filename)


def reset_profiling():
    """Reset profiling statistics."""
    _global_profiler.reset()
