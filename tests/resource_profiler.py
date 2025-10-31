"""
Resource Profiling Tools for GraphRAG-SDK

This module provides comprehensive resource profiling capabilities including:
- Memory usage tracking and analysis
- CPU utilization monitoring
- I/O operations profiling
- Performance bottleneck identification
- Resource usage optimization recommendations
"""

import time
import threading
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from pathlib import Path

# Optional dependencies for system monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Optional dependencies for plotting
try:
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


@dataclass
class ResourceSnapshot:
    """Single snapshot of system resources"""
    timestamp: float
    memory_rss_mb: float
    memory_vms_mb: float
    memory_percent: float
    cpu_percent: float
    disk_read_mb: float
    disk_write_mb: float
    network_recv_mb: float
    network_sent_mb: float
    open_files: int = 0
    threads: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class ProfileSummary:
    """Summary of profiling session"""
    start_time: float
    end_time: float
    duration: float
    peak_memory_mb: float
    avg_memory_mb: float
    peak_cpu_percent: float
    avg_cpu_percent: float
    total_disk_read_mb: float
    total_disk_write_mb: float
    total_network_mb: float
    snapshots_count: int
    memory_efficiency_score: float = 0.0
    cpu_efficiency_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


class ResourceProfiler:
    """Main resource profiling class"""
    
    def __init__(self, sampling_interval: float = 0.5):
        self.sampling_interval = sampling_interval
        self.snapshots: List[ResourceSnapshot] = []
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None
        self.end_time = None
        
        # Initial system state
        self.initial_disk_io = None
        self.initial_network_io = None
        
    def start_profiling(self):
        """Start resource profiling"""
        if not HAS_PSUTIL:
            print("Warning: psutil not available, profiling will be limited")
            return
            
        self.monitoring = True
        self.start_time = time.time()
        self.snapshots.clear()
        
        # Record initial I/O counters
        try:
            self.initial_disk_io = psutil.disk_io_counters()
            self.initial_network_io = psutil.net_io_counters()
        except (psutil.AccessDenied, AttributeError, TypeError):
            self.initial_disk_io = None
            self.initial_network_io = None
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_profiling(self) -> ProfileSummary:
        """Stop profiling and return summary"""
        self.monitoring = False
        self.end_time = time.time()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        return self._generate_summary()
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                snapshot = self._take_snapshot(process)
                self.snapshots.append(snapshot)
                time.sleep(self.sampling_interval)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
    
    def _take_snapshot(self, process) -> ResourceSnapshot:
        """Take a single resource snapshot"""
        timestamp = time.time()
        
        # Memory information
        try:
            memory_info = process.memory_info()
            memory_rss_mb = memory_info.rss / (1024 * 1024)
            memory_vms_mb = memory_info.vms / (1024 * 1024)
            memory_percent = process.memory_percent()
        except (psutil.AccessDenied, AttributeError):
            memory_rss_mb = memory_vms_mb = memory_percent = 0.0
        
        # CPU information
        try:
            cpu_percent = process.cpu_percent()
        except (psutil.AccessDenied, AttributeError):
            cpu_percent = 0.0
        
        # Disk I/O information
        try:
            current_disk_io = psutil.disk_io_counters()
            if (self.initial_disk_io and current_disk_io and 
                hasattr(current_disk_io, 'read_bytes') and hasattr(self.initial_disk_io, 'read_bytes')):
                disk_read_mb = (current_disk_io.read_bytes - self.initial_disk_io.read_bytes) / (1024 * 1024)
                disk_write_mb = (current_disk_io.write_bytes - self.initial_disk_io.write_bytes) / (1024 * 1024)
            else:
                disk_read_mb = disk_write_mb = 0.0
        except (psutil.AccessDenied, AttributeError, TypeError):
            disk_read_mb = disk_write_mb = 0.0
        
        # Network I/O information
        try:
            current_network_io = psutil.net_io_counters()
            if (self.initial_network_io and current_network_io and 
                hasattr(current_network_io, 'bytes_recv') and hasattr(self.initial_network_io, 'bytes_recv')):
                network_recv_mb = (current_network_io.bytes_recv - self.initial_network_io.bytes_recv) / (1024 * 1024)
                network_sent_mb = (current_network_io.bytes_sent - self.initial_network_io.bytes_sent) / (1024 * 1024)
            else:
                network_recv_mb = network_sent_mb = 0.0
        except (psutil.AccessDenied, AttributeError, TypeError):
            network_recv_mb = network_sent_mb = 0.0
        
        # Other metrics
        try:
            open_files = len(process.open_files())
            threads = process.num_threads()
        except (psutil.AccessDenied, AttributeError):
            open_files = threads = 0
        
        return ResourceSnapshot(
            timestamp=timestamp,
            memory_rss_mb=memory_rss_mb,
            memory_vms_mb=memory_vms_mb,
            memory_percent=memory_percent,
            cpu_percent=cpu_percent,
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb,
            network_recv_mb=network_recv_mb,
            network_sent_mb=network_sent_mb,
            open_files=open_files,
            threads=threads
        )
    
    def _generate_summary(self) -> ProfileSummary:
        """Generate profiling summary"""
        if not self.snapshots:
            return ProfileSummary(
                start_time=self.start_time or 0.0,
                end_time=self.end_time or 0.0,
                duration=0.0,
                peak_memory_mb=0.0,
                avg_memory_mb=0.0,
                peak_cpu_percent=0.0,
                avg_cpu_percent=0.0,
                total_disk_read_mb=0.0,
                total_disk_write_mb=0.0,
                total_network_mb=0.0,
                snapshots_count=0
            )
        
        duration = (self.end_time or 0.0) - (self.start_time or 0.0)
        
        # Memory metrics
        memory_values = [s.memory_rss_mb for s in self.snapshots]
        peak_memory_mb = max(memory_values)
        avg_memory_mb = sum(memory_values) / len(memory_values)
        
        # CPU metrics
        cpu_values = [s.cpu_percent for s in self.snapshots]
        peak_cpu_percent = max(cpu_values)
        avg_cpu_percent = sum(cpu_values) / len(cpu_values)
        
        # I/O metrics
        total_disk_read_mb = max(s.disk_read_mb for s in self.snapshots)
        total_disk_write_mb = max(s.disk_write_mb for s in self.snapshots)
        total_network_mb = max(s.network_recv_mb + s.network_sent_mb for s in self.snapshots)
        
        # Efficiency scores (0-100, higher is better)
        memory_efficiency_score = max(0, 100 - (peak_memory_mb / 1024))  # Penalize >1GB memory usage
        cpu_efficiency_score = max(0, 100 - avg_cpu_percent)  # Penalize high CPU usage
        
        return ProfileSummary(
            start_time=self.start_time or 0.0,
            end_time=self.end_time or 0.0,
            duration=duration,
            peak_memory_mb=peak_memory_mb,
            avg_memory_mb=avg_memory_mb,
            peak_cpu_percent=peak_cpu_percent,
            avg_cpu_percent=avg_cpu_percent,
            total_disk_read_mb=total_disk_read_mb,
            total_disk_write_mb=total_disk_write_mb,
            total_network_mb=total_network_mb,
            snapshots_count=len(self.snapshots),
            memory_efficiency_score=memory_efficiency_score,
            cpu_efficiency_score=cpu_efficiency_score
        )
    
    def get_snapshots(self) -> List[ResourceSnapshot]:
        """Get all collected snapshots"""
        return self.snapshots.copy()
    
    def save_profile(self, filename: str):
        """Save profiling data to file"""
        data = {
            'summary': self._generate_summary().to_dict(),
            'snapshots': [s.to_dict() for s in self.snapshots],
            'sampling_interval': self.sampling_interval
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_profile(self, filename: str):
        """Load profiling data from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.snapshots = [ResourceSnapshot(**s) for s in data['snapshots']]
        self.sampling_interval = data['sampling_interval']
    
    def create_visualization(self, output_file: Optional[str] = None):
        """Create visualization of resource usage"""
        if not HAS_PLOTTING:
            print("Matplotlib not available for visualization")
            return
        
        if not self.snapshots:
            print("No profiling data available")
            return
        
        if output_file is None:
            output_file = f"resource_profile_{int(time.time())}.png"
        
        # Extract data
        timestamps = [s.timestamp - self.snapshots[0].timestamp for s in self.snapshots]
        memory_mb = [s.memory_rss_mb for s in self.snapshots]
        cpu_percent = [s.cpu_percent for s in self.snapshots]
        disk_read_mb = [s.disk_read_mb for s in self.snapshots]
        disk_write_mb = [s.disk_write_mb for s in self.snapshots]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(4, 1, figsize=(12, 10))
        
        # Memory usage
        ax1.plot(timestamps, memory_mb, 'b-', linewidth=2)
        ax1.set_ylabel('Memory (MB)')
        ax1.set_title('Memory Usage Over Time')
        ax1.grid(True, alpha=0.3)
        
        # CPU usage
        ax2.plot(timestamps, cpu_percent, 'r-', linewidth=2)
        ax2.set_ylabel('CPU (%)')
        ax2.set_title('CPU Usage Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Disk I/O
        ax3.plot(timestamps, disk_read_mb, 'g-', label='Read', linewidth=2)
        ax3.plot(timestamps, disk_write_mb, 'orange', label='Write', linewidth=2)
        ax3.set_ylabel('Disk I/O (MB)')
        ax3.set_title('Disk I/O Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Combined efficiency
        memory_norm = [m / max(memory_mb) * 100 if max(memory_mb) > 0 else 0 for m in memory_mb]
        cpu_norm = cpu_percent
        ax4.plot(timestamps, memory_norm, 'b-', label='Memory Load', linewidth=2, alpha=0.7)
        ax4.plot(timestamps, cpu_norm, 'r-', label='CPU Load', linewidth=2, alpha=0.7)
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Load (%)')
        ax4.set_title('System Load Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Resource profile visualization saved to: {output_file}")
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance and provide recommendations"""
        if not self.snapshots:
            return {"error": "No profiling data available"}
        
        summary = self._generate_summary()
        analysis = {
            "summary": summary.to_dict(),
            "recommendations": [],
            "bottlenecks": [],
            "efficiency_grade": "F"
        }
        
        # Memory analysis
        if summary.peak_memory_mb > 2048:  # > 2GB
            analysis["bottlenecks"].append("High memory usage detected")
            analysis["recommendations"].append("Consider implementing streaming or chunking to reduce memory footprint")
        elif summary.peak_memory_mb > 1024:  # > 1GB
            analysis["recommendations"].append("Memory usage is moderate, monitor for growth")
        
        # CPU analysis
        if summary.avg_cpu_percent > 80:
            analysis["bottlenecks"].append("High CPU usage detected")
            analysis["recommendations"].append("Consider optimizing algorithms or implementing parallel processing")
        elif summary.avg_cpu_percent > 60:
            analysis["recommendations"].append("CPU usage is high, investigate optimization opportunities")
        
        # I/O analysis
        if summary.total_disk_read_mb > 1000:  # > 1GB
            analysis["recommendations"].append("High disk I/O detected, consider caching strategies")
        
        # Efficiency grading
        efficiency_score = (summary.memory_efficiency_score + summary.cpu_efficiency_score) / 2
        
        if efficiency_score >= 80:
            analysis["efficiency_grade"] = "A"
        elif efficiency_score >= 70:
            analysis["efficiency_grade"] = "B"
        elif efficiency_score >= 60:
            analysis["efficiency_grade"] = "C"
        elif efficiency_score >= 50:
            analysis["efficiency_grade"] = "D"
        else:
            analysis["efficiency_grade"] = "F"
        
        return analysis


@contextmanager
def profile_function(name: str = "function", sampling_interval: float = 0.5):
    """Context manager for profiling a function"""
    profiler = ResourceProfiler(sampling_interval)
    
    print(f"Starting profiling for: {name}")
    profiler.start_profiling()
    
    try:
        yield profiler
    finally:
        summary = profiler.stop_profiling()
        print(f"\nProfiling completed for: {name}")
        print(f"Duration: {summary.duration:.2f}s")
        print(f"Peak Memory: {summary.peak_memory_mb:.1f}MB")
        print(f"Avg CPU: {summary.avg_cpu_percent:.1f}%")
        print(f"Efficiency Grade: {profiler.analyze_performance()['efficiency_grade']}")


class PerformanceComparator:
    """Compare performance between different runs"""
    
    def __init__(self):
        self.profiles: Dict[str, ProfileSummary] = {}
    
    def add_profile(self, name: str, profile: ProfileSummary):
        """Add a profile for comparison"""
        self.profiles[name] = profile
    
    def compare(self) -> Dict[str, Any]:
        """Compare all profiles and return analysis"""
        if len(self.profiles) < 2:
            return {"error": "Need at least 2 profiles for comparison"}
        
        comparison = {
            "profiles": {name: profile.to_dict() for name, profile in self.profiles.items()},
            "rankings": {},
            "improvements": {}
        }
        
        # Rank by different metrics
        metrics = ['duration', 'peak_memory_mb', 'avg_cpu_percent']
        
        for metric in metrics:
            sorted_profiles = sorted(self.profiles.items(), key=lambda x: getattr(x[1], metric))
            comparison["rankings"][metric] = [name for name, _ in sorted_profiles]
        
        # Calculate improvements (compare first vs last)
        if len(self.profiles) >= 2:
            first_profile = list(self.profiles.values())[0]
            last_profile = list(self.profiles.values())[-1]
            
            comparison["improvements"] = {
                "speed_improvement": (first_profile.duration - last_profile.duration) / first_profile.duration * 100,
                "memory_improvement": (first_profile.peak_memory_mb - last_profile.peak_memory_mb) / first_profile.peak_memory_mb * 100,
                "cpu_improvement": (first_profile.avg_cpu_percent - last_profile.avg_cpu_percent) / first_profile.avg_cpu_percent * 100
            }
        
        return comparison


# Utility functions
def profile_method(method: Callable, *args, **kwargs) -> tuple:
    """Profile a method call and return (result, profile_summary)"""
    profiler = ResourceProfiler()
    profiler.start_profiling()
    
    try:
        result = method(*args, **kwargs)
        return result, profiler.stop_profiling()
    except Exception as e:
        profiler.stop_profiling()
        raise e


def create_performance_report(profiler: ResourceProfiler, output_file: Optional[str] = None) -> str:
    """Create a detailed performance report"""
    analysis = profiler.analyze_performance()
    summary = profiler._generate_summary()
    
    report = []
    report.append("# Resource Performance Report\n")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report.append("## Summary\n")
    report.append(f"- Duration: {summary.duration:.2f} seconds")
    report.append(f"- Peak Memory: {summary.peak_memory_mb:.1f} MB")
    report.append(f"- Average Memory: {summary.avg_memory_mb:.1f} MB")
    report.append(f"- Peak CPU: {summary.peak_cpu_percent:.1f}%")
    report.append(f"- Average CPU: {summary.avg_cpu_percent:.1f}%")
    report.append(f"- Efficiency Grade: {analysis['efficiency_grade']}")
    report.append("")
    
    if analysis.get("bottlenecks"):
        report.append("## Bottlenecks Identified\n")
        for bottleneck in analysis["bottlenecks"]:
            report.append(f"- {bottleneck}")
        report.append("")
    
    if analysis.get("recommendations"):
        report.append("## Recommendations\n")
        for rec in analysis["recommendations"]:
            report.append(f"- {rec}")
        report.append("")
    
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Performance report saved to: {output_file}")
    
    return report_text