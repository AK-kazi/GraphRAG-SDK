"""
Performance Testing Framework for GraphRAG-SDK

This module provides comprehensive performance testing capabilities including:
- Benchmarking of ingestion and extraction operations
- Resource usage monitoring
- Performance regression detection
- Comparative analysis between optimized and non-optimized versions
"""

import time
import threading
import json
import statistics
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from pathlib import Path

# Optional dependencies for visualization
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Optional dependency for system monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single test run"""
    test_name: str
    start_time: float
    end_time: float
    duration: float
    documents_processed: int
    entities_created: int
    relations_created: int
    peak_memory_mb: float
    avg_cpu_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_mb: float = 0.0
    error_count: int = 0
    batch_count: int = 0
    
    @property
    def docs_per_second(self) -> float:
        """Documents processed per second"""
        return self.documents_processed / self.duration if self.duration > 0 else 0
    
    @property
    def entities_per_second(self) -> float:
        """Entities created per second"""
        return self.entities_created / self.duration if self.duration > 0 else 0
    
    @property
    def memory_per_doc(self) -> float:
        """Memory usage per document in KB"""
        return (self.peak_memory_mb * 1024) / self.documents_processed if self.documents_processed > 0 else 0


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests"""
    test_data_sizes: Optional[List[int]] = None
    test_iterations: int = 3
    warmup_iterations: int = 1
    enable_optimizations: bool = True
    enable_memory_monitoring: bool = True
    enable_cpu_monitoring: bool = True
    enable_io_monitoring: bool = True
    output_dir: str = "benchmark_results"
    
    def __post_init__(self):
        if self.test_data_sizes is None:
            self.test_data_sizes = [100, 500, 1000, 5000]


class ResourceMonitor:
    """Monitor system resources during test execution"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.metrics = {
            'memory': [],
            'cpu': [],
            'disk_io': [],
            'network_io': []
        }
        self.start_io_counters = None
        
    def start_monitoring(self):
        """Start resource monitoring"""
        if not HAS_PSUTIL:
            return
            
        self.monitoring = True
        self.start_io_counters = psutil.disk_io_counters()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return collected metrics"""
        if not HAS_PSUTIL:
            return {
                'peak_memory_mb': 0,
                'avg_cpu_percent': 0,
                'disk_io_read_mb': 0,
                'disk_io_write_mb': 0,
            }
            
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
        # Calculate final metrics
        end_io_counters = psutil.disk_io_counters()
        
        results = {
            'peak_memory_mb': max([m['rss'] for m in self.metrics['memory']]) / (1024 * 1024) if self.metrics['memory'] else 0,
            'avg_cpu_percent': statistics.mean([m['percent'] for m in self.metrics['cpu']]) if self.metrics['cpu'] else 0,
            'disk_io_read_mb': (end_io_counters.read_bytes - self.start_io_counters.read_bytes) / (1024 * 1024) if self.start_io_counters and end_io_counters else 0,
            'disk_io_write_mb': (end_io_counters.write_bytes - self.start_io_counters.write_bytes) / (1024 * 1024) if self.start_io_counters and end_io_counters else 0,
        }
        
        return results
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        if not HAS_PSUTIL:
            return
            
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # Memory metrics
                memory_info = process.memory_info()
                self.metrics['memory'].append({
                    'rss': memory_info.rss,
                    'vms': memory_info.vms,
                    'timestamp': time.time()
                })
                
                # CPU metrics
                cpu_percent = process.cpu_percent()
                self.metrics['cpu'].append({
                    'percent': cpu_percent,
                    'timestamp': time.time()
                })
                
                time.sleep(0.1)  # Monitor every 100ms
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break


class PerformanceBenchmark:
    """Main benchmarking framework"""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.results: List[PerformanceMetrics] = []
        self.resource_monitor = ResourceMonitor()
        
        # Create output directory
        Path(self.config.output_dir).mkdir(exist_ok=True)
        
    def run_benchmark(self, test_func: Callable, test_name: str, **kwargs) -> PerformanceMetrics:
        """Run a single benchmark test"""
        print(f"Running benchmark: {test_name}")
        
        # Warmup iterations
        for i in range(self.config.warmup_iterations):
            print(f"  Warmup iteration {i+1}/{self.config.warmup_iterations}")
            try:
                test_func(**kwargs)
            except Exception as e:
                print(f"  Warmup iteration failed: {e}")
        
        # Main test iterations
        iteration_results = []
        
        for iteration in range(self.config.test_iterations):
            print(f"  Test iteration {iteration+1}/{self.config.test_iterations}")
            
            # Start monitoring
            if self.config.enable_memory_monitoring or self.config.enable_cpu_monitoring or self.config.enable_io_monitoring:
                self.resource_monitor.start_monitoring()
            
            start_time = time.time()
            
            try:
                # Run the test function
                result = test_func(**kwargs)
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Stop monitoring and get resource metrics
                resource_metrics = {}
                if self.config.enable_memory_monitoring or self.config.enable_cpu_monitoring or self.config.enable_io_monitoring:
                    resource_metrics = self.resource_monitor.stop_monitoring()
                
                # Create performance metrics
                metrics = PerformanceMetrics(
                    test_name=test_name,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    documents_processed=int(result.get('documents_processed', 0)),
                    entities_created=int(result.get('entities_created', 0)),
                    relations_created=int(result.get('relations_created', 0)),
                    peak_memory_mb=resource_metrics.get('peak_memory_mb', 0),
                    avg_cpu_percent=resource_metrics.get('avg_cpu_percent', 0),
                    disk_io_read_mb=resource_metrics.get('disk_io_read_mb', 0),
                    disk_io_write_mb=resource_metrics.get('disk_io_write_mb', 0),
                    error_count=int(result.get('error_count', 0)),
                    batch_count=int(result.get('batch_count', 0))
                )
                
                iteration_results.append(metrics)
                print(f"    Duration: {duration:.2f}s, Docs: {metrics.documents_processed}, "
                      f"Entities: {metrics.entities_created}, Memory: {metrics.peak_memory_mb:.1f}MB")
                
            except Exception as e:
                print(f"  Test iteration failed: {e}")
                continue
        
        # Calculate average metrics
        if iteration_results:
            avg_metrics = self._average_metrics(iteration_results)
            self.results.append(avg_metrics)
            return avg_metrics
        else:
            raise RuntimeError("All test iterations failed")
    
    def _average_metrics(self, metrics_list: List[PerformanceMetrics]) -> PerformanceMetrics:
        """Calculate average metrics from multiple iterations"""
        return PerformanceMetrics(
            test_name=metrics_list[0].test_name,
            start_time=min(m.start_time for m in metrics_list),
            end_time=max(m.end_time for m in metrics_list),
            duration=statistics.mean([m.duration for m in metrics_list]),
            documents_processed=int(statistics.mean([m.documents_processed for m in metrics_list])),
            entities_created=int(statistics.mean([m.entities_created for m in metrics_list])),
            relations_created=int(statistics.mean([m.relations_created for m in metrics_list])),
            peak_memory_mb=statistics.mean([m.peak_memory_mb for m in metrics_list]),
            avg_cpu_percent=statistics.mean([m.avg_cpu_percent for m in metrics_list]),
            disk_io_read_mb=statistics.mean([m.disk_io_read_mb for m in metrics_list]),
            disk_io_write_mb=statistics.mean([m.disk_io_write_mb for m in metrics_list]),
            network_io_mb=statistics.mean([m.network_io_mb for m in metrics_list]),
            error_count=int(statistics.mean([m.error_count for m in metrics_list])),
            batch_count=int(statistics.mean([m.batch_count for m in metrics_list]))
        )
    
    def compare_implementations(self, test_funcs: Dict[str, Callable], test_name: str, **kwargs) -> Dict[str, PerformanceMetrics]:
        """Compare multiple implementations of the same functionality"""
        results = {}
        
        for impl_name, test_func in test_funcs.items():
            full_test_name = f"{test_name}_{impl_name}"
            try:
                metrics = self.run_benchmark(test_func, full_test_name, **kwargs)
                results[impl_name] = metrics
            except Exception as e:
                print(f"Failed to benchmark {impl_name}: {e}")
        
        return results
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive performance report"""
        if not self.results:
            return "No benchmark results available"
        
        # Create DataFrame for easier analysis
        data = []
        for metric in self.results:
            data.append({
                'Test': metric.test_name,
                'Duration (s)': metric.duration,
                'Docs/sec': metric.docs_per_second,
                'Entities/sec': metric.entities_per_second,
                'Peak Memory (MB)': metric.peak_memory_mb,
                'Avg CPU (%)': metric.avg_cpu_percent,
                'Disk Read (MB)': metric.disk_io_read_mb,
                'Disk Write (MB)': metric.disk_io_write_mb,
                'Memory/Doc (KB)': metric.memory_per_doc,
                'Errors': metric.error_count
            })
        
        if not HAS_PLOTTING:
            # Simple text-based report without pandas
            report = []
            report.append("# GraphRAG-SDK Performance Benchmark Report\n")
            report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            report.append(f"Total tests: {len(self.results)}\n")
            
            report.append("## Performance Summary\n")
            for metric in self.results:
                report.append(f"{metric.test_name}:")
                report.append(f"  Duration: {metric.duration:.2f}s")
                report.append(f"  Docs/sec: {metric.docs_per_second:.1f}")
                report.append(f"  Memory: {metric.peak_memory_mb:.1f}MB")
                report.append("")
            
            report_text = "\n".join(report)
            
            # Save report
            if output_file is None:
                output_file = f"{self.config.output_dir}/benchmark_report_{int(time.time())}.md"
            
            with open(output_file, 'w') as f:
                f.write(report_text)
            
            print(f"Benchmark report saved to: {output_file}")
            return report_text
        
        df = pd.DataFrame(data)
        
        # Generate report
        report = []
        report.append("# GraphRAG-SDK Performance Benchmark Report\n")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"Total tests: {len(self.results)}\n")
        
        # Summary table
        report.append("## Performance Summary\n")
        report.append(df.to_string(index=False))
        report.append("\n")
        
        # Performance analysis
        report.append("## Performance Analysis\n")
        
        # Best performers
        if len(df) > 1:
            report.append("### Best Performers\n")
            report.append(f"- Fastest processing: {df.loc[df['Docs/sec'].idxmax(), 'Test']} ({df['Docs/sec'].max():.1f} docs/sec)")
            report.append(f"- Most memory efficient: {df.loc[df['Memory/Doc (KB)'].idxmin(), 'Test']} ({df['Memory/Doc (KB)'].min():.1f} KB/doc)")
            report.append(f"- Lowest CPU usage: {df.loc[df['Avg CPU (%)'].idxmin(), 'Test']} ({df['Avg CPU (%)'].min():.1f}%)")
            report.append("\n")
        
        # Performance improvements
        optimized_tests = [r for r in self.results if 'optimized' in r.test_name.lower()]
        baseline_tests = [r for r in self.results if 'baseline' in r.test_name.lower() or 'original' in r.test_name.lower()]
        
        if optimized_tests and baseline_tests:
            report.append("### Optimization Impact\n")
            for opt_test in optimized_tests:
                # Find corresponding baseline test
                base_name = opt_test.test_name.replace('_optimized', '').replace('_baseline', '')
                baseline = next((b for b in baseline_tests if base_name in b.test_name), None)
                
                if baseline:
                    speedup = opt_test.docs_per_second / baseline.docs_per_second if baseline.docs_per_second > 0 else 0
                    memory_reduction = (baseline.peak_memory_mb - opt_test.peak_memory_mb) / baseline.peak_memory_mb * 100 if baseline.peak_memory_mb > 0 else 0
                    
                    report.append(f"- {base_name}:")
                    report.append(f"  - Speedup: {speedup:.2f}x")
                    report.append(f"  - Memory reduction: {memory_reduction:.1f}%")
                    report.append(f"  - Throughput: {baseline.docs_per_second:.1f} → {opt_test.docs_per_second:.1f} docs/sec")
        
        report_text = "\n".join(report)
        
        # Save report
        if output_file is None:
            output_file = f"{self.config.output_dir}/benchmark_report_{int(time.time())}.md"
        
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        # Save raw data
        json_file = output_file.replace('.md', '.json')
        with open(json_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        
        print(f"Benchmark report saved to: {output_file}")
        return report_text
    
    def create_visualizations(self, output_file: Optional[str] = None):
        """Create performance visualization charts"""
        if not self.results:
            print("No benchmark results available for visualization")
            return
        
        if not HAS_PLOTTING:
            print("Matplotlib and pandas not available for visualization")
            return
        
        if output_file is None:
            output_file = f"{self.config.output_dir}/benchmark_charts_{int(time.time())}.png"
        
        # Prepare data
        test_names = [r.test_name for r in self.results]
        docs_per_sec = [r.docs_per_second for r in self.results]
        memory_mb = [r.peak_memory_mb for r in self.results]
        cpu_percent = [r.avg_cpu_percent for r in self.results]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Throughput chart
        ax1.bar(range(len(test_names)), docs_per_sec)
        ax1.set_xlabel('Test')
        ax1.set_ylabel('Documents per Second')
        ax1.set_title('Processing Throughput')
        ax1.set_xticks(range(len(test_names)))
        ax1.set_xticklabels(test_names, rotation=45, ha='right')
        
        # Memory usage chart
        ax2.bar(range(len(test_names)), memory_mb)
        ax2.set_xlabel('Test')
        ax2.set_ylabel('Peak Memory (MB)')
        ax2.set_title('Memory Usage')
        ax2.set_xticks(range(len(test_names)))
        ax2.set_xticklabels(test_names, rotation=45, ha='right')
        
        # CPU usage chart
        ax3.bar(range(len(test_names)), cpu_percent)
        ax3.set_xlabel('Test')
        ax3.set_ylabel('Average CPU (%)')
        ax3.set_title('CPU Usage')
        ax3.set_xticks(range(len(test_names)))
        ax3.set_xticklabels(test_names, rotation=45, ha='right')
        
        # Efficiency chart (docs/sec per MB)
        efficiency = [d/m if m > 0 else 0 for d, m in zip(docs_per_sec, memory_mb)]
        ax4.bar(range(len(test_names)), efficiency)
        ax4.set_xlabel('Test')
        ax4.set_ylabel('Efficiency (docs/sec per MB)')
        ax4.set_title('Processing Efficiency')
        ax4.set_xticks(range(len(test_names)))
        ax4.set_xticklabels(test_names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance charts saved to: {output_file}")


@contextmanager
def benchmark_context(test_name: str, monitor_resources: bool = True):
    """Context manager for easy benchmarking"""
    monitor = ResourceMonitor()
    
    if monitor_resources:
        monitor.start_monitoring()
    
    start_time = time.time()
    
    try:
        yield monitor
    finally:
        end_time = time.time()
        
        if monitor_resources:
            resource_metrics = monitor.stop_monitoring()
            print(f"\nBenchmark: {test_name}")
            print(f"Duration: {end_time - start_time:.2f}s")
            print(f"Peak Memory: {resource_metrics.get('peak_memory_mb', 0):.1f}MB")
            print(f"Avg CPU: {resource_metrics.get('avg_cpu_percent', 0):.1f}%")
            print(f"Disk Read: {resource_metrics.get('disk_io_read_mb', 0):.1f}MB")
            print(f"Disk Write: {resource_metrics.get('disk_io_write_mb', 0):.1f}MB")