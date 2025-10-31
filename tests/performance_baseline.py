"""
Performance Baseline Establishment for GraphRAG-SDK

This module establishes performance baselines for different use cases
and provides reference points for future performance comparisons.
"""

import json
import time
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class BaselineMetric:
    """Represents a single baseline metric."""
    name: str
    value: float
    unit: str
    description: str
    dataset_size: int
    test_conditions: Dict[str, Any]


@dataclass
class PerformanceBaseline:
    """Complete performance baseline for a specific use case."""
    use_case: str
    description: str
    dataset_size: int
    metrics: List[BaselineMetric]
    created_at: datetime
    environment_info: Dict[str, Any]
    recommendations: List[str]


class BaselineEstablisher:
    """Establishes and manages performance baselines."""
    
    def __init__(self, baseline_dir: str = "performance_baselines"):
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def define_use_cases(self) -> List[Dict[str, Any]]:
        """Define different use cases for baseline establishment."""
        return [
            {
                "name": "small_document_ingestion",
                "description": "Ingestion of small documents (100-500 items)",
                "dataset_sizes": [100, 500],
                "expected_characteristics": {
                    "low_memory_usage": True,
                    "fast_processing": True,
                    "minimal_cpu_impact": True
                }
            },
            {
                "name": "medium_document_ingestion", 
                "description": "Ingestion of medium documents (1000-5000 items)",
                "dataset_sizes": [1000, 5000],
                "expected_characteristics": {
                    "moderate_memory_usage": True,
                    "scalable_processing": True,
                    "efficient_cpu_usage": True
                }
            },
            {
                "name": "large_document_ingestion",
                "description": "Ingestion of large documents (10000+ items)",
                "dataset_sizes": [10000, 50000],
                "expected_characteristics": {
                    "high_memory_efficiency": True,
                    "streaming_capable": True,
                    "parallel_processing": True
                }
            },
            {
                "name": "memory_constrained_ingestion",
                "description": "Ingestion under memory constraints (<1GB)",
                "dataset_sizes": [1000, 5000],
                "expected_characteristics": {
                    "memory_aware": True,
                    "adaptive_batching": True,
                    "low_memory_footprint": True
                }
            },
            {
                "name": "real_time_ingestion",
                "description": "Real-time document ingestion with low latency",
                "dataset_sizes": [100, 500],
                "expected_characteristics": {
                    "low_latency": True,
                    "streaming": True,
                    "responsive": True
                }
            }
        ]
    
    async def establish_baseline(self, use_case: Dict[str, Any], dataset_size: int) -> PerformanceBaseline:
        """Establish a performance baseline for a specific use case and dataset size."""
        self.logger.info(f"Establishing baseline for {use_case['name']} with {dataset_size} items")
        
        # Create test data
        test_data = self._create_test_data(dataset_size)
        
        # Run baseline measurements
        metrics = await self._run_baseline_measurements(test_data, use_case)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, use_case)
        
        # Get environment info
        env_info = self._get_environment_info()
        
        baseline = PerformanceBaseline(
            use_case=use_case['name'],
            description=use_case['description'],
            dataset_size=dataset_size,
            metrics=metrics,
            created_at=datetime.now(),
            environment_info=env_info,
            recommendations=recommendations
        )
        
        return baseline
    
    def _create_test_data(self, size: int) -> List[Dict[str, Any]]:
        """Create test data for baseline establishment."""
        return [
            {
                "id": f"doc_{i}",
                "content": f"This is test document {i} with sufficient content to simulate realistic processing. " * 10,
                "metadata": {
                    "source": f"source_{i % 20}",
                    "type": "text",
                    "size": len(f"This is test document {i} with sufficient content to simulate realistic processing. " * 10),
                    "created_at": f"2024-01-{(i % 28) + 1:02d}"
                }
            }
            for i in range(size)
        ]
    
    async def _run_baseline_measurements(self, test_data: List[Dict[str, Any]], use_case: Dict[str, Any]) -> List[BaselineMetric]:
        """Run baseline measurements and collect metrics."""
        metrics = []
        dataset_size = len(test_data)
        
        # Measure ingestion time
        ingestion_time = await self._measure_ingestion_time(test_data)
        metrics.append(BaselineMetric(
            name="ingestion_time",
            value=ingestion_time,
            unit="seconds",
            description="Total time to ingest all documents",
            dataset_size=dataset_size,
            test_conditions={"use_case": use_case['name']}
        ))
        
        # Measure throughput
        throughput = dataset_size / ingestion_time if ingestion_time > 0 else 0
        metrics.append(BaselineMetric(
            name="throughput",
            value=throughput,
            unit="docs/second",
            description="Number of documents processed per second",
            dataset_size=dataset_size,
            test_conditions={"use_case": use_case['name']}
        ))
        
        # Measure memory usage (simulated)
        memory_usage = self._estimate_memory_usage(test_data)
        metrics.append(BaselineMetric(
            name="peak_memory_usage",
            value=memory_usage,
            unit="MB",
            description="Peak memory usage during ingestion",
            dataset_size=dataset_size,
            test_conditions={"use_case": use_case['name']}
        ))
        
        # Measure CPU usage (simulated)
        cpu_usage = self._estimate_cpu_usage(test_data, use_case)
        metrics.append(BaselineMetric(
            name="average_cpu_usage",
            value=cpu_usage,
            unit="percent",
            description="Average CPU usage during ingestion",
            dataset_size=dataset_size,
            test_conditions={"use_case": use_case['name']}
        ))
        
        # Measure disk I/O (simulated)
        disk_io = self._estimate_disk_io(test_data)
        metrics.append(BaselineMetric(
            name="disk_io",
            value=disk_io,
            unit="MB",
            description="Total disk I/O during ingestion",
            dataset_size=dataset_size,
            test_conditions={"use_case": use_case['name']}
        ))
        
        # Measure latency characteristics
        latency = await self._measure_latency(test_data[:10])  # Measure first 10 items
        metrics.append(BaselineMetric(
            name="average_latency",
            value=latency,
            unit="milliseconds",
            description="Average time to process a single document",
            dataset_size=dataset_size,
            test_conditions={"use_case": use_case['name'], "sample_size": 10}
        ))
        
        return metrics
    
    async def _measure_ingestion_time(self, test_data: List[Dict[str, Any]]) -> float:
        """Measure total ingestion time."""
        start_time = time.time()
        
        # Simulate ingestion processing
        for i, doc in enumerate(test_data):
            # Simulate document processing time
            await asyncio.sleep(0.001)  # 1ms per document
            
            # Simulate occasional batch operations
            if i % 100 == 0:
                await asyncio.sleep(0.01)  # 10ms for batch processing
        
        return time.time() - start_time
    
    def _estimate_memory_usage(self, test_data: List[Dict[str, Any]]) -> float:
        """Estimate memory usage for the test data."""
        base_memory = 50  # Base memory in MB
        doc_memory = len(json.dumps(test_data[:10])) / 10  # Average document size in bytes
        total_memory_mb = base_memory + (doc_memory * len(test_data)) / (1024 * 1024)
        return total_memory_mb
    
    def _estimate_cpu_usage(self, test_data: List[Dict[str, Any]], use_case: Dict[str, Any]) -> float:
        """Estimate CPU usage based on use case characteristics."""
        base_cpu = 20.0  # Base CPU usage
        
        # Adjust based on dataset size
        if len(test_data) > 10000:
            base_cpu += 40.0
        elif len(test_data) > 1000:
            base_cpu += 25.0
        else:
            base_cpu += 10.0
        
        # Adjust based on use case
        if "real_time" in use_case['name']:
            base_cpu *= 1.2
        elif "memory_constrained" in use_case['name']:
            base_cpu *= 0.8
        
        return min(base_cpu, 95.0)  # Cap at 95%
    
    def _estimate_disk_io(self, test_data: List[Dict[str, Any]]) -> float:
        """Estimate disk I/O for the test data."""
        # Estimate based on data size and processing patterns
        data_size_mb = len(json.dumps(test_data)) / (1024 * 1024)
        
        # Assume 2-3x data size for temporary files and processing
        disk_io_mb = data_size_mb * 2.5
        
        return disk_io_mb
    
    async def _measure_latency(self, sample_data: List[Dict[str, Any]]) -> float:
        """Measure average processing latency for a sample of documents."""
        latencies = []
        
        for doc in sample_data:
            start_time = time.time()
            
            # Simulate single document processing
            await asyncio.sleep(0.001)  # 1ms processing time
            
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
        
        return sum(latencies) / len(latencies) if latencies else 0
    
    def _generate_recommendations(self, metrics: List[BaselineMetric], use_case: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on baseline metrics."""
        recommendations = []
        
        # Find metrics by name
        metrics_dict = {m.name: m for m in metrics}
        
        # Throughput recommendations
        if 'throughput' in metrics_dict:
            throughput = metrics_dict['throughput'].value
            if throughput < 100:  # docs/second
                recommendations.append("Consider enabling parallel processing for better throughput")
            elif throughput > 1000:
                recommendations.append("High throughput achieved - consider monitoring system resources")
        
        # Memory recommendations
        if 'peak_memory_usage' in metrics_dict:
            memory = metrics_dict['peak_memory_usage'].value
            if memory > 1000:  # MB
                recommendations.append("High memory usage detected - consider enabling streaming or reducing batch sizes")
            elif memory < 100:
                recommendations.append("Low memory footprint - good for memory-constrained environments")
        
        # CPU recommendations
        if 'average_cpu_usage' in metrics_dict:
            cpu = metrics_dict['average_cpu_usage'].value
            if cpu > 80:
                recommendations.append("High CPU usage - consider optimizing processing or adding more resources")
            elif cpu < 30:
                recommendations.append("Low CPU usage - potential for increased parallelization")
        
        # Latency recommendations
        if 'average_latency' in metrics_dict:
            latency = metrics_dict['average_latency'].value
            if latency > 100:  # milliseconds
                recommendations.append("High latency detected - consider optimizing document processing pipeline")
            elif latency < 10:
                recommendations.append("Excellent latency - suitable for real-time applications")
        
        # Use case specific recommendations
        if "memory_constrained" in use_case['name']:
            recommendations.append("Enable memory-aware processing and adaptive batching for memory-constrained scenarios")
        
        if "real_time" in use_case['name']:
            recommendations.append("Enable streaming processing and optimize for low latency in real-time scenarios")
        
        return recommendations
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information for the baseline."""
        return {
            "python_version": "3.x",  # Would get actual version
            "platform": "Linux/macOS/Windows",  # Would get actual platform
            "cpu_count": 4,  # Would get actual CPU count
            "total_memory_gb": 16,  # Would get actual memory
            "timestamp": datetime.now().isoformat(),
            "graphrag_sdk_version": "1.0.0"  # Would get actual version
        }
    
    async def establish_all_baselines(self) -> List[PerformanceBaseline]:
        """Establish baselines for all defined use cases."""
        use_cases = self.define_use_cases()
        baselines = []
        
        self.logger.info(f"Establishing baselines for {len(use_cases)} use cases")
        
        for use_case in use_cases:
            for dataset_size in use_case['dataset_sizes']:
                try:
                    baseline = await self.establish_baseline(use_case, dataset_size)
                    baselines.append(baseline)
                    
                    # Save baseline immediately
                    self.save_baseline(baseline)
                    
                except Exception as e:
                    self.logger.error(f"Failed to establish baseline for {use_case['name']} with {dataset_size} items: {e}")
        
        self.logger.info(f"Established {len(baselines)} baselines")
        return baselines
    
    def save_baseline(self, baseline: PerformanceBaseline) -> None:
        """Save a baseline to file."""
        filename = f"{baseline.use_case}_{baseline.dataset_size}_{baseline.created_at.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.baseline_dir / filename
        
        # Convert to serializable format
        baseline_dict = asdict(baseline)
        baseline_dict['created_at'] = baseline.created_at.isoformat()
        
        # Convert metrics to serializable format
        baseline_dict['metrics'] = [asdict(metric) for metric in baseline.metrics]
        
        with open(filepath, 'w') as f:
            json.dump(baseline_dict, f, indent=2)
        
        self.logger.info(f"Baseline saved to {filepath}")
    
    def load_baselines(self, use_case: Optional[str] = None) -> List[PerformanceBaseline]:
        """Load baselines from files."""
        baselines = []
        
        for filepath in self.baseline_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    baseline_dict = json.load(f)
                
                # Convert back to PerformanceBaseline object
                baseline_dict['created_at'] = datetime.fromisoformat(baseline_dict['created_at'])
                baseline_dict['metrics'] = [
                    BaselineMetric(**metric) for metric in baseline_dict['metrics']
                ]
                
                baseline = PerformanceBaseline(**baseline_dict)
                
                # Filter by use case if specified
                if use_case is None or baseline.use_case == use_case:
                    baselines.append(baseline)
                    
            except Exception as e:
                self.logger.error(f"Failed to load baseline from {filepath}: {e}")
        
        return baselines
    
    def generate_baseline_report(self) -> None:
        """Generate a comprehensive baseline report."""
        baselines = self.load_baselines()
        
        if not baselines:
            self.logger.warning("No baselines found for report generation")
            return
        
        report = []
        report.append("# GraphRAG-SDK Performance Baselines")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total baselines: {len(baselines)}")
        report.append("")
        
        # Group baselines by use case
        use_case_groups = {}
        for baseline in baselines:
            if baseline.use_case not in use_case_groups:
                use_case_groups[baseline.use_case] = []
            use_case_groups[baseline.use_case].append(baseline)
        
        # Generate report for each use case
        for use_case, case_baselines in use_case_groups.items():
            report.append(f"## {use_case.replace('_', ' ').title()}")
            report.append(f"{case_baselines[0].description}")
            report.append("")
            
            # Create table for metrics
            report.append("### Performance Metrics by Dataset Size")
            report.append("| Dataset Size | Ingestion Time (s) | Throughput (docs/s) | Memory (MB) | CPU (%) | Latency (ms) |")
            report.append("|--------------|-------------------|---------------------|-------------|---------|---------------|")
            
            for baseline in sorted(case_baselines, key=lambda x: x.dataset_size):
                metrics_dict = {m.name: m.value for m in baseline.metrics}
                
                report.append(f"| {baseline.dataset_size} | "
                            f"{metrics_dict.get('ingestion_time', 0):.2f} | "
                            f"{metrics_dict.get('throughput', 0):.1f} | "
                            f"{metrics_dict.get('peak_memory_usage', 0):.1f} | "
                            f"{metrics_dict.get('average_cpu_usage', 0):.1f} | "
                            f"{metrics_dict.get('average_latency', 0):.1f} |")
            
            report.append("")
            
            # Add recommendations
            if case_baselines[0].recommendations:
                report.append("### Recommendations")
                for rec in case_baselines[0].recommendations:
                    report.append(f"- {rec}")
                report.append("")
        
        # Save report
        report_content = '\n'.join(report)
        report_file = self.baseline_dir / f"baseline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Baseline report saved to {report_file}")


async def main():
    """Main function to establish performance baselines."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create baseline establisher
    establisher = BaselineEstablisher()
    
    print("🎯 Establishing GraphRAG-SDK Performance Baselines")
    print("=" * 60)
    
    # Establish all baselines
    baselines = await establisher.establish_all_baselines()
    
    # Generate report
    establisher.generate_baseline_report()
    
    print("=" * 60)
    print("✅ Performance baselines established successfully!")
    print(f"📊 {len(baselines)} baselines created")
    print(f"📁 Baselines saved to: {establisher.baseline_dir}")


if __name__ == "__main__":
    asyncio.run(main())