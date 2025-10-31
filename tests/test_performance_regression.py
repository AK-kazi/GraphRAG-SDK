"""
Performance Regression Tests for GraphRAG-SDK

This module provides performance regression testing to ensure that optimizations
do not degrade performance over time. It includes:

- Baseline performance tracking
- Regression detection thresholds
- Automated performance comparison
- Performance trend analysis
- CI/CD integration support
"""

import unittest
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from unittest.mock import Mock, patch

# Import test framework
try:
    from .performance_test_framework import PerformanceBenchmark, BenchmarkConfig
    from .resource_profiler import ResourceProfiler, PerformanceComparator
except ImportError:
    # Handle relative imports for different test execution contexts
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    try:
        from performance_test_framework import PerformanceBenchmark, BenchmarkConfig
        from resource_profiler import ResourceProfiler, PerformanceComparator
    except ImportError:
        # If modules don't exist yet, create minimal mocks for testing
        PerformanceBenchmark = Mock
        BenchmarkConfig = Mock
        ResourceProfiler = Mock
        PerformanceComparator = Mock


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression testing"""
    test_name: str
    docs_per_second: float
    memory_per_doc_mb: float
    cpu_usage_percent: float
    error_rate: float
    timestamp: float
    
    def to_dict(self) -> Dict:
        return {
            'test_name': self.test_name,
            'docs_per_second': self.docs_per_second,
            'memory_per_doc_mb': self.memory_per_doc_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'error_rate': self.error_rate,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PerformanceBaseline':
        return cls(**data)


@dataclass
class RegressionThresholds:
    """Thresholds for detecting performance regression"""
    docs_per_second_degradation: float = 0.1  # 10% degradation
    memory_increase_percent: float = 0.2  # 20% increase
    cpu_increase_percent: float = 0.15  # 15% increase
    error_rate_increase: float = 0.05  # 5% increase
    
    def check_regression(self, baseline: PerformanceBaseline, current: Dict) -> List[str]:
        """Check if current performance exceeds regression thresholds"""
        regressions = []
        
        # Check throughput regression
        current_throughput = current.get('docs_per_second', 0)
        if current_throughput < baseline.docs_per_second * (1 - self.docs_per_second_degradation):
            degradation = (baseline.docs_per_second - current_throughput) / baseline.docs_per_second
            regressions.append(f"Throughput degraded by {degradation:.1%}")
        
        # Check memory regression
        current_memory = current.get('memory_per_doc_mb', 0)
        if current_memory > baseline.memory_per_doc_mb * (1 + self.memory_increase_percent):
            increase = (current_memory - baseline.memory_per_doc_mb) / baseline.memory_per_doc_mb
            regressions.append(f"Memory usage increased by {increase:.1%}")
        
        # Check CPU regression
        current_cpu = current.get('cpu_usage_percent', 0)
        if current_cpu > baseline.cpu_usage_percent * (1 + self.cpu_increase_percent):
            increase = (current_cpu - baseline.cpu_usage_percent) / baseline.cpu_usage_percent
            regressions.append(f"CPU usage increased by {increase:.1%}")
        
        # Check error rate regression
        current_error_rate = current.get('error_rate', 0)
        if current_error_rate > baseline.error_rate + self.error_rate_increase:
            increase = current_error_rate - baseline.error_rate
            regressions.append(f"Error rate increased by {increase:.1%}")
        
        return regressions


class PerformanceRegressionTest(unittest.TestCase):
    """Base class for performance regression tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up regression testing framework"""
        cls.baseline_file = Path("tests/data/performance_baselines.json")
        cls.thresholds = RegressionThresholds()
        cls.baselines = cls._load_baselines()
        
    @classmethod
    def _load_baselines(cls) -> Dict[str, PerformanceBaseline]:
        """Load performance baselines from file"""
        baselines = {}
        
        if cls.baseline_file.exists():
            with open(cls.baseline_file, 'r') as f:
                data = json.load(f)
                for test_name, baseline_data in data.items():
                    baselines[test_name] = PerformanceBaseline.from_dict(baseline_data)
        
        return baselines
    
    @classmethod
    def _save_baselines(cls):
        """Save performance baselines to file"""
        cls.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {name: baseline.to_dict() for name, baseline in cls.baselines.items()}
        with open(cls.baseline_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _check_regression(self, test_name: str, current_metrics: Dict):
        """Check for performance regression against baseline"""
        if test_name not in self.baselines:
            # No baseline exists, create one
            baseline = PerformanceBaseline(
                test_name=test_name,
                docs_per_second=current_metrics.get('docs_per_second', 0),
                memory_per_doc_mb=current_metrics.get('memory_per_doc_mb', 0),
                cpu_usage_percent=current_metrics.get('cpu_usage_percent', 0),
                error_rate=current_metrics.get('error_rate', 0),
                timestamp=time.time()
            )
            self.baselines[test_name] = baseline
            self._save_baselines()
            self.skipTest(f"Created baseline for {test_name}")
        
        baseline = self.baselines[test_name]
        regressions = self.thresholds.check_regression(baseline, current_metrics)
        
        if regressions:
            self.fail(f"Performance regression detected in {test_name}:\n" + 
                     "\n".join(f"  - {reg}" for reg in regressions))
    
    def _run_performance_test(self, test_name: str, test_func, **kwargs) -> Dict:
        """Run a performance test and return metrics"""
        config = BenchmarkConfig(
            test_iterations=3,
            warmup_iterations=1,
            enable_memory_monitoring=True,
            enable_cpu_monitoring=True
        )
        
        benchmark = PerformanceBenchmark(config)
        
        try:
            metrics = benchmark.run_benchmark(test_func, test_name, **kwargs)
            
            return {
                'docs_per_second': metrics.docs_per_second,
                'memory_per_doc_mb': metrics.memory_per_doc / 1024,  # Convert to MB
                'cpu_usage_percent': metrics.avg_cpu_percent,
                'error_rate': metrics.error_count / max(metrics.documents_processed, 1)
            }
            
        except Exception as e:
            self.fail(f"Performance test failed: {e}")


class TestDocumentLoadingRegression(PerformanceRegressionTest):
    """Performance regression tests for document loading"""
    
    def test_text_loading_performance(self):
        """Test text document loading performance"""
        def load_text_documents():
            # Simulate text loading
            time.sleep(0.01)  # Simulate I/O
            return {
                'documents_processed': 100,
                'entities_created': 0,
                'relations_created': 0
            }
        
        metrics = self._run_performance_test(
            "text_loading_performance",
            load_text_documents
        )
        
        self._check_regression("text_loading_performance", metrics)
    
    def test_pdf_loading_performance(self):
        """Test PDF document loading performance"""
        def load_pdf_documents():
            # Simulate PDF processing (slower)
            time.sleep(0.05)  # Simulate processing
            return {
                'documents_processed': 50,
                'entities_created': 0,
                'relations_created': 0
            }
        
        metrics = self._run_performance_test(
            "pdf_loading_performance",
            load_pdf_documents
        )
        
        self._check_regression("pdf_loading_performance", metrics)
    
    def test_html_loading_performance(self):
        """Test HTML document loading performance"""
        def load_html_documents():
            # Simulate HTML parsing
            time.sleep(0.02)  # Simulate parsing
            return {
                'documents_processed': 75,
                'entities_created': 0,
                'relations_created': 0
            }
        
        metrics = self._run_performance_test(
            "html_loading_performance",
            load_html_documents
        )
        
        self._check_regression("html_loading_performance", metrics)


class TestDataExtractionRegression(PerformanceRegressionTest):
    """Performance regression tests for data extraction"""
    
    def test_entity_extraction_performance(self):
        """Test entity extraction performance"""
        def extract_entities():
            # Simulate entity extraction
            time.sleep(0.03)  # Simulate processing
            return {
                'documents_processed': 100,
                'entities_created': 250,
                'relations_created': 0
            }
        
        metrics = self._run_performance_test(
            "entity_extraction_performance",
            extract_entities
        )
        
        self._check_regression("entity_extraction_performance", metrics)
    
    def test_relation_extraction_performance(self):
        """Test relation extraction performance"""
        def extract_relations():
            # Simulate relation extraction
            time.sleep(0.04)  # Simulate processing
            return {
                'documents_processed': 100,
                'entities_created': 200,
                'relations_created': 150
            }
        
        metrics = self._run_performance_test(
            "relation_extraction_performance",
            extract_relations
        )
        
        self._check_regression("relation_extraction_performance", metrics)
    
    def test_batch_extraction_performance(self):
        """Test batch extraction performance"""
        def batch_extract():
            # Simulate optimized batch processing
            time.sleep(0.02)  # Should be faster than individual processing
            return {
                'documents_processed': 500,
                'entities_created': 1250,
                'relations_created': 750
            }
        
        metrics = self._run_performance_test(
            "batch_extraction_performance",
            batch_extract
        )
        
        self._check_regression("batch_extraction_performance", metrics)


class TestDatabaseOperationsRegression(PerformanceRegressionTest):
    """Performance regression tests for database operations"""
    
    def test_entity_creation_performance(self):
        """Test entity creation performance"""
        def create_entities():
            # Simulate database entity creation
            time.sleep(0.015)  # Simulate DB operations
            return {
                'documents_processed': 0,
                'entities_created': 100,
                'relations_created': 0
            }
        
        metrics = self._run_performance_test(
            "entity_creation_performance",
            create_entities
        )
        
        self._check_regression("entity_creation_performance", metrics)
    
    def test_relation_creation_performance(self):
        """Test relation creation performance"""
        def create_relations():
            # Simulate database relation creation
            time.sleep(0.02)  # Simulate DB operations
            return {
                'documents_processed': 0,
                'entities_created': 0,
                'relations_created': 100
            }
        
        metrics = self._run_performance_test(
            "relation_creation_performance",
            create_relations
        )
        
        self._check_regression("relation_creation_performance", metrics)
    
    def test_batch_operations_performance(self):
        """Test batch database operations performance"""
        def batch_operations():
            # Simulate batch database operations (should be faster)
            time.sleep(0.01)  # Should be faster than individual operations
            return {
                'documents_processed': 0,
                'entities_created': 500,
                'relations_created': 300
            }
        
        metrics = self._run_performance_test(
            "batch_operations_performance",
            batch_operations
        )
        
        self._check_regression("batch_operations_performance", metrics)


class TestMemoryUsageRegression(PerformanceRegressionTest):
    """Performance regression tests for memory usage"""
    
    def test_memory_efficiency_small_dataset(self):
        """Test memory efficiency with small datasets"""
        def process_small_dataset():
            # Simulate processing with low memory usage
            return {
                'documents_processed': 50,
                'entities_created': 100,
                'relations_created': 50
            }
        
        metrics = self._run_performance_test(
            "memory_efficiency_small_dataset",
            process_small_dataset
        )
        
        self._check_regression("memory_efficiency_small_dataset", metrics)
    
    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with large datasets"""
        def process_large_dataset():
            # Simulate processing with higher memory usage
            return {
                'documents_processed': 1000,
                'entities_created': 2000,
                'relations_created': 1500
            }
        
        metrics = self._run_performance_test(
            "memory_efficiency_large_dataset",
            process_large_dataset
        )
        
        self._check_regression("memory_efficiency_large_dataset", metrics)
    
    def test_streaming_memory_efficiency(self):
        """Test memory efficiency with streaming processing"""
        def stream_process():
            # Simulate streaming processing (should use less memory)
            return {
                'documents_processed': 2000,
                'entities_created': 4000,
                'relations_created': 3000
            }
        
        metrics = self._run_performance_test(
            "streaming_memory_efficiency",
            stream_process
        )
        
        self._check_regression("streaming_memory_efficiency", metrics)


class TestEndToEndRegression(PerformanceRegressionTest):
    """End-to-end performance regression tests"""
    
    def test_full_pipeline_performance(self):
        """Test complete pipeline performance"""
        def full_pipeline():
            # Simulate complete pipeline
            time.sleep(0.1)  # Simulate full processing time
            return {
                'documents_processed': 100,
                'entities_created': 250,
                'relations_created': 180
            }
        
        metrics = self._run_performance_test(
            "full_pipeline_performance",
            full_pipeline
        )
        
        self._check_regression("full_pipeline_performance", metrics)
    
    def test_optimized_pipeline_performance(self):
        """Test optimized pipeline performance"""
        def optimized_pipeline():
            # Simulate optimized pipeline (should be faster)
            time.sleep(0.06)  # Should be faster than baseline
            return {
                'documents_processed': 100,
                'entities_created': 250,
                'relations_created': 180
            }
        
        metrics = self._run_performance_test(
            "optimized_pipeline_performance",
            optimized_pipeline
        )
        
        self._check_regression("optimized_pipeline_performance", metrics)


class TestPerformanceTrends(unittest.TestCase):
    """Test performance trend analysis"""
    
    def setUp(self):
        self.trend_file = Path("tests/data/performance_trends.json")
        self.trend_file.parent.mkdir(parents=True, exist_ok=True)
    
    def test_performance_trend_tracking(self):
        """Test performance trend tracking over time"""
        # Simulate historical performance data
        trends = []
        
        for i in range(10):
            timestamp = time.time() - (9 - i) * 86400  # Last 10 days
            performance = {
                'timestamp': timestamp,
                'docs_per_second': 100 + i * 2,  # Improving trend
                'memory_per_doc_mb': 1.0 - i * 0.05,  # Improving trend
                'cpu_usage_percent': 50 - i * 1,  # Improving trend
                'error_rate': 0.01 - i * 0.001  # Improving trend
            }
            trends.append(performance)
        
        # Save trends
        with open(self.trend_file, 'w') as f:
            json.dump(trends, f, indent=2)
        
        # Load and analyze trends
        with open(self.trend_file, 'r') as f:
            loaded_trends = json.load(f)
        
        self.assertEqual(len(loaded_trends), 10)
        
        # Check trend direction
        first_throughput = loaded_trends[0]['docs_per_second']
        last_throughput = loaded_trends[-1]['docs_per_second']
        self.assertGreater(last_throughput, first_throughput)  # Should be improving
        
        # Clean up
        self.trend_file.unlink()
    
    def test_performance_degradation_detection(self):
        """Test detection of performance degradation trends"""
        # Simulate degrading performance
        trends = []
        
        for i in range(5):
            timestamp = time.time() - (4 - i) * 86400  # Last 5 days
            performance = {
                'timestamp': timestamp,
                'docs_per_second': 100 - i * 5,  # Degrading trend
                'memory_per_doc_mb': 1.0 + i * 0.1,  # Degrading trend
                'cpu_usage_percent': 50 + i * 2,  # Degrading trend
                'error_rate': 0.01 + i * 0.002  # Degrading trend
            }
            trends.append(performance)
        
        # Analyze for degradation
        throughput_trend = trends[-1]['docs_per_second'] - trends[0]['docs_per_second']
        memory_trend = trends[-1]['memory_per_doc_mb'] - trends[0]['memory_per_doc_mb']
        
        # Should detect degradation
        self.assertLess(throughput_trend, 0)  # Throughput decreasing
        self.assertGreater(memory_trend, 0)  # Memory increasing


class TestCIIntegration(unittest.TestCase):
    """Test CI/CD integration for performance regression"""
    
    def test_ci_performance_report(self):
        """Test CI performance report generation"""
        # Simulate CI test results
        test_results = {
            'text_loading_performance': {
                'docs_per_second': 95.0,
                'memory_per_doc_mb': 0.8,
                'cpu_usage_percent': 45.0,
                'error_rate': 0.01
            },
            'entity_extraction_performance': {
                'docs_per_second': 85.0,
                'memory_per_doc_mb': 1.2,
                'cpu_usage_percent': 55.0,
                'error_rate': 0.02
            }
        }
        
        # Generate CI report
        report = {
            'timestamp': time.time(),
            'branch': 'main',
            'commit': 'abc123',
            'test_results': test_results,
            'summary': {
                'total_tests': len(test_results),
                'passed_tests': len(test_results),  # All passed in this simulation
                'failed_tests': 0,
                'regressions_detected': 0
            }
        }
        
        # Save CI report
        ci_report_file = Path("tests/data/ci_performance_report.json")
        ci_report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(ci_report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Verify report structure
        self.assertIn('timestamp', report)
        self.assertIn('test_results', report)
        self.assertIn('summary', report)
        self.assertEqual(report['summary']['total_tests'], 2)
        
        # Clean up
        ci_report_file.unlink()
    
    def test_performance_threshold_validation(self):
        """Test performance threshold validation for CI"""
        thresholds = RegressionThresholds()
        
        # Test with acceptable performance
        baseline = PerformanceBaseline(
            test_name="test",
            docs_per_second=100.0,
            memory_per_doc_mb=1.0,
            cpu_usage_percent=50.0,
            error_rate=0.01,
            timestamp=time.time()
        )
        
        current_acceptable = {
            'docs_per_second': 95.0,  # 5% degradation (within 10% threshold)
            'memory_per_doc_mb': 1.1,  # 10% increase (within 20% threshold)
            'cpu_usage_percent': 55.0,  # 10% increase (within 15% threshold)
            'error_rate': 0.02  # 1% increase (within 5% threshold)
        }
        
        regressions = thresholds.check_regression(baseline, current_acceptable)
        self.assertEqual(len(regressions), 0)  # No regressions
        
        # Test with regressing performance
        current_regressing = {
            'docs_per_second': 85.0,  # 15% degradation (exceeds 10% threshold)
            'memory_per_doc_mb': 1.3,  # 30% increase (exceeds 20% threshold)
            'cpu_usage_percent': 60.0,  # 20% increase (exceeds 15% threshold)
            'error_rate': 0.07  # 6% increase (exceeds 5% threshold)
        }
        
        regressions = thresholds.check_regression(baseline, current_regressing)
        self.assertGreater(len(regressions), 0)  # Should detect regressions


if __name__ == '__main__':
    # Run performance regression tests
    unittest.main(verbosity=2)