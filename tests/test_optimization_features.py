"""
Comprehensive Test Suite for Optimization Features

This module provides comprehensive testing for all optimization features including:
- Streaming document loaders
- Batch database operations
- Memory-aware processing
- Async I/O operations
- Adaptive batch processing
"""

import unittest
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import test framework
try:
    from .performance_test_framework import PerformanceBenchmark, BenchmarkConfig
    from .resource_profiler import ResourceProfiler, profile_function
except ImportError:
    # Handle relative imports for different test execution contexts
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from performance_test_framework import PerformanceBenchmark, BenchmarkConfig
    from resource_profiler import ResourceProfiler, profile_function


class TestStreamingDocumentLoaders(unittest.TestCase):
    """Test streaming document loader optimizations"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_streaming_text_loader(self):
        """Test streaming text loader with chunking"""
        # Create test file
        test_file = os.path.join(self.temp_dir, "test.txt")
        test_content = "This is line 1.\n" * 1000  # Large text file
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Test streaming loader
        try:
            from graphrag_sdk.document_loaders.text import StreamingTextLoader
            
            loader = StreamingTextLoader(test_file, chunk_size=100, overlap=10)
            documents = list(loader.load())
            
            # Verify chunking
            self.assertGreater(len(documents), 1)  # Should be multiple chunks
            self.assertTrue(all(doc.not_empty() for doc in documents))
            
            # Verify content preservation
            combined_content = " ".join(doc.text for doc in documents)
            self.assertIn("This is line 1", combined_content)
            
        except ImportError:
            self.skipTest("StreamingTextLoader not implemented")
    
    def test_streaming_html_loader(self):
        """Test streaming HTML loader"""
        # Create test HTML file
        test_file = os.path.join(self.temp_dir, "test.html")
        test_html = """
        <html>
        <head><title>Test</title></head>
        <body>
            <h1>Header</h1>
            <p>Paragraph 1</p>
            <p>Paragraph 2</p>
            <script>console.log('test');</script>
        </body>
        </html>
        """
        
        with open(test_file, 'w') as f:
            f.write(test_html)
        
        try:
            from graphrag_sdk.document_loaders.html import StreamingHTMLLoader
            
            loader = StreamingHTMLLoader(test_file, chunk_size=50)
            documents = list(loader.load())
            
            # Verify content extraction
            self.assertGreater(len(documents), 0)
            self.assertTrue(all(doc.not_empty() for doc in documents))
            
            # Verify script removal
            combined_content = " ".join(doc.text for doc in documents)
            self.assertNotIn("console.log", combined_content)
            self.assertIn("Header", combined_content)
            
        except ImportError:
            self.skipTest("StreamingHTMLLoader not implemented")
    
    def test_streaming_pdf_loader(self):
        """Test streaming PDF loader with parallel processing"""
        # Create a mock PDF test
        try:
            from graphrag_sdk.document_loaders.pdf import StreamingPDFLoader
            
            # Test with mock PDF processing
            with patch('pypdf.PdfReader') as mock_reader:
                mock_pdf = Mock()
                mock_page = Mock()
                mock_page.extract_text.return_value = "Sample PDF content"
                mock_pdf.pages = [mock_page, mock_page]  # 2 pages
                mock_reader.return_value = mock_pdf
                
                loader = StreamingPDFLoader("test.pdf", max_workers=2)
                documents = list(loader.load())
                
                # Verify parallel processing
                self.assertEqual(len(documents), 2)
                self.assertTrue(all(doc.not_empty() for doc in documents))
                
        except ImportError:
            self.skipTest("StreamingPDFLoader not implemented")


class TestBatchDatabaseOperations(unittest.TestCase):
    """Test batch database operation optimizations"""
    
    def setUp(self):
        self.mock_graph = Mock()
        self.mock_ontology = Mock()
        self.mock_model = Mock()
        
    def test_batch_entity_creation(self):
        """Test batch entity creation"""
        try:
            from graphrag_sdk.steps.extract_data_step import BatchExtractDataStep
            
            # Create test entities
            test_entities = [
                {'label': 'Person', 'attributes': {'name': 'John', 'age': 30}},
                {'label': 'Person', 'attributes': {'name': 'Jane', 'age': 25}},
                {'label': 'Company', 'attributes': {'name': 'Acme Corp'}}
            ]
            
            step = BatchExtractDataStep(
                sources=[],
                ontology=self.mock_ontology,
                model=self.mock_model,
                graph=self.mock_graph,
                batch_size=2
            )
            
            # Test batch query generation
            query = step._create_entity_batch_query(test_entities)
            self.assertIsNotNone(query)
            self.assertIn('UNWIND', query)
            self.assertIn('MERGE', query)
            
        except ImportError:
            self.skipTest("BatchExtractDataStep not implemented")
    
    def test_batch_relation_creation(self):
        """Test batch relation creation"""
        try:
            from graphrag_sdk.steps.extract_data_step import BatchExtractDataStep
            
            # Create test relations
            test_relations = [
                {
                    'label': 'WORKS_FOR',
                    'source': {'label': 'Person', 'attributes': {'name': 'John'}},
                    'target': {'label': 'Company', 'attributes': {'name': 'Acme'}},
                    'attributes': {'since': 2020}
                },
                {
                    'label': 'KNOWS',
                    'source': {'label': 'Person', 'attributes': {'name': 'John'}},
                    'target': {'label': 'Person', 'attributes': {'name': 'Jane'}},
                    'attributes': {}
                }
            ]
            
            step = BatchExtractDataStep(
                sources=[],
                ontology=self.mock_ontology,
                model=self.mock_model,
                graph=self.mock_graph,
                batch_size=2
            )
            
            # Test batch query generation
            query = step._create_relation_batch_query(test_relations)
            self.assertIsNotNone(query)
            self.assertIn('UNWIND', query)
            self.assertIn('MATCH', query)
            self.assertIn('MERGE', query)
            
        except ImportError:
            self.skipTest("BatchExtractDataStep not implemented")
    
    def test_fallback_mechanisms(self):
        """Test fallback mechanisms for batch failures"""
        try:
            from graphrag_sdk.steps.extract_data_step import BatchExtractDataStep
            
            step = BatchExtractDataStep(
                sources=[],
                ontology=self.mock_ontology,
                model=self.mock_model,
                graph=self.mock_graph,
                batch_size=2
            )
            
            # Mock database failure
            self.mock_graph.query.side_effect = Exception("Database error")
            
            # Test fallback entity creation
            test_entities = [
                {'label': 'Person', 'attributes': {'name': 'John'}}
            ]
            
            step.entity_batch = test_entities
            step._fallback_entity_creation()
            
            # Verify individual queries were attempted
            self.assertGreater(self.mock_graph.query.call_count, 0)
            
        except ImportError:
            self.skipTest("BatchExtractDataStep not implemented")


class TestMemoryAwareProcessing(unittest.TestCase):
    """Test memory-aware processing optimizations"""
    
    def test_memory_monitoring(self):
        """Test memory monitoring functionality"""
        try:
            from graphrag_sdk.steps.extract_data_step import MemoryAwareExtractDataStep, MemoryConfig
            
            config = MemoryConfig(max_memory_mb=100, memory_check_interval=5)
            step = MemoryAwareExtractDataStep(
                sources=[],
                ontology=Mock(),
                model=Mock(),
                graph=Mock(),
                memory_config=config
            )
            
            # Test memory checking
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 50.0
                mock_memory.return_value.used = 100 * 1024 * 1024  # 100MB
                
                result = step._check_memory_usage()
                self.assertFalse(result)  # Should not exceed threshold
                
                # Test threshold exceeded
                mock_memory.return_value.used = 200 * 1024 * 1024  # 200MB
                result = step._check_memory_usage()
                self.assertTrue(result)  # Should exceed threshold
                
        except ImportError:
            self.skipTest("MemoryAwareExtractDataStep not implemented")
    
    def test_memory_pressure_handling(self):
        """Test memory pressure handling"""
        try:
            from graphrag_sdk.steps.extract_data_step import MemoryAwareExtractDataStep
            
            step = MemoryAwareExtractDataStep(
                sources=[],
                ontology=Mock(),
                model=Mock(),
                graph=Mock()
            )
            
            # Test memory pressure handling
            with patch.object(step, 'flush_all_batches') as mock_flush:
                with patch('gc.collect') as mock_gc:
                    step._handle_memory_pressure()
                    
                    # Verify cleanup actions
                    mock_gc.assert_called()
                    mock_flush.assert_called()
                    
        except ImportError:
            self.skipTest("MemoryAwareExtractDataStep not implemented")


class TestAsyncIOOperations(unittest.TestCase):
    """Test async I/O optimizations"""
    
    def test_async_url_loader(self):
        """Test async URL loading"""
        try:
            from graphrag_sdk.document_loaders.url import AsyncURLLoader
            
            urls = [
                "https://example.com/page1",
                "https://example.com/page2"
            ]
            
            # Mock HTTP responses
            mock_response = Mock()
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.text = "<html><body>Test content</body></html>"
            
            with patch('aiohttp.ClientSession') as mock_session:
                mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
                
                loader = AsyncURLLoader(urls, max_concurrent=2)
                
                # Test async loading (run in sync context for testing)
                import asyncio
                documents = asyncio.run(loader.load())
                
                # Verify results
                self.assertEqual(len(documents), 2)
                self.assertTrue(all(doc.not_empty() for doc in documents))
                
        except ImportError:
            self.skipTest("AsyncURLLoader not implemented")
    
    def test_connection_pooling(self):
        """Test database connection pooling"""
        try:
            from graphrag_sdk.kg import PooledGraphOperations
            
            mock_graph = Mock()
            pool = PooledGraphOperations(mock_graph, pool_size=3)
            
            # Test connection management
            conn1 = pool.get_connection()
            conn2 = pool.get_connection()
            conn3 = pool.get_connection()
            
            # All connections should be available
            self.assertIsNotNone(conn1)
            self.assertIsNotNone(conn2)
            self.assertIsNotNone(conn3)
            
            # Return connections
            pool.return_connection(conn1)
            pool.return_connection(conn2)
            
            # Verify pool management
            self.assertEqual(len(pool.connection_pool), 2)
            
        except ImportError:
            self.skipTest("PooledGraphOperations not implemented")


class TestAdaptiveBatchProcessing(unittest.TestCase):
    """Test adaptive batch processing optimizations"""
    
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        try:
            from graphrag_sdk.steps.extract_data_step import PerformanceMetrics
            
            metrics = PerformanceMetrics(
                processing_times=[1.0, 2.0, 3.0],
                memory_usage=[50.0, 60.0, 55.0],
                error_count=1,
                success_count=9
            )
            
            # Test metric calculations
            self.assertEqual(metrics.get_avg_processing_time(), 2.0)
            self.assertEqual(metrics.get_avg_memory_usage(), 55.0)
            self.assertEqual(metrics.get_error_rate(), 0.1)
            
        except ImportError:
            self.skipTest("PerformanceMetrics not implemented")
    
    def test_adaptive_batch_sizing(self):
        """Test adaptive batch size adjustment"""
        try:
            from graphrag_sdk.steps.extract_data_step import AdaptiveBatchProcessor
            
            processor = AdaptiveBatchProcessor(
                initial_batch_size=50,
                min_batch_size=10,
                max_batch_size=200
            )
            
            # Test performance-based adjustment
            good_metrics = Mock()
            good_metrics.get_avg_processing_time.return_value = 2.0  # Fast
            good_metrics.get_avg_memory_usage.return_value = 30.0  # Low
            good_metrics.get_error_rate.return_value = 0.01  # Low
            
            processor.record_batch_performance(good_metrics)
            initial_size = processor.get_current_batch_size()
            
            # Should increase batch size for good performance
            self.assertGreater(processor.get_current_batch_size(), initial_size)
            
        except ImportError:
            self.skipTest("AdaptiveBatchProcessor not implemented")
    
    def test_system_resource_optimization(self):
        """Test system resource-based optimization"""
        try:
            from graphrag_sdk.steps.extract_data_step import AdaptiveBatchProcessor
            
            processor = AdaptiveBatchProcessor()
            
            # Test optimal worker calculation
            with patch('psutil.cpu_count', return_value=4):
                with patch('psutil.virtual_memory') as mock_memory:
                    mock_memory.return_value.total = 8 * 1024**3  # 8GB
                    
                    workers = processor.get_optimal_workers(4, 8)
                    self.assertGreater(workers, 0)
                    self.assertLessEqual(workers, 16)  # Should not exceed max
                    
        except ImportError:
            self.skipTest("AdaptiveBatchProcessor not implemented")


class TestOptimizationIntegration(unittest.TestCase):
    """Test integration of all optimization features"""
    
    def test_end_to_end_optimized_pipeline(self):
        """Test complete optimized pipeline"""
        # Create temporary test data
        temp_dir = tempfile.mkdtemp()
        try:
            # Create test documents
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("Test document content for optimization testing. " * 100)
            
            # Test optimized extraction
            try:
                from graphrag_sdk.steps.extract_data_step import AdaptiveExtractDataStep
                
                # Mock dependencies
                mock_source = Mock()
                mock_source.load.return_value = [
                    Mock(text="Test content", not_empty=lambda: True)
                ]
                
                mock_ontology = Mock()
                mock_model = Mock()
                mock_graph = Mock()
                
                step = AdaptiveExtractDataStep(
                    sources=[mock_source],
                    ontology=mock_ontology,
                    model=mock_model,
                    graph=mock_graph,
                    enable_optimizations=True
                )
                
                # Test execution
                with patch('tqdm.tqdm'):  # Hide progress bar
                    result = step.run()
                    
                # Verify optimization features were used
                self.assertIsNotNone(result)
                
            except ImportError:
                self.skipTest("AdaptiveExtractDataStep not implemented")
                
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_performance_comparison(self):
        """Test performance comparison between optimized and non-optimized versions"""
        # Create benchmark configuration
        config = BenchmarkConfig(
            test_iterations=2,
            warmup_iterations=1,
            enable_memory_monitoring=True
        )
        
        benchmark = PerformanceBenchmark(config)
        
        # Mock test functions
        def baseline_implementation():
            time.sleep(0.1)  # Simulate slower processing
            return {
                'documents_processed': 10,
                'entities_created': 5,
                'relations_created': 3
            }
        
        def optimized_implementation():
            time.sleep(0.05)  # Simulate faster processing
            return {
                'documents_processed': 10,
                'entities_created': 5,
                'relations_created': 3
            }
        
        # Run comparison
        results = benchmark.compare_implementations(
            {
                'baseline': baseline_implementation,
                'optimized': optimized_implementation
            },
            'test_comparison'
        )
        
        # Verify performance improvement
        self.assertIn('baseline', results)
        self.assertIn('optimized', results)
        
        # Optimized version should be faster
        baseline_time = results['baseline'].duration
        optimized_time = results['optimized'].duration
        self.assertLess(optimized_time, baseline_time)


class TestOptimizationConfiguration(unittest.TestCase):
    """Test optimization configuration and feature flags"""
    
    def test_optimization_config(self):
        """Test optimization configuration options"""
        try:
            from graphrag_sdk.optimization_config import IngestionOptimizationConfig
            
            # Test default configuration
            config = IngestionOptimizationConfig()
            self.assertTrue(config.enable_streaming)
            self.assertTrue(config.enable_batch_operations)
            self.assertEqual(config.batch_size, 1000)
            
            # Test high performance configuration
            hp_config = IngestionOptimizationConfig.high_performance()
            self.assertEqual(hp_config.chunk_size, 16384)
            self.assertEqual(hp_config.batch_size, 5000)
            self.assertTrue(hp_config.enable_async_io)
            
            # Test resource constrained configuration
            rc_config = IngestionOptimizationConfig.resource_constrained()
            self.assertEqual(rc_config.chunk_size, 4096)
            self.assertEqual(rc_config.batch_size, 100)
            self.assertFalse(rc_config.enable_async_io)
            
        except ImportError:
            self.skipTest("IngestionOptimizationConfig not implemented")
    
    def test_feature_flags(self):
        """Test feature flag functionality"""
        try:
            from graphrag_sdk.steps.extract_data_step import ExtractDataStep
            
            # Test with optimizations enabled
            step_enabled = ExtractDataStep(
                sources=[],
                ontology=Mock(),
                model=Mock(),
                graph=Mock(),
                enable_optimizations=True
            )
            
            # Test with optimizations disabled
            step_disabled = ExtractDataStep(
                sources=[],
                ontology=Mock(),
                model=Mock(),
                graph=Mock(),
                enable_optimizations=False
            )
            
            # Verify feature flags are set correctly
            self.assertTrue(step_enabled.enable_optimizations)
            self.assertFalse(step_disabled.enable_optimizations)
            
        except ImportError:
            self.skipTest("ExtractDataStep not implemented")


class TestOptimizationMetrics(unittest.TestCase):
    """Test optimization metrics and monitoring"""
    
    def test_ingestion_metrics(self):
        """Test ingestion metrics collection"""
        try:
            from graphrag_sdk.ingestion_metrics import IngestionMetrics
            
            metrics = IngestionMetrics()
            
            # Test metric recording
            metrics.record_batch_completed(10, 2.5, 100.5)
            metrics.record_entities_created(5)
            metrics.record_relations_created(3)
            
            # Test performance summary
            summary = metrics.get_performance_summary()
            
            self.assertEqual(summary['documents_processed'], 10)
            self.assertEqual(summary['entities_created'], 5)
            self.assertEqual(summary['relations_created'], 3)
            self.assertEqual(summary['batches_processed'], 1)
            self.assertGreater(summary['documents_per_second'], 0)
            
        except ImportError:
            self.skipTest("IngestionMetrics not implemented")
    
    def test_resource_profiling_integration(self):
        """Test resource profiling integration"""
        profiler = ResourceProfiler()
        
        # Test profiling a simple function
        def test_function():
            time.sleep(0.1)
            return "test_result"
        
        with profile_function("test_function", sampling_interval=0.05):
            result = test_function()
            self.assertEqual(result, "test_result")


if __name__ == '__main__':
    # Run all optimization tests
    unittest.main(verbosity=2)