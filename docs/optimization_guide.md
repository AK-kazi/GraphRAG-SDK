# GraphRAG-SDK Optimization Guide

This guide provides comprehensive documentation for optimizing the performance of GraphRAG-SDK ingestion and extraction operations.

## Overview

The GraphRAG-SDK includes several optimization features that can significantly improve performance:

- **Streaming Document Loaders**: Process large files without loading everything into memory
- **Batch Database Operations**: Reduce database round trips with batched queries
- **Memory-Aware Processing**: Monitor and adapt to memory constraints
- **Async I/O Operations**: Improve I/O throughput with asynchronous processing
- **Adaptive Batch Processing**: Dynamically adjust batch sizes based on performance

## Quick Start

### Basic Optimization

```python
from graphrag_sdk import KnowledgeGraph
from graphrag_sdk.optimization_config import IngestionOptimizationConfig

# Create optimized configuration
config = IngestionOptimizationConfig.high_performance()

# Use with KnowledgeGraph
kg = KnowledgeGraph(
    model=your_model,
    ontology=your_ontology,
    optimization_config=config
)

# Run optimized extraction
kg.extract_data_from_sources(sources)
```

### Simple Configuration

```python
from graphrag_sdk.steps.extract_data_step import ExtractDataStep

# Enable all optimizations
step = ExtractDataStep(
    sources=sources,
    ontology=ontology,
    model=model,
    graph=graph,
    enable_optimizations=True  # Enable all optimizations
)
```

## Configuration Options

### IngestionOptimizationConfig

The main configuration class for all optimization features.

```python
@dataclass
class IngestionOptimizationConfig:
    # Document loading
    enable_streaming: bool = True
    chunk_size: int = 8192
    chunk_overlap: int = 100
    max_concurrent_files: int = 10
    
    # Batch processing
    enable_batch_operations: bool = True
    batch_size: int = 1000
    adaptive_batching: bool = True
    
    # Memory management
    max_memory_mb: int = 2048
    memory_check_interval: int = 10
    enable_memory_monitoring: bool = True
    
    # Async operations
    enable_async_io: bool = True
    connection_pool_size: int = 5
    max_concurrent_requests: int = 20
    
    # Performance optimization
    enable_adaptive_processing: bool = True
    performance_window: int = 10
    adjustment_interval: int = 60
```

### Predefined Configurations

#### High Performance Configuration

```python
config = IngestionOptimizationConfig.high_performance()
```

Optimized for maximum throughput on systems with abundant resources:

- `chunk_size`: 16384 bytes
- `max_concurrent_files`: 20
- `batch_size`: 5000
- `max_memory_mb`: 8192 MB
- `enable_async_io`: True
- `connection_pool_size`: 10
- `max_concurrent_requests`: 50

#### Resource Constrained Configuration

```python
config = IngestionOptimizationConfig.resource_constrained()
```

Optimized for systems with limited resources:

- `chunk_size`: 4096 bytes
- `max_concurrent_files`: 3
- `batch_size`: 100
- `max_memory_mb`: 1024 MB
- `enable_async_io`: False
- `connection_pool_size`: 2
- `max_concurrent_requests`: 5

## Document Loading Optimizations

### Streaming Text Loader

```python
from graphrag_sdk.document_loaders.text import StreamingTextLoader

# Create streaming loader with custom chunking
loader = StreamingTextLoader(
    path="large_file.txt",
    chunk_size=8192,      # 8KB chunks
    overlap=100           # 100 character overlap
)

# Process documents as they're loaded
for document in loader.load():
    # Process document chunk
    process_document(document)
```

### Streaming HTML Loader

```python
from graphrag_sdk.document_loaders.html import StreamingHTMLLoader

# Create streaming HTML loader
loader = StreamingHTMLLoader(
    path="large_page.html",
    chunk_size=8192
)

# Process cleaned HTML chunks
for document in loader.load():
    # Scripts and styles automatically removed
    process_document(document)
```

### Streaming PDF Loader

```python
from graphrag_sdk.document_loaders.pdf import StreamingPDFLoader

# Create parallel PDF loader
loader = StreamingPDFLoader(
    path="document.pdf",
    max_workers=4        # Process 4 pages in parallel
)

# Process pages as they're extracted
for document in loader.load():
    # Each document represents one page
    process_document(document)
```

## Database Operation Optimizations

### Batch Entity Creation

```python
from graphrag_sdk.steps.extract_data_step import BatchExtractDataStep

# Create batch-enabled extraction step
step = BatchExtractDataStep(
    sources=sources,
    ontology=ontology,
    model=model,
    graph=graph,
    batch_size=1000,      # Process 1000 entities at once
    max_memory_mb=2048     # Memory limit for batch processing
)

# Run with automatic batching
result = step.run()
```

### Batch Relation Creation

Relations are automatically batched when using `BatchExtractDataStep`. The system groups relations by type and creates them in optimized batches.

### Connection Pooling

```python
from graphrag_sdk.kg import PooledGraphOperations

# Create connection pool
pool = PooledGraphOperations(
    graph=graph,
    pool_size=5    # Maintain 5 database connections
)

# Execute queries using pooled connections
result = pool.execute_query("MATCH (n) RETURN count(n)")
```

## Memory Management

### Memory-Aware Processing

```python
from graphrag_sdk.steps.extract_data_step import MemoryAwareExtractDataStep, MemoryConfig

# Configure memory limits
memory_config = MemoryConfig(
    max_memory_mb=2048,        # 2GB memory limit
    memory_check_interval=10,   # Check memory every 10 documents
    gc_threshold=0.8,          # Trigger GC at 80% memory
    chunk_size_reduction=0.5    # Halve chunk size under pressure
)

# Create memory-aware step
step = MemoryAwareExtractDataStep(
    sources=sources,
    ontology=ontology,
    model=model,
    graph=graph,
    memory_config=memory_config
)
```

### Memory Monitoring

```python
from tests.resource_profiler import ResourceProfiler, profile_function

# Profile a function
with profile_function("data_extraction", sampling_interval=0.5):
    result = extract_data(sources)

# Or use profiler directly
profiler = ResourceProfiler()
profiler.start_profiling()

# Run your code
result = extract_data(sources)

# Get performance summary
summary = profiler.stop_profiling()
print(f"Peak memory: {summary.peak_memory_mb:.1f}MB")
print(f"Average CPU: {summary.avg_cpu_percent:.1f}%")
```

## Async I/O Operations

### Async URL Loading

```python
from graphrag_sdk.document_loaders.url import AsyncURLLoader
import asyncio

# Create async URL loader
loader = AsyncURLLoader(
    urls=[
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3"
    ],
    max_concurrent=10,    # Process 10 URLs concurrently
    timeout=30            # 30 second timeout per URL
)

# Load URLs asynchronously
async def load_urls():
    documents = []
    async for document in loader.load():
        documents.append(document)
    return documents

# Run async operation
documents = asyncio.run(load_urls())
```

## Adaptive Processing

### Adaptive Batch Processing

```python
from graphrag_sdk.steps.extract_data_step import AdaptiveExtractDataStep

# Create adaptive processing step
step = AdaptiveExtractDataStep(
    sources=sources,
    ontology=ontology,
    model=model,
    graph=graph,
    adaptive_config={
        'initial_batch_size': 50,
        'min_batch_size': 10,
        'max_batch_size': 500,
        'adjustment_factor': 0.2
    }
)

# Run with automatic optimization
result = step.run()
```

The system will automatically:
- Monitor performance metrics
- Adjust batch size based on throughput
- Optimize memory usage
- Adapt to system resources

## Performance Monitoring

### Benchmarking

```python
from tests.performance_test_framework import PerformanceBenchmark, BenchmarkConfig

# Configure benchmark
config = BenchmarkConfig(
    test_iterations=5,
    warmup_iterations=2,
    enable_memory_monitoring=True,
    enable_cpu_monitoring=True
)

# Create benchmark
benchmark = PerformanceBenchmark(config)

# Run performance test
def test_function():
    return extract_data(sources)

metrics = benchmark.run_benchmark(test_function, "extraction_test")
print(f"Throughput: {metrics.docs_per_second:.1f} docs/sec")
print(f"Memory: {metrics.peak_memory_mb:.1f}MB")
```

### Performance Comparison

```python
# Compare different implementations
implementations = {
    'baseline': baseline_extract,
    'optimized': optimized_extract
}

results = benchmark.compare_implementations(
    implementations,
    'extraction_comparison'
)

# Generate comparison report
benchmark.generate_report()
benchmark.create_visualizations()
```

## Best Practices

### 1. Choose the Right Configuration

- **High Performance**: Use on servers with 8GB+ RAM and multiple CPU cores
- **Resource Constrained**: Use on systems with 2GB RAM or fewer CPU cores
- **Custom**: Adjust parameters based on your specific workload

### 2. Monitor Resource Usage

```python
# Always monitor in production
from tests.resource_profiler import profile_function

@profile_function("production_extraction")
def extract_with_monitoring():
    return kg.extract_data_from_sources(sources)
```

### 3. Use Streaming for Large Files

```python
# For files > 100MB, always use streaming
if file_size > 100 * 1024 * 1024:
    loader = StreamingTextLoader(file_path, chunk_size=16384)
else:
    loader = TextLoader(file_path)
```

### 4. Optimize Batch Sizes

```python
# Start with these batch sizes and adjust:
# - Small documents: 1000-5000 per batch
# - Large documents: 100-500 per batch
# - Memory constrained: 10-100 per batch
```

### 5. Enable Connection Pooling

```python
# Always use connection pooling for production
pool = PooledGraphOperations(graph, pool_size=min(10, cpu_count))
```

## Troubleshooting

### Memory Issues

**Problem**: Out of memory errors
**Solution**: 
- Reduce `chunk_size` and `batch_size`
- Enable `enable_memory_monitoring`
- Set lower `max_memory_mb`

### Slow Performance

**Problem**: Processing is slower than expected
**Solution**:
- Increase `batch_size` if memory allows
- Enable `enable_async_io`
- Check if `max_concurrent_files` is too low

### Database Timeouts

**Problem**: Database operation timeouts
**Solution**:
- Reduce `batch_size`
- Increase `connection_pool_size`
- Check database connection limits

## Performance Metrics

### Key Metrics to Monitor

1. **Documents per Second**: Primary throughput metric
2. **Memory per Document**: Memory efficiency
3. **CPU Usage**: Processor utilization
4. **Batch Efficiency**: Actual vs. optimal batch size
5. **Error Rate**: Reliability metric

### Target Performance

| Metric | Good | Excellent |
|--------|------|-----------|
| Docs/sec | > 50 | > 100 |
| Memory/Doc | < 100KB | < 50KB |
| CPU Usage | 50-80% | 70-90% |
| Error Rate | < 1% | < 0.1% |

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI
from graphrag_sdk.optimization_config import IngestionOptimizationConfig

app = FastAPI()

@app.post("/extract")
async def extract_documents(file_paths: List[str]):
    config = IngestionOptimizationConfig.high_performance()
    kg = KnowledgeGraph(
        model=model,
        ontology=ontology,
        optimization_config=config
    )
    
    sources = [TEXT(path) for path in file_paths]
    result = await kg.extract_data_from_sources_async(sources)
    
    return {
        "documents_processed": len(result.documents),
        "entities_created": len(result.entities),
        "relations_created": len(result.relations)
    }
```

### Celery Integration

```python
from celery import Celery
from graphrag_sdk.optimization_config import IngestionOptimizationConfig

app = Celery('graphrag_tasks')

@app.task
def process_large_dataset(file_paths):
    config = IngestionOptimizationConfig.resource_constrained()
    kg = KnowledgeGraph(
        model=model,
        ontology=ontology,
        optimization_config=config
    )
    
    sources = [TEXT(path) for path in file_paths]
    return kg.extract_data_from_sources(sources)
```

## API Reference

### Classes

- `IngestionOptimizationConfig`: Main configuration class
- `StreamingTextLoader`: Streaming text document loader
- `StreamingHTMLLoader`: Streaming HTML document loader
- `StreamingPDFLoader`: Streaming PDF document loader
- `BatchExtractDataStep`: Batch-enabled extraction step
- `MemoryAwareExtractDataStep`: Memory-aware extraction step
- `AdaptiveExtractDataStep`: Adaptive extraction step
- `ResourceProfiler`: Resource monitoring and profiling
- `PerformanceBenchmark`: Performance testing framework

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_streaming` | bool | True | Enable streaming document loaders |
| `chunk_size` | int | 8192 | Document chunk size in bytes |
| `batch_size` | int | 1000 | Database operation batch size |
| `max_memory_mb` | int | 2048 | Maximum memory usage in MB |
| `enable_async_io` | bool | True | Enable async I/O operations |
| `enable_adaptive_processing` | bool | True | Enable adaptive batch processing |

For more detailed information, see the [API Documentation](api_reference.md).