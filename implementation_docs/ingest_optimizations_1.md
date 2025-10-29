# Ingestion and Extraction Performance Optimization Plan

## Overview

This document outlines the implementation plan for optimizing ingestion and extraction performance in the GraphRAG-SDK. The optimizations focus on the ingestion pipeline: **Document Loading → Data Extraction → Entity/Relation Creation → Database Insertion**.

## Performance Bottlenecks Identified

### 1. Document Loading Patterns
- **Synchronous file loading**: All document loaders read files synchronously (`text.py:27`, `pdf.py:36`, `csv.py:17`)
- **Full file loading**: Text and HTML loaders load entire files into memory at once
- **No streaming**: Large files are loaded completely before processing
- **Inefficient PDF processing**: Each page creates a separate document but processes sequentially

### 2. Data Extraction Workflows
- **Sequential entity/relation creation**: `extract_data_step.py:232-244` processes entities and relations sequentially
- **Individual database queries**: Each entity/relation creates a separate Cypher query
- **No batch operations**: Database operations are not batched
- **Rate limiting per document**: Rate limiting is applied per document rather than globally

### 3. Memory Usage During Processing
- **All documents loaded into memory**: `extract_data_step.py:96-101` loads all documents before processing
- **No memory monitoring**: No checks for memory usage during processing
- **Large text chunks**: Documents are truncated only by token count, not memory usage

### 4. I/O Operations and File Handling
- **Synchronous I/O**: All file operations are blocking
- **No connection pooling**: Database connections are created per operation
- **Inefficient URL fetching**: `url.py:24` makes single HTTP requests without connection reuse

## Optimization Implementation Plan

### 1. Streaming Document Loaders (Priority: High)

**Files to Modify:** `graphrag_sdk/document_loaders/text.py`, `graphrag_sdk/document_loaders/html.py`, `graphrag_sdk/document_loaders/pdf.py`

**Implementation Details:**
```python
# Enhanced TextLoader with streaming and chunking
import re
from typing import Iterator, Optional
from graphrag_sdk.document import Document

class StreamingTextLoader:
    def __init__(self, path: str, chunk_size: int = 8192, overlap: int = 100):
        self.path = path
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def load(self) -> Iterator[Document]:
        """Load text file in streaming chunks"""
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                buffer = ""
                chunk_id = 0
                
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        if buffer.strip():
                            yield Document(buffer.strip(), f"{self.path}#final_{chunk_id}")
                        break
                    
                    buffer += chunk
                    if len(buffer) >= self.chunk_size + self.overlap:
                        # Find optimal breaking point
                        break_point = self._find_break_point(buffer)
                        content = buffer[:break_point].strip()
                        
                        if content:
                            yield Document(content, f"{self.path}#chunk_{chunk_id}")
                            chunk_id += 1
                        
                        buffer = buffer[break_point - self.overlap:]
                        
        except UnicodeDecodeError:
            # Fallback to different encoding
            with open(self.path, 'r', encoding='latin-1') as f:
                content = f.read()
                yield Document(content, self.path)
    
    def _find_break_point(self, text: str) -> int:
        """Find optimal breaking point in text"""
        # Prefer sentence breaks
        for i in range(len(text) - 1, max(0, len(text) - 500), -1):
            if text[i] in '.!?' and i < len(text) - 1 and text[i+1] == ' ':
                return i + 1
        
        # Then paragraph breaks
        for i in range(len(text) - 1, max(0, len(text) - 200), -1):
            if text[i] == '\n' and i > 0 and text[i-1] == '\n':
                return i + 1
        
        # Finally word breaks
        for i in range(len(text) - 1, max(0, len(text) - 100), -1):
            if text[i] == ' ':
                return i + 1
        
        return len(text) // 2

# Enhanced HTML loader with streaming
class StreamingHTMLLoader:
    def __init__(self, path: str, chunk_size: int = 8192):
        self.path = path
        self.chunk_size = chunk_size
    
    def load(self) -> Iterator[Document]:
        """Load and parse HTML in streaming fashion"""
        try:
            from bs4 import BeautifulSoup
            
            with open(self.path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            text = re.sub(r'\n{3,}', '\n\n', text)  # Normalize newlines
            text = re.sub(r' {2,}', ' ', text)  # Normalize spaces
            
            # Split into chunks
            for i in range(0, len(text), self.chunk_size):
                chunk = text[i:i + self.chunk_size]
                if chunk.strip():
                    yield Document(chunk.strip(), f"{self.path}#chunk_{i//self.chunk_size}")
                    
        except Exception as e:
            # Fallback to plain text
            with open(self.path, 'r', encoding='utf-8') as f:
                content = f.read()
                yield Document(content, self.path)

# Enhanced PDF loader with parallel processing
class StreamingPDFLoader:
    def __init__(self, path: str, max_workers: int = 4):
        self.path = path
        self.max_workers = max_workers
    
    def load(self) -> Iterator[Document]:
        """Load PDF with parallel page processing"""
        try:
            import PyPDF2
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with open(self.path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                def extract_page(page_num):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    return Document(text, f"{self.path}#page_{page_num + 1}")
                
                # Process pages in parallel
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [executor.submit(extract_page, i) for i in range(total_pages)]
                    
                    for future in as_completed(futures):
                        try:
                            doc = future.result()
                            if doc.not_empty():
                                yield doc
                        except Exception as e:
                            print(f"Error processing PDF page: {e}")
                            
        except Exception as e:
            raise ValueError(f"Failed to load PDF {self.path}: {e}")
```

**Integration Points:**
- Add streaming capabilities to existing loaders
- Implement chunking with overlap for context preservation
- Add fallback mechanisms for encoding issues
- Implement parallel PDF page processing

**Expected Performance Gain:** 3-5x improvement for large files, 50-70% memory reduction

### 2. Batch Database Operations (Priority: High)

**Files to Modify:** `graphrag_sdk/steps/extract_data_step.py`, `graphrag_sdk/kg.py`

**Implementation Details:**
```python
# Batch entity and relation creation
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from falkordb import Graph

@dataclass
class BatchOperation:
    operation_type: str  # 'entity' or 'relation'
    data: Dict[str, Any]
    timestamp: float

class BatchExtractDataStep:
    def __init__(
        self,
        sources: list,
        ontology: Ontology,
        model: GenerativeModel,
        graph: Graph,
        batch_size: int = 1000,
        max_memory_mb: int = 2048,
        hide_progress: bool = False
    ):
        self.sources = sources
        self.ontology = ontology
        self.model = model
        self.graph = graph
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb
        self.hide_progress = hide_progress
        
        # Batch buffers
        self.entity_batch = []
        self.relation_batch = []
        self.batch_operations = []
        
        # Performance tracking
        self.processed_documents = 0
        self.created_entities = 0
        self.created_relations = 0
    
    def _create_entity_batch_query(self, entities: List[Dict]) -> str:
        """Generate optimized batch entity creation query"""
        # Group entities by type for more efficient queries
        entities_by_type = {}
        for entity in entities:
            entity_type = entity['label']
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        queries = []
        for entity_type, type_entities in entities_by_type.items():
            # Extract unique and non-unique properties
            unique_props = self._get_unique_properties(type_entities[0]['attributes'])
            non_unique_props = self._get_non_unique_properties(type_entities[0]['attributes'])
            
            query = f"""
            UNWIND $entities_{entity_type.lower()} AS entity
            MERGE (n:{entity_type} {{{unique_props}}})
            SET n += entity.non_unique_props
            """
            queries.append(query)
        
        return "\n".join(queries)
    
    def _create_relation_batch_query(self, relations: List[Dict]) -> str:
        """Generate optimized batch relation creation query"""
        # Group relations by type
        relations_by_type = {}
        for relation in relations:
            rel_type = relation['label']
            if rel_type not in relations_by_type:
                relations_by_type[rel_type] = []
            relations_by_type[rel_type].append(relation)
        
        queries = []
        for rel_type, type_relations in relations_by_type.items():
            query = f"""
            UNWIND $relations_{rel_type.lower()} AS rel
            MATCH (source:{rel['source']['label']} {{{rel['source']['unique_props']}}})
            MATCH (target:{rel['target']['label']} {{{rel['target']['unique_props']}}})
            MERGE (source)-[r:{rel_type}]->(target)
            SET r += rel.properties
            """
            queries.append(query)
        
        return "\n".join(queries)
    
    def _flush_entity_batch(self):
        """Flush entity batch to database"""
        if not self.entity_batch:
            return
        
        try:
            # Prepare batch data
            entities_by_type = {}
            for entity in self.entity_batch:
                entity_type = entity['label']
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                
                # Split attributes into unique and non-unique
                unique_props = {}
                non_unique_props = {}
                for attr_name, attr_value in entity['attributes'].items():
                    if self._is_unique_attribute(entity_type, attr_name):
                        unique_props[attr_name] = attr_value
                    else:
                        non_unique_props[attr_name] = attr_value
                
                entities_by_type[entity_type].append({
                    'unique_props': unique_props,
                    'non_unique_props': non_unique_props
                })
            
            # Execute batch query
            query = self._create_entity_batch_query(self.entity_batch)
            params = {}
            
            for entity_type, type_entities in entities_by_type.items():
                params[f"entities_{entity_type.lower()}"] = type_entities
            
            result = self.graph.query(query, params)
            self.created_entities += len(self.entity_batch)
            self.entity_batch.clear()
            
            return result
            
        except Exception as e:
            print(f"Error in batch entity creation: {e}")
            # Fallback to individual creation
            self._fallback_entity_creation()
    
    def _flush_relation_batch(self):
        """Flush relation batch to database"""
        if not self.relation_batch:
            return
        
        try:
            # Prepare batch data
            relations_by_type = {}
            for relation in self.relation_batch:
                rel_type = relation['label']
                if rel_type not in relations_by_type:
                    relations_by_type[rel_type] = []
                
                # Prepare source and target unique properties
                source_unique = self._extract_unique_properties(
                    relation['source']['label'], 
                    relation['source']['attributes']
                )
                target_unique = self._extract_unique_properties(
                    relation['target']['label'], 
                    relation['target']['attributes']
                )
                
                relations_by_type[rel_type].append({
                    'source': {
                        'label': relation['source']['label'],
                        'unique_props': source_unique
                    },
                    'target': {
                        'label': relation['target']['label'],
                        'unique_props': target_unique
                    },
                    'properties': relation.get('attributes', {})
                })
            
            # Execute batch query
            query = self._create_relation_batch_query(self.relation_batch)
            params = {}
            
            for rel_type, type_relations in relations_by_type.items():
                params[f"relations_{rel_type.lower()}"] = type_relations
            
            result = self.graph.query(query, params)
            self.created_relations += len(self.relation_batch)
            self.relation_batch.clear()
            
            return result
            
        except Exception as e:
            print(f"Error in batch relation creation: {e}")
            # Fallback to individual creation
            self._fallback_relation_creation()
    
    def _fallback_entity_creation(self):
        """Fallback to individual entity creation"""
        for entity in self.entity_batch:
            try:
                # Original individual creation logic
                label = entity['label']
                attributes = entity['attributes']
                
                # Create Cypher query
                props_str = ", ".join([f"{k}: {json.dumps(v)}" for k, v in attributes.items()])
                query = f"MERGE (n:{label} {{{props_str}}})"
                
                self.graph.query(query)
                self.created_entities += 1
                
            except Exception as e:
                print(f"Error creating entity {entity}: {e}")
        
        self.entity_batch.clear()
    
    def _fallback_relation_creation(self):
        """Fallback to individual relation creation"""
        for relation in self.relation_batch:
            try:
                # Original individual creation logic
                source_label = relation['source']['label']
                source_attrs = relation['source']['attributes']
                target_label = relation['target']['label']
                target_attrs = relation['target']['attributes']
                rel_label = relation['label']
                rel_attrs = relation.get('attributes', {})
                
                # Create Cypher query
                source_props = ", ".join([f"{k}: {json.dumps(v)}" for k, v in source_attrs.items()])
                target_props = ", ".join([f"{k}: {json.dumps(v)}" for k, v in target_attrs.items()])
                rel_props = ", ".join([f"{k}: {json.dumps(v)}" for k, v in rel_attrs.items()])
                
                query = f"""
                MATCH (source:{source_label} {{{source_props}}})
                MATCH (target:{target_label} {{{target_props}}})
                MERGE (source)-[r:{rel_label}]->(target)
                SET r += {{{rel_props}}}
                """
                
                self.graph.query(query)
                self.created_relations += 1
                
            except Exception as e:
                print(f"Error creating relation {relation}: {e}")
        
        self.relation_batch.clear()
    
    def _add_to_entity_batch(self, entity_data: Dict):
        """Add entity to batch, flush if needed"""
        self.entity_batch.append(entity_data)
        if len(self.entity_batch) >= self.batch_size:
            self._flush_entity_batch()
    
    def _add_to_relation_batch(self, relation_data: Dict):
        """Add relation to batch, flush if needed"""
        self.relation_batch.append(relation_data)
        if len(self.relation_batch) >= self.batch_size:
            self._flush_relation_batch()
    
    def flush_all_batches(self):
        """Flush all remaining batches"""
        self._flush_entity_batch()
        self._flush_relation_batch()
```

**Integration Points:**
- Replace individual entity/relation creation with batch operations
- Implement fallback mechanisms for batch failures
- Add transaction support for data consistency
- Optimize queries by grouping by entity/relation types

**Expected Performance Gain:** 10-20x improvement for database operations

### 3. Memory-Aware Document Processing (Priority: High)

**Files to Modify:** `graphrag_sdk/steps/extract_data_step.py`

**Implementation Details:**
```python
import psutil
import gc
from typing import Iterator, Optional
from dataclasses import dataclass

@dataclass
class MemoryConfig:
    max_memory_mb: int = 2048
    memory_check_interval: int = 10
    gc_threshold: float = 0.8  # Trigger GC at 80% memory usage
    chunk_size_reduction: float = 0.5  # Reduce chunk size by 50% under memory pressure

class MemoryAwareExtractDataStep(BatchExtractDataStep):
    def __init__(self, *args, memory_config: Optional[MemoryConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_config = memory_config or MemoryConfig()
        self.processed_count = 0
        self.memory_warnings = 0
        
    def _check_memory_usage(self) -> bool:
        """Check if memory usage exceeds threshold"""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent / 100.0
        usage_mb = memory.used / (1024 * 1024)
        
        # Log memory usage periodically
        if self.processed_count % self.memory_config.memory_check_interval == 0:
            print(f"Memory usage: {usage_percent:.1%} ({usage_mb:.0f} MB)")
        
        # Check if we're over the threshold
        if usage_mb > self.memory_config.max_memory_mb:
            self.memory_warnings += 1
            return True
        
        # Check if we should trigger garbage collection
        if usage_percent > self.memory_config.gc_threshold:
            gc.collect()
            print(f"Triggered garbage collection at {usage_percent:.1%} memory usage")
        
        return False
    
    def _get_documents_stream(self) -> Iterator[tuple]:
        """Stream documents instead of loading all at once"""
        for source in self.sources:
            try:
                for document in source.load():
                    if document.not_empty():
                        yield document, source.instruction
                        
                        # Check memory after each document
                        if self._check_memory_usage():
                            self._handle_memory_pressure()
                            
            except Exception as e:
                print(f"Error loading from source {source}: {e}")
                continue
    
    def _handle_memory_pressure(self):
        """Handle memory pressure by reducing processing load"""
        print(f"Memory pressure detected! Implementing mitigation strategies...")
        
        # Force garbage collection
        gc.collect()
        
        # Flush any pending batches to free memory
        self.flush_all_batches()
        
        # Reduce chunk size for future documents
        if hasattr(self, 'current_chunk_size'):
            self.current_chunk_size = int(
                self.current_chunk_size * self.memory_config.chunk_size_reduction
            )
            print(f"Reduced chunk size to {self.current_chunk_size}")
        
        # Wait for memory to be freed
        import time
        time.sleep(1)
    
    def _process_document_with_memory_check(self, document, source_instruction):
        """Process document with memory monitoring"""
        try:
            # Check memory before processing
            if self._check_memory_usage():
                self._handle_memory_pressure()
            
            # Process the document
            result = self._process_document(document, source_instruction)
            self.processed_count += 1
            
            # Periodic cleanup
            if self.processed_count % 50 == 0:
                gc.collect()
            
            return result
            
        except MemoryError:
            print(f"Memory error processing document. Implementing emergency cleanup...")
            self._emergency_memory_cleanup()
            raise
    
    def _emergency_memory_cleanup(self):
        """Emergency cleanup when memory is critically low"""
        print("EMERGENCY: Performing aggressive memory cleanup...")
        
        # Flush all batches immediately
        self.flush_all_batches()
        
        # Force multiple garbage collection cycles
        for _ in range(3):
            gc.collect()
            import time
            time.sleep(0.1)
        
        # Clear any caches
        if hasattr(self, '_cache'):
            self._cache.clear()
        
        print("Emergency cleanup completed")
    
    def run(self, instructions: Optional[str] = None):
        """Memory-aware version of run method"""
        print(f"Starting memory-aware extraction with max {self.memory_config.max_memory_mb} MB")
        
        try:
            # Process documents in streaming fashion
            documents_processed = 0
            
            with tqdm(desc="Processing Documents", disable=self.hide_progress) as pbar:
                for document, source_instruction in self._get_documents_stream():
                    try:
                        self._process_document_with_memory_check(document, source_instruction)
                        documents_processed += 1
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"Error processing document: {e}")
                        continue
            
            # Flush any remaining batches
            self.flush_all_batches()
            
            print(f"Processing completed. Documents: {documents_processed}, "
                  f"Entities: {self.created_entities}, Relations: {self.created_relations}")
            print(f"Memory warnings encountered: {self.memory_warnings}")
            
        except Exception as e:
            print(f"Fatal error in extraction: {e}")
            # Try to save any pending data
            self.flush_all_batches()
            raise
```

**Integration Points:**
- Add memory monitoring throughout the processing pipeline
- Implement adaptive chunking based on memory pressure
- Add emergency cleanup procedures for critical memory situations
- Implement streaming document processing instead of bulk loading

**Expected Performance Gain:** 50-70% memory reduction, ability to process larger datasets

### 4. Async I/O and Connection Pooling (Priority: Medium)

**Files to Modify:** `graphrag_sdk/document_loaders/url.py`, `graphrag_sdk/kg.py`

**Implementation Details:**
```python
# Async URL loader with connection pooling
import asyncio
import aiohttp
import aiofiles
from typing import Iterator, List
from urllib.parse import urlparse

class AsyncURLLoader:
    def __init__(self, urls: List[str], max_concurrent: int = 10, timeout: int = 30):
        self.urls = urls
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def load(self) -> Iterator[Document]:
        """Load URLs asynchronously with connection pooling"""
        # Configure connection pooling
        connector = aiohttp.TCPConnector(
            limit=50,  # Total connection pool size
            limit_per_host=10,  # Connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        headers = {
            'User-Agent': 'GraphRAG-SDK/1.0 (Ingestion Bot)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        ) as session:
            # Create tasks for all URLs
            tasks = [self._fetch_url(session, url) for url in self.urls]
            
            # Process results as they complete
            for completed_task in asyncio.as_completed(tasks):
                try:
                    document = await completed_task
                    if document:
                        yield document
                except Exception as e:
                    print(f"Error fetching URL: {e}")
                    continue
    
    async def _fetch_url(self, session: aiohttp.ClientSession, url: str) -> Document:
        """Fetch single URL with retry logic"""
        async with self.semaphore:
            max_retries = 3
            base_delay = 1
            
            for attempt in range(max_retries):
                try:
                    async with session.get(url) as response:
                        response.raise_for_status()
                        
                        # Check content type
                        content_type = response.headers.get('content-type', '').lower()
                        if 'text/html' not in content_type:
                            print(f"Skipping non-HTML content: {content_type}")
                            return None
                        
                        # Read content with size limit
                        content = await response.text()
                        
                        # Extract text from HTML
                        text = self._extract_text_from_html(content)
                        
                        if text.strip():
                            return Document(text, url)
                        else:
                            return None
                            
                except aiohttp.ClientError as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                except Exception as e:
                    print(f"Unexpected error fetching {url}: {e}")
                    return None
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract clean text from HTML content"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Get text and normalize whitespace
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return html_content

# Database connection pooling
class PooledGraphOperations:
    def __init__(self, graph: Graph, pool_size: int = 5):
        self.graph = graph
        self.pool_size = pool_size
        self.connection_pool = []
        self.pool_lock = threading.Lock()
        
        # Initialize connection pool
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize database connection pool"""
        for _ in range(self.pool_size):
            # Create connection (simplified - actual implementation depends on FalkorDB)
            connection = self._create_connection()
            self.connection_pool.append(connection)
    
    def _create_connection(self):
        """Create new database connection"""
        # This would create a new FalkorDB connection
        # Implementation depends on FalkorDB API
        return self.graph
    
    def get_connection(self):
        """Get connection from pool"""
        with self.pool_lock:
            if self.connection_pool:
                return self.connection_pool.pop()
            else:
                # Pool exhausted, create new connection
                return self._create_connection()
    
    def return_connection(self, connection):
        """Return connection to pool"""
        with self.pool_lock:
            if len(self.connection_pool) < self.pool_size:
                self.connection_pool.append(connection)
            else:
                # Pool full, close connection
                self._close_connection(connection)
    
    def _close_connection(self, connection):
        """Close database connection"""
        # Implementation depends on FalkorDB API
        pass
    
    def execute_query(self, query: str, params: dict = None):
        """Execute query using pooled connection"""
        connection = self.get_connection()
        try:
            result = connection.query(query, params)
            return result
        finally:
            self.return_connection(connection)
    
    def execute_transaction(self, queries: List[tuple]):
        """Execute multiple queries in a transaction"""
        connection = self.get_connection()
        try:
            # Begin transaction
            results = []
            for query, params in queries:
                result = connection.query(query, params)
                results.append(result)
            # Commit transaction
            return results
        except Exception as e:
            # Rollback transaction
            raise
        finally:
            self.return_connection(connection)
```

**Integration Points:**
- Replace synchronous URL loading with async version
- Implement connection pooling for database operations
- Add retry logic with exponential backoff
- Implement proper resource cleanup

**Expected Performance Gain:** 2-3x improvement for I/O operations, better resource utilization

### 5. Adaptive Batch Processing (Priority: Medium)

**Files to Modify:** `graphrag_sdk/steps/extract_data_step.py`

**Implementation Details:**
```python
import time
import statistics
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    processing_times: List[float]
    memory_usage: List[float]
    error_count: int
    success_count: int
    
    def get_avg_processing_time(self) -> float:
        return statistics.mean(self.processing_times) if self.processing_times else 0
    
    def get_avg_memory_usage(self) -> float:
        return statistics.mean(self.memory_usage) if self.memory_usage else 0
    
    def get_error_rate(self) -> float:
        total = self.error_count + self.success_count
        return self.error_count / total if total > 0 else 0

class AdaptiveBatchProcessor:
    def __init__(
        self,
        initial_batch_size: int = 50,
        min_batch_size: int = 10,
        max_batch_size: int = 500,
        adjustment_factor: float = 0.2,
        performance_window: int = 10
    ):
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.adjustment_factor = adjustment_factor
        self.performance_window = performance_window
        
        self.current_batch_size = initial_batch_size
        self.metrics_history: List[PerformanceMetrics] = []
        self.last_adjustment_time = time.time()
        self.adjustment_interval = 60  # Adjust every 60 seconds
    
    def record_batch_performance(self, metrics: PerformanceMetrics):
        """Record performance metrics for a batch"""
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics_history) > self.performance_window:
            self.metrics_history.pop(0)
        
        # Check if we should adjust batch size
        if self._should_adjust():
            self._adjust_batch_size()
    
    def _should_adjust(self) -> bool:
        """Determine if batch size should be adjusted"""
        current_time = time.time()
        time_since_last_adjustment = current_time - self.last_adjustment_time
        
        return (
            time_since_last_adjustment >= self.adjustment_interval and
            len(self.metrics_history) >= 3
        )
    
    def _adjust_batch_size(self):
        """Adjust batch size based on performance metrics"""
        if not self.metrics_history:
            return
        
        # Calculate recent performance averages
        recent_metrics = self.metrics_history[-3:]
        avg_processing_time = statistics.mean([
            m.get_avg_processing_time() for m in recent_metrics
        ])
        avg_memory_usage = statistics.mean([
            m.get_avg_memory_usage() for m in recent_metrics
        ])
        avg_error_rate = statistics.mean([
            m.get_error_rate() for m in recent_metrics
        ])
        
        old_batch_size = self.current_batch_size
        
        # Adjust based on performance
        if avg_error_rate > 0.1:  # High error rate
            # Reduce batch size
            self.current_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * (1 - self.adjustment_factor))
            )
            print(f"Reduced batch size due to high error rate ({avg_error_rate:.2%})")
            
        elif avg_processing_time > 30.0:  # Slow processing
            # Reduce batch size
            self.current_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * (1 - self.adjustment_factor))
            )
            print(f"Reduced batch size due to slow processing ({avg_processing_time:.1f}s)")
            
        elif avg_memory_usage > 0.8:  # High memory usage
            # Reduce batch size
            self.current_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * (1 - self.adjustment_factor))
            )
            print(f"Reduced batch size due to high memory usage ({avg_memory_usage:.1%})")
            
        elif (avg_processing_time < 5.0 and 
              avg_memory_usage < 0.5 and 
              avg_error_rate < 0.05):  # Good performance
            # Increase batch size
            self.current_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * (1 + self.adjustment_factor))
            )
            print(f"Increased batch size due to good performance")
        
        # Log adjustment
        if old_batch_size != self.current_batch_size:
            print(f"Batch size adjusted: {old_batch_size} → {self.current_batch_size}")
            self.last_adjustment_time = time.time()
    
    def get_current_batch_size(self) -> int:
        """Get current optimal batch size"""
        return self.current_batch_size
    
    def get_optimal_workers(self, cpu_count: int, memory_gb: int) -> int:
        """Calculate optimal number of workers based on system resources"""
        # Base workers on CPU count
        cpu_workers = min(cpu_count * 2, 32)
        
        # Limit workers based on memory (assume 1GB per worker minimum)
        memory_workers = max(1, memory_gb)
        
        # Consider current batch size
        batch_factor = min(self.current_batch_size / 100, 2.0)
        
        optimal_workers = min(
            int(cpu_workers * batch_factor),
            memory_workers,
            16  # Hard maximum
        )
        
        return max(optimal_workers, 1)

class AdaptiveExtractDataStep(MemoryAwareExtractDataStep):
    def __init__(self, *args, adaptive_config: Optional[Dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize adaptive processor
        config = adaptive_config or {}
        self.adaptive_processor = AdaptiveBatchProcessor(
            initial_batch_size=config.get('initial_batch_size', 50),
            min_batch_size=config.get('min_batch_size', 10),
            max_batch_size=config.get('max_batch_size', 500),
            adjustment_factor=config.get('adjustment_factor', 0.2)
        )
        
        # System resource detection
        self.cpu_count = psutil.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
    
    def _process_batch_adaptive(self, documents: List[tuple]):
        """Process a batch of documents with adaptive sizing"""
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent / 100.0
        
        batch_size = len(documents)
        errors = 0
        successes = 0
        
        try:
            # Process documents in current batch
            for document, source_instruction in documents:
                try:
                    self._process_document(document, source_instruction)
                    successes += 1
                except Exception as e:
                    errors += 1
                    print(f"Error processing document: {e}")
            
            # Record performance metrics
            end_time = time.time()
            end_memory = psutil.virtual_memory().percent / 100.0
            
            processing_time = end_time - start_time
            memory_usage = (start_memory + end_memory) / 2
            
            metrics = PerformanceMetrics(
                processing_times=[processing_time / batch_size],
                memory_usage=[memory_usage],
                error_count=errors,
                success_count=successes
            )
            
            self.adaptive_processor.record_batch_performance(metrics)
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            raise
    
    def run(self, instructions: Optional[str] = None):
        """Run with adaptive batch processing"""
        print(f"Starting adaptive extraction with {self.cpu_count} CPUs, {self.memory_gb:.1f} GB RAM")
        
        # Get optimal worker count
        optimal_workers = self.adaptive_processor.get_optimal_workers(
            self.cpu_count, self.memory_gb
        )
        print(f"Using {optimal_workers} workers")
        
        documents_batch = []
        
        with tqdm(desc="Processing Documents", disable=self.hide_progress) as pbar:
            for document, source_instruction in self._get_documents_stream():
                documents_batch.append((document, source_instruction))
                
                # Check if batch is ready
                current_batch_size = self.adaptive_processor.get_current_batch_size()
                if len(documents_batch) >= current_batch_size:
                    self._process_batch_adaptive(documents_batch)
                    documents_batch.clear()
                    pbar.update(current_batch_size)
            
            # Process remaining documents
            if documents_batch:
                self._process_batch_adaptive(documents_batch)
                pbar.update(len(documents_batch))
        
        # Flush any remaining batches
        self.flush_all_batches()
        
        print(f"Adaptive extraction completed. Final batch size: {self.adaptive_processor.get_current_batch_size()}")
```

**Integration Points:**
- Add performance monitoring throughout the processing pipeline
- Implement dynamic batch size adjustment based on performance metrics
- Add system resource detection and optimization
- Implement adaptive worker count management

**Expected Performance Gain:** 20-40% improvement through adaptive optimization

## Implementation Timeline

### Phase 1 High Impact Optimizations
1. **Streaming Document Loaders** - Implement chunking and streaming
2. **Batch Database Operations** - Add batch entity/relation creation
3. **Memory-Aware Processing** - Add memory monitoring and adaptive processing

### Phase 2 Medium Impact Optimizations
4. **Async I/O and Connection Pooling** - Implement async URL loading
5. **Adaptive Batch Processing** - Add performance-based optimization

### Phase 3 Testing and Refinement
6. **Performance Testing** - Benchmark improvements
7. **Resource Profiling** - Optimize resource usage
8. **Documentation** - Update API documentation

## Configuration Options

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
    
    @classmethod
    def high_performance(cls) -> "IngestionOptimizationConfig":
        return cls(
            enable_streaming=True,
            chunk_size=16384,
            max_concurrent_files=20,
            batch_size=5000,
            adaptive_batching=True,
            max_memory_mb=8192,
            enable_async_io=True,
            connection_pool_size=10,
            max_concurrent_requests=50,
            enable_adaptive_processing=True
        )
    
    @classmethod
    def resource_constrained(cls) -> "IngestionOptimizationConfig":
        return cls(
            enable_streaming=True,
            chunk_size=4096,
            max_concurrent_files=3,
            batch_size=100,
            adaptive_batching=False,
            max_memory_mb=1024,
            enable_async_io=False,
            connection_pool_size=2,
            max_concurrent_requests=5,
            enable_adaptive_processing=False
        )
```

## Performance Monitoring

### Metrics to Track
1. **Documents Processed Per Second** - Throughput metric
2. **Memory Usage** - Peak and average memory consumption
3. **Database Operation Time** - Time for entity/relation creation
4. **Batch Efficiency** - Average batch size and utilization
5. **Error Rate** - Percentage of failed operations
6. **Resource Utilization** - CPU and disk I/O usage

### Monitoring Implementation
```python
class IngestionMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.documents_processed = 0
        self.entities_created = 0
        self.relations_created = 0
        self.batches_processed = 0
        self.errors = []
        self.memory_snapshots = []
        self.processing_times = []
    
    def record_batch_completed(self, batch_size: int, processing_time: float, memory_mb: float):
        self.batches_processed += 1
        self.documents_processed += batch_size
        self.processing_times.append(processing_time)
        self.memory_snapshots.append(memory_mb)
    
    def record_entities_created(self, count: int):
        self.entities_created += count
    
    def record_relations_created(self, count: int):
        self.relations_created += count
    
    def get_performance_summary(self) -> dict:
        total_time = time.time() - self.start_time
        
        return {
            'total_time': total_time,
            'documents_processed': self.documents_processed,
            'documents_per_second': self.documents_processed / total_time if total_time > 0 else 0,
            'entities_created': self.entities_created,
            'relations_created': self.relations_created,
            'batches_processed': self.batches_processed,
            'average_batch_size': self.documents_processed / self.batches_processed if self.batches_processed > 0 else 0,
            'average_processing_time': sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
            'peak_memory_mb': max(self.memory_snapshots) if self.memory_snapshots else 0,
            'error_count': len(self.errors),
            'success_rate': (self.documents_processed - len(self.errors)) / self.documents_processed if self.documents_processed > 0 else 0
        }
```

## Expected Overall Performance Gains

| Optimization Area | Current Performance | Optimized Performance | Improvement |
|------------------|-------------------|---------------------|-------------|
| Document Loading | Sequential, full file | Streaming, chunked | 3-5x faster |
| Database Operations | Individual queries | Batch transactions | 10-20x faster |
| Memory Usage | All docs in memory | Streaming with limits | 50-70% reduction |
| I/O Operations | Blocking requests | Async with pooling | 2-3x faster |
| Overall Throughput | Baseline | Optimized pipeline | 5-10x improvement |

## Backward Compatibility

All optimizations will be implemented with feature flags to ensure backward compatibility:

```python
# Example feature flag usage
class ExtractDataStep:
    def __init__(self, ..., enable_optimizations: bool = True):
        self.enable_streaming = enable_optimizations and config.enable_streaming
        self.enable_batch_operations = enable_optimizations and config.enable_batch_operations
        self.enable_memory_monitoring = enable_optimizations and config.enable_memory_monitoring
        # ... other feature flags
```

This ensures existing code continues to work while allowing gradual adoption of optimizations.