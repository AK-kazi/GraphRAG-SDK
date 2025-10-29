# Query and Chat Session Performance Optimization Plan

## Overview

This document outlines the implementation plan for optimizing query performance and chat session management in the GraphRAG-SDK. The optimizations focus on the query path: **Question → Cypher Generation → Query Execution → QA Response**.

## Performance Bottlenecks Identified

1. **Repeated LLM API calls** for similar queries
2. **Chat session creation overhead** for each new session
3. **Ontology processing redundancy** in each session initialization
4. **Sequential query execution** without parallelization
5. **Growing chat history** increasing token usage and latency
6. **Inefficient retry mechanisms** without exponential backoff

## Optimization Implementation Plan

### 1. Query Result Caching (Priority: High)

**Files to Modify:** `graphrag_sdk/chat_session.py`

**Implementation Details:**
```python
import hashlib
import time
from functools import lru_cache
from typing import Optional, Dict, Tuple

class ChatSession:
    def __init__(self, ...):
        # Existing initialization
        self._query_cache: Dict[str, Tuple[str, float]] = {}
        self._cache_ttl = 3600  # 1 hour cache duration
        
    def _get_cache_key(self, message: str, cypher: str) -> str:
        """Generate MD5 hash cache key from message and cypher"""
        content = f"{message}:{cypher}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is within TTL"""
        return time.time() - timestamp < self._cache_ttl
    
    def _get_cached_response(self, message: str, cypher: str) -> Optional[str]:
        """Retrieve cached response if valid"""
        cache_key = self._get_cache_key(message, cypher)
        if cache_key in self._query_cache:
            response, timestamp = self._query_cache[cache_key]
            if self._is_cache_valid(timestamp):
                return response
            else:
                del self._query_cache[cache_key]
        return None
    
    def _cache_response(self, message: str, cypher: str, response: str):
        """Cache the response with timestamp"""
        cache_key = self._get_cache_key(message, cypher)
        self._query_cache[cache_key] = (response, time.time())
```

**Integration Points:**
- Modify `send_message()` method to check cache before QA step
- Add cache invalidation on ontology updates
- Implement cache size limits to prevent memory bloat

**Expected Performance Gain:** 40-60% improvement for repeated queries

### 2. Chat Session Pooling (Priority: High)

**Files to Modify:** `graphrag_sdk/kg.py`, `graphrag_sdk/chat_session.py`

**Implementation Details:**
```python
# In kg.py
import threading
from collections import defaultdict
from typing import List

class KnowledgeGraph:
    def __init__(self, ...):
        # Existing initialization
        self._session_pools: defaultdict = defaultdict(list)
        self._pool_lock = threading.Lock()
        self._max_pool_size = 5
        
    def get_pooled_chat_session(self) -> ChatSession:
        """Get or create a pooled chat session"""
        with self._pool_lock:
            pool_key = id(self._model_config)
            
            if self._session_pools[pool_key]:
                session = self._session_pools[pool_key].pop()
                self._reset_session_state(session)
                return session
                
            return self.chat_session()
    
    def return_session_to_pool(self, session: ChatSession):
        """Return session to pool for reuse"""
        with self._pool_lock:
            pool_key = id(self._model_config)
            if len(self._session_pools[pool_key]) < self._max_pool_size:
                self._session_pools[pool_key].append(session)
    
    def _reset_session_state(self, session: ChatSession):
        """Reset session state for reuse"""
        session.last_complete_response = {
            "question": None, "response": None, "context": None, "cypher": None
        }
        # Clear chat history if needed
        if hasattr(session, 'cypher_chat_session'):
            session.cypher_chat_session.clear_history()
        if hasattr(session, 'qa_chat_session'):
            session.qa_chat_session.clear_history()
```

**Integration Points:**
- Add session pool management to KnowledgeGraph class
- Implement session lifecycle management
- Add pool cleanup for unused sessions

**Expected Performance Gain:** 20-30% improvement by avoiding session creation overhead

### 3. Ontology Processing Caching (Priority: Medium)

**Files to Modify:** `graphrag_sdk/chat_session.py`

**Implementation Details:**
```python
import threading

class ChatSession:
    _ontology_cache: Dict[int, str] = {}
    _cache_lock = threading.Lock()
    
    def clean_ontology_for_prompt(self, ontology: dict) -> str:
        """Cached version of ontology processing"""
        cache_key = hash(str(ontology.to_json()))
        
        with self._cache_lock:
            if cache_key not in self._ontology_cache:
                self._ontology_cache[cache_key] = self._process_ontology(ontology)
            return self._ontology_cache[cache_key]
    
    def _process_ontology(self, ontology: dict) -> str:
        """Original ontology processing logic"""
        ontology = ontology.to_json()
        
        # Remove unique and required attributes
        for entity in ontology["entities"]:
            for attribute in entity["attributes"]:
                del attribute['unique']
                del attribute['required']
        
        for relation in ontology["relations"]:
            for attribute in relation["attributes"]:
                del attribute['unique']
                del attribute['required']
        
        return json.dumps(ontology)
```

**Integration Points:**
- Replace existing `clean_ontology_for_prompt` method
- Add cache invalidation on ontology changes
- Implement cache size management

**Expected Performance Gain:** 15-25% improvement for session initialization

### 4. Parallel Query Execution (Priority: High)

**Files to Modify:** `graphrag_sdk/chat_session.py`, `graphrag_sdk/steps/graph_query_step.py`

**Implementation Details:**
```python
# In chat_session.py
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

class ChatSession:
    def __init__(self, ...):
        # Existing initialization
        self._executor = ThreadPoolExecutor(max_workers=2)
        
    def send_message_async(self, message: str) -> dict:
        """Async version of send_message"""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(self._executor, self.send_message, message)
    
    def _generate_cypher_optimized(self, message: str) -> tuple:
        """Optimized cypher generation with parallel validation"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit cypher generation
            cypher_future = executor.submit(self._generate_cypher_query, message)
            
            # Wait for cypher generation
            context, cypher = cypher_future.result()
            
            return (context, cypher)

# In graph_query_step.py
class GraphQueryGenerationStep:
    def run_async(self, question: str, retries: Optional[int] = 10) -> tuple:
        """Async version with parallel validation"""
        async def _run_with_retry():
            for i in range(retries):
                try:
                    # Generate cypher
                    cypher = await self._generate_cypher_async(question)
                    
                    if not cypher:
                        return (None, None, None)
                    
                    # Parallel validation and execution
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        validation_future = executor.submit(validate_cypher, cypher, self.ontology)
                        validation_errors = validation_future.result()
                        
                        if validation_errors:
                            raise Exception("\n".join(validation_errors))
                        
                        # Execute query
                        query_result = self.graph.query(cypher)
                        result_set = query_result.result_set
                        execution_time = query_result.run_time_ms
                        context = stringify_falkordb_response(result_set)
                        
                        return (context, cypher, execution_time)
                        
                except Exception as e:
                    if i == retries - 1:
                        raise
                    await asyncio.sleep(2 ** i)  # Exponential backoff
        
        return asyncio.run(_run_with_retry())
```

**Integration Points:**
- Add async versions of critical methods
- Implement parallel validation and execution
- Add proper error handling for async operations

**Expected Performance Gain:** 30-50% improvement for complex queries

### 5. Chat History Optimization (Priority: Medium)

**Files to Modify:** `graphrag_sdk/models/litellm.py`

**Implementation Details:**
```python
class LiteModelChatSession:
    def __init__(self, model: LiteModel, system_instruction: Optional[str] = None):
        # Existing initialization
        self._max_history_length = 20
        self._compression_threshold = 10
        
    def _compress_history_if_needed(self):
        """Compress chat history to reduce token usage"""
        if len(self._chat_history) > self._compression_threshold:
            # Keep system message and last N messages
            system_msgs = [msg for msg in self._chat_history if msg["role"] == "system"]
            recent_msgs = self._chat_history[-self._max_history_length:]
            
            # Optional: Summarize older messages
            if len(self._chat_history) > self._max_history_length + 5:
                summary = self._summarize_old_messages()
                if summary:
                    recent_msgs.insert(0, {"role": "system", "content": f"Previous conversation summary: {summary}"})
            
            self._chat_history = system_msgs + recent_msgs
    
    def _summarize_old_messages(self) -> Optional[str]:
        """Summarize older messages to preserve context"""
        old_messages = self._chat_history[1:-self._max_history_length]
        if len(old_messages) < 4:
            return None
        
        # Create summary prompt
        summary_prompt = "Summarize this conversation in 2-3 sentences:\n"
        for msg in old_messages:
            summary_prompt += f"{msg['role']}: {msg['content']}\n"
        
        try:
            summary_response = completion(
                model=self._model.model,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=100
            )
            return summary_response.choices[0].message.content
        except:
            return None
    
    def send_message(self, message: str) -> GenerationResponse:
        """Optimized send_message with history management"""
        self._compress_history_if_needed()
        
        # Existing send_message logic
        self._chat_history.append({"role": "user", "content": message})
        # ... rest of existing code ...
```

**Integration Points:**
- Add history compression logic
- Implement optional message summarization
- Add configurable history limits

**Expected Performance Gain:** 10-20% reduction in token usage and faster responses

### 6. Smart Retry with Exponential Backoff (Priority: Medium)

**Files to Modify:** `graphrag_sdk/steps/graph_query_step.py`

**Implementation Details:**
```python
import time
import random
import logging

class GraphQueryGenerationStep:
    def __init__(self, ...):
        # Existing initialization
        self.base_delay = 0.1
        self.max_delay = 2.0
        self.jitter_factor = 0.1
        
    def run(self, question: str, retries: Optional[int] = 10) -> tuple:
        """Enhanced run with exponential backoff and jitter"""
        for i in range(retries):
            try:
                # Existing query generation logic
                cypher_prompt = (
                    (self.cypher_prompt.format(question=question) 
                    if self.last_answer is None
                    else self.cypher_prompt_with_history.format(question=question, last_answer=self.last_answer))
                )   
                
                logger.debug(f"Cypher Prompt: {cypher_prompt}")
                cypher_statement_response = self.chat_session.send_message(cypher_prompt)
                logger.debug(f"Cypher Statement Response: {cypher_statement_response}")
                cypher = extract_cypher(cypher_statement_response.text)
                logger.debug(f"Cypher: {cypher}")

                if not cypher or len(cypher) == 0:
                    return (None, None, None)

                validation_errors = validate_cypher(cypher, self.ontology)
                if validation_errors is not None:
                    raise Exception("\n".join(validation_errors))

                if cypher is not None:
                    query_result = self.graph.query(cypher)
                    result_set = query_result.result_set
                    execution_time = query_result.run_time_ms
                    context = stringify_falkordb_response(result_set)
                    
                    return (context, cypher, execution_time)
                    
            except Exception as e:
                logger.debug(f"Error: {e}")
                
                if i == retries - 1:
                    logger.error(f"Failed after {retries} retries: {e}")
                    raise
                
                # Calculate delay with exponential backoff and jitter
                delay = min(
                    self.base_delay * (2 ** i) + random.uniform(0, self.jitter_factor),
                    self.max_delay
                )
                
                logger.debug(f"Retry {i+1} after {delay:.2f}s delay")
                time.sleep(delay)
```

**Integration Points:**
- Replace existing retry logic in `run` method
- Add configurable delay parameters
- Implement proper logging for retry attempts

**Expected Performance Gain:** 15-25% improvement in error recovery and reduced API rate limiting

## Implementation Timeline

### Phase 1: High Impact Optimizations
1. **Query Result Caching** - Implement caching layer with TTL
2. **Chat Session Pooling** - Add session management and pooling
3. **Parallel Query Execution** - Add async support and parallel processing

### Phase 2: Medium Impact Optimizations
4. **Ontology Processing Caching** - Cache processed ontology
5. **Chat History Optimization** - Implement history compression
6. **Smart Retry Logic** - Add exponential backoff with jitter

### Phase 3: Testing and Refinement
7. **Memory Profiling** - Optimize memory usage
8. **Documentation** - Update API documentation

## Configuration Options

```python
# Add to model_config.py
class QueryOptimizationConfig:
    def __init__(self):
        # Caching settings
        self.enable_query_caching = True
        self.cache_ttl = 3600  # 1 hour
        self.max_cache_size = 1000
        
        # Session pooling
        self.enable_session_pooling = True
        self.max_pool_size = 5
        
        # Parallel execution
        self.enable_parallel_execution = True
        self.max_workers = 2
        
        # History optimization
        self.max_history_length = 20
        self.compression_threshold = 10
        self.enable_summarization = True
        
        # Retry settings
        self.max_retries = 10
        self.base_delay = 0.1
        self.max_delay = 2.0
        self.jitter_factor = 0.1
```

## Performance Monitoring

### Metrics to Track
1. **Query Response Time** - Average time per query
2. **Cache Hit Rate** - Percentage of queries served from cache
3. **Session Reuse Rate** - Percentage of sessions reused from pool
4. **Token Usage** - Average tokens per request
5. **Error Rate** - Percentage of failed queries
6. **Memory Usage** - Memory consumption over time

### Monitoring Implementation
```python
# Add to chat_session.py
import time
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class QueryMetrics:
    query_time: float
    cache_hit: bool
    token_count: int
    success: bool

class PerformanceMonitor:
    def __init__(self):
        self.metrics: List[QueryMetrics] = []
        self.start_time = time.time()
    
    def record_query(self, metrics: QueryMetrics):
        self.metrics.append(metrics)
    
    def get_average_response_time(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(m.query_time for m in self.metrics) / len(self.metrics)
    
    def get_cache_hit_rate(self) -> float:
        if not self.metrics:
            return 0.0
        cache_hits = sum(1 for m in self.metrics if m.cache_hit)
        return cache_hits / len(self.metrics)
```

## Expected Overall Performance Gains

- **Query Response Time:** 40-70% improvement
- **Memory Usage:** 20-40% reduction
- **API Call Efficiency:** 50-80% reduction in redundant calls
- **Session Management:** 30-50% improvement in session reuse
- **Error Recovery:** 25-40% improvement in retry efficiency

## Backward Compatibility

All optimizations will be implemented with feature flags to ensure backward compatibility:

```python
# Example feature flag usage
class ChatSession:
    def __init__(self, ..., enable_optimizations: bool = True):
        self.enable_caching = enable_optimizations and config.enable_query_caching
        self.enable_pooling = enable_optimizations and config.enable_session_pooling
        # ... other feature flags
```

This ensures existing code continues to work while allowing gradual adoption of optimizations.