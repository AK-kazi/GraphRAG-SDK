import json
import hashlib
import time
import asyncio
import threading
import gc
import psutil
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from falkordb import Graph
from typing import Iterator, Optional, Dict, Tuple, List
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.steps.qa_step import QAStep
from graphrag_sdk.steps.stream_qa_step import StreamingQAStep
from graphrag_sdk.model_config import KnowledgeGraphModelConfig
from graphrag_sdk.steps.graph_query_step import GraphQueryGenerationStep
from graphrag_sdk.optimization_config import QueryOptimizationConfig, DEFAULT_OPTIMIZATION_CONFIG

CYPHER_ERROR_RES = "Sorry, I could not find the answer to your question"

@dataclass
class QueryMetrics:
    """Metrics for individual query performance"""
    query_time: float
    cache_hit: bool
    token_count: int
    success: bool
    cypher_generation_time: float
    qa_time: float

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics: List[QueryMetrics] = []
        self.start_time = time.time()
        self._lock = threading.Lock()
    
    def record_query(self, metrics: QueryMetrics):
        """Record query metrics"""
        with self._lock:
            self.metrics.append(metrics)
    
    def get_average_response_time(self) -> float:
        """Get average query response time"""
        if not self.metrics:
            return 0.0
        return sum(m.query_time for m in self.metrics) / len(self.metrics)
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate percentage"""
        if not self.metrics:
            return 0.0
        cache_hits = sum(1 for m in self.metrics if m.cache_hit)
        return cache_hits / len(self.metrics)
    
    def get_success_rate(self) -> float:
        """Get query success rate"""
        if not self.metrics:
            return 0.0
        successful_queries = sum(1 for m in self.metrics if m.success)
        return successful_queries / len(self.metrics)
    
    def get_total_queries(self) -> int:
        """Get total number of queries"""
        return len(self.metrics)
    
    def get_uptime(self) -> float:
        """Get monitor uptime in seconds"""
        return time.time() - self.start_time
    
    def get_metrics_summary(self) -> dict:
        """Get comprehensive metrics summary"""
        return {
            "total_queries": self.get_total_queries(),
            "average_response_time": self.get_average_response_time(),
            "cache_hit_rate": self.get_cache_hit_rate(),
            "success_rate": self.get_success_rate(),
            "uptime_seconds": self.get_uptime(),
            "queries_per_second": self.get_total_queries() / max(self.get_uptime(), 1)
        }

class MemoryProfiler:
    """Monitor and optimize memory usage"""
    
    def __init__(self, check_interval: int = 300):
        self.check_interval = check_interval
        self.last_check = time.time()
        self.memory_snapshots = []
        self._lock = threading.Lock()
    
    def check_memory_usage(self) -> dict:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "timestamp": time.time()
        }
    
    def should_check_memory(self) -> bool:
        """Check if it's time to monitor memory"""
        return time.time() - self.last_check >= self.check_interval
    
    def record_memory_snapshot(self):
        """Record a memory usage snapshot"""
        with self._lock:
            snapshot = self.check_memory_usage()
            self.memory_snapshots.append(snapshot)
            self.last_check = time.time()
            
            # Keep only last 100 snapshots
            if len(self.memory_snapshots) > 100:
                self.memory_snapshots = self.memory_snapshots[-100:]
    
    def get_memory_stats(self) -> dict:
        """Get memory usage statistics"""
        if not self.memory_snapshots:
            return {}
        
        rss_values = [s["rss_mb"] for s in self.memory_snapshots]
        vms_values = [s["vms_mb"] for s in self.memory_snapshots]
        percent_values = [s["percent"] for s in self.memory_snapshots]
        
        return {
            "avg_rss_mb": sum(rss_values) / len(rss_values),
            "max_rss_mb": max(rss_values),
            "avg_vms_mb": sum(vms_values) / len(vms_values),
            "max_vms_mb": max(vms_values),
            "avg_percent": sum(percent_values) / len(percent_values),
            "max_percent": max(percent_values),
            "snapshot_count": len(self.memory_snapshots)
        }
    
    def optimize_memory(self):
        """Perform memory optimization"""
        # Force garbage collection
        gc.collect()
        
        # Clear old cache entries if memory is high
        current_memory = self.check_memory_usage()
        if current_memory["percent"] > 80:  # If using more than 80% memory
            return True  # Signal that cache clearing is needed
        return False

class ChatSession:
    # Class-level ontology cache for all instances
    _ontology_cache: Dict[int, str] = {}
    _cache_lock = threading.Lock()
    _max_ontology_cache_size = 100
    """
    Represents a chat session with a Knowledge Graph.

    Args:
        model_config (KnowledgeGraphModelConfig): The model configuration to use.
        ontology (Ontology): The ontology to use.
        graph (Graph): The graph to query.

    Examples:
        >>> from graphrag_sdk import KnowledgeGraph, Orchestrator
        >>> from graphrag_sdk.ontology import Ontology
        >>> from graphrag_sdk.model_config import KnowledgeGraphModelConfig
        >>> model_config = KnowledgeGraphModelConfig.with_model(model)
        >>> kg = KnowledgeGraph("test_kg", model_config, ontology)
        >>> chat_session = kg.start_chat()
        >>> chat_session.send_message("What is the capital of France?")
    """

    def __init__(self, model_config: KnowledgeGraphModelConfig, ontology: Ontology, graph: Graph,
                cypher_system_instruction: str, qa_system_instruction: str,
                cypher_gen_prompt: str, qa_prompt: str, cypher_gen_prompt_history: str,
                optimization_config: Optional[QueryOptimizationConfig] = None):
        """
        Initializes a new ChatSession object.

        Args:
            model_config (KnowledgeGraphModelConfig): The model configuration.
            ontology (Ontology): The ontology object.
            graph (Graph): The graph object.
            cypher_system_instruction (str): Cypher system instruction.
            qa_system_instruction (str): QA system instruction.
            cypher_gen_prompt (str): Cypher generation prompt.
            qa_prompt (str): QA prompt.
            cypher_gen_prompt_history (str): Cypher prompt with history.
            optimization_config (Optional[QueryOptimizationConfig]): Optimization configuration.

        Attributes:
            model_config (KnowledgeGraphModelConfig): The model configuration.
            ontology (Ontology): The ontology object.
            graph (Graph): The graph object.
            cypher_chat_session (CypherChatSession): The Cypher chat session object.
            qa_chat_session (QAChatSession): The QA chat session object.
        """
        self.model_config = model_config
        self.graph = graph
        self.ontology = ontology
        
        # Optimization configuration
        self.optimization_config = optimization_config or DEFAULT_OPTIMIZATION_CONFIG
        
        # Query result caching
        self._query_cache: Dict[str, Tuple[str, float]] = {}
        self._cache_ttl = self.optimization_config.cache_ttl
        self._max_cache_size = self.optimization_config.max_cache_size
        
        # Parallel execution
        self._executor = ThreadPoolExecutor(max_workers=self.optimization_config.max_workers)
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self._enable_monitoring = self.optimization_config.enable_performance_monitoring
        
        # Memory profiling
        self.memory_profiler = MemoryProfiler(
            check_interval=self.optimization_config.memory_check_interval
        )
        self._enable_memory_profiling = self.optimization_config.enable_memory_profiling
        
        # Filter the ontology to remove unique and required attributes that are not needed for Q&A. 
        ontology_prompt = self.clean_ontology_for_prompt(ontology)
                
        cypher_system_instruction = cypher_system_instruction.format(ontology=ontology_prompt)
        
        self.cypher_prompt = cypher_gen_prompt
        self.qa_prompt = qa_prompt
        self.cypher_prompt_with_history = cypher_gen_prompt_history
        
        self.cypher_chat_session = model_config.cypher_generation.start_chat(
                cypher_system_instruction
            )
        self.qa_chat_session = model_config.qa.start_chat(
                qa_system_instruction
            )
        self.last_complete_response = {
            "question": None, 
            "response": None, 
            "context": None, 
            "cypher": None
            }
        
        # Metadata to store additional information about the chat session (currently only last query execution time)
        self.metadata = {"last_query_execution_time": None}
        
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
        
        # Remove oldest entries if cache is full
        if len(self._query_cache) >= self._max_cache_size:
            oldest_key = min(self._query_cache.keys(), 
                           key=lambda k: self._query_cache[k][1])
            del self._query_cache[oldest_key]
        
        self._query_cache[cache_key] = (response, time.time())

    async def send_message_async(self, message: str) -> dict:
        """
        Async version of send_message.
        
        Args:
            message (str): The message to send.
            
        Returns:
            dict: The response dictionary.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.send_message, message)
    
    def _generate_cypher_optimized(self, message: str) -> tuple:
        """
        Optimized cypher generation with parallel validation.
        
        Args:
            message (str): The message to generate cypher for.
            
        Returns:
            tuple: A tuple containing (context, cypher)
        """
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit cypher generation
            cypher_future = executor.submit(self._generate_cypher_query, message)
            
            # Wait for cypher generation
            context, cypher = cypher_future.result()
            
            return (context, cypher)

    def _generate_cypher_query(self, message: str) -> tuple:
        """
        Generate a Cypher query for the given message.
        
        Args:
            message (str): The message to generate a query for.
            
        Returns:
            tuple: A tuple containing (context, cypher)
        """
        cypher_step = GraphQueryGenerationStep(
            graph=self.graph,
            chat_session=self.cypher_chat_session,
            ontology=self.ontology,
            last_answer=self.last_complete_response["response"],
            cypher_prompt=self.cypher_prompt,
            cypher_prompt_with_history=self.cypher_prompt_with_history
        )

        (context, cypher, query_execution_time) = cypher_step.run(message)
        self.metadata["last_query_execution_time"] = query_execution_time
        
        return (context, cypher)

    def send_message(self, message: str) -> dict:
        """
        Sends a message to the chat session.

        Args:
            message (str): The message to send.

        Returns:
            dict: The response to the message in the following format:
                    {"question": message, 
                    "response": answer, 
                    "context": context, 
                    "cypher": cypher}
        """
        start_time = time.time()
        cache_hit = False
        success = True
        cypher_gen_start = time.time()
        
        # Memory profiling check
        if self._enable_memory_profiling and self.memory_profiler.should_check_memory():
            self.memory_profiler.record_memory_snapshot()
            if self.memory_profiler.optimize_memory():
                # Clear caches if memory is high
                self._query_cache.clear()
                ChatSession._ontology_cache.clear()
        
        (context, cypher) = self._generate_cypher_query(message)
        cypher_gen_time = time.time() - cypher_gen_start

        # If the cypher is empty, return an error message
        if not cypher or len(cypher) == 0:
            self.last_complete_response = {
                "question": message,
                "response": CYPHER_ERROR_RES,
                "context": None,
                "cypher": None
            }
            
            if self._enable_monitoring:
                metrics = QueryMetrics(
                    query_time=time.time() - start_time,
                    cache_hit=False,
                    token_count=0,
                    success=False,
                    cypher_generation_time=cypher_gen_time,
                    qa_time=0
                )
                self.performance_monitor.record_query(metrics)
            
            return self.last_complete_response
        
        # Check cache for existing response
        cached_answer = self._get_cached_response(message, cypher)
        if cached_answer:
            cache_hit = True
            self.last_complete_response = {
                "question": message, 
                "response": cached_answer, 
                "context": context, 
                "cypher": cypher
            }
            
            if self._enable_monitoring:
                metrics = QueryMetrics(
                    query_time=time.time() - start_time,
                    cache_hit=True,
                    token_count=len(cached_answer.split()),
                    success=True,
                    cypher_generation_time=cypher_gen_time,
                    qa_time=0
                )
                self.performance_monitor.record_query(metrics)
            
            return self.last_complete_response
        
        qa_step = QAStep(
            chat_session=self.qa_chat_session,
            qa_prompt=self.qa_prompt,
        )

        qa_start = time.time()
        answer = qa_step.run(message, cypher, context)
        qa_time = time.time() - qa_start
        
        # Cache the response
        self._cache_response(message, cypher, answer)

        self.last_complete_response = {
            "question": message, 
            "response": answer, 
            "context": context, 
            "cypher": cypher
        }
        
        # Record performance metrics
        if self._enable_monitoring:
            metrics = QueryMetrics(
                query_time=time.time() - start_time,
                cache_hit=cache_hit,
                token_count=len(answer.split()),
                success=success,
                cypher_generation_time=cypher_gen_time,
                qa_time=qa_time
            )
            self.performance_monitor.record_query(metrics)
        
        return self.last_complete_response
    
    def send_message_stream(self, message: str) -> Iterator[str]:

        """
        Sends a message to the chat session and streams the response.

        Args:
            message (str): The message to send.

        Yields:
            str: Chunks of the response as they're generated.
        """
        (context, cypher) = self._generate_cypher_query(message)

        if not cypher or len(cypher) == 0:
            # Stream the error message for consistency with successful responses
            yield CYPHER_ERROR_RES
            
            self.last_complete_response = {
                "question": message,
                "response": CYPHER_ERROR_RES,
                "context": None,
                "cypher": None
            }
            return

        qa_step = StreamingQAStep(
            chat_session=self.qa_chat_session,
            qa_prompt=self.qa_prompt,
        )

        # Yield chunks of the response as they're generated
        for chunk in qa_step.run(message, cypher, context):
            yield chunk

        # Set the last answer using chat history to ensure we have the complete response
        self.last_complete_response = {
            "question": message, 
            "response": qa_step.chat_session.get_chat_history()[-1]['content'], 
            "context": context, 
            "cypher": cypher
        }
        
    def clean_ontology_for_prompt(self, ontology: Ontology) -> str:
        """
        Cached version of ontology cleaning that removes 'unique' and 'required' keys.

        Args:
            ontology (Ontology): The ontology to clean and transform.

        Returns:
            str: The cleaned ontology as a JSON string.
        """
        cache_key = hash(str(ontology.to_json()))
        
        with self._cache_lock:
            if cache_key not in self._ontology_cache:
                self._ontology_cache[cache_key] = self._process_ontology(ontology)
                
                # Manage cache size
                if len(self._ontology_cache) > self._max_ontology_cache_size:
                    # Remove oldest entries (simple strategy - remove first 20%)
                    keys_to_remove = list(self._ontology_cache.keys())[:self._max_ontology_cache_size // 5]
                    for key in keys_to_remove:
                        del self._ontology_cache[key]
            
            return self._ontology_cache[cache_key]
    
    def _process_ontology(self, ontology: Ontology) -> str:
        """
        Original ontology processing logic.

        Args:
            ontology (Ontology): The ontology to process.

        Returns:
            str: The processed ontology as JSON string.
        """
        # Convert the ontology object to a JSON.
        ontology_dict = ontology.to_json()
        
        # Remove unique and required attributes from the ontology.
        for entity in ontology_dict["entities"]:
            for attribute in entity["attributes"]:
                del attribute['unique']
                del attribute['required']
        
        for relation in ontology_dict["relations"]:
            for attribute in relation["attributes"]:
                del attribute['unique']
                del attribute['required']
        
        # Return the transformed ontology as a JSON string
        return json.dumps(ontology_dict)
    
    def get_performance_metrics(self) -> dict:
        """
        Get performance metrics for the chat session.
        
        Returns:
            dict: Performance metrics summary.
        """
        if not self._enable_monitoring:
            return {"monitoring_enabled": False}
        
        return self.performance_monitor.get_metrics_summary()
    
    def enable_performance_monitoring(self, enabled: bool = True):
        """
        Enable or disable performance monitoring.
        
        Args:
            enabled (bool): Whether to enable monitoring.
        """
        self._enable_monitoring = enabled
    
    def reset_performance_metrics(self):
        """Reset performance metrics"""
        if self._enable_monitoring:
            self.performance_monitor = PerformanceMonitor()
    
    def get_memory_stats(self) -> dict:
        """
        Get memory usage statistics.
        
        Returns:
            dict: Memory usage statistics.
        """
        if not self._enable_memory_profiling:
            return {"memory_profiling_enabled": False}
        
        return self.memory_profiler.get_memory_stats()
    
    def enable_memory_profiling(self, enabled: bool = True):
        """
        Enable or disable memory profiling.
        
        Args:
            enabled (bool): Whether to enable memory profiling.
        """
        self._enable_memory_profiling = enabled