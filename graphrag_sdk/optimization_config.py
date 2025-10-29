"""
Configuration options for query and chat session performance optimizations.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class QueryOptimizationConfig:
    """
    Configuration for query performance optimizations.
    
    This class provides centralized configuration for all optimization features
    implemented in Phase 1, 2, and 3.
    """
    
    # Query result caching settings
    enable_query_caching: bool = True
    cache_ttl: int = 3600  # 1 hour in seconds
    max_cache_size: int = 1000
    
    # Session pooling settings
    enable_session_pooling: bool = True
    max_pool_size: int = 5
    
    # Parallel execution settings
    enable_parallel_execution: bool = True
    max_workers: int = 2
    
    # History optimization settings
    max_history_length: int = 20
    compression_threshold: int = 10
    enable_summarization: bool = True
    
    # Retry settings
    max_retries: int = 10
    base_delay: float = 0.1
    max_delay: float = 2.0
    jitter_factor: float = 0.1
    
    # Ontology caching settings
    enable_ontology_caching: bool = True
    max_ontology_cache_size: int = 100
    
    # Performance monitoring settings
    enable_performance_monitoring: bool = True
    metrics_retention_hours: int = 24  # Keep metrics for 24 hours
    
    # Memory profiling settings
    enable_memory_profiling: bool = False
    memory_check_interval: int = 300  # Check memory every 5 minutes
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "query_caching": {
                "enabled": self.enable_query_caching,
                "ttl_seconds": self.cache_ttl,
                "max_size": self.max_cache_size
            },
            "session_pooling": {
                "enabled": self.enable_session_pooling,
                "max_pool_size": self.max_pool_size
            },
            "parallel_execution": {
                "enabled": self.enable_parallel_execution,
                "max_workers": self.max_workers
            },
            "history_optimization": {
                "max_history_length": self.max_history_length,
                "compression_threshold": self.compression_threshold,
                "enable_summarization": self.enable_summarization
            },
            "retry_logic": {
                "max_retries": self.max_retries,
                "base_delay": self.base_delay,
                "max_delay": self.max_delay,
                "jitter_factor": self.jitter_factor
            },
            "ontology_caching": {
                "enabled": self.enable_ontology_caching,
                "max_cache_size": self.max_ontology_cache_size
            },
            "performance_monitoring": {
                "enabled": self.enable_performance_monitoring,
                "metrics_retention_hours": self.metrics_retention_hours
            },
            "memory_profiling": {
                "enabled": self.enable_memory_profiling,
                "check_interval_seconds": self.memory_check_interval
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "QueryOptimizationConfig":
        """Create configuration from dictionary."""
        return cls(
            enable_query_caching=config_dict.get("enable_query_caching", True),
            cache_ttl=config_dict.get("cache_ttl", 3600),
            max_cache_size=config_dict.get("max_cache_size", 1000),
            enable_session_pooling=config_dict.get("enable_session_pooling", True),
            max_pool_size=config_dict.get("max_pool_size", 5),
            enable_parallel_execution=config_dict.get("enable_parallel_execution", True),
            max_workers=config_dict.get("max_workers", 2),
            max_history_length=config_dict.get("max_history_length", 20),
            compression_threshold=config_dict.get("compression_threshold", 10),
            enable_summarization=config_dict.get("enable_summarization", True),
            max_retries=config_dict.get("max_retries", 10),
            base_delay=config_dict.get("base_delay", 0.1),
            max_delay=config_dict.get("max_delay", 2.0),
            jitter_factor=config_dict.get("jitter_factor", 0.1),
            enable_ontology_caching=config_dict.get("enable_ontology_caching", True),
            max_ontology_cache_size=config_dict.get("max_ontology_cache_size", 100),
            enable_performance_monitoring=config_dict.get("enable_performance_monitoring", True),
            metrics_retention_hours=config_dict.get("metrics_retention_hours", 24),
            enable_memory_profiling=config_dict.get("enable_memory_profiling", False),
            memory_check_interval=config_dict.get("memory_check_interval", 300)
        )
    
    @classmethod
    def conservative(cls) -> "QueryOptimizationConfig":
        """Create a conservative configuration with minimal optimizations."""
        return cls(
            enable_query_caching=True,
            cache_ttl=1800,  # 30 minutes
            max_cache_size=500,
            enable_session_pooling=True,
            max_pool_size=3,
            enable_parallel_execution=False,
            max_workers=1,
            max_history_length=15,
            compression_threshold=8,
            enable_summarization=False,
            max_retries=5,
            base_delay=0.2,
            max_delay=1.0,
            jitter_factor=0.05,
            enable_ontology_caching=True,
            max_ontology_cache_size=50,
            enable_performance_monitoring=True,
            metrics_retention_hours=12,
            enable_memory_profiling=False,
            memory_check_interval=600
        )
    
    @classmethod
    def aggressive(cls) -> "QueryOptimizationConfig":
        """Create an aggressive configuration with maximum optimizations."""
        return cls(
            enable_query_caching=True,
            cache_ttl=7200,  # 2 hours
            max_cache_size=2000,
            enable_session_pooling=True,
            max_pool_size=10,
            enable_parallel_execution=True,
            max_workers=4,
            max_history_length=30,
            compression_threshold=15,
            enable_summarization=True,
            max_retries=15,
            base_delay=0.05,
            max_delay=5.0,
            jitter_factor=0.2,
            enable_ontology_caching=True,
            max_ontology_cache_size=200,
            enable_performance_monitoring=True,
            metrics_retention_hours=48,
            enable_memory_profiling=True,
            memory_check_interval=120
        )
    
    @classmethod
    def disabled(cls) -> "QueryOptimizationConfig":
        """Create a configuration with all optimizations disabled."""
        return cls(
            enable_query_caching=False,
            enable_session_pooling=False,
            enable_parallel_execution=False,
            enable_summarization=False,
            enable_ontology_caching=False,
            enable_performance_monitoring=False,
            enable_memory_profiling=False
        )


# Default configuration instance
DEFAULT_OPTIMIZATION_CONFIG = QueryOptimizationConfig()