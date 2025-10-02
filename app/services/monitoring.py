"""
Monitoring service for Prometheus metrics and observability
"""

import time
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from loguru import logger


class MonitoringService:
    """Service for monitoring and metrics collection"""
    
    def __init__(self):
        # Define Prometheus metrics
        self.document_processing_counter = Counter(
            'di_documents_processed_total',
            'Total number of documents processed',
            ['status']
        )
        
        self.query_processing_counter = Counter(
            'di_queries_processed_total',
            'Total number of queries processed',
            ['type', 'status']
        )
        
        self.query_response_time = Histogram(
            'di_query_response_time_seconds',
            'Time taken to process queries',
            ['type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        self.agent_execution_counter = Counter(
            'di_agent_executions_total',
            'Total number of agent executions',
            ['agent_id', 'status']
        )
        
        self.agent_execution_time = Histogram(
            'di_agent_execution_time_seconds',
            'Time taken for agent executions',
            ['agent_id'],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0]
        )
        
        self.cache_hit_counter = Counter(
            'di_cache_hits_total',
            'Total number of cache hits',
            ['cache_type']
        )
        
        self.cache_miss_counter = Counter(
            'di_cache_misses_total',
            'Total number of cache misses',
            ['cache_type']
        )
        
        self.vector_search_time = Histogram(
            'di_vector_search_time_seconds',
            'Time taken for vector similarity search',
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
        )
        
        self.active_documents_gauge = Gauge(
            'di_active_documents',
            'Number of active documents in the system'
        )
        
        self.active_queries_gauge = Gauge(
            'di_active_queries',
            'Number of active queries being processed'
        )
        
        self.vector_store_size_gauge = Gauge(
            'di_vector_store_size',
            'Number of vectors in the vector store'
        )
        
        logger.info("Monitoring service initialized")
    
    def record_document_processing(self, status: str):
        """Record document processing metric"""
        self.document_processing_counter.labels(status=status).inc()
    
    def record_query_processing(self, query_type: str, status: str, duration: float):
        """Record query processing metric"""
        self.query_processing_counter.labels(type=query_type, status=status).inc()
        self.query_response_time.labels(type=query_type).observe(duration)
    
    def record_agent_execution(self, agent_id: str, status: str, duration: float):
        """Record agent execution metric"""
        self.agent_execution_counter.labels(agent_id=agent_id, status=status).inc()
        self.agent_execution_time.labels(agent_id=agent_id).observe(duration)
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit"""
        self.cache_hit_counter.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss"""
        self.cache_miss_counter.labels(cache_type=cache_type).inc()
    
    def record_vector_search(self, duration: float):
        """Record vector search time"""
        self.vector_search_time.observe(duration)
    
    def update_active_documents(self, count: int):
        """Update active documents gauge"""
        self.active_documents_gauge.set(count)
    
    def update_active_queries(self, count: int):
        """Update active queries gauge"""
        self.active_queries_gauge.set(count)
    
    def update_vector_store_size(self, size: int):
        """Update vector store size gauge"""
        self.vector_store_size_gauge.set(size)
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest()
    
    def get_metrics_content_type(self) -> str:
        """Get content type for metrics endpoint"""
        return CONTENT_TYPE_LATEST


# Global monitoring instance
monitoring = MonitoringService()


def setup_monitoring():
    """Setup monitoring and metrics collection"""
    logger.info("Setting up monitoring and metrics collection")
    
    # Start metrics collection
    # In a real implementation, you would start background tasks here
    # to collect system metrics periodically
    
    return monitoring
