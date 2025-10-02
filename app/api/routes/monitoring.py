"""
API routes for monitoring and observability
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any
from loguru import logger
from sqlalchemy import func

from app.core.database import get_db
from app.services.vector_store import VectorStoreService
from app.services.cache_service import CacheService
from app.models import Document, Query, AgentExecution


router = APIRouter()


@router.get("/stats")
async def get_system_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    try:
        # Document statistics
        total_documents = db.query(Document).count()
        processed_documents = db.query(Document).filter(Document.processed == True).count()
        
        # Query statistics
        total_queries = db.query(Query).count()
        avg_processing_time = db.query(func.avg(Query.processing_time)).scalar() or 0
        
        # Agent execution statistics
        total_executions = db.query(AgentExecution).count()
        successful_executions = db.query(AgentExecution).filter(
            AgentExecution.success == True
        ).count()
        
        # Vector store statistics
        vector_store = VectorStoreService()
        vector_stats = vector_store.get_stats()
        
        # Cache statistics
        cache_service = CacheService()
        cache_stats = await cache_service.get_cache_stats()
        
        return {
            "documents": {
                "total": total_documents,
                "processed": processed_documents,
                "pending": total_documents - processed_documents
            },
            "queries": {
                "total": total_queries,
                "average_processing_time": round(avg_processing_time, 3)
            },
            "agents": {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "success_rate": round(
                    successful_executions / max(total_executions, 1) * 100, 2
                )
            },
            "vector_store": vector_stats,
            "cache": cache_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",
            "components": {}
        }
        
        # Check database connection
        try:
            from app.core.database import engine
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            health_status["components"]["database"] = "healthy"
        except Exception as e:
            health_status["components"]["database"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check Redis connection
        try:
            cache_service = CacheService()
            await cache_service.get_cache_stats()
            health_status["components"]["redis"] = "healthy"
        except Exception as e:
            health_status["components"]["redis"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check vector store
        try:
            vector_store = VectorStoreService()
            vector_store.get_stats()
            health_status["components"]["vector_store"] = "healthy"
        except Exception as e:
            health_status["components"]["vector_store"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": "2024-01-01T00:00:00Z",
            "error": str(e)
        }


@router.get("/metrics")
async def get_metrics():
    """Get Prometheus-compatible metrics"""
    try:
        # This would integrate with Prometheus client
        # For now, return basic metrics structure
        return {
            "metrics": {
                "document_processing_rate": "0.0",
                "query_response_time": "0.0",
                "cache_hit_rate": "0.0",
                "vector_search_latency": "0.0"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_cache():
    """Clear all cache entries"""
    try:
        cache_service = CacheService()
        success = await cache_service.clear_all_cache()
        
        if success:
            return {"message": "Cache cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear cache")
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
