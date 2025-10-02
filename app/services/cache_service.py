"""
Cache service for Redis-based query result caching
"""

import json
import hashlib
from typing import Dict, Any, Optional
import redis
from loguru import logger

from app.core.config import settings


class CacheService:
    """Service for caching query results and embeddings"""
    
    def __init__(self):
        self.redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        self.cache_ttl = settings.cache_ttl
        self.max_cache_size = settings.max_cache_size
    
    async def get_query_result(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached query result"""
        try:
            cache_key = self._generate_query_cache_key(query)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                result = json.loads(cached_data)
                result['cached'] = True
                logger.info(f"Cache hit for query: {query[:50]}...")
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached query result: {str(e)}")
            return None
    
    async def cache_query_result(self, query: str, result: Dict[str, Any]) -> bool:
        """Cache query result"""
        try:
            cache_key = self._generate_query_cache_key(query)
            
            # Remove cached flag before storing
            cache_data = result.copy()
            cache_data.pop('cached', None)
            
            # Store in Redis with TTL
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(cache_data)
            )
            
            logger.info(f"Cached query result: {query[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error caching query result: {str(e)}")
            return False
    
    async def get_embedding(self, text: str) -> Optional[list]:
        """Get cached embedding"""
        try:
            cache_key = self._generate_embedding_cache_key(text)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached embedding: {str(e)}")
            return None
    
    async def cache_embedding(self, text: str, embedding: list) -> bool:
        """Cache embedding"""
        try:
            cache_key = self._generate_embedding_cache_key(text)
            
            # Store in Redis with longer TTL for embeddings
            self.redis_client.setex(
                cache_key,
                self.cache_ttl * 24,  # 24 hours for embeddings
                json.dumps(embedding)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error caching embedding: {str(e)}")
            return False
    
    async def invalidate_query_cache(self, query_pattern: str = None) -> int:
        """Invalidate query cache entries"""
        try:
            if query_pattern:
                pattern = f"query:{query_pattern}*"
            else:
                pattern = "query:*"
            
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Error invalidating cache: {str(e)}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            info = self.redis_client.info()
            
            return {
                'total_keys': info.get('db0', {}).get('keys', 0),
                'memory_usage': info.get('used_memory_human', '0B'),
                'hit_rate': info.get('keyspace_hits', 0) / max(
                    info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1
                ),
                'connected_clients': info.get('connected_clients', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {}
    
    def _generate_query_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        return f"query:{query_hash}"
    
    def _generate_embedding_cache_key(self, text: str) -> str:
        """Generate cache key for embedding"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"embedding:{text_hash}"
    
    async def clear_all_cache(self) -> bool:
        """Clear all cache entries"""
        try:
            self.redis_client.flushdb()
            logger.info("Cleared all cache entries")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
