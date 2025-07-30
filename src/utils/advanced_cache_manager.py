"""
Advanced Cache Manager with NATS Integration.
Provides distributed caching capabilities with Redis and NATS messaging.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import hashlib
import pickle
import gzip

import redis.asyncio as redis
import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg

logger = logging.getLogger(__name__)

class AdvancedCacheManager:
    """
    Advanced Cache Manager with NATS integration for distributed caching.
    Supports multiple cache layers, automatic invalidation, and performance monitoring.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 nats_url: str = "nats://localhost:4222",
                 cache_config: Dict = None):
        """
        Initialize Advanced Cache Manager.
        
        Args:
            redis_url: Redis connection URL
            nats_url: NATS connection URL
            cache_config: Cache configuration
        """
        self.redis_url = redis_url
        self.nats_url = nats_url
        self.cache_config = cache_config or {
            'default_ttl': 3600,
            'max_size': 1000,
            'compression_enabled': True,
            'distributed_cache_enabled': True,
            'cache_layers': ['memory', 'redis', 'distributed']
        }
        
        # Initialize connections
        self.redis_client = None
        self.nats_client = None
        
        # Cache statistics
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0,
            'compression_savings': 0
        }
        
        # Memory cache (L1)
        self.memory_cache = {}
        self.memory_cache_size = 0
        self.memory_cache_max_size = 100
        
        # Cache invalidation patterns
        self.invalidation_patterns = {}
        
        # Performance tracking
        self.performance_metrics = {
            'response_times': [],
            'cache_hit_rates': [],
            'memory_usage': []
        }
        
        logger.info("Advanced Cache Manager initialized")
    
    async def initialize(self) -> bool:
        """Initialize cache manager connections."""
        try:
            # Initialize Redis connection
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=False,
                max_connections=20
            )
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
            # Initialize NATS connection
            self.nats_client = NATS()
            await self.nats_client.connect(self.nats_url)
            logger.info("NATS connection established")
            
            # Subscribe to cache invalidation events
            await self._subscribe_to_cache_events()
            
            # Start background tasks
            asyncio.create_task(self._background_cleanup())
            asyncio.create_task(self._performance_monitoring())
            
            logger.info("Advanced Cache Manager initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing cache manager: {str(e)}")
            return False
    
    async def _subscribe_to_cache_events(self):
        """Subscribe to NATS cache events."""
        try:
            # Subscribe to cache invalidation events
            await self.nats_client.subscribe(
                "cache.invalidate",
                cb=self._handle_cache_invalidation
            )
            
            # Subscribe to cache statistics events
            await self.nats_client.subscribe(
                "cache.stats",
                cb=self._handle_cache_stats
            )
            
            # Subscribe to cache performance events
            await self.nats_client.subscribe(
                "cache.performance",
                cb=self._handle_performance_event
            )
            
            logger.info("Subscribed to cache events")
            
        except Exception as e:
            logger.error(f"Error subscribing to cache events: {str(e)}")
    
    async def _handle_cache_invalidation(self, msg: Msg):
        """Handle cache invalidation events."""
        try:
            data = json.loads(msg.data.decode())
            pattern = data.get('pattern')
            keys = data.get('keys', [])
            
            if pattern:
                await self._invalidate_by_pattern(pattern)
            elif keys:
                await self._invalidate_keys(keys)
                
        except Exception as e:
            logger.error(f"Error handling cache invalidation: {str(e)}")
    
    async def _handle_cache_stats(self, msg: Msg):
        """Handle cache statistics events."""
        try:
            data = json.loads(msg.data.decode())
            # Update local stats with distributed stats
            for key, value in data.items():
                if key in self.cache_stats:
                    self.cache_stats[key] += value
                    
        except Exception as e:
            logger.error(f"Error handling cache stats: {str(e)}")
    
    async def _handle_performance_event(self, msg: Msg):
        """Handle performance monitoring events."""
        try:
            data = json.loads(msg.data.decode())
            # Update performance metrics
            if 'response_time' in data:
                self.performance_metrics['response_times'].append(data['response_time'])
                
        except Exception as e:
            logger.error(f"Error handling performance event: {str(e)}")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache with multi-layer support.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        start_time = time.time()
        
        try:
            # Try memory cache first (L1)
            if key in self.memory_cache:
                self.cache_stats['hits'] += 1
                self._update_performance_metrics(time.time() - start_time)
                return self.memory_cache[key]['value']
            
            # Try Redis cache (L2)
            if self.redis_client:
                redis_value = await self.redis_client.get(self._encode_key(key))
                if redis_value:
                    # Decompress if needed
                    value = self._decompress_value(redis_value)
                    
                    # Update memory cache
                    await self._update_memory_cache(key, value)
                    
                    self.cache_stats['hits'] += 1
                    self._update_performance_metrics(time.time() - start_time)
                    return value
            
            # Cache miss
            self.cache_stats['misses'] += 1
            self._update_performance_metrics(time.time() - start_time)
            return default
            
        except Exception as e:
            self.cache_stats['errors'] += 1
            logger.error(f"Error getting cache key {key}: {str(e)}")
            return default
    
    async def set(self, key: str, value: Any, ttl: int = None, 
                  cache_layer: str = 'all') -> bool:
        """
        Set value in cache with multi-layer support.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            cache_layer: Cache layer to use ('memory', 'redis', 'all')
            
        Returns:
            True if successful
        """
        start_time = time.time()
        
        try:
            ttl = ttl or self.cache_config['default_ttl']
            expiry = datetime.now() + timedelta(seconds=ttl)
            
            # Compress value if enabled
            if self.cache_config['compression_enabled']:
                compressed_value = self._compress_value(value)
                compression_savings = len(str(value).encode()) - len(compressed_value)
                self.cache_stats['compression_savings'] += compression_savings
            else:
                compressed_value = pickle.dumps(value)
            
            # Set in memory cache (L1)
            if cache_layer in ['memory', 'all']:
                await self._set_memory_cache(key, value, expiry)
            
            # Set in Redis cache (L2)
            if cache_layer in ['redis', 'all'] and self.redis_client:
                await self.redis_client.setex(
                    self._encode_key(key),
                    ttl,
                    compressed_value
                )
            
            # Publish to distributed cache if enabled
            if self.cache_config['distributed_cache_enabled'] and self.nats_client:
                await self._publish_cache_update(key, value, ttl)
            
            self.cache_stats['sets'] += 1
            self._update_performance_metrics(time.time() - start_time)
            return True
            
        except Exception as e:
            self.cache_stats['errors'] += 1
            logger.error(f"Error setting cache key {key}: {str(e)}")
            return False
    
    async def delete(self, key: str, cache_layer: str = 'all') -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            cache_layer: Cache layer to use
            
        Returns:
            True if successful
        """
        try:
            # Delete from memory cache
            if cache_layer in ['memory', 'all']:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    self.memory_cache_size -= 1
            
            # Delete from Redis cache
            if cache_layer in ['redis', 'all'] and self.redis_client:
                await self.redis_client.delete(self._encode_key(key))
            
            # Publish invalidation event
            if self.nats_client:
                await self._publish_cache_invalidation(key)
            
            self.cache_stats['deletes'] += 1
            return True
            
        except Exception as e:
            self.cache_stats['errors'] += 1
            logger.error(f"Error deleting cache key {key}: {str(e)}")
            return False
    
    async def invalidate_pattern(self, pattern: str) -> bool:
        """
        Invalidate cache keys matching pattern.
        
        Args:
            pattern: Pattern to match keys
            
        Returns:
            True if successful
        """
        try:
            # Invalidate from Redis
            if self.redis_client:
                keys = await self.redis_client.keys(self._encode_key(pattern))
                if keys:
                    await self.redis_client.delete(*keys)
            
            # Invalidate from memory cache
            keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k]
            for key in keys_to_delete:
                del self.memory_cache[key]
                self.memory_cache_size -= 1
            
            # Publish pattern invalidation
            if self.nats_client:
                await self._publish_pattern_invalidation(pattern)
            
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating pattern {pattern}: {str(e)}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            # Calculate hit rate
            total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
            
            # Get Redis info
            redis_info = {}
            if self.redis_client:
                info = await self.redis_client.info()
                redis_info = {
                    'used_memory': info.get('used_memory', 0),
                    'connected_clients': info.get('connected_clients', 0),
                    'total_commands_processed': info.get('total_commands_processed', 0)
                }
            
            return {
                'cache_stats': self.cache_stats,
                'hit_rate': hit_rate,
                'memory_cache_size': self.memory_cache_size,
                'redis_info': redis_info,
                'performance_metrics': self.performance_metrics,
                'compression_savings_mb': self.cache_stats['compression_savings'] / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {}
    
    async def _set_memory_cache(self, key: str, value: Any, expiry: datetime):
        """Set value in memory cache with LRU eviction."""
        try:
            # Check if cache is full
            if self.memory_cache_size >= self.memory_cache_max_size:
                # Remove least recently used item
                lru_key = min(self.memory_cache.keys(), 
                             key=lambda k: self.memory_cache[k]['last_accessed'])
                del self.memory_cache[lru_key]
                self.memory_cache_size -= 1
            
            # Add new item
            self.memory_cache[key] = {
                'value': value,
                'expiry': expiry,
                'last_accessed': time.time()
            }
            self.memory_cache_size += 1
            
        except Exception as e:
            logger.error(f"Error setting memory cache: {str(e)}")
    
    async def _update_memory_cache(self, key: str, value: Any):
        """Update memory cache with accessed value."""
        try:
            if key in self.memory_cache:
                self.memory_cache[key]['last_accessed'] = time.time()
            else:
                await self._set_memory_cache(key, value, datetime.now() + timedelta(hours=1))
                
        except Exception as e:
            logger.error(f"Error updating memory cache: {str(e)}")
    
    def _encode_key(self, key: str) -> str:
        """Encode cache key."""
        return f"trading_bot:{key}"
    
    def _compress_value(self, value: Any) -> bytes:
        """Compress value using gzip."""
        try:
            serialized = pickle.dumps(value)
            return gzip.compress(serialized)
        except Exception as e:
            logger.error(f"Error compressing value: {str(e)}")
            return pickle.dumps(value)
    
    def _decompress_value(self, compressed_value: bytes) -> Any:
        """Decompress value using gzip."""
        try:
            decompressed = gzip.decompress(compressed_value)
            return pickle.loads(decompressed)
        except Exception as e:
            logger.error(f"Error decompressing value: {str(e)}")
            return pickle.loads(compressed_value)
    
    async def _publish_cache_update(self, key: str, value: Any, ttl: int):
        """Publish cache update to NATS."""
        try:
            # Convert datetime to string to make it JSON serializable
            message = {
                'action': 'update',
                'key': key,
                'value': value,
                'ttl': ttl,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.nats_client.publish(
                "cache.update",
                json.dumps(message, default=str).encode()
            )
            
        except Exception as e:
            logger.error(f"Error publishing cache update: {str(e)}")
    
    async def _publish_cache_invalidation(self, key: str):
        """Publish cache invalidation to NATS."""
        try:
            message = {
                'action': 'invalidate',
                'key': key,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.nats_client.publish(
                "cache.invalidate",
                json.dumps(message, default=str).encode()
            )
            
        except Exception as e:
            logger.error(f"Error publishing cache invalidation: {str(e)}")
    
    async def _publish_pattern_invalidation(self, pattern: str):
        """Publish pattern invalidation to NATS."""
        try:
            message = {
                'action': 'invalidate_pattern',
                'pattern': pattern,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.nats_client.publish(
                "cache.invalidate",
                json.dumps(message, default=str).encode()
            )
            
        except Exception as e:
            logger.error(f"Error publishing pattern invalidation: {str(e)}")
    
    async def _background_cleanup(self):
        """Background task for cache cleanup."""
        while True:
            try:
                # Clean expired items from memory cache
                current_time = time.time()
                expired_keys = []
                
                for key, item in self.memory_cache.items():
                    if item['expiry'].timestamp() < current_time:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.memory_cache[key]
                    self.memory_cache_size -= 1
                
                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired cache items")
                
                # Publish cache statistics
                if self.nats_client:
                    await self.nats_client.publish(
                        "cache.stats",
                        json.dumps(self.cache_stats, default=str).encode()
                    )
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in background cleanup: {str(e)}")
                await asyncio.sleep(60)
    
    async def _performance_monitoring(self):
        """Background task for performance monitoring."""
        while True:
            try:
                # Calculate average response time
                if self.performance_metrics['response_times']:
                    avg_response_time = sum(self.performance_metrics['response_times']) / len(self.performance_metrics['response_times'])
                    
                    # Publish performance metrics
                    if self.nats_client:
                        performance_data = {
                            'avg_response_time': avg_response_time,
                            'cache_hit_rate': self.cache_stats['hits'] / (self.cache_stats['hits'] + self.cache_stats['misses']) if (self.cache_stats['hits'] + self.cache_stats['misses']) > 0 else 0,
                            'memory_usage': self.memory_cache_size,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        await self.nats_client.publish(
                            "cache.performance",
                            json.dumps(performance_data, default=str).encode()
                        )
                
                # Clear old performance data
                self.performance_metrics['response_times'] = self.performance_metrics['response_times'][-100:]
                
                await asyncio.sleep(30)  # Run every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {str(e)}")
                await asyncio.sleep(30)
    
    def _update_performance_metrics(self, response_time: float):
        """Update performance metrics."""
        self.performance_metrics['response_times'].append(response_time)
    
    async def close(self):
        """Close cache manager connections."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            if self.nats_client:
                await self.nats_client.close()
            
            logger.info("Advanced Cache Manager closed")
            
        except Exception as e:
            logger.error(f"Error closing cache manager: {str(e)}") 