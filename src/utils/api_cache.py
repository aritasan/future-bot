"""
API cache utility for caching API responses.
"""

from typing import Any, Optional
from datetime import datetime, timedelta
import logging

class APICache:
    """
    Cache for API responses with TTL (Time To Live) support.
    """
    
    def __init__(self, ttl_seconds: int = 60):
        """
        Initialize the API cache.
        
        Args:
            ttl_seconds: Time in seconds before cache entries expire
        """
        self.logger = logging.getLogger(__name__)
        self.ttl_seconds = ttl_seconds
        self.cache: dict[str, tuple[Any, datetime]] = {}
        
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Cached value if exists and not expired, None otherwise
        """
        if key not in self.cache:
            return None
            
        value, timestamp = self.cache[key]
        if datetime.now() - timestamp > timedelta(seconds=self.ttl_seconds):
            del self.cache[key]
            return None
            
        return value
        
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        try:
            self.cache[key] = (value, datetime.now())
        except Exception as e:
            self.logger.error(f"Error setting cache value: {str(e)}")
            
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        
    def remove_expired(self) -> None:
        """Remove all expired cache entries."""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp > timedelta(seconds=self.ttl_seconds)
        ]
        for key in expired_keys:
            del self.cache[key] 