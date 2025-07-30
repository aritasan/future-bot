"""
Rate limiter utility for managing API requests to avoid rate limit errors.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 2400  # Binance default
    requests_per_second: int = 40    # Conservative limit
    burst_limit: int = 100           # Allow burst for important requests
    retry_after_429: int = 60        # Wait 60 seconds after 429
    exponential_backoff: bool = True
    max_retry_delay: int = 300       # 5 minutes max delay

class RateLimiter:
    """Advanced rate limiter with multiple strategies."""
    
    def __init__(self, config: RateLimitConfig = None):
        """Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        self._request_history = defaultdict(deque)
        self._last_429_time = 0
        self._consecutive_429_count = 0
        self._backoff_multiplier = 1
        self._lock = asyncio.Lock()
        self._priority_queue = asyncio.PriorityQueue()
        self._processing_task = None
        self._stopped = False
        
        # Request type weights (lower = higher priority)
        self._request_weights = {
            'order': 1,           # Trading orders (highest priority)
            'position': 2,        # Position management
            'balance': 3,         # Account balance
            'market_data': 4,     # Market data
            'ticker': 5,          # Price tickers
            'orderbook': 6,       # Order book data
            'trades': 7,          # Trade history
            'funding': 8,         # Funding rate
            'open_interest': 9,   # Open interest
            'klines': 10,         # Historical data
            'symbols': 11,        # Symbol list
            'default': 12         # Default weight
        }
        
    async def start(self):
        """Start the rate limiter processing."""
        if self._processing_task is None:
            self._processing_task = asyncio.create_task(self._process_queue())
            logger.info("Rate limiter started")
    
    async def stop(self):
        """Stop the rate limiter processing."""
        self._stopped = True
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None
        logger.info("Rate limiter stopped")
    
    async def _process_queue(self):
        """Process requests from the priority queue."""
        while not self._stopped:
            try:
                # Get next request from queue with timeout
                try:
                    priority, timestamp, request_id, func, args, kwargs, future = await asyncio.wait_for(
                        self._priority_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Check if we can make the request
                await self._wait_if_needed()
                
                # Execute the request
                try:
                    result = await func(*args, **kwargs)
                    if not future.done():
                        future.set_result(result)
                except Exception as e:
                    if not future.done():
                        future.set_exception(e)
                
                # Record the request
                await self._record_request(request_id)
                
                # Mark task as done
                self._priority_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("Rate limiter processing cancelled")
                break
            except Exception as e:
                logger.error(f"Error in rate limiter processing: {str(e)}")
                # Continue processing other requests
                continue
    
    async def _wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        async with self._lock:
            current_time = time.time()
            
            # Check if we're in 429 backoff period
            if current_time - self._last_429_time < self.config.retry_after_429:
                wait_time = self.config.retry_after_429 - (current_time - self._last_429_time)
                logger.warning(f"Waiting {wait_time:.1f}s due to recent 429 error")
                await asyncio.sleep(wait_time)
                return
            
            # Check rate limits
            requests_last_minute = len([t for t in self._request_history['minute'] 
                                      if current_time - t < 60])
            requests_last_second = len([t for t in self._request_history['second'] 
                                      if current_time - t < 1])
            
            # Calculate wait time
            wait_time = 0
            
            if requests_last_minute >= self.config.requests_per_minute:
                # Wait until we can make another request
                oldest_request = min(self._request_history['minute'])
                wait_time = max(wait_time, 60 - (current_time - oldest_request))
            
            if requests_last_second >= self.config.requests_per_second:
                # Wait until next second
                wait_time = max(wait_time, 1 - (current_time % 1))
            
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
    
    async def _record_request(self, request_id: str):
        """Record a request for rate limiting."""
        current_time = time.time()
        
        # Record in different time windows
        self._request_history['minute'].append(current_time)
        self._request_history['second'].append(current_time)
        
        # Clean up old records
        cutoff_minute = current_time - 60
        cutoff_second = current_time - 1
        
        while self._request_history['minute'] and self._request_history['minute'][0] < cutoff_minute:
            self._request_history['minute'].popleft()
        
        while self._request_history['second'] and self._request_history['second'][0] < cutoff_second:
            self._request_history['second'].popleft()
    
    async def execute(self, func: Callable, *args, request_type: str = 'default', **kwargs) -> Any:
        """Execute a function with rate limiting.
        
        Args:
            func: Function to execute
            *args: Function arguments
            request_type: Type of request for priority
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        # Create future for the result
        future = asyncio.Future()
        
        # Calculate priority based on request type and timestamp
        priority = self._request_weights.get(request_type, self._request_weights['default'])
        timestamp = time.time()
        request_id = f"{request_type}_{timestamp}_{id(func)}"
        
        # Add to priority queue
        await self._priority_queue.put((priority, timestamp, request_id, func, args, kwargs, future))
        
        # Wait for result
        return await future
    
    async def handle_429_error(self, error: Exception):
        """Handle 429 rate limit error."""
        async with self._lock:
            current_time = time.time()
            self._last_429_time = current_time
            self._consecutive_429_count += 1
            
            # Calculate backoff delay
            if self.config.exponential_backoff:
                delay = min(self.config.retry_after_429 * (2 ** (self._consecutive_429_count - 1)), 
                           self.config.max_retry_delay)
            else:
                delay = self.config.retry_after_429
            
            logger.warning(f"429 error detected (count: {self._consecutive_429_count}), "
                          f"backing off for {delay}s")
            
            # Clear recent request history to allow recovery
            self._request_history['minute'].clear()
            self._request_history['second'].clear()
            
            # Wait for backoff period
            await asyncio.sleep(delay)
    
    async def reset_429_backoff(self):
        """Reset 429 backoff when successful requests are made."""
        async with self._lock:
            if self._consecutive_429_count > 0:
                self._consecutive_429_count = 0
                self._backoff_multiplier = 1
                logger.info("429 backoff reset after successful request")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        current_time = time.time()
        
        return {
            'requests_last_minute': len([t for t in self._request_history['minute'] 
                                       if current_time - t < 60]),
            'requests_last_second': len([t for t in self._request_history['second'] 
                                       if current_time - t < 1]),
            'queue_size': self._priority_queue.qsize(),
            'consecutive_429_count': self._consecutive_429_count,
            'last_429_time': self._last_429_time,
            'time_since_last_429': current_time - self._last_429_time
        }

class AdaptiveRateLimiter(RateLimiter):
    """Rate limiter that adapts to API response patterns."""
    
    def __init__(self, config: RateLimitConfig = None):
        super().__init__(config)
        self._success_count = 0
        self._error_count = 0
        self._adaptive_limits = {
            'requests_per_minute': self.config.requests_per_minute,
            'requests_per_second': self.config.requests_per_second
        }
        self._adaptation_window = 300  # 5 minutes
        self._adaptation_history = deque(maxlen=100)
    
    async def _adapt_limits(self, success: bool):
        """Adapt rate limits based on success/failure."""
        current_time = time.time()
        self._adaptation_history.append((current_time, success))
        
        # Calculate success rate in adaptation window
        recent_requests = [(t, s) for t, s in self._adaptation_history 
                          if current_time - t < self._adaptation_window]
        
        if len(recent_requests) >= 10:  # Need minimum data
            success_rate = sum(1 for _, s in recent_requests if s) / len(recent_requests)
            
            if success_rate > 0.95:  # High success rate, can increase limits
                self._adaptive_limits['requests_per_minute'] = min(
                    self._adaptive_limits['requests_per_minute'] * 1.1,
                    self.config.requests_per_minute
                )
                self._adaptive_limits['requests_per_second'] = min(
                    self._adaptive_limits['requests_per_second'] * 1.05,
                    self.config.requests_per_second
                )
                logger.debug(f"Adaptive limits increased: {self._adaptive_limits}")
                
            elif success_rate < 0.8:  # Low success rate, decrease limits
                self._adaptive_limits['requests_per_minute'] = max(
                    self._adaptive_limits['requests_per_minute'] * 0.8,
                    self.config.requests_per_minute * 0.5
                )
                self._adaptive_limits['requests_per_second'] = max(
                    self._adaptive_limits['requests_per_second'] * 0.9,
                    self.config.requests_per_second * 0.5
                )
                logger.debug(f"Adaptive limits decreased: {self._adaptive_limits}")
    
    async def _wait_if_needed(self):
        """Wait if rate limit would be exceeded (with adaptive limits)."""
        async with self._lock:
            current_time = time.time()
            
            # Check if we're in 429 backoff period
            if current_time - self._last_429_time < self.config.retry_after_429:
                wait_time = self.config.retry_after_429 - (current_time - self._last_429_time)
                logger.warning(f"Waiting {wait_time:.1f}s due to recent 429 error")
                await asyncio.sleep(wait_time)
                return
            
            # Check rate limits with adaptive limits
            requests_last_minute = len([t for t in self._request_history['minute'] 
                                      if current_time - t < 60])
            requests_last_second = len([t for t in self._request_history['second'] 
                                      if current_time - t < 1])
            
            # Calculate wait time
            wait_time = 0
            
            if requests_last_minute >= self._adaptive_limits['requests_per_minute']:
                oldest_request = min(self._request_history['minute'])
                wait_time = max(wait_time, 60 - (current_time - oldest_request))
            
            if requests_last_second >= self._adaptive_limits['requests_per_second']:
                wait_time = max(wait_time, 1 - (current_time % 1))
            
            if wait_time > 0:
                logger.debug(f"Adaptive rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
    
    async def execute(self, func: Callable, *args, request_type: str = 'default', **kwargs) -> Any:
        """Execute a function with adaptive rate limiting."""
        try:
            result = await super().execute(func, *args, request_type=request_type, **kwargs)
            await self._adapt_limits(True)
            return result
        except Exception as e:
            await self._adapt_limits(False)
            raise e 