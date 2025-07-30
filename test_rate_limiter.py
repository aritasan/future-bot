#!/usr/bin/env python3
"""
Test script to verify rate limiter functionality.
"""

import sys
import os
import asyncio
import logging
import time
from typing import Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.rate_limiter import AdaptiveRateLimiter, RateLimitConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_rate_limiter():
    """Test rate limiter functionality."""
    
    try:
        # Create rate limiter with conservative settings for testing
        config = RateLimitConfig(
            requests_per_minute=100,  # Low limit for testing
            requests_per_second=2,    # Very low limit for testing
            retry_after_429=10,      # Short retry for testing
            exponential_backoff=True,
            max_retry_delay=30
        )
        
        rate_limiter = AdaptiveRateLimiter(config)
        await rate_limiter.start()
        
        logger.info("Rate limiter started")
        
        # Test function that simulates API call
        async def mock_api_call(request_id: str, delay: float = 0.1) -> str:
            """Mock API call with configurable delay."""
            await asyncio.sleep(delay)
            return f"Response for {request_id}"
        
        # Test normal requests
        logger.info("Testing normal requests...")
        start_time = time.time()
        
        tasks = []
        for i in range(10):
            task = rate_limiter.execute(
                mock_api_call, 
                f"request_{i}", 
                request_type='ticker'
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        logger.info(f"Completed {len(results)} requests in {end_time - start_time:.2f}s")
        logger.info(f"Results: {results}")
        
        # Test rate limiting
        logger.info("Testing rate limiting...")
        start_time = time.time()
        
        # Try to make many requests quickly
        tasks = []
        for i in range(20):
            task = rate_limiter.execute(
                mock_api_call, 
                f"burst_request_{i}", 
                request_type='orderbook'
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        logger.info(f"Completed {len(results)} burst requests in {end_time - start_time:.2f}s")
        
        # Test 429 error handling
        logger.info("Testing 429 error handling...")
        
        async def mock_api_call_with_429(request_id: str) -> str:
            """Mock API call that sometimes returns 429."""
            await asyncio.sleep(0.1)
            if request_id.endswith('_429'):
                raise Exception("429 Too Many Requests")
            return f"Response for {request_id}"
        
        try:
            # Make some normal requests
            for i in range(3):
                result = await rate_limiter.execute(
                    mock_api_call_with_429, 
                    f"normal_{i}", 
                    request_type='ticker'
                )
                logger.info(f"Normal request {i}: {result}")
            
            # Make a request that triggers 429
            result = await rate_limiter.execute(
                mock_api_call_with_429, 
                "test_429", 
                request_type='ticker'
            )
            logger.info(f"429 request result: {result}")
            
        except Exception as e:
            logger.info(f"Expected 429 error caught: {str(e)}")
        
        # Get statistics
        stats = rate_limiter.get_stats()
        logger.info(f"Rate limiter stats: {stats}")
        
        # Stop rate limiter
        await rate_limiter.stop()
        logger.info("Rate limiter stopped")
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")

async def test_priority_queue():
    """Test priority queue functionality."""
    
    try:
        config = RateLimitConfig(
            requests_per_minute=100,
            requests_per_second=5,
            retry_after_429=5
        )
        
        rate_limiter = AdaptiveRateLimiter(config)
        await rate_limiter.start()
        
        logger.info("Testing priority queue...")
        
        async def mock_api_call(request_id: str) -> str:
            await asyncio.sleep(0.1)
            return f"Response for {request_id}"
        
        # Create requests with different priorities
        tasks = [
            rate_limiter.execute(mock_api_call, "low_priority", request_type='klines'),
            rate_limiter.execute(mock_api_call, "high_priority", request_type='order'),
            rate_limiter.execute(mock_api_call, "medium_priority", request_type='ticker'),
            rate_limiter.execute(mock_api_call, "very_high_priority", request_type='position'),
        ]
        
        results = await asyncio.gather(*tasks)
        logger.info(f"Priority queue results: {results}")
        
        await rate_limiter.stop()
        
    except Exception as e:
        logger.error(f"Error in priority queue test: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_rate_limiter())
    asyncio.run(test_priority_queue()) 