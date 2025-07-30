#!/usr/bin/env python3
"""
Test script to verify rate limiter integration with BinanceService.
"""

import sys
import os
import asyncio
import logging
import time
from typing import Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import load_config
from src.services.binance_service import BinanceService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_binance_rate_limiter():
    """Test rate limiter integration with BinanceService."""
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize Binance service
        binance_service = BinanceService(config)
        await binance_service.initialize()
        logger.info("Binance service initialized with rate limiter")
        
        # Test different types of requests
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        logger.info("Testing ticker requests (low priority)...")
        for symbol in test_symbols:
            try:
                ticker = await binance_service.get_ticker(symbol)
                logger.info(f"Ticker for {symbol}: {ticker.get('last', 'N/A')}")
                await asyncio.sleep(0.1)  # Small delay between requests
            except Exception as e:
                logger.error(f"Error getting ticker for {symbol}: {str(e)}")
        
        logger.info("Testing order book requests (medium priority)...")
        for symbol in test_symbols:
            try:
                orderbook = await binance_service.get_order_book(symbol)
                if orderbook:
                    logger.info(f"Order book for {symbol}: {len(orderbook.get('bids', []))} bids, {len(orderbook.get('asks', []))} asks")
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error getting order book for {symbol}: {str(e)}")
        
        logger.info("Testing funding rate requests (low priority)...")
        for symbol in test_symbols:
            try:
                funding_rate = await binance_service.get_funding_rate(symbol)
                logger.info(f"Funding rate for {symbol}: {funding_rate}")
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error getting funding rate for {symbol}: {str(e)}")
        
        logger.info("Testing open interest requests (low priority)...")
        for symbol in test_symbols:
            try:
                open_interest = await binance_service.get_open_interest(symbol)
                logger.info(f"Open interest for {symbol}: {open_interest}")
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error getting open interest for {symbol}: {str(e)}")
        
        # Test burst requests to see rate limiting in action
        logger.info("Testing burst requests to trigger rate limiting...")
        tasks = []
        for i in range(10):
            symbol = test_symbols[i % len(test_symbols)]
            task = binance_service.get_ticker(symbol)
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for r in results if not isinstance(r, Exception))
            logger.info(f"Burst test completed: {successful}/{len(results)} successful")
        except Exception as e:
            logger.error(f"Error in burst test: {str(e)}")
        
        # Get rate limiter statistics
        if hasattr(binance_service, '_rate_limiter'):
            stats = binance_service._rate_limiter.get_stats()
            logger.info(f"Rate limiter stats: {stats}")
        
        # Close service
        await binance_service.close()
        logger.info("Binance service closed")
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")

async def test_rate_limiter_configuration():
    """Test different rate limiter configurations."""
    
    try:
        from src.utils.rate_limiter import AdaptiveRateLimiter, RateLimitConfig
        
        logger.info("Testing rate limiter configurations...")
        
        # Test conservative configuration
        conservative_config = RateLimitConfig(
            requests_per_minute=100,
            requests_per_second=2,
            retry_after_429=30,
            exponential_backoff=True,
            max_retry_delay=120
        )
        
        conservative_limiter = AdaptiveRateLimiter(conservative_config)
        await conservative_limiter.start()
        
        # Test aggressive configuration
        aggressive_config = RateLimitConfig(
            requests_per_minute=2000,
            requests_per_second=30,
            retry_after_429=10,
            exponential_backoff=True,
            max_retry_delay=60
        )
        
        aggressive_limiter = AdaptiveRateLimiter(aggressive_config)
        await aggressive_limiter.start()
        
        logger.info("Conservative config stats:")
        logger.info(f"- Requests per minute: {conservative_config.requests_per_minute}")
        logger.info(f"- Requests per second: {conservative_config.requests_per_second}")
        logger.info(f"- Retry after 429: {conservative_config.retry_after_429}s")
        
        logger.info("Aggressive config stats:")
        logger.info(f"- Requests per minute: {aggressive_config.requests_per_minute}")
        logger.info(f"- Requests per second: {aggressive_config.requests_per_second}")
        logger.info(f"- Retry after 429: {aggressive_config.retry_after_429}s")
        
        # Cleanup
        await conservative_limiter.stop()
        await aggressive_limiter.stop()
        
        logger.info("Rate limiter configuration test completed")
        
    except Exception as e:
        logger.error(f"Error in configuration test: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_rate_limiter_configuration())
    asyncio.run(test_binance_rate_limiter()) 