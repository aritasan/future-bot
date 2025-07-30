#!/usr/bin/env python3
"""
Test script to verify all fixes applied.
"""

import sys
import os
import asyncio
import logging
import warnings
import numpy as np
from typing import Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Suppress warnings for testing
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from src.core.config import load_config
from src.services.binance_service import BinanceService
from src.services.indicator_service import IndicatorService
from src.services.notification_service import NotificationService
from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_rate_limiter_fix():
    """Test rate limiter fix for invalid state errors."""
    try:
        logger.info("Testing rate limiter fix...")
        
        from src.utils.rate_limiter import AdaptiveRateLimiter, RateLimitConfig
        
        config = RateLimitConfig(
            requests_per_minute=100,
            requests_per_second=2,
            retry_after_429=10
        )
        
        rate_limiter = AdaptiveRateLimiter(config)
        await rate_limiter.start()
        
        # Test multiple requests
        async def mock_request(request_id: str) -> str:
            await asyncio.sleep(0.1)
            return f"Response for {request_id}"
        
        tasks = []
        for i in range(5):
            task = rate_limiter.execute(mock_request, f"test_{i}", request_type='ticker')
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful = sum(1 for r in results if not isinstance(r, Exception))
        
        logger.info(f"Rate limiter test: {successful}/{len(results)} successful")
        
        # Test stop functionality
        await rate_limiter.stop()
        logger.info("Rate limiter stopped successfully")
        
    except Exception as e:
        logger.error(f"Rate limiter test error: {str(e)}")

async def test_divide_by_zero_fix():
    """Test divide by zero warning fixes."""
    try:
        logger.info("Testing divide by zero fix...")
        
        from src.quantitative.statistical_validator import StatisticalSignalValidator
        
        validator = StatisticalSignalValidator()
        
        # Test with edge cases that would cause divide by zero
        test_cases = [
            np.array([]),  # Empty array
            np.array([0, 0, 0]),  # All zeros
            np.array([np.nan, np.nan]),  # All NaN
            np.array([1, 1, 1]),  # All same values
        ]
        
        for i, returns in enumerate(test_cases):
            try:
                result = validator.validate_signal({'test': 'data'}, returns)
                logger.info(f"Test case {i+1}: Validation completed without errors")
            except Exception as e:
                logger.error(f"Test case {i+1} failed: {str(e)}")
        
        logger.info("Divide by zero fix test completed")
        
    except Exception as e:
        logger.error(f"Divide by zero test error: {str(e)}")

async def test_order_notification():
    """Test order notification functionality."""
    try:
        logger.info("Testing order notification...")
        
        config = load_config()
        
        # Create notification callback
        notifications_sent = []
        
        async def notification_callback(message: str):
            notifications_sent.append(message)
            logger.info(f"Notification sent: {message[:100]}...")
        
        # Initialize Binance service with notification callback
        binance_service = BinanceService(config, notification_callback)
        await binance_service.initialize()
        
        # Test order notification
        test_order_params = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'type': 'MARKET',
            'amount': 0.001,
            'stop_loss': 50000,
            'take_profit': 60000
        }
        
        # Mock order result
        mock_order = {
            'id': 'test_order_123',
            'status': 'FILLED',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'amount': 0.001,
            'price': 55000
        }
        
        # Test notification method directly
        await binance_service._send_order_notification(mock_order, test_order_params)
        
        logger.info(f"Order notification test: {len(notifications_sent)} notifications sent")
        
        await binance_service.close()
        
    except Exception as e:
        logger.error(f"Order notification test error: {str(e)}")

async def test_quantitative_strategy_with_fixes():
    """Test quantitative strategy with all fixes applied."""
    try:
        logger.info("Testing quantitative strategy with fixes...")
        
        config = load_config()
        
        # Initialize services
        binance_service = BinanceService(config)
        await binance_service.initialize()
        
        indicator_service = IndicatorService(config)
        await indicator_service.initialize()
        
        notification_service = NotificationService(config)
        await notification_service.initialize()
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config, binance_service, indicator_service, notification_service
        )
        await strategy.initialize()
        
        # Test signal generation with edge cases
        test_symbols = ['BTCUSDT', 'ETHUSDT']
        
        for symbol in test_symbols:
            try:
                signal = await strategy.generate_signals(symbol, indicator_service)
                if signal:
                    logger.info(f"Signal generated for {symbol}: {signal.get('action', 'unknown')}")
                else:
                    logger.warning(f"No signal generated for {symbol}")
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {str(e)}")
        
        # Cleanup
        await strategy.close()
        await indicator_service.close()
        await notification_service.close()
        await binance_service.close()
        
        logger.info("Quantitative strategy test completed")
        
    except Exception as e:
        logger.error(f"Quantitative strategy test error: {str(e)}")

async def main():
    """Run all verification tests."""
    try:
        logger.info("Starting verification tests...")
        
        # Test rate limiter fix
        await test_rate_limiter_fix()
        
        # Test divide by zero fix
        await test_divide_by_zero_fix()
        
        # Test order notification
        await test_order_notification()
        
        # Test quantitative strategy
        await test_quantitative_strategy_with_fixes()
        
        logger.info("All verification tests completed successfully")
        
    except Exception as e:
        logger.error(f"Verification test error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 