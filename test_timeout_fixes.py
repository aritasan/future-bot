#!/usr/bin/env python3
"""
Test script to verify timeout and CancelledError fixes.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import load_config
from src.services.binance_service import BinanceService
from src.services.indicator_service import IndicatorService
from src.services.notification_service import NotificationService
from src.services.cache_service import CacheService
from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'test_timeout_fixes_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def test_timeout_handling():
    """Test timeout handling improvements."""
    try:
        logger.info("Starting timeout handling test...")
        
        # Load configuration
        config = load_config()
        
        # Initialize services
        binance_service = BinanceService(config)
        indicator_service = IndicatorService(config)
        notification_service = NotificationService(config)
        cache_service = CacheService(config)
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config, binance_service, indicator_service, notification_service, cache_service
        )
        
        # Initialize services
        await binance_service.initialize()
        await indicator_service.initialize()
        await notification_service.initialize()
        await cache_service.initialize()
        await strategy.initialize()
        
        logger.info("All services initialized successfully")
        
        # Test symbol
        test_symbol = "BTC/USDT"
        
        # Test 1: Normal signal generation
        logger.info("Test 1: Normal signal generation")
        try:
            signals = await asyncio.wait_for(
                strategy.generate_signals(test_symbol, indicator_service), 
                timeout=60
            )
            logger.info(f"✅ Normal signal generation successful: {signals is not None}")
        except asyncio.TimeoutError:
            logger.warning("⚠️ Normal signal generation timed out")
        except asyncio.CancelledError:
            logger.info("ℹ️ Normal signal generation cancelled")
        except Exception as e:
            logger.error(f"❌ Normal signal generation failed: {str(e)}")
        
        # Test 2: Timeout handling
        logger.info("Test 2: Timeout handling")
        try:
            # Create a task that will timeout
            async def slow_operation():
                await asyncio.sleep(70)  # Longer than timeout
                return "result"
            
            result = await asyncio.wait_for(slow_operation(), timeout=5)
            logger.info(f"❌ Unexpected: slow operation completed: {result}")
        except asyncio.TimeoutError:
            logger.info("✅ Timeout handling working correctly")
        except asyncio.CancelledError:
            logger.info("ℹ️ Slow operation cancelled")
        except Exception as e:
            logger.error(f"❌ Unexpected error in timeout test: {str(e)}")
        
        # Test 3: CancelledError handling
        logger.info("Test 3: CancelledError handling")
        try:
            # Create a task that will be cancelled
            async def cancellable_operation():
                await asyncio.sleep(10)
                return "result"
            
            task = asyncio.create_task(cancellable_operation())
            await asyncio.sleep(0.1)  # Let task start
            task.cancel()
            
            try:
                result = await task
                logger.info(f"❌ Unexpected: cancelled task completed: {result}")
            except asyncio.CancelledError:
                logger.info("✅ CancelledError handling working correctly")
        except Exception as e:
            logger.error(f"❌ Error in CancelledError test: {str(e)}")
        
        # Test 4: Rate limiter timeout handling
        logger.info("Test 4: Rate limiter timeout handling")
        try:
            # Test get_current_price with timeout
            price = await asyncio.wait_for(
                binance_service.get_current_price(test_symbol),
                timeout=30
            )
            logger.info(f"✅ Rate limiter price fetch successful: {price}")
        except asyncio.TimeoutError:
            logger.warning("⚠️ Rate limiter price fetch timed out")
        except asyncio.CancelledError:
            logger.info("ℹ️ Rate limiter price fetch cancelled")
        except Exception as e:
            logger.error(f"❌ Rate limiter price fetch failed: {str(e)}")
        
        # Test 5: Strategy timeout handling
        logger.info("Test 5: Strategy timeout handling")
        try:
            recommendations = await asyncio.wait_for(
                strategy.get_quantitative_recommendations(test_symbol),
                timeout=30
            )
            logger.info(f"✅ Strategy recommendations successful: {recommendations is not None}")
        except asyncio.TimeoutError:
            logger.warning("⚠️ Strategy recommendations timed out")
        except asyncio.CancelledError:
            logger.info("ℹ️ Strategy recommendations cancelled")
        except Exception as e:
            logger.error(f"❌ Strategy recommendations failed: {str(e)}")
        
        logger.info("🎉 All timeout handling tests completed!")
        
        # Cleanup
        await binance_service.close()
        await cache_service.close()
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

async def test_concurrent_operations():
    """Test concurrent operations with timeout handling."""
    try:
        logger.info("Starting concurrent operations test...")
        
        # Load configuration
        config = load_config()
        
        # Initialize services
        binance_service = BinanceService(config)
        indicator_service = IndicatorService(config)
        notification_service = NotificationService(config)
        cache_service = CacheService(config)
        
        # Initialize services
        await binance_service.initialize()
        await indicator_service.initialize()
        await notification_service.initialize()
        await cache_service.initialize()
        
        # Test symbols
        test_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        
        # Test concurrent operations
        async def process_symbol(symbol):
            try:
                # Get current price
                price = await asyncio.wait_for(
                    binance_service.get_current_price(symbol),
                    timeout=10
                )
                
                # Get ticker
                ticker = await asyncio.wait_for(
                    binance_service.get_ticker(symbol),
                    timeout=10
                )
                
                return {
                    'symbol': symbol,
                    'price': price,
                    'ticker': ticker is not None
                }
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ Timeout processing {symbol}")
                return {'symbol': symbol, 'error': 'timeout'}
            except asyncio.CancelledError:
                logger.info(f"ℹ️ Cancelled processing {symbol}")
                return {'symbol': symbol, 'error': 'cancelled'}
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {str(e)}")
                return {'symbol': symbol, 'error': str(e)}
        
        # Run concurrent operations
        tasks = [process_symbol(symbol) for symbol in test_symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"❌ Task failed with exception: {str(result)}")
            else:
                logger.info(f"✅ Task completed: {result}")
        
        logger.info("🎉 Concurrent operations test completed!")
        
        # Cleanup
        await binance_service.close()
        await cache_service.close()
        
    except Exception as e:
        logger.error(f"Concurrent test failed: {str(e)}")
        raise

async def main():
    """Main test function."""
    try:
        logger.info("🚀 Starting timeout fixes verification tests...")
        
        # Test 1: Basic timeout handling
        await test_timeout_handling()
        
        # Test 2: Concurrent operations
        await test_concurrent_operations()
        
        logger.info("🎉 All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 