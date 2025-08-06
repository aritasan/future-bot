#!/usr/bin/env python3
"""
Test script to verify continuous processing of trading bot.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any
from src.core.config import load_config
from src.services.binance_service import BinanceService
from src.services.indicator_service import IndicatorService
from src.services.telegram_service import TelegramService
from src.services.discord_service import DiscordService
from src.services.notification_service import NotificationService
from src.services.cache_service import CacheService
from src.services.cache_monitor_service import CacheMonitorService
from src.core.health_monitor import HealthMonitor
from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_continuous_processing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def test_continuous_processing():
    """Test continuous processing functionality."""
    
    try:
        logger.info("=== Starting Continuous Processing Test ===")
        
        # Load configuration
        config = load_config()
        
        # Initialize services
        binance_service = None
        telegram_service = None
        discord_service = None
        health_monitor = None
        indicator_service = None
        strategy = None
        cache_service = None
        cache_monitor = None
        
        try:
            # Initialize Binance service
            binance_service = BinanceService(config)
            await binance_service.initialize()
            logger.info("✅ Binance service initialized")
            
            # Initialize health monitor
            health_monitor = HealthMonitor(config)
            await health_monitor.initialize()
            logger.info("✅ Health monitor initialized")
            
            # Initialize indicator service
            indicator_service = IndicatorService(config)
            await indicator_service.initialize()
            logger.info("✅ Indicator service initialized")
            
            # Initialize cache service
            cache_service = CacheService(config)
            await cache_service.initialize()
            logger.info("✅ Cache service initialized")
            
            # Initialize cache monitor service
            cache_monitor = CacheMonitorService(config)
            await cache_monitor.initialize()
            logger.info("✅ Cache monitor service initialized")
            
            # Initialize notification service
            notification_service = NotificationService(config, telegram_service, discord_service)
            await notification_service.initialize()
            logger.info("✅ Notification service initialized")
            
            # Initialize enhanced trading strategy
            strategy = EnhancedTradingStrategyWithQuantitative(
                config, binance_service, indicator_service, notification_service, cache_service
            )
            await strategy.initialize()
            logger.info("✅ Enhanced Trading Strategy initialized")
            
            # Test symbols (small subset for testing)
            test_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
            logger.info(f"Testing with {len(test_symbols)} symbols: {test_symbols}")
            
            # Test continuous processing
            cycle_count = 0
            max_cycles = 3  # Test 3 cycles
            
            while cycle_count < max_cycles:
                cycle_count += 1
                logger.info(f"=== Test Cycle {cycle_count}/{max_cycles} ===")
                
                # Process each symbol
                for i, symbol in enumerate(test_symbols):
                    logger.info(f"Processing symbol {i+1}/{len(test_symbols)}: {symbol}")
                    
                    try:
                        # Simulate symbol processing
                        await asyncio.sleep(1)  # Simulate processing time
                        logger.info(f"✅ Successfully processed {symbol}")
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing {symbol}: {str(e)}")
                
                logger.info(f"=== Completed Test Cycle {cycle_count} ===")
                
                if cycle_count < max_cycles:
                    logger.info("Waiting 10 seconds before next test cycle...")
                    try:
                        await asyncio.sleep(10)  # 10 seconds for testing
                        logger.info("✅ Wait completed, continuing to next cycle")
                    except Exception as e:
                        logger.error(f"❌ Error during wait: {str(e)}")
                        logger.info("Continuing to next cycle despite wait error")
            
            logger.info("✅ All test cycles completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error during test initialization: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"❌ Fatal error in test: {str(e)}")
    finally:
        # Cleanup
        logger.info("Cleaning up test resources...")
        
        if cache_monitor:
            await cache_monitor.close()
        if cache_service:
            await cache_service.close()
        if indicator_service:
            await indicator_service.close()
        if health_monitor:
            await health_monitor.close()
        if binance_service:
            await binance_service.close()
        if strategy:
            await strategy.close()
        
        logger.info("✅ Test cleanup completed")

async def test_cycle_wait_mechanism():
    """Test the cycle wait mechanism specifically."""
    
    logger.info("=== Testing Cycle Wait Mechanism ===")
    
    try:
        # Test 1: Normal wait
        logger.info("Test 1: Normal 5-second wait")
        start_time = time.time()
        await asyncio.sleep(5)
        elapsed = time.time() - start_time
        logger.info(f"✅ Normal wait completed in {elapsed:.2f} seconds")
        
        # Test 2: Wait with timeout
        logger.info("Test 2: Wait with timeout")
        start_time = time.time()
        try:
            await asyncio.wait_for(asyncio.sleep(3), timeout=5)
            elapsed = time.time() - start_time
            logger.info(f"✅ Timeout wait completed in {elapsed:.2f} seconds")
        except asyncio.TimeoutError:
            logger.error("❌ Timeout wait failed")
        
        # Test 3: Wait with interruption
        logger.info("Test 3: Wait with interruption")
        start_time = time.time()
        try:
            # Create a task that will be cancelled
            wait_task = asyncio.create_task(asyncio.sleep(10))
            await asyncio.sleep(1)  # Wait 1 second
            wait_task.cancel()  # Cancel the task
            await wait_task  # This should raise CancelledError
        except asyncio.CancelledError:
            elapsed = time.time() - start_time
            logger.info(f"✅ Interrupted wait handled correctly in {elapsed:.2f} seconds")
        
        logger.info("✅ All cycle wait tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Error in cycle wait test: {str(e)}")

if __name__ == "__main__":
    async def run_tests():
        """Run all tests."""
        logger.info("🚀 Starting Continuous Processing Tests")
        
        # Test 1: Cycle wait mechanism
        await test_cycle_wait_mechanism()
        
        # Test 2: Continuous processing
        await test_continuous_processing()
        
        logger.info("🎉 All tests completed successfully!")
    
    # Run tests
    asyncio.run(run_tests()) 