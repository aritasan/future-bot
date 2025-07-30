#!/usr/bin/env python3
"""
Debug script to identify and fix errors in the trading system.
"""

import sys
import os
import asyncio
import logging
import traceback
from typing import Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import load_config
from src.services.binance_service import BinanceService
from src.services.indicator_service import IndicatorService
from src.services.notification_service import NotificationService
from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_discord_service():
    """Test Discord service initialization."""
    try:
        logger.info("Testing Discord service...")
        
        config = load_config()
        from src.services.discord_service import DiscordService
        
        discord_service = DiscordService(config)
        await discord_service.initialize()
        logger.info("Discord service initialized successfully")
        
        # Test sending a message
        success = await discord_service.send_message("🧪 Debug test message")
        logger.info(f"Message sent: {success}")
        
        await discord_service.close()
        logger.info("Discord service closed")
        
    except Exception as e:
        logger.error(f"Discord service error: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")

async def test_quantitative_strategy():
    """Test quantitative strategy initialization and signal generation."""
    try:
        logger.info("Testing quantitative strategy...")
        
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
        
        logger.info("Strategy initialized successfully")
        
        # Test signal generation for a simple symbol
        test_symbol = 'BTCUSDT'
        logger.info(f"Testing signal generation for {test_symbol}...")
        
        try:
            signal = await strategy.generate_signals(test_symbol, indicator_service)
            if signal:
                logger.info(f"Signal generated successfully: {signal}")
            else:
                logger.warning("No signal generated")
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        # Cleanup
        await strategy.close()
        await indicator_service.close()
        await notification_service.close()
        await binance_service.close()
        
    except Exception as e:
        logger.error(f"Quantitative strategy error: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")

async def test_quantitative_integration():
    """Test quantitative integration module."""
    try:
        logger.info("Testing quantitative integration...")
        
        from src.quantitative.integration import QuantitativeIntegration
        
        config = {
            'confidence_level': 0.95,
            'max_position_size': 0.02,
            'risk_free_rate': 0.02,
            'optimization_method': 'markowitz',
            'n_factors': 5,
            'var_limit': 0.02,
            'quantitative_integration_enabled': True
        }
        
        integration = QuantitativeIntegration(config)
        logger.info("Quantitative integration initialized successfully")
        
        # Test with mock data
        mock_signal = {
            'symbol': 'BTCUSDT',
            'action': 'BUY',
            'confidence': 0.7,
            'position_size': 0.01
        }
        
        mock_market_data = {
            'current_price': 50000,
            'returns': [0.01, -0.02, 0.03, -0.01, 0.02],
            'volatility': 0.02
        }
        
        try:
            enhanced_signal = await integration.enhance_trading_signal(
                'BTCUSDT', mock_signal, mock_market_data
            )
            logger.info(f"Enhanced signal: {enhanced_signal}")
        except Exception as e:
            logger.error(f"Error enhancing signal: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        
    except Exception as e:
        logger.error(f"Quantitative integration error: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")

async def test_concurrent_tasks():
    """Test concurrent task handling."""
    try:
        logger.info("Testing concurrent task handling...")
        
        async def mock_task(symbol: str, delay: float = 0.1):
            """Mock task that simulates trading operations."""
            try:
                await asyncio.sleep(delay)
                logger.info(f"Task completed for {symbol}")
                return f"Success for {symbol}"
            except Exception as e:
                logger.error(f"Task failed for {symbol}: {str(e)}")
                raise
        
        # Test with limited concurrency
        max_concurrent = 5
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_task(symbol):
            async with semaphore:
                return await mock_task(symbol)
        
        # Create multiple tasks
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT']
        tasks = [limited_task(symbol) for symbol in symbols]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Concurrent test completed: {successful}/{len(results)} successful")
        
    except Exception as e:
        logger.error(f"Concurrent task error: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")

async def test_memory_usage():
    """Test memory usage and cleanup."""
    try:
        logger.info("Testing memory usage...")
        
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Create some objects
        test_objects = []
        for i in range(1000):
            test_objects.append({'data': 'x' * 1000, 'index': i})
        
        memory_after_creation = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory after object creation: {memory_after_creation:.2f} MB")
        
        # Cleanup
        del test_objects
        gc.collect()
        
        memory_after_cleanup = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory after cleanup: {memory_after_cleanup:.2f} MB")
        
        logger.info("Memory test completed")
        
    except Exception as e:
        logger.error(f"Memory test error: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")

async def main():
    """Run all debug tests."""
    try:
        logger.info("Starting debug tests...")
        
        # Test Discord service
        await test_discord_service()
        
        # Test quantitative integration
        await test_quantitative_integration()
        
        # Test quantitative strategy
        await test_quantitative_strategy()
        
        # Test concurrent tasks
        await test_concurrent_tasks()
        
        # Test memory usage
        await test_memory_usage()
        
        logger.info("All debug tests completed")
        
    except Exception as e:
        logger.error(f"Debug test error: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main()) 