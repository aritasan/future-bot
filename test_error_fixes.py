#!/usr/bin/env python3
"""
Test script to verify error fixes.
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

async def test_binance_service_methods():
    """Test BinanceService methods including get_recent_trades."""
    try:
        logger.info("Testing BinanceService methods...")
        
        config = load_config()
        binance_service = BinanceService(config)
        await binance_service.initialize()
        
        # Test get_recent_trades method
        test_symbol = 'BTCUSDT'
        logger.info(f"Testing get_recent_trades for {test_symbol}...")
        
        try:
            trades = await binance_service.get_recent_trades(test_symbol)
            logger.info(f"get_recent_trades successful: {len(trades)} trades")
        except Exception as e:
            logger.error(f"get_recent_trades failed: {str(e)}")
        
        # Test get_trades method
        try:
            trades = await binance_service.get_trades(test_symbol)
            logger.info(f"get_trades successful: {len(trades)} trades")
        except Exception as e:
            logger.error(f"get_trades failed: {str(e)}")
        
        # Test other methods
        try:
            ticker = await binance_service.get_ticker(test_symbol)
            logger.info(f"get_ticker successful: {ticker.get('last', 'N/A')}")
        except Exception as e:
            logger.error(f"get_ticker failed: {str(e)}")
        
        await binance_service.close()
        logger.info("BinanceService test completed")
        
    except Exception as e:
        logger.error(f"BinanceService test error: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")

async def test_quantitative_strategy_fixed():
    """Test quantitative strategy with error fixes."""
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
        
        logger.info("Strategy initialized successfully")
        
        # Test signal generation for multiple symbols
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        for symbol in test_symbols:
            logger.info(f"Testing signal generation for {symbol}...")
            
            try:
                signal = await strategy.generate_signals(symbol, indicator_service)
                if signal:
                    logger.info(f"Signal generated for {symbol}: {signal.get('action', 'unknown')}")
                else:
                    logger.warning(f"No signal generated for {symbol}")
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {str(e)}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        # Cleanup
        await strategy.close()
        await indicator_service.close()
        await notification_service.close()
        await binance_service.close()
        
        logger.info("Quantitative strategy test completed")
        
    except Exception as e:
        logger.error(f"Quantitative strategy test error: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")

async def test_concurrent_processing():
    """Test concurrent processing with limited concurrency."""
    try:
        logger.info("Testing concurrent processing...")
        
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
        
        # Test concurrent signal generation
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT']
        
        async def process_symbol(symbol):
            try:
                signal = await strategy.generate_signals(symbol, indicator_service)
                return {'symbol': symbol, 'success': True, 'signal': signal}
            except Exception as e:
                return {'symbol': symbol, 'success': False, 'error': str(e)}
        
        # Process with limited concurrency
        max_concurrent = 3
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_process(symbol):
            async with semaphore:
                return await process_symbol(symbol)
        
        tasks = [limited_process(symbol) for symbol in test_symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))
        logger.info(f"Concurrent processing completed: {successful}/{len(results)} successful")
        
        # Cleanup
        await strategy.close()
        await indicator_service.close()
        await notification_service.close()
        await binance_service.close()
        
        logger.info("Concurrent processing test completed")
        
    except Exception as e:
        logger.error(f"Concurrent processing test error: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")

async def test_discord_service_fixed():
    """Test Discord service with fixes."""
    try:
        logger.info("Testing Discord service with fixes...")
        
        config = load_config()
        from src.services.discord_service import DiscordService
        
        discord_service = DiscordService(config)
        await discord_service.initialize()
        logger.info("Discord service initialized successfully")
        
        # Test sending a message
        success = await discord_service.send_message("🔧 Error fixes test message")
        logger.info(f"Message sent: {success}")
        
        await discord_service.close()
        logger.info("Discord service test completed")
        
    except Exception as e:
        logger.error(f"Discord service test error: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")

async def main():
    """Run all error fix tests."""
    try:
        logger.info("Starting error fix tests...")
        
        # Test BinanceService methods
        await test_binance_service_methods()
        
        # Test Discord service
        await test_discord_service_fixed()
        
        # Test quantitative strategy
        await test_quantitative_strategy_fixed()
        
        # Test concurrent processing
        await test_concurrent_processing()
        
        logger.info("All error fix tests completed successfully")
        
    except Exception as e:
        logger.error(f"Error fix test error: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main()) 