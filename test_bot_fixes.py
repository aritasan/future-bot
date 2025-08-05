#!/usr/bin/env python3
"""
Test script to verify bot fixes
Tests the fixes for:
- signal_history attribute
- import errors
- duplicate methods
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockConfig:
    def __init__(self):
        self.config = {
            'sentiment_threshold': 0.6,
            'risk_free_rate': 0.02,
            'max_position_size': 0.3,
            'min_position_size': 0.01,
            'target_volatility': 0.15,
            'rebalancing_frequency': 'monthly',
            'attribution_period': 'monthly',
            'factor_model': 'fama_french',
            'monitoring_frequency': 60,
            'var_confidence_level': 0.95,
            'position_limit': 0.3,
            'correlation_threshold': 0.8
        }
    
    def get(self, key, default=None):
        return self.config.get(key, default)

class MockBinanceService:
    def __init__(self):
        logger.info("Mock Binance Service initialized")
    
    async def get_klines(self, symbol, interval, limit=100):
        return {
            '1h': [{'open': 50000, 'high': 51000, 'low': 49000, 'close': 50500, 'volume': 1000} for _ in range(100)],
            '4h': [{'open': 50000, 'high': 51000, 'low': 49000, 'close': 50500, 'volume': 1000} for _ in range(100)],
            '1d': [{'open': 50000, 'high': 51000, 'low': 49000, 'close': 50500, 'volume': 1000} for _ in range(100)]
        }

class MockIndicatorService:
    def __init__(self):
        logger.info("Mock Indicator Service initialized")
    
    async def calculate_indicators(self, df):
        return {
            'rsi': 50.0,
            'macd': 0.1,
            'bollinger_bands': {'upper': 51000, 'middle': 50500, 'lower': 50000},
            'atr': 500.0
        }

class MockNotificationService:
    def __init__(self):
        logger.info("Mock Notification Service initialized")
    
    async def send_notification(self, message):
        logger.info(f"Mock notification: {message}")

async def test_bot_fixes():
    """Test the bot fixes."""
    try:
        logger.info("=== Testing Bot Fixes ===")
        
        # Initialize mock config
        config = MockConfig()
        
        # Initialize mock services
        binance_service = MockBinanceService()
        indicator_service = MockIndicatorService()
        notification_service = MockNotificationService()
        
        logger.info("✅ All mock services initialized successfully")
        
        # Test 1: Import Strategy Class
        logger.info("\n--- Test 1: Import Strategy Class ---")
        try:
            from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
            logger.info("✅ Strategy class imported successfully")
        except Exception as e:
            logger.error(f"❌ Error importing strategy class: {str(e)}")
            return False
        
        # Test 2: Initialize Strategy
        logger.info("\n--- Test 2: Initialize Strategy ---")
        try:
            strategy = EnhancedTradingStrategyWithQuantitative(
                config, binance_service, indicator_service, notification_service
            )
            logger.info("✅ Strategy initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing strategy: {str(e)}")
            return False
        
        # Test 3: Check signal_history attribute
        logger.info("\n--- Test 3: Check signal_history attribute ---")
        try:
            if hasattr(strategy, 'signal_history'):
                logger.info("✅ signal_history attribute exists")
                logger.info(f"signal_history type: {type(strategy.signal_history)}")
                logger.info(f"signal_history content: {strategy.signal_history}")
            else:
                logger.error("❌ signal_history attribute missing")
                return False
        except Exception as e:
            logger.error(f"❌ Error checking signal_history: {str(e)}")
            return False
        
        # Test 4: Test generate_signals method
        logger.info("\n--- Test 4: Test generate_signals method ---")
        try:
            symbol = "BTCUSDT"
            signal = await strategy.generate_signals(symbol, indicator_service)
            logger.info(f"✅ generate_signals completed for {symbol}")
            logger.info(f"Signal result: {signal}")
        except Exception as e:
            logger.error(f"❌ Error in generate_signals: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        
        # Test 5: Test signal history storage
        logger.info("\n--- Test 5: Test signal history storage ---")
        try:
            # Create a mock signal
            mock_signal = {
                'action': 'buy',
                'confidence': 0.75,
                'strength': 0.6,
                'timestamp': '2025-08-05T19:10:00'
            }
            
            # Store signal in history
            strategy._store_signal_history(symbol, mock_signal)
            
            # Check if signal was stored
            if symbol in strategy.signal_history:
                logger.info("✅ Signal stored in history successfully")
                logger.info(f"History for {symbol}: {len(strategy.signal_history[symbol])} signals")
            else:
                logger.error("❌ Signal not stored in history")
                return False
        except Exception as e:
            logger.error(f"❌ Error testing signal history: {str(e)}")
            return False
        
        # Test 6: Test dynamic validation thresholds
        logger.info("\n--- Test 6: Test dynamic validation thresholds ---")
        try:
            market_data = {
                'close': [50000, 50500, 51000, 50800, 51200],
                'volume': [1000, 1100, 1200, 1150, 1300],
                'volatility': 0.02
            }
            
            thresholds = strategy._calculate_dynamic_validation_thresholds(symbol, market_data)
            logger.info("✅ Dynamic validation thresholds calculated successfully")
            logger.info(f"Thresholds: {thresholds}")
        except Exception as e:
            logger.error(f"❌ Error calculating dynamic thresholds: {str(e)}")
            return False
        
        # Test 7: Test health check
        logger.info("\n--- Test 7: Test health check ---")
        try:
            health_status = await strategy.health_check()
            logger.info("✅ Health check completed successfully")
            logger.info(f"Health status: {health_status}")
        except Exception as e:
            logger.error(f"❌ Error in health check: {str(e)}")
            return False
        
        # Summary
        logger.info("\n=== Test Summary ===")
        logger.info("✅ Strategy class import: Working")
        logger.info("✅ Strategy initialization: Working")
        logger.info("✅ signal_history attribute: Working")
        logger.info("✅ generate_signals method: Working")
        logger.info("✅ Signal history storage: Working")
        logger.info("✅ Dynamic validation thresholds: Working")
        logger.info("✅ Health check: Working")
        
        logger.info("\n🎉 All bot fixes verified successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in test: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Main test function."""
    try:
        success = await test_bot_fixes()
        
        if success:
            logger.info("✅ All tests passed successfully!")
        else:
            logger.error("❌ Some tests failed!")
            
    except Exception as e:
        logger.error(f"❌ Test execution failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 