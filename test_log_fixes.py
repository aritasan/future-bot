#!/usr/bin/env python3
"""
Test script to verify that the log errors are fixed.
"""

import asyncio
import logging
from unittest.mock import Mock, AsyncMock
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
from src.services.indicator_service import IndicatorService
from src.services.notification_service import NotificationService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockBinanceService:
    """Mock BinanceService for testing."""
    
    def __init__(self):
        self.mock_account_info = {
            'totalWalletBalance': '1000.0',
            'availableBalance': '1000.0'
        }
    
    async def get_account_info(self):
        """Mock get_account_info."""
        return self.mock_account_info

class MockIndicatorService:
    """Mock IndicatorService for testing."""
    
    async def get_klines(self, symbol, timeframe='1h', limit=100):
        """Mock get_klines."""
        return {
            'close': [50000, 50100, 50200, 50300, 50400],
            'high': [50500, 50600, 50700, 50800, 50900],
            'low': [49500, 49600, 49700, 49800, 49900],
            'volume': [100, 110, 120, 130, 140]
        }

class MockNotificationService:
    """Mock NotificationService for testing."""
    
    async def send_notification(self, message):
        """Mock send_notification."""
        logger.info(f"Mock notification: {message}")

async def test_indicator_service_get_klines():
    """Test that IndicatorService has get_klines method."""
    logger.info("Testing IndicatorService get_klines method...")
    
    # Create config
    config = {
        'api': {
            'binance': {
                'use_testnet': True,
                'testnet': {
                    'api_key': 'test_key',
                    'api_secret': 'test_secret'
                }
            }
        }
    }
    
    # Create indicator service
    indicator_service = IndicatorService(config)
    
    # Check if get_klines method exists
    assert hasattr(indicator_service, 'get_klines'), "IndicatorService should have get_klines method"
    logger.info("✅ IndicatorService has get_klines method")

async def test_strategy_performance_metrics():
    """Test that strategy performance metrics work correctly."""
    logger.info("Testing strategy performance metrics...")
    
    # Create mock services
    mock_binance = MockBinanceService()
    mock_indicator = MockIndicatorService()
    mock_notification = MockNotificationService()
    
    # Create config
    config = {
        'trading': {
            'default_position_size': 0.01,
            'max_position_size': 0.1,
            'risk_per_trade': 0.02
        },
        'api': {
            'binance': {
                'use_testnet': True,
                'testnet': {
                    'api_key': 'test_key',
                    'api_secret': 'test_secret'
                }
            }
        }
    }
    
    # Create strategy instance
    strategy = EnhancedTradingStrategyWithQuantitative(
        config=config,
        binance_service=mock_binance,
        indicator_service=mock_indicator,
        notification_service=mock_notification
    )
    
    # Test get_performance_metrics
    try:
        metrics = await strategy.get_performance_metrics()
        logger.info(f"Performance metrics: {metrics}")
        assert isinstance(metrics, dict), "Performance metrics should be a dictionary"
        logger.info("✅ Strategy performance metrics work correctly")
    except Exception as e:
        logger.error(f"❌ Strategy performance metrics failed: {e}")
        raise

async def test_market_conditions():
    """Test that _get_market_conditions works correctly."""
    logger.info("Testing _get_market_conditions...")
    
    # Create mock services
    mock_binance = MockBinanceService()
    mock_indicator = MockIndicatorService()
    mock_notification = MockNotificationService()
    
    # Create config
    config = {
        'trading': {
            'default_position_size': 0.01,
            'max_position_size': 0.1,
            'risk_per_trade': 0.02
        },
        'api': {
            'binance': {
                'use_testnet': True,
                'testnet': {
                    'api_key': 'test_key',
                    'api_secret': 'test_secret'
                }
            }
        }
    }
    
    # Create strategy instance
    strategy = EnhancedTradingStrategyWithQuantitative(
        config=config,
        binance_service=mock_binance,
        indicator_service=mock_indicator,
        notification_service=mock_notification
    )
    
    # Test _get_market_conditions
    try:
        market_conditions = await strategy._get_market_conditions('BTCUSDT')
        logger.info(f"Market conditions: {market_conditions}")
        assert isinstance(market_conditions, dict), "Market conditions should be a dictionary"
        assert 'volatility' in market_conditions, "Market conditions should have volatility"
        assert 'price_change_24h' in market_conditions, "Market conditions should have price_change_24h"
        logger.info("✅ _get_market_conditions works correctly")
    except Exception as e:
        logger.error(f"❌ _get_market_conditions failed: {e}")
        raise

async def test_discord_service_none_handling():
    """Test that Discord service None handling works correctly."""
    logger.info("Testing Discord service None handling...")
    
    # Create mock services
    mock_binance = MockBinanceService()
    mock_indicator = MockIndicatorService()
    mock_notification = MockNotificationService()
    
    # Create config
    config = {
        'trading': {
            'default_position_size': 0.01,
            'max_position_size': 0.1,
            'risk_per_trade': 0.02
        },
        'api': {
            'binance': {
                'use_testnet': True,
                'testnet': {
                    'api_key': 'test_key',
                    'api_secret': 'test_secret'
                }
            }
        }
    }
    
    # Create strategy instance
    strategy = EnhancedTradingStrategyWithQuantitative(
        config=config,
        binance_service=mock_binance,
        indicator_service=mock_indicator,
        notification_service=mock_notification
    )
    
    # Test with None Discord service
    try:
        # This should not raise an error
        signals = {
            'optimized_position_size': 0.02,
            'current_price': 50000.0,
            'atr': 1000.0,
            'confidence': 0.8,
            'action': 'BUY'
        }
        
        # Test execute functions with None services
        await strategy._execute_buy_order('BTCUSDT', signals)
        logger.info("✅ Discord service None handling works correctly")
    except Exception as e:
        logger.error(f"❌ Discord service None handling failed: {e}")
        raise

async def main():
    """Run all tests."""
    logger.info("🚀 Starting log fixes tests...")
    
    try:
        await test_indicator_service_get_klines()
        await test_strategy_performance_metrics()
        await test_market_conditions()
        await test_discord_service_none_handling()
        
        logger.info("🎉 All log fixes tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 