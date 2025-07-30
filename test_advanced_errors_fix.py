#!/usr/bin/env python3
"""
Test script to verify that the advanced errors are fixed.
"""

import asyncio
import logging
from unittest.mock import Mock, AsyncMock
import sys
import os
import pandas as pd

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
            'open': [50000, 50100, 50200, 50300, 50400],
            'high': [50500, 50600, 50700, 50800, 50900],
            'low': [49500, 49600, 49700, 49800, 49900],
            'close': [50000, 50100, 50200, 50300, 50400],
            'volume': [100, 110, 120, 130, 140]
        }

class MockNotificationService:
    """Mock NotificationService for testing."""
    
    async def send_notification(self, message):
        """Mock send_notification."""
        logger.info(f"Mock notification: {message}")

async def test_convert_klines_to_dataframe():
    """Test that _convert_klines_to_dataframe works correctly."""
    logger.info("Testing _convert_klines_to_dataframe...")
    
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
    
    # Test klines conversion
    klines = {
        'open': [50000, 50100, 50200],
        'high': [50500, 50600, 50700],
        'low': [49500, 49600, 49700],
        'close': [50000, 50100, 50200],
        'volume': [100, 110, 120]
    }
    
    df = strategy._convert_klines_to_dataframe(klines)
    logger.info(f"Converted DataFrame shape: {df.shape}")
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    
    assert not df.empty, "DataFrame should not be empty"
    assert 'close' in df.columns, "DataFrame should have 'close' column"
    assert len(df) == 3, "DataFrame should have 3 rows"
    
    logger.info("✅ _convert_klines_to_dataframe works correctly")

async def test_calculate_advanced_indicators():
    """Test that _calculate_advanced_indicators works correctly."""
    logger.info("Testing _calculate_advanced_indicators...")
    
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
    
    # Create test DataFrame
    test_df = pd.DataFrame({
        'open': [50000, 50100, 50200, 50300, 50400],
        'high': [50500, 50600, 50700, 50800, 50900],
        'low': [49500, 49600, 49700, 49800, 49900],
        'close': [50000, 50100, 50200, 50300, 50400],
        'volume': [100, 110, 120, 130, 140]
    })
    
    # Test advanced indicators calculation
    result_df = await strategy._calculate_advanced_indicators(test_df)
    
    logger.info(f"Result DataFrame shape: {result_df.shape}")
    logger.info(f"Result DataFrame columns: {result_df.columns.tolist()}")
    
    # Check that advanced indicators were added
    expected_indicators = ['sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd', 'rsi', 'atr']
    for indicator in expected_indicators:
        assert indicator in result_df.columns, f"DataFrame should have '{indicator}' column"
    
    logger.info("✅ _calculate_advanced_indicators works correctly")

async def test_get_comprehensive_market_data():
    """Test that _get_comprehensive_market_data works correctly."""
    logger.info("Testing _get_comprehensive_market_data...")
    
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
    
    # Test comprehensive market data
    market_data = await strategy._get_comprehensive_market_data('BTCUSDT')
    
    logger.info(f"Market data: {market_data}")
    
    assert 'symbol' in market_data, "Market data should have 'symbol'"
    assert 'returns' in market_data, "Market data should have 'returns'"
    assert 'volatility' in market_data, "Market data should have 'volatility'"
    assert market_data['symbol'] == 'BTCUSDT', "Symbol should be BTCUSDT"
    
    logger.info("✅ _get_comprehensive_market_data works correctly")

async def test_apply_quantitative_analysis():
    """Test that _apply_quantitative_analysis works correctly."""
    logger.info("Testing _apply_quantitative_analysis...")
    
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
    
    # Test quantitative analysis
    signal = {
        'symbol': 'BTCUSDT',
        'action': 'buy',
        'strength': 0.5,
        'confidence': 0.7
    }
    
    market_data = {
        'symbol': 'BTCUSDT',
        'returns': [0.01, -0.02, 0.03],
        'volatility': 0.02
    }
    
    result_signal = await strategy._apply_quantitative_analysis('BTCUSDT', signal, market_data)
    
    logger.info(f"Result signal: {result_signal}")
    
    assert 'symbol' in result_signal, "Result signal should have 'symbol'"
    assert result_signal['symbol'] == 'BTCUSDT', "Symbol should be BTCUSDT"
    
    logger.info("✅ _apply_quantitative_analysis works correctly")

async def main():
    """Run all tests."""
    logger.info("🚀 Starting advanced errors fix tests...")
    
    try:
        await test_convert_klines_to_dataframe()
        await test_calculate_advanced_indicators()
        await test_get_comprehensive_market_data()
        await test_apply_quantitative_analysis()
        
        logger.info("🎉 All advanced errors fix tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 