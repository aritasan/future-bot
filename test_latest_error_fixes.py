#!/usr/bin/env python3
"""
Test script to verify the latest error fixes in the trading bot.
"""

import asyncio
import logging
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from typing import Dict, Optional, List
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockBinanceService:
    """Mock Binance service for testing."""
    
    def __init__(self):
        self._is_initialized = True
    
    async def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        """Mock funding rate."""
        return {'fundingRate': 0.0001}
    
    async def get_24h_ticker(self, symbol: str) -> Optional[Dict]:
        """Mock 24h ticker."""
        return {
            'volume': '1000000',
            'priceChangePercent': '2.5'
        }

class MockIndicatorService:
    """Mock Indicator service for testing."""
    
    def __init__(self):
        self._is_initialized = True
        self._is_closed = False
    
    async def get_klines(self, symbol: str, timeframe: str = '5m', limit: int = 100) -> Optional[Dict]:
        """Mock klines data."""
        # Return mock klines data
        return {
            'open': [50000, 50100, 50200, 50300, 50400],
            'high': [50100, 50200, 50300, 50400, 50500],
            'low': [49900, 50000, 50100, 50200, 50300],
            'close': [50100, 50200, 50300, 50400, 50500],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }

class MockNotificationService:
    """Mock Notification service for testing."""
    
    def __init__(self):
        pass

async def test_market_conditions_fix():
    """Test that _get_market_conditions works correctly with the latest fix."""
    logger.info("Testing _get_market_conditions fix...")
    
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
    
    # Import and create strategy instance
    from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
    
    strategy = EnhancedTradingStrategyWithQuantitative(
        config=config,
        binance_service=mock_binance,
        indicator_service=mock_indicator,
        notification_service=mock_notification
    )
    
    try:
        # Test _get_market_conditions
        market_conditions = await strategy._get_market_conditions('BTCUSDT')
        
        logger.info(f"Market conditions: {market_conditions}")
        
        # Verify the result
        assert isinstance(market_conditions, dict), "Market conditions should be a dictionary"
        assert 'volatility' in market_conditions, "Market conditions should have volatility"
        assert 'price_change_24h' in market_conditions, "Market conditions should have price_change_24h"
        assert isinstance(market_conditions['volatility'], float), "Volatility should be a float"
        assert isinstance(market_conditions['price_change_24h'], float), "Price change should be a float"
        
        logger.info("✅ _get_market_conditions fix works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ _get_market_conditions fix failed: {e}")
        return False

async def test_comprehensive_market_data_fix():
    """Test that _get_comprehensive_market_data works correctly with the latest fix."""
    logger.info("Testing _get_comprehensive_market_data fix...")
    
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
    
    # Import and create strategy instance
    from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
    
    strategy = EnhancedTradingStrategyWithQuantitative(
        config=config,
        binance_service=mock_binance,
        indicator_service=mock_indicator,
        notification_service=mock_notification
    )
    
    try:
        # Test _get_comprehensive_market_data
        market_data = await strategy._get_comprehensive_market_data('BTCUSDT')
        
        logger.info(f"Comprehensive market data: {market_data}")
        
        # Verify the result
        assert isinstance(market_data, dict), "Market data should be a dictionary"
        assert 'symbol' in market_data, "Market data should have symbol"
        assert 'returns' in market_data, "Market data should have returns"
        assert 'volatility' in market_data, "Market data should have volatility"
        assert market_data['symbol'] == 'BTCUSDT', "Symbol should be BTCUSDT"
        assert isinstance(market_data['returns'], list), "Returns should be a list"
        assert isinstance(market_data['volatility'], float), "Volatility should be a float"
        
        logger.info("✅ _get_comprehensive_market_data fix works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ _get_comprehensive_market_data fix failed: {e}")
        return False

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
    
    # Import and create strategy instance
    from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
    
    strategy = EnhancedTradingStrategyWithQuantitative(
        config=config,
        binance_service=mock_binance,
        indicator_service=mock_indicator,
        notification_service=mock_notification
    )
    
    try:
        # Test data
        klines = {
            'open': [50000, 50100, 50200],
            'high': [50100, 50200, 50300],
            'low': [49900, 50000, 50100],
            'close': [50100, 50200, 50300],
            'volume': [1000, 1100, 1200]
        }
        
        # Test conversion
        df = strategy._convert_klines_to_dataframe(klines)
        
        logger.info(f"Converted DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        
        # Verify the result
        assert isinstance(df, pd.DataFrame), "Result should be a pandas DataFrame"
        assert len(df) == 3, "DataFrame should have 3 rows"
        assert 'open' in df.columns, "DataFrame should have 'open' column"
        assert 'high' in df.columns, "DataFrame should have 'high' column"
        assert 'low' in df.columns, "DataFrame should have 'low' column"
        assert 'close' in df.columns, "DataFrame should have 'close' column"
        assert 'volume' in df.columns, "DataFrame should have 'volume' column"
        
        logger.info("✅ _convert_klines_to_dataframe works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ _convert_klines_to_dataframe failed: {e}")
        return False

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
    
    # Import and create strategy instance
    from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
    
    strategy = EnhancedTradingStrategyWithQuantitative(
        config=config,
        binance_service=mock_binance,
        indicator_service=mock_indicator,
        notification_service=mock_notification
    )
    
    try:
        # Test data
        signal = {
            'action': 'buy',
            'strength': 0.7,
            'confidence': 0.8
        }
        
        market_data = {
            'symbol': 'BTCUSDT',
            'returns': [0.01, -0.02, 0.03],
            'volatility': 0.25
        }
        
        # Test quantitative analysis
        result = await strategy._apply_quantitative_analysis('BTCUSDT', signal, market_data)
        
        logger.info(f"Quantitative analysis result: {result}")
        
        # Verify the result
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'action' in result, "Result should have action"
        assert 'strength' in result, "Result should have strength"
        assert 'confidence' in result, "Result should have confidence"
        
        logger.info("✅ _apply_quantitative_analysis works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ _apply_quantitative_analysis failed: {e}")
        return False

async def test_edge_cases():
    """Test edge cases for the fixes."""
    logger.info("Testing edge cases...")
    
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
    
    # Import and create strategy instance
    from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
    
    strategy = EnhancedTradingStrategyWithQuantitative(
        config=config,
        binance_service=mock_binance,
        indicator_service=mock_indicator,
        notification_service=mock_notification
    )
    
    try:
        # Test with None klines
        market_conditions = await strategy._get_market_conditions('INVALID')
        assert isinstance(market_conditions, dict), "Should return default dict for invalid symbol"
        assert 'volatility' in market_conditions, "Should have default volatility"
        assert 'price_change_24h' in market_conditions, "Should have default price_change_24h"
        
        # Test with empty klines
        market_data = await strategy._get_comprehensive_market_data('INVALID')
        assert isinstance(market_data, dict), "Should return default dict for invalid symbol"
        assert 'symbol' in market_data, "Should have symbol"
        assert 'returns' in market_data, "Should have returns"
        assert 'volatility' in market_data, "Should have volatility"
        
        logger.info("✅ Edge cases work correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Edge cases failed: {e}")
        return False

async def main():
    """Run all tests."""
    logger.info("🧪 Starting latest error fixes verification...")
    
    tests = [
        test_market_conditions_fix,
        test_comprehensive_market_data_fix,
        test_convert_klines_to_dataframe,
        test_apply_quantitative_analysis,
        test_edge_cases
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
        except Exception as e:
            logger.error(f"❌ Test {test.__name__} failed with exception: {e}")
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Latest error fixes are working correctly.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 