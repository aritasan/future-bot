#!/usr/bin/env python3
"""
Test script to verify the updated execute buy/sell functions in enhanced_trading_strategy_with_quantitative.py
"""

import asyncio
import logging
from unittest.mock import Mock, AsyncMock
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
from src.services.binance_service import BinanceService
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
        self.mock_positions = [
            {
                'symbol': 'BTCUSDT',
                'info': {
                    'positionAmt': '0.1',
                    'positionSide': 'LONG'
                }
            }
        ]
        self.place_order_calls = []
    
    async def get_account_info(self):
        """Mock get_account_info."""
        return self.mock_account_info
    
    async def get_positions(self):
        """Mock get_positions."""
        return self.mock_positions
    
    async def place_order(self, order_params):
        """Mock place_order to track calls."""
        self.place_order_calls.append(order_params)
        logger.info(f"Mock place_order called with: {order_params}")
        return {
            'id': 'test_order_id',
            'symbol': order_params.get('symbol'),
            'side': order_params.get('side'),
            'type': order_params.get('type'),
            'amount': order_params.get('amount'),
            'status': 'FILLED'
        }

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

async def test_execute_buy_order():
    """Test the updated _execute_buy_order function."""
    logger.info("Testing _execute_buy_order function...")
    
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
    
    # Test signals
    signals = {
        'optimized_position_size': 0.02,
        'current_price': 50000.0,
        'atr': 1000.0,  # 2% of current price
        'confidence': 0.8,
        'action': 'BUY'
    }
    
    # Execute buy order
    await strategy._execute_buy_order('BTCUSDT', signals)
    
    # Verify place_order was called correctly
    assert len(mock_binance.place_order_calls) == 1, f"Expected 1 place_order call, got {len(mock_binance.place_order_calls)}"
    
    order_params = mock_binance.place_order_calls[0]
    logger.info(f"Order parameters: {order_params}")
    
    # Verify required parameters
    assert order_params['symbol'] == 'BTCUSDT'
    assert order_params['side'] == 'BUY'
    assert order_params['type'] == 'MARKET'
    assert 'amount' in order_params
    assert order_params['amount'] > 0
    
    # Verify SL/TP parameters are included
    assert 'stop_loss' in order_params, "Stop loss should be included in order parameters"
    assert 'take_profit' in order_params, "Take profit should be included in order parameters"
    assert order_params['stop_loss'] > 0, "Stop loss should be positive"
    assert order_params['take_profit'] > 50000, "Take profit should be above current price"
    
    logger.info("✅ _execute_buy_order test passed!")

async def test_execute_sell_order():
    """Test the updated _execute_sell_order function."""
    logger.info("Testing _execute_sell_order function...")
    
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
    
    # Test signals
    signals = {
        'optimized_position_size': 0.02,
        'current_price': 50000.0,
        'atr': 1000.0,  # 2% of current price
        'confidence': 0.8,
        'action': 'SELL'
    }
    
    # Execute sell order
    await strategy._execute_sell_order('BTCUSDT', signals)
    
    # Verify place_order was called correctly
    assert len(mock_binance.place_order_calls) == 1, f"Expected 1 place_order call, got {len(mock_binance.place_order_calls)}"
    
    order_params = mock_binance.place_order_calls[0]
    logger.info(f"Order parameters: {order_params}")
    
    # Verify required parameters
    assert order_params['symbol'] == 'BTCUSDT'
    assert order_params['side'] == 'SELL'
    assert order_params['type'] == 'MARKET'
    assert 'amount' in order_params
    assert order_params['amount'] > 0
    
    # Verify SL/TP parameters are included
    assert 'stop_loss' in order_params, "Stop loss should be included in order parameters"
    assert 'take_profit' in order_params, "Take profit should be included in order parameters"
    assert order_params['stop_loss'] > 50000, "Stop loss should be above current price for SHORT"
    assert order_params['take_profit'] < 50000, "Take profit should be below current price for SHORT"
    
    logger.info("✅ _execute_sell_order test passed!")

async def test_no_position_sell():
    """Test sell order when no position exists."""
    logger.info("Testing sell order with no position...")
    
    # Create mock services with no positions
    mock_binance = MockBinanceService()
    mock_binance.mock_positions = []  # No positions
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
    
    # Test signals
    signals = {
        'optimized_position_size': 0.02,
        'current_price': 50000.0,
        'atr': 1000.0,
        'confidence': 0.8,
        'action': 'SELL'
    }
    
    # Execute sell order (should not place order due to no position)
    await strategy._execute_sell_order('BTCUSDT', signals)
    
    # Verify no order was placed
    assert len(mock_binance.place_order_calls) == 0, "No order should be placed when no position exists"
    
    logger.info("✅ No position sell test passed!")

async def test_error_handling():
    """Test error handling in execute functions."""
    logger.info("Testing error handling...")
    
    # Create mock services that raise exceptions
    mock_binance = MockBinanceService()
    mock_binance.get_account_info = AsyncMock(side_effect=Exception("API Error"))
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
    
    # Test signals
    signals = {
        'optimized_position_size': 0.02,
        'current_price': 50000.0,
        'atr': 1000.0,
        'confidence': 0.8,
        'action': 'BUY'
    }
    
    # Execute buy order (should handle error gracefully)
    try:
        await strategy._execute_buy_order('BTCUSDT', signals)
        logger.info("✅ Error handling test passed - exception was caught and logged")
    except Exception as e:
        logger.error(f"❌ Error handling test failed - exception was not caught: {e}")
        raise
    
    logger.info("✅ Error handling test passed!")

async def main():
    """Run all tests."""
    logger.info("🚀 Starting execute functions fix tests...")
    
    try:
        await test_execute_buy_order()
        await test_execute_sell_order()
        await test_no_position_sell()
        await test_error_handling()
        
        logger.info("🎉 All tests passed! The execute functions are working correctly.")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 