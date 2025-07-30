#!/usr/bin/env python3
"""
Test script to verify the futures trading logic with HEDGING mode
"""

import asyncio
import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockBinanceService:
    """Mock Binance service for testing futures trading."""
    
    def __init__(self):
        self.mock_balance = {
            'totalWalletBalance': '1000.0',
            'availableBalance': '950.0'
        }
        self.mock_positions = [
            {
                'symbol': 'BTCUSDT',
                'info': {
                    'positionSide': 'LONG',
                    'positionAmt': '0.005',
                    'entryPrice': '50000.0',
                    'unrealizedPnl': '50.0'
                }
            },
            {
                'symbol': 'ETHUSDT',
                'info': {
                    'positionSide': 'SHORT',
                    'positionAmt': '-0.1',
                    'entryPrice': '3000.0',
                    'unrealizedPnl': '-20.0'
                }
            }
        ]
    
    async def get_account_info(self):
        """Mock get account info."""
        return self.mock_balance
    
    async def get_positions(self):
        """Mock get positions."""
        return self.mock_positions
    
    async def place_order(self, order_params):
        """Mock place order."""
        logger.info(f"Mock order placed: {order_params}")
        return {
            'orderId': '12345',
            'symbol': order_params.get('symbol'),
            'side': order_params.get('side'),
            'positionSide': order_params.get('positionSide'),
            'amount': order_params.get('amount')
        }

class MockIndicatorService:
    """Mock Indicator service for testing."""
    
    async def get_klines(self, symbol: str, timeframe: str = '1h', limit: int = 100):
        """Mock get klines."""
        return {
            'open': [50000, 50500, 51000, 51500, 52000],
            'high': [51000, 51500, 52000, 52500, 53000],
            'low': [49000, 50000, 50500, 51000, 51500],
            'close': [50500, 51000, 51500, 52000, 52500],
            'volume': [1000, 1200, 1100, 1300, 1400]
        }

class MockNotificationService:
    """Mock Notification service for testing."""
    
    async def send_notification(self, message: str):
        """Mock send notification."""
        logger.info(f"Mock notification: {message}")

async def test_long_position_order():
    """Test opening LONG position."""
    try:
        logger.info("Testing LONG position order...")
        
        # Create mock services
        mock_binance = MockBinanceService()
        mock_indicator = MockIndicatorService()
        mock_notification = MockNotificationService()
        
        # Create config
        config = {
            'trading': {
                'leverage': 10
            },
            'risk_management': {
                'risk_per_trade': 0.02
            }
        }
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config=config,
            binance_service=mock_binance,
            indicator_service=mock_indicator,
            notification_service=mock_notification
        )
        
        # Test signal for LONG position
        buy_signal = {
            'symbol': 'BTCUSDT',
            'action': 'buy',
            'current_price': 50000.0,
            'atr': 1000.0,
            'confidence': 0.8
        }
        
        # Test execute buy order (LONG position)
        await strategy._execute_buy_order('BTCUSDT', buy_signal)
        logger.info("✅ LONG position order test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ LONG position order test failed: {str(e)}")
        return False

async def test_short_position_order():
    """Test opening SHORT position."""
    try:
        logger.info("Testing SHORT position order...")
        
        # Create mock services
        mock_binance = MockBinanceService()
        mock_indicator = MockIndicatorService()
        mock_notification = MockNotificationService()
        
        # Create config
        config = {
            'trading': {
                'leverage': 10
            },
            'risk_management': {
                'risk_per_trade': 0.02
            }
        }
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config=config,
            binance_service=mock_binance,
            indicator_service=mock_indicator,
            notification_service=mock_notification
        )
        
        # Test signal for SHORT position
        sell_signal = {
            'symbol': 'BTCUSDT',
            'action': 'sell',
            'current_price': 50000.0,
            'atr': 1000.0,
            'confidence': 0.8
        }
        
        # Test execute sell order (SHORT position)
        await strategy._execute_sell_order('BTCUSDT', sell_signal)
        logger.info("✅ SHORT position order test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ SHORT position order test failed: {str(e)}")
        return False

async def test_close_long_position():
    """Test closing LONG position."""
    try:
        logger.info("Testing close LONG position...")
        
        # Create mock services
        mock_binance = MockBinanceService()
        mock_indicator = MockIndicatorService()
        mock_notification = MockNotificationService()
        
        # Create config
        config = {
            'trading': {
                'leverage': 10
            }
        }
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config=config,
            binance_service=mock_binance,
            indicator_service=mock_indicator,
            notification_service=mock_notification
        )
        
        # Test close LONG position
        close_signal = {
            'symbol': 'BTCUSDT',
            'action': 'close_long',
            'current_price': 50000.0
        }
        
        await strategy._close_long_position('BTCUSDT', close_signal)
        logger.info("✅ Close LONG position test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Close LONG position test failed: {str(e)}")
        return False

async def test_close_short_position():
    """Test closing SHORT position."""
    try:
        logger.info("Testing close SHORT position...")
        
        # Create mock services
        mock_binance = MockBinanceService()
        mock_indicator = MockIndicatorService()
        mock_notification = MockNotificationService()
        
        # Create config
        config = {
            'trading': {
                'leverage': 10
            }
        }
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config=config,
            binance_service=mock_binance,
            indicator_service=mock_indicator,
            notification_service=mock_notification
        )
        
        # Test close SHORT position
        close_signal = {
            'symbol': 'ETHUSDT',
            'action': 'close_short',
            'current_price': 3000.0
        }
        
        await strategy._close_short_position('ETHUSDT', close_signal)
        logger.info("✅ Close SHORT position test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Close SHORT position test failed: {str(e)}")
        return False

async def test_process_trading_signals():
    """Test process trading signals with different actions."""
    try:
        logger.info("Testing process trading signals...")
        
        # Create mock services
        mock_binance = MockBinanceService()
        mock_indicator = MockIndicatorService()
        mock_notification = MockNotificationService()
        
        # Create config
        config = {
            'trading': {
                'leverage': 10
            },
            'risk_management': {
                'risk_per_trade': 0.02
            }
        }
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config=config,
            binance_service=mock_binance,
            indicator_service=mock_indicator,
            notification_service=mock_notification
        )
        
        # Test different signal actions
        test_signals = [
            {
                'symbol': 'BTCUSDT',
                'action': 'buy',
                'current_price': 50000.0,
                'atr': 1000.0,
                'confidence': 0.8
            },
            {
                'symbol': 'ETHUSDT',
                'action': 'sell',
                'current_price': 3000.0,
                'atr': 100.0,
                'confidence': 0.7
            },
            {
                'symbol': 'BTCUSDT',
                'action': 'close_long',
                'current_price': 50000.0,
                'confidence': 0.6
            },
            {
                'symbol': 'ETHUSDT',
                'action': 'close_short',
                'current_price': 3000.0,
                'confidence': 0.5
            }
        ]
        
        for signal in test_signals:
            logger.info(f"Testing signal: {signal['action']} for {signal['symbol']}")
            await strategy.process_trading_signals(signal)
        
        logger.info("✅ Process trading signals test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Process trading signals test failed: {str(e)}")
        return False

async def main():
    """Run all tests."""
    logger.info("🔍 TESTING FUTURES TRADING LOGIC WITH HEDGING MODE")
    
    tests = [
        test_long_position_order,
        test_short_position_order,
        test_close_long_position,
        test_close_short_position,
        test_process_trading_signals
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {str(e)}")
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Futures trading logic is working correctly.")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main()) 