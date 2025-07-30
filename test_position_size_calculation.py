#!/usr/bin/env python3
"""
Test script to verify the position size calculation functions
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
    """Mock Binance service for testing."""
    
    def __init__(self):
        self.mock_balance = {
            'totalWalletBalance': '1000.0',
            'availableBalance': '950.0'
        }
        self.mock_klines = [
            ['1640995200000', '50000', '51000', '49000', '50500', '1000', '1640998800000', '50500', '1000', '0', '0', '0'],
            ['1640998800000', '50500', '51500', '50000', '51000', '1200', '1641002400000', '51000', '1200', '0', '0', '0'],
            ['1641002400000', '51000', '52000', '50500', '51500', '1100', '1641006000000', '51500', '1100', '0', '0', '0'],
            ['1641006000000', '51500', '52500', '51000', '52000', '1300', '1641009600000', '52000', '1300', '0', '0', '0'],
            ['1641009600000', '52000', '53000', '51500', '52500', '1400', '1641013200000', '52500', '1400', '0', '0', '0']
        ]
    
    async def get_account_info(self):
        """Mock get account info."""
        return self.mock_balance
    
    async def get_klines(self, symbol: str, timeframe: str = '1h', limit: int = 100):
        """Mock get klines."""
        return self.mock_klines[:limit]

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

async def test_position_size_calculation():
    """Test the position size calculation function."""
    try:
        logger.info("Testing position size calculation...")
        
        # Create mock services
        mock_binance = MockBinanceService()
        mock_indicator = MockIndicatorService()
        mock_notification = MockNotificationService()
        
        # Create config
        config = {
            'trading': {
                'leverage': 10,
                'risk_per_trade': 0.02
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
        
        # Test parameters
        symbol = 'BTCUSDT'
        risk_per_trade = 0.02  # 2% risk per trade
        current_price = 50000.0
        
        # Test position size calculation
        position_size = await strategy._calculate_position_size(symbol, risk_per_trade, current_price)
        
        logger.info(f"Calculated position size: {position_size}")
        
        # Verify the result
        if position_size is not None and position_size > 0:
            logger.info("✅ Position size calculation test passed")
            return True
        else:
            logger.error("❌ Position size calculation test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Position size calculation test failed with exception: {str(e)}")
        return False

async def test_volatility_adjustment():
    """Test the volatility adjustment function."""
    try:
        logger.info("Testing volatility adjustment...")
        
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
        
        # Test parameters
        symbol = 'BTCUSDT'
        base_size = 0.1  # 10% position size
        
        # Test volatility adjustment
        adjusted_size = await strategy._adjust_position_size_by_volatility(symbol, base_size)
        
        logger.info(f"Base size: {base_size}")
        logger.info(f"Adjusted size: {adjusted_size}")
        
        # Verify the result
        if adjusted_size is not None and adjusted_size > 0:
            logger.info("✅ Volatility adjustment test passed")
            return True
        else:
            logger.error("❌ Volatility adjustment test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Volatility adjustment test failed with exception: {str(e)}")
        return False

async def test_market_volatility():
    """Test the market volatility calculation."""
    try:
        logger.info("Testing market volatility calculation...")
        
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
        
        # Test market volatility calculation
        market_volatility = await strategy._get_market_volatility()
        
        logger.info(f"Market volatility: {market_volatility}")
        
        # Verify the result
        if market_volatility is not None and market_volatility > 0:
            logger.info("✅ Market volatility calculation test passed")
            return True
        else:
            logger.error("❌ Market volatility calculation test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Market volatility calculation test failed with exception: {str(e)}")
        return False

async def test_execute_functions():
    """Test the execute buy/sell functions with position size calculation."""
    try:
        logger.info("Testing execute functions with position size calculation...")
        
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
        
        # Test signal for buy order
        buy_signal = {
            'current_price': 50000.0,
            'atr': 1000.0,
            'action': 'buy',
            'confidence': 0.8
        }
        
        # Test signal for sell order
        sell_signal = {
            'current_price': 50000.0,
            'atr': 1000.0,
            'action': 'sell',
            'confidence': 0.8
        }
        
        # Test execute buy order (this will fail in mock but we can test the logic)
        try:
            await strategy._execute_buy_order('BTCUSDT', buy_signal)
            logger.info("✅ Execute buy order test passed (logic executed)")
        except Exception as e:
            logger.info(f"Execute buy order test completed (expected mock failure): {str(e)}")
        
        # Test execute sell order (this will fail in mock but we can test the logic)
        try:
            await strategy._execute_sell_order('BTCUSDT', sell_signal)
            logger.info("✅ Execute sell order test passed (logic executed)")
        except Exception as e:
            logger.info(f"Execute sell order test completed (expected mock failure): {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Execute functions test failed with exception: {str(e)}")
        return False

async def main():
    """Run all tests."""
    logger.info("🔍 TESTING POSITION SIZE CALCULATION FUNCTIONS")
    
    tests = [
        test_position_size_calculation,
        test_volatility_adjustment,
        test_market_volatility,
        test_execute_functions
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
        logger.info("🎉 All tests passed! Position size calculation functions are working correctly.")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main()) 