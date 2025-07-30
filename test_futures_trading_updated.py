#!/usr/bin/env python3
"""
Simple test script to verify the updated futures trading logic
"""

import asyncio
import sys
import os
import logging
from typing import Dict, Optional

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

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
    
    async def get_account_info(self):
        """Mock get account info."""
        return self.mock_balance
    
    async def close_position(self, symbol: str, position_side: str = None) -> bool:
        """Mock close position."""
        logger.info(f"Mock close position: {symbol} {position_side}")
        return True
    
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

async def test_futures_trading_logic():
    """Test the updated futures trading logic."""
    try:
        logger.info("🔍 TESTING UPDATED FUTURES TRADING LOGIC")
        
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
        
        # Import strategy
        from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config=config,
            binance_service=mock_binance,
            indicator_service=mock_indicator,
            notification_service=mock_notification
        )
        
        # Test different trading actions
        test_cases = [
            {
                'name': 'Open LONG Position',
                'signal': {
                    'symbol': 'BTCUSDT',
                    'action': 'buy',
                    'current_price': 50000.0,
                    'atr': 1000.0,
                    'confidence': 0.8
                }
            },
            {
                'name': 'Open SHORT Position',
                'signal': {
                    'symbol': 'BTCUSDT',
                    'action': 'sell',
                    'current_price': 50000.0,
                    'atr': 1000.0,
                    'confidence': 0.7
                }
            },
            {
                'name': 'Close LONG Position',
                'signal': {
                    'symbol': 'BTCUSDT',
                    'action': 'close_long',
                    'current_price': 52000.0,
                    'confidence': 0.6
                }
            },
            {
                'name': 'Close SHORT Position',
                'signal': {
                    'symbol': 'ETHUSDT',
                    'action': 'close_short',
                    'current_price': 3000.0,
                    'confidence': 0.5
                }
            }
        ]
        
        for test_case in test_cases:
            logger.info(f"\n🧪 Testing: {test_case['name']}")
            logger.info(f"Signal: {test_case['signal']}")
            
            try:
                await strategy.process_trading_signals(test_case['signal'])
                logger.info(f"✅ {test_case['name']} - PASSED")
            except Exception as e:
                logger.error(f"❌ {test_case['name']} - FAILED: {str(e)}")
        
        logger.info("\n🎉 All tests completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {str(e)}")
        return False

async def main():
    """Run the test."""
    success = await test_futures_trading_logic()
    if success:
        logger.info("🎉 Futures trading logic test completed successfully!")
    else:
        logger.error("❌ Futures trading logic test failed!")

if __name__ == "__main__":
    asyncio.run(main()) 