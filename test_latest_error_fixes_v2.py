#!/usr/bin/env python3
"""
Test script to verify the latest error fixes:
1. validate_signal method in QuantitativeTradingSystem
2. position_size key in signals
3. funding_rate and ticker data handling
"""

import asyncio
import sys
import os
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.quantitative.quantitative_trading_system import QuantitativeTradingSystem
from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockBinanceService:
    """Mock Binance service for testing."""
    
    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Mock funding rate."""
        return 0.0001  # Return a float, not a dict
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Mock ticker data."""
        return {
            'volume': 1000000.0,
            'percentage': 2.5,
            'last': 50000.0
        }

class MockIndicatorService:
    """Mock indicator service for testing."""
    
    async def get_klines(self, symbol: str, timeframe: str, limit: int = 100) -> Dict:
        """Mock klines data."""
        return {
            'open': [100.0] * limit,
            'high': [110.0] * limit,
            'low': [90.0] * limit,
            'close': [105.0] * limit,
            'volume': [1000.0] * limit
        }

class MockNotificationService:
    """Mock notification service for testing."""
    
    async def send_notification(self, message: str) -> None:
        """Mock notification."""
        pass

async def test_validate_signal_method():
    """Test the validate_signal method in QuantitativeTradingSystem."""
    try:
        logger.info("Testing validate_signal method...")
        
        # Initialize quantitative system
        qts = QuantitativeTradingSystem()
        
        # Create test signal and market data
        signal_data = {
            'strength': 0.7,
            'confidence': 0.8,
            'position_size': 0.01,
            'action': 'buy'
        }
        
        market_data = {
            'returns': [0.01, -0.005, 0.02, -0.01, 0.015],
            'volatility': 0.02
        }
        
        # Test validate_signal method (now async)
        validation_result = await qts.validate_signal(signal_data, market_data)
        
        logger.info(f"Validation result: {validation_result}")
        
        # Check that the method exists and returns expected structure
        assert 'is_valid' in validation_result
        assert 'quantitative_validation' in validation_result
        assert 'signal_strength' in validation_result['quantitative_validation']
        
        logger.info("✅ validate_signal method test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ validate_signal method test failed: {str(e)}")
        return False

async def test_position_size_in_signals():
    """Test that signals have position_size key."""
    try:
        logger.info("Testing position_size in signals...")
        
        # Initialize strategy with mock services
        config = {'test_mode': True}
        binance_service = MockBinanceService()
        indicator_service = MockIndicatorService()
        notification_service = MockNotificationService()
        
        strategy = EnhancedTradingStrategyWithQuantitative(
            config, binance_service, indicator_service, notification_service
        )
        
        # Test _combine_timeframe_signals method
        timeframes = {
            '1h': {'signal': 'buy', 'strength': 0.5, 'confidence': 0.6, 'reasons': ['test']},
            '4h': {'signal': 'buy', 'strength': 0.4, 'confidence': 0.5, 'reasons': ['test']},
            '1d': {'signal': 'buy', 'strength': 0.6, 'confidence': 0.7, 'reasons': ['test']}
        }
        
        combined_signal = strategy._combine_timeframe_signals(timeframes)
        
        logger.info(f"Combined signal: {combined_signal}")
        
        # Check that position_size is present
        assert 'position_size' in combined_signal
        assert combined_signal['position_size'] == 0.01
        
        # Test _create_base_signal method
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'open': [99.0, 100.0, 101.0],
            'high': [103.0, 104.0, 105.0],
            'low': [98.0, 99.0, 100.0],
            'volume': [1000.0, 1100.0, 1200.0]
        })
        
        base_signal = strategy._create_base_signal('BTCUSDT', df, {'test': True})
        
        logger.info(f"Base signal: {base_signal}")
        
        # Check that position_size is present
        assert 'position_size' in base_signal
        assert base_signal['position_size'] == 0.01
        
        logger.info("✅ position_size in signals test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ position_size in signals test failed: {str(e)}")
        return False

async def test_funding_rate_and_ticker_handling():
    """Test the fixed funding rate and ticker handling."""
    try:
        logger.info("Testing funding rate and ticker handling...")
        
        # Initialize strategy with mock services
        config = {'test_mode': True}
        binance_service = MockBinanceService()
        indicator_service = MockIndicatorService()
        notification_service = MockNotificationService()
        
        strategy = EnhancedTradingStrategyWithQuantitative(
            config, binance_service, indicator_service, notification_service
        )
        
        # Test _get_comprehensive_market_data method
        market_data = await strategy._get_comprehensive_market_data('BTCUSDT')
        
        logger.info(f"Market data: {market_data}")
        
        # Check that funding_rate and ticker data are properly handled
        assert 'funding_rate' in market_data
        assert 'volume_24h' in market_data
        assert 'price_change_24h' in market_data
        
        logger.info("✅ funding rate and ticker handling test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ funding rate and ticker handling test failed: {str(e)}")
        return False

async def test_risk_management_methods():
    """Test that risk management methods work with position_size."""
    try:
        logger.info("Testing risk management methods...")
        
        # Initialize strategy with mock services
        config = {'test_mode': True}
        binance_service = MockBinanceService()
        indicator_service = MockIndicatorService()
        notification_service = MockNotificationService()
        
        strategy = EnhancedTradingStrategyWithQuantitative(
            config, binance_service, indicator_service, notification_service
        )
        
        # Create test signal with position_size
        signal = {
            'action': 'buy',
            'strength': 0.7,
            'confidence': 0.8,
            'position_size': 0.01,
            'reasons': ['test']
        }
        
        market_data = {
            'returns': [0.01, -0.005, 0.02, -0.01, 0.015],
            'volatility': 0.02
        }
        
        # Test _apply_advanced_risk_management
        risk_adjusted_signal = await strategy._apply_advanced_risk_management('BTCUSDT', signal, market_data)
        
        logger.info(f"Risk adjusted signal: {risk_adjusted_signal}")
        
        # Check that position_size is still present and modified
        assert 'position_size' in risk_adjusted_signal
        
        # Test _apply_volatility_regime_analysis
        volatility_signal = await strategy._apply_volatility_regime_analysis('BTCUSDT', signal, market_data)
        
        logger.info(f"Volatility signal: {volatility_signal}")
        
        # Check that position_size is still present and modified
        assert 'position_size' in volatility_signal
        
        logger.info("✅ risk management methods test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ risk management methods test failed: {str(e)}")
        return False

async def main():
    """Run all tests."""
    logger.info("Starting latest error fixes verification...")
    
    tests = [
        test_validate_signal_method,
        test_position_size_in_signals,
        test_funding_rate_and_ticker_handling,
        test_risk_management_methods
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
        logger.info("🎉 All tests passed! Latest error fixes are working correctly.")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main()) 