#!/usr/bin/env python3
"""
Test script to verify the latest error fixes:
1. get_recommendations method in QuantitativeTradingSystem
2. Signal type handling in process_trading_signals
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

async def test_get_recommendations_method():
    """Test the get_recommendations method in QuantitativeTradingSystem."""
    try:
        logger.info("Testing get_recommendations method...")
        
        # Initialize quantitative system
        qts = QuantitativeTradingSystem()
        
        # Test get_recommendations method
        recommendations = await qts.get_recommendations('BTCUSDT')
        
        logger.info(f"Recommendations: {recommendations}")
        
        # Check that the method exists and returns expected structure
        assert 'symbol' in recommendations
        assert 'timestamp' in recommendations
        assert 'market_analysis' in recommendations
        assert 'risk_assessment' in recommendations
        assert 'trading_recommendation' in recommendations
        
        logger.info("✅ get_recommendations method test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ get_recommendations method test failed: {str(e)}")
        return False

async def test_signal_type_handling():
    """Test that process_trading_signals handles different signal types correctly."""
    try:
        logger.info("Testing signal type handling...")
        
        # Initialize strategy with mock services
        config = {'test_mode': True}
        binance_service = MockBinanceService()
        indicator_service = MockIndicatorService()
        notification_service = MockNotificationService()
        
        strategy = EnhancedTradingStrategyWithQuantitative(
            config, binance_service, indicator_service, notification_service
        )
        
        # Test with mixed signal types (dict and string)
        signals = {
            'BTCUSDT': {
                'action': 'buy',
                'strength': 0.7,
                'confidence': 0.8,
                'position_size': 0.01
            },
            'ETHUSDT': 'invalid_signal_string',  # This should be skipped
            'ADAUSDT': {
                'action': 'hold',
                'strength': 0.0,
                'confidence': 0.0,
                'position_size': 0.01
            }
        }
        
        # This should not raise an error
        await strategy.process_trading_signals(signals)
        
        logger.info("✅ signal type handling test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ signal type handling test failed: {str(e)}")
        return False

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

async def main():
    """Run all tests."""
    logger.info("Starting latest error fixes verification v3...")
    
    tests = [
        test_get_recommendations_method,
        test_signal_type_handling,
        test_validate_signal_method
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