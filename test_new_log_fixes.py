#!/usr/bin/env python3
"""
Test script to verify new log error fixes
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
from src.services.binance_service import BinanceService
from src.services.indicator_service import IndicatorService
from src.services.notification_service import NotificationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockBinanceService:
    async def get_klines(self, symbol, interval, limit=100):
        # Mock klines data
        return {
            'symbol': symbol,
            'interval': interval,
            'klines': [
                [datetime.now().timestamp() * 1000, 50000, 51000, 49000, 50500, 1000, 0, 0, 0, 0, 0, 0],
                [datetime.now().timestamp() * 1000, 50500, 52000, 50000, 51500, 1200, 0, 0, 0, 0, 0, 0],
                [datetime.now().timestamp() * 1000, 51500, 53000, 51000, 52500, 1100, 0, 0, 0, 0, 0, 0],
            ]
        }

class MockIndicatorService:
    async def calculate_indicators(self, df):
        return df

class MockNotificationService:
    async def send_notification(self, message):
        pass

class MockCacheService:
    async def get_market_data(self, symbol, data_type):
        # Mock market data
        if data_type == "returns":
            return [0.01, -0.02, 0.03, -0.01, 0.02]
        return None

class MockStatisticalArbitrageEngine:
    """Mock engine that doesn't have the analyze_mean_reversion method"""
    def __init__(self):
        pass

class MockImpliedVolatilityEngine:
    """Mock engine that doesn't have the analyze_volatility_regime method"""
    def __init__(self):
        pass

async def test_momentum_mean_reversion_with_invalid_signal():
    """Test momentum mean reversion analysis with invalid signal parameter"""
    logger.info("Testing momentum mean reversion analysis with invalid signal...")
    
    # Create mock services
    config = {
        'risk_per_trade': 0.02,
        'max_positions': 5,
        'min_confidence': 0.7
    }
    
    strategy = EnhancedTradingStrategyWithQuantitative(
        config=config,
        binance_service=MockBinanceService(),
        indicator_service=MockIndicatorService(),
        notification_service=MockNotificationService(),
        cache_service=MockCacheService()
    )
    
    # Test data
    market_data = {
        'returns': [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.02, 0.03, -0.01, 0.02],
        'volume': [1000, 1200, 1100, 1300, 1400, 1200, 1100, 1300, 1400, 1200, 1100, 1300, 1400, 1200, 1100],
        'current_price': 50000
    }
    
    # Test with invalid signal (string instead of dict)
    invalid_signal = "invalid_signal"
    
    try:
        result = await strategy._apply_momentum_mean_reversion_analysis('BTCUSDT', invalid_signal, market_data)
        logger.info(f"✅ Momentum mean reversion analysis handled invalid signal gracefully")
        logger.info(f"Result: {result}")
        return True
    except Exception as e:
        logger.error(f"❌ Error in momentum mean reversion analysis with invalid signal: {str(e)}")
        return False

async def test_momentum_mean_reversion_with_missing_engine():
    """Test momentum mean reversion analysis with missing statistical arbitrage engine"""
    logger.info("Testing momentum mean reversion analysis with missing engine...")
    
    # Create mock services
    config = {
        'risk_per_trade': 0.02,
        'max_positions': 5,
        'min_confidence': 0.7
    }
    
    strategy = EnhancedTradingStrategyWithQuantitative(
        config=config,
        binance_service=MockBinanceService(),
        indicator_service=MockIndicatorService(),
        notification_service=MockNotificationService(),
        cache_service=MockCacheService()
    )
    
    # Add mock engine without the required method
    strategy.statistical_arbitrage_engine = MockStatisticalArbitrageEngine()
    
    # Test data
    market_data = {
        'returns': [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.02, 0.03, -0.01, 0.02],
        'volume': [1000, 1200, 1100, 1300, 1400, 1200, 1100, 1300, 1400, 1200, 1100, 1300, 1400, 1200, 1100],
        'current_price': 50000
    }
    
    signal = {
        'action': 'buy',
        'confidence': 0.8,
        'strength': 0.6
    }
    
    try:
        result = await strategy._apply_momentum_mean_reversion_analysis('BTCUSDT', signal, market_data)
        logger.info(f"✅ Momentum mean reversion analysis handled missing engine gracefully")
        logger.info(f"Result: {result}")
        return True
    except Exception as e:
        logger.error(f"❌ Error in momentum mean reversion analysis with missing engine: {str(e)}")
        return False

async def test_volatility_regime_with_missing_engine():
    """Test volatility regime analysis with missing implied volatility engine"""
    logger.info("Testing volatility regime analysis with missing engine...")
    
    # Create mock services
    config = {
        'risk_per_trade': 0.02,
        'max_positions': 5,
        'min_confidence': 0.7
    }
    
    strategy = EnhancedTradingStrategyWithQuantitative(
        config=config,
        binance_service=MockBinanceService(),
        indicator_service=MockIndicatorService(),
        notification_service=MockNotificationService(),
        cache_service=MockCacheService()
    )
    
    # Add mock engine without the required method
    strategy.volatility_engine = MockImpliedVolatilityEngine()
    
    # Test data
    market_data = {
        'returns': [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.02, 0.03, -0.01, 0.02],
        'current_price': 50000
    }
    
    signal = {
        'action': 'buy',
        'confidence': 0.8,
        'strength': 0.6
    }
    
    try:
        result = await strategy._apply_volatility_regime_analysis('BTCUSDT', signal, market_data)
        logger.info(f"✅ Volatility regime analysis handled missing engine gracefully")
        logger.info(f"Result: {result}")
        return True
    except Exception as e:
        logger.error(f"❌ Error in volatility regime analysis with missing engine: {str(e)}")
        return False

async def test_normal_operation():
    """Test normal operation without any missing engines"""
    logger.info("Testing normal operation...")
    
    # Create mock services
    config = {
        'risk_per_trade': 0.02,
        'max_positions': 5,
        'min_confidence': 0.7
    }
    
    strategy = EnhancedTradingStrategyWithQuantitative(
        config=config,
        binance_service=MockBinanceService(),
        indicator_service=MockIndicatorService(),
        notification_service=MockNotificationService(),
        cache_service=MockCacheService()
    )
    
    # Test data
    market_data = {
        'returns': [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.02, 0.03, -0.01, 0.02],
        'current_price': 50000
    }
    
    signal = {
        'action': 'buy',
        'confidence': 0.8,
        'strength': 0.6
    }
    
    try:
        # Test momentum mean reversion
        result1 = await strategy._apply_momentum_mean_reversion_analysis('BTCUSDT', signal, market_data)
        logger.info(f"✅ Momentum mean reversion analysis completed successfully")
        
        # Test volatility regime
        result2 = await strategy._apply_volatility_regime_analysis('BTCUSDT', signal, market_data)
        logger.info(f"✅ Volatility regime analysis completed successfully")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error in normal operation: {str(e)}")
        return False

async def main():
    """Main test function"""
    logger.info("Starting new log error fixes test...")
    
    tests = [
        test_momentum_mean_reversion_with_invalid_signal,
        test_momentum_mean_reversion_with_missing_engine,
        test_volatility_regime_with_missing_engine,
        test_normal_operation
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {str(e)}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"TEST SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        logger.info("✅ All tests passed! New log errors have been fixed.")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
