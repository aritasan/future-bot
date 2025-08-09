#!/usr/bin/env python3
"""
Test script to verify the momentum mean reversion analysis error fix.
"""

import asyncio
import logging
import sys
import os
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_momentum_error_fix():
    """Test the momentum mean reversion analysis error fix."""
    logger.info("Testing momentum mean reversion analysis error fix...")
    
    try:
        # Import the strategy class
        from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
        from src.services.indicator_service import IndicatorService
        from src.services.binance_service import BinanceService
        from src.services.notification_service import NotificationService
        
        # Mock configuration
        config = {
            'trading': {
                'statistical_significance_level': 0.05,
                'min_sample_size': 100
            },
            'risk_management': {
                'max_position_size': 0.1,
                'stop_loss_percentage': 0.02
            }
        }
        
        # Mock services
        mock_binance_service = None
        mock_indicator_service = None
        mock_notification_service = None
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config, mock_binance_service, mock_indicator_service, mock_notification_service
        )
        
        # Test data
        symbol = 'BTCUSDT'
        
        # Test case 1: Valid signal with complete market data
        valid_signal = {
            'action': 'buy',
            'confidence': 0.8,
            'strength': 0.7,
            'timestamp': 1234567890
        }
        
        complete_market_data = {
            'returns': [0.01, 0.02, -0.01, 0.03, 0.01, 0.02, -0.01, 0.03, 0.01, 0.02, -0.01, 0.03, 0.01, 0.02, -0.01, 0.03, 0.01, 0.02, -0.01, 0.03],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900],
            'bollinger_bands': {
                'upper': 105.0,
                'lower': 95.0
            },
            'rsi': 65.0,
            'current_price': 100.0
        }
        
        logger.info("Test 1: Valid signal with complete market data")
        try:
            result1 = await strategy._apply_momentum_mean_reversion_analysis(symbol, valid_signal, complete_market_data)
            logger.info(f"✅ Test 1 passed: {result1.get('action', 'unknown')}")
        except Exception as e:
            logger.error(f"❌ Test 1 failed: {str(e)}")
            return False
        
        # Test case 2: Signal with missing mean reversion data
        incomplete_market_data = {
            'returns': [0.01, 0.02, -0.01, 0.03, 0.01, 0.02, -0.01, 0.03, 0.01, 0.02, -0.01, 0.03, 0.01, 0.02, -0.01, 0.03, 0.01, 0.02, -0.01, 0.03],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900],
            'rsi': 65.0,
            'current_price': 100.0
            # Missing bollinger_bands
        }
        
        logger.info("Test 2: Signal with missing mean reversion data")
        try:
            result2 = await strategy._apply_momentum_mean_reversion_analysis(symbol, valid_signal, incomplete_market_data)
            logger.info(f"✅ Test 2 passed: {result2.get('action', 'unknown')}")
        except Exception as e:
            logger.error(f"❌ Test 2 failed: {str(e)}")
            return False
        
        # Test case 3: Invalid signal parameter
        invalid_signal = "not_a_dict"
        
        logger.info("Test 3: Invalid signal parameter")
        try:
            result3 = await strategy._apply_momentum_mean_reversion_analysis(symbol, invalid_signal, complete_market_data)
            logger.info(f"✅ Test 3 passed: {result3.get('action', 'unknown')}")
        except Exception as e:
            logger.error(f"❌ Test 3 failed: {str(e)}")
            return False
        
        # Test case 4: Empty market data
        empty_market_data = {}
        
        logger.info("Test 4: Empty market data")
        try:
            result4 = await strategy._apply_momentum_mean_reversion_analysis(symbol, valid_signal, empty_market_data)
            logger.info(f"✅ Test 4 passed: {result4.get('action', 'unknown')}")
        except Exception as e:
            logger.error(f"❌ Test 4 failed: {str(e)}")
            return False
        
        # Test case 5: Market data with insufficient returns
        insufficient_market_data = {
            'returns': [0.01, 0.02, -0.01],  # Only 3 data points
            'volume': [1000, 1100, 1200],
            'current_price': 100.0
        }
        
        logger.info("Test 5: Market data with insufficient returns")
        try:
            result5 = await strategy._apply_momentum_mean_reversion_analysis(symbol, valid_signal, insufficient_market_data)
            logger.info(f"✅ Test 5 passed: {result5.get('action', 'unknown')}")
        except Exception as e:
            logger.error(f"❌ Test 5 failed: {str(e)}")
            return False
        
        logger.info("\n" + "="*60)
        logger.info("ALL TESTS PASSED! ✅")
        logger.info("The momentum mean reversion analysis error has been fixed.")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        return False

async def main():
    """Main function to run the test."""
    try:
        logger.info("Starting momentum error fix test...")
        
        success = await test_momentum_error_fix()
        
        if success:
            logger.info("✅ All tests completed successfully!")
            logger.info("The momentum mean reversion analysis error has been fixed.")
        else:
            logger.error("❌ Some tests failed. Please check the errors above.")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in main test: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(main())
