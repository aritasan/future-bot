#!/usr/bin/env python3
"""
Test script to verify the ML ensemble missing columns error fix.
"""

import asyncio
import logging
import sys
import os
from typing import Dict, Any
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_ml_columns_error_fix():
    """Test the ML ensemble missing columns error fix."""
    logger.info("Testing ML ensemble missing columns error fix...")
    
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
        
        # Test case 1: Complete market data
        complete_market_data = {
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'open': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }
        
        valid_signal = {
            'action': 'buy',
            'confidence': 0.8,
            'strength': 0.7,
            'timestamp': 1234567890
        }
        
        logger.info("Test 1: Complete market data")
        try:
            result1 = await strategy._apply_advanced_ml_analysis(symbol, valid_signal, complete_market_data)
            logger.info(f"✅ Test 1 passed: {result1.get('action', 'unknown')}")
        except Exception as e:
            logger.error(f"❌ Test 1 failed: {str(e)}")
            return False
        
        # Test case 2: Missing columns market data
        incomplete_market_data = {
            'price': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
            # Missing close, high, low, open
        }
        
        logger.info("Test 2: Missing columns market data")
        try:
            result2 = await strategy._apply_advanced_ml_analysis(symbol, valid_signal, incomplete_market_data)
            logger.info(f"✅ Test 2 passed: {result2.get('action', 'unknown')}")
        except Exception as e:
            logger.error(f"❌ Test 2 failed: {str(e)}")
            return False
        
        # Test case 3: Empty market data
        empty_market_data = {}
        
        logger.info("Test 3: Empty market data")
        try:
            result3 = await strategy._apply_advanced_ml_analysis(symbol, valid_signal, empty_market_data)
            logger.info(f"✅ Test 3 passed: {result3.get('action', 'unknown')}")
        except Exception as e:
            logger.error(f"❌ Test 3 failed: {str(e)}")
            return False
        
        # Test case 4: Market data with only some columns
        partial_market_data = {
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
            # Missing high, low, open
        }
        
        logger.info("Test 4: Partial market data")
        try:
            result4 = await strategy._apply_advanced_ml_analysis(symbol, valid_signal, partial_market_data)
            logger.info(f"✅ Test 4 passed: {result4.get('action', 'unknown')}")
        except Exception as e:
            logger.error(f"❌ Test 4 failed: {str(e)}")
            return False
        
        # Test case 5: Market data with different column names
        different_columns_market_data = {
            'price': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'max_price': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'min_price': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'start_price': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }
        
        logger.info("Test 5: Different column names")
        try:
            result5 = await strategy._apply_advanced_ml_analysis(symbol, valid_signal, different_columns_market_data)
            logger.info(f"✅ Test 5 passed: {result5.get('action', 'unknown')}")
        except Exception as e:
            logger.error(f"❌ Test 5 failed: {str(e)}")
            return False
        
        logger.info("\n" + "="*60)
        logger.info("ALL TESTS PASSED! ✅")
        logger.info("The ML ensemble missing columns error has been fixed.")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        return False

async def main():
    """Main function to run the test."""
    try:
        logger.info("Starting ML columns error fix test...")
        
        success = await test_ml_columns_error_fix()
        
        if success:
            logger.info("✅ All tests completed successfully!")
            logger.info("The ML ensemble missing columns error has been fixed.")
        else:
            logger.error("❌ Some tests failed. Please check the errors above.")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in main test: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(main())

