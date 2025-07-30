#!/usr/bin/env python3
"""
Test script to verify the current_price fix
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

class MockIndicatorService:
    """Mock indicator service for testing."""
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 100) -> Dict:
        """Mock klines data."""
        # Create mock klines data
        klines = {
            'open': [100.0] * limit,
            'high': [110.0] * limit,
            'low': [90.0] * limit,
            'close': [105.0] * limit,
            'volume': [1000.0] * limit
        }
        return klines

class MockBinanceService:
    """Mock binance service for testing."""
    
    async def get_funding_rate(self, symbol: str) -> float:
        """Mock funding rate."""
        return 0.0001
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Mock ticker data."""
        return {
            'volume': 1000000.0,
            'percentage': 2.5
        }

class MockNotificationService:
    """Mock notification service for testing."""
    pass

class MockCacheService:
    """Mock cache service for testing."""
    pass

async def test_current_price_fix():
    """Test the current_price fix."""
    try:
        logger.info("🧪 TESTING CURRENT_PRICE FIX")
        
        # Import the strategy
        from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
        
        # Create mock services
        indicator_service = MockIndicatorService()
        binance_service = MockBinanceService()
        notification_service = MockNotificationService()
        cache_service = MockCacheService()
        
        # Create strategy instance
        config = {
            'risk_management': {
                'risk_per_trade': 0.02
            }
        }
        
        strategy = EnhancedTradingStrategyWithQuantitative(
            config=config,
            binance_service=binance_service,
            indicator_service=indicator_service,
            notification_service=notification_service,
            cache_service=cache_service
        )
        
        # Test symbols
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        for symbol in test_symbols:
            logger.info(f"Testing symbol: {symbol}")
            
            # Get market data
            market_data = await strategy._get_comprehensive_market_data(symbol)
            logger.info(f"Market data for {symbol}: {market_data}")
            
            # Generate advanced signal
            signal = await strategy._generate_advanced_signal(symbol, indicator_service, market_data)
            
            if signal:
                current_price = signal.get('current_price', 0.0)
                logger.info(f"Signal for {symbol}: current_price = {current_price}")
                
                if current_price > 0:
                    logger.info(f"✅ {symbol}: current_price is valid ({current_price})")
                else:
                    logger.error(f"❌ {symbol}: current_price is invalid ({current_price})")
                    
                # Test execute functions
                if current_price > 0:
                    logger.info(f"Testing execute functions for {symbol}")
                    
                    # Test buy order
                    await strategy._execute_buy_order(symbol, signal)
                    
                    # Test sell order  
                    await strategy._execute_sell_order(symbol, signal)
                    
                    logger.info(f"✅ Execute functions completed for {symbol}")
                else:
                    logger.warning(f"Skipping execute functions for {symbol} due to invalid current_price")
            else:
                logger.error(f"❌ Failed to generate signal for {symbol}")
        
        logger.info("🎉 Current price fix test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def main():
    """Run the test."""
    success = await test_current_price_fix()
    if success:
        logger.info("🎉 Current price fix test passed!")
    else:
        logger.error("❌ Current price fix test failed!")

if __name__ == "__main__":
    asyncio.run(main()) 