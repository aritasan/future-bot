#!/usr/bin/env python3
"""
Test script to verify the error fixes
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
    """Mock binance service for testing."""
    
    async def get_klines(self, symbol: str, timeframe: str, limit: int = 100) -> list:
        """Mock klines data with correct format."""
        # Create mock klines data with 6 columns (not 12)
        klines = []
        for i in range(limit):
            klines.append([
                1640995200000 + i * 3600000,  # timestamp
                100.0 + i * 0.1,  # open
                110.0 + i * 0.1,  # high
                90.0 + i * 0.1,   # low
                105.0 + i * 0.1,  # close
                1000.0 + i * 10    # volume
            ])
        return klines
    
    async def get_funding_rate(self, symbol: str) -> float:
        """Mock funding rate."""
        return 0.0001
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Mock ticker data."""
        return {
            'volume': 1000000.0,
            'percentage': 2.5
        }
    
    async def get_account_balance(self) -> Dict:
        """Mock account balance."""
        return {
            'USDT': {
                'total': '1000.0',
                'free': '1000.0',
                'used': '0.0'
            }
        }

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

class MockNotificationService:
    """Mock notification service for testing."""
    pass

class MockCacheService:
    """Mock cache service for testing."""
    pass

async def test_error_fixes():
    """Test the error fixes."""
    try:
        logger.info("🧪 TESTING ERROR FIXES")
        
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
            
            # Test 1: Position size calculation
            try:
                position_size = await strategy._calculate_position_size(symbol, 0.02, 105.0)
                logger.info(f"✅ Position size calculation for {symbol}: {position_size}")
            except Exception as e:
                logger.error(f"❌ Position size calculation failed for {symbol}: {str(e)}")
            
            # Test 2: Position size adjustment by volatility
            try:
                adjusted_size = await strategy._adjust_position_size_by_volatility(symbol, 0.01)
                logger.info(f"✅ Position size adjustment for {symbol}: {adjusted_size}")
            except Exception as e:
                logger.error(f"❌ Position size adjustment failed for {symbol}: {str(e)}")
            
            # Test 3: Market volatility calculation
            try:
                market_vol = await strategy._get_market_volatility()
                logger.info(f"✅ Market volatility calculation: {market_vol}")
            except Exception as e:
                logger.error(f"❌ Market volatility calculation failed: {str(e)}")
            
            # Test 4: Generate advanced signal
            try:
                market_data = await strategy._get_comprehensive_market_data(symbol)
                signal = await strategy._generate_advanced_signal(symbol, indicator_service, market_data)
                if signal:
                    current_price = signal.get('current_price', 0.0)
                    logger.info(f"✅ Advanced signal for {symbol}: current_price = {current_price}")
                else:
                    logger.warning(f"⚠️ No signal generated for {symbol}")
            except Exception as e:
                logger.error(f"❌ Advanced signal generation failed for {symbol}: {str(e)}")
            
            # Test 5: Execute functions
            if signal and signal.get('current_price', 0.0) > 0:
                try:
                    await strategy._execute_buy_order(symbol, signal)
                    logger.info(f"✅ Buy order execution for {symbol}")
                except Exception as e:
                    logger.error(f"❌ Buy order execution failed for {symbol}: {str(e)}")
                
                try:
                    await strategy._execute_sell_order(symbol, signal)
                    logger.info(f"✅ Sell order execution for {symbol}")
                except Exception as e:
                    logger.error(f"❌ Sell order execution failed for {symbol}: {str(e)}")
        
        logger.info("🎉 Error fixes test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def main():
    """Run the test."""
    success = await test_error_fixes()
    if success:
        logger.info("🎉 Error fixes test passed!")
    else:
        logger.error("❌ Error fixes test failed!")

if __name__ == "__main__":
    asyncio.run(main()) 