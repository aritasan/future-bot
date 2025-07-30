#!/usr/bin/env python3
"""
Test script to verify the discord service fix
"""

import asyncio
import sys
from unittest.mock import Mock, AsyncMock

# Mock the necessary imports
sys.path.append('.')

class MockTelegramService:
    def __init__(self):
        self._paused = False
    
    def is_trading_paused(self):
        return self._paused
    
    async def pause_trading(self):
        self._paused = True
    
    async def wait_for_trading_resume(self):
        while self._paused:
            await asyncio.sleep(0.1)

class MockDiscordService:
    def __init__(self):
        self._paused = False
    
    def is_trading_paused(self):
        return self._paused
    
    async def pause_trading(self):
        self._paused = True
    
    async def wait_for_trading_resume(self):
        while self._paused:
            await asyncio.sleep(0.1)

class MockBinanceService:
    async def get_account_info(self):
        return {'totalWalletBalance': '1000.0'}

class MockHealthMonitor:
    async def check_health(self):
        return True

class MockIndicatorService:
    async def get_klines(self, symbol, interval, limit=100):
        return {
            'close': [50000.0 + i * 100 for i in range(limit)],
            'high': [50050.0 + i * 100 for i in range(limit)],
            'low': [49950.0 + i * 100 for i in range(limit)],
            'volume': [1000 + i * 10 for i in range(limit)]
        }

class MockCacheService:
    async def get_market_data(self, symbol, timeframe):
        return None
    
    async def cache_market_data(self, symbol, timeframe, data, ttl):
        pass
    
    async def cache_analysis(self, symbol, analysis_type, data, ttl):
        pass

class MockStrategy:
    async def check_profit_target(self):
        return False
    
    async def generate_signals(self, symbol, indicator_service):
        return {
            'action': 'hold',
            'quantitative_confidence': 0.5,
            'optimized_position_size': 0.01
        }
    
    async def process_trading_signals(self, signals):
        pass
    
    async def get_quantitative_recommendations(self, symbol):
        return {'trading_recommendation': {'action': 'hold', 'confidence': 0.5}}

async def test_discord_service_none():
    """Test that the function handles None discord_service correctly"""
    print("🧪 Testing Discord Service None handling...")
    
    # Create mock services
    binance_service = MockBinanceService()
    telegram_service = MockTelegramService()
    discord_service = None  # This is the key test - None discord service
    health_monitor = MockHealthMonitor()
    strategy = MockStrategy()
    indicator_service = MockIndicatorService()
    cache_service = MockCacheService()
    
    # Set test mode
    import main_with_quantitative
    main_with_quantitative.test_mode = True
    
    # Import the function
    from main_with_quantitative import process_symbol_with_quantitative
    
    try:
        # This should not raise an error even with None discord_service
        await asyncio.wait_for(
            process_symbol_with_quantitative(
                'BTCUSDT',
                binance_service,
                telegram_service,
                discord_service,
                health_monitor,
                strategy,
                indicator_service,
                cache_service
            ),
            timeout=5.0  # 5 seconds timeout for test
        )
        print("✅ Test passed: Function handles None discord_service correctly")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

async def test_telegram_service_none():
    """Test that the function handles None telegram_service correctly"""
    print("\n🧪 Testing Telegram Service None handling...")
    
    # Create mock services
    binance_service = MockBinanceService()
    telegram_service = None  # This is the key test - None telegram service
    discord_service = MockDiscordService()
    health_monitor = MockHealthMonitor()
    strategy = MockStrategy()
    indicator_service = MockIndicatorService()
    cache_service = MockCacheService()
    
    # Set test mode
    import main_with_quantitative
    main_with_quantitative.test_mode = True
    
    # Import the function
    from main_with_quantitative import process_symbol_with_quantitative
    
    try:
        # This should not raise an error even with None telegram_service
        await asyncio.wait_for(
            process_symbol_with_quantitative(
                'BTCUSDT',
                binance_service,
                telegram_service,
                discord_service,
                health_monitor,
                strategy,
                indicator_service,
                cache_service
            ),
            timeout=5.0  # 5 seconds timeout for test
        )
        print("✅ Test passed: Function handles None telegram_service correctly")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

async def test_both_services_none():
    """Test that the function handles both services being None correctly"""
    print("\n🧪 Testing Both Services None handling...")
    
    # Create mock services
    binance_service = MockBinanceService()
    telegram_service = None
    discord_service = None
    health_monitor = MockHealthMonitor()
    strategy = MockStrategy()
    indicator_service = MockIndicatorService()
    cache_service = MockCacheService()
    
    # Set test mode
    import main_with_quantitative
    main_with_quantitative.test_mode = True
    
    # Import the function
    from main_with_quantitative import process_symbol_with_quantitative
    
    try:
        # This should not raise an error even with both services being None
        await asyncio.wait_for(
            process_symbol_with_quantitative(
                'BTCUSDT',
                binance_service,
                telegram_service,
                discord_service,
                health_monitor,
                strategy,
                indicator_service,
                cache_service
            ),
            timeout=5.0  # 5 seconds timeout for test
        )
        print("✅ Test passed: Function handles both services being None correctly")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

async def test_notification_function():
    """Test the send_quantitative_notification function with None services"""
    print("\n🧪 Testing Notification Function...")
    
    # Import the function
    from main_with_quantitative import send_quantitative_notification
    
    # Test data
    symbol = 'BTCUSDT'
    signals = {
        'quantitative_confidence': 0.8,
        'action': 'buy',
        'optimized_position_size': 0.01,
        'quantitative_validation': {'is_valid': True, 'sharpe_ratio': 1.5},
        'var_estimate': 0.02,
        'market_efficiency': 0.7
    }
    recommendations = {
        'trading_recommendation': {
            'action': 'buy',
            'confidence': 0.8,
            'reasoning': ['Strong momentum', 'Good risk/reward']
        }
    }
    
    # Test with None services
    try:
        await send_quantitative_notification(
            symbol, signals, recommendations, None, None
        )
        print("✅ Test passed: Notification function handles None services correctly")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

async def main():
    """Run all tests"""
    print("🧪 Testing Discord Service Fix...")
    print("=" * 50)
    
    tests = [
        test_discord_service_none,
        test_telegram_service_none,
        test_both_services_none,
        test_notification_function
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if await test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Discord service fix is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please review the fix.")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 