#!/usr/bin/env python3
"""
Comprehensive test script to verify all error fixes
"""

import asyncio
import sys
import pandas as pd
import numpy as np
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
    
    async def analyze_portfolio_optimization(self, symbols):
        return {'optimization_success': True}
    
    async def analyze_factor_exposures(self, symbols):
        return {'factor_analysis': True}
    
    async def get_performance_metrics(self):
        return {'total_analyses': 0}

async def test_discord_service_fix():
    """Test Discord service None handling"""
    print("🧪 Testing Discord Service Fix...")
    
    # Create mock services
    binance_service = MockBinanceService()
    telegram_service = MockTelegramService()
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
            timeout=5.0
        )
        print("✅ Discord service fix test passed")
        return True
    except Exception as e:
        print(f"❌ Discord service fix test failed: {str(e)}")
        return False

async def test_portfolio_optimizer_fix():
    """Test portfolio optimizer with list data"""
    print("\n🧪 Testing Portfolio Optimizer Fix...")
    
    try:
        from src.quantitative.portfolio_optimizer import PortfolioOptimizer
        
        optimizer = PortfolioOptimizer()
        
        # Test with list data (should return error)
        list_data = [0.01, -0.02, 0.03]
        result = optimizer.optimize_portfolio(list_data, method='markowitz')
        
        if 'error' in result and 'list' in result['error']:
            print("✅ Portfolio optimizer correctly handles list data")
            return True
        else:
            print("❌ Portfolio optimizer should have returned error for list data")
            return False
            
    except Exception as e:
        print(f"❌ Portfolio optimizer test failed: {str(e)}")
        return False

async def test_factor_model_fix():
    """Test factor model with list data"""
    print("\n🧪 Testing Factor Model Fix...")
    
    try:
        from src.quantitative.factor_model import FactorModel
        
        factor_model = FactorModel()
        
        # Test with list data (should return error)
        list_data = [0.01, -0.02, 0.03]
        result = factor_model.build_factor_model(list_data)
        
        if 'error' in result and 'list' in result['error']:
            print("✅ Factor model correctly handles list data")
            return True
        else:
            print("❌ Factor model should have returned error for list data")
            return False
            
    except Exception as e:
        print(f"❌ Factor model test failed: {str(e)}")
        return False

async def test_quantitative_system_methods():
    """Test QuantitativeTradingSystem methods"""
    print("\n🧪 Testing QuantitativeTradingSystem Methods...")
    
    try:
        from src.quantitative.quantitative_trading_system import QuantitativeTradingSystem
        
        system = QuantitativeTradingSystem()
        
        # Test get_performance_metrics
        metrics = system.get_performance_metrics()
        if isinstance(metrics, dict):
            print("✅ get_performance_metrics method exists and works")
            return True
        else:
            print("❌ get_performance_metrics should return dict")
            return False
            
    except Exception as e:
        print(f"❌ QuantitativeTradingSystem test failed: {str(e)}")
        return False

async def test_strategy_methods():
    """Test strategy methods with proper data"""
    print("\n🧪 Testing Strategy Methods...")
    
    try:
        from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
        
        # Create mock services for strategy
        config = {'trading': {'profit_target': 0.05}}
        binance_service = MockBinanceService()
        indicator_service = MockIndicatorService()
        notification_service = Mock()
        cache_service = MockCacheService()
        
        strategy = EnhancedTradingStrategyWithQuantitative(
            config, binance_service, indicator_service, notification_service, cache_service
        )
        
        # Test analyze_portfolio_optimization
        symbols = ['BTCUSDT', 'ETHUSDT']
        result = await strategy.analyze_portfolio_optimization(symbols)
        
        if isinstance(result, dict):
            print("✅ analyze_portfolio_optimization works")
            return True
        else:
            print("❌ analyze_portfolio_optimization should return dict")
            return False
            
    except Exception as e:
        print(f"❌ Strategy methods test failed: {str(e)}")
        return False

async def test_notification_function():
    """Test notification function with None services"""
    print("\n🧪 Testing Notification Function...")
    
    try:
        from main_with_quantitative import send_quantitative_notification
        
        symbol = 'BTCUSDT'
        signals = {
            'quantitative_confidence': 0.8,
            'action': 'buy',
            'optimized_position_size': 0.01
        }
        recommendations = {
            'trading_recommendation': {
                'action': 'buy',
                'confidence': 0.8
            }
        }
        
        await send_quantitative_notification(
            symbol, signals, recommendations, None, None
        )
        print("✅ Notification function handles None services correctly")
        return True
        
    except Exception as e:
        print(f"❌ Notification function test failed: {str(e)}")
        return False

async def main():
    """Run all comprehensive tests"""
    print("🧪 Comprehensive Error Fixes Test")
    print("=" * 50)
    
    tests = [
        test_discord_service_fix,
        test_portfolio_optimizer_fix,
        test_factor_model_fix,
        test_quantitative_system_methods,
        test_strategy_methods,
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
        print("🎉 All tests passed! All error fixes are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please review the fixes.")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 