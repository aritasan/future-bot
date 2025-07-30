#!/usr/bin/env python3
"""
Test script to verify stop loss and take profit functions
"""

import asyncio
import sys
import numpy as np
from unittest.mock import Mock, AsyncMock

# Mock the necessary imports
sys.path.append('.')

class MockBinanceService:
    def __init__(self):
        self.orders = []
    
    async def get_account_info(self):
        return {'totalWalletBalance': '1000.0'}
    
    async def get_positions(self):
        return [
            {
                'symbol': 'BTCUSDT',
                'info': {'positionAmt': '0.001'},
                'unrealizedPnl': '10.0'
            }
        ]
    
    async def place_order(self, symbol, side, order_type, amount, **kwargs):
        order = {
            'symbol': symbol,
            'side': side,
            'orderType': order_type,
            'quantity': amount,
            **kwargs
        }
        self.orders.append(order)
        return order

class MockIndicatorService:
    async def get_klines(self, symbol, interval, limit=100):
        # Create mock price data
        base_price = 50000.0
        prices = [base_price + i * 100 for i in range(limit)]
        volumes = [1000 + i * 10 for i in range(limit)]
        
        return {
            'close': prices,
            'high': [p + 50 for p in prices],
            'low': [p - 50 for p in prices],
            'volume': volumes
        }

class MockNotificationService:
    async def send_notification(self, message):
        print(f"Notification: {message}")

async def test_stop_loss_calculation():
    """Test stop loss calculation for different scenarios"""
    print("🧪 Testing Stop Loss Calculation...")
    
    # Mock config
    config = {
        'risk_management': {
            'stop_loss_atr_multiplier': 2.0,
            'min_stop_distance': 0.01
        }
    }
    
    # Create mock services
    binance_service = MockBinanceService()
    indicator_service = MockIndicatorService()
    notification_service = MockNotificationService()
    
    # Import and create strategy instance
    from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
    
    strategy = EnhancedTradingStrategyWithQuantitative(
        config, binance_service, indicator_service, notification_service
    )
    
    # Test cases
    test_cases = [
        {
            'symbol': 'BTCUSDT',
            'position_type': 'LONG',
            'current_price': 50000.0,
            'atr': 1000.0,
            'expected_range': (48000, 49000)
        },
        {
            'symbol': 'BTCUSDT',
            'position_type': 'SHORT',
            'current_price': 50000.0,
            'atr': 1000.0,
            'expected_range': (51000, 52000)
        },
        {
            'symbol': 'ETHUSDT',
            'position_type': 'LONG',
            'current_price': 3000.0,
            'atr': 50.0,
            'expected_range': (2900, 2950)
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            stop_loss = await strategy._calculate_stop_loss(
                test_case['symbol'],
                test_case['position_type'],
                test_case['current_price'],
                test_case['atr']
            )
            
            if stop_loss is not None:
                min_expected, max_expected = test_case['expected_range']
                if min_expected <= stop_loss <= max_expected:
                    print(f"✅ Test {i} passed: Stop loss = {stop_loss:.2f}")
                    passed += 1
                else:
                    print(f"❌ Test {i} failed: Stop loss = {stop_loss:.2f} (expected: {min_expected:.2f}-{max_expected:.2f})")
            else:
                print(f"❌ Test {i} failed: Stop loss is None")
                
        except Exception as e:
            print(f"❌ Test {i} failed with error: {str(e)}")
    
    print(f"\n📊 Stop Loss Tests: {passed}/{total} passed")
    return passed == total

async def test_take_profit_calculation():
    """Test take profit calculation for different scenarios"""
    print("\n🧪 Testing Take Profit Calculation...")
    
    # Mock config
    config = {
        'risk_management': {
            'take_profit_multiplier': 2.0,
            'min_tp_distance': 0.01
        }
    }
    
    # Create mock services
    binance_service = MockBinanceService()
    indicator_service = MockIndicatorService()
    notification_service = MockNotificationService()
    
    # Import and create strategy instance
    from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
    
    strategy = EnhancedTradingStrategyWithQuantitative(
        config, binance_service, indicator_service, notification_service
    )
    
    # Test cases
    test_cases = [
        {
            'symbol': 'BTCUSDT',
            'position_type': 'LONG',
            'current_price': 50000.0,
            'stop_loss': 48000.0,
            'expected_range': (52000, 54000)
        },
        {
            'symbol': 'BTCUSDT',
            'position_type': 'SHORT',
            'current_price': 50000.0,
            'stop_loss': 52000.0,
            'expected_range': (48000, 50000)
        },
        {
            'symbol': 'ETHUSDT',
            'position_type': 'LONG',
            'current_price': 3000.0,
            'stop_loss': 2900.0,
            'expected_range': (3100, 3300)
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            take_profit = await strategy._calculate_take_profit(
                test_case['symbol'],
                test_case['position_type'],
                test_case['current_price'],
                test_case['stop_loss']
            )
            
            if take_profit is not None:
                min_expected, max_expected = test_case['expected_range']
                if min_expected <= take_profit <= max_expected:
                    print(f"✅ Test {i} passed: Take profit = {take_profit:.2f}")
                    passed += 1
                else:
                    print(f"❌ Test {i} failed: Take profit = {take_profit:.2f} (expected: {min_expected:.2f}-{max_expected:.2f})")
            else:
                print(f"❌ Test {i} failed: Take profit is None")
                
        except Exception as e:
            print(f"❌ Test {i} failed with error: {str(e)}")
    
    print(f"\n📊 Take Profit Tests: {passed}/{total} passed")
    return passed == total

async def test_market_conditions():
    """Test market conditions calculation"""
    print("\n🧪 Testing Market Conditions...")
    
    # Mock config
    config = {
        'risk_management': {
            'stop_loss_atr_multiplier': 2.0,
            'min_stop_distance': 0.01
        }
    }
    
    # Create mock services
    binance_service = MockBinanceService()
    indicator_service = MockIndicatorService()
    notification_service = MockNotificationService()
    
    # Import and create strategy instance
    from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
    
    strategy = EnhancedTradingStrategyWithQuantitative(
        config, binance_service, indicator_service, notification_service
    )
    
    try:
        market_conditions = await strategy._get_market_conditions('BTCUSDT')
        
        if 'volatility' in market_conditions and 'price_change_24h' in market_conditions:
            print(f"✅ Market conditions test passed:")
            print(f"   Volatility: {market_conditions['volatility']:.4f}")
            print(f"   Price change 24h: {market_conditions['price_change_24h']:.2f}%")
            return True
        else:
            print("❌ Market conditions test failed: Missing required fields")
            return False
            
    except Exception as e:
        print(f"❌ Market conditions test failed with error: {str(e)}")
        return False

async def test_edge_cases():
    """Test edge cases for stop loss and take profit"""
    print("\n🧪 Testing Edge Cases...")
    
    # Mock config
    config = {
        'risk_management': {
            'stop_loss_atr_multiplier': 2.0,
            'take_profit_multiplier': 2.0,
            'min_stop_distance': 0.01,
            'min_tp_distance': 0.01
        }
    }
    
    # Create mock services
    binance_service = MockBinanceService()
    indicator_service = MockIndicatorService()
    notification_service = MockNotificationService()
    
    # Import and create strategy instance
    from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
    
    strategy = EnhancedTradingStrategyWithQuantitative(
        config, binance_service, indicator_service, notification_service
    )
    
    edge_cases = [
        # Very low ATR
        {
            'name': 'Low ATR',
            'position_type': 'LONG',
            'current_price': 50000.0,
            'atr': 10.0
        },
        # Very high ATR
        {
            'name': 'High ATR',
            'position_type': 'LONG',
            'current_price': 50000.0,
            'atr': 5000.0
        },
        # Zero ATR
        {
            'name': 'Zero ATR',
            'position_type': 'LONG',
            'current_price': 50000.0,
            'atr': 0.0
        },
        # SHORT position with high ATR
        {
            'name': 'SHORT High ATR',
            'position_type': 'SHORT',
            'current_price': 50000.0,
            'atr': 3000.0
        }
    ]
    
    passed = 0
    total = len(edge_cases)
    
    for i, case in enumerate(edge_cases, 1):
        try:
            stop_loss = await strategy._calculate_stop_loss(
                'BTCUSDT',
                case['position_type'],
                case['current_price'],
                case['atr']
            )
            
            if stop_loss is not None and stop_loss > 0:
                print(f"✅ Edge case {i} ({case['name']}) passed: Stop loss = {stop_loss:.2f}")
                passed += 1
            else:
                print(f"❌ Edge case {i} ({case['name']}) failed: Stop loss = {stop_loss}")
                
        except Exception as e:
            print(f"❌ Edge case {i} ({case['name']}) failed with error: {str(e)}")
    
    print(f"\n📊 Edge Cases Tests: {passed}/{total} passed")
    return passed == total

async def main():
    """Run all tests"""
    print("🧪 Testing Stop Loss and Take Profit Implementation...")
    print("=" * 60)
    
    tests = [
        test_stop_loss_calculation,
        test_take_profit_calculation,
        test_market_conditions,
        test_edge_cases
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if await test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Overall Test Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All tests passed! Stop loss and take profit implementation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 