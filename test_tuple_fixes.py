#!/usr/bin/env python3
"""
Test script to verify tuple handling fixes in the trading strategy.
"""

import sys
import os
import asyncio
import logging
from unittest.mock import Mock, AsyncMock

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.strategies.enhanced_trading_strategy import EnhancedTradingStrategy
from src.services.binance_service import BinanceService
from src.services.indicator_service import IndicatorService
from src.services.notification_service import NotificationService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_tuple_handling():
    """Test that tuple data is handled correctly."""
    
    # Mock configuration
    config = {
        'trading': {
            'max_positions': 5,
            'risk_per_trade': 0.02,
            'min_volume_ratio': 1.2,
            'max_volatility_ratio': 2.0,
            'min_adx': 25
        },
        'cache': {
            'enabled': True,
            'ttl': 300
        }
    }
    
    # Create mock services
    binance_service = Mock(spec=BinanceService)
    indicator_service = Mock(spec=IndicatorService)
    notification_service = Mock(spec=NotificationService)
    
    # Create strategy instance
    strategy = EnhancedTradingStrategy(
        config=config,
        binance_service=binance_service,
        indicator_service=indicator_service,
        notification_service=notification_service
    )
    
    # Test data - tuple that would cause errors
    tuple_data = ({"test": "data"}, 1234567890.0)  # (data, timestamp)
    
    print("Testing tuple handling fixes...")
    
    # Test 1: _check_trend_following_signal with tuple data
    print("\n1. Testing _check_trend_following_signal with tuple data...")
    market_conditions = {'df': tuple_data}
    result = strategy._check_trend_following_signal("TEST/USDT", market_conditions)
    print(f"Result: {result}")
    assert result is None, "Should return None for tuple data"
    
    # Test 2: _check_breakout_signal with tuple data
    print("\n2. Testing _check_breakout_signal with tuple data...")
    result = strategy._check_breakout_signal("TEST/USDT", market_conditions)
    print(f"Result: {result}")
    assert result is None, "Should return None for tuple data"
    
    # Test 3: _calculate_volume_score with tuple data
    print("\n3. Testing _calculate_volume_score with tuple data...")
    result = strategy._calculate_volume_score(tuple_data, "LONG")
    print(f"Result: {result}")
    assert result == 0.0, "Should return 0.0 for tuple data"
    
    # Test 4: _calculate_volatility_score with tuple data
    print("\n4. Testing _calculate_volatility_score with tuple data...")
    result = strategy._calculate_volatility_score(tuple_data, "LONG")
    print(f"Result: {result}")
    assert result == 0.0, "Should return 0.0 for tuple data"
    
    # Test 5: _calculate_sentiment_score with tuple data
    print("\n5. Testing _calculate_sentiment_score with tuple data...")
    result = strategy._calculate_sentiment_score(tuple_data, "LONG")
    print(f"Result: {result}")
    assert result == 0.0, "Should return 0.0 for tuple data"
    
    # Test 6: _calculate_structure_score with tuple data
    print("\n6. Testing _calculate_structure_score with tuple data...")
    result = strategy._calculate_structure_score(tuple_data, "LONG")
    print(f"Result: {result}")
    assert result == 0.0, "Should return 0.0 for tuple data"
    
    # Test 7: _calculate_volume_profile_score with tuple data
    print("\n7. Testing _calculate_volume_profile_score with tuple data...")
    result = strategy._calculate_volume_profile_score(tuple_data, "LONG")
    print(f"Result: {result}")
    assert result == 0.0, "Should return 0.0 for tuple data"
    
    # Test 8: _calculate_funding_rate_score with tuple data
    print("\n8. Testing _calculate_funding_rate_score with tuple data...")
    result = strategy._calculate_funding_rate_score(tuple_data, "LONG")
    print(f"Result: {result}")
    assert result == 0.0, "Should return 0.0 for tuple data"
    
    # Test 9: _calculate_open_interest_score with tuple data
    print("\n9. Testing _calculate_open_interest_score with tuple data...")
    result = strategy._calculate_open_interest_score(tuple_data, "LONG")
    print(f"Result: {result}")
    assert result == 0.0, "Should return 0.0 for tuple data"
    
    # Test 10: _calculate_order_book_score with tuple data
    print("\n10. Testing _calculate_order_book_score with tuple data...")
    result = strategy._calculate_order_book_score(tuple_data, "LONG")
    print(f"Result: {result}")
    assert result == 0.0, "Should return 0.0 for tuple data"
    
    # Test 11: _calculate_support_resistance with tuple data
    print("\n11. Testing _calculate_support_resistance with tuple data...")
    result = strategy._calculate_support_resistance(tuple_data)
    print(f"Result: {result}")
    assert isinstance(result, dict), "Should return a dictionary"
    assert 'support_levels' in result, "Should have support_levels key"
    
    # Test 12: _calculate_value_area with tuple data
    print("\n12. Testing _calculate_value_area with tuple data...")
    result = strategy._calculate_value_area(tuple_data)
    print(f"Result: {result}")
    assert isinstance(result, dict), "Should return a dictionary"
    assert 'value_area_high' in result, "Should have value_area_high key"
    
    # Test 13: _check_candlestick_patterns with tuple data
    print("\n13. Testing _check_candlestick_patterns with tuple data...")
    result = strategy._check_candlestick_patterns(tuple_data, "LONG")
    print(f"Result: {result}")
    assert result == 0.0, "Should return 0.0 for tuple data"
    
    # Test 14: check_volume_condition with tuple data
    print("\n14. Testing check_volume_condition with tuple data...")
    result = strategy.check_volume_condition(tuple_data)
    print(f"Result: {result}")
    assert result is False, "Should return False for tuple data"
    
    # Test 15: check_volatility_condition with tuple data
    print("\n15. Testing check_volatility_condition with tuple data...")
    result = strategy.check_volatility_condition(tuple_data)
    print(f"Result: {result}")
    assert result is False, "Should return False for tuple data"
    
    # Test 16: check_adx_condition with tuple data
    print("\n16. Testing check_adx_condition with tuple data...")
    result = strategy.check_adx_condition(tuple_data)
    print(f"Result: {result}")
    assert result is False, "Should return False for tuple data"
    
    # Test 17: check_bollinger_condition with tuple data
    print("\n17. Testing check_bollinger_condition with tuple data...")
    result = strategy.check_bollinger_condition(tuple_data)
    print(f"Result: {result}")
    assert result is False, "Should return False for tuple data"
    
    # Test 18: _calculate_momentum with tuple data
    print("\n18. Testing _calculate_momentum with tuple data...")
    result = strategy._calculate_momentum(tuple_data)
    print(f"Result: {result}")
    assert isinstance(result, dict), "Should return a dictionary"
    assert 'strength' in result, "Should have strength key"
    
    # Test 19: _should_exit_by_momentum with tuple data
    print("\n19. Testing _should_exit_by_momentum with tuple data...")
    position = {'symbol': 'TEST/USDT', 'side': 'LONG'}
    result = strategy._should_exit_by_momentum(tuple_data, position)
    print(f"Result: {result}")
    assert result is False, "Should return False for tuple data"
    
    print("\n✅ All tuple handling tests passed!")
    print("The tuple handling fixes are working correctly.")

if __name__ == "__main__":
    asyncio.run(test_tuple_handling()) 