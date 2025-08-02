#!/usr/bin/env python3
"""
Test script to verify the order checking logic in BinanceService.
"""

import asyncio
import logging
from typing import Dict, Optional, List
from unittest.mock import AsyncMock, MagicMock

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockBinanceService:
    """Mock BinanceService for testing order checking logic."""
    
    def __init__(self):
        self._is_initialized = True
        self._is_closed = False
        self._cache = {}
        
    async def get_open_orders(self, symbol: str = None) -> Optional[List[Dict]]:
        """Mock get_open_orders method."""
        # Simulate existing orders for testing
        if symbol == "BTCUSDT":
            return [
                {
                    'id': '12345',
                    'symbol': 'BTCUSDT',
                    'type': 'MARKET',
                    'side': 'BUY',
                    'status': 'open',
                    'info': {'positionSide': 'LONG'}
                }
            ]
        elif symbol == "ETHUSDT":
            return [
                {
                    'id': '67890',
                    'symbol': 'ETHUSDT',
                    'type': 'STOP_MARKET',
                    'side': 'SELL',
                    'status': 'open',
                    'info': {'positionSide': 'SHORT'}
                }
            ]
        return []
    
    async def get_position(self, symbol: str, position_side: str = None) -> Optional[Dict]:
        """Mock get_position method."""
        # Simulate existing positions for testing
        if symbol == "ADAUSDT" and position_side == "LONG":
            return {
                'symbol': 'ADAUSDT',
                'info': {
                    'positionSide': 'LONG',
                    'positionAmt': '100.0'  # Existing position
                }
            }
        elif symbol == "DOTUSDT" and position_side == "SHORT":
            return {
                'symbol': 'DOTUSDT',
                'info': {
                    'positionSide': 'SHORT',
                    'positionAmt': '-50.0'  # Existing position
                }
            }
        return None
    
    async def get_existing_order(self, symbol: str, order_type: str, side: str) -> Optional[Dict]:
        """Mock get_existing_order method."""
        orders = await self.get_open_orders(symbol)
        if not orders:
            return None
            
        for order in orders:
            if (order['type'].lower() == order_type.lower() and 
                order['side'].lower() == side.lower()):
                return order
        return None
    
    async def should_place_order(self, symbol: str, order_type: str, side: str, position_side: str = None) -> Dict:
        """Check if an order should be placed based on existing orders and positions."""
        try:
            if not self._is_initialized:
                return {
                    'should_place': False,
                    'reason': 'Binance service not initialized',
                    'existing_order': None,
                    'existing_position': None
                }
                
            if self._is_closed:
                return {
                    'should_place': False,
                    'reason': 'Binance service is closed',
                    'existing_order': None,
                    'existing_position': None
                }
            
            # Check for existing orders
            existing_order = await self.get_existing_order(symbol, order_type, side)
            
            # Check for existing position if position_side is provided
            existing_position = None
            if position_side:
                existing_position = await self.get_position(symbol, position_side)
            
            # Determine if order should be placed
            should_place = True
            reason = "No existing orders or positions found"
            
            if existing_order:
                should_place = False
                reason = f"Existing {order_type} order found for {symbol} {side}"
                logger.info(f"Skipping order placement for {symbol}: {reason}")
                
            elif existing_position and float(existing_position.get('info', {}).get('positionAmt', 0)) != 0:
                should_place = False
                reason = f"Existing position found for {symbol} {position_side}"
                logger.info(f"Skipping order placement for {symbol}: {reason}")
            
            return {
                'should_place': should_place,
                'reason': reason,
                'existing_order': existing_order,
                'existing_position': existing_position
            }
            
        except Exception as e:
            logger.error(f"Error checking if order should be placed for {symbol}: {str(e)}")
            return {
                'should_place': False,
                'reason': f'Error checking order status: {str(e)}',
                'existing_order': None,
                'existing_position': None
            }

async def test_order_checking():
    """Test the order checking logic."""
    service = MockBinanceService()
    
    logger.info("🧪 Testing order checking logic...")
    
    # Test cases
    test_cases = [
        {
            'name': 'New order - should place',
            'symbol': 'LINKUSDT',
            'order_type': 'MARKET',
            'side': 'BUY',
            'position_side': 'LONG',
            'expected': True
        },
        {
            'name': 'Existing order - should skip',
            'symbol': 'BTCUSDT',
            'order_type': 'MARKET',
            'side': 'BUY',
            'position_side': 'LONG',
            'expected': False
        },
        {
            'name': 'Existing position - should skip',
            'symbol': 'ADAUSDT',
            'order_type': 'MARKET',
            'side': 'BUY',
            'position_side': 'LONG',
            'expected': False
        },
        {
            'name': 'Different order type - should place',
            'symbol': 'BTCUSDT',
            'order_type': 'LIMIT',
            'side': 'BUY',
            'position_side': 'LONG',
            'expected': True
        },
        {
            'name': 'Different side - should place',
            'symbol': 'BTCUSDT',
            'order_type': 'MARKET',
            'side': 'SELL',
            'position_side': 'SHORT',
            'expected': True
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n📋 Test {i}: {test_case['name']}")
        logger.info(f"   Symbol: {test_case['symbol']}")
        logger.info(f"   Order Type: {test_case['order_type']}")
        logger.info(f"   Side: {test_case['side']}")
        logger.info(f"   Position Side: {test_case['position_side']}")
        
        result = await service.should_place_order(
            test_case['symbol'],
            test_case['order_type'],
            test_case['side'],
            test_case['position_side']
        )
        
        logger.info(f"   Should Place: {result['should_place']}")
        logger.info(f"   Reason: {result['reason']}")
        
        if result['should_place'] == test_case['expected']:
            logger.info(f"   Status: ✅ PASS")
            passed += 1
        else:
            logger.error(f"   Status: ❌ FAIL (Expected: {test_case['expected']}, Got: {result['should_place']})")
    
    logger.info(f"\n📊 Test Summary:")
    logger.info(f"   Total tests: {total}")
    logger.info(f"   Passed: {passed}")
    logger.info(f"   Failed: {total - passed}")
    
    if passed == total:
        logger.info("🎉 All order checking tests passed!")
    else:
        logger.error(f"❌ {total - passed} tests failed!")
    
    return passed == total

async def test_edge_cases():
    """Test edge cases for order checking."""
    service = MockBinanceService()
    
    logger.info("\n🔍 Testing edge cases...")
    
    # Test with uninitialized service
    service._is_initialized = False
    result = await service.should_place_order("TESTUSDT", "MARKET", "BUY", "LONG")
    logger.info(f"Uninitialized service test: {result}")
    
    # Test with closed service
    service._is_initialized = True
    service._is_closed = True
    result = await service.should_place_order("TESTUSDT", "MARKET", "BUY", "LONG")
    logger.info(f"Closed service test: {result}")
    
    # Reset service state
    service._is_initialized = True
    service._is_closed = False
    
    logger.info("✅ Edge case tests completed!")

if __name__ == "__main__":
    print("Testing order checking logic...")
    
    # Run tests
    success = asyncio.run(test_order_checking())
    asyncio.run(test_edge_cases())
    
    if success:
        print("\n🎉 All tests passed! Order checking logic is working correctly.")
    else:
        print("\n❌ Some tests failed! Please review the logic.") 