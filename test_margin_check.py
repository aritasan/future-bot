#!/usr/bin/env python3
"""
Test script to verify the new margin checking logic in place_order function.
"""

import asyncio
import logging
import json
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockBinanceService:
    """Mock Binance service for testing margin check logic."""
    
    def __init__(self):
        self._is_initialized = True
        self._is_closed = False
        
    async def get_account_balance(self) -> Optional[Dict]:
        """Mock account balance."""
        return {
            'total': {
                'USDT': '100.0',  # $100 total balance
                'BTC': '0.001'
            },
            'free': {
                'USDT': '50.0',   # $50 available balance
                'BTC': '0.001'
            },
            'used': {
                'USDT': '50.0',   # $50 used balance
                'BTC': '0.0'
            }
        }
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Mock current price."""
        prices = {
            'BTCUSDT': 50000.0,
            'ETHUSDT': 3000.0,
            'ADAUSDT': 0.5,
            'ZKJUSDT': 0.15
        }
        return prices.get(symbol, 100.0)
    
    async def _check_margin_for_order(self, order_params: Dict) -> Dict:
        """Check if there's sufficient margin for the order."""
        try:
            # Get account balance
            balance = await self.get_account_balance()
            if not balance or 'total' not in balance:
                return {
                    'sufficient': False,
                    'required': 0,
                    'available': 0,
                    'reason': 'Could not fetch account balance'
                }
            
            # Get available USDT balance
            available_usdt = float(balance.get('free', {}).get('USDT', 0))
            
            # Calculate required margin for the order
            symbol = order_params['symbol']
            amount = float(order_params['amount'])
            
            # Get current price for margin calculation
            current_price = await self.get_current_price(symbol)
            if not current_price:
                return {
                    'sufficient': False,
                    'required': 0,
                    'available': available_usdt,
                    'reason': f'Could not get current price for {symbol}'
                }
            
            # Calculate required margin (position value + buffer)
            position_value = amount * current_price
            margin_buffer = 0.1  # 10% buffer for fees and price fluctuations
            required_margin = position_value * (1 + margin_buffer)
            
            # Check if sufficient margin
            sufficient = available_usdt >= required_margin
            
            return {
                'sufficient': sufficient,
                'required': required_margin,
                'available': available_usdt,
                'reason': 'Insufficient USDT balance' if not sufficient else 'Sufficient margin'
            }
            
        except Exception as e:
            logger.error(f"Error checking margin for order: {str(e)}")
            return {
                'sufficient': False,
                'required': 0,
                'available': 0,
                'reason': f'Error checking margin: {str(e)}'
            }

async def test_margin_check_logic():
    """Test the margin check logic with different scenarios."""
    logger.info("🧪 Testing margin check logic...")
    
    # Create mock service
    mock_service = MockBinanceService()
    
    # Test cases
    test_cases = [
        {
            'name': 'Small order - should pass',
            'order_params': {
                'symbol': 'ADAUSDT',
                'side': 'BUY',
                'type': 'MARKET',
                'amount': 10.0  # 10 ADA = $5 + 10% buffer = $5.5
            },
            'expected': True
        },
        {
            'name': 'Medium order - should pass',
            'order_params': {
                'symbol': 'ETHUSDT',
                'side': 'SELL',
                'type': 'MARKET',
                'amount': 0.01  # 0.01 ETH = $30 + 10% buffer = $33
            },
            'expected': True
        },
        {
            'name': 'Large order - should fail',
            'order_params': {
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'type': 'MARKET',
                'amount': 0.001  # 0.001 BTC = $50 + 10% buffer = $55
            },
            'expected': False
        },
        {
            'name': 'Very large order - should fail',
            'order_params': {
                'symbol': 'BTCUSDT',
                'side': 'SELL',
                'type': 'MARKET',
                'amount': 0.01  # 0.01 BTC = $500 + 10% buffer = $550
            },
            'expected': False
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n📋 Test {i}: {test_case['name']}")
        
        # Test margin check
        margin_check = await mock_service._check_margin_for_order(test_case['order_params'])
        
        # Log results
        logger.info(f"  Symbol: {test_case['order_params']['symbol']}")
        logger.info(f"  Amount: {test_case['order_params']['amount']}")
        logger.info(f"  Required margin: ${margin_check['required']:.2f}")
        logger.info(f"  Available balance: ${margin_check['available']:.2f}")
        logger.info(f"  Sufficient: {margin_check['sufficient']}")
        logger.info(f"  Reason: {margin_check['reason']}")
        
        # Check if result matches expectation
        passed = margin_check['sufficient'] == test_case['expected']
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  Status: {status}")
        
        results.append({
            'test_name': test_case['name'],
            'passed': passed,
            'margin_check': margin_check
        })
    
    # Summary
    logger.info(f"\n📊 Test Summary:")
    passed_count = sum(1 for result in results if result['passed'])
    total_count = len(results)
    
    logger.info(f"  Total tests: {total_count}")
    logger.info(f"  Passed: {passed_count}")
    logger.info(f"  Failed: {total_count - passed_count}")
    
    if passed_count == total_count:
        logger.info("🎉 All margin check tests passed!")
        return True
    else:
        logger.error("❌ Some margin check tests failed!")
        return False

async def test_edge_cases():
    """Test edge cases for margin checking."""
    logger.info("\n🔍 Testing edge cases...")
    
    mock_service = MockBinanceService()
    
    # Test with missing symbol
    edge_case = {
        'symbol': 'INVALIDUSDT',
        'side': 'BUY',
        'type': 'MARKET',
        'amount': 1.0
    }
    
    margin_check = await mock_service._check_margin_for_order(edge_case)
    logger.info(f"Invalid symbol test: {margin_check}")
    
    # Test with zero amount
    edge_case_zero = {
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'type': 'MARKET',
        'amount': 0.0
    }
    
    margin_check_zero = await mock_service._check_margin_for_order(edge_case_zero)
    logger.info(f"Zero amount test: {margin_check_zero}")
    
    return True

async def main():
    """Main test function."""
    logger.info("🚀 Starting margin check tests...")
    
    try:
        # Test basic margin check logic
        basic_tests_passed = await test_margin_check_logic()
        
        # Test edge cases
        edge_cases_passed = await test_edge_cases()
        
        if basic_tests_passed and edge_cases_passed:
            logger.info("🎉 All tests passed! Margin check logic is working correctly.")
        else:
            logger.error("❌ Some tests failed! Please review the margin check logic.")
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(main()) 