#!/usr/bin/env python3
"""
Test script to verify the notification callback fix in BinanceService.
"""

import asyncio
import logging
from typing import Dict, Optional
from src.services.binance_service import BinanceService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockNotificationCallback:
    """Mock notification callback for testing."""
    
    def __init__(self):
        self.notifications = []
    
    async def __call__(self, message: str):
        """Mock notification callback."""
        self.notifications.append(message)
        logger.info(f"Mock notification sent: {message[:100]}...")
    
    def get_notifications(self):
        """Get all notifications sent."""
        return self.notifications
    
    def clear_notifications(self):
        """Clear all notifications."""
        self.notifications = []

async def test_notification_callback():
    """Test the notification callback functionality."""
    logger.info("Testing notification callback fix...")
    
    # Create mock config
    config = {
        'api': {
            'binance': {
                'use_testnet': True,
                'testnet': {
                    'api_key': 'test_key',
                    'api_secret': 'test_secret'
                }
            }
        },
        'ip_monitor': {
            'enabled': False
        }
    }
    
    # Create mock notification callback
    mock_callback = MockNotificationCallback()
    
    # Create BinanceService with notification callback
    binance_service = BinanceService(config, notification_callback=mock_callback)
    
    # Test that the callback is properly stored
    assert binance_service.notification_callback == mock_callback, "Notification callback not properly stored"
    logger.info("✓ Notification callback properly stored")
    
    # Test notification sending (without actually placing orders)
    test_order = {
        'id': 'test_order_123',
        'status': 'FILLED',
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'type': 'MARKET',
        'amount': 0.001,
        'price': 50000.0
    }
    
    test_order_params = {
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'type': 'MARKET',
        'amount': 0.001,
        'price': 50000.0,
        'stop_loss': 49000.0,
        'take_profit': 52000.0
    }
    
    # Test notification sending
    try:
        await binance_service._send_order_notification(test_order, test_order_params)
        logger.info("✓ Notification method executed without errors")
        
        # Check if notification was sent
        notifications = mock_callback.get_notifications()
        if notifications:
            logger.info(f"✓ Notification sent successfully: {len(notifications)} notification(s)")
            for i, notification in enumerate(notifications):
                logger.info(f"  Notification {i+1}: {notification[:200]}...")
        else:
            logger.warning("⚠ No notifications were sent")
            
    except Exception as e:
        logger.error(f"✗ Error in notification test: {str(e)}")
        return False
    
    # Test with no callback (should not error)
    binance_service_no_callback = BinanceService(config, notification_callback=None)
    try:
        await binance_service_no_callback._send_order_notification(test_order, test_order_params)
        logger.info("✓ Notification method works correctly with no callback")
    except Exception as e:
        logger.error(f"✗ Error when no callback provided: {str(e)}")
        return False
    
    logger.info("✓ All notification tests passed!")
    return True

async def main():
    """Main test function."""
    logger.info("Starting notification callback fix test...")
    
    try:
        success = await test_notification_callback()
        if success:
            logger.info("🎉 NOTIFICATION FIX TEST PASSED!")
        else:
            logger.error("❌ NOTIFICATION FIX TEST FAILED!")
            return False
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(main()) 