#!/usr/bin/env python3
"""
Test script to verify that notification callback can be set after initialization.
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
    
    async def send_message(self, message: str):
        """Mock notification callback."""
        self.notifications.append(message)
        logger.info(f"Mock notification sent: {message[:100]}...")
    
    def get_notifications(self):
        """Get all notifications sent."""
        return self.notifications
    
    def clear_notifications(self):
        """Clear all notifications."""
        self.notifications = []

async def test_notification_callback_set():
    """Test setting notification callback after initialization."""
    logger.info("Testing notification callback set functionality...")
    
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
    
    # Create BinanceService without notification callback initially
    binance_service = BinanceService(config)
    
    # Test that no callback is set initially
    assert binance_service.notification_callback is None, "Notification callback should be None initially"
    logger.info("✓ Notification callback is None initially")
    
    # Test notification sending without callback (should not error)
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
    
    # Test notification sending without callback
    try:
        await binance_service._send_order_notification(test_order, test_order_params)
        logger.info("✓ Notification method works correctly without callback")
    except Exception as e:
        logger.error(f"✗ Error when no callback provided: {str(e)}")
        return False
    
    # Create mock notification callback
    mock_callback = MockNotificationCallback()
    
    # Set notification callback after initialization
    binance_service.set_notification_callback(mock_callback.send_message)
    
    # Test that callback is now set
    assert binance_service.notification_callback == mock_callback.send_message, "Notification callback not properly set"
    logger.info("✓ Notification callback properly set after initialization")
    
    # Test notification sending with callback
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
    
    # Test changing callback
    new_mock_callback = MockNotificationCallback()
    binance_service.set_notification_callback(new_mock_callback.send_message)
    
    # Test notification with new callback
    try:
        await binance_service._send_order_notification(test_order, test_order_params)
        new_notifications = new_mock_callback.get_notifications()
        if new_notifications:
            logger.info(f"✓ New notification callback works: {len(new_notifications)} notification(s)")
        else:
            logger.warning("⚠ New callback didn't send notifications")
    except Exception as e:
        logger.error(f"✗ Error with new callback: {str(e)}")
        return False
    
    logger.info("✓ All notification callback set tests passed!")
    return True

async def main():
    """Main test function."""
    logger.info("Starting notification callback set test...")
    
    try:
        success = await test_notification_callback_set()
        if success:
            logger.info("🎉 NOTIFICATION CALLBACK SET TEST PASSED!")
        else:
            logger.error("❌ NOTIFICATION CALLBACK SET TEST FAILED!")
            return False
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(main()) 