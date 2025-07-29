"""
Test script for IP monitoring functionality.
"""

import asyncio
import logging
from src.core.config import load_config
from src.services.ip_monitor_service import IPMonitorService
from src.services.notification_service import NotificationService
from src.services.telegram_service import TelegramService
from src.services.discord_service import DiscordService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_ip_monitor():
    """Test IP monitoring functionality."""
    try:
        # Load configuration
        config = load_config()
        
        # Create notification services
        telegram_service = TelegramService(config) if config.get('api', {}).get('telegram', {}).get('enabled', True) else None
        discord_service = DiscordService(config) if config.get('api', {}).get('discord', {}).get('enabled', True) else None
        
        notification_service = NotificationService(
            config=config,
            telegram_service=telegram_service,
            discord_service=discord_service
        )
        
        # Initialize notification service
        await notification_service.initialize()
        
        # Create IP monitor with notification callback
        ip_monitor = IPMonitorService(config, notification_service.send_message)
        
        # Initialize IP monitor
        if await ip_monitor.initialize():
            logger.info("IP monitor initialized successfully")
            
            # Get current IP
            current_ip = await ip_monitor.get_current_ip()
            logger.info(f"Current IP: {current_ip}")
            
            # Start monitoring
            await ip_monitor.start_monitoring()
            logger.info("IP monitoring started")
            
            # Test for 30 seconds
            logger.info("Testing IP monitoring for 30 seconds...")
            await asyncio.sleep(30)
            
            # Force IP check
            logger.info("Forcing IP check...")
            new_ip = await ip_monitor.force_ip_check()
            logger.info(f"Force check result: {new_ip}")
            
            # Stop monitoring
            await ip_monitor.stop_monitoring()
            logger.info("IP monitoring stopped")
            
        else:
            logger.error("Failed to initialize IP monitor")
            
    except Exception as e:
        logger.error(f"Error in IP monitor test: {str(e)}")
        
    finally:
        # Cleanup
        if 'notification_service' in locals():
            await notification_service.close()
        if 'ip_monitor' in locals():
            await ip_monitor.close()

async def test_ip_error_detection():
    """Test IP error detection in Binance service."""
    try:
        from src.services.binance_service import BinanceService
        
        # Load configuration
        config = load_config()
        
        # Create notification services
        telegram_service = TelegramService(config) if config.get('api', {}).get('telegram', {}).get('enabled', True) else None
        discord_service = DiscordService(config) if config.get('api', {}).get('discord', {}).get('enabled', True) else None
        
        notification_service = NotificationService(
            config=config,
            telegram_service=telegram_service,
            discord_service=discord_service
        )
        
        # Initialize notification service
        await notification_service.initialize()
        
        # Create Binance service with notification callback
        binance_service = BinanceService(config, notification_service.send_message)
        
        # Test IP error detection
        logger.info("Testing IP error detection...")
        
        # Simulate IP error
        class MockIPError(Exception):
            def __str__(self):
                return "Invalid API-key, IP, or permissions for action"
        
        mock_error = MockIPError()
        
        # Test IP error detection
        is_ip_error = await binance_service._is_ip_error(mock_error)
        logger.info(f"Is IP error: {is_ip_error}")
        
        if is_ip_error:
            await binance_service._handle_ip_error(mock_error)
            logger.info("IP error handled successfully")
        
    except Exception as e:
        logger.error(f"Error in IP error detection test: {str(e)}")
        
    finally:
        # Cleanup
        if 'notification_service' in locals():
            await notification_service.close()

if __name__ == "__main__":
    print("Testing IP monitoring functionality...")
    
    # Test 1: IP Monitor
    print("\n=== Test 1: IP Monitor ===")
    asyncio.run(test_ip_monitor())
    
    # Test 2: IP Error Detection
    print("\n=== Test 2: IP Error Detection ===")
    asyncio.run(test_ip_error_detection())
    
    print("\nTests completed!") 