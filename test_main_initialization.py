#!/usr/bin/env python3
"""
Test script to check complete initialization process as in main script.
"""

import sys
import os
import asyncio
import logging
from typing import Dict, Any, Optional

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import load_config
from src.services.binance_service import BinanceService
from src.services.telegram_service import TelegramService
from src.services.discord_service import DiscordService
from src.services.indicator_service import IndicatorService
from src.services.notification_service import NotificationService
from src.core.health_monitor import HealthMonitor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_main_initialization():
    """Test complete initialization process as in main script."""
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize services
        binance_service = None
        telegram_service = None
        discord_service = None
        health_monitor = None
        indicator_service = None
        notification_service = None
        
        try:
            # Initialize Binance service
            logger.info("Initializing Binance service...")
            binance_service = BinanceService(config)
            await binance_service.initialize()
            logger.info("Binance service initialized")
            
            # Initialize Telegram service
            if config.get('api', {}).get('telegram', {}).get('enabled', False):
                logger.info("Initializing Telegram service...")
                telegram_service = TelegramService(config)
                await telegram_service.initialize()
                logger.info("Telegram service initialized")
            else:
                logger.info("Telegram service not enabled")
            
            # Initialize Discord service
            if config.get('api', {}).get('discord', {}).get('enabled', False):
                try:
                    logger.info("Attempting to initialize Discord service...")
                    discord_service = DiscordService(config)
                    await discord_service.initialize()
                    logger.info("Discord service initialized successfully")
                except Exception as e:
                    logger.error(f"Error initializing Discord service: {str(e)}")
                    import traceback
                    logger.error(f"Discord initialization traceback:\n{traceback.format_exc()}")
                    discord_service = None
            else:
                logger.info("Discord service not enabled")
            
            # Initialize health monitor
            logger.info("Initializing health monitor...")
            health_monitor = HealthMonitor(config)
            await health_monitor.initialize()
            logger.info("Health monitor initialized")
            
            # Initialize indicator service
            logger.info("Initializing indicator service...")
            indicator_service = IndicatorService(config)
            await indicator_service.initialize()
            logger.info("Indicator service initialized")
            
            # Initialize notification service
            logger.info("Initializing notification service...")
            notification_service = NotificationService(config, telegram_service, discord_service)
            await notification_service.initialize()
            logger.info("Notification service initialized")
            
            # Test notification service
            logger.info("Testing notification service...")
            test_message = "🧪 Test message from complete initialization test"
            success = await notification_service.send_message(test_message)
            logger.info(f"Notification test result: {success}")
            
            # Log service status
            logger.info("Service initialization summary:")
            logger.info(f"- Binance service: {binance_service is not None}")
            logger.info(f"- Telegram service: {telegram_service is not None}")
            logger.info(f"- Discord service: {discord_service is not None}")
            logger.info(f"- Health monitor: {health_monitor is not None}")
            logger.info(f"- Indicator service: {indicator_service is not None}")
            logger.info(f"- Notification service: {notification_service is not None}")
            
            # Cleanup
            logger.info("Cleaning up services...")
            if notification_service:
                await notification_service.close()
            if indicator_service:
                await indicator_service.close()
            if health_monitor:
                await health_monitor.close()
            if discord_service:
                await discord_service.close()
            if telegram_service:
                await telegram_service.close()
            if binance_service:
                await binance_service.close()
            logger.info("All services cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            import traceback
            logger.error(f"Initialization traceback:\n{traceback.format_exc()}")
            
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(test_main_initialization()) 