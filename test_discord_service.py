#!/usr/bin/env python3
"""
Test script to check Discord service initialization.
"""

import sys
import os
import asyncio
import logging
from typing import Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import load_config
from src.services.discord_service import DiscordService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_discord_service():
    """Test Discord service initialization."""
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Check Discord configuration
        discord_config = config.get('api', {}).get('discord', {})
        logger.info(f"Discord config: {discord_config}")
        
        # Check if Discord is enabled
        discord_enabled = discord_config.get('enabled', False)
        logger.info(f"Discord enabled: {discord_enabled}")
        
        # Check bot token
        bot_token = discord_config.get('bot_token')
        logger.info(f"Bot token available: {bot_token is not None}")
        
        # Check webhook URL
        webhook_url = discord_config.get('webhook_url')
        logger.info(f"Webhook URL available: {webhook_url is not None}")
        
        # Check channel ID
        channel_id = discord_config.get('channel_id')
        logger.info(f"Channel ID available: {channel_id is not None}")
        
        if not discord_enabled:
            logger.warning("Discord is not enabled in configuration")
            return
        
        if not bot_token and not webhook_url:
            logger.error("Neither bot token nor webhook URL is configured")
            return
        
        # Try to initialize Discord service
        logger.info("Attempting to initialize Discord service...")
        discord_service = DiscordService(config)
        
        try:
            await discord_service.initialize()
            logger.info("Discord service initialized successfully")
            
            # Test sending a message
            test_message = "🧪 Test message from Discord service"
            success = await discord_service.send_message(test_message)
            logger.info(f"Message sent successfully: {success}")
            
            # Close the service
            await discord_service.close()
            logger.info("Discord service closed successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Discord service: {str(e)}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(test_discord_service()) 