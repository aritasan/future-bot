#!/usr/bin/env python3
"""
Test script to check main script Discord service initialization.
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

async def test_main_discord_initialization():
    """Test Discord service initialization as in main script."""
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Check Discord configuration
        discord_config = config.get('api', {}).get('discord', {})
        logger.info(f"Discord config: {discord_config}")
        
        # Check if Discord is enabled
        discord_enabled = config.get('api', {}).get('discord', {}).get('enabled', False)
        logger.info(f"Discord enabled check: {discord_enabled}")
        
        # Initialize Discord service (same logic as main script)
        discord_service = None
        if discord_enabled:
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
        
        logger.info(f"Final discord_service object: {discord_service}")
        logger.info(f"discord_service is None: {discord_service is None}")
        
        if discord_service:
            # Test sending a message
            test_message = "🧪 Test message from main script Discord initialization"
            success = await discord_service.send_message(test_message)
            logger.info(f"Message sent successfully: {success}")
            
            # Close the service
            await discord_service.close()
            logger.info("Discord service closed successfully")
        else:
            logger.warning("Discord service was not initialized")
            
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(test_main_discord_initialization()) 