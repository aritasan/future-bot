"""
Notification service for managing different notification channels.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.services.telegram_service import TelegramService
from src.services.discord_service import DiscordService

logger = logging.getLogger(__name__)

class NotificationService:
    def __init__(self, config: Dict[str, Any], telegram_service: Optional[TelegramService] = None, discord_service: Optional[DiscordService] = None):
        """
        Initialize notification service.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
            telegram_service (Optional[TelegramService]): Telegram service instance
            discord_service (Optional[DiscordService]): Discord service instance
        """
        self.config = config
        self.telegram_service = telegram_service
        self.discord_service = discord_service
        self._is_initialized = False
        
        # Get service enabled status from config
        self.telegram_enabled = config.get('api', {}).get('telegram', {}).get('enabled', True)
        self.discord_enabled = config.get('api', {}).get('discord', {}).get('enabled', True)

    async def initialize(self) -> bool:
        """Initialize the notification service."""
        try:
            if self.telegram_enabled and self.telegram_service:
                await self.telegram_service.initialize()
                
            if self.discord_enabled and self.discord_service:
                await self.discord_service.initialize()
                
            self._is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing notification service: {str(e)}")
            return False

    async def close(self) -> None:
        """Close the notification service."""
        try:
            if self.telegram_service:
                await self.telegram_service.close()
            if self.discord_service:
                await self.discord_service.close()
        except Exception as e:
            logger.error(f"Error closing notification service: {str(e)}")

    def is_initialized(self) -> bool:
        """Check if the service is initialized."""
        return self._is_initialized

    async def send_message(self, message: str, embed: Optional[Dict[str, Any]] = None) -> None:
        """
        Send a message through all enabled notification channels.
        
        Args:
            message (str): Message to send
            embed (Optional[Dict[str, Any]]): Optional embed data for rich messages
        """
        if not self._is_initialized:
            logger.warning("Notification service not initialized")
            return

        try:
            if self.telegram_enabled and self.telegram_service:
                await self.telegram_service.send_message(message)
                
            if self.discord_enabled and self.discord_service:
                await self.discord_service.send_message(message, embed)
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")

    async def send_trade_notification(self, 
                                    symbol: str, 
                                    action: str, 
                                    price: float, 
                                    quantity: float,
                                    reason: Optional[str] = None) -> None:
        """
        Send a trade notification through all enabled channels.
        
        Args:
            symbol (str): Trading pair symbol
            action (str): Trade action (BUY/SELL)
            price (float): Trade price
            quantity (float): Trade quantity
            reason (Optional[str]): Optional reason for the trade
        """
        if not self._is_initialized:
            logger.warning("Notification service not initialized")
            return

        try:
            if self.telegram_enabled and self.telegram_service:
                await self.telegram_service.send_trade_notification(
                    symbol=symbol,
                    action=action,
                    price=price,
                    quantity=quantity,
                    reason=reason
                )
                
            if self.discord_enabled and self.discord_service:
                await self.discord_service.send_trade_notification(
                    symbol=symbol,
                    action=action,
                    price=price,
                    quantity=quantity,
                    reason=reason
                )
        except Exception as e:
            logger.error(f"Error sending trade notification: {str(e)}")

    async def send_error_notification(self, error_message: str) -> None:
        """
        Send an error notification through all enabled channels.
        
        Args:
            error_message (str): Error message to send
        """
        if not self._is_initialized:
            logger.warning("Notification service not initialized")
            return

        try:
            if self.telegram_enabled and self.telegram_service:
                await self.telegram_service.send_error_notification(error_message)
                
            if self.discord_enabled and self.discord_service:
                await self.discord_service.send_error_notification(error_message)
        except Exception as e:
            logger.error(f"Error sending error notification: {str(e)}")

    async def send_health_status(self, status: Dict[str, Any]) -> None:
        """
        Send health status through all enabled channels.
        
        Args:
            status (Dict[str, Any]): Health status information
        """
        if not self._is_initialized:
            logger.warning("Notification service not initialized")
            return

        try:
            if self.telegram_enabled and self.telegram_service:
                await self.telegram_service.send_health_status(status)
                
            if self.discord_enabled and self.discord_service:
                await self.discord_service.send_health_status(status)
        except Exception as e:
            logger.error(f"Error sending health status: {str(e)}")

    def toggle_telegram(self, enabled: bool) -> None:
        """
        Toggle Telegram notifications on/off.
        
        Args:
            enabled (bool): Whether to enable or disable Telegram notifications
        """
        self.telegram_enabled = enabled
        logger.info(f"Telegram notifications {'enabled' if enabled else 'disabled'}")

    def toggle_discord(self, enabled: bool) -> None:
        """
        Toggle Discord notifications on/off.
        
        Args:
            enabled (bool): Whether to enable or disable Discord notifications
        """
        self.discord_enabled = enabled
        logger.info(f"Discord notifications {'enabled' if enabled else 'disabled'}")

    def get_status(self) -> Dict[str, bool]:
        """
        Get the current status of notification services.
        
        Returns:
            Dict[str, bool]: Dictionary containing enabled status for each service
        """
        return {
            'telegram': self.telegram_enabled,
            'discord': self.discord_enabled
        } 