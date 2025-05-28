"""
Discord notification service for the trading bot.
"""

import aiohttp
import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class DiscordService:
    def __init__(self, webhook_url: str):
        """
        Initialize Discord service with webhook URL.
        
        Args:
            webhook_url (str): Discord webhook URL for sending messages
        """
        self.webhook_url = webhook_url
        self.session: Optional[aiohttp.ClientSession] = None
        self._trading_paused = False

    async def initialize(self) -> None:
        """Initialize the Discord service."""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close(self) -> None:
        """Close the Discord service."""
        if self.session:
            await self.session.close()
            self.session = None

    def is_trading_paused(self) -> bool:
        """Check if trading is paused."""
        return self._trading_paused

    async def send_message(self, content: str, embed: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send a message to Discord channel.
        
        Args:
            content (str): Message content
            embed (Optional[Dict[str, Any]]): Optional embed data for rich messages
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        if not self.session:
            await self.initialize()

        try:
            payload = {"content": content}
            if embed:
                payload["embeds"] = [embed]

            async with self.session.post(self.webhook_url, json=payload) as response:
                if response.status == 204:  # Discord returns 204 on success
                    return True
                else:
                    logger.error(f"Failed to send Discord message. Status: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error sending Discord message: {str(e)}")
            return False

    async def send_trade_notification(self, 
                                    symbol: str, 
                                    action: str, 
                                    price: float, 
                                    quantity: float,
                                    reason: Optional[str] = None) -> None:
        """
        Send a trade notification to Discord.
        
        Args:
            symbol (str): Trading pair symbol
            action (str): Trade action (BUY/SELL)
            price (float): Trade price
            quantity (float): Trade quantity
            reason (Optional[str]): Optional reason for the trade
        """
        embed = {
            "title": f"Trade {action}",
            "color": 0x00ff00 if action == "BUY" else 0xff0000,
            "fields": [
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Price", "value": f"{price:.8f}", "inline": True},
                {"name": "Quantity", "value": f"{quantity:.8f}", "inline": True},
                {"name": "Time", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "inline": True}
            ]
        }
        
        if reason:
            embed["fields"].append({"name": "Reason", "value": reason, "inline": False})
            
        await self.send_message("", embed=embed)

    async def send_error_notification(self, error_message: str) -> None:
        """
        Send an error notification to Discord.
        
        Args:
            error_message (str): Error message to send
        """
        embed = {
            "title": "Error Alert",
            "color": 0xff0000,
            "description": error_message,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.send_message("", embed=embed)

    async def send_health_status(self, status: Dict[str, Any]) -> None:
        """
        Send health status to Discord.
        
        Args:
            status (Dict[str, Any]): Health status information
        """
        embed = {
            "title": "Health Status",
            "color": 0x00ff00 if status.get("healthy", False) else 0xff0000,
            "fields": [
                {"name": "Status", "value": "Healthy" if status.get("healthy", False) else "Unhealthy", "inline": True},
                {"name": "Last Check", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "inline": True}
            ]
        }
        
        if "details" in status:
            embed["fields"].append({"name": "Details", "value": status["details"], "inline": False})
            
        await self.send_message("", embed=embed) 