"""
Discord notification service for the trading bot.
"""

import aiohttp
import logging
import os
import asyncio
import discord
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime
from aiohttp import ClientTimeout, TCPConnector
from aiohttp.resolver import AsyncResolver
import ssl

logger = logging.getLogger(__name__)

class DiscordService:
    def __init__(self, bot_token: str, channel_id: int, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize Discord service with bot token.
        
        Args:
            bot_token (str): Discord bot token
            channel_id (int): Channel ID to send messages to
            max_retries (int): Maximum number of retry attempts for failed requests
            retry_delay (float): Delay between retry attempts in seconds
        """
        self.bot_token = bot_token
        self.channel_id = channel_id
        # Configure intents to allow access to channels and messages
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.messages = True
        self.client = discord.Client(intents=intents)
        self._trading_paused = False
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._is_ready = False
        self._ready_event = asyncio.Event()
        
        # Command handlers
        self._command_handlers: Dict[str, Callable] = {
            'start': self._handle_start,
            'help': self._handle_help,
            'status': self._handle_status,
            'pause': self._handle_pause,
            'unpause': self._handle_unpause,
            'balance': self._handle_balance,
            'cleanup': self._handle_cleanup,
            'report': self._handle_report
        }
        
        # Set up event handlers
        @self.client.event
        async def on_ready():
            logger.info(f'Bot logged in as {self.client.user}')
            # Log available channels for debugging
            for guild in self.client.guilds:
                logger.info(f"Bot is in guild: {guild.name} (ID: {guild.id})")
                for channel in guild.channels:
                    logger.info(f"Channel: {channel.name} (ID: {channel.id})")
            self._is_ready = True
            self._ready_event.set()
            
        @self.client.event
        async def on_message(message):
            if message.author == self.client.user:
                return
                
            if message.content.startswith('/'):
                command = message.content[1:].split()[0].lower()
                args = message.content.split()[1:] if len(message.content.split()) > 1 else None
                
                try:
                    response = await self.handle_command(command, args)
                    await message.channel.send(response)
                except Exception as e:
                    logger.error(f"Error handling command: {str(e)}")
                    await message.channel.send(f"Error executing command: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the Discord service."""
        try:
            # Start the bot in the background
            asyncio.create_task(self.client.start(self.bot_token))
            
            # Wait for the bot to be ready
            try:
                await asyncio.wait_for(self._ready_event.wait(), timeout=30.0)
                logger.info("Discord bot is ready")
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for Discord bot to be ready")
                raise
                
        except Exception as e:
            logger.error(f"Error initializing Discord bot: {str(e)}")
            raise

    async def close(self) -> None:
        """Close the Discord service."""
        try:
            await self.client.close()
            self._is_ready = False
            self._ready_event.clear()
        except Exception as e:
            logger.error(f"Error closing Discord bot: {str(e)}")

    def is_initialized(self) -> bool:
        """Check if the service is initialized and ready."""
        return self._is_ready

    def is_trading_paused(self) -> bool:
        """Check if trading is paused."""
        return self._trading_paused

    async def handle_command(self, command: str, args: List[str] = None) -> str:
        """
        Handle Discord commands.
        
        Args:
            command (str): Command to handle
            args (List[str], optional): Command arguments
            
        Returns:
            str: Response message
        """
        command = command.lower()
        if command in self._command_handlers:
            try:
                return await self._command_handlers[command](args)
            except Exception as e:
                logger.error(f"Error handling command {command}: {str(e)}")
                return f"Error executing command: {str(e)}"
        else:
            return "Unknown command. Use /help to see available commands."

    async def _handle_start(self, args: List[str] = None) -> str:
        """Handle /start command."""
        self._trading_paused = False
        return "Bot started and subscribed to updates."

    async def _handle_help(self, args: List[str] = None) -> str:
        """Handle /help command."""
        help_text = (
            "Available commands:\n"
            "/start - Start the bot and subscribe to updates\n"
            "/help - Show this help message\n"
            "/status - Show current bot status\n"
            "/pause - Pause the bot\n"
            "/unpause - Unpause the bot\n"
            "/balance - Show current balance\n"
            "/cleanup - Clean up orders\n"
            "/report - Show detailed report"
        )
        return help_text

    async def _handle_status(self, args: List[str] = None) -> str:
        """Handle /status command."""
        status = "Active" if not self._trading_paused else "Paused"
        return f"Bot status: {status}"

    async def _handle_pause(self, args: List[str] = None) -> str:
        """Handle /pause command."""
        self._trading_paused = True
        return "Bot paused."

    async def _handle_unpause(self, args: List[str] = None) -> str:
        """Handle /unpause command."""
        self._trading_paused = False
        return "Bot unpaused."

    async def _handle_balance(self, args: List[str] = None) -> str:
        """Handle /balance command."""
        # This should be implemented to show actual balance
        return "Balance information not available."

    async def _handle_cleanup(self, args: List[str] = None) -> str:
        """Handle /cleanup command."""
        # This should be implemented to clean up orders
        return "Cleanup command received."

    async def _handle_report(self, args: List[str] = None) -> str:
        """Handle /report command."""
        # This should be implemented to show detailed report
        return "Detailed report not available."

    async def send_message(self, content: str, embed: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send a message to Discord channel.
        
        Args:
            content (str): Message content
            embed (Optional[Dict[str, Any]]): Optional embed data for rich messages
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        if not self._is_ready:
            logger.warning("Discord bot is not ready yet")
            return False
            
        try:
            # First try to get channel from cache
            channel = self.client.get_channel(self.channel_id)
            if not channel:
                logger.info(f"Channel {self.channel_id} not found in cache, fetching from Discord API...")
                # Try to fetch channel directly from Discord API
                try:
                    channel = await self.client.fetch_channel(self.channel_id)
                    if channel:
                        logger.info(f"Successfully fetched channel: {channel.name} (ID: {channel.id})")
                except Exception as e:
                    logger.error(f"Failed to fetch channel from Discord API: {str(e)}")
                    return False
            
            if channel:
                if embed:
                    discord_embed = discord.Embed.from_dict(embed)
                    await channel.send(content=content, embed=discord_embed)
                else:
                    logger.info(f"Sending message to Discord channel {channel.name}: {content}")
                    await channel.send(content=content)
                return True
            else:
                logger.error(f"Could not find or fetch Discord channel with ID {self.channel_id}")
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