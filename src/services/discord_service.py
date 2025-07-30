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
from discord.ext import commands
from src.services.base_notification_service import BaseNotificationService

logger = logging.getLogger(__name__)

class DiscordService(BaseNotificationService):
    """Service for handling Discord bot operations."""
    
    def __init__(self, config: Dict):
        """Initialize the service."""
        super().__init__(config)
        self._is_ready = False
        self._ready_event = asyncio.Event()
        self._is_closed = False
        self._is_running = False
        self._shutdown_event = asyncio.Event()
        self._event_loop = None
        self._subscribed_users = set()
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Configure intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.messages = True
        
        # Create bot instance
        self.bot = commands.Bot(command_prefix='/', intents=intents)
        
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
        @self.bot.event
        async def on_ready():
            logger.info(f'Bot logged in as {self.bot.user}')
            # Log available channels for debugging
            for guild in self.bot.guilds:
                logger.info(f"Bot is in guild: {guild.name} (ID: {guild.id})")
                for channel in guild.channels:
                    logger.info(f"Channel: {channel.name} (ID: {channel.id})")
            self._is_ready = True
            self._ready_event.set()
            
        @self.bot.event
        async def on_message(message):
            if message.author == self.bot.user:
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
            # Initialize base service first
            await super().initialize()
            
            # Check if bot token is available
            discord_config = self.config.get('api', {}).get('discord', {})
            bot_token = discord_config.get('bot_token')
            
            if not bot_token:
                logger.warning("Discord bot token not found, using webhook mode only")
                self._is_ready = True
                self._ready_event.set()
                return
            
            # Start the bot in the background only if not already running
            if not self._is_running:
                self._is_running = True
                asyncio.create_task(self.bot.start(bot_token))
            
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
            await self.bot.close()
            self._is_ready = False
            self._ready_event.clear()
            self._is_closed = True
        except Exception as e:
            logger.error(f"Error closing Discord bot: {str(e)}")

    def is_initialized(self) -> bool:
        """Check if the service is initialized and ready."""
        return self._is_ready

    def is_trading_paused(self) -> bool:
        """Check if trading is paused."""
        return super().is_trading_paused()

    async def wait_for_trading_resume(self) -> None:
        """Wait for trading to be resumed."""
        await super().wait_for_trading_resume()

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
        self.unpause_trading()
        return "Bot started and subscribed to updates."

    async def _handle_help(self, args: List[str] = None) -> str:
        """Handle /help command."""
        return self.help_content()

    async def _handle_status(self, args: List[str] = None) -> str:
        """Handle /status command."""
        status = await self._get_status_message()
        return f"Bot status: {status}"

    async def _handle_pause(self, args: List[str] = None) -> str:
        """Handle /pause command."""
        if not self._is_ready:
            return "Bot is not initialized. Please try again later."
        
        if self.is_trading_paused():
            return "Bot is already paused"
        
        self.pause_trading()
        return "⏸ Bot trading paused successfully\n⚠️ Note: Only new trades are paused. Trailing stops and position management will continue to work."

    async def _handle_unpause(self, args: List[str] = None) -> str:
        """Handle /unpause command."""
        if not self._is_ready:
            return "Bot is not initialized. Please try again later."
        
        if not self.is_trading_paused():
            return "Bot is not paused"
        
        self.unpause_trading()
        
        # Reset profit target tracking if strategy exists
        if hasattr(self.strategy, 'reset_profit_target'):
            await self.strategy.reset_profit_target()
            
        return "▶️ Bot trading resumed successfully"

    async def _handle_balance(self, ctx: commands.Context) -> str:
        """Handle the /balance command."""
        try:
            if not self._is_ready:
                return "Bot is not initialized. Please try again later."
            
            if not self.binance_service:
                return "Binance service not available"
            
            # Get balance report from base class
            return await self.generate_balance_report()
            
        except Exception as e:
            logger.error(f"Error handling balance command: {str(e)}")
            return "An error occurred while generating the balance."

    async def _handle_cleanup(self, args: List[str] = None) -> str:
        """Handle /cleanup command."""
        try:
            if not self._is_ready:
                return "Bot is not initialized. Please try again later."
            
            if not self.binance_service:
                return "Binance service not available"
            
            # Get cleanup message from base class
            return await self.cleanup_orders()
            
        except Exception as e:
            logger.error(f"Error handling cleanup command: {str(e)}")
            return "An error occurred while cleaning up orders."

    async def _handle_report(self, args: List[str] = None) -> str:
        """Handle /report command."""
        try:
            if not self._is_ready:
                return "Bot is not initialized. Please try again later."
            
            if not self.binance_service:
                return "Binance service not available"
            
            # Get report chunks from base class
            report_chunks = await self.generate_report()
            
            # Send each chunk
            for chunk in report_chunks:
                await self.send_message(chunk)
                await asyncio.sleep(0.5)  # Small delay between messages
            
            return "Report sent successfully."
            
        except Exception as e:
            logger.error(f"Error handling report command: {str(e)}")
            return "An error occurred while generating the report."

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
            channel_id = self.config['api']['discord']['channel_id']
            channel = self.bot.get_channel(channel_id)
            if not channel:
                logger.info(f"Channel {channel_id} not found in cache, fetching from Discord API...")
                # Try to fetch channel directly from Discord API
                try:
                    channel = await self.bot.fetch_channel(channel_id)
                except Exception as e:
                    logger.error(f"Failed to fetch channel from Discord API: {str(e)}")
                    return False
            
            if channel:
                if embed:
                    discord_embed = discord.Embed.from_dict(embed)
                    await channel.send(content=content, embed=discord_embed)
                else:
                    await channel.send(content=content)
                return True
            else:
                logger.error(f"Could not find or fetch Discord channel with ID {channel_id}")
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