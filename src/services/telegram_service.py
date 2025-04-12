"""
Telegram service for sending notifications and alerts.
"""
from typing import Dict, Any
import logging
import telegram
from telegram.ext import Application, CommandHandler, ContextTypes
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class TelegramService:
    """
    Service for handling Telegram notifications and commands.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Telegram service.
        
        Args:
            config: Configuration dictionary containing Telegram settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.bot = None
        self.application = None
        self.chat_id = config['api']['telegram']['chat_id']
        self.last_balance_check = None
        self.last_positions_check = None
        self.token = config['telegram']['bot_token']
        self.balance_check_task = None
        self.is_running = True
        self.binance_service = None  # Will be set after initialization
        
    async def initialize(self) -> bool:
        """
        Initialize the Telegram bot and set up command handlers.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Initialize bot with token
            self.application = Application.builder().token(self.token).build()
            
            # Add command handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CommandHandler("status", self.status_command))
            self.application.add_handler(CommandHandler("balance", self.balance_command))
            self.application.add_handler(CommandHandler("positions", self.positions_command))
            self.application.add_handler(CommandHandler("stop", self.stop_command))
            self.application.add_handler(CommandHandler("start_trading", self.start_trading_command))
            self.application.add_handler(CommandHandler("stop_trading", self.stop_trading_command))
            
            # Start the bot
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            
            # Send startup message
            await self.send_startup_message()
            
            # Start balance check task
            self.balance_check_task = asyncio.create_task(self.periodic_balance_check())
            
            return True
        except Exception as e:
            self.logger.error(f"Error initializing Telegram service: {e}")
            return False
            
    async def close(self) -> None:
        """Close the Telegram service."""
        try:
            if self.application and self.application.updater.running:
                await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
            logger.info("Telegram service closed")
        except Exception as e:
            logger.error(f"Error closing Telegram service: {e}")
            
    async def send_message(self, text: str) -> bool:
        """Send message to Telegram chat.
        
        Args:
            text: Message text
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        try:
            if not self.application or not self.application.bot:
                self.logger.error("Telegram bot is not initialized")
                return False
            
            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode='Markdown'
            )
            return True
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {str(e)}")
            return False
            
    async def send_trade_notification(self, symbol: str, side: str, size: float, 
                                    price: float, stop_loss: float, take_profit: float) -> bool:
        """
        Send a trade notification.
        
        Args:
            symbol: Trading pair symbol
            side: Trade side (buy/sell)
            size: Position size
            price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        try:
            message = (
                f"🔄 New Trade Alert\n"
                f"Symbol: {symbol}\n"
                f"Side: {side.upper()}\n"
                f"Size: {size}\n"
                f"Entry: {price}\n"
                f"Stop Loss: {stop_loss}\n"
                f"Take Profit: {take_profit}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return await self.send_message(message)
        except Exception as e:
            self.logger.error(f"Error sending trade notification: {e}")
            return False
            
    async def send_position_update(self, symbol: str, unrealized_pnl: float, 
                                 current_price: float) -> bool:
        """
        Send a position update notification.
        
        Args:
            symbol: Trading pair symbol
            unrealized_pnl: Unrealized P&L
            current_price: Current market price
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        try:
            message = (
                f"📊 Position Update\n"
                f"Symbol: {symbol}\n"
                f"Unrealized P&L: {unrealized_pnl}\n"
                f"Current Price: {current_price}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return await self.send_message(message)
        except Exception as e:
            self.logger.error(f"Error sending position update: {e}")
            return False
            
    async def send_error_notification(self, error_message: str) -> bool:
        """
        Send an error notification.
        
        Args:
            error_message: Error message to send
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        try:
            message = (
                f"⚠️ Error Alert\n"
                f"Message: {error_message}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return await self.send_message(message)
        except Exception as e:
            self.logger.error(f"Error sending error notification: {e}")
            return False
            
    async def send_balance_update(self, balance: float) -> bool:
        """
        Send a balance update notification.
        
        Args:
            balance: Current account balance
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        try:
            message = (
                f"💰 Balance Update\n"
                f"Current Balance: {balance} USDT\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return await self.send_message(message)
        except Exception as e:
            self.logger.error(f"Error sending balance update: {e}")
            return False
            
    async def start_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        try:
            message = (
                "🤖 *Welcome to Trading Bot*\n\n"
                "Available commands:\n"
                "/help - Show all commands\n"
                "/status - Check bot status\n"
                "/balance - Check account balance\n"
                "/positions - View open positions\n"
                "/start_trading - Start trading\n"
                "/stop_trading - Stop trading\n"
                "/stop - Stop the bot"
            )
            await update.message.reply_text(message, parse_mode='Markdown')
        except Exception as e:
            self.logger.error(f"Error handling start command: {str(e)}")
            
    async def help_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        try:
            message = (
                "📚 *Available Commands*\n\n"
                "🤖 Bot Control:\n"
                "/start - Start the bot\n"
                "/stop - Stop the bot\n"
                "/status - Check bot status\n\n"
                "💰 Trading:\n"
                "/start_trading - Start trading\n"
                "/stop_trading - Stop trading\n"
                "/balance - Check account balance\n"
                "/positions - View open positions"
            )
            await update.message.reply_text(message, parse_mode='Markdown')
        except Exception as e:
            self.logger.error(f"Error handling help command: {str(e)}")
            
    async def status_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        try:
            status = "🟢 Running" if self.is_running else "🔴 Stopped"
            trading_status = "🟢 Active" if self.binance_service and self.binance_service.is_trading else "🔴 Inactive"
            
            message = (
                "📊 *Bot Status*\n\n"
                f"Bot: {status}\n"
                f"Trading: {trading_status}\n"
                f"Last Update: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
            )
            await update.message.reply_text(message, parse_mode='Markdown')
        except Exception as e:
            self.logger.error(f"Error handling status command: {str(e)}")
            
    async def balance_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command."""
        try:
            if not self.binance_service:
                await update.message.reply_text("❌ Binance service not initialized")
                return
                
            balance = await self.get_current_balance()
            if balance is not None:
                message = (
                    "💰 *Account Balance*\n\n"
                    f"Current Balance: `{balance:.2f} USDT`\n"
                    f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
                )
                await update.message.reply_text(message, parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ Failed to fetch balance")
        except Exception as e:
            self.logger.error(f"Error handling balance command: {str(e)}")
            
    async def positions_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command."""
        try:
            if not self.binance_service:
                await update.message.reply_text("❌ Binance service not initialized")
                return
                
            positions = await self.binance_service.get_positions()
            if positions:
                message = "📊 *Open Positions*\n\n"
                for symbol, pos in positions.items():
                    message += (
                        f"Symbol: `{symbol}`\n"
                        f"Side: `{pos['side']}`\n"
                        f"Size: `{pos['size']}`\n"
                        f"Entry: `{pos['entry_price']}`\n"
                        f"P&L: `{pos['unrealized_pnl']}`\n\n"
                    )
                await update.message.reply_text(message, parse_mode='Markdown')
            else:
                await update.message.reply_text("No open positions")
        except Exception as e:
            self.logger.error(f"Error handling positions command: {str(e)}")
            
    async def stop_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command."""
        try:
            await update.message.reply_text("🛑 Stopping bot...")
            self.is_running = False
            await self.close()
        except Exception as e:
            self.logger.error(f"Error handling stop command: {str(e)}")
            
    async def start_trading_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start_trading command."""
        try:
            if not self.binance_service:
                await update.message.reply_text("❌ Binance service not initialized")
                return
                
            if self.binance_service.start_trading():
                await update.message.reply_text("✅ Trading started successfully")
            else:
                await update.message.reply_text("❌ Failed to start trading")
        except Exception as e:
            self.logger.error(f"Error handling start_trading command: {str(e)}")
            
    async def stop_trading_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop_trading command."""
        try:
            if not self.binance_service:
                await update.message.reply_text("❌ Binance service not initialized")
                return
                
            if self.binance_service.stop_trading():
                await update.message.reply_text("✅ Trading stopped successfully")
            else:
                await update.message.reply_text("❌ Failed to stop trading")
        except Exception as e:
            self.logger.error(f"Error handling stop_trading command: {str(e)}")
            
    def set_binance_service(self, binance_service):
        """Set the Binance service instance.
        
        Args:
            binance_service: BinanceService instance
        """
        self.binance_service = binance_service
            
    async def send_startup_message(self) -> bool:
        """Send startup notification.
        
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        try:
            # Get current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Format message with emojis and markdown
            message = (
                "🤖 *Trading Bot Started Successfully*\n\n"
                f"⏰ Time: `{current_time}`\n"
                "📊 Bot is now running and will monitor:\n"
                "• Trading signals\n"
                "• Market conditions\n"
                "• Risk management\n\n"
                "🔔 You will receive notifications for:\n"
                "• New orders placed\n"
                "• Balance updates (every 10 minutes)\n"
                "• Bot status changes\n\n"
                "🛑 Use /stop to stop the bot\n"
                "ℹ️ Use /help for more commands"
            )
            
            # Send message
            await self.send_message(message)
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending startup message: {str(e)}")
            return False 