"""
Telegram service for sending notifications and receiving commands.
"""
import logging
from typing import Dict, Optional
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TelegramService:
    """Service for handling Telegram bot operations."""
    
    def __init__(self, config: Dict):
        """Initialize the service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.bot = None
        self.application = None
        self._is_initialized = False
        self._is_closed = False
        self._is_running = False
        self.binance_service = None
        self.last_balance_check = None
        self.balance_check_interval = timedelta(minutes=5)
        
    async def initialize(self) -> bool:
        """Initialize the Telegram service."""
        try:
            if self._is_initialized:
                logger.warning("Telegram service already initialized")
                return True
                
            # Initialize bot and application
            self.bot = Bot(token=self.config['api']['telegram']['bot_token'])
            self.application = Application.builder().token(self.config['api']['telegram']['bot_token']).build()
            
            # Add command handlers
            self.application.add_handler(CommandHandler("start", self._handle_start))
            self.application.add_handler(CommandHandler("help", self._handle_help))
            self.application.add_handler(CommandHandler("status", self._handle_status))
            self.application.add_handler(CommandHandler("balance", self._handle_balance))
            self.application.add_handler(CommandHandler("positions", self._handle_positions))
            
            # Initialize the application
            await self.application.initialize()
            
            self._is_initialized = True
            logger.info("Telegram service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Telegram service: {str(e)}")
            return False
            
    def set_binance_service(self, binance_service):
        """Set the Binance service for balance and position checks.
        
        Args:
            binance_service: Binance service instance
        """
        self.binance_service = binance_service
        
    async def send_message(self, message: str) -> bool:
        """Send a message to the configured chat.
        
        Args:
            message: Message to send
            
        Returns:
            bool: True if message sent successfully, False otherwise
        """
        try:
            if not self._is_initialized or self._is_closed:
                logger.error("Telegram service not initialized or closed")
                return False
                
            await self.bot.send_message(
                chat_id=self.config['api']['telegram']['chat_id'],
                text=message,
                parse_mode=ParseMode.HTML
            )
            return True
            
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            return False
            
    async def send_order_notification(self, order: Dict, signals: Optional[Dict] = None) -> bool:
        """Send an order notification.
        
        Args:
            order: Order details
            signals: Trading signals containing SL/TP information
            
        Returns:
            bool: True if notification sent successfully, False otherwise
        """
        try:
            if not self._is_initialized or self._is_closed:
                logger.error("Telegram service not initialized or closed")
                return False
                
            message = (
                f"📊 <b>New Order</b>\n\n"
                f"Symbol: {order['symbol']}\n"
                f"Side: {order['side']}\n"
                f"Size: {order['amount']}\n"
                f"Price: {order['price']}\n"
                f"Type: {order['type']}"
            )
            
            # Add SL/TP information from signals if available
            if signals:
                if 'stop_loss' in signals:
                    sl_price = signals['stop_loss']
                    sl_percent = abs((sl_price - order['price']) / order['price'] * 100)
                    message += f"\nStop Loss: {sl_price} ({sl_percent:.2f}%)"
                if 'take_profit' in signals:
                    tp_price = signals['take_profit']
                    tp_percent = abs((tp_price - order['price']) / order['price'] * 100)
                    message += f"\nTake Profit: {tp_price} ({tp_percent:.2f}%)"
                
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending order notification: {str(e)}")
            return False
            
    async def send_startup_notification(self) -> bool:
        """Send bot startup notification.
        
        Returns:
            bool: True if notification sent successfully, False otherwise
        """
        try:
            if not self._is_initialized or self._is_closed:
                logger.error("Telegram service not initialized or closed")
                return False
                
            message = (
                "🤖 <b>Trading Bot Started</b>\n\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                "Status: Running"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending startup notification: {str(e)}")
            return False
            
    async def send_shutdown_notification(self) -> bool:
        """Send bot shutdown notification.
        
        Returns:
            bool: True if notification sent successfully, False otherwise
        """
        try:
            if not self._is_initialized or self._is_closed:
                logger.error("Telegram service not initialized or closed")
                return False
                
            message = (
                "🤖 <b>Trading Bot Stopped</b>\n\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                "Status: Stopped"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending shutdown notification: {str(e)}")
            return False
            
    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        try:
            if not self._is_initialized or self._is_closed:
                await update.message.reply_text("❌ Bot is not ready")
                return
                
            await update.message.reply_text(
                "🤖 Welcome to the Trading Bot!\n\n"
                "Use /help to see available commands."
            )
        except Exception as e:
            logger.error(f"Error handling start command: {str(e)}")
            
    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        try:
            if not self._is_initialized or self._is_closed:
                await update.message.reply_text("❌ Bot is not ready")
                return
                
            await update.message.reply_text(
                "📚 <b>Available Commands:</b>\n\n"
                "/start - Start the bot\n"
                "/help - Show this help message\n"
                "/status - Check bot status\n"
                "/balance - Check account balance\n"
                "/positions - Check open positions"
            )
        except Exception as e:
            logger.error(f"Error handling help command: {str(e)}")
            
    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        try:
            if not self._is_initialized or self._is_closed:
                await update.message.reply_text("❌ Bot is not ready")
                return
                
            status = "🟢 Running" if self._is_running else "🔴 Stopped"
            await update.message.reply_text(f"Bot Status: {status}")
        except Exception as e:
            logger.error(f"Error handling status command: {str(e)}")
            
    async def _handle_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /balance command."""
        try:
            if not self._is_initialized or self._is_closed:
                await update.message.reply_text("❌ Bot is not ready")
                return
                
            if not self.binance_service:
                await update.message.reply_text("❌ Binance service not connected")
                return
                
            balance = await self.binance_service.get_account_balance()
            if not balance:
                await update.message.reply_text("❌ Failed to get balance")
                return
                
            message = "💰 <b>Account Balance</b>\n\n"
            for asset, data in balance.items():
                if isinstance(data, dict):
                    amount = data.get('total', 0)
                else:
                    amount = data
                    
                if float(amount) > 0:
                    message += f"{asset}: {amount}\n"
                    
            await update.message.reply_text(message)
        except Exception as e:
            logger.error(f"Error handling balance command: {str(e)}")
            
    async def _handle_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /positions command."""
        try:
            if not self._is_initialized or self._is_closed:
                await update.message.reply_text("❌ Bot is not ready")
                return
                
            if not self.binance_service:
                await update.message.reply_text("❌ Binance service not connected")
                return
                
            positions = await self.binance_service.get_positions()
            if not positions:
                await update.message.reply_text("No open positions")
                return
                
            message = "📊 <b>Open Positions</b>\n\n"
            for symbol, position in positions.items():
                message += (
                    f"Symbol: {symbol}\n"
                    f"Size: {position['size']}\n"
                    f"Entry Price: {position['entry_price']}\n"
                    f"Side: {position['side']}\n"
                    f"Unrealized PnL: {position['unrealized_pnl']}\n\n"
                )
                
            await update.message.reply_text(message)
        except Exception as e:
            logger.error(f"Error handling positions command: {str(e)}")
            
    async def periodic_balance_check(self) -> None:
        """Periodically check and send balance updates."""
        try:
            while self._is_running and not self._is_closed:
                if not self.binance_service:
                    await asyncio.sleep(60)
                    continue
                    
                current_time = datetime.now()
                if (self.last_balance_check is None or 
                    current_time - self.last_balance_check >= self.balance_check_interval):
                    
                    balance = await self.binance_service.get_account_balance()
                    if balance:
                        message = "💰 <b>Periodic Balance Update</b>\n\n"
                        for asset, data in balance.items():
                            if isinstance(data, dict):
                                amount = data.get('total', 0)
                            else:
                                amount = data
                                
                            if amount and float(amount) > 0:  # Add check for None
                                message += f"{asset}: {amount}\n"
                                
                        if message != "💰 <b>Periodic Balance Update</b>\n\n":  # Only send if we have balances
                            await self.send_message(message)
                            self.last_balance_check = current_time
                            
                await asyncio.sleep(60)
                
        except Exception as e:
            logger.error(f"Error in periodic balance check: {str(e)}")
            
    async def run(self) -> None:
        """Run the Telegram service."""
        try:
            if not self._is_initialized:
                logger.error("Telegram service not initialized")
                return
                
            if self._is_closed:
                logger.error("Telegram service is closed")
                return
                
            self._is_running = True
            
            # Initialize the application if not already initialized
            if not self.application.running:
                await self.application.initialize()
                await self.application.start()
            
            # Run the application in a separate task
            polling_task = asyncio.create_task(self.application.run_polling())
            
            # Wait for the polling task to complete
            try:
                await polling_task
            except asyncio.CancelledError:
                logger.info("Telegram polling task was cancelled")
            except Exception as e:
                logger.error(f"Error in Telegram polling task: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error running Telegram service: {str(e)}")
            # Ensure proper cleanup on error
            if self.application:
                try:
                    await self.application.stop()
                    await self.application.shutdown()
                except Exception as shutdown_error:
                    logger.error(f"Error during shutdown: {str(shutdown_error)}")
        finally:
            self._is_running = False
            
    async def close(self) -> None:
        """Close the Telegram service."""
        try:
            if not self._is_initialized:
                logger.warning("Telegram service was not initialized")
                return
                
            if self._is_closed:
                logger.warning("Telegram service already closed")
                return
                
            self._is_running = False
            
            # Stop the application
            if self.application:
                try:
                    # Stop the application and wait for it to complete
                    await self.application.stop()
                    await self.application.shutdown()
                    # Wait for any pending tasks to complete
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Error stopping application: {str(e)}")
                    
            self._is_closed = True
            logger.info("Telegram service closed")
            
        except Exception as e:
            logger.error(f"Error closing Telegram service: {str(e)}")
            # Ensure we mark as closed even if there's an error
            self._is_closed = True 