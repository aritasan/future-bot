"""
Telegram service for sending notifications and receiving commands.
"""
import logging
from typing import Dict
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode
import asyncio
from datetime import datetime, timedelta
import telegram.error
from src.services.base_notification_service import BaseNotificationService

logger = logging.getLogger(__name__)

class TelegramService(BaseNotificationService):
    """Service for handling Telegram bot operations."""
    
    def __init__(self, config: Dict):
        """Initialize the service."""
        super().__init__(config)
        self.bot = None
        self.application = None
        self._is_initialized = False
        self._is_closed = False
        self._is_running = False
        self._is_paused = False
        self.binance_service = None
        self.last_balance_check = None
        self.balance_check_interval = timedelta(minutes=5)
        self.status_report_interval = timedelta(minutes=30)
        self.last_status_report = None
        self.polling_task = None
        self.status_task = None
        self._shutdown_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._event_loop = None
        self._subscribed_users = set()
        
    async def initialize(self) -> None:
        """Initialize the Telegram service."""
        try:
            # Initialize base service first
            await super().initialize()
            
            # Create bot and application
            self.bot = Bot(token=self.config['api']['telegram']['bot_token'])
            self.application = Application.builder().token(self.config['api']['telegram']['bot_token']).build()
            
            # Add command handlers
            self.application.add_handler(CommandHandler("start", self._handle_start_command))
            self.application.add_handler(CommandHandler("help", self._handle_help_command))
            self.application.add_handler(CommandHandler("status", self._handle_status_command))
            self.application.add_handler(CommandHandler("pause", self._handle_pause_command))
            self.application.add_handler(CommandHandler("unpause", self._handle_unpause_command))
            self.application.add_handler(CommandHandler("report", self._handle_report_command))
            self.application.add_handler(CommandHandler("balance", self._handle_balance_command))
            self.application.add_handler(CommandHandler("cleanup", self._handle_cleanup_command))

            # Initialize application
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            
            logger.info("Telegram service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Telegram service: {str(e)}")
            raise
            
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
            
    async def _handle_start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /start command."""
        try:
            user_id = update.effective_user.id
            self._subscribed_users.add(user_id)
            await update.message.reply_text(
                "Welcome to the Trading Bot! 🚀\n\n"
                "You are now subscribed to updates.\n"
                "Use /help to see available commands."
            )
        except Exception as e:
            logger.error(f"Error handling start command: {str(e)}")
            await update.message.reply_text("An error occurred. Please try again later.")

    async def _handle_help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /help command."""
        try:
            help_text = self.help_content()
            await update.message.reply_text(help_text, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Error handling help command: {str(e)}")
            await update.message.reply_text("An error occurred. Please try again later.")

    async def _handle_status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /status command."""
        try:
            status_message = await self._get_status_message()
            await update.message.reply_text(status_message, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Error handling status command: {str(e)}")
            await update.message.reply_text("An error occurred. Please try again later.")

    async def _handle_pause_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /pause command."""
        try:
            if not self._is_initialized:
                await update.message.reply_text("Bot is not initialized. Please try again later.")
                return
            
            if self._is_paused:
                await update.message.reply_text("Bot is already paused")
                return
            
            self._is_paused = True
            self._pause_event.set()  # Set the pause event
            await update.message.reply_text(
                "⏸ Bot trading paused successfully\n"
                "⚠️ Note: Only new trades are paused. Trailing stops and position management will continue to work."
            )
            logger.info("Bot trading paused via Telegram command - Only new trades are paused")
        except Exception as e:
            logger.error(f"Error handling pause command: {str(e)}")
            await update.message.reply_text("An error occurred while pausing the bot.")

    async def _handle_unpause_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /unpause command."""
        try:
            if not self._is_initialized:
                await update.message.reply_text("Bot is not initialized. Please try again later.")
                return
            
            if not self._is_paused:
                await update.message.reply_text("Bot is not paused")
                return
            
            self._is_paused = False
            self._pause_event.clear()  # Clear the pause event
            await update.message.reply_text("▶️ Bot trading resumed successfully")
            logger.info("Bot trading resumed via Telegram command")
        except Exception as e:
            logger.error(f"Error handling unpause command: {str(e)}")
            await update.message.reply_text("An error occurred while unpausing the bot.")

    async def _handle_report_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /report command."""
        try:
            if not self._is_initialized:
                await update.message.reply_text("Bot is not initialized. Please try again later.")
                return
            
            if not self.binance_service:
                await update.message.reply_text("Binance service not available")
                return
            
            # Get report chunks from base class
            report_chunks = await self.generate_report()
            
            # Send each chunk
            for chunk in report_chunks:
                # Convert markdown to HTML for Telegram
                chunk = chunk.replace('**', '<b>').replace('</b>', '</b>')
                await update.message.reply_text(chunk, parse_mode='HTML')
                await asyncio.sleep(0.5)  # Small delay between messages
            
        except Exception as e:
            logger.error(f"Error handling report command: {str(e)}")
            await update.message.reply_text("An error occurred while generating the report.")
    
    async def _handle_balance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /balance command."""
        try:
            if not self._is_initialized:
                await update.message.reply_text("Bot is not initialized. Please try again later.")
                return
            
            if not self.binance_service:
                await update.message.reply_text("Binance service not available")
                return
            
            # Get balance report from base class
            message = await self.generate_balance_report()
            
            # Convert markdown to HTML for Telegram
            message = message.replace('**', '<b>').replace('</b>', '</b>')
            
            # Send message
            await update.message.reply_text(message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error handling balance command: {str(e)}")
            await update.message.reply_text("An error occurred while generating the balance.")

    async def _handle_cleanup_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /cleanup command."""
        try:
            if not self._is_initialized:
                await update.message.reply_text("Bot is not initialized. Please try again later.")  

            message = await self.cleanup_orders()
            await update.message.reply_text(message, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Error handling cleanup command: {str(e)}")
            await update.message.reply_text("An error occurred while cleaning up orders.")


    async def _handle_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming Telegram commands."""
        try:
            if not self._is_initialized:
                await update.message.reply_text("Bot is not initialized. Please try again later.")
                return
            
            command = update.message.text.lower()
            
            if command == '/start':
                await self._handle_start_command(update, context)
            elif command == '/help':
                await self._handle_help_command(update, context)
            elif command == '/status':
                await self._handle_status_command(update, context)
            elif command == '/pause':
                await self._handle_pause_command(update, context)
            elif command == '/unpause':
                await self._handle_unpause_command(update, context)
            elif command == '/report':
                await self._handle_report_command(update, context)
            elif command == '/balance':
                await self._handle_balance_command(update, context)
            elif command == '/cleanup':
                await self._handle_cleanup_command(update, context)
            else:
                await update.message.reply_text(
                    "Unknown command. Use /help to see available commands."
                )
            
        except Exception as e:
            logger.error(f"Error handling command: {str(e)}")
            await update.message.reply_text(
                "An error occurred while processing your command. Please try again later."
            )

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
            
    async def run(self):
        """Run the Telegram service."""
        try:
            if not self._is_initialized:
                logger.error("Telegram service not initialized")
                return
            
            if self._is_running:
                logger.warning("Telegram service is already running")
                return
            
            self._is_running = True
            self._start_time = datetime.now()
            
            # Create a new event loop if needed
            if not self._event_loop:
                self._event_loop = asyncio.get_event_loop()
            
            # Start polling for updates with retry mechanism
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Initialize the application first
                    await self.application.initialize()
                    await self.application.start()
                    
                    # Start polling
                    self.polling_task = asyncio.create_task(self.application.run_polling(
                        allowed_updates=Update.ALL_TYPES,
                        drop_pending_updates=True,
                        timeout=30
                    ))
                    logger.info("Started polling for updates")
                    break
                except telegram.error.Conflict as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(f"Conflict detected, retrying ({retry_count}/{max_retries})...")
                        await asyncio.sleep(2)
                    else:
                        logger.error("Max retries reached for polling")
                        raise
                except Exception as e:
                    logger.error(f"Error during polling: {str(e)}")
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(f"Error detected, retrying ({retry_count}/{max_retries})...")
                        await asyncio.sleep(2)
                    else:
                        logger.error("Max retries reached for polling")
                        raise
            
            # Start sending status updates
            self.status_task = asyncio.create_task(self._send_status_updates())
            logger.info("Started sending status updates")
            
            # Wait for both tasks to complete
            try:
                await asyncio.gather(self.polling_task, self.status_task)
            except asyncio.CancelledError:
                logger.info("Telegram service tasks cancelled")
                # Cancel tasks on cancellation
                if self.polling_task and not self.polling_task.done():
                    self.polling_task.cancel()
                if self.status_task and not self.status_task.done():
                    self.status_task.cancel()
            except Exception as e:
                logger.error(f"Error running Telegram service: {str(e)}")
                # Cancel tasks on error
                if self.polling_task and not self.polling_task.done():
                    self.polling_task.cancel()
                if self.status_task and not self.status_task.done():
                    self.status_task.cancel()
            
        except Exception as e:
            logger.error(f"Error running Telegram service: {str(e)}")
            self._is_running = False
            raise
            
    async def _send_status_updates(self):
        """Send periodic status updates to Telegram."""
        while self._is_running and not self._is_closed:
            try:
                if not self._is_initialized:
                    logger.error("Telegram service not initialized")
                    return
                    
                # Get current status
                status = self.get_status()
                
                # Format status message
                message = (
                    f"📊 Trading Bot Status\n\n"
                    f"🔄 Last Update: {status['last_update']}\n"
                    f"📈 Active Trades: {status['active_trades']}\n"
                    f"💰 Total P&L: {status['total_pnl']}\n"
                    f"📉 Max Drawdown: {status['max_drawdown']}\n"
                    f"⚡ Risk Level: {status['risk_level']}\n"
                    f"🔔 Alerts: {status['alerts']}\n"
                )
                
                # Send status to all subscribed users
                for user_id in self._subscribed_users:
                    try:
                        await self.bot.send_message(
                            chat_id=user_id,
                            text=message,
                            parse_mode='HTML'
                        )
                    except Exception as e:
                        logger.error(f"Error sending status to user {user_id}: {str(e)}")
                
                # Wait for next update
                await asyncio.sleep(60)  # Update every minute
                
            except asyncio.CancelledError:
                logger.info("Status updates task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in status updates: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying
            
    async def _send_status_report(self):
        """Send a detailed status report."""
        try:
            if not self.binance_service:
                return
                
            # Get balance
            balance = await self.binance_service.get_account_balance()
            if not balance:
                return
                
            # Get positions
            positions = await self.binance_service.get_positions()
            
            # Calculate total PnL
            total_pnl = 0.0
            if positions:
                for pos in positions:
                    if pos and isinstance(pos, dict):
                        unrealized_pnl = pos.get('unrealized_pnl')
                        if unrealized_pnl is not None:
                            try:
                                total_pnl += float(unrealized_pnl)
                            except (ValueError, TypeError):
                                pass
                        
            # Format message
            message = (
                "📊 <b>Status Report</b>\n\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Paused: {'Yes' if self._is_paused else 'No'}\n\n"
                "💰 <b>Balance</b>\n"
            )
            
            # Add balance information
            for asset, data in balance.items():
                if isinstance(data, dict):
                    amount = data.get('total')
                else:
                    amount = data
                    
                if amount is not None:
                    try:
                        amount_float = float(amount)
                        if amount_float > 0:
                            message += f"{asset}: {amount_float}\n"
                    except (ValueError, TypeError):
                        continue
                    
            # Add positions information
            if positions:
                message += "\n📈 <b>Open Positions</b>\n"
                for pos in positions:
                    if pos and isinstance(pos, dict):
                        size = pos.get('size')
                        entry_price = pos.get('entry_price')
                        side = pos.get('side')
                        unrealized_pnl = pos.get('unrealized_pnl')
                        
                        if size and float(size) > 0:
                            message += (
                                f"\nSymbol: {pos.get('symbol', 'N/A')}\n"
                                f"Size: {size}\n"
                                f"Entry Price: {entry_price if entry_price else 'N/A'}\n"
                                f"Side: {side if side else 'N/A'}\n"
                                f"Unrealized PnL: {unrealized_pnl if unrealized_pnl else 'N/A'}\n"
                            )
                        
            message += f"\n💵 Total Unrealized PnL: {total_pnl:.2f} USDT"
            
            await self.send_message(message)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error sending status report: {str(e)}")
            
    async def close(self) -> None:
        """Close the Telegram service."""
        try:
            if not self._is_running:
                return
                
            self._is_running = False
            
            # Cancel all tasks
            if self.polling_task and not self.polling_task.done():
                self.polling_task.cancel()
                try:
                    await self.polling_task
                except asyncio.CancelledError:
                    pass
                
            if self.status_task and not self.status_task.done():
                self.status_task.cancel()
                try:
                    await self.status_task
                except asyncio.CancelledError:
                    pass

            # Shutdown the application
            if self.application:
                try:
                    # Stop the updater first
                    if self.application.updater:
                        await self.application.updater.stop()
                        # Clear the update queue and bot instance
                        self.application.updater._update_queue = None
                        self.application.updater.bot = None
                    
                    # Then stop the application
                    await self.application.stop()
                    await self.application.shutdown()
                    
                    # Clear the bot instance and application
                    self.application.bot = None
                    self.application = None
                    
                    # Give time for pending operations to complete
                    await asyncio.sleep(2)
                except Exception as e:
                    logger.error(f"Error during application shutdown: {str(e)}")
                    
            # Reset state
            self._start_time = None
            self.polling_task = None
            self.status_task = None
            self._is_initialized = False
            self._is_closed = True
            
            logger.info("Telegram service closed successfully")
        except Exception as e:
            logger.error(f"Error closing Telegram service: {str(e)}")
            raise

    def is_trading_paused(self) -> bool:
        """Check if trading is paused.
        
        Returns:
            bool: True if trading is paused, False otherwise
        """
        return self._is_paused

    async def wait_for_trading_resume(self) -> None:
        """Wait for trading to be resumed.
        
        This method will block until trading is resumed or the service is closed.
        """
        if not self._is_paused:
            return
        
        try:
            await asyncio.wait_for(self._pause_event.wait(), timeout=None)
        except asyncio.CancelledError:
            logger.info("Wait for trading resume cancelled")
            raise
        except Exception as e:
            logger.error(f"Error waiting for trading resume: {str(e)}")
            raise
    