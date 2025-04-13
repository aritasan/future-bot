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
        """Initialize the service."""
        self.config = config
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
        self._event_loop = None
        self._subscribed_users = set()
        
    async def initialize(self) -> None:
        """Initialize the Telegram service."""
        try:
            # Create bot and application
            self.bot = Bot(token=self.config['api']['telegram']['bot_token'])
            self.application = Application.builder().token(self.config['api']['telegram']['bot_token']).build()
            
            # Add command handlers
            self.application.add_handler(CommandHandler("start", self._handle_start_command))
            self.application.add_handler(CommandHandler("help", self._handle_help_command))
            self.application.add_handler(CommandHandler("status", self._handle_status_command))
            self.application.add_handler(CommandHandler("subscribe", self._handle_subscribe_command))
            self.application.add_handler(CommandHandler("unsubscribe", self._handle_unsubscribe_command))
            self.application.add_handler(CommandHandler("settings", self._handle_settings_command))
            self.application.add_handler(CommandHandler("trades", self._handle_trades_command))
            self.application.add_handler(CommandHandler("risk", self._handle_risk_command))
            self.application.add_handler(CommandHandler("alerts", self._handle_alerts_command))
            
            # Initialize application
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            
            self._is_initialized = True
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
            help_text = (
                "🤖 <b>Available Commands</b>\n\n"
                "/start - Start the bot and subscribe to updates\n"
                "/help - Show this help message\n"
                "/status - Show current bot status\n"
                "/subscribe - Subscribe to updates\n"
                "/unsubscribe - Unsubscribe from updates\n"
                "/settings - Show current settings\n"
                "/trades - Show active trades\n"
                "/risk - Show current risk level\n"
                "/alerts - Show recent alerts"
            )
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

    async def _handle_subscribe_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /subscribe command."""
        try:
            user_id = update.effective_user.id
            self._subscribed_users.add(user_id)
            await update.message.reply_text("You are now subscribed to updates! ✅")
        except Exception as e:
            logger.error(f"Error handling subscribe command: {str(e)}")
            await update.message.reply_text("An error occurred. Please try again later.")

    async def _handle_unsubscribe_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /unsubscribe command."""
        try:
            user_id = update.effective_user.id
            self._subscribed_users.discard(user_id)
            await update.message.reply_text("You are now unsubscribed from updates. ❌")
        except Exception as e:
            logger.error(f"Error handling unsubscribe command: {str(e)}")
            await update.message.reply_text("An error occurred. Please try again later.")

    async def _handle_settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /settings command."""
        try:
            settings_message = self._get_settings_message()
            await update.message.reply_text(settings_message, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Error handling settings command: {str(e)}")
            await update.message.reply_text("An error occurred. Please try again later.")

    async def _handle_trades_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /trades command."""
        try:
            trades_message = self._get_active_trades_message()
            await update.message.reply_text(trades_message, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Error handling trades command: {str(e)}")
            await update.message.reply_text("An error occurred. Please try again later.")

    async def _handle_risk_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /risk command."""
        try:
            risk_message = self._get_risk_message()
            await update.message.reply_text(risk_message, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Error handling risk command: {str(e)}")
            await update.message.reply_text("An error occurred. Please try again later.")

    async def _handle_alerts_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /alerts command."""
        try:
            alerts_message = self._get_alerts_message()
            await update.message.reply_text(alerts_message, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Error handling alerts command: {str(e)}")
            await update.message.reply_text("An error occurred. Please try again later.")

    async def _get_status_message(self) -> str:
        """Get the current bot status message."""
        try:
            status = "🟢 Running" if self._is_running else "🔴 Stopped"
            uptime = str(datetime.now() - self._start_time).split('.')[0] if self._start_time else "N/A"
            
            message = (
                f"🤖 <b>Bot Status</b>\n\n"
                f"Status: {status}\n"
                f"Uptime: {uptime}\n"
                f"Subscribed Users: {len(self._subscribed_users)}\n"
                f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return message
        except Exception as e:
            logger.error(f"Error getting status message: {str(e)}")
            return "Error getting status information."

    def _get_settings_message(self) -> str:
        """Get the current settings message."""
        try:
            message = (
                "⚙️ <b>Current Settings</b>\n\n"
                f"Risk Level: {self.risk_manager.risk_level}\n"
                f"Max Position Size: {self.risk_manager.max_position_size}\n"
                f"Max Open Trades: {self.risk_manager.max_open_trades}\n"
                f"Stop Loss: {self.risk_manager.stop_loss_percentage}%\n"
                f"Take Profit: {self.risk_manager.take_profit_percentage}%"
            )
            return message
        except Exception as e:
            logger.error(f"Error getting settings message: {str(e)}")
            return "Error getting settings information."

    def _get_active_trades_message(self) -> str:
        """Get the active trades message."""
        try:
            active_trades = self.risk_manager.get_active_trades()
            if not active_trades:
                return "No active trades at the moment."
            
            message = "📊 <b>Active Trades</b>\n\n"
            for trade in active_trades:
                message += (
                    f"Symbol: {trade['symbol']}\n"
                    f"Side: {trade['side']}\n"
                    f"Entry Price: {trade['entry_price']}\n"
                    f"Current Price: {trade['current_price']}\n"
                    f"P&L: {trade['pnl']}%\n"
                    f"Stop Loss: {trade['stop_loss']}\n"
                    f"Take Profit: {trade['take_profit']}\n\n"
                )
            return message
        except Exception as e:
            logger.error(f"Error getting active trades message: {str(e)}")
            return "Error getting active trades information."

    def _get_risk_message(self) -> str:
        """Get the current risk level message."""
        try:
            risk_level = self.risk_manager.risk_level
            position_size = self.risk_manager.max_position_size
            open_trades = self.risk_manager.get_active_trades_count()
            max_trades = self.risk_manager.max_open_trades
            
            message = (
                "⚠️ <b>Risk Information</b>\n\n"
                f"Current Risk Level: {risk_level}\n"
                f"Max Position Size: {position_size}\n"
                f"Open Trades: {open_trades}/{max_trades}\n"
                f"Available Margin: {self.risk_manager.get_available_margin()}\n"
                f"Total Exposure: {self.risk_manager.get_total_exposure()}"
            )
            return message
        except Exception as e:
            logger.error(f"Error getting risk message: {str(e)}")
            return "Error getting risk information."

    def _get_alerts_message(self) -> str:
        """Get the recent alerts message."""
        try:
            alerts = self.risk_manager.get_recent_alerts()
            if not alerts:
                return "No recent alerts."
            
            message = "🔔 <b>Recent Alerts</b>\n\n"
            for alert in alerts:
                message += (
                    f"Time: {alert['timestamp']}\n"
                    f"Type: {alert['type']}\n"
                    f"Message: {alert['message']}\n\n"
                )
            return message
        except Exception as e:
            logger.error(f"Error getting alerts message: {str(e)}")
            return "Error getting alerts information."

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
                
            self._is_running = True
            self._start_time = datetime.now()
            
            # Start polling for updates
            self.polling_task = asyncio.create_task(self.application.run_polling())
            logger.info("Started polling for updates")
            
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
                f"Status: {'🟢 Running' if self._is_running else '🔴 Stopped'}\n"
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
            
    async def _handle_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /pause command."""
        try:
            if not self._is_initialized or self._is_closed:
                await update.message.reply_text("❌ Bot is not ready")
                return
                
            self._is_paused = True
            await update.message.reply_text("⏸ Bot paused")
            
        except Exception as e:
            logger.error(f"Error handling pause command: {str(e)}")
            
    async def _handle_unpause(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /unpause command."""
        try:
            if not self._is_initialized or self._is_closed:
                await update.message.reply_text("❌ Bot is not ready")
                return
                
            self._is_paused = False
            await update.message.reply_text("▶️ Bot unpaused")
            
        except Exception as e:
            logger.error(f"Error handling unpause command: {str(e)}")
            
    async def _handle_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /report command."""
        try:
            if not self._is_initialized or self._is_closed:
                await update.message.reply_text("❌ Bot is not ready")
                return
                
            await self._send_status_report()
            
        except Exception as e:
            logger.error(f"Error handling report command: {str(e)}")
            
    async def close(self) -> None:
        """Close the Telegram service."""
        try:
            if not self._is_running:
                logger.warning("Telegram service is not running")
                return
                
            self._is_running = False
            
            # Cancel all tasks
            if self.polling_task and not self.polling_task.done():
                self.polling_task.cancel()
            if self.status_task and not self.status_task.done():
                self.status_task.cancel()

            # Wait for tasks to complete
            try:
                if self.polling_task and not self.polling_task.done():
                    await self.polling_task
                if self.status_task and not self.status_task.done():
                    await self.status_task
            except asyncio.CancelledError:
                pass

            # Shutdown the application
            if self.application:
                try:
                    # Stop the updater first
                    if self.application.updater:
                        await self.application.updater.stop()
                    
                    # Then stop the application
                    await self.application.stop()
                    await self.application.shutdown()
                    
                    # Give time for pending operations to complete
                    await asyncio.sleep(2)  # Increased delay to 2 seconds
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
            
    async def _handle_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming Telegram commands."""
        try:
            if not self._is_initialized:
                await update.message.reply_text("Bot is not initialized. Please try again later.")
                return
            
            command = update.message.text.lower()
            user_id = update.effective_user.id
            
            if command == '/start':
                await self._handle_start_command(update, context)
            elif command == '/help':
                await self._handle_help_command(update, context)
            elif command == '/status':
                await self._handle_status_command(update, context)
            elif command == '/subscribe':
                await self._handle_subscribe_command(update, context)
            elif command == '/unsubscribe':
                await self._handle_unsubscribe_command(update, context)
            elif command == '/settings':
                await self._handle_settings_command(update, context)
            elif command == '/trades':
                await self._handle_trades_command(update, context)
            elif command == '/risk':
                await self._handle_risk_command(update, context)
            elif command == '/alerts':
                await self._handle_alerts_command(update, context)
            else:
                await update.message.reply_text(
                    "Unknown command. Use /help to see available commands."
                )
            
        except Exception as e:
            logger.error(f"Error handling command: {str(e)}")
            await update.message.reply_text(
                "An error occurred while processing your command. Please try again later."
            ) 