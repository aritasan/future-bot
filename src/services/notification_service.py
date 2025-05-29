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
        
        # Track service health
        self._service_health = {
            'telegram': True,
            'discord': True
        }

    async def initialize(self) -> bool:
        """Initialize the notification service."""
        try:
            if self.telegram_enabled and self.telegram_service:
                await self.telegram_service.initialize()
                self._service_health['telegram'] = True
                
            if self.discord_enabled and self.discord_service:
                await self.discord_service.initialize()
                self._service_health['discord'] = True
                
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

    async def send_message(self, message: str, embed: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send a message through all enabled notification channels.
        
        Args:
            message (str): Message to send
            embed (Optional[Dict[str, Any]]): Optional embed data for rich messages
            
        Returns:
            bool: True if message was sent successfully through at least one channel
        """
        if not self._is_initialized:
            logger.warning("Notification service not initialized")
            return False

        success = False
        try:
            if self.telegram_enabled and self.telegram_service and self._service_health['telegram']:
                try:
                    await self.telegram_service.send_message(message)
                    success = True
                except Exception as e:
                    logger.error(f"Error sending Telegram message: {str(e)}")
                    self._service_health['telegram'] = False
                
            logger.info(f"Discord: {self._service_health} {self.discord_enabled} {self.discord_service} {self.discord_service is not None}")
            if self.discord_enabled and self.discord_service and self._service_health['discord']:
                try:
                    cleaned_message = message.replace('<b>', '**').replace('</b>', '**')
                    if await self.discord_service.send_message(cleaned_message, embed):
                        success = True
                except Exception as e:
                    logger.error(f"Error sending Discord message: {str(e)}")
                    self._service_health['discord'] = False
                    
            return success
        except Exception as e:
            logger.error(f"Error in notification service: {str(e)}")
            return False

    async def send_trade_notification(self, 
                                    symbol: str, 
                                    action: str, 
                                    price: float, 
                                    quantity: float,
                                    reason: Optional[str] = None) -> bool:
        """
        Send a trade notification through all enabled channels.
        
        Args:
            symbol (str): Trading pair symbol
            action (str): Trade action (BUY/SELL)
            price (float): Trade price
            quantity (float): Trade quantity
            reason (Optional[str]): Optional reason for the trade
            
        Returns:
            bool: True if notification was sent successfully through at least one channel
        """
        if not self._is_initialized:
            logger.warning("Notification service not initialized")
            return False

        success = False
        try:
            if self.telegram_enabled and self.telegram_service and self._service_health['telegram']:
                try:
                    await self.telegram_service.send_trade_notification(
                        symbol=symbol,
                        action=action,
                        price=price,
                        quantity=quantity,
                        reason=reason
                    )
                    success = True
                except Exception as e:
                    logger.error(f"Error sending Telegram trade notification: {str(e)}")
                    self._service_health['telegram'] = False
                
            if self.discord_enabled and self.discord_service and self._service_health['discord']:
                try:
                    if await self.discord_service.send_trade_notification(
                        symbol=symbol,
                        action=action,
                        price=price,
                        quantity=quantity,
                        reason=reason
                    ):
                        success = True
                except Exception as e:
                    logger.error(f"Error sending Discord trade notification: {str(e)}")
                    self._service_health['discord'] = False
                    
            return success
        except Exception as e:
            logger.error(f"Error in notification service: {str(e)}")
            return False

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
        
    
    def is_trading_paused(self) -> bool:
        """Check if trading is paused."""
        if not self._is_initialized:
            logger.warning("Notification service not initialized")
            return

        try:
            if self.telegram_enabled and self.telegram_service:
                self.telegram_service.is_trading_paused()
                
            if self.discord_enabled and self.discord_service:
                self.discord_service.is_trading_paused()
        except Exception as e:
            logger.error(f"Error is_trading_paused: {str(e)}")
    
    async def send_dca_notification(self, dca_details: Dict) -> bool:
        """Send DCA execution notification.
        
        Args:
            dca_details: Dictionary containing DCA execution details
            
        Returns:
            bool: True if notification sent successfully, False otherwise
        """
        try:
            if not self._is_initialized:
                logger.error("Notification service not initialized or closed")
                return False
                
            message = (
                f"💰 <b>DCA Executed</b>\n\n"
                f"Symbol: {dca_details['symbol']}\n"
                f"Amount: {dca_details['dca_amount']:.8f}\n"
                f"New Entry: {dca_details['new_entry_price']:.8f}\n"
                f"Price Drop: {dca_details['price_drop']:.2f}%\n"
                f"Order ID: {dca_details['order_id']}"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending DCA notification: {str(e)}")
            return False
            
    async def send_trailing_stop_notification(self, symbol: str, new_stop: float, position_side: str) -> bool:
        """Send trailing stop update notification.
        
        Args:
            symbol: Trading pair symbol
            new_stop: New stop loss price
            position_side: Position side (LONG/SHORT)
            
        Returns:
            bool: True if notification sent successfully, False otherwise
        """
        try:
            if not self._is_initialized:
                logger.error("Notification service not initialized or closed")
                return False
                
            message = (
                f"🛑 <b>Trailing Stop Updated</b>\n\n"
                f"Symbol: {symbol}\n"
                f"Position: {position_side}\n"
                f"New Stop: {new_stop:.8f}"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending trailing stop notification: {str(e)}")
            return False

    async def send_stop_loss_notification(self, symbol: str, position_side: str, 
                                        position_size: float, entry_price: float, 
                                        stop_price: float, pnl_usd: float) -> bool:
        """Send a stop loss notification.
        
        Args:
            symbol: Trading pair symbol
            position_side: Position side (LONG/SHORT)
            position_size: Position size
            entry_price: Position entry price
            stop_price: Stop loss price
            pnl_usd: USDT of profit/loss
            
        Returns:
            bool: True if notification sent successfully, False otherwise
        """
        try:
            if not self._is_initialized:
                logger.error("Notification service not initialized or closed")
                return False
                
            message = (
                f"🛑 <b>Stop Loss Triggered</b>\n\n"
                f"Symbol: {symbol}\n"
                f"Position: {position_side}\n"
                f"Size: {position_size}\n"
                f"Entry Price: {entry_price:.8f}\n"
                f"Stop Price: {stop_price:.8f}\n"
                f"PnL: {pnl_usd:.2f} USDT"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending stop loss notification: {str(e)}")
            return False
            
    async def send_take_profit_notification(self, symbol: str, position_side: str,
                                        position_size: float, entry_price: float,
                                          tp_price: float, pnl_usd: float) -> bool:
        """Send a take profit notification.
        
        Args:
            symbol: Trading pair symbol
            position_side: Position side (LONG/SHORT)
            position_size: Position size
            entry_price: Position entry price
            tp_price: Take profit price
            pnl_usd: USDT of profit/loss
            
        Returns:
            bool: True if notification sent successfully, False otherwise
        """
        try:
            if not self._is_initialized:
                logger.error("Notification service not initialized or closed")
                return False
                
            message = (
                f"🎯 <b>Take Profit Triggered</b>\n\n"
                f"Symbol: {symbol}\n"
                f"Position: {position_side}\n"
                f"Size: {position_size}\n"
                f"Entry Price: {entry_price:.8f}\n"
                f"TP Price: {tp_price:.8f}\n"
                f"PnL: {pnl_usd:.2f} USDT"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending take profit notification: {str(e)}")
            return False

    async def send_order_notification(self, order: Dict) -> bool:
        """Send an order notification.
        
        Args:
            order: Order details
            signals: Trading signals containing SL/TP information
            
        Returns:
            bool: True if notification sent successfully, False otherwise
        """
        try:
            if not self._is_initialized:
                logger.error("Notification service not initialized or closed")
                return False
          
            message = (
                f"📊 <b>New Order</b>\n\n"
                f"Symbol: {order['symbol']}\n"
                f"Side: {order['side']}\n"
                f"Size: {order['amount']}\n"
                f"Price: {order['price']}\n"
                f"Type: {order['type']}"
            )
            
            # Add SL/TP information from order parameters
            if 'stop_loss' in order:
                sl_price = float(order['stop_loss'])
                sl_percent = abs((sl_price - float(order['price'])) / float(order['price']) * 100)
                message += f"\nStop Loss: {sl_price:.8f} ({sl_percent:.2f}%)"
                
            if 'take_profit' in order:
                tp_price = float(order['take_profit'])
                tp_percent = abs((tp_price - float(order['price'])) / float(order['price']) * 100)
                message += f"\nTake Profit: {tp_price:.8f} ({tp_percent:.2f}%)"
                
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending order notification: {str(e)}")
            return False