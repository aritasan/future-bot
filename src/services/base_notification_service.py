"""
Base notification service class that contains shared functionality.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from src.services.binance_service import BinanceService

logger = logging.getLogger(__name__)

class BaseNotificationService:
    """Base class for notification services."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize base notification service.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self._is_initialized = False
        self._is_paused = False
        self.binance_service = None
        self._start_time = datetime.now()
        self._pause_event = asyncio.Event()
        
    async def initialize(self) -> None:
        """Initialize the service and its dependencies."""
        try:
            # Initialize Binance service
            self.binance_service = BinanceService(self.config)
            await self.binance_service.initialize()
            
            self._is_initialized = True
            logger.info("Base notification service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing base notification service: {str(e)}")
            raise
        
    def set_binance_service(self, binance_service):
        """Set the Binance service for balance and position checks.
        
        Args:
            binance_service: Binance service instance
        """
        self.binance_service = binance_service
        
    def pause_trading(self) -> None:
        """Pause trading and set the pause event."""
        if not self._is_paused:
            self._is_paused = True
            self._pause_event.set()
            logger.info("Trading paused")
            
    def unpause_trading(self) -> None:
        """Unpause trading and clear the pause event."""
        if self._is_paused:
            self._is_paused = False
            self._pause_event.clear()
            logger.info("Trading unpaused")
            
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
        
    def is_trading_paused(self) -> bool:
        """Check if trading is paused."""
        return self._is_paused
       
    async def _get_status_message(self) -> str:
        """Get the current bot status message."""
        try:
            status = "🟢 Running" if not self._is_paused else "🔴 Stopped"
            uptime = str(datetime.now() - self._start_time).split('.')[0] if self._start_time else "N/A"
            
            message = (
                f"🤖 <b>Bot Status</b>\n\n"
                f"Status: {status}\n"
                f"Uptime: {uptime}\n"
                f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return message
        except Exception as e:
            logger.error(f"Error getting status message: {str(e)}")
            return "Error getting status information."
        
    def help_content(self) -> str:
        """Return help content for the service."""
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
    
    async def generate_balance_report(self) -> str:
        """Generate a balance report.
        
        Returns:
            str: Formatted balance report
        """
        try:
            if not self.binance_service:
                return "Binance service not available"
            
            # Get account balance
            balance = await self.binance_service.get_account_balance()
            if not balance:
                return "Failed to get account balance"
            
            # Get position statistics
            position_stats = await self.binance_service.get_position_statistics()
            if position_stats is None:
                return "Failed to get position statistics"
            
            # Format header message
            header_message = (
                "📊 **Detailed Report**\n\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Paused: {'Yes' if self._is_paused else 'No'}\n\n"
                "💰 **Balance**\n"
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
                            header_message += f"{asset}: {amount_float}\n"
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid balance amount for {asset}: {amount}")
                        continue
            
            # Add positions summary
            header_message += f"\n📈 Active Positions: {position_stats['active_positions']}\n"
            header_message += f"💵 Total Unrealized PnL: {position_stats['total_pnl']:.2f} USDT\n"
            
            return header_message
            
        except Exception as e:
            logger.error(f"Error generating balance report: {str(e)}")
            return "An error occurred while generating the balance report."

    async def generate_report(self) -> List[str]:
        """Generate a detailed report including balance and position details.
        
        Returns:
            List[str]: List of report chunks
        """
        try:
            if not self.binance_service:
                return ["Binance service not available"]
            
            # Get account balance
            balance = await self.binance_service.get_account_balance()
            if not balance:
                return ["Failed to get account balance"]
            
            # Get position statistics
            position_stats = await self.binance_service.get_position_statistics()
            if position_stats is None:
                return ["Failed to get position statistics"]
            
            # Format header message
            header_message = (
                "📊 **Detailed Report**\n\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Paused: {'Yes' if self._is_paused else 'No'}\n\n"
                "💰 **Balance**\n"
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
                            header_message += f"{asset}: {amount_float}\n"
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid balance amount for {asset}: {amount}")
                        continue
            
            # Add positions summary
            header_message += f"\n📈 Active Positions: {position_stats['active_positions']}\n"
            header_message += f"💵 Total Unrealized PnL: {position_stats['total_pnl']:.2f} USDT\n"
            
            # Initialize report chunks with header
            report_chunks = [header_message]
            
            # Add position details if available
            if position_stats['position_details']:
                current_chunk = "📋 **Position Details**\n"
                
                for pos in position_stats['position_details']:
                    position_info = (
                        f"Symbol: {pos['symbol']}\n"
                        f"Size: {pos['size']}\n"
                        f"PnL: {pos['pnl']:.2f} USDT\n"
                        f"Entry Price: {pos['entry_price']}\n"
                        f"Mark Price: {pos['mark_price']}\n"
                        f"Leverage: {pos['leverage']}x\n"
                        f"Side: {pos['side']}\n\n"
                    )
                    
                    # If adding this position would make the chunk too long, start a new chunk
                    if len(current_chunk) + len(position_info) > 4000:
                        report_chunks.append(current_chunk)
                        current_chunk = "📋 **Position Details (continued)**\n" + position_info
                    else:
                        current_chunk += position_info
                
                # Add the last chunk if it's not empty
                if current_chunk:
                    report_chunks.append(current_chunk)
            
            return report_chunks
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return ["An error occurred while generating the report."]

    async def cleanup_orders(self) -> str:
        """Clean up orders and return a formatted message.
        
        Returns:
            str: Formatted cleanup message
        """
        try:
            if not self.binance_service:
                return "Binance service not available"
            
            deleted_orders = await self.binance_service.cleanup_orders()
            if deleted_orders:
                message = "🧹 **Order Cleanup**\n\n"
                for symbol, count in deleted_orders.items():
                    message += f"✅ Cleaned up {count} orders for {symbol}\n"
                return message
            else:
                return "No orders to clean up."
                
        except Exception as e:
            logger.error(f"Error cleaning up orders: {str(e)}")
            return "An error occurred while cleaning up orders."