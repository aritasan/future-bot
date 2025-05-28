import asyncio
import logging
import sys
import os

# Set event loop policy for Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.config import load_config
from src.services.binance_service import BinanceService
from src.services.telegram_service import TelegramService
from src.services.discord_service import DiscordService
from src.services.notification_service import NotificationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OrderCleanup:
    def __init__(self):
        self.config = load_config()
        self.binance_service = BinanceService(self.config)
        
        # Initialize notification services based on config
        self.telegram_service = TelegramService(self.config) if self.config.get('api', {}).get('telegram', {}).get('enabled', True) else None
        self.discord_service = DiscordService(self.config['api']['discord']['webhook_url']) if self.config.get('api', {}).get('discord', {}).get('enabled', True) else None
        
        # Create notification service
        self.notification_service = NotificationService(
            config=self.config,
            telegram_service=self.telegram_service,
            discord_service=self.discord_service
        )
        
    async def initialize(self):
        """Initialize services."""
        await self.binance_service.initialize()
        await self.notification_service.initialize()
        
    async def cleanup(self):
        """Clean up SL and TP orders for symbols without open positions."""
        try:
            # Get all positions
            positions = await self.binance_service.get_positions()
            if not positions:
                logger.error("Failed to fetch positions")
                return
                
            # Create a set of symbols with open positions
            active_symbols = {
                pos['symbol'] for pos in positions 
                if float(pos.get('contracts', 0)) > 0
            }
            
            # Get all open orders
            all_orders = await self.binance_service.get_open_orders()
            if not all_orders:
                logger.info("No open orders found")
                return
                
            # Filter and cancel SL/TP orders for inactive symbols
            cancelled_count = 0
            for order in all_orders:
                symbol = order['symbol']
                
                # Skip if symbol has active position
                if symbol in active_symbols:
                    continue
                    
                # Check if order is SL or TP
                order_type = order.get('type', '').lower()
                if order_type.lower() in ['stop_market', 'take_profit_market']:
                    # Cancel the order
                    success = await self.binance_service.cancel_order(
                        symbol=symbol,
                        order_id=order['id']
                    )
                    
                    if success:
                        cancelled_count += 1
                        logger.info(f"Cancelled {order_type} order {order['id']} for {symbol}")
                        
                        # Send notification with embed for Discord
                        embed = {
                            "title": "Order Cleanup",
                            "color": 0x00ff00,  # Green color
                            "fields": [
                                {"name": "Symbol", "value": symbol, "inline": True},
                                {"name": "Order Type", "value": order_type, "inline": True},
                                {"name": "Position Side", "value": order['info']['positionSide'], "inline": True}
                            ]
                        }
                        
                        await self.notification_service.send_message(
                            f"🧹 Cleaned up {order_type} order for {symbol}\n"
                            f"Position Type: {order['info']['positionSide']}",
                            embed=embed
                        )
                    else:
                        logger.error(f"Failed to cancel order {order['id']} for {symbol}")
            
            # Send summary notification
            if cancelled_count > 0:
                embed = {
                    "title": "Cleanup Complete",
                    "color": 0x00ff00,  # Green color
                    "description": f"Cancelled {cancelled_count} SL/TP orders for symbols without positions"
                }
                
                await self.notification_service.send_message(
                    f"✅ Cleanup complete\n"
                    f"Cancelled {cancelled_count} SL/TP orders for symbols without positions",
                    embed=embed
                )
            else:
                logger.info("No orders needed cleanup")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            embed = {
                "title": "Cleanup Error",
                "color": 0xff0000,  # Red color
                "description": str(e)
            }
            
            await self.notification_service.send_error_notification(
                f"❌ Error during order cleanup: {str(e)}"
            )
            
    async def close(self):
        """Close services."""
        await self.binance_service.close()
        await self.notification_service.close()

async def main():
    cleanup = OrderCleanup()
    try:
        await cleanup.initialize()
        await cleanup.cleanup()
    finally:
        await cleanup.close()

if __name__ == "__main__":
    asyncio.run(main()) 