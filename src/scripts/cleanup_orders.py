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
        self.telegram_service = TelegramService(self.config)
        
    async def initialize(self):
        """Initialize services."""
        await self.binance_service.initialize()
        await self.telegram_service.initialize()
        
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
                print(f"order: {order}")
                if order_type.lower() in ['stop_market', 'take_profit_market']:
                    # Cancel the order
                    success = await self.binance_service.cancel_order(
                        symbol=symbol,
                        order_id=order['id']
                    )
                    
                    if success:
                        cancelled_count += 1
                        logger.info(f"Cancelled {order_type} order {order['id']} for {symbol}")
                        
                        # Send notification
                        await self.telegram_service.send_message(
                            f"🧹 Cleaned up {order_type} order for {symbol}\n"
                            f"Order ID: {order['id']}\n"
                            f"Price: {order.get('price', 'N/A')}"
                        )
                    else:
                        logger.error(f"Failed to cancel order {order['id']} for {symbol}")
            
            # Send summary notification
            if cancelled_count > 0:
                await self.telegram_service.send_message(
                    f"✅ Cleanup complete\n"
                    f"Cancelled {cancelled_count} SL/TP orders for symbols without positions"
                )
            else:
                logger.info("No orders needed cleanup")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            await self.telegram_service.send_message(
                f"❌ Error during order cleanup: {str(e)}"
            )
            
    async def close(self):
        """Close services."""
        await self.binance_service.close()
        await self.telegram_service.close()

async def main():
    cleanup = OrderCleanup()
    try:
        await cleanup.initialize()
        await cleanup.cleanup()
    finally:
        await cleanup.close()

if __name__ == "__main__":
    asyncio.run(main()) 