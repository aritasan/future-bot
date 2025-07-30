"""
Script to test the place_order function from binance_service.py
"""
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_get_open_orders(
    symbol: str
) -> None:
    """Test the get_open_orders function with given parameters.
    
    Args:
        symbol: Trading pair symbol (e.g. 'BTC/USDT')
    """
    try:
        # Load configuration
        config = load_config()
        
        # Initialize services
        binance_service = BinanceService(config)
        
        # Initialize Binance service
        if not await binance_service.initialize():
            logger.error("Failed to initialize Binance service")
            return
        
        # Get open orders
        orders = await binance_service.get_open_orders(symbol)
        logger.info(f"Open orders for {symbol}: {orders}")
        
        # existing_order = await binance_service.get_existing_order(symbol, 'STOP_MARKET', 'SELL')
        # logger.info(f"Existing order for {symbol}: {existing_order}")
        
        # cancelled = await binance_service._cancel_existing_orders(symbol, 'STOP_MARKET', 'LONG')
        # logger.info(f"Cancelled order for {symbol}: {cancelled}")
        
        # deleted_orders = await binance_service.cleanup_orders()
        # logger.info(f"Deleted orders: {deleted_orders}")
        
        stop_price = await binance_service.get_stop_price(symbol, 'LONG', 'STOP_MARKET')
        logger.info(f"Stop price for {symbol}: {stop_price}")
        take_profit_price = await binance_service.get_stop_price(symbol, 'LONG', 'TAKE_PROFIT_MARKET')
        logger.info(f"Take profit price for {symbol}: {take_profit_price}")
        
        # await binance_service.close_position(symbol, 'SHORT')
    except Exception as e:
        logger.error(f"Error testing get_open_orders: {str(e)}")
    finally:
        # Cleanup
        await binance_service.close()
        
async def main():
    try:
        # Add optional parameters

        # Place the order
        logger.info(f"Test get_open_orders")
        symbol = "GRIFFAIN/USDT"
        await test_get_open_orders(symbol)
            
    except Exception as e:
        logger.error(f"Error testing place_order: {str(e)}")
    finally:
        # Cleanup
        pass

if __name__ == "__main__":
    # Set event loop policy for Windows
    if sys.platform.lower() == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the test
    asyncio.run(main()) 