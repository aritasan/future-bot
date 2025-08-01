"""
Script to test the place_order function from binance_service.py
"""
import asyncio
import logging
import sys
import os
from typing import Optional

# Set event loop policy for Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.config import load_config
from src.services.binance_service import BinanceService
from src.services.notification_service import NotificationService
from src.services.telegram_service import TelegramService
from src.services.discord_service import DiscordService


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_place_order(
    symbol: str,
    side: str,
    order_type: str,
    amount: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None
) -> None:
    """Test the place_order function with given parameters.
    
    Args:
        symbol: Trading pair symbol (e.g. 'BTC/USDT')
        side: Order side ('BUY' or 'SELL')
        order_type: Order type ('MARKET' or 'LIMIT')
        amount: Order amount
        stop_loss: Optional stop loss price
        take_profit: Optional take profit price
    """
    try:
        # Load configuration
        config = load_config()
        
        # Initialize services
        binance_service = BinanceService(config)
        # Initialize notification services based on config
        telegram_service = TelegramService(config) if config.get('api', {}).get('telegram', {}).get('enabled', True) else None
        discord_service = DiscordService(config) if config.get('api', {}).get('discord', {}).get('enabled', True) else None
        notification_service = NotificationService(config, telegram_service, discord_service)
        
        
        # Initialize Binance service
        if not await binance_service.initialize():
            logger.error("Failed to initialize Binance service")
            return
            
        
        # Prepare order parameters
        order_params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'amount': amount
        }
        
        # Add optional parameters
        if stop_loss:
            order_params['stop_loss'] = stop_loss
        if take_profit:
            order_params['take_profit'] = take_profit
            
        # Place the order
        logger.info(f"Placing order with parameters: {order_params}")
        result = await binance_service.place_order(order_params)
        
        if result:
            logger.info(f"Order placed successfully: {result}")
            await notification_service.send_order_notification(result)
        else:
            logger.error("Failed to place order")
            
    except Exception as e:
        logger.error(f"Error testing place_order: {str(e)}")
    finally:
        # Cleanup
        await binance_service.close()
        await notification_service.close()

async def main():
    """Main function to run the test."""
    # Example parameters - modify these as needed
    # symbol = "BTC/USDT"
    # side = "BUY"
    # order_type = "MARKET"
    # amount = 0.005  # Small amount for testing
    # stop_loss = 93500  # Optional
    # take_profit = 97500  # Optional
    # order_params = {
    #     'symbol': 'BTC/USDT',
    #     'side': 'BUY',
    #     'type': 'MARKET',
    #     'amount': 0.005,
    #     'stop_loss': 93500,
    #     'take_profit': 97500
    # }
    order_params = {
        'symbol': 'ME/USDT',
        'side': 'BUY',
        'order_type': 'MARKET',
        'amount': 7,
        'stop_loss': 0.3,
        'take_profit': 1
    }
    
    await test_place_order(
        **order_params
    )

if __name__ == "__main__":
    # Set event loop policy for Windows
    if sys.platform.lower() == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the test
    asyncio.run(main()) 