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
from src.services.telegram_service import TelegramService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def get_balance() -> None:
    """Get the balance of the account.
    """
    binance_service = None
    telegram_service = None
    try:
        # Load configuration
        config = load_config()
        
        # Initialize services
        binance_service = BinanceService(config)
        telegram_service = TelegramService(config)
        
        # Initialize Binance service
        if not await binance_service.initialize():
            logger.error("Failed to initialize Binance service")
            return
            
        # Set up Telegram service
        telegram_service.set_binance_service(binance_service)
        await telegram_service.initialize()
        
        # Send the balance to the Telegram channel
        balance = await binance_service.get_account_balance()
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
                await telegram_service.send_message(message)
    except Exception as e:
        logger.error(f"Error getting balance: {str(e)}")
    finally:
        # Cleanup
        if binance_service:
            await binance_service.close()
        if telegram_service:
            try:
                await telegram_service.close()
            except Exception as e:
                logger.error(f"Error closing Telegram service: {str(e)}")
            # Give time for pending operations to complete
            await asyncio.sleep(2)

async def main():
    """Main function to run the test."""
    await get_balance()

if __name__ == "__main__":
    # Set event loop policy for Windows
    if sys.platform.lower() == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the test
    asyncio.run(main()) 