import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.strategies.enhanced_trading_strategy import EnhancedTradingStrategy
from src.services.binance_service import BinanceService
from src.services.indicator_service import IndicatorService
from src.services.notification_service import NotificationService
from src.services.telegram_service import TelegramService
from src.services.discord_service import DiscordService
from src.core.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/handle_dca_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def test_handle_dca(symbol: str, position: Dict):
    """Test the _handle_dca function with real services."""
    try:
        # Load configuration
        config = load_config()
        
        # Initialize services
        binance_service = BinanceService(config)
        indicator_service = IndicatorService(config)
        telegram_service = TelegramService(config)
        discord_service = DiscordService(config)
        notification_service = NotificationService(config, telegram_service, discord_service)
        
        # Initialize strategy
        strategy = EnhancedTradingStrategy(
            config=config,
            binance_service=binance_service,
            indicator_service=indicator_service,
            notification_service=notification_service
        )
        
        # Initialize services
        await binance_service.initialize()
        await indicator_service.initialize()
        await notification_service.initialize()
        await strategy.initialize()
        
        logger.info(f"Testing DCA for {symbol}")
        logger.info(f"Position details: {position}")
        
        # Execute DCA
        result = await strategy._handle_dca(symbol, position)
        
        if result:
            logger.info(f"DCA executed successfully: {result}")
        else:
            logger.error("DCA execution failed")
        
        # Cleanup
        await strategy.close()
        await binance_service.close()
        await indicator_service.close()
        await notification_service.close()
        
    except Exception as e:
        logger.error(f"Error testing DCA: {str(e)}")
        raise

async def main():
    """Main function to run the DCA test."""
    # Example position for testing
    symbol = "SYNUSDT"
    position = {
        'info': {
            'positionAmt': '100.0',
            'side': 'LONG'
        },
        'entryPrice': 0.2165,
    }
    
    try:
        await test_handle_dca(symbol, position)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Set event loop policy for Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main()) 