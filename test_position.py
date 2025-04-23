import asyncio
import logging
import platform
import json
from src.core.config import load_config
from src.services.binance_service import BinanceService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_get_position():
    try:
        # Load config
        config = load_config()
        
        # Initialize Binance service
        binance_service = BinanceService(config)
        if not await binance_service.initialize():
            logger.error("Failed to initialize Binance service")
            return
            
        # Test symbols
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BIO/USDT:USDT']
        
        for symbol in test_symbols:
            logger.info(f"\nTesting get_position for {symbol}")
            
            # Get position
            position = await binance_service.get_position(symbol)
            
            if position:
                logger.info(f"Raw position data: {json.dumps(position, indent=2)}")
                logger.info(f"Position details for {symbol}:")
                logger.info(f"Symbol: {position.get('symbol')}")
                logger.info(f"Position Amount: {float(position.get('info').get('positionAmt', 0))}")
                logger.info(f"Entry Price: {float(position.get('entryPrice', 0))}")
                logger.info(f"Unrealized PnL: {float(position.get('info').get('unRealizedProfit', 0))}")
                logger.info(f"Leverage: {position.get('leverage')}")
                logger.info(f"Margin Type: {position.get('marginType')}")
                logger.info(f"Position Side: {position.get('info').get('positionSide')}")
                
                # Calculate position value
                amt = float(position.get('info').get('positionAmt', 0))
                price = float(position.get('entryPrice', 0))
                value = abs(amt * price)
                logger.info(f"Position Value: {value:.2f} USDT")
            else:
                logger.info(f"No position found for {symbol}")
                
        # Close service
        await binance_service.close()
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")

if __name__ == "__main__":
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_get_position()) 