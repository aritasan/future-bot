"""
Script to test stop loss and take profit calculations with different configurations.
"""

import asyncio
import logging
import sys
import os
from typing import Dict, Any
import json

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.strategies.enhanced_trading_strategy import EnhancedTradingStrategy
from src.services.binance_service import BinanceService
from src.services.indicator_service import IndicatorService
from src.services.notification_service import NotificationService
from src.core.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def calculate_sl_tp(
    symbol: str,
    position_type: str
) -> None:
    """Calculate and display SL/TP for given parameters."""
    try:
        # Load configuration
        config = load_config()
        
        # Create service instances
        binance_service = BinanceService(config)
        indicator_service = IndicatorService(config)
        
        # Initialize services
        if not await binance_service.initialize():
            logger.error("Failed to initialize Binance service")
            return
            
        if not await indicator_service.initialize():
            logger.error("Failed to initialize Indicator service")
            return
            
        notification_service = NotificationService(config)
        
        # Get current price and ATR
        current_price = await binance_service.get_current_price(symbol)
        if current_price is None:
            logger.error(f"Failed to get current price for {symbol}")
            return
            
        atr = await indicator_service.calculate_atr(symbol)
        if atr is None:
            logger.error(f"Failed to calculate ATR for {symbol}")
            return
        
        # Create strategy instance
        strategy = EnhancedTradingStrategy(
            config=config,
            binance_service=binance_service,
            indicator_service=indicator_service,
            notification_service=notification_service
        )
        
        # Calculate stop loss
        stop_loss = await strategy._calculate_stop_loss(
            symbol=symbol,
            position_type=position_type,
            current_price=current_price,
            atr=atr
        )
        
        if stop_loss is None:
            logger.error("Failed to calculate stop loss")
            return
            
        # Calculate take profit
        take_profit = await strategy._calculate_take_profit(
            symbol=symbol,
            position_type=position_type,
            current_price=current_price,
            stop_loss=stop_loss
        )
        
        if take_profit is None:
            logger.error("Failed to calculate take profit")
            return
            
        # Calculate distances
        sl_distance = abs(current_price - stop_loss) / current_price * 100
        tp_distance = abs(current_price - take_profit) / current_price * 100
        risk_reward = tp_distance / sl_distance if sl_distance > 0 else 0
        
        # Print results
        print("\n=== SL/TP Calculation Results ===")
        print(f"Symbol: {symbol}")
        print(f"Position Type: {position_type}")
        print(f"Current Price: {current_price:.8f}")
        print(f"ATR: {atr:.8f}")
        print(f"\nStop Loss: {stop_loss:.8f} ({sl_distance:.2f}% from current price)")
        print(f"Take Profit: {take_profit:.8f} ({tp_distance:.2f}% from current price)")
        print(f"Risk/Reward Ratio: {risk_reward:.2f}")
        
        # Print config values used
        print("\nConfig Values Used:")
        print(f"Stop Loss ATR Multiplier: {config['risk_management']['stop_loss_atr_multiplier']}")
        print(f"Take Profit Multiplier: {config['risk_management']['take_profit_multiplier']}")
        print(f"Min Stop Distance: {config['risk_management']['min_stop_distance']}")
        print(f"Min TP Distance: {config['risk_management']['min_tp_distance']}")
        
    except Exception as e:
        logger.error(f"Error calculating SL/TP: {str(e)}")
    finally:
        # Clean up services
        if 'binance_service' in locals():
            await binance_service.close()
        if 'indicator_service' in locals():
            await indicator_service.close()

async def main():
    """Main function."""
    
    # Test cases
    test_cases = [
        {
            "symbol": "BTCUSDT",
            "position_type": "LONG"
        },
        {
            "symbol": "ETHUSDT",
            "position_type": "SHORT"
        },
        {
            "symbol": "SCRUSDT",
            "position_type": "SHORT"
        }
    ]
    
    # Run test cases
    for test_case in test_cases:
        print("\n" + "="*50)
        await calculate_sl_tp(**test_case)

if __name__ == "__main__":
    asyncio.run(main()) 