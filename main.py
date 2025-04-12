"""
Main entry point for the trading bot.
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, Any

from src.core.config import load_config
from src.services.binance_service import BinanceService
from src.services.telegram_service import TelegramService
from src.core.health_monitor import HealthMonitor
from src.utils.risk import RiskManager
from src.strategies.enhanced_trading_strategy import EnhancedTradingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
is_running = True
trading_bot = None
tasks = []

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global is_running
    logger.info(f"Received signal {signum}")
    is_running = False

async def process_symbol(
    symbol: str,
    binance_service: BinanceService,
    telegram_service: TelegramService,
    health_monitor: HealthMonitor,
    risk_manager: RiskManager,
    strategy: EnhancedTradingStrategy
) -> None:
    """Process a single trading symbol."""
    try:
        # Main trading loop
        while is_running:
            try:
                # Check health
                health_status = await health_monitor.check_health()
                if not health_status:
                    logger.warning("Health check failed, pausing trading")
                    await asyncio.sleep(60)
                    continue

                # Process trading signals
                signals = await strategy.generate_signals(symbol)
                if signals:
                    for signal in signals:
                        risk_check = await risk_manager.check_risk(signal)
                        if risk_check:
                            # Place order and send notification
                            order = binance_service.place_order(signal)
                            if order:
                                await telegram_service.send_order_notification(order)

                await asyncio.sleep(1)  # Prevent CPU overload

            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {str(e)}")
                await asyncio.sleep(5)

    except Exception as e:
        logger.error(f"Fatal error processing symbol {symbol}: {str(e)}")
    finally:
        # Cleanup
        binance_service.close()
        await telegram_service.close()

async def main():
    """Main entry point."""
    global trading_bot, tasks
    binance_service = None
    telegram_service = None

    try:
        # Load configuration
        config = load_config()

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Create service instances
        binance_service = BinanceService(config)
        telegram_service = TelegramService(config)
        health_monitor = HealthMonitor(config)
        risk_manager = RiskManager(config)
        strategy = EnhancedTradingStrategy(config)

        # Initialize services
        binance_service.initialize()
        if not await telegram_service.initialize():
            logger.error("Failed to initialize Telegram service")
            return
        await health_monitor.initialize()

        await telegram_service.send_startup_message()
        
        # Start processing trading pairs
        with open("filtered_pairs.txt", "r") as f:
            trading_pairs = [line.strip() for line in f.readlines()]
        for symbol in trading_pairs:
            # Create task for the symbol
            task = asyncio.create_task(process_symbol(
                symbol=symbol,
                binance_service=binance_service,
                telegram_service=telegram_service,
                health_monitor=health_monitor,
                risk_manager=risk_manager,
                strategy=strategy
            ))
            tasks.append(task)

        # Wait for all tasks to complete or signal received
        while is_running:
            await asyncio.sleep(1)
            
        # Cancel all tasks and wait for them to complete
        logger.info("Shutting down...")
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to complete with timeout
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Some tasks did not complete in time")
        except Exception as e:
            logger.error(f"Error waiting for tasks to complete: {e}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Fatal error in main: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        if binance_service:
            binance_service.close()
        if telegram_service:
            await telegram_service.close()
        logger.info("Shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())