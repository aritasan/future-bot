"""
Main entry point for the trading bot.
"""

import asyncio
import logging
import signal
import sys
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from typing import List, Set, Optional
import platform

from src.core.config import load_config
from src.services.binance_service import BinanceService
from src.services.telegram_service import TelegramService
from src.services.indicator_service import IndicatorService
from src.core.health_monitor import HealthMonitor
from src.utils.risk import RiskManager
from src.strategies.enhanced_trading_strategy import EnhancedTradingStrategy

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging
log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create file handler with rotation
log_file = f'logs/trading_bot_{datetime.now().strftime("%Y%m%d")}.log'
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,  # Keep 5 backup files
    encoding='utf-8'
)
file_handler.setFormatter(log_formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)

# Remove all existing handlers to prevent duplicate logging
for handler in root_logger.handlers[:]:
    if not isinstance(handler, RotatingFileHandler):
        root_logger.removeHandler(handler)

# Configure specific loggers
logging.getLogger('telegram').setLevel(logging.WARNING)  # Reduce telegram library logging
logging.getLogger('ccxt').setLevel(logging.WARNING)  # Reduce ccxt library logging

logger = logging.getLogger(__name__)

# Global variables
is_running = True
trading_bot = None
tasks: List[asyncio.Task] = []
closed_services: Set[str] = set()

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global is_running
    logger.info("Received signal %s", signum)
    is_running = False

async def process_symbol(
    symbol: str,
    binance_service: BinanceService,
    telegram_service: TelegramService,
    health_monitor: HealthMonitor,
    risk_manager: RiskManager,
    strategy: EnhancedTradingStrategy,
    indicator_service: IndicatorService
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
                signals = await strategy.generate_signals(symbol, indicator_service)
                if signals is not None:
                    # Check risk before placing order
                    risk_check = await risk_manager.check_risk(symbol, signals.get('position_size', 0))
                    if risk_check:
                        try:
                            # Place order and send notification
                            order = await binance_service.place_order(signals)
                            if order:
                                await telegram_service.send_order_notification(order, signals)
                        except Exception as e:
                            logger.error(f"Error placing order for {symbol}: {str(e)}")
                            health_monitor.record_error()
                            continue

                await asyncio.sleep(1)  # Prevent CPU overload

            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {str(e)}")
                health_monitor.record_error()
                await asyncio.sleep(5)

    except Exception as e:
        logger.error(f"Fatal error processing symbol {symbol}: {str(e)}")
        health_monitor.record_error()
    finally:
        # Cleanup
        try:
            if 'binance' not in closed_services:
                await binance_service.close()
                closed_services.add('binance')
        except Exception as e:
            logger.error(f"Error closing Binance service for {symbol}: {str(e)}")
            
        try:
            if 'telegram' not in closed_services:
                await telegram_service.close()
                closed_services.add('telegram')
        except Exception as e:
            logger.error(f"Error closing Telegram service for {symbol}: {str(e)}")

async def cleanup_services(
    binance_service: Optional[BinanceService],
    telegram_service: Optional[TelegramService],
    health_monitor: Optional[HealthMonitor],
    indicator_service: Optional[IndicatorService],
    strategy: Optional[EnhancedTradingStrategy]
) -> None:
    """Cleanup all services in a controlled manner."""
    logger.info("Starting service cleanup...")
    
    # Close services in reverse order of initialization
    if strategy and 'strategy' not in closed_services:
        try:
            await strategy.close()
            closed_services.add('strategy')
        except Exception as e:
            logger.error(f"Error closing strategy: {str(e)}")
            
    if indicator_service and 'indicator' not in closed_services:
        try:
            await indicator_service.close()
            closed_services.add('indicator')
        except Exception as e:
            logger.error(f"Error closing indicator service: {str(e)}")
            
    if health_monitor and 'health_monitor' not in closed_services:
        try:
            await health_monitor.close()
            closed_services.add('health_monitor')
        except Exception as e:
            logger.error(f"Error closing health monitor: {str(e)}")
            
    if telegram_service and 'telegram' not in closed_services:
        try:
            # Give Telegram service time to process pending messages
            await asyncio.sleep(1)
            await telegram_service.close()
            closed_services.add('telegram')
        except Exception as e:
            logger.error(f"Error closing Telegram service: {str(e)}")
            
    if binance_service and 'binance' not in closed_services:
        try:
            await binance_service.close()
            closed_services.add('binance')
        except Exception as e:
            logger.error(f"Error closing Binance service: {str(e)}")
            
    logger.info("Service cleanup complete")

async def main():
    """Main entry point."""
    global trading_bot, tasks
    binance_service = None
    telegram_service = None
    health_monitor = None
    indicator_service = None
    strategy = None

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
        indicator_service = IndicatorService(config)

        # Set binance_service for services that need it
        risk_manager.set_binance_service(binance_service)
        strategy.set_binance_service(binance_service)
        telegram_service.set_binance_service(binance_service)

        # Initialize services
        try:
            if not binance_service.initialize():
                logger.error("Failed to initialize Binance service")
                return
        except Exception as e:
            logger.error(f"Failed to initialize Binance service: {str(e)}")
            return
            
        try:
            if not await telegram_service.initialize():
                logger.error("Failed to initialize Telegram service")
                return
        except Exception as e:
            logger.error(f"Failed to initialize Telegram service: {str(e)}")
            return
            
        try:
            if not await health_monitor.initialize():
                logger.error("Failed to initialize health monitor")
                return
        except Exception as e:
            logger.error(f"Failed to initialize health monitor: {str(e)}")
            return
            
        try:
            if not await indicator_service.initialize():
                logger.error("Failed to initialize indicator service")
                return
        except Exception as e:
            logger.error(f"Failed to initialize indicator service: {str(e)}")
            return
            
        try:
            if not await strategy.initialize():
                logger.error("Failed to initialize strategy")
                return
        except Exception as e:
            logger.error(f"Failed to initialize strategy: {str(e)}")
            return

        # Send startup notification
        await telegram_service.send_startup_notification()

        # Create a dedicated task for Telegram service
        telegram_task = asyncio.create_task(telegram_service.run())
        tasks.append(telegram_task)
        logger.info("Telegram service task started")
        
        # Start processing trading pairs
        try:
            with open("filtered_pairs.txt", "r") as f:
                trading_pairs = [line.strip() for line in f.readlines()]
        except Exception as e:
            logger.error(f"Error reading trading pairs: {str(e)}")
            return
            
        # Create tasks for each trading pair
        for symbol in trading_pairs:
            task = asyncio.create_task(process_symbol(
                symbol=symbol,
                binance_service=binance_service,
                telegram_service=telegram_service,
                health_monitor=health_monitor,
                risk_manager=risk_manager,
                strategy=strategy,
                indicator_service=indicator_service
            ))
            tasks.append(task)

        # Create task for periodic balance check
        balance_check_task = asyncio.create_task(telegram_service.periodic_balance_check())
        tasks.append(balance_check_task)

        # Main loop to keep the bot running
        while is_running:
            try:
                # Check if any tasks have completed
                done, pending = await asyncio.wait(
                    tasks,
                    timeout=0.1,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Handle completed tasks
                for task in done:
                    try:
                        await task
                    except asyncio.CancelledError:
                        logger.info("Task was cancelled")
                    except Exception as e:
                        logger.error(f"Task failed: {str(e)}")
                        health_monitor.record_error()
                        
                # Update tasks list with pending tasks
                tasks = list(pending)
                
                # Add back any completed tasks that need to be restarted
                for task in done:
                    if not task.done() or not task.cancelled():
                        tasks.append(task)
            
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                health_monitor.record_error()
                await asyncio.sleep(0.1)
            
        # Send shutdown notification
        await telegram_service.send_shutdown_notification()
            
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
        import traceback
        traceback.print_exc()
        logger.error(f"Fatal error in main: {str(e)}")
        # Send error notification
        if telegram_service:
            try:
                await telegram_service.send_message(f"❌ Bot stopped due to error: {str(e)}")
            except Exception as e:
                logger.error(f"Error sending error notification: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup services only once
        await cleanup_services(
            binance_service=binance_service,
            telegram_service=telegram_service,
            health_monitor=health_monitor,
            indicator_service=indicator_service,
            strategy=strategy
        )
        logger.info("Shutdown complete")

if __name__ == "__main__":
    # Configure event loop for Windows
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
    
    # Run the main function
    asyncio.run(main())