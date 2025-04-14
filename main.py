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
import time

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
                # Check if trading is paused
                if telegram_service.is_trading_paused():
                    await telegram_service.wait_for_trading_resume()
                    continue

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
        # Only cleanup if the service is not already closed
        if not is_running:
            try:
                if 'binance' not in closed_services:
                    await binance_service.close()
                    closed_services.add('binance')
            except Exception as e:
                logger.error(f"Error closing Binance service for {symbol}: {str(e)}")
                
            try:
                if 'telegram' not in closed_services:
                    # Wait for any pending messages to be sent
                    await asyncio.sleep(2)
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
    
    # Set a timeout for cleanup
    cleanup_timeout = 10.0  # Increased timeout for better reliability
    
    try:
        # First, close non-critical services concurrently
        non_critical_tasks = []
        
        if strategy and 'strategy' not in closed_services:
            logger.info("Closing strategy service...")
            non_critical_tasks.append(asyncio.create_task(strategy.close()))
            closed_services.add('strategy')
            
        if indicator_service and 'indicator' not in closed_services:
            logger.info("Closing indicator service...")
            non_critical_tasks.append(asyncio.create_task(indicator_service.close()))
            closed_services.add('indicator')
            
        if health_monitor and 'health_monitor' not in closed_services:
            logger.info("Closing health monitor...")
            non_critical_tasks.append(asyncio.create_task(health_monitor.close()))
            closed_services.add('health_monitor')
        
        # Wait for non-critical services to close with timeout
        if non_critical_tasks:
            try:
                logger.info(f"Waiting for {len(non_critical_tasks)} non-critical services to close...")
                await asyncio.wait_for(asyncio.gather(*non_critical_tasks, return_exceptions=True), 
                                     timeout=cleanup_timeout/2)
                logger.info("Non-critical services closed successfully")
            except asyncio.TimeoutError:
                logger.warning("Timeout while closing non-critical services")
                for task in non_critical_tasks:
                    if not task.done():
                        task.cancel()
            except Exception as e:
                logger.error(f"Error closing non-critical services: {str(e)}")
        
        # Then close critical services sequentially
        if telegram_service and 'telegram' not in closed_services:
            try:
                logger.info("Closing Telegram service...")
                await asyncio.wait_for(telegram_service.close(), timeout=cleanup_timeout/4)
                closed_services.add('telegram')
                logger.info("Telegram service closed successfully")
            except asyncio.TimeoutError:
                logger.warning("Timeout while closing Telegram service")
            except Exception as e:
                logger.error(f"Error closing Telegram service: {str(e)}")
        
        # Finally close Binance service
        if binance_service and 'binance' not in closed_services:
            try:
                logger.info("Closing Binance service...")
                await asyncio.wait_for(binance_service.close(), timeout=cleanup_timeout/4)
                closed_services.add('binance')
                logger.info("Binance service closed successfully")
            except asyncio.TimeoutError:
                logger.warning("Timeout while closing Binance service")
            except Exception as e:
                logger.error(f"Error closing Binance service: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error during service cleanup: {str(e)}")
    finally:
        # Log final cleanup status
        logger.info(f"Service cleanup completed. Closed services: {closed_services}")
        
        # Clear any remaining tasks
        for task in asyncio.all_tasks():
            if not task.done() and task != asyncio.current_task():
                task.cancel()
                
        logger.info("All remaining tasks cancelled")


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

        # Set up services
        risk_manager.set_binance_service(binance_service)
        strategy.set_binance_service(binance_service)
        telegram_service.set_binance_service(binance_service)

        # Initialize services
        if not binance_service.initialize():
            logger.error("Failed to initialize Binance service")
            return
            
        await telegram_service.initialize()
        if not telegram_service._is_initialized:
            logger.error("Telegram service initialization failed")
            return
            
        if not await health_monitor.initialize():
            logger.error("Failed to initialize health monitor")
            return
            
        if not await indicator_service.initialize():
            logger.error("Failed to initialize indicator service")
            return
            
        if not await strategy.initialize():
            logger.error("Failed to initialize strategy")
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
                # Read trading pairs from file
                try:
                    with open("filtered_pairs.txt", "r") as f:
                        symbols = [line.strip() for line in f.readlines()]
                except Exception as e:
                    logger.error(f"Error reading trading pairs: {str(e)}")
                    await asyncio.sleep(60)
                    continue
                    
                if not symbols:
                    logger.warning("No trading symbols available in filtered_pairs.txt, retrying in 60 seconds...")
                    await asyncio.sleep(60)
                    continue
                    
                logger.info(f"Processing {len(symbols)} trading symbols from filtered_pairs.txt")
                
                # Process each symbol
                for symbol in symbols:
                    try:
                        # Generate signals with indicator_service
                        signals = await strategy.generate_signals(symbol, indicator_service)
                        
                        # Process signals
                        if signals:
                            await strategy.process_signals(symbol, signals)
                            
                    except Exception as e:
                        logger.error(f"Error processing symbol {symbol}: {str(e)}")
                        continue
                
                # Sleep for a while before next iteration
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying
            
        # Send shutdown notification
        if telegram_service:
            try:
                await telegram_service.send_shutdown_notification()
            except Exception as e:
                logger.error(f"Error sending shutdown notification: {str(e)}")
        
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
            # Force cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Fatal error in main: {str(e)}")
        if telegram_service:
            try:
                await telegram_service.send_message(f"❌ Bot stopped due to error: {str(e)}")
            except Exception as e:
                logger.error(f"Error sending error notification: {str(e)}")
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
    # Set event loop policy for Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        # Run main function using asyncio.run()
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1) 