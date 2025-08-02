#!/usr/bin/env python3
"""
Main entry point for the trading bot with Quantitative Trading Integration.
"""

import asyncio
import logging

# Setup logging configuration
from src.utils.logging_config import setup_logging
setup_logging()

# Disable werkzeug logs
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('dash').setLevel(logging.ERROR)
logging.getLogger('dash.dash').setLevel(logging.ERROR)
logging.getLogger('flask').setLevel(logging.ERROR)
logging.getLogger('aiohttp.access').setLevel(logging.ERROR)
logging.getLogger('websockets.server').setLevel(logging.WARNING)


import signal
import traceback
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from typing import List, Set, Optional, Dict

from src.core.config import load_config
from src.services.binance_service import BinanceService
from src.services.telegram_service import TelegramService
from src.services.discord_service import DiscordService
from src.services.indicator_service import IndicatorService
from src.core.health_monitor import HealthMonitor
from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
from src.services.notification_service import NotificationService
from src.services.cache_service import CacheService
from src.services.cache_monitor_service import CacheMonitorService

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging
log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create file handler with rotation
log_file = f'logs/trading_bot_quantitative_{datetime.now().strftime("%Y%m%d")}.log'
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
logging.getLogger('telegram').setLevel(logging.WARNING)
logging.getLogger('ccxt').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Global variables
is_running = True
trading_bot = None
tasks: List[asyncio.Task] = []
closed_services: Set[str] = set()
shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global is_running
    logger.info("Received signal %s", signum)
    logger.info('Frame: %s', frame)
    is_running = False
    # Set the shutdown event to wake up any sleeping tasks
    if shutdown_event:
        shutdown_event.set()

async def process_symbol_with_quantitative(
    symbol: str,
    binance_service: BinanceService,
    telegram_service: Optional[TelegramService],
    discord_service: Optional[DiscordService],
    health_monitor: HealthMonitor,
    strategy: EnhancedTradingStrategyWithQuantitative,
    indicator_service: IndicatorService,
    cache_service: CacheService
) -> None:
    """Process a single trading symbol with quantitative analysis."""
    try:
        logger.info(f"Starting quantitative trading for symbol: {symbol}")
        
        # Check if trading is paused
        if (telegram_service is not None and telegram_service.is_trading_paused()):
            logger.info(f"Trading paused for {symbol}, skipping")
            return
        
        if (discord_service is not None and discord_service.is_trading_paused()):
            logger.info(f"Trading paused for {symbol}, skipping")
            return

        # Check profit target
        if await strategy.check_profit_target():
            logger.info(f"Profit target reached, skipping {symbol}")
            return

        # Check cache for existing signals
        cached_signals = await cache_service.get_market_data(symbol, "5m")
        if cached_signals:
            logger.info(f"Using cached signals for {symbol}")
            signals = cached_signals
        else:
            # Generate signals with quantitative analysis
            signals = await asyncio.wait_for(strategy.generate_signals(symbol, indicator_service), timeout=60)
            
            # Cache the signals
            if signals:
                await cache_service.cache_market_data(symbol, "5m", signals, ttl=300)  # 5 minutes TTL
        
        if signals:
            logger.info(f"Generated quantitative signals for {symbol}: {signals}")
            
            # Process signals
            await asyncio.wait_for(strategy.process_trading_signals(signals), timeout=60)
            
            # Get quantitative recommendations
            recommendations = await asyncio.wait_for(strategy.get_quantitative_recommendations(symbol), timeout=60)
            if recommendations and 'error' not in recommendations:
                logger.info(f"Quantitative recommendations for {symbol}: {recommendations}")
                # Cache recommendations
                await cache_service.cache_analysis(symbol, "quantitative_recommendations", recommendations, ttl=600)  # 10 minutes TTL
            
            # Send notifications if significant
            if signals.get('quantitative_confidence', 0) > 0.7:
                await asyncio.wait_for(send_quantitative_notification(
                    symbol, signals, recommendations, telegram_service, discord_service
                ), timeout=60)
        
        # Health check
        if health_monitor:
            await asyncio.wait_for(health_monitor.check_health(), timeout=30)
        
        logger.info(f"Completed processing for {symbol}")
        
    except asyncio.CancelledError:
        logger.info(f"Processing cancelled for {symbol}")
        raise
    except Exception as e:
        logger.error(f"Fatal error processing symbol {symbol}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Attempt recovery
        if hasattr(strategy, 'recover_from_error'):
            try:
                recovery_success = await strategy.recover_from_error(e)
                if recovery_success:
                    logger.info(f"Recovery successful for {symbol}")
                else:
                    logger.error(f"Recovery failed for {symbol}")
            except Exception as recovery_error:
                logger.error(f"Recovery attempt failed: {str(recovery_error)}")
        
        # Continue with next symbol instead of crashing
        logger.error(f"Error processing symbol {symbol}: {str(e)}")
        return

async def send_quantitative_notification(
    symbol: str, 
    signals: Dict, 
    recommendations: Dict,
    telegram_service: Optional[TelegramService],
    discord_service: Optional[DiscordService]
) -> None:
    """Send quantitative analysis notifications."""
    try:
        message = f"🔬 Quantitative Analysis Alert for {symbol}\n\n"
        
        # Signal information
        message += f"📊 Signal Strength: {signals.get('quantitative_confidence', 0):.2f}\n"
        message += f"📈 Action: {signals.get('action', 'hold').upper()}\n"
        message += f"💰 Position Size: {signals.get('optimized_position_size', 0):.4f}\n"
        
        # Statistical validation
        validation = signals.get('quantitative_validation', {})
        message += f"✅ Statistically Valid: {validation.get('is_valid', False)}\n"
        message += f"📊 Sharpe Ratio: {validation.get('sharpe_ratio', 0):.2f}\n"
        
        # Risk metrics
        message += f"⚠️ VaR Estimate: {signals.get('var_estimate', 0):.4f}\n"
        message += f"🎯 Market Efficiency: {signals.get('market_efficiency', 0):.2f}\n"
        
        # Recommendations
        if recommendations and 'trading_recommendation' in recommendations:
            rec = recommendations['trading_recommendation']
            message += f"\n🎯 Quantitative Recommendation:\n"
            message += f"Action: {rec.get('action', 'hold').upper()}\n"
            message += f"Confidence: {rec.get('confidence', 0):.2f}\n"
            
            reasoning = rec.get('reasoning', [])
            if reasoning:
                message += f"Reasons: {', '.join(reasoning[:3])}\n"
        
        # Send notifications
        if telegram_service:
            await telegram_service.send_message(message)
        
        if discord_service:
            await discord_service.send_message(message)
            
    except Exception as e:
        logger.error(f"Error sending quantitative notification: {str(e)}")

async def run_portfolio_analysis(
    strategy: EnhancedTradingStrategyWithQuantitative,
    symbols: List[str],
    cache_service: CacheService
) -> None:
    """Run periodic portfolio analysis."""
    try:
        while is_running:
            try:
                logger.info("Running portfolio optimization analysis...")
                
                # Check cache for portfolio analysis
                cached_optimization = await cache_service.get_portfolio_analysis("optimization")
                if cached_optimization:
                    logger.info("Using cached portfolio optimization results")
                    optimization_results = cached_optimization
                else:
                    # Analyze portfolio optimization
                    try:
                        optimization_results = await asyncio.wait_for(strategy.analyze_portfolio_optimization(symbols), timeout=120)
                        if optimization_results and 'error' not in optimization_results:
                            # Cache optimization results
                            await cache_service.cache_portfolio_analysis("optimization", optimization_results, ttl=3600)  # 1 hour TTL
                    except Exception as e:
                        logger.error(f"Error in portfolio optimization analysis: {str(e)}")
                        optimization_results = None
                
                # Check cache for factor analysis
                cached_factors = await cache_service.get_portfolio_analysis("factors")
                if cached_factors:
                    logger.info("Using cached factor analysis results")
                    factor_results = cached_factors
                else:
                    # Analyze factor exposures
                    try:
                        factor_results = await asyncio.wait_for(strategy.analyze_factor_exposures(symbols), timeout=120)
                        if factor_results and 'error' not in factor_results:
                            # Cache factor results
                            await cache_service.cache_portfolio_analysis("factors", factor_results, ttl=3600)  # 1 hour TTL
                    except Exception as e:
                        logger.error(f"Error in factor analysis: {str(e)}")
                        factor_results = None
                
                # Get performance metrics
                metrics = await asyncio.wait_for(strategy.get_performance_metrics(), timeout=60)
                # Cache performance metrics
                await cache_service.cache_performance_metrics(metrics, ttl=1800)  # 30 minutes TTL
                
                # Wait 6 hours before next analysis - use wait_for with shutdown event
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=21600)  # 6 hours
                    if shutdown_event.is_set():
                        break
                except asyncio.TimeoutError:
                    continue
                
            except asyncio.CancelledError:
                logger.info("Portfolio analysis task received cancellation.")
                raise
            except Exception as e:
                logger.error(f"Error in portfolio analysis: {str(e)}")
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=3600)
                    if shutdown_event.is_set():
                        break
                except asyncio.CancelledError:
                    raise
                
    except asyncio.CancelledError:
        logger.info("run_portfolio_analysis cancelled.")
        raise
    except Exception as e:
        logger.error(f"Fatal error in portfolio analysis: {str(e)}")

async def cleanup_services(
    binance_service: Optional[BinanceService],
    telegram_service: Optional[TelegramService],
    discord_service: Optional[DiscordService],
    health_monitor: Optional[HealthMonitor],
    indicator_service: Optional[IndicatorService],
    strategy: Optional[EnhancedTradingStrategyWithQuantitative],
    cache_service: Optional[CacheService],
    cache_monitor: Optional[CacheMonitorService]
) -> None:
    """Clean up all services with improved timeout handling."""
    try:
        logger.info("Cleaning up services...")
        
        # Create cleanup tasks with individual timeouts
        cleanup_tasks = []
        
        # Close strategy
        if strategy:
            task = asyncio.create_task(strategy.close())
            cleanup_tasks.append(("strategy", task, 30))
        
        # Close indicator service
        if indicator_service:
            task = asyncio.create_task(indicator_service.close())
            cleanup_tasks.append(("indicator_service", task, 30))
        
        # Close health monitor
        if health_monitor:
            task = asyncio.create_task(health_monitor.close())
            cleanup_tasks.append(("health_monitor", task, 30))
        
        # Close Discord service
        if discord_service:
            task = asyncio.create_task(discord_service.close())
            cleanup_tasks.append(("discord_service", task, 30))
        
        # Close Telegram service
        if telegram_service:
            task = asyncio.create_task(telegram_service.close())
            cleanup_tasks.append(("telegram_service", task, 30))
        
        # Close cache monitor service
        if cache_monitor:
            task = asyncio.create_task(cache_monitor.close())
            cleanup_tasks.append(("cache_monitor", task, 30))
        
        # Close cache service
        if cache_service:
            task = asyncio.create_task(cache_service.close())
            cleanup_tasks.append(("cache_service", task, 30))
        
        # Close Binance service
        if binance_service:
            task = asyncio.create_task(binance_service.close())
            cleanup_tasks.append(("binance_service", task, 30))
        
        # Wait for all cleanup tasks with individual timeouts
        for service_name, task, timeout in cleanup_tasks:
            try:
                await asyncio.wait_for(task, timeout=timeout)
                closed_services.add(service_name)
                logger.info(f"{service_name} closed successfully")
            except asyncio.TimeoutError:
                logger.warning(f"{service_name} cleanup timed out after {timeout}s")
                if not task.done():
                    task.cancel()
            except Exception as e:
                logger.error(f"Error closing {service_name}: {str(e)}")
                if not task.done():
                    task.cancel()
        
        logger.info("All services cleaned up")
        
    except asyncio.CancelledError:
        logger.info("Cleanup services received cancellation.")
        raise
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

async def cancel_all_tasks(tasks: List[asyncio.Task], timeout: float = 30.0) -> None:
    """Cancel all tasks and wait for them to complete with timeout."""
    if not tasks:
        return
    
    logger.info(f"Cancelling {len(tasks)} tasks...")
    
    # Cancel all tasks
    for task in tasks:
        if not task.done():
            task.cancel()
    
    # Wait for all tasks to complete with timeout
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)
        logger.info("All tasks cancelled successfully")
    except asyncio.TimeoutError:
        logger.warning(f"Task cancellation timed out after {timeout}s")
        # Force cancel any remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        # Give a final chance for cleanup
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
        except asyncio.TimeoutError:
            logger.error("Some tasks could not be cancelled properly")

async def main():
    """Main function with quantitative trading integration."""
    global is_running, trading_bot, shutdown_event
    
    try:
        logger.info("Starting Trading Bot with Quantitative Trading Integration")
        
        # Load configuration
        config = load_config()
        
        # Initialize services
        binance_service = None
        telegram_service = None
        discord_service = None
        health_monitor = None
        indicator_service = None
        strategy = None
        cache_service = None
        cache_monitor = None
        
        try:
            # Initialize Binance service
            binance_service = BinanceService(config)
            await binance_service.initialize()
            logger.info("Binance service initialized")
            
            # Initialize Telegram service
            if config.get('telegram_enabled', False):
                telegram_service = TelegramService(config)
                await telegram_service.initialize()
                logger.info("Telegram service initialized")
            
            # Initialize Discord service
            if config.get('api', {}).get('discord', {}).get('enabled', False):
                try:
                    logger.info("Attempting to initialize Discord service...")
                    discord_service = DiscordService(config)
                    await discord_service.initialize()
                    logger.info("Discord service initialized successfully")
                except Exception as e:
                    logger.error(f"Error initializing Discord service: {str(e)}")
                    import traceback
                    logger.error(f"Discord initialization traceback:\n{traceback.format_exc()}")
                    discord_service = None
            
            # Initialize health monitor
            health_monitor = HealthMonitor(config)
            await health_monitor.initialize()
            logger.info("Health monitor initialized")
            
            # Initialize indicator service
            indicator_service = IndicatorService(config)
            await indicator_service.initialize()
            logger.info("Indicator service initialized")
            
            # Initialize cache service
            cache_service = CacheService(config)
            await cache_service.initialize()
            logger.info("Cache service initialized")
            
            # Initialize cache monitor service
            cache_monitor = CacheMonitorService(config)
            await cache_monitor.initialize()
            logger.info("Cache monitor service initialized")
            
            # Initialize notification service
            notification_service = NotificationService(config, telegram_service, discord_service)
            await notification_service.initialize()
            logger.info("Notification service initialized")
            
            # Set notification callback for Binance service
            binance_service.set_notification_callback(notification_service.send_message)
            logger.info("Notification callback set for Binance service")
            
            # Initialize enhanced trading strategy with quantitative integration
            strategy = EnhancedTradingStrategyWithQuantitative(
                config, binance_service, indicator_service, notification_service, cache_service
            )
            await strategy.initialize()
            logger.info("Enhanced Trading Strategy with Quantitative Integration initialized")
            
            # Đọc tất cả các cặp trading pairs từ file future_symbols.txt
            try:
                with open("future_symbols.txt", "r") as f:
                    symbols = [line.strip() for line in f if line.strip()]
            except Exception as e:
                logger.error(f"Error reading trading pairs: {str(e)}")
                symbols = ['BTCUSDT']
            
            logger.info(f"Loaded {len(symbols)} trading symbols from future_symbols.txt")
            logger.info(f"First 10 symbols: {symbols[:10]}")
            logger.info(f"Last 10 symbols: {symbols[-10:]}")
            logger.info(f"Total symbols to process: {len(symbols)}")
            
            # Start trading tasks with limited concurrency
            max_concurrent_tasks = 10  # Limit concurrent tasks
            semaphore = asyncio.Semaphore(max_concurrent_tasks)
            
            # Progress tracking
            processed_count = 0
            total_symbols = len(symbols)
            
            async def process_symbol_batch(symbol_batch: List[str]):
                """Process a batch of symbols."""
                nonlocal processed_count
                async with semaphore:
                    for symbol in symbol_batch:
                        processed_count += 1
                        logger.info(f"Processing symbol {processed_count}/{total_symbols}: {symbol}")
                        try:
                            await process_symbol_with_quantitative(
                                symbol, binance_service, telegram_service, discord_service,
                                health_monitor, strategy, indicator_service, cache_service
                            )
                        except Exception as e:
                            logger.error(f"Error processing symbol {symbol}: {str(e)}")
            
            logger.info(f"Starting continuous processing of {total_symbols} symbols with max {max_concurrent_tasks} concurrent batches")
            
            # Send startup notification
            startup_message = "🚀 Trading Bot with Quantitative Trading Integration started successfully!"
            if telegram_service:
                await telegram_service.send_message(startup_message)
            if discord_service:
                await discord_service.send_message(startup_message)
            
            # Continuous processing loop
            cycle_count = 0
            while is_running and not shutdown_event.is_set():
                cycle_count += 1
                logger.info(f"=== Starting cycle {cycle_count} ===")
                
                # Clear previous tasks
                tasks.clear()
                
                # Process symbols in batches
                batch_size = max_concurrent_tasks
                symbol_batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
                
                for batch in symbol_batches:
                    task = asyncio.create_task(process_symbol_batch(batch))
                    tasks.append(task)
                
                # Start portfolio analysis task
                portfolio_task = asyncio.create_task(
                    run_portfolio_analysis(strategy, symbols, cache_service)
                )
                tasks.append(portfolio_task)
                
                # Wait for all tasks to complete or shutdown signal
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    logger.info(f"=== Completed cycle {cycle_count} ===")
                    
                    # Wait before starting next cycle (5 minutes)
                    logger.info("Waiting 5 minutes before starting next cycle...")
                    await asyncio.sleep(300)  # 5 minutes
                    
                except asyncio.CancelledError:
                    logger.info("Main task gathering cancelled")
                    raise
                except Exception as e:
                    logger.error(f"Error in cycle {cycle_count}: {str(e)}")
                    # Wait a bit before retrying
                    await asyncio.sleep(60)  # 1 minute
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
    finally:
        # Set running flag to False and trigger shutdown event
        is_running = False
        shutdown_event.set()
        
        # Cancel all tasks with timeout
        await cancel_all_tasks(tasks, timeout=30.0)
        
        # Cleanup services
        await cleanup_services(
            binance_service, telegram_service, discord_service,
            health_monitor, indicator_service, strategy,
            cache_service, cache_monitor
        )
        
        logger.info("Trading Bot with Quantitative Trading Integration stopped")

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the main function
    asyncio.run(main()) 