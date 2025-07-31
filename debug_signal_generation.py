#!/usr/bin/env python3
"""
Debug script to investigate why signal generation is not working.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils.logging_config import setup_logging
from src.core.config import load_config
from src.services.binance_service import BinanceService
from src.services.indicator_service import IndicatorService
from src.services.notification_service import NotificationService
from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative

async def debug_signal_generation():
    """Debug signal generation for a specific symbol."""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load config
        config = load_config()
        logger.info("Config loaded successfully")
        
        # Initialize services
        binance_service = BinanceService(config)
        await binance_service.initialize()  # Initialize binance service
        logger.info("Binance service initialized")
        
        indicator_service = IndicatorService(config)  # Pass config instead of binance_service
        await indicator_service.initialize()  # Initialize indicator service
        logger.info("Indicator service initialized")
        
        notification_service = NotificationService(config)
        await notification_service.initialize()  # Initialize notification service
        logger.info("Notification service initialized")
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config, binance_service, indicator_service, notification_service
        )
        await strategy.initialize()  # Initialize strategy
        logger.info("Strategy initialized")
        
        # Test symbol
        test_symbol = "BTC/USDT"
        logger.info(f"Testing signal generation for {test_symbol}")
        
        # Test step by step
        logger.info("Step 1: Getting comprehensive market data...")
        market_data = await strategy._get_comprehensive_market_data(test_symbol)
        logger.info(f"Market data keys: {list(market_data.keys()) if market_data else 'None'}")
        
        logger.info("Step 2: Getting klines data...")
        klines_1h = await indicator_service.get_klines(test_symbol, '1h', limit=100)
        klines_4h = await indicator_service.get_klines(test_symbol, '4h', limit=100)
        klines_1d = await indicator_service.get_klines(test_symbol, '1d', limit=100)
        
        logger.info(f"Klines 1h: {len(klines_1h) if klines_1h else 0} items")
        logger.info(f"Klines 4h: {len(klines_4h) if klines_4h else 0} items")
        logger.info(f"Klines 1d: {len(klines_1d) if klines_1d else 0} items")
        
        if klines_1h and klines_4h and klines_1d:
            logger.info("Step 3: Converting klines to DataFrames...")
            df_1h = strategy._convert_klines_to_dataframe(klines_1h)
            df_4h = strategy._convert_klines_to_dataframe(klines_4h)
            df_1d = strategy._convert_klines_to_dataframe(klines_1d)
            
            logger.info(f"DataFrame 1h shape: {df_1h.shape}")
            logger.info(f"DataFrame 4h shape: {df_4h.shape}")
            logger.info(f"DataFrame 1d shape: {df_1d.shape}")
            
            if not df_1h.empty and not df_4h.empty and not df_1d.empty:
                logger.info("Step 4: Calculating advanced indicators...")
                try:
                    df_1h = await strategy._calculate_advanced_indicators(df_1h)
                    df_4h = await strategy._calculate_advanced_indicators(df_4h)
                    df_1d = await strategy._calculate_advanced_indicators(df_1d)
                    logger.info("Advanced indicators calculated successfully")
                except Exception as e:
                    logger.error(f"Error calculating advanced indicators: {str(e)}")
                    return
                
                logger.info("Step 5: Creating advanced signal...")
                try:
                    signal = strategy._create_advanced_signal(test_symbol, df_1h, df_4h, df_1d, market_data)
                    logger.info(f"Signal created: {signal}")
                except Exception as e:
                    logger.error(f"Error creating advanced signal: {str(e)}")
                    return
                
                if signal:
                    logger.info("Step 6: Applying quantitative analysis...")
                    try:
                        signal = await strategy._apply_quantitative_analysis(test_symbol, signal, market_data)
                        logger.info("Quantitative analysis applied successfully")
                    except Exception as e:
                        logger.error(f"Error applying quantitative analysis: {str(e)}")
                        return
                    
                    logger.info("Step 7: Applying statistical validation...")
                    try:
                        signal = await strategy._apply_statistical_validation(test_symbol, signal, market_data)
                        logger.info("Statistical validation applied successfully")
                    except Exception as e:
                        logger.error(f"Error applying statistical validation: {str(e)}")
                        return
                    
                    logger.info(f"Final signal: {signal}")
                else:
                    logger.error("Failed to create signal")
            else:
                logger.error("Empty DataFrames after conversion")
        else:
            logger.error("Missing klines data")
        
        # Test generate_signals method
        logger.info("Step 8: Testing generate_signals method...")
        try:
            signals = await strategy.generate_signals(test_symbol, indicator_service)
            if signals:
                logger.info(f"Signal generation successful: {signals}")
            else:
                logger.error("Signal generation returned None")
        except Exception as e:
            logger.error(f"Error in generate_signals: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Cleanup
        await binance_service.close()
        await indicator_service.close()
        await notification_service.close()
        await strategy.close()
        
    except Exception as e:
        logger.error(f"Error in debug script: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(debug_signal_generation()) 