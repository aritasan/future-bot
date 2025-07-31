#!/usr/bin/env python3
"""
Test script to verify signal generation fixes.
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

async def test_signal_generation_fixes():
    """Test signal generation with fixes applied."""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load config
        config = load_config()
        logger.info("Config loaded successfully")
        
        # Initialize services
        binance_service = BinanceService(config)
        await binance_service.initialize()
        logger.info("Binance service initialized")
        
        indicator_service = IndicatorService(config)
        await indicator_service.initialize()
        logger.info("Indicator service initialized")
        
        notification_service = NotificationService(config)
        await notification_service.initialize()
        logger.info("Notification service initialized")
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config, binance_service, indicator_service, notification_service
        )
        await strategy.initialize()
        logger.info("Strategy initialized")
        
        # Test symbols
        test_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        
        for symbol in test_symbols:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing signal generation for {symbol}")
            logger.info(f"{'='*50}")
            
            # Test 1: Check initial signal history
            initial_history = len(strategy.signal_history.get(symbol, []))
            logger.info(f"Initial signal history for {symbol}: {initial_history}")
            
            # Test 2: Generate signal
            signal = await strategy.generate_signals(symbol, indicator_service)
            
            if signal:
                logger.info(f"✅ Signal generated successfully for {symbol}")
                logger.info(f"   Action: {signal.get('action', 'unknown')}")
                logger.info(f"   Strength: {signal.get('strength', 0):.3f}")
                logger.info(f"   Confidence: {signal.get('confidence', 0):.3f}")
                
                # Check if statistical validation passed
                if 'statistical_validation' in signal:
                    validation = signal['statistical_validation']
                    logger.info(f"   Statistical validation: {validation.get('significance_test', {}).get('significant', False)}")
                    logger.info(f"   P-value: {validation.get('significance_test', {}).get('p_value', 1.0):.4f}")
                
                # Check signal history after generation
                final_history = len(strategy.signal_history.get(symbol, []))
                logger.info(f"   Signal history after generation: {final_history}")
                
            else:
                logger.error(f"❌ Signal generation failed for {symbol}")
            
            # Test 3: Check dynamic thresholds
            market_data = await strategy._get_comprehensive_market_data(symbol)
            dynamic_thresholds = strategy._calculate_dynamic_validation_thresholds(symbol, market_data)
            logger.info(f"Dynamic thresholds for {symbol}: {dynamic_thresholds}")
            
            # Test 4: Accumulate signals
            logger.info(f"Accumulating signals for {symbol}...")
            await strategy._accumulate_signals_for_symbol(symbol)
            
            accumulated_history = len(strategy.signal_history.get(symbol, []))
            logger.info(f"Signal history after accumulation: {accumulated_history}")
            
            # Test 5: Generate signal again after accumulation
            signal_after_accumulation = await strategy.generate_signals(symbol, indicator_service)
            
            if signal_after_accumulation:
                logger.info(f"✅ Signal generated after accumulation for {symbol}")
                logger.info(f"   Action: {signal_after_accumulation.get('action', 'unknown')}")
                logger.info(f"   Strength: {signal_after_accumulation.get('strength', 0):.3f}")
                logger.info(f"   Confidence: {signal_after_accumulation.get('confidence', 0):.3f}")
            else:
                logger.warning(f"⚠️ No signal generated after accumulation for {symbol}")
        
        # Test 6: Check overall improvements
        logger.info(f"\n{'='*50}")
        logger.info("OVERALL IMPROVEMENTS SUMMARY")
        logger.info(f"{'='*50}")
        
        total_signals = sum(len(history) for history in strategy.signal_history.values())
        logger.info(f"Total signals across all symbols: {total_signals}")
        
        successful_signals = 0
        for symbol, history in strategy.signal_history.items():
            if history:
                successful_signals += 1
                logger.info(f"Symbol {symbol}: {len(history)} signals")
        
        logger.info(f"Symbols with signals: {successful_signals}/{len(test_symbols)}")
        
        # Cleanup
        await binance_service.close()
        await indicator_service.close()
        await notification_service.close()
        await strategy.close()
        
        logger.info("✅ All tests completed successfully")
        
    except Exception as e:
        logger.error(f"Error in test script: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(test_signal_generation_fixes()) 