#!/usr/bin/env python3
"""
Test script to verify signal boosting functionality.
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

async def test_signal_boosting():
    """Test signal boosting functionality."""
    
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
        
        # Test symbol
        test_symbol = "BTC/USDT"
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing signal boosting for {test_symbol}")
        logger.info(f"{'='*60}")
        
        # Test 1: Check initial signal history
        initial_history = len(strategy.signal_history.get(test_symbol, []))
        logger.info(f"Initial signal history: {initial_history}")
        
        # Test 2: Check dynamic thresholds before boosting
        market_data = await strategy._get_comprehensive_market_data(test_symbol)
        dynamic_thresholds = strategy._calculate_dynamic_validation_thresholds(test_symbol, market_data)
        logger.info(f"Dynamic thresholds before boosting: {dynamic_thresholds}")
        
        # Test 3: Generate signal without boosting
        logger.info("\n--- Testing signal generation WITHOUT boosting ---")
        signal_without_boost = await strategy.generate_signals(test_symbol, indicator_service)
        
        if signal_without_boost:
            logger.info(f"✅ Signal generated WITHOUT boosting:")
            logger.info(f"   Confidence: {signal_without_boost.get('confidence', 0):.3f}")
            logger.info(f"   Strength: {signal_without_boost.get('strength', 0):.3f}")
            logger.info(f"   Action: {signal_without_boost.get('action', 'unknown')}")
            
            # Check boost metrics
            if 'quality_metrics' in signal_without_boost and 'boost_applied' in signal_without_boost['quality_metrics']:
                boost_metrics = signal_without_boost['quality_metrics']['boost_applied']
                logger.info(f"   Boost multiplier: {boost_metrics.get('boost_multiplier', 1.0):.2f}x")
                logger.info(f"   Original confidence: {boost_metrics.get('original_confidence', 0):.3f}")
                logger.info(f"   Boosted confidence: {boost_metrics.get('boosted_confidence', 0):.3f}")
        else:
            logger.error("❌ No signal generated WITHOUT boosting")
        
        # Test 4: Test signal boosting directly
        logger.info("\n--- Testing signal boosting directly ---")
        
        # Create a mock signal with low confidence
        mock_signal = {
            'action': 'hold',
            'strength': 0.05,
            'confidence': 0.1,  # Low confidence
            'symbol': test_symbol,
            'timeframes': {
                '1h': {'confidence': 0.1, 'strength': 0.05},
                '4h': {'confidence': 0.12, 'strength': 0.06},
                '1d': {'confidence': 0.08, 'strength': 0.04}
            },
            'quality_metrics': {}
        }
        
        # Test boosting
        boosted_signal = await strategy._boost_signal_quality(mock_signal, market_data, test_symbol)
        
        logger.info(f"Original signal confidence: {mock_signal.get('confidence', 0):.3f}")
        logger.info(f"Boosted signal confidence: {boosted_signal.get('confidence', 0):.3f}")
        
        if 'quality_metrics' in boosted_signal and 'boost_applied' in boosted_signal['quality_metrics']:
            boost_metrics = boosted_signal['quality_metrics']['boost_applied']
            logger.info(f"Boost multiplier: {boost_metrics.get('boost_multiplier', 1.0):.2f}x")
            logger.info(f"Boost reason: {boost_metrics.get('boost_reason', 'Unknown')}")
        
        # Test 5: Test with different history sizes
        logger.info("\n--- Testing with different history sizes ---")
        
        # Simulate different history sizes
        test_cases = [
            ("New symbol (0 history)", 0),
            ("New symbol (5 history)", 5),
            ("New symbol (15 history)", 15),
            ("Established symbol (50 history)", 50)
        ]
        
        for case_name, history_size in test_cases:
            logger.info(f"\n{case_name}:")
            
            # Create mock history
            strategy.signal_history[test_symbol] = [{'timestamp': '2025-01-01'} for _ in range(history_size)]
            
            # Test thresholds
            thresholds = strategy._calculate_dynamic_validation_thresholds(test_symbol, market_data)
            logger.info(f"   Confidence threshold: {thresholds['confidence_threshold']:.3f}")
            logger.info(f"   Strength threshold: {thresholds['strength_threshold']:.3f}")
            
            # Test boosting
            mock_signal['confidence'] = 0.1  # Reset to low confidence
            boosted = await strategy._boost_signal_quality(mock_signal, market_data, test_symbol)
            logger.info(f"   Boosted confidence: {boosted.get('confidence', 0):.3f}")
            
            # Check if boosted signal passes thresholds
            passes_confidence = boosted.get('confidence', 0) >= thresholds['confidence_threshold']
            passes_strength = abs(boosted.get('strength', 0)) >= thresholds['strength_threshold']
            
            logger.info(f"   Passes confidence threshold: {'✅' if passes_confidence else '❌'}")
            logger.info(f"   Passes strength threshold: {'✅' if passes_strength else '❌'}")
        
        # Cleanup
        await binance_service.close()
        await indicator_service.close()
        await notification_service.close()
        await strategy.close()
        
        logger.info(f"\n{'='*60}")
        logger.info("✅ Signal boosting test completed successfully")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error in test script: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(test_signal_boosting()) 