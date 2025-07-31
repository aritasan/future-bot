#!/usr/bin/env python3
"""
Test script to verify performance metrics fix.
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import load_config
from src.services.binance_service import BinanceService
from src.services.indicator_service import IndicatorService
from src.services.notification_service import NotificationService
from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_performance_metrics_fix():
    """Test the performance metrics fix."""
    try:
        logger.info("Testing performance metrics fix...")
        
        # Load config
        config = load_config()
        
        # Initialize services
        binance_service = BinanceService(config)
        indicator_service = IndicatorService(config)
        notification_service = NotificationService(config)
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config, binance_service, indicator_service, notification_service
        )
        
        # Initialize strategy
        await strategy.initialize()
        
        # Add some test signals to history
        test_signals = [
            {
                'action': 'buy',
                'confidence': 0.7,
                'strength': 0.8,
                'quantitative_confidence': 0.6,
                'timestamp': '2025-08-01T06:00:00'
            },
            {
                'action': 'sell',
                'confidence': 0.6,
                'strength': 0.7,
                'quantitative_confidence': 0.5,
                'timestamp': '2025-08-01T06:01:00'
            },
            {
                'action': 'buy',
                'confidence': 0.8,
                'strength': 0.9,
                'quantitative_confidence': 0.7,
                'timestamp': '2025-08-01T06:02:00'
            }
        ]
        
        # Store test signals
        for i, signal in enumerate(test_signals):
            strategy._store_signal_history(f'TEST{i}', signal)
        
        logger.info(f"Stored {len(test_signals)} test signals")
        logger.info(f"Signal history size: {len(strategy.signal_history)}")
        
        # Test performance metrics
        metrics = await strategy.get_performance_metrics()
        
        logger.info("Performance metrics retrieved successfully:")
        logger.info(f"Signal history count: {metrics.get('signal_history_count', 0)}")
        logger.info(f"Signal success rate: {metrics.get('signal_success_rate', 0):.3f}")
        logger.info(f"Quantitative integration status: {metrics.get('quantitative_integration_status', 'unknown')}")
        
        # Test confidence analytics
        if 'confidence_analytics' in metrics:
            conf_analytics = metrics['confidence_analytics']
            logger.info(f"Buy executions: {conf_analytics.get('buy_executions', 0)}")
            logger.info(f"Sell executions: {conf_analytics.get('sell_executions', 0)}")
            logger.info(f"Buy success rate: {conf_analytics.get('buy_success_rate', 0):.3f}")
            logger.info(f"Sell success rate: {conf_analytics.get('sell_success_rate', 0):.3f}")
        
        # Close strategy
        await strategy.close()
        
        logger.info("Performance metrics test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in performance metrics test: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_performance_metrics_fix())
    if success:
        logger.info("Performance metrics fix verified successfully")
    else:
        logger.error("Performance metrics fix verification failed")
        sys.exit(1) 