#!/usr/bin/env python3
"""
Strategy Recovery Script
"""

import asyncio
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def recover_strategy():
    """Recover the strategy with improved error handling."""
    try:
        logger.info("🔄 Starting strategy recovery...")
        
        # Import strategy components
        from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
        from src.services.binance_service import BinanceService
        from src.services.indicator_service import IndicatorService
        from src.services.notification_service import NotificationService
        from src.services.cache_service import CacheService
        
        # Load config
        import json
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            logger.error("❌ config.json not found")
            return False
        
        # Initialize services with improved error handling
        binance_service = None
        indicator_service = None
        notification_service = None
        cache_service = None
        strategy = None
        
        try:
            # Initialize Binance service
            logger.info("📡 Initializing Binance service...")
            binance_service = BinanceService(config)
            await binance_service.initialize()
            logger.info("✅ Binance service initialized")
            
            # Initialize other services
            logger.info("📊 Initializing indicator service...")
            indicator_service = IndicatorService(config)
            await indicator_service.initialize()
            logger.info("✅ Indicator service initialized")
            
            logger.info("💬 Initializing notification service...")
            notification_service = NotificationService(config, None, None)
            await notification_service.initialize()
            logger.info("✅ Notification service initialized")
            
            logger.info("💾 Initializing cache service...")
            cache_service = CacheService(config)
            await cache_service.initialize()
            logger.info("✅ Cache service initialized")
            
            # Set notification callback
            binance_service.set_notification_callback(notification_service.send_message)
            logger.info("✅ Notification callback set")
            
            # Initialize strategy with improved error handling
            logger.info("🎯 Initializing enhanced trading strategy...")
            strategy = EnhancedTradingStrategyWithQuantitative(
                config, binance_service, indicator_service, notification_service, cache_service
            )
            await strategy.initialize()
            logger.info("✅ Strategy initialized successfully")
            
            # Test strategy functionality
            logger.info("🧪 Testing strategy functionality...")
            
            # Test signal generation
            test_symbol = "BTCUSDT"
            signals = await strategy.generate_signals(test_symbol, indicator_service)
            if signals:
                logger.info(f"✅ Signal generation test passed for {test_symbol}")
            else:
                logger.warning(f"⚠️ No signals generated for {test_symbol}")
            
            # Test quantitative analysis
            try:
                recommendations = await strategy.get_quantitative_recommendations(test_symbol)
                if recommendations:
                    logger.info(f"✅ Quantitative analysis test passed for {test_symbol}")
                else:
                    logger.warning(f"⚠️ No quantitative recommendations for {test_symbol}")
            except Exception as e:
                logger.error(f"❌ Quantitative analysis test failed: {str(e)}")
            
            logger.info("🎉 Strategy recovery completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during strategy recovery: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
            
        finally:
            # Cleanup
            if strategy:
                try:
                    await strategy.close()
                    logger.info("✅ Strategy closed")
                except Exception as e:
                    logger.error(f"❌ Error closing strategy: {str(e)}")
            
            if binance_service:
                try:
                    await binance_service.close()
                    logger.info("✅ Binance service closed")
                except Exception as e:
                    logger.error(f"❌ Error closing Binance service: {str(e)}")

async def main():
    """Main recovery function."""
    logger.info("🚀 Starting strategy investigation and recovery...")
    
    # Check current health
    await check_strategy_health()
    
    # Create recovery script
    await create_strategy_recovery_script()
    
    logger.info("📝 Recovery script created: strategy_recovery.py")
    logger.info("💡 To recover the strategy, run: python strategy_recovery.py")

if __name__ == "__main__":
    asyncio.run(main())
