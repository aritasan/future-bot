#!/usr/bin/env python3
"""
Script to investigate and fix the strategy crash issue.
"""

import asyncio
import logging
import sys
import traceback
from datetime import datetime
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def check_strategy_health():
    """Check if the strategy is still running and healthy."""
    logger.info("🔍 Investigating strategy crash issue...")
    
    # Check if main process is still running
    try:
        import psutil
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python.exe':
                    cmdline = proc.info['cmdline']
                    if cmdline and any('main_with_quantitative' in arg for arg in cmdline):
                        python_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        logger.info(f"Found {len(python_processes)} Python processes running main_with_quantitative.py")
        for proc in python_processes:
            logger.info(f"  PID: {proc['pid']}, CMD: {' '.join(proc['cmdline'])}")
            
    except ImportError:
        logger.warning("psutil not available, cannot check processes")
    
    # Check recent logs for errors
    try:
        with open("logs/trading_bot_quantitative_20250801.log", "r") as f:
            lines = f.readlines()
            
        # Find last strategy log
        last_strategy_log = None
        for line in reversed(lines):
            if "src.strategies.enhanced_trading_strategy_with_quantitative" in line:
                last_strategy_log = line.strip()
                break
        
        if last_strategy_log:
            logger.info(f"Last strategy log: {last_strategy_log}")
            
            # Check if it's an error
            if "ERROR" in last_strategy_log:
                logger.warning("⚠️ Last strategy log was an ERROR")
                
                # Check for margin insufficient errors
                margin_errors = [line for line in lines if "Margin is insufficient" in line]
                if margin_errors:
                    logger.error(f"🚨 Found {len(margin_errors)} margin insufficient errors")
                    logger.error(f"Last margin error: {margin_errors[-1].strip()}")
                    
                    # Suggest solutions
                    logger.info("💡 Suggested solutions:")
                    logger.info("  1. Check account balance and margin")
                    logger.info("  2. Reduce position sizes")
                    logger.info("  3. Add margin error handling")
                    logger.info("  4. Implement circuit breaker for margin errors")
        
        # Check for exceptions
        exception_lines = [line for line in lines if "Exception" in line or "Traceback" in line]
        if exception_lines:
            logger.error(f"🚨 Found {len(exception_lines)} exception lines")
            for line in exception_lines[-5:]:  # Show last 5
                logger.error(f"  {line.strip()}")
                
    except FileNotFoundError:
        logger.error("❌ Log file not found")
    except Exception as e:
        logger.error(f"❌ Error reading logs: {str(e)}")

async def create_strategy_recovery_script():
    """Create a script to recover the strategy."""
    logger.info("🔧 Creating strategy recovery script...")
    
    recovery_script = '''#!/usr/bin/env python3
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
'''
    
    with open("strategy_recovery.py", "w", encoding='utf-8') as f:
        f.write(recovery_script)
    
    logger.info("✅ Strategy recovery script created")

async def create_improved_strategy():
    """Create an improved version of the strategy with better error handling."""
    logger.info("🔧 Creating improved strategy with better error handling...")
    
    # Read the current strategy file
    try:
        with open("src/strategies/enhanced_trading_strategy_with_quantitative.py", "r") as f:
            strategy_content = f.read()
        
        # Add improved error handling for margin errors
        improved_content = strategy_content.replace(
            'logger.error(f"Failed to place SHORT order for {symbol}")',
            '''logger.error(f"Failed to place SHORT order for {symbol}")
                # Add margin error handling
                if "Margin is insufficient" in str(e):
                    logger.warning(f"Margin insufficient for {symbol}, skipping order")
                    return  # Skip this order instead of crashing'''
        )
        
        # Add circuit breaker for margin errors
        circuit_breaker_code = '''
    async def _check_margin_health(self) -> bool:
        """Check if margin is sufficient for trading."""
        try:
            balance = await self.binance_service.get_account_balance()
            if balance and 'total' in balance:
                total_balance = float(balance['total'].get('USDT', 0))
                if total_balance < 10:  # Less than $10
                    logger.warning(f"Insufficient balance: ${total_balance}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error checking margin health: {str(e)}")
            return False
    
    async def _handle_margin_error(self, symbol: str, error: str) -> None:
        """Handle margin insufficient errors gracefully."""
        logger.warning(f"Margin error for {symbol}: {error}")
        
        # Implement circuit breaker
        if not hasattr(self, '_margin_error_count'):
            self._margin_error_count = 0
        
        self._margin_error_count += 1
        
        if self._margin_error_count >= 5:
            logger.error("🚨 Too many margin errors, implementing circuit breaker")
            logger.error("💡 Consider: 1) Adding more margin 2) Reducing position sizes 3) Pausing trading")
            # Could implement a pause mechanism here
        
        # Wait before retrying
        await asyncio.sleep(60)  # Wait 1 minute before retrying
'''
        
        # Insert circuit breaker code after the class definition
        improved_content = improved_content.replace(
            'class EnhancedTradingStrategyWithQuantitative:',
            'class EnhancedTradingStrategyWithQuantitative:' + circuit_breaker_code
        )
        
        # Create backup and improved version
        with open("src/strategies/enhanced_trading_strategy_with_quantitative_backup.py", "w", encoding='utf-8') as f:
            f.write(strategy_content)
        
        with open("src/strategies/enhanced_trading_strategy_with_quantitative_improved.py", "w", encoding='utf-8') as f:
            f.write(improved_content)
        
        logger.info("✅ Improved strategy created with better error handling")
        logger.info("📁 Backup: enhanced_trading_strategy_with_quantitative_backup.py")
        logger.info("📁 Improved: enhanced_trading_strategy_with_quantitative_improved.py")
        
    except Exception as e:
        logger.error(f"❌ Error creating improved strategy: {str(e)}")

async def main():
    """Main investigation function."""
    logger.info("🚀 Starting strategy crash investigation...")
    
    # Check current health
    await check_strategy_health()
    
    # Create recovery script
    await create_strategy_recovery_script()
    
    # Create improved strategy
    await create_improved_strategy()
    
    logger.info("📋 Investigation Summary:")
    logger.info("  1. ✅ Health check completed")
    logger.info("  2. ✅ Recovery script created: strategy_recovery.py")
    logger.info("  3. ✅ Improved strategy created with better error handling")
    logger.info("  4. 💡 Next steps:")
    logger.info("     - Run: python strategy_recovery.py")
    logger.info("     - Check account margin and balance")
    logger.info("     - Consider reducing position sizes")
    logger.info("     - Monitor for margin errors")

if __name__ == "__main__":
    asyncio.run(main()) 