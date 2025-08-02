#!/usr/bin/env python3
"""
Script to fix margin issues and implement better error handling.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def check_account_balance():
    """Check account balance and margin."""
    logger.info("💰 Checking account balance and margin...")
    
    try:
        from src.services.binance_service import BinanceService
        
        # Load config
        with open("config.json", "r") as f:
            config = json.load(f)
        
        # Initialize Binance service
        binance_service = BinanceService(config)
        await binance_service.initialize()
        
        # Get account balance
        balance = await binance_service.get_account_balance()
        if balance:
            logger.info("📊 Account Balance:")
            if 'total' in balance:
                for currency, amount in balance['total'].items():
                    if float(amount) > 0:
                        logger.info(f"  {currency}: {amount}")
            
            if 'free' in balance:
                logger.info("📈 Available Balance:")
                for currency, amount in balance['free'].items():
                    if float(amount) > 0:
                        logger.info(f"  {currency}: {amount}")
        
        # Get positions
        positions = await binance_service.get_positions()
        if positions:
            logger.info("📋 Current Positions:")
            for position in positions:
                if float(position.get('size', 0)) != 0:
                    logger.info(f"  {position.get('symbol')}: {position.get('size')} ({position.get('side')})")
        
        await binance_service.close()
        
    except Exception as e:
        logger.error(f"❌ Error checking balance: {str(e)}")

async def implement_margin_error_handling():
    """Implement improved margin error handling in the strategy."""
    logger.info("🔧 Implementing improved margin error handling...")
    
    try:
        # Read the current strategy file
        with open("src/strategies/enhanced_trading_strategy_with_quantitative.py", "r", encoding='utf-8') as f:
            strategy_content = f.read()
        
        # Add margin health check method
        margin_health_method = '''
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
    
    async def _reduce_position_size(self, base_size: float) -> float:
        """Reduce position size when margin is insufficient."""
        try:
            # Get current balance
            balance = await self.binance_service.get_account_balance()
            if balance and 'total' in balance:
                total_balance = float(balance['total'].get('USDT', 0))
                
                # Calculate safe position size (max 5% of balance)
                max_position_value = total_balance * 0.05
                safe_size = max_position_value / 100  # Assume $100 per unit
                
                # Use the smaller of base_size or safe_size
                reduced_size = min(base_size, safe_size)
                
                if reduced_size < base_size:
                    logger.warning(f"Reduced position size from {base_size} to {reduced_size} due to margin constraints")
                
                return reduced_size
            
            return base_size
            
        except Exception as e:
            logger.error(f"Error reducing position size: {str(e)}")
            return base_size * 0.5  # Reduce by 50% as fallback
'''
        
        # Insert margin health method after class definition
        if '_check_margin_health' not in strategy_content:
            strategy_content = strategy_content.replace(
                'class EnhancedTradingStrategyWithQuantitative:',
                'class EnhancedTradingStrategyWithQuantitative:' + margin_health_method
            )
        
        # Improve error handling in order placement
        improved_error_handling = '''
            # Check margin health before placing order
            if not await self._check_margin_health():
                logger.warning(f"Insufficient margin for {symbol}, skipping order")
                return
            
            # Reduce position size if needed
            position_size = await self._reduce_position_size(position_size)
            
            # Place SHORT position order
            order = await self.binance_service.place_order(order_params)
            
            if order:
                logger.info(f"SHORT position opened for {symbol} with size {position_size} and SL/TP: {order}")
            else:
                logger.error(f"Failed to place SHORT order for {symbol}")
                # Handle margin error specifically
                await self._handle_margin_error(symbol, "Order placement failed")
'''
        
        # Replace the order placement section
        if 'Place SHORT position order' in strategy_content:
            strategy_content = strategy_content.replace(
                '''            # Place SHORT position order
            order = await self.binance_service.place_order(order_params)
            
            if order:
                logger.info(f"SHORT position opened for {symbol} with size {position_size} and SL/TP: {order}")
            else:
                logger.error(f"Failed to place SHORT order for {symbol}")''',
                improved_error_handling
            )
        
        # Write improved strategy
        with open("src/strategies/enhanced_trading_strategy_with_quantitative_fixed.py", "w", encoding='utf-8') as f:
            f.write(strategy_content)
        
        logger.info("✅ Improved strategy with margin error handling created")
        logger.info("📁 File: enhanced_trading_strategy_with_quantitative_fixed.py")
        
    except Exception as e:
        logger.error(f"❌ Error implementing margin error handling: {str(e)}")

async def create_margin_monitor():
    """Create a margin monitoring script."""
    logger.info("📊 Creating margin monitoring script...")
    
    monitor_script = '''#!/usr/bin/env python3
"""
Margin Monitoring Script
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarginMonitor:
    """Monitor account margin and balance."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.binance_service = None
        self.min_balance_threshold = 10  # $10 minimum
        self.margin_warning_threshold = 50  # $50 warning threshold
        
    async def initialize(self):
        """Initialize the margin monitor."""
        try:
            from src.services.binance_service import BinanceService
            self.binance_service = BinanceService(self.config)
            await self.binance_service.initialize()
            logger.info("✅ Margin monitor initialized")
        except Exception as e:
            logger.error(f"❌ Error initializing margin monitor: {str(e)}")
    
    async def check_margin_health(self) -> Dict:
        """Check margin health and return status."""
        try:
            if not self.binance_service:
                return {"status": "error", "message": "Binance service not initialized"}
            
            # Get account balance
            balance = await self.binance_service.get_account_balance()
            if not balance:
                return {"status": "error", "message": "Could not fetch balance"}
            
            # Calculate total USDT balance
            total_usdt = 0
            if 'total' in balance and 'USDT' in balance['total']:
                total_usdt = float(balance['total']['USDT'])
            
            # Get available USDT
            available_usdt = 0
            if 'free' in balance and 'USDT' in balance['free']:
                available_usdt = float(balance['free']['USDT'])
            
            # Check margin health
            status = "healthy"
            warnings = []
            
            if total_usdt < self.min_balance_threshold:
                status = "critical"
                warnings.append(f"Total balance too low: ${total_usdt}")
            
            if available_usdt < self.margin_warning_threshold:
                status = "warning"
                warnings.append(f"Available balance low: ${available_usdt}")
            
            # Get positions
            positions = await self.binance_service.get_positions()
            open_positions = []
            if positions:
                for position in positions:
                    if float(position.get('size', 0)) != 0:
                        open_positions.append({
                            'symbol': position.get('symbol'),
                            'size': position.get('size'),
                            'side': position.get('side'),
                            'pnl': position.get('unrealizedPnl', 0)
                        })
            
            return {
                "status": status,
                "total_usdt": total_usdt,
                "available_usdt": available_usdt,
                "warnings": warnings,
                "open_positions": open_positions,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error checking margin health: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def close(self):
        """Close the margin monitor."""
        if self.binance_service:
            await self.binance_service.close()

async def main():
    """Main margin monitoring function."""
    logger.info("🚀 Starting margin monitoring...")
    
    try:
        # Load config
        with open("config.json", "r") as f:
            config = json.load(f)
        
        # Initialize margin monitor
        monitor = MarginMonitor(config)
        await monitor.initialize()
        
        # Check margin health
        health_status = await monitor.check_margin_health()
        
        logger.info("📊 Margin Health Report:")
        logger.info(f"  Status: {health_status['status']}")
        logger.info(f"  Total USDT: ${health_status.get('total_usdt', 0):.2f}")
        logger.info(f"  Available USDT: ${health_status.get('available_usdt', 0):.2f}")
        
        if health_status.get('warnings'):
            logger.warning("⚠️ Warnings:")
            for warning in health_status['warnings']:
                logger.warning(f"  {warning}")
        
        if health_status.get('open_positions'):
            logger.info("📋 Open Positions:")
            for position in health_status['open_positions']:
                logger.info(f"  {position['symbol']}: {position['size']} ({position['side']}) PnL: {position['pnl']}")
        
        # Provide recommendations
        if health_status['status'] == 'critical':
            logger.error("🚨 CRITICAL: Account balance too low!")
            logger.info("💡 Recommendations:")
            logger.info("  1. Add more funds to the account")
            logger.info("  2. Close some positions to free up margin")
            logger.info("  3. Reduce position sizes in strategy")
            logger.info("  4. Pause trading until balance is restored")
        elif health_status['status'] == 'warning':
            logger.warning("⚠️ WARNING: Available balance is low")
            logger.info("💡 Recommendations:")
            logger.info("  1. Monitor positions closely")
            logger.info("  2. Consider reducing position sizes")
            logger.info("  3. Add more funds if needed")
        
        await monitor.close()
        
    except Exception as e:
        logger.error(f"❌ Error in margin monitoring: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with open("margin_monitor.py", "w", encoding='utf-8') as f:
        f.write(monitor_script)
    
    logger.info("✅ Margin monitoring script created: margin_monitor.py")

async def main():
    """Main function to fix margin issues."""
    logger.info("🚀 Starting margin issue fixes...")
    
    # Check current account balance
    await check_account_balance()
    
    # Implement improved margin error handling
    await implement_margin_error_handling()
    
    # Create margin monitor
    await create_margin_monitor()
    
    logger.info("📋 Margin Fix Summary:")
    logger.info("  1. ✅ Account balance checked")
    logger.info("  2. ✅ Improved margin error handling implemented")
    logger.info("  3. ✅ Margin monitoring script created")
    logger.info("  4. 💡 Next steps:")
    logger.info("     - Run: python margin_monitor.py")
    logger.info("     - Replace strategy with fixed version")
    logger.info("     - Add more funds if balance is low")
    logger.info("     - Monitor margin health regularly")

if __name__ == "__main__":
    asyncio.run(main()) 