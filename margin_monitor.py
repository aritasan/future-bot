#!/usr/bin/env python3
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
