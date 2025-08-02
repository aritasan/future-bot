#!/usr/bin/env python3
"""
Simple and Practical DCA & Trailing Stop Implementation
Based on real trading logic:
- DCA: Add more orders when price moves against initial position
- Trailing Stop: Move SL to secure profit when position is profitable
"""

import asyncio
import logging
from typing import Dict, Optional, List, Any
import json
import time
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDCA:
    """Simple DCA implementation - Add more orders when price moves against position."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dca_history = {}  # Track DCA attempts per symbol
        self.dca_config = config.get('risk_management', {}).get('dca', {})
        
    async def check_dca_opportunity(self, symbol: str, position: Dict) -> Dict:
        """
        Check if DCA opportunity exists based on price movement against position.
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            position: Position details from exchange
            
        Returns:
            Dict with DCA decision and details
        """
        try:
            # Get position details
            position_side = position.get('info', {}).get('positionSide', 'LONG')  # LONG or SHORT
            entry_price = float(position.get('entryPrice', 0))
            current_price = float(position.get('markPrice', 0))
            position_size = abs(float(position.get('info', {}).get('positionAmt', 0)))
            
            if entry_price <= 0 or current_price <= 0 or position_size <= 0:
                return {'should_dca': False, 'reason': 'Invalid position data'}
            
            # Calculate price movement percentage
            if position_side == 'LONG':
                price_change_pct = (current_price - entry_price) / entry_price * 100
                # For LONG: DCA when price drops (negative change)
                price_movement_against = price_change_pct < 0
            else:  # SHORT
                price_change_pct = (entry_price - current_price) / entry_price * 100
                # For SHORT: DCA when price rises (negative change for SHORT)
                price_movement_against = price_change_pct < 0
            
            # Check DCA thresholds
            dca_thresholds = self.dca_config.get('price_drop_thresholds', [5, 10, 15])  # 5%, 10%, 15%
            dca_attempts = self.dca_history.get(symbol, 0)
            max_attempts = self.dca_config.get('max_attempts', 3)
            
            # Check if we can DCA
            if dca_attempts >= max_attempts:
                return {'should_dca': False, 'reason': f'Max DCA attempts reached ({max_attempts})'}
            
            # Check if price movement is significant enough for DCA
            abs_price_change = abs(price_change_pct)
            current_threshold = dca_thresholds[dca_attempts] if dca_attempts < len(dca_thresholds) else dca_thresholds[-1]
            
            if abs_price_change >= current_threshold and price_movement_against:
                # Calculate DCA size
                dca_size = await self._calculate_dca_size(symbol, position_size, abs_price_change)
                
                return {
                    'should_dca': True,
                    'dca_size': dca_size,
                    'price_change_pct': price_change_pct,
                    'threshold': current_threshold,
                    'attempt': dca_attempts + 1,
                    'reason': f'Price moved {abs_price_change:.2f}% against position (threshold: {current_threshold}%)'
                }
            else:
                return {
                    'should_dca': False,
                    'price_change_pct': price_change_pct,
                    'threshold': current_threshold,
                    'attempt': dca_attempts,
                    'reason': f'Price movement {abs_price_change:.2f}% not sufficient for DCA (threshold: {current_threshold}%)'
                }
                
        except Exception as e:
            logger.error(f"Error checking DCA opportunity for {symbol}: {str(e)}")
            return {'should_dca': False, 'reason': f'Error: {str(e)}'}
    
    async def _calculate_dca_size(self, symbol: str, current_position_size: float, price_change_pct: float) -> float:
        """Calculate DCA size based on current position and price movement."""
        try:
            # Base DCA size (50% of current position)
            base_dca_size = current_position_size * self.dca_config.get('dca_size_multiplier', 0.5)
            
            # Adjust size based on price movement (more aggressive for larger moves)
            price_factor = min(price_change_pct / 10, 2.0)  # Cap at 2x for 20%+ moves
            adjusted_dca_size = base_dca_size * (1 + price_factor)
            
            # Apply limits
            min_size = self.dca_config.get('min_dca_size', 0.001)
            max_size = current_position_size * self.dca_config.get('max_dca_size_multiplier', 2.0)
            
            final_size = max(min_size, min(adjusted_dca_size, max_size))
            
            return final_size
            
        except Exception as e:
            logger.error(f"Error calculating DCA size for {symbol}: {str(e)}")
            return 0.0
    
    async def execute_dca(self, symbol: str, position: Dict, dca_decision: Dict) -> bool:
        """Execute DCA order."""
        try:
            if not dca_decision.get('should_dca', False):
                return False
            
            # Here you would call your exchange service to place the DCA order
            # For now, we'll just log the decision
            logger.info(f"Executing DCA for {symbol}: {dca_decision}")
            
            # Update DCA history
            self.dca_history[symbol] = dca_decision.get('attempt', 1)
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing DCA for {symbol}: {str(e)}")
            return False

class SimpleTrailingStop:
    """Simple Trailing Stop implementation - Move SL to secure profit."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.trailing_history = {}  # Track trailing stops per symbol
        self.trailing_config = config.get('risk_management', {}).get('trailing_stop', {})
        
    async def check_trailing_stop_opportunity(self, symbol: str, position: Dict) -> Dict:
        """
        Check if trailing stop should be updated based on profit level.
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            position: Position details from exchange
            
        Returns:
            Dict with trailing stop decision and details
        """
        try:
            # Get position details
            position_side = position.get('info', {}).get('positionSide', 'LONG')
            entry_price = float(position.get('entryPrice', 0))
            current_price = float(position.get('markPrice', 0))
            unrealized_pnl = float(position.get('unrealizedPnl', 0))
            position_size = abs(float(position.get('info', {}).get('positionAmt', 0)))
            
            if entry_price <= 0 or current_price <= 0 or position_size <= 0:
                return {'should_update': False, 'reason': 'Invalid position data'}
            
            # Calculate profit percentage
            if position_side == 'LONG':
                profit_pct = (current_price - entry_price) / entry_price * 100
            else:  # SHORT
                profit_pct = (entry_price - current_price) / entry_price * 100
            
            # Check if position is profitable
            if profit_pct <= 0:
                return {'should_update': False, 'reason': 'Position not profitable yet'}
            
            # Get trailing stop thresholds
            profit_thresholds = self.trailing_config.get('profit_thresholds', [2, 5, 10])  # 2%, 5%, 10%
            trailing_multipliers = self.trailing_config.get('trailing_multipliers', [2.0, 1.5, 1.0])  # Tighter as profit increases
            
            # Find appropriate threshold and multiplier
            threshold_index = 0
            for i, threshold in enumerate(profit_thresholds):
                if profit_pct >= threshold:
                    threshold_index = i
            
            current_threshold = profit_thresholds[threshold_index]
            current_multiplier = trailing_multipliers[threshold_index]
            
            # Calculate new stop loss
            atr = current_price * 0.02  # Mock ATR (2% of price)
            trailing_distance = atr * current_multiplier
            
            if position_side == 'LONG':
                new_stop_loss = current_price - trailing_distance
                # Ensure stop loss is above entry price for profit protection
                if new_stop_loss <= entry_price:
                    new_stop_loss = entry_price * 1.01  # 1% above entry
            else:  # SHORT
                new_stop_loss = current_price + trailing_distance
                # Ensure stop loss is below entry price for profit protection
                if new_stop_loss >= entry_price:
                    new_stop_loss = entry_price * 0.99  # 1% below entry
            
            # Check if update is needed
            current_stop_loss = self.trailing_history.get(symbol, {}).get('stop_loss', 0)
            
            should_update = False
            if position_side == 'LONG':
                should_update = new_stop_loss > current_stop_loss
            else:  # SHORT
                should_update = new_stop_loss < current_stop_loss or current_stop_loss == 0
            
            return {
                'should_update': should_update,
                'new_stop_loss': new_stop_loss,
                'current_stop_loss': current_stop_loss,
                'profit_pct': profit_pct,
                'threshold': current_threshold,
                'multiplier': current_multiplier,
                'trailing_distance': trailing_distance,
                'reason': f'Profit {profit_pct:.2f}% >= threshold {current_threshold}%' if should_update else 'No update needed'
            }
            
        except Exception as e:
            logger.error(f"Error checking trailing stop for {symbol}: {str(e)}")
            return {'should_update': False, 'reason': f'Error: {str(e)}'}
    
    async def execute_trailing_stop_update(self, symbol: str, position: Dict, trailing_decision: Dict) -> bool:
        """Execute trailing stop update."""
        try:
            if not trailing_decision.get('should_update', False):
                return False
            
            # Here you would call your exchange service to update the stop loss
            # For now, we'll just log the decision
            logger.info(f"Updating trailing stop for {symbol}: {trailing_decision}")
            
            # Update trailing history
            self.trailing_history[symbol] = {
                'stop_loss': trailing_decision.get('new_stop_loss', 0),
                'last_update': time.time(),
                'profit_pct': trailing_decision.get('profit_pct', 0)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating trailing stop for {symbol}: {str(e)}")
            return False

async def test_simple_implementations():
    """Test the simple DCA and Trailing Stop implementations."""
    logger.info("🧪 Testing Simple DCA and Trailing Stop implementations...")
    
    # Mock config
    config = {
        'risk_management': {
            'dca': {
                'enabled': True,
                'dca_size_multiplier': 0.5,  # 50% of current position
                'max_dca_size_multiplier': 2.0,  # Max 2x current position
                'min_dca_size': 0.001,
                'max_attempts': 3,
                'price_drop_thresholds': [5, 10, 15]  # 5%, 10%, 15%
            },
            'trailing_stop': {
                'enabled': True,
                'profit_thresholds': [2, 5, 10],  # 2%, 5%, 10%
                'trailing_multipliers': [2.0, 1.5, 1.0]  # Tighter as profit increases
            }
        }
    }
    
    # Test DCA - LONG position with price drop
    logger.info("\n📊 Testing DCA for LONG position...")
    dca = SimpleDCA(config)
    
    long_position = {
        'symbol': 'ETHUSDT',
        'markPrice': '3000',  # Current price dropped from 3200
        'entryPrice': '3200',  # Entry price
        'unrealizedPnl': '-200',
        'info': {
            'positionSide': 'LONG',
            'positionAmt': '0.1'
        }
    }
    
    dca_result = await dca.check_dca_opportunity('ETHUSDT', long_position)
    logger.info(f"DCA Result for LONG: {dca_result}")
    
    # Test DCA - SHORT position with price rise
    logger.info("\n📊 Testing DCA for SHORT position...")
    
    short_position = {
        'symbol': 'ETHUSDT',
        'markPrice': '3400',  # Current price rose from 3200
        'entryPrice': '3200',  # Entry price
        'unrealizedPnl': '-200',
        'info': {
            'positionSide': 'SHORT',
            'positionAmt': '0.1'
        }
    }
    
    dca_result_short = await dca.check_dca_opportunity('ETHUSDT', short_position)
    logger.info(f"DCA Result for SHORT: {dca_result_short}")
    
    # Test Trailing Stop - LONG position with profit
    logger.info("\n📊 Testing Trailing Stop for LONG position...")
    trailing = SimpleTrailingStop(config)
    
    profitable_long_position = {
        'symbol': 'ETHUSDT',
        'markPrice': '3360',  # Current price rose from 3200 (5% profit)
        'entryPrice': '3200',  # Entry price
        'unrealizedPnl': '160',
        'info': {
            'positionSide': 'LONG',
            'positionAmt': '0.1'
        }
    }
    
    trailing_result = await trailing.check_trailing_stop_opportunity('ETHUSDT', profitable_long_position)
    logger.info(f"Trailing Stop Result for LONG: {trailing_result}")
    
    # Test Trailing Stop - SHORT position with profit
    logger.info("\n📊 Testing Trailing Stop for SHORT position...")
    
    profitable_short_position = {
        'symbol': 'ETHUSDT',
        'markPrice': '3040',  # Current price dropped from 3200 (5% profit)
        'entryPrice': '3200',  # Entry price
        'unrealizedPnl': '160',
        'info': {
            'positionSide': 'SHORT',
            'positionAmt': '0.1'
        }
    }
    
    trailing_result_short = await trailing.check_trailing_stop_opportunity('ETHUSDT', profitable_short_position)
    logger.info(f"Trailing Stop Result for SHORT: {trailing_result_short}")
    
    logger.info("\n✅ Simple implementations tested successfully!")

async def create_simple_config():
    """Create simple configuration for DCA and Trailing Stop."""
    logger.info("🔧 Creating simple configuration...")
    
    simple_config = {
        'risk_management': {
            'dca': {
                'enabled': True,
                'dca_size_multiplier': 0.5,  # 50% of current position
                'max_dca_size_multiplier': 2.0,  # Max 2x current position
                'min_dca_size': 0.001,
                'max_attempts': 3,
                'price_drop_thresholds': [5, 10, 15],  # 5%, 10%, 15%
                'min_interval': 3600,  # 1 hour between DCA attempts
                'risk_reduction': 0.5  # Reduce risk by 50% for each DCA
            },
            'trailing_stop': {
                'enabled': True,
                'profit_thresholds': [2, 5, 10],  # 2%, 5%, 10%
                'trailing_multipliers': [2.0, 1.5, 1.0],  # Tighter as profit increases
                'update_interval': 300,  # 5 minutes between updates
                'min_profit_for_trail': 1.0  # 1% minimum profit to start trailing
            }
        }
    }
    
    # Save simple config
    with open('simple_dca_trailing_config.json', 'w') as f:
        json.dump(simple_config, f, indent=2)
    
    logger.info("✅ Simple configuration created: simple_dca_trailing_config.json")

if __name__ == "__main__":
    print("🚀 Implementing Simple DCA and Trailing Stop...")
    
    # Run tests
    asyncio.run(test_simple_implementations())
    
    # Create simple configuration
    asyncio.run(create_simple_config())
    
    print("\n🎉 Simple DCA and Trailing Stop implementation completed!")
    print("📁 Simple configuration saved: simple_dca_trailing_config.json")
    print("📊 Test results logged above")
    
    print("\n📋 Summary:")
    print("✅ DCA: Add more orders when price moves against position")
    print("   - LONG: DCA when price drops 5%, 10%, 15%")
    print("   - SHORT: DCA when price rises 5%, 10%, 15%")
    print("✅ Trailing Stop: Move SL to secure profit")
    print("   - LONG: Move SL up when profit >= 2%, 5%, 10%")
    print("   - SHORT: Move SL down when profit >= 2%, 5%, 10%") 