"""
WorldQuant DCA and Trailing Stop Implementation
Integrated with quantitative trading strategy.
"""

import asyncio
import logging
from typing import Dict, Optional, List, Any
import numpy as np
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class WorldQuantDCA:
    """WorldQuant-level DCA implementation with quantitative analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dca_history = {}
        self.dca_config = config.get('risk_management', {}).get('dca', {})
        
    async def check_dca_opportunity(self, symbol: str, position: Dict, market_data: Dict) -> Dict:
        """Check if DCA opportunity exists based on price movement against position."""
        try:
            # Get position details
            position_side = position.get('info', {}).get('positionSide', 'LONG')
            entry_price = float(position.get('entryPrice', 0))
            current_price = float(position.get('markPrice', 0))
            position_size = abs(float(position.get('info', {}).get('positionAmt', 0)))
            
            if entry_price <= 0 or current_price <= 0 or position_size <= 0:
                return {'should_dca': False, 'reason': 'Invalid position data'}
            
            # Calculate price movement percentage
            if position_side == 'LONG':
                price_change_pct = (current_price - entry_price) / entry_price * 100
                price_movement_against = price_change_pct < 0
            else:  # SHORT
                price_change_pct = (entry_price - current_price) / entry_price * 100
                price_movement_against = price_change_pct < 0
            
            # Check DCA thresholds
            dca_thresholds = self.dca_config.get('price_drop_thresholds', [5, 10, 15])
            dca_attempts = self.dca_history.get(symbol, 0)
            max_attempts = self.dca_config.get('max_attempts', 3)
            
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
            
            # Adjust size based on price movement
            price_factor = min(price_change_pct / 10, 2.0)
            adjusted_dca_size = base_dca_size * (1 + price_factor)
            
            # Apply limits
            min_size = self.dca_config.get('min_dca_size', 0.001)
            max_size = current_position_size * self.dca_config.get('max_dca_size_multiplier', 2.0)
            
            final_size = max(min_size, min(adjusted_dca_size, max_size))
            
            return final_size
            
        except Exception as e:
            logger.error(f"Error calculating DCA size for {symbol}: {str(e)}")
            return 0.0
    
    async def execute_dca(self, symbol: str, position: Dict, dca_decision: Dict, binance_service) -> bool:
        """Execute DCA order."""
        try:
            if not dca_decision.get('should_dca', False):
                return False
            
            # Prepare DCA order parameters
            position_side = position.get('info', {}).get('positionSide', 'LONG')
            dca_size = dca_decision.get('dca_size', 0)
            
            order_params = {
                'symbol': symbol,
                'side': 'BUY' if position_side == 'LONG' else 'SELL',
                'type': 'MARKET',
                'positionSide': position_side,
                'amount': dca_size,
                'isDCA': True  # Flag to bypass existing order check
            }
            
            # Place DCA order
            order = await binance_service.place_order(order_params)
            
            if order:
                logger.info(f"DCA executed for {symbol}: {dca_decision}")
                
                # Calculate new SL and TP levels after DCA
                await self._update_sl_tp_after_dca(symbol, position, dca_decision, binance_service)
                
                # Update DCA history
                self.dca_history[symbol] = dca_decision.get('attempt', 1)
                
                return True
            else:
                logger.error(f"Failed to execute DCA for {symbol}")
                return False
            
        except Exception as e:
            logger.error(f"Error executing DCA for {symbol}: {str(e)}")
            return False
    
    async def _update_sl_tp_after_dca(self, symbol: str, position: Dict, dca_decision: Dict, binance_service) -> None:
        """Update SL and TP levels after DCA execution."""
        try:
            # Get current position details
            current_position_size = abs(float(position.get('info', {}).get('positionAmt', 0)))
            entry_price = float(position.get('entryPrice', 0))
            dca_size = dca_decision.get('dca_size', 0)
            
            # Get current market price for DCA entry
            current_price = await binance_service.get_current_price(symbol)
            if not current_price:
                logger.error(f"Could not get current price for {symbol} to calculate new SL/TP")
                return
            
            # Calculate new average entry price
            total_position_size = current_position_size + dca_size
            if total_position_size > 0:
                new_average_entry = ((current_position_size * entry_price) + (dca_size * current_price)) / total_position_size
            else:
                logger.error(f"Invalid position size for {symbol}")
                return
            
            logger.info(f"DCA: New average entry price for {symbol}: {new_average_entry:.2f}")
            
            # Calculate new SL and TP levels
            new_sl, new_tp = await self._calculate_new_sl_tp(symbol, position, new_average_entry, total_position_size, binance_service)
            
            if new_sl and new_tp:
                # Update SL
                sl_success = await binance_service._update_stop_loss(symbol, position, new_sl)
                if sl_success:
                    logger.info(f"DCA: Updated SL for {symbol} to {new_sl:.2f}")
                else:
                    logger.error(f"DCA: Failed to update SL for {symbol}")
                
                # Update TP
                tp_success = await binance_service._update_take_profit(symbol, position, new_tp)
                if tp_success:
                    logger.info(f"DCA: Updated TP for {symbol} to {new_tp:.2f}")
                else:
                    logger.error(f"DCA: Failed to update TP for {symbol}")
                
                # Log the DCA summary
                logger.info(f"DCA Summary for {symbol}:")
                logger.info(f"  - Original position: {current_position_size} @ {entry_price:.2f}")
                logger.info(f"  - DCA addition: {dca_size} @ {current_price:.2f}")
                logger.info(f"  - New average: {total_position_size} @ {new_average_entry:.2f}")
                logger.info(f"  - New SL: {new_sl:.2f}, New TP: {new_tp:.2f}")
            else:
                logger.error(f"DCA: Failed to calculate new SL/TP for {symbol}")
                
        except Exception as e:
            logger.error(f"Error updating SL/TP after DCA for {symbol}: {str(e)}")
    
    async def _calculate_new_sl_tp(self, symbol: str, position: Dict, new_average_entry: float, total_position_size: float, binance_service) -> tuple:
        """Calculate new SL and TP levels after DCA."""
        try:
            position_side = position.get('info', {}).get('positionSide', 'LONG')
            
            # Get current market data for ATR calculation
            klines = await binance_service.get_klines(symbol, timeframe='1h', limit=20)
            if not klines or len(klines) < 14:
                logger.warning(f"Insufficient kline data for {symbol}, using default ATR")
                atr = new_average_entry * 0.02  # 2% default ATR
            else:
                # Calculate ATR (Average True Range)
                atr = self._calculate_atr(klines)
            
            # Get current SL and TP levels
            current_sl = await binance_service.get_stop_price(symbol, position_side, 'STOP_MARKET')
            current_tp = await binance_service.get_stop_price(symbol, position_side, 'TAKE_PROFIT_MARKET')
            
            # Calculate new SL and TP based on position side
            if position_side == 'LONG':
                # For LONG positions
                if current_sl:
                    # Keep the better SL (higher one)
                    new_sl = max(current_sl, new_average_entry - (atr * 2))
                else:
                    new_sl = new_average_entry - (atr * 2)
                
                if current_tp:
                    # Keep the better TP (higher one)
                    new_tp = max(current_tp, new_average_entry + (atr * 3))
                else:
                    new_tp = new_average_entry + (atr * 3)
                    
            else:  # SHORT
                # For SHORT positions
                if current_sl:
                    # Keep the better SL (lower one)
                    new_sl = min(current_sl, new_average_entry + (atr * 2))
                else:
                    new_sl = new_average_entry + (atr * 2)
                
                if current_tp:
                    # Keep the better TP (lower one)
                    new_tp = min(current_tp, new_average_entry - (atr * 3))
                else:
                    new_tp = new_average_entry - (atr * 3)
            
            # Ensure SL and TP are reasonable
            if position_side == 'LONG':
                if new_sl >= new_average_entry:
                    new_sl = new_average_entry - (atr * 1.5)
                if new_tp <= new_average_entry:
                    new_tp = new_average_entry + (atr * 2)
            else:  # SHORT
                if new_sl <= new_average_entry:
                    new_sl = new_average_entry + (atr * 1.5)
                if new_tp >= new_average_entry:
                    new_tp = new_average_entry - (atr * 2)
            
            return new_sl, new_tp
            
        except Exception as e:
            logger.error(f"Error calculating new SL/TP for {symbol}: {str(e)}")
            return None, None
    
    def _calculate_atr(self, klines: list, period: int = 14) -> float:
        """Calculate Average True Range (ATR)."""
        try:
            if len(klines) < period + 1:
                return klines[-1][4] * 0.02  # 2% of current price as fallback
            
            true_ranges = []
            for i in range(1, len(klines)):
                high = float(klines[i][2])
                low = float(klines[i][3])
                prev_close = float(klines[i-1][4])
                
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                
                true_range = max(tr1, tr2, tr3)
                true_ranges.append(true_range)
            
            # Calculate ATR as simple moving average of true ranges
            atr = sum(true_ranges[-period:]) / period
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return 0.0

class WorldQuantTrailingStop:
    """WorldQuant-level trailing stop implementation with quantitative analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.trailing_history = {}
        self.trailing_config = config.get('risk_management', {}).get('trailing_stop', {})
        
    async def check_trailing_stop_opportunity(self, symbol: str, position: Dict, market_data: Dict) -> Dict:
        """Check if trailing stop should be updated based on profit level."""
        try:
            # Get position details
            position_side = position.get('info', {}).get('positionSide', 'LONG')
            entry_price = float(position.get('entryPrice', 0))
            current_price = float(position.get('markPrice', 0))
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
            profit_thresholds = self.trailing_config.get('profit_thresholds', [2, 5, 10])
            trailing_multipliers = self.trailing_config.get('trailing_multipliers', [2.0, 1.5, 1.0])
            
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
    
    async def execute_trailing_stop_update(self, symbol: str, position: Dict, trailing_decision: Dict, binance_service) -> bool:
        """Execute trailing stop update."""
        try:
            if not trailing_decision.get('should_update', False):
                return False
            
            new_stop_loss = trailing_decision.get('new_stop_loss', 0)
            if new_stop_loss <= 0:
                logger.error(f"Invalid new stop loss price: {new_stop_loss}")
                return False
            
            logger.info(f"Updating trailing stop for {symbol}: {trailing_decision}")
            
            # Call binance_service to update stop loss
            success = await binance_service._update_stop_loss(symbol, position, new_stop_loss)
            
            if success:
                logger.info(f"Successfully updated trailing stop for {symbol} to {new_stop_loss}")
                
                # Update trailing history
                self.trailing_history[symbol] = {
                    'stop_loss': new_stop_loss,
                    'last_update': time.time(),
                    'profit_pct': trailing_decision.get('profit_pct', 0)
                }
                
                return True
            else:
                logger.error(f"Failed to update trailing stop for {symbol}")
                return False
            
        except Exception as e:
            logger.error(f"Error updating trailing stop for {symbol}: {str(e)}")
            return False 