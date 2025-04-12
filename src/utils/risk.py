"""
Risk management utilities for the trading bot.
"""
import logging
from typing import Dict, Optional
import numpy as np
import ccxt

logger = logging.getLogger(__name__)

class RiskManager:
    """Class for managing trading risk."""
    
    def __init__(self, config: Dict):
        """
        Initialize risk manager.
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.initial_balance = None
        self.position_history = {}
        self.price_history = {}
        
    async def check_risk(self, symbol: str, position_size: float) -> bool:
        """
        Check if a trade meets risk requirements.
        
        Args:
            symbol: Trading pair symbol
            position_size: Proposed position size
            
        Returns:
            bool: True if trade meets risk requirements, False otherwise
        """
        try:
            # Check position size
            if not await self.check_position_size(position_size):
                return False
                
            # Check correlation with existing positions
            if not await self.check_correlation(symbol):
                return False
                
            # Check market volatility
            if not await self.check_volatility(symbol):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk for {symbol}: {str(e)}")
            return False
            
    async def check_position_size(self, position_size: float) -> bool:
        """
        Check if position size is within limits.
        
        Args:
            position_size: Proposed position size
            
        Returns:
            bool: True if position size is acceptable, False otherwise
        """
        try:
            # Get current balance
            balance = await self.get_account_balance()
            if not balance:
                return False
                
            # Calculate position size
            calculated_position_size = await self.calculate_position_size(balance, self.config['trading']['order_risk_percent'], position_size)
            
            # Check against maximum position size
            max_position_size = balance * self.config['risk_management']['max_position_size']
            if calculated_position_size > max_position_size:
                logger.warning(f"Position size {calculated_position_size} exceeds maximum {max_position_size}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking position size: {str(e)}")
            return False
            
    async def check_correlation(self, symbol: str) -> bool:
        """
        Check correlation with existing positions.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            bool: True if correlation is acceptable, False otherwise
        """
        try:
            # Get existing positions
            positions = await self.get_positions()
            if not positions:
                return True
                
            # Calculate correlation
            correlation = await self.calculate_correlation(symbol, positions)
            
            # Check against maximum correlation
            max_correlation = self.config['risk_management']['max_correlation']
            if correlation > max_correlation:
                logger.warning(f"Correlation {correlation} exceeds maximum {max_correlation}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking correlation: {str(e)}")
            return False
            
    async def check_volatility(self, symbol: str) -> bool:
        """
        Check market volatility.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            bool: True if volatility is acceptable, False otherwise
        """
        try:
            # Calculate volatility
            volatility = await self.calculate_volatility(symbol)
            
            # Check against maximum volatility
            max_volatility = self.config['risk_management']['max_volatility']
            if volatility > max_volatility:
                logger.warning(f"Volatility {volatility} exceeds maximum {max_volatility}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking volatility: {str(e)}")
            return False
            
    async def calculate_position_size(self, account_balance: float, risk_per_trade: float,
                              stop_loss_distance: float) -> float:
        """
        Calculate appropriate position size.
        
        Args:
            account_balance: Current account balance
            risk_per_trade: Maximum risk per trade as a percentage
            stop_loss_distance: Distance to stop loss
            
        Returns:
            float: Calculated position size
        """
        try:
            # Calculate risk amount
            risk_amount = account_balance * risk_per_trade
            
            # Calculate position size based on stop loss
            position_size = risk_amount / stop_loss_distance
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
            
    async def check_drawdown(self) -> bool:
        """
        Check if drawdown limit is reached.
        
        Returns:
            bool: True if drawdown limit is reached, False otherwise
        """
        try:
            # Get current balance
            current_balance = await self.get_account_balance()
            if not current_balance or not self.initial_balance:
                return False
                
            # Calculate drawdown
            drawdown = (self.initial_balance - current_balance) / self.initial_balance
            
            # Check against maximum drawdown
            max_drawdown = self.config['trading']['max_drawdown']
            if drawdown > max_drawdown:
                logger.warning(f"Drawdown {drawdown} exceeds maximum {max_drawdown}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking drawdown: {e}")
            return False
            
    async def get_account_balance(self) -> Optional[float]:
        """
        Get current account balance.
        
        Returns:
            Optional[float]: Account balance or None if failed
        """
        try:
            # Initialize Binance client
            client = ccxt.binance({
                'apiKey': self.config['binance']['api_key'],
                'secret': self.config['binance']['api_secret']
            })
            
            # Fetch balance
            balance = await client.fetch_balance()
            usdt_balance = float(balance['total']['USDT'])
            
            # Set initial balance if not set
            if self.initial_balance is None:
                self.initial_balance = usdt_balance
                
            return usdt_balance
            
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return None
            
    async def get_positions(self) -> Dict:
        """
        Get current positions.
        
        Returns:
            Dict: Current positions
        """
        try:
            # Initialize Binance client
            client = ccxt.binance({
                'apiKey': self.config['binance']['api_key'],
                'secret': self.config['binance']['api_secret']
            })
            
            # Fetch positions
            positions = await client.fetch_positions()
            
            # Filter for open positions
            open_positions = {
                pos['symbol']: {
                    'size': float(pos['contracts']),
                    'entry_price': float(pos['entryPrice']),
                    'side': pos['side'],
                    'unrealized_pnl': float(pos['unrealizedPnl'])
                }
                for pos in positions
                if float(pos['contracts']) > 0
            }
            
            return open_positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
            
    async def calculate_correlation(self, symbol: str, positions: Dict) -> float:
        """
        Calculate correlation with existing positions.
        
        Args:
            symbol: Trading pair symbol
            positions: Existing positions
            
        Returns:
            float: Correlation value between 0 and 1
        """
        try:
            if not positions:
                return 0.0
                
            # Get price history for all symbols
            symbols = list(positions.keys()) + [symbol]
            
            for sym in symbols:
                if sym not in self.price_history:
                    # Fetch historical prices
                    client = ccxt.binance()
                    ohlcv = await client.fetch_ohlcv(sym, timeframe='1h', limit=100)
                    prices = [candle[4] for candle in ohlcv]  # Close prices
                    self.price_history[sym] = prices
                    
            # Calculate returns
            returns = {}
            for sym, prices in self.price_history.items():
                returns[sym] = np.diff(prices) / prices[:-1]
                
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef([returns[sym] for sym in symbols])
            
            # Get average correlation with new symbol
            new_symbol_index = symbols.index(symbol)
            correlations = correlation_matrix[new_symbol_index]
            avg_correlation = np.mean(np.abs(correlations[:-1]))  # Exclude self-correlation
            
            return float(avg_correlation)
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return 1.0  # Return high correlation on error to be safe
            
    async def calculate_volatility(self, symbol: str) -> float:
        """
        Calculate market volatility.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            float: Volatility value (standard deviation of returns)
        """
        try:
            if symbol not in self.price_history:
                # Fetch historical prices
                client = ccxt.binance()
                ohlcv = await client.fetch_ohlcv(symbol, timeframe='1h', limit=100)
                prices = [candle[4] for candle in ohlcv]  # Close prices
                self.price_history[symbol] = prices
                
            # Calculate returns
            returns = np.diff(self.price_history[symbol]) / self.price_history[symbol][:-1]
            
            # Calculate volatility (standard deviation of returns)
            volatility = np.std(returns)
            
            return float(volatility)
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 1.0  # Return high volatility on error to be safe 