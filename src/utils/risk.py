"""
Risk management utilities for the trading bot.
"""
import logging
from typing import Dict, Optional
import numpy as np
import ccxt
from datetime import datetime, timedelta

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
        self.last_update = {}
        self.cache_expiry = timedelta(minutes=5)
        self.binance_service = None  # Will be set by set_binance_service
        self._is_initialized = True
        self._is_closed = False
        
    def set_binance_service(self, binance_service):
        """Set the Binance service instance."""
        self.binance_service = binance_service
        
    async def check_risk(self, symbol: str, position_size: float) -> bool:
        """Check if a trade meets risk management criteria."""
        try:
            if not self.binance_service:
                logger.error("Binance service not set")
                return False
                
            # Get current price from Binance service
            ticker = await self.binance_service.get_ticker(symbol)
            if not ticker:
                logger.error(f"Failed to get ticker for {symbol}")
                return False
                
            current_price = ticker.get('last', 0.0)
            if current_price <= 0:
                logger.error(f"Invalid current price: {current_price}")
                return False
                
            # Check position size with actual current price
            if not await self.check_position_size(symbol, position_size, current_price):
                return False
                
            # Check correlation with existing positions
            if not await self.check_correlation(symbol):
                return False
                
            # Check market volatility
            if not await self.check_volatility(symbol):
                return False
                
            # Check drawdown
            if await self.check_drawdown():
                logger.warning("Drawdown limit reached, stopping trading")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk for {symbol}: {str(e)}")
            return False
            
    async def check_position_size(self, symbol: str, position_size: float, current_price: float) -> bool:
        """Check if position size is within limits.
        
        Args:
            symbol: Trading pair symbol
            position_size: Position size to check
            current_price: Current price of the trading pair
            
        Returns:
            bool: True if position size is valid, False otherwise
        """
        try:
            logger.debug(f"Checking position size for {symbol}: size={position_size}, price={current_price}")
            
            # Validate inputs
            if position_size <= 0:
                logger.error(f"Invalid position size: {position_size}")
                return False
                
            if current_price <= 0:
                logger.error(f"Invalid current price: {current_price}")
                return False
                
            # Check minimum notional value (5 USDT)
            min_notional = 5.0  # Binance minimum notional value
            position_notional = position_size * current_price
            logger.debug(f"Position notional value: {position_notional} USDT")
            
            if position_notional < min_notional:
                logger.warning(f"Position notional value {position_notional} below minimum {min_notional} USDT")
                return False
                
            # Get account balance
            balance = await self.get_account_balance()
            if not balance:
                logger.error(f"Failed to get balance for {symbol}")
                return False
                
            # Get USDT balance
            usdt_balance = balance.get('USDT', {}).get('total', 0)
            if not usdt_balance or float(usdt_balance) <= 0:
                logger.error(f"Invalid USDT balance: {usdt_balance}")
                return False
                
            logger.debug(f"USDT balance: {usdt_balance}")
                
            # Get leverage from config
            leverage = self.config['trading'].get('leverage', 10)
            if leverage <= 0:
                logger.error(f"Invalid leverage: {leverage}")
                return False
                
            logger.debug(f"Leverage: {leverage}")
                
            # Check maximum position size based on available balance and leverage
            max_position_size = (float(usdt_balance) * leverage) / current_price
            logger.debug(f"Maximum position size: {max_position_size}")
            
            if position_size > max_position_size:
                logger.warning(f"Position size {position_size} exceeds available balance with leverage")
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
            if not self.binance_service:
                logger.error("Binance service not set")
                return False
                
            # Get historical data
            ohlcv = await self.binance_service.fetch_ohlcv(symbol, timeframe='1h', limit=100)
            if not ohlcv or len(ohlcv) < 2:
                logger.error(f"Not enough data for {symbol}")
                return False
                
            # Extract close prices
            prices = [float(candle[4]) for candle in ohlcv]  # Close prices
            
            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            volatility = float(np.std(returns))
            
            # Check against maximum volatility
            max_volatility = float(self.config['risk_management']['max_volatility'])
            if volatility > max_volatility:
                logger.warning(f"Volatility {volatility} exceeds maximum {max_volatility}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking volatility: {str(e)}")
            return False
            
    async def calculate_position_size(self, symbol: str, risk_per_trade: float, current_price: float) -> Optional[float]:
        """Calculate position size based on risk management.
        
        Args:
            symbol: Trading pair symbol
            risk_per_trade: Risk per trade as a percentage
            current_price: Current price of the trading pair
            
        Returns:
            Optional[float]: Position size or None if calculation fails
        """
        try:
            # Get account balance
            balance = await self.get_account_balance()
            if not balance:
                logger.error(f"Failed to get balance for {symbol}")
                return None
                
            # Get USDT balance
            usdt_balance = balance.get('USDT', {}).get('total', 0)
            if not usdt_balance:
                logger.error(f"No USDT balance available")
                return None
                
            # Calculate position size based on risk
            risk_amount = float(usdt_balance) * (risk_per_trade / 100)
            position_size = risk_amount / current_price
            
            # Check if position size is valid
            if not await self.check_position_size(symbol, position_size, current_price):
                return None
                
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return None
            
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
            
    async def get_account_balance(self) -> Optional[Dict]:
        """Get account balance from Binance service."""
        try:
            if not self.binance_service:
                logger.error("Binance service not set")
                return None
                
            balance = await self.binance_service.get_account_balance()
            if not balance:
                logger.error("Failed to get account balance")
                return None
                
            return balance
            
        except Exception as e:
            logger.error(f"Error getting account balance: {str(e)}")
            return None
            
    async def get_positions(self) -> Dict:
        """
        Get current positions.
        
        Returns:
            Dict: Current positions
        """
        try:
            if not self.binance_service:
                logger.error("Binance service not set")
                return {}
                
            # Get positions from binance service
            positions = await self.binance_service.get_positions()
            if not positions:
                return {}
                
            # Format positions
            open_positions = {}
            for pos in positions:
                if float(pos.get('size', 0)) > 0:
                    open_positions[pos['symbol']] = {
                        'size': float(pos.get('size', 0)),
                        'entry_price': float(pos.get('entry_price', 0)),
                        'side': pos.get('side', ''),
                        'unrealized_pnl': float(pos.get('unrealized_pnl', 0))
                    }
            
            return open_positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
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
            current_time = datetime.now()
            
            for sym in symbols:
                # Check cache
                if sym in self.price_history and sym in self.last_update:
                    if current_time - self.last_update[sym] < self.cache_expiry:
                        continue
                
                # Fetch historical prices
                client = ccxt.binance()
                ohlcv = await client.fetch_ohlcv(sym, timeframe='1h', limit=100)
                prices = [candle[4] for candle in ohlcv]  # Close prices
                self.price_history[sym] = prices
                self.last_update[sym] = current_time
                    
            # Calculate returns
            returns = {}
            for sym, prices in self.price_history.items():
                if len(prices) > 1:
                    returns[sym] = np.diff(prices) / prices[:-1]
                else:
                    returns[sym] = np.array([0.0])
                
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
            if not self.binance_service:
                logger.error("Binance service not set")
                return 1.0
                
            # Get historical data
            ohlcv = await self.binance_service.fetch_ohlcv(symbol, timeframe='1h', limit=100)
            if not ohlcv or len(ohlcv) < 2:
                logger.error(f"Not enough data for {symbol}")
                return 1.0
                
            # Extract close prices
            prices = [candle[4] for candle in ohlcv]  # Close prices
            
            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)
            
            return float(volatility)
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return 1.0  # Return high volatility on error to be safe 