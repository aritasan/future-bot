"""
Service for interacting with Binance exchange.
"""

import logging
from typing import Dict, Optional
import ccxt.async_support as ccxt
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

class BinanceService:
    """Service for interacting with Binance exchange."""
    
    def __init__(self, config: Dict):
        """Initialize the service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._is_initialized = False
        self._is_closed = False
        self.exchange = None
        self.order_cache = {}
        self.balance_cache = {}
        self.position_cache = {}
        self.cache_expiry = timedelta(minutes=5)
        self.last_update = {}
        
    def initialize(self) -> bool:
        """Initialize the Binance service.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            if self._is_initialized:
                logger.warning("Binance service already initialized")
                return True
                
            # Initialize exchange
            self.exchange = ccxt.binance({
                'apiKey': self.config['api']['binance']['api_key'],
                'secret': self.config['api']['binance']['api_secret'],
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True
                }
            })
            
            self._is_initialized = True
            logger.info("Binance service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Binance service: {str(e)}")
            return False
            
    async def place_order(self, order_params: Dict) -> Optional[Dict]:
        """Place an order on Binance.
        
        Args:
            order_params: Order parameters
            
        Returns:
            Optional[Dict]: Order details if successful, None otherwise
        """
        try:
            if not self._is_initialized:
                logger.error("Binance service not initialized")
                return None
                
            if self._is_closed:
                logger.error("Binance service is closed")
                return None
                
            # Check for duplicate orders
            order_key = f"{order_params['symbol']}_{order_params['side']}_{order_params['type']}"
            if order_key in self.order_cache:
                cached_order, timestamp = self.order_cache[order_key]
                if datetime.now() - timestamp < self.cache_expiry:
                    logger.warning("Duplicate order detected")
                    return None
                    
            # Place order
            order = await self.exchange.create_order(
                symbol=order_params['symbol'],
                type=order_params['type'],
                side=order_params['side'],
                amount=order_params['amount'],
                price=order_params.get('price'),
                params=order_params.get('params', {})
            )
            
            # Cache order
            self.order_cache[order_key] = (order, datetime.now())
            
            return order
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None
            
    async def get_account_balance(self) -> Optional[Dict]:
        """Get account balance.
        
        Returns:
            Optional[Dict]: Account balance if successful, None otherwise
        """
        try:
            if not self._is_initialized:
                logger.error("Binance service not initialized")
                return None
                
            if self._is_closed:
                logger.error("Binance service is closed")
                return None
                
            # Check cache
            current_time = datetime.now()
            if 'balance' in self.balance_cache:
                cached_balance, timestamp = self.balance_cache['balance']
                if current_time - timestamp < self.cache_expiry:
                    return cached_balance
                    
            # Fetch balance
            balance = await self.exchange.fetch_balance()
            
            # Cache balance
            self.balance_cache['balance'] = (balance, current_time)
            
            return balance
            
        except Exception as e:
            logger.error(f"Error getting account balance: {str(e)}")
            return None
            
    async def get_positions(self) -> Optional[Dict]:
        """Get open positions.
        
        Returns:
            Optional[Dict]: Open positions if successful, None otherwise
        """
        try:
            if not self._is_initialized:
                logger.error("Binance service not initialized")
                return None
                
            if self._is_closed:
                logger.error("Binance service is closed")
                return None
                
            # Check cache
            current_time = datetime.now()
            if 'positions' in self.position_cache:
                cached_positions, timestamp = self.position_cache['positions']
                if current_time - timestamp < self.cache_expiry:
                    return cached_positions
                    
            # Fetch positions
            positions = await self.exchange.fetch_positions()
            
            # Cache positions
            self.position_cache['positions'] = (positions, current_time)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return None
            
    async def fetch_ohlcv(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            limit: Number of candles to fetch
            
        Returns:
            Optional[pd.DataFrame]: OHLCV data if successful, None otherwise
        """
        try:
            if not self._is_initialized:
                logger.error("Binance service not initialized")
                return None
                
            if self._is_closed:
                logger.error("Binance service is closed")
                return None
                
            # Fetch data
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {str(e)}")
            return None
            
    def clear_cache(self):
        """Clear all cached data."""
        self.order_cache.clear()
        self.balance_cache.clear()
        self.position_cache.clear()
        self.last_update.clear()
        
    async def close(self):
        """Close the Binance service."""
        try:
            if not self._is_initialized:
                logger.warning("Binance service was not initialized")
                return
                
            if self._is_closed:
                logger.warning("Binance service already closed")
                return
                
            # Clear cache
            self.clear_cache()
            
            # Close exchange
            if self.exchange:
                await self.exchange.close()
                
            self._is_closed = True
            logger.info("Binance service closed")
            
        except Exception as e:
            logger.error(f"Error closing Binance service: {str(e)}") 