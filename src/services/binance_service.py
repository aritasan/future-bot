"""
Binance service for trading operations.
"""

import logging
from typing import Dict, Optional, List
import ccxt
from cachetools import TTLCache

logger = logging.getLogger(__name__)

class BinanceService:
    def __init__(self, config: Dict):
        self.config = config
        self.exchange = None
        self.price_cache = TTLCache(maxsize=100, ttl=60)  # Cache giá trong 60 giây
        
    def initialize(self) -> None:
        """Initialize Binance client."""
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.config['api']['binance']['api_key'],
                'secret': self.config['api']['binance']['api_secret'],
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'  # Enable futures trading
                }
            })
            # Test connection
            self.exchange.fetch_balance()
            logger.info("Binance service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Binance service: {e}")
            raise
            
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List:
        """Fetch OHLCV data."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            return []
            
    def get_account_balance(self) -> Optional[float]:
        """Get account balance in USDT."""
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = float(balance['total']['USDT'])
            return usdt_balance
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
            return None
            
    def place_order(self, signal: Dict) -> bool:
        """Place an order based on signal."""
        try:
            order = self.exchange.create_order(
                symbol=signal['symbol'],
                type=signal['type'],
                side=signal['side'],
                amount=signal['amount'],
                price=signal.get('price'),
                params={'reduceOnly': signal.get('reduce_only', False)}
            )
            logger.info(f"Order placed: {order}")
            return True
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return False
            
    def get_position_info(self, symbol: str) -> Optional[Dict]:
        """Get position information."""
        try:
            positions = self.exchange.fetch_positions([symbol])
            if positions:
                return positions[0]
            return None
        except Exception as e:
            logger.error(f"Error getting position info: {e}")
            return None
            
    def close_position(self, symbol: str) -> bool:
        """Close position for a symbol."""
        try:
            position = self.get_position_info(symbol)
            if position and float(position['contracts']) > 0:
                side = 'sell' if position['side'] == 'long' else 'buy'
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=side,
                    amount=float(position['contracts']),
                    params={'reduceOnly': True}
                )
                logger.info(f"Position closed: {order}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
            
    def get_realtime_price(self, symbol: str) -> Optional[float]:
        """Get realtime price for a symbol."""
        try:
            if symbol in self.price_cache:
                return self.price_cache[symbol]
                
            ticker = self.exchange.fetch_ticker(symbol)
            price = float(ticker['last'])
            self.price_cache[symbol] = price
            return price
        except Exception as e:
            logger.error(f"Error getting realtime price: {e}")
            return None
            
    def close(self) -> None:
        """Close the exchange connection."""
        try:
            # Clear the price cache
            self.price_cache.clear()
            logger.info("Binance service closed")
        except Exception as e:
            logger.error(f"Error closing Binance service: {e}") 