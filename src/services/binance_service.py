"""
Service for interacting with Binance exchange.
"""

import logging
from typing import Dict, Optional, List
import ccxt.async_support as ccxt
from datetime import datetime, timedelta
import pandas as pd
import time
import asyncio

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
        self._market_cache = {}
        self._position_cache = {}
        self._balance_cache = {}
        self._cache_ttl = 60  # 1 minute
        self._last_update = {}
        
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
            order_params: Order parameters including:
                - symbol: Trading pair
                - side: buy/sell
                - type: market/limit/stop
                - amount: Order amount
                - price: Order price (for limit/stop orders)
                - stop_loss: Stop loss price (optional)
                - take_profit: Take profit price (optional)
                - reduceOnly: Whether the order is reduce-only (for SL/TP)
            
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
            order_key = f"{order_params['symbol']}_{order_params['amount']}_{order_params['type']}"
            if order_key in self.order_cache:
                cached_order, timestamp = self.order_cache[order_key]
                if datetime.now() - timestamp < self.cache_expiry:
                    logger.warning("Duplicate order detected")
                    return None
                    
            # Get current market price
            ticker = await self.exchange.fetch_ticker(order_params['symbol'])
            current_price = float(ticker['last'])
            
            # Set position side based on order side
            position_side = "LONG" if order_params['side'] == 'buy' else "SHORT"
            
            # Base params for main order
            params = {
                'positionSide': position_side,
                **order_params.get('params', {})
            }
            
            # Place main order
            order = await self.exchange.create_order(
                symbol=order_params['symbol'],
                type=order_params['type'],
                side=order_params['side'],
                amount=order_params['amount'],
                price=order_params.get('price'),
                params=params
            )
            
            # Cache main order
            self.order_cache[order_key] = (order, datetime.now())
            
            # Place stop loss order if specified
            if 'stop_loss' in order_params:
                sl_side = 'sell' if order_params['side'] == 'buy' else 'buy'
                sl_price = float(order_params['stop_loss'])
                
                try:
                    sl_order = await self.exchange.create_order(
                        symbol=order_params['symbol'],
                        type='STOP_MARKET',
                        side=sl_side,
                        amount=order_params['amount'],
                        params={
                            'positionSide': position_side,
                            'stopPrice': sl_price
                        }
                    )
                    logger.info(f"Stop loss order placed: {sl_order}")
                except Exception as e:
                    logger.error(f"Failed to place stop loss order: {str(e)}")
                    # Continue with main order even if stop loss fails
                    return order
            
            # Place take profit order if specified
            if 'take_profit' in order_params:
                tp_side = 'sell' if order_params['side'] == 'buy' else 'buy'
                tp_price = float(order_params['take_profit'])
                
                try:
                    tp_order = await self.exchange.create_order(
                        symbol=order_params['symbol'],
                        type='TAKE_PROFIT_MARKET',
                        side=tp_side,
                        amount=order_params['amount'],
                        params={
                            'positionSide': position_side,
                            'stopPrice': tp_price
                        }
                    )
                    logger.info(f"Take profit order placed: {tp_order}")
                except Exception as e:
                    logger.error(f"Failed to place take profit order: {str(e)}")
                    # Continue with main order even if take profit fails
                    return order
            
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
            
    async def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get ticker information for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Optional[Dict]: Ticker information if successful, None otherwise
        """
        try:
            if not self._is_initialized:
                logger.error("Binance service not initialized")
                return None
                
            if self._is_closed:
                logger.error("Binance service is closed")
                return None
                
            # Fetch ticker
            ticker = await self.exchange.fetch_ticker(symbol)
            
            if not ticker:
                logger.error(f"Failed to fetch ticker for {symbol}")
                return None
                
            # Format ticker with safe float conversion
            formatted_ticker = {
                'last': float(ticker.get('last', 0)) if ticker.get('last') is not None else 0,
                'bid': float(ticker.get('bid', 0)) if ticker.get('bid') is not None else 0,
                'ask': float(ticker.get('ask', 0)) if ticker.get('ask') is not None else 0,
                'high': float(ticker.get('high', 0)) if ticker.get('high') is not None else 0,
                'low': float(ticker.get('low', 0)) if ticker.get('low') is not None else 0,
                'volume': float(ticker.get('volume', 0)) if ticker.get('volume') is not None else 0,
                'timestamp': ticker.get('timestamp', 0)
            }
            
            return formatted_ticker
            
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {str(e)}")
            return None
            
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[List]:
        """Fetch OHLCV data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for OHLCV data
            limit: Number of candles to fetch
            
        Returns:
            Optional[List]: OHLCV data if successful, None otherwise
        """
        try:
            if not self._is_initialized:
                logger.error("Binance service not initialized")
                return None
                
            if self._is_closed:
                logger.error("Binance service is closed")
                return None
                
            # Fetch OHLCV data
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            
            if not ohlcv:
                logger.error(f"Failed to fetch OHLCV data for {symbol}")
                return None
                
            return ohlcv
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {str(e)}")
            return None
            
    def clear_cache(self):
        """Clear all cached data."""
        self.order_cache.clear()
        self.balance_cache.clear()
        self.position_cache.clear()
        self.last_update.clear()
        self._market_cache.clear()
        self._position_cache.clear()
        self._balance_cache.clear()
        
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
            
    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get market data with caching."""
        try:
            current_time = time.time()
            cache_key = f"market_{symbol}"
            
            if cache_key in self._market_cache:
                cached_data, timestamp = self._market_cache[cache_key]
                if current_time - timestamp < self._cache_ttl:
                    return cached_data
            
            # Fetch fresh data if cache expired
            data = await self._fetch_market_data(symbol)
            if data:
                self._market_cache[cache_key] = (data, current_time)
                self._last_update[cache_key] = current_time
            return data
            
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return None
            
    async def get_position_statistics(self) -> Dict:
        """Get position statistics with caching."""
        try:
            current_time = time.time()
            cache_key = "position_stats"
            
            if cache_key in self._position_cache:
                cached_data, timestamp = self._position_cache[cache_key]
                if current_time - timestamp < self._cache_ttl:
                    return cached_data
            
            # Calculate fresh statistics
            stats = await self._calculate_position_statistics()
            if stats:
                self._position_cache[cache_key] = (stats, current_time)
                self._last_update[cache_key] = current_time
            return stats
            
        except Exception as e:
            logger.error(f"Error getting position statistics: {str(e)}")
            return None
            
    async def _fetch_market_data(self, symbol: str) -> Optional[Dict]:
        """Fetch market data efficiently."""
        try:
            # Fetch multiple data points in parallel
            tasks = [
                self.exchange.fetch_ticker(symbol),
                self.exchange.fetch_ohlcv(symbol, timeframe='1m', limit=1),
                self.exchange.fetch_order_book(symbol, limit=5)
            ]
            
            ticker, ohlcv, order_book = await asyncio.gather(*tasks)
            
            if not all([ticker, ohlcv, order_book]):
                return None
                
            return {
                'price': ticker['last'],
                'volume': ticker['quoteVolume'],
                'bid': order_book['bids'][0][0],
                'ask': order_book['asks'][0][0],
                'high': ohlcv[0][2],
                'low': ohlcv[0][3]
            }
            
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return None
            
    async def _calculate_position_statistics(self) -> Dict:
        """Calculate position statistics efficiently."""
        try:
            # Get positions and account info in parallel
            tasks = [
                self.exchange.fetch_positions(),
                self.exchange.fetch_balance()
            ]
            
            positions, balance = await asyncio.gather(*tasks)
            
            if not positions:
                return {
                    'total_pnl': 0.0,
                    'active_positions': 0,
                    'position_details': []
                }
                
            # Process positions efficiently
            total_pnl = 0.0
            active_positions = 0
            position_details = []
            
            for pos in positions:
                if not pos or not isinstance(pos, dict):
                    continue
                    
                size = float(pos.get('contracts', 0))
                if size <= 0:
                    continue
                    
                pnl = float(pos.get('unrealizedPnl', 0))
                total_pnl += pnl
                active_positions += 1
                
                position_details.append({
                    'symbol': pos.get('symbol', 'Unknown'),
                    'size': size,
                    'pnl': pnl,
                    'entry_price': pos.get('entryPrice'),
                    'mark_price': pos.get('markPrice'),
                    'leverage': pos.get('leverage'),
                    'side': pos.get('side')
                })
                
            return {
                'total_pnl': total_pnl,
                'active_positions': active_positions,
                'position_details': position_details
            }
            
        except Exception as e:
            logger.error(f"Error calculating position statistics: {str(e)}")
            return None 