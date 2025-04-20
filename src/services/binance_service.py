"""
Service for interacting with Binance exchange.
"""

import logging
from typing import Dict, Optional, List, Any
import ccxt.async_support as ccxt
from datetime import timedelta
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
        self._time_offset = 0  # Time difference between local and server time
        self._last_sync_time = 0  # Last time synchronization timestamp
        self._sync_interval = 300  # Sync every 5 minutes
        self._max_retries = 3  # Maximum number of retries
        self._retry_delay = 1  # Initial retry delay in seconds
        self._max_retry_delay = 32  # Maximum retry delay in seconds
        self._cache = {}  # In-memory cache for indicators
        self._ws_connections = {}  # WebSocket connections
        self._ws_subscriptions = {}  # WebSocket subscriptions
        
    async def initialize(self) -> bool:
        """Initialize the Binance service.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            if self._is_initialized:
                return True
                
            # Get API credentials based on mode
            use_testnet = self.config['api']['binance']['use_testnet']
            api_config = self.config['api']['binance']['testnet' if use_testnet else 'mainnet']
            
            # Create exchange instance
            self.exchange = ccxt.binance({
                'apiKey': api_config['api_key'],
                'secret': api_config['api_secret'],
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000,
                    'defaultTimeInForce': 'GTC',
                    'createMarketBuyOrderRequiresPrice': False,
                    'warnOnFetchOpenOrdersWithoutSymbol': False,
                    'defaultPositionMode': 'hedge'  # Enable hedge mode
                },
                'urls': {
                    'api': {
                        'public': 'https://testnet.binancefuture.com/fapi/v1' if use_testnet else 'https://fapi.binance.com/fapi/v1',
                        'private': 'https://testnet.binancefuture.com/fapi/v1' if use_testnet else 'https://fapi.binance.com/fapi/v1',
                    }
                }
            })
            
            # Load markets
            await self.exchange.load_markets()
            
            # Sync time
            await self._sync_time()
            
            self._is_initialized = True
            logger.info(f"Binance service initialized successfully in {'testnet' if use_testnet else 'mainnet'} mode")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Binance service: {str(e)}")
            return False
            
    async def place_order(self, order_params: Dict) -> Optional[Dict]:
        """Place an order with enhanced error handling and retry logic."""
        try:
            symbol = order_params.get('symbol')
            side = order_params.get('side')
            order_type = order_params.get('type', 'market')
            amount = order_params.get('amount')
            price = order_params.get('price')
            stop_loss = order_params.get('stop_loss')
            take_profit = order_params.get('take_profit')
            reduce_only = order_params.get('reduce_only')
            position_side = order_params.get('position_side')
            
            # Validate required parameters
            if not all([symbol, side, amount]):
                logger.error(f"Missing required parameters for order: {order_params}")
                return None
                
            # Get current market price if not provided
            if not price and order_type == 'limit':
                ticker = await self.get_ticker(symbol)
                if not ticker:
                    logger.error(f"Failed to get ticker for {symbol}")
                    return None
                price = ticker.get('last')
                
            # Place the order
            order = await self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=amount,
                params={
                    "positionSide": position_side
                }
            )
            
            if not order:
                logger.error(f"Failed to place order for {symbol}")
                return None
                
            # Cache the order
            cache_key = f"{symbol}_{side}_{order_type}"
            self.order_cache[cache_key] = order
            
            # Xác định position side và order side cho Hedge Mode
            if side == "buy":
                position_side = "LONG"
                close_side = "SELL"
            else:
                position_side = "SHORT"
                close_side = "BUY"
                
            # Place stop loss order if provided
            if stop_loss:
                try:
                    # Place stop loss order
                    logger.info(f"Placing stop loss order for {symbol} at {stop_loss}")
                    sl_order = await self.exchange.create_order(
                        symbol=symbol,
                        type="STOP_MARKET",
                        side=close_side,
                        amount=amount,
                        params={
                            "stopPrice": stop_loss,
                            "positionSide": position_side
                        }
                    )
                    
                    if sl_order:
                        logger.info(f"Stop loss order placed: {sl_order}")
                    else:
                        logger.error(f"Failed to place stop loss order for {symbol}")
                except Exception as e:
                    logger.error(f"{symbol} Error placing stop loss order: {str(e)}")
                    
            # Place take profit order if provided
            if take_profit:
                try:
                    # Place take profit order
                    logger.info(f"Placing take profit order for {symbol} at {take_profit}")
                    tp_order = await self.exchange.create_order(
                        symbol=symbol,
                        type="TAKE_PROFIT_MARKET",
                        side=close_side,
                        amount=amount,
                        params={
                            "stopPrice": take_profit,
                            "positionSide": position_side
                        }
                    )
                    if tp_order:
                        logger.info(f"Take profit order placed: {tp_order}")
                    else:
                        logger.error(f"Failed to place take profit order for {symbol}")
                except Exception as e:
                    logger.error(f"{symbol} Error placing take profit order: {str(e)}")
                    
            logger.info(f"Order placed: {order}")
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
            # Check cache first
            cache_key = "account_balance"
            cached_data = await self._get_cached_data(cache_key, ttl=60)  # Cache for 1 minute
            if cached_data:
                return cached_data

            # Use REST API with retry mechanism
            balance = await self._make_request(self.exchange.fetch_balance)
            if balance:
                self._set_cached_data(cache_key, balance)
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
            # Check cache first
            cache_key = "positions"
            cached_data = await self._get_cached_data(cache_key, ttl=30)  # Cache for 30 seconds
            if cached_data:
                return cached_data

            # Use REST API with retry mechanism
            positions = await self._make_request(self.exchange.fetch_positions)
            if positions:
                self._set_cached_data(cache_key, positions)
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return None
            
    async def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for a specific symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Optional[Dict]: Position details if successful, None otherwise
        """
        try:
            # Get all positions
            positions = await self.get_positions()
            if not positions:
                return None
                
            # Find position for the specified symbol
            for position in positions:
                if position.get('symbol') == symbol and float(position.get('contracts', 0)) != 0:
                    return position
                    
            return None
        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {str(e)}")
            return None
            
    async def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get ticker information for a symbol."""
        try:
            # Check cache first
            cache_key = f"ticker_{symbol}"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data

            # Try to use WebSocket if available
            if symbol in self._ws_connections and 'ticker' in self._ws_connections[symbol]:
                ticker = await self._ws_connections[symbol]['ticker']
                self._set_cached_data(cache_key, ticker)
                return ticker

            # Fallback to REST API
            ticker = await self._make_request(self.exchange.fetch_ticker, symbol)
            if ticker:
                self._set_cached_data(cache_key, ticker)
            return ticker
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {str(e)}")
            return None

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            # Always fetch fresh data from REST API
            ticker = await self._make_request(self.exchange.fetch_ticker, symbol)
            if ticker and 'last' in ticker:
                price = float(ticker['last'])
                # Validate price
                if price <= 0:
                    logger.error(f"Invalid price {price} for {symbol}")
                    return None
                logger.info(f"Current price for {symbol}: {price}")
                return price
            return None
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
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
            # Close all WebSocket connections
            for symbol in list(self._ws_connections.keys()):
                for channel in list(self._ws_connections[symbol].keys()):
                    await self._cleanup_websocket(symbol, channel)

            # Clear cache
            self._cache.clear()
            
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
            # Check if symbol exists and is active
            try:
                markets = await self.exchange.load_markets()
                if symbol not in markets:
                    logger.warning(f"Symbol {symbol} not found in markets")
                    return None
                if not markets[symbol].get('active', False):
                    logger.warning(f"Symbol {symbol} is not active")
                    return None
            except Exception as e:
                logger.error(f"Error checking symbol status for {symbol}: {str(e)}")
                return None

            # Fetch multiple data points in parallel
            tasks = [
                self.exchange.fetch_ticker(symbol),
                self.exchange.fetch_ohlcv(symbol, timeframe='1m', limit=1),
                self.exchange.fetch_order_book(symbol, limit=5)
            ]
            
            ticker, ohlcv, order_book = await asyncio.gather(*tasks)
            
            # Check if all data is available and has the expected structure
            if not all([ticker, ohlcv, order_book]):
                logger.warning(f"Missing data for {symbol}: ticker={bool(ticker)}, ohlcv={bool(ohlcv)}, order_book={bool(order_book)}")
                return None
                
            # Check if ohlcv has at least one candle
            if not ohlcv or len(ohlcv) == 0:
                logger.warning(f"No OHLCV data available for {symbol}")
                return None
                
            # Check if order book has bids and asks
            if not order_book.get('bids') or not order_book.get('asks'):
                logger.warning(f"No order book data available for {symbol}")
                return None
                
            # Check if there are any bids or asks
            if len(order_book['bids']) == 0 or len(order_book['asks']) == 0:
                logger.warning(f"Empty order book for {symbol}")
                return None
                
            return {
                'price': ticker.get('last'),
                'volume': ticker.get('quoteVolume'),
                'bid': order_book['bids'][0][0],
                'ask': order_book['asks'][0][0],
                'high': ohlcv[0][2],
                'low': ohlcv[0][3]
            }
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {str(e)}")
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
            
    async def get_klines(self, symbol: str, timeframe: str = '5m', limit: int = 100) -> Optional[List]:
        """Get kline/candlestick data for a symbol."""
        try:
            # Check cache first
            cache_key = f"klines_{symbol}_{timeframe}_{limit}"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data

            # Use REST API with retry mechanism
            klines = await self._make_request(
                self.exchange.fetch_ohlcv,
                symbol,
                timeframe=timeframe,
                limit=limit
            )
            if klines:
                self._set_cached_data(cache_key, klines)
            return klines
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {str(e)}")
            return None
            
    async def update_stop_loss(self, symbol: str, stop_price: float, side: str, position_side: str = None, amount: float = None) -> bool:
        """Update stop loss order for a position.
        
        Args:
            symbol: Trading pair symbol
            stop_price: New stop loss price
            side: Order side (BUY/SELL)
            position_side: Position side (LONG/SHORT)
            amount: Order amount (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Use REST API with retry mechanism
            params = {
                'stopPrice': stop_price
            }
            
            # Only add reduceOnly if we're in a position that requires it
            # Check if we have an open position for this symbol
            position = await self.get_position(symbol)
            if position and float(position.get('positionAmt', 0)) != 0:
                params['reduceOnly'] = True
            
            if position_side:
                params['positionSide'] = position_side
                
            result = await self._make_request(
                self.exchange.create_order,
                symbol=symbol,
                type='STOP_MARKET',
                side=side,
                amount=amount,
                params=params
            )
            return bool(result)
        except Exception as e:
            logger.error(f"Error updating stop loss: {str(e)}")
            return False

    async def update_take_profit(self, symbol: str, take_profit_price: float, side: str, amount: float) -> bool:
        """Update take profit order for a position."""
        try:
            # Use REST API with retry mechanism
            params = {
                'stopPrice': take_profit_price
            }
            
            # Only add reduceOnly if we're in a position that requires it
            # Check if we have an open position for this symbol
            position = await self.get_position(symbol)
            if position and float(position.get('positionAmt', 0)) != 0:
                params['reduceOnly'] = True
                
            result = await self._make_request(
                self.exchange.create_order,
                symbol=symbol,
                type='TAKE_PROFIT_MARKET',
                side=side,
                amount=amount,
                params=params
            )
            return bool(result)
        except Exception as e:
            logger.error(f"Error updating take profit: {str(e)}")
            return False

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order by its ID.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to cancel
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._is_initialized:
                logger.error("Binance service not initialized")
                return False
                
            if self._is_closed:
                logger.error("Binance service is closed")
                return False
                
            # Use REST API with retry mechanism
            result = await self._make_request(
                self.exchange.cancel_order,
                id=order_id,
                symbol=symbol
            )
            
            # Clear cache for open orders
            cache_key = f"open_orders_{symbol}"
            self._cache.pop(cache_key, None)
            
            return bool(result)
        except Exception as e:
            logger.error(f"Error canceling order {order_id} for {symbol}: {str(e)}")
            return False

    async def _sync_time(self) -> None:
        """Synchronize local time with Binance server time."""
        try:
            current_time = time.time()
            if current_time - self._last_sync_time < self._sync_interval:
                return

            # Get server time with retry mechanism
            for attempt in range(3):
                try:
                    server_time = await self.exchange.fetch_time()
                    local_time = int(time.time() * 1000)
                    self._time_offset = server_time - local_time
                    self._last_sync_time = current_time
                    
                    # Increase recvWindow to avoid timestamp errors
                    if hasattr(self.exchange, 'options'):
                        self.exchange.options['recvWindow'] = 60000  # 60 seconds
                        logger.info(f"Time synchronized. Offset: {self._time_offset}ms, recvWindow: 60s")
                    else:
                        logger.warning("Exchange options not available for recvWindow adjustment")
                    return
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"Time sync attempt {attempt + 1} failed: {str(e)}")
                        await asyncio.sleep(1)
                    else:
                        raise e
                    
        except Exception as e:
            logger.error(f"Error synchronizing time: {str(e)}")
            # Force sync on next request
            self._last_sync_time = 0

    async def _make_request(self, func, *args, **kwargs):
        """Make a request with retry mechanism and time synchronization."""
        retries = 0
        last_error = None

        while retries < self._max_retries:
            try:
                # Sync time before making request
                await self._sync_time()
                
                # Add timestamp to request if needed
                if 'params' in kwargs:
                    if 'timestamp' not in kwargs['params']:
                        kwargs['params']['timestamp'] = int(time.time() * 1000) + self._time_offset
                    if 'recvWindow' not in kwargs['params']:
                        kwargs['params']['recvWindow'] = 60000  # 60 seconds
                
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                last_error = e
                retries += 1
                
                if '-1021' in str(e):  # Timestamp error
                    logger.warning(f"Timestamp error detected: {str(e)}. Forcing time sync...")
                    self._last_sync_time = 0  # Force sync on next attempt
                    await self._sync_time()
                    continue
                
                if retries < self._max_retries:
                    delay = min(self._retry_delay * (2 ** (retries - 1)), self._max_retry_delay)
                    logger.warning(f"Request failed, retrying in {delay}s (attempt {retries}/{self._max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Request failed after {retries} attempts: {str(last_error)}")
                    raise last_error

    async def _get_cached_data(self, key: str, ttl: int = None) -> Optional[Any]:
        """Get data from cache if available and not expired."""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < (ttl or self._cache_ttl):
                return data
        return None

    def _set_cached_data(self, key: str, data: Any) -> None:
        """Store data in cache with timestamp."""
        self._cache[key] = (data, time.time())

    async def _setup_websocket(self, symbol: str, channel: str) -> None:
        """Setup WebSocket connection for a symbol and channel."""
        try:
            if symbol not in self._ws_connections:
                self._ws_connections[symbol] = {}
            
            if channel not in self._ws_connections[symbol]:
                ws = await self.exchange.watch_ticker(symbol)
                self._ws_connections[symbol][channel] = ws
                logger.info(f"WebSocket connection established for {symbol} {channel}")
        except Exception as e:
            logger.error(f"Error setting up WebSocket for {symbol} {channel}: {str(e)}")

    async def _cleanup_websocket(self, symbol: str, channel: str) -> None:
        """Cleanup WebSocket connection."""
        try:
            if symbol in self._ws_connections and channel in self._ws_connections[symbol]:
                await self._ws_connections[symbol][channel].close()
                del self._ws_connections[symbol][channel]
                logger.info(f"WebSocket connection closed for {symbol} {channel}")
        except Exception as e:
            logger.error(f"Error cleaning up WebSocket for {symbol} {channel}: {str(e)}")

    async def get_markets(self) -> Optional[Dict]:
        """Get all available markets from Binance.
        
        Returns:
            Optional[Dict]: Dictionary of market information if successful, None otherwise
        """
        try:
            if not self._is_initialized:
                logger.error("Binance service not initialized")
                return None
                
            if self._is_closed:
                logger.error("Binance service is closed")
                return None
                
            # Check cache first
            cache_key = "markets"
            cached_data = await self._get_cached_data(cache_key, ttl=3600)  # Cache for 1 hour
            if cached_data:
                return cached_data
                
            # Fetch markets from exchange
            markets = await self._make_request(self.exchange.load_markets)
            if markets:
                self._set_cached_data(cache_key, markets)
            return markets
            
        except Exception as e:
            logger.error(f"Error getting markets: {str(e)}")
            return None

    async def get_open_orders(self, symbol: str = None) -> Optional[List[Dict]]:
        """Get open orders for a symbol or all symbols.
        
        Args:
            symbol: Trading pair symbol (optional)
            
        Returns:
            Optional[List[Dict]]: List of open orders or None if error
        """
        try:
            if not self._is_initialized:
                logger.error("Binance service not initialized")
                return None
                
            if self._is_closed:
                logger.error("Binance service is closed")
                return None
                
            # Check cache first
            cache_key = f"open_orders_{symbol if symbol else 'all'}"
            cached_data = await self._get_cached_data(cache_key, ttl=30)  # Cache for 30 seconds
            if cached_data:
                return cached_data
                
            # Use REST API with retry mechanism
            if symbol:
                orders = await self._make_request(self.exchange.fetch_open_orders, symbol)
            else:
                orders = await self._make_request(self.exchange.fetch_open_orders)
                
            if orders:
                self._set_cached_data(cache_key, orders)
            return orders
            
        except Exception as e:
            logger.error(f"Error getting open orders: {str(e)}")
            return None

    async def close_position(self, symbol: str) -> bool:
        """Close position for a specific symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            bool: True if position closed successfully, False otherwise
        """
        try:
            # Get position details
            position = await self.get_position(symbol)
            if not position:
                logger.warning(f"No position found for {symbol}")
                return False
                
            # Get position amount
            position_amt = float(position.get('positionAmt', 0))
            if position_amt == 0:
                logger.warning(f"No position amount for {symbol}")
                return False
                
            # Determine side based on position amount
            side = 'sell' if position_amt > 0 else 'buy'
            amount = abs(position_amt)
            
            # Get position side
            position_side = position.get('positionSide', 'LONG' if position_amt > 0 else 'SHORT')
            
            # Create market order to close position
            order_params = {
                'symbol': symbol,
                'side': side,
                'type': 'market',
                'amount': amount,
                'params': {
                    'reduceOnly': True,
                    'positionSide': position_side
                }
            }
            
            # Place order
            result = await self.place_order(order_params)
            if result:
                logger.info(f"Position closed for {symbol}: {result}")
                # Clear position cache
                self._cache.pop('positions', None)
                return True
            else:
                logger.error(f"Failed to close position for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {str(e)}")
            return False 