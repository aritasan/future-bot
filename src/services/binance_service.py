"""
Service for interacting with Binance exchange.
"""

import logging
from typing import Dict, Optional, List, Any
import ccxt.async_support as ccxt
from datetime import timedelta
import time
import asyncio
from src.utils.helpers import is_same_symbol, is_same_side, is_long_side

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
        self._cache_ttl = {
            'market': 60,  # 1 minute
            'ticker': 30,  # 30 seconds
            'order': 30,   # 30 seconds
            'position': 30, # 30 seconds
            'balance': 60,  # 1 minute
            'klines': 60,   # 1 minute
            'markets': 3600 # 1 hour
        }
        self._time_offset = 0
        self._last_sync_time = 0
        self._sync_interval = 300  # 5 minutes
        self._max_retries = 3
        self._retry_delay = 1
        self._max_retry_delay = 32
        self._cache = {}  # Unified cache storage
        self._ws_connections = {}
        self._ws_subscriptions = {}
        
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
        """Place an order on Binance.
        
        Args:
            order_params: Dictionary containing order parameters
                Required keys:
                - symbol: Trading pair symbol
                - side: Order side (BUY/SELL)
                - type: Order type (MARKET/LIMIT/STOP_MARKET/STOP_LIMIT)
                - amount: Order amount
                Optional keys:
                - price: Price for limit orders
                - stop_loss: Stop loss price
                - take_profit: Take profit price
                
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
                
            # Validate required parameters
            required_params = ['symbol', 'side', 'type', 'amount']
            for param in required_params:
                if param not in order_params:
                    logger.error(f"Missing required parameter: {param}")
                    return None
                    
            # Convert side and type to uppercase
            order_params['side'] = order_params['side'].upper()
            order_params['type'] = order_params['type'].upper()
            
            # Get symbol and position side
            symbol = order_params['symbol']
                    
            # Place main order first
            main_order_params = {
                'symbol': symbol,
                'side': order_params['side'],
                'type': order_params['type'].upper(),
                'params': {
                    'positionSide': 'LONG' if is_long_side(order_params['side']) else 'SHORT',
                },
                'amount': order_params['amount']
            }
            
              
            # Place main order
            main_order = await self._make_request(
                self.exchange.create_order,
                **main_order_params
            )
            
            if not main_order:
                logger.error(f"Failed to place main order for {symbol} {main_order_params}")
                return None
               
            logger.info(f"Main order placed successfully for {symbol} {main_order_params['side']}: {main_order['id']}")
            
            # Check for existing SL/TP orders
            existing_orders = await self.get_open_orders(symbol)
            open_position_side = "SELL" if is_long_side(order_params['side']) else "BUY"
            if existing_orders:
                existing_sl = await self.get_existing_order(symbol, 'STOP_MARKET', open_position_side)
                existing_tp = await self.get_existing_order(symbol, 'TAKE_PROFIT_MARKET', open_position_side)
                logger.info(f"Existing SL/TP orders for {symbol}: {existing_sl} {existing_tp}")

                # Cancel existing SL/TP orders
                if existing_sl:
                    await self.cancel_order(symbol, existing_sl['id'])
                if existing_tp:
                    await self.cancel_order(symbol, existing_tp['id'])
                    
            # Place SL/TP orders if specified
            if 'stop_loss' in order_params:
                sl_order_params = {
                    'symbol': symbol,
                    'side': 'SELL' if is_long_side(order_params['side']) else 'BUY',
                    'type': 'STOP_MARKET',
                    'amount': order_params['amount'],
                    'params': {
                        'stopPrice': order_params['stop_loss'],
                        'positionSide': 'LONG' if is_long_side(order_params['side']) else 'SHORT',
                        'workingType': 'MARK_PRICE',
                        'priceProtect': True,
                        'timeInForce': 'GTC',
                        'closePosition': True
                    }
                }
                
                sl_order = await self._make_request(
                    self.exchange.create_order,
                    **sl_order_params
                )
                
                if not sl_order:
                    logger.error(f"Failed to place stop loss order for {symbol}")
                    
            if 'take_profit' in order_params:
                tp_order_params = {
                    'symbol': symbol,
                    'side': 'SELL' if is_long_side(order_params['side']) else 'BUY',
                    'type': 'TAKE_PROFIT_MARKET',
                    'amount': order_params['amount'],
                    'params': {
                        'stopPrice': order_params['take_profit'],
                        'positionSide': 'LONG' if is_long_side(order_params['side']) else 'SHORT',
                        'workingType': 'MARK_PRICE',
                        'priceProtect': True,
                        'timeInForce': 'GTC',
                        'closePosition': True
                    }
                }
                
                tp_order = await self._make_request(
                    self.exchange.create_order,
                    **tp_order_params
                )
                
                if not tp_order:
                    logger.error(f"Failed to place take profit order for {symbol}")
                    
            return main_order
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error placing order: {str(e)}")
            return None
            
    async def _validate_order_params(self, order_params: Dict) -> bool:
        """Validate order parameters.
        
        Args:
            order_params: Dictionary containing order parameters
            
        Returns:
            bool: True if parameters are valid, False otherwise
        """
        required_params = ['symbol', 'type', 'side', 'amount']
        for param in required_params:
            if param not in order_params:
                logger.error(f"Missing required parameter: {param}")
                return False
            
        # Validate order type
        if not await self._validate_order_type(order_params['type']):
            return False
            
        return True
        
            
    async def _validate_order_type(self, order_type: str) -> bool:
        """Validate order type.
        
        Args:
            order_type: Order type
            
        Returns:
            bool: True if order type is valid, False otherwise
        """
        valid_types = ['MARKET', 'LIMIT', 'STOP_MARKET', 'TAKE_PROFIT_MARKET']
        if order_type.upper() not in valid_types:
            logger.error(f"Invalid order type: {order_type}")
            return False
        return True
        
            
    async def _place_main_order(self, order_params: Dict) -> Optional[Dict]:
        """Place the main order.
        
        Args:
            order_params: Dictionary containing order parameters
            
        Returns:
            Optional[Dict]: Order details if successful, None otherwise
        """
        try:
            # Place order
            order = await self._make_request(
                self.exchange.create_order,
                **order_params
            )
            
            if order:
                logger.info(f"Order placed successfully for {order_params['symbol']} {order_params['side']}: {order['id']}")
                return order
            return None
            
        except Exception as e:
            logger.error(f"Error placing main order: {str(e)}")
            return None
            
    async def _update_stop_order(self, symbol: str, position: Dict, new_price: float, 
                               order_type: str) -> bool:
        """Update stop loss or take profit order.
        
        Args:
            symbol: Trading pair symbol
            position: Position details
            new_price: New stop price
            order_type: Order type (STOP_MARKET/TAKE_PROFIT_MARKET)
            
        Returns:
            bool: True if order updated successfully, False otherwise
        """
        try:
            if not self._is_initialized:
                logger.error("Binance service not initialized")
                return False
                
            if self._is_closed:
                logger.error("Binance service is closed")
                return False
                
            # Get current position
            current_position = await self.get_position(symbol, position.get('info', {}).get('positionSide', None))
            if not current_position:
                logger.error(f"No position found for {symbol}")
                return False
                
            # Prepare new order parameters
            order_params = {
                'symbol': symbol,
                'type': order_type,
                'side': 'SELL' if is_long_side(position['info']['positionSide']) else 'BUY',
                'amount': abs(float(current_position['contracts'])),
                'params': {
                    'stopPrice': new_price,
                    'positionSide': position['info']['positionSide'],
                    'workingType': 'MARK_PRICE',
                    'timeInForce': 'GTC',
                    'closePosition': True
                }
            }
            
            # Get existing orders
            existing_orders = await self.get_open_orders(symbol)
            if not existing_orders:
                logger.info(f"No orders found for {symbol}, creating new {order_type} order")
                # Create new order since no orders exist
                return await self._place_main_order(order_params)
                
            # Find existing order
            logger.info(f"Existing orders {symbol}: {existing_orders}")
            open_position_side = "SELL" if is_long_side(position['info']['positionSide']) else "BUY"
            existing_order = await self.get_existing_order(symbol, order_type, open_position_side)
            logger.info(f"Existing order {order_type} {symbol} {open_position_side}: {existing_order}")
            if not existing_order:
                logger.info(f"No {order_type} order found for {symbol}, creating new order")
                # Create new order since no order of this type exists
                return await self._place_main_order(order_params)
                
            # Cancel existing order
            if not await self.cancel_order(symbol, existing_order['id']):
                logger.error(f"Failed to cancel existing {order_type} order")
                return False
            
            # Place new order
            return await self._place_main_order(order_params)
            
        except Exception as e:
            logger.error(f"Error updating {order_type} order: {str(e)}")
            return False
            
    async def _update_stop_loss(self, symbol: str, position: Dict, new_stop_loss: float) -> bool:
        """Update stop loss order.
        
        Args:
            symbol: Trading pair symbol
            position: Position details
            new_stop_loss: New stop loss price
            
        Returns:
            bool: True if update successful, False otherwise
        """
        return await self._update_stop_order(symbol, position, new_stop_loss, 'STOP_MARKET')
        
    async def _update_take_profit(self, symbol: str, position: Dict, new_take_profit: float) -> bool:
        """Update take profit order.
        
        Args:
            symbol: Trading pair symbol
            position: Position details
            new_take_profit: New take profit price
            
        Returns:
            bool: True if update successful, False otherwise
        """
        return await self._update_stop_order(symbol, position, new_take_profit, 'TAKE_PROFIT_MARKET')

    async def get_account_balance(self) -> Optional[Dict]:
        """Get account balance.
        
        Returns:
            Optional[Dict]: Account balance if successful, None otherwise
        """
        try:
            # Check cache first
            cache_key = "account_balance"
            cached_data = await self._get_cached_data(cache_key)
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
            cached_data = await self._get_cached_data(cache_key)
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
            
    async def get_position(self, symbol: str, position_side: str = None) -> Optional[Dict]:
        """Get position for a specific symbol and position side.
        
        Args:
            symbol: Trading pair symbol
            position_side: Position side (LONG/SHORT). If None, returns first position found.
            
        Returns:
            Optional[Dict]: Position details if successful, None otherwise
        """
        try:
            # Get all positions
            positions = await self.get_positions()
            if not positions:
                return None
                
            # Find position for the specified symbol and side
            # Convert BIO/USDT:USDT => BIOUSDT
            symbol = symbol.split(':')[0].replace('/', '')
            for position in positions:
                if is_same_symbol(position.get('info').get('symbol'), symbol):
                    # Check position side if specified
                    if position_side:
                        if is_same_side(position.get('info').get('positionSide'), position_side) and float(position.get('contracts', 0)) != 0:
                            return position
                    # If position_side not specified, return first position found
                    elif float(position.get('contracts', 0)) != 0:
                        return position
                    
            return None
        except Exception as e:
            logger.error(f"Error getting position for {symbol} with side {position_side}: {str(e)}")
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
                # logger.info(f"Current price for {symbol}: {price}")
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
            
    def clear_cache(self, cache_type: str = None):
        """Clear cache data.
        
        Args:
            cache_type: Type of cache to clear. If None, clears all cache.
        """
        if cache_type:
            # Clear specific cache type
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{cache_type}_")]
            for key in keys_to_remove:
                self._cache.pop(key, None)
        else:
            # Clear all cache
            self._cache.clear()
            
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
        except Exception as e:
            logger.error(f"Error closing Binance service: {str(e)}")
            
    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get market data with caching."""
        try:
            cache_key = f"market_{symbol}"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # Fetch fresh data
            data = await self._fetch_market_data(symbol)
            if data:
                self._set_cached_data(cache_key, data)
            return data
            
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return None
            
    async def get_position_statistics(self) -> Dict:
        """Get position statistics with caching."""
        try:
            cache_key = "position_stats"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # Calculate fresh statistics
            stats = await self._calculate_position_statistics()
            if stats:
                self._set_cached_data(cache_key, stats)
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
            
            logger.info(f"Order {order_id} cancelled for {symbol}")
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

    async def _get_cached_data(self, key: str) -> Optional[Any]:
        """Get data from cache if available and not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data if available and not expired, None otherwise
        """
        if key in self._cache:
            data, timestamp, ttl = self._cache[key]
            if time.time() - timestamp < ttl:
                return data
        return None

    def _set_cached_data(self, key: str, data: Any) -> None:
        """Store data in cache with timestamp.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        # Set TTL based on cache type
        ttl = self._cache_ttl.get('market', 60) if 'market' in key else 60
        self._cache[key] = (data, time.time(), ttl)

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
            cached_data = await self._get_cached_data(cache_key)
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
            cached_data = await self._get_cached_data(cache_key)
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

    async def get_existing_order(self, symbol: str, order_type: str, side: str) -> Optional[List[Dict]]:
        """Get existing order for a symbol.
        
        Args:
            symbol: Trading pair symbol
        """
        try:
            # Get open orders
            orders = await self.get_open_orders(symbol)
            if not orders:
                return None

            # Get existing order
            for order in orders:
                if order['type'].upper() == order_type.upper() and order['side'].upper() == side.upper():
                    return order
            return None
            
        except Exception as e:
            logger.error(f"Error getting existing order: {str(e)}")
            return None
    
    
    async def close_position(self, symbol: str, position_side: str = None) -> bool:
        """Close position for a specific symbol and position side.
        
        Args:
            symbol: Trading pair symbol
            position_side: Position side (LONG/SHORT). If None, returns first position found.
        Returns:
            bool: True if position closed successfully, False otherwise
        """
        try:
            # Get position details
            position = await self.get_position(symbol, position_side)
            if not position:
                logger.warning(f"No position found for {symbol} {position_side}")
                return False
                
            # Get position amount
            position_amt = float(position.get('info').get('positionAmt', 0))
            if position_amt == 0:
                logger.warning(f"No position amount for {symbol} {position_side}")
                return False
                
            # Determine side based on position amount
            side = 'SELL' if position_amt > 0 else 'BUY'
            amount = abs(position_amt)
            
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
                logger.info(f"Position closed for {symbol} {position_side}: {result}")
                # Clear position cache
                self._cache.pop('positions', None)
                return True
            else:
                logger.error(f"Failed to close position for {symbol} {position_side}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position for {symbol} {position_side}: {str(e)}")
            return False
