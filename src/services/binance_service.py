"""
Service for interacting with Binance exchange.
"""

import logging
from typing import Dict, Optional, List, Any, Callable
import ccxt.async_support as ccxt
from datetime import datetime
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
        self.exchange = None
        self._is_initialized = False
        self._is_closed = False
        self._cache = {}
        self._cache_ttl = {
            "orders": 5,  # 5 minutes
            "balances": 5,
            "positions": 5,
            "tickers": 1,  # 1 minute
            "orderbook": 1,
            "trades": 1,
        }
        self._last_update = {}
        self._time_offset = 0
        self._last_sync_time = 0
        self._sync_interval = 300
        self._max_retries = 3
        self._retry_delay = 1
        self._max_retry_delay = 32
        self._request_count = 0
        self._last_request_time = 0
        self._rate_limit = 2400
        self._rate_limit_window = 60
        self._rate_limit_queue = asyncio.Queue()
        self._rate_limit_task = None
        self._ws_connections = {}
        self._ws_subscriptions = {}
        self._ws_data = {}
        self._ws_callbacks = {}
        
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
            
            # Place SL/TP orders if specified
            if 'stop_loss' in order_params:
                await self._update_stop_loss(symbol, main_order, order_params['stop_loss'])

            if 'take_profit' in order_params:
                await self._update_take_profit(symbol, main_order, order_params['take_profit'])

            return main_order
            
        except Exception as e:
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
        
    
    async def _cancel_existing_orders(self, symbol: str, order_type: str, position_side: str) -> bool:
        """Cancel all existing orders of specified type for a symbol.
        
        Args:
            symbol: Trading pair symbol
            order_type: Order type to cancel (e.g. STOP_MARKET, TAKE_PROFIT_MARKET)
            position_side: Position side (e.g. LONG, SHORT)

        Returns:
            bool: True if orders cancelled successfully, False otherwise
        """
        try:
            if not self._is_initialized:
                logger.error("Binance service not initialized")
                return False
                
            if self._is_closed:
                logger.error("Binance service is closed")
                return False
                
            # Get existing orders
            existing_orders = await self.get_open_orders(symbol)
            if not existing_orders:
                logger.info(f"No existing orders found for {symbol}")
                return True
                
            # Cancel matching orders
            cancelled = True
            attempt = 0
            maxAttempts = 3
            while attempt < maxAttempts:
                for order in existing_orders:
                    if order['type'].upper() == order_type.upper() and is_same_side(order['info']['positionSide'], position_side):
                        success = await self._make_request(
                            self.exchange.cancel_order,
                            id=order['id'],
                            symbol=symbol
                        )
                        if not success:
                            cancelled = False
                            logger.error(f"Failed to cancel {order_type} order {order['id']} for {symbol}")
                        else:
                            logger.info(f"Cancelled {order_type} order {order['id']} for {symbol}")
                        
                if cancelled:
                    return True
                attempt += 1
                logger.info(f"Attempt {attempt} of {maxAttempts} failed to cancel {order_type} order for {symbol} {position_side}")
                await asyncio.sleep(5)
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling existing orders: {str(e)}")
            return False
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
            # logger.info(f"_update_stop_order: Position: {position}")
            position_side = position.get('info', {}).get('positionSide', None)
            # logger.info(f"_update_stop_order: Position side: {position_side}")
            # current_position = await self.get_position(symbol, position_side)
            # logger.info(f"_update_stop_order: Current position: {current_position}")
            # if not current_position:
            #     logger.error(f"No position found for {symbol}")
            #     return False
            
            # Cancel existing orders
            if not await self._cancel_existing_orders(symbol, order_type, position_side):
                logger.error(f"Failed to cancel existing {order_type} order")
                return False
            
            # Prepare new order parameters
            order_params = {
                'symbol': symbol,
                'type': order_type,
                'side': 'SELL' if is_long_side(position['info']['positionSide']) else 'BUY',
                'amount': abs(float(position.get('info', {}).get('positionAmt', position.get('amount', 0)))),
                'params': {
                    'stopPrice': new_price,
                    'positionSide': position['info']['positionSide'],
                    'workingType': 'MARK_PRICE',
                    'timeInForce': 'GTC',
                    'closePosition': True
                }
            }
            
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
                
            # Normalize symbol format
            # Handle both formats: ATOMUSDT and ATOM/USDT
            normalized_symbol = symbol.split(':')[0].replace('/', '')
            
            # Find position for the specified symbol and side
            for position in positions:
                if not position or not isinstance(position, dict):
                    continue
                    
                # Get position info
                info = position.get('info', {})
                if not info:
                    continue
                    
                # Normalize position symbol
                pos_symbol = info.get('symbol', '').replace('/', '')
                
                # Check if symbols match
                if is_same_symbol(pos_symbol, normalized_symbol):
                    # Check position side if specified
                    if position_side:
                        if is_same_side(info.get('positionSide', ''), position_side) and float(info.get('positionAmt', 0)) != 0:
                            return position
                    # If position_side not specified, return first position found
                    elif float(info.get('positionAmt', 0)) != 0:
                        return position
                    
            return None
        except Exception as e:
            logger.error(f"Error getting position for {symbol} with side {position_side}: {str(e)}")
            return None
       
    async def cleanup_orders(self) -> Optional[Dict[str, int]]:
        """Clean up orders.
        
        Returns:
            Optional[Dict[str, int]]: Dictionary with symbol as key and number of deleted orders as value, or None if error
        """
        try:
            # Get all open orders
            open_orders = await self._make_request(self.exchange.fetch_open_orders)
            if not open_orders:
                return {}

            # Group orders by symbol, position side and order type
            orders_by_symbol = {}
            deleted_orders = {}
            
            for order in open_orders:
                if not order or not isinstance(order, dict):
                    continue

                order_type = order.get('type')
                if order_type.upper() not in ['STOP_MARKET', 'TAKE_PROFIT_MARKET']:
                    continue

                symbol = order.get('symbol')
                position_side = order.get('info', {}).get('positionSide')
                
                if not symbol or not position_side:
                    continue

                # Key format: BTCUSDT_LONG_STOP_MARKET or BTCUSDT_SHORT_TAKE_PROFIT_MARKET
                key = f"{symbol}_{position_side}_{order_type}"
                if key not in orders_by_symbol:
                    orders_by_symbol[key] = []
                orders_by_symbol[key].append(order)

            # For each symbol/side/type combination, keep only the latest order
            for key, orders in orders_by_symbol.items():
                if len(orders) <= 1:
                    continue

                # Sort by timestamp descending to keep the most recent
                sorted_orders = sorted(orders, key=lambda x: x.get('timestamp', 0), reverse=True)
                
                # Cancel all except the latest order
                for order in sorted_orders[1:]:
                    try:
                        await self._make_request(
                            self.exchange.cancel_order,
                            order['id'],
                            order['symbol']
                        )
                        symbol = order['symbol']
                        print(f"Order: {order}")
                        deleted_orders[symbol] = deleted_orders.get(symbol, 0) + 1
                        logger.info(f"Cancelled duplicate {key} order {order['id']} for {symbol}")
                    except Exception as e:
                        logger.error(f"Error cancelling order {order['id']}: {str(e)}")

            return deleted_orders

        except Exception as e:
            logger.error(f"Error cleaning up orders: {str(e)}")
            return None
    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker data with REST API fallback."""
        try:
            # Check cache first
            cache_key = f"{symbol}_ticker"
            if cache_key in self._cache:
                cache_data = self._cache[cache_key]
                if time.time() - cache_data['timestamp'] < self._cache_ttl['tickers']:
                    return cache_data['data']
                    
            # Use REST API directly since websocket is not supported
            ticker = await self._make_request(self.exchange.fetch_ticker, symbol)
            if ticker:
                self._cache[cache_key] = {
                    'data': ticker,
                    'timestamp': time.time()
                }
            return ticker or {}
            
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {str(e)}")
            return {}
            
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
        """Get comprehensive market data for a trading pair.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Optional[Dict]: Market data or None if error
        """
        try:
            if not self._is_initialized:
                logger.error("Binance service not initialized")
                return None
            
            # Get multiple market data points concurrently
            tasks = [
                self.get_ticker(symbol),
                self.get_funding_rate(symbol),
                self.get_open_interest(symbol),
                self.get_order_book(symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for errors
            if any(isinstance(r, Exception) for r in results):
                logger.error(f"Error getting market data for {symbol}")
                return None
            
            ticker, funding_rate, open_interest, order_book = results
            
            if not all([ticker, funding_rate is not None, open_interest is not None, order_book]):
                logger.error(f"Missing market data for {symbol}")
                return None
            
            # Format market data
            market_data = {
                'ticker': ticker,
                'funding_rate': funding_rate,
                'open_interest': open_interest,
                'order_book': order_book,
                'timestamp': datetime.now().isoformat()
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
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
        """Cancel an order.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to cancel
            
        Returns:
            bool: True if order canceled successfully, False otherwise
        """
        try:
            if not self._is_initialized:
                logger.error("Binance service not initialized")
                return False
                
            if self._is_closed:
                logger.error("Binance service is closed")
                return False
                
            # First check if the order still exists
            try:
                order = await self._make_request(
                    self.exchange.fetch_order,
                    order_id,
                    symbol
                )
                if not order:
                    logger.info(f"Order {order_id} for {symbol} not found, assuming already canceled")
                    return True
                    
                if order['status'].lower() in ['closed', 'canceled']:
                    logger.info(f"Order {order_id} for {symbol} is already {order['status']}")
                    return True
                    
            except Exception as e:
                # If we get an "Unknown order" error, the order is already gone
                if '-2011' in str(e):  # Unknown order error code
                    logger.info(f"Order {order_id} for {symbol} not found, assuming already canceled")
                    return True
                # For other errors, log and continue with cancellation attempt
                logger.warning(f"Error checking order status: {str(e)}")
                
            # Try to cancel the order
            try:
                result = await self._make_request(
                    self.exchange.cancel_order,
                    order_id,
                    symbol
                )
                if result:
                    logger.info(f"Order {order_id} for {symbol} canceled successfully")
                    return True
                return False
                
            except Exception as e:
                # If we get an "Unknown order" error during cancellation, the order is already gone
                if '-2011' in str(e):  # Unknown order error code
                    logger.info(f"Order {order_id} for {symbol} not found during cancellation, assuming already canceled")
                    return True
                # For other errors, log and return False
                logger.error(f"Error canceling order {order_id} for {symbol}: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Error in cancel_order: {str(e)}")
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

    async def _setup_websocket(self, symbol: str, channel: str, callback: Optional[Callable] = None) -> None:
        """Setup websocket connection for a symbol and channel.
        
        Args:
            symbol: Trading pair symbol
            channel: Websocket channel (e.g. 'ticker', 'orderbook', 'trades')
            callback: Optional callback function to handle websocket data
        """
        try:
            if symbol not in self._ws_connections:
                self._ws_connections[symbol] = {}
                
            if channel not in self._ws_connections[symbol]:
                # Use supported websocket methods
                if channel == 'ticker':
                    # Fallback to REST API for ticker
                    ticker = await self._make_request(self.exchange.fetch_ticker, symbol)
                    if ticker:
                        self._ws_data[symbol] = ticker
                        self._cache[f"{symbol}_ticker"] = {
                            'data': ticker,
                            'timestamp': time.time()
                        }
                    return
                    
                elif channel == 'orderbook':
                    ws = await self.exchange.watch_order_book(symbol)
                elif channel == 'trades':
                    ws = await self.exchange.watch_trades(symbol)
                else:
                    raise ValueError(f"Unsupported websocket channel: {channel}")
                    
                self._ws_connections[symbol][channel] = ws
                self._ws_data[symbol] = {}
                self._ws_callbacks[symbol] = callback
                
                # Start background task to process websocket data
                asyncio.create_task(self._process_ws_data(symbol, channel))
                
        except Exception as e:
            logger.error(f"Error setting up websocket for {symbol} {channel}: {str(e)}")
            # Fallback to REST API
            await self._fallback_to_rest_api(symbol, channel)
            
    async def _fallback_to_rest_api(self, symbol: str, channel: str) -> None:
        """Fallback to REST API when websocket fails."""
        try:
            if channel == 'ticker':
                data = await self._make_request(self.exchange.fetch_ticker, symbol)
            elif channel == 'orderbook':
                data = await self._make_request(self.exchange.fetch_order_book, symbol)
            elif channel == 'trades':
                data = await self._make_request(self.exchange.fetch_trades, symbol)
            else:
                return
                
            if data:
                self._ws_data[symbol] = data
                self._cache[f"{symbol}_{channel}"] = {
                    'data': data,
                    'timestamp': time.time()
                }
                
        except Exception as e:
            logger.error(f"Error in REST API fallback for {symbol} {channel}: {str(e)}")
            
    async def _process_ws_data(self, symbol: str, channel: str) -> None:
        """Process websocket data for a symbol and channel."""
        try:
            ws = self._ws_connections[symbol][channel]
            while True:
                data = await ws.recv()
                self._ws_data[symbol] = data
                
                # Update cache
                self._cache[f"{symbol}_{channel}"] = {
                    'data': data,
                    'timestamp': time.time()
                }
                
                # Call callback if provided
                if symbol in self._ws_callbacks and self._ws_callbacks[symbol]:
                    await self._ws_callbacks[symbol](data)
                    
        except Exception as e:
            logger.error(f"Error processing websocket data for {symbol} {channel}: {str(e)}")
            await self._cleanup_websocket(symbol, channel)
            
    async def _cleanup_websocket(self, symbol: str, channel: str) -> None:
        """Cleanup websocket connection."""
        try:
            if symbol in self._ws_connections and channel in self._ws_connections[symbol]:
                await self._ws_connections[symbol][channel].close()
                del self._ws_connections[symbol][channel]
                if not self._ws_connections[symbol]:
                    del self._ws_connections[symbol]
                if symbol in self._ws_data:
                    del self._ws_data[symbol]
                if symbol in self._ws_callbacks:
                    del self._ws_callbacks[symbol]
                    
        except Exception as e:
            logger.error(f"Error cleaning up websocket for {symbol} {channel}: {str(e)}")
            
    async def get_orderbook(self, symbol: str) -> Dict:
        """Get orderbook data with websocket support."""
        try:
            # Check cache first
            cache_key = f"{symbol}_orderbook"
            if cache_key in self._cache:
                cache_data = self._cache[cache_key]
                if time.time() - cache_data['timestamp'] < self._cache_ttl['orderbook']:
                    return cache_data['data']
                    
            # Setup websocket if not already connected
            if symbol not in self._ws_connections or 'orderbook' not in self._ws_connections[symbol]:
                await self._setup_websocket(symbol, 'orderbook')
                
            # Return websocket data if available
            if symbol in self._ws_data:
                return self._ws_data[symbol]
                
            # Fallback to REST API
            return await self._make_request(self.exchange.fetch_order_book, symbol)
            
        except Exception as e:
            logger.error(f"Error getting orderbook for {symbol}: {str(e)}")
            return {}
            
    async def get_trades(self, symbol: str) -> List[Dict]:
        """Get recent trades with websocket support."""
        try:
            # Check cache first
            cache_key = f"{symbol}_trades"
            if cache_key in self._cache:
                cache_data = self._cache[cache_key]
                if time.time() - cache_data['timestamp'] < self._cache_ttl['trades']:
                    return cache_data['data']
                    
            # Setup websocket if not already connected
            if symbol not in self._ws_connections or 'trades' not in self._ws_connections[symbol]:
                await self._setup_websocket(symbol, 'trades')
                
            # Return websocket data if available
            if symbol in self._ws_data:
                return self._ws_data[symbol]
                
            # Fallback to REST API
            return await self._make_request(self.exchange.fetch_trades, symbol)
            
        except Exception as e:
            logger.error(f"Error getting trades for {symbol}: {str(e)}")
            return []

    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get the current funding rate for a futures trading pair.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Optional[float]: Current funding rate or None if error
        """
        try:
            if not self._is_initialized:
                logger.error("Binance service not initialized")
                return None
            
            # Get funding rate from Binance
            funding_rate = await self._make_request(
                self.exchange.fetch_funding_rate,
                symbol
            )
            
            if not funding_rate:
                logger.error(f"Failed to get funding rate for {symbol}")
                return None
            
            return float(funding_rate['fundingRate'])
            
        except Exception as e:
            logger.error(f"Error getting funding rate for {symbol}: {str(e)}")
            return None

    async def get_open_interest(self, symbol: str) -> Optional[float]:
        """Get current open interest for a futures trading pair.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Optional[float]: Open interest value or None if error
        """
        try:
            if not self._is_initialized:
                logger.error("Binance service not initialized")
                return None
                
            if self._is_closed:
                logger.error("Binance service is closed")
                return None
                
            # Get open interest data
            open_interest_data = await self._make_request(
                self.exchange.fetch_open_interest,
                symbol
            )
            
            if not open_interest_data:
                logger.error(f"Failed to get open interest data for {symbol}")
                return None
                
            # For Binance, the response structure is:
            # {
            #     'symbol': 'BTC/USDT',
            #     'openInterest': 12345.67,
            #     'timestamp': 1234567890
            # }
            if isinstance(open_interest_data, dict):
                # Try different possible field names
                open_interest = open_interest_data.get('openInterest') or \
                              open_interest_data.get('open_interest') or \
                              open_interest_data.get('oi') or \
                              open_interest_data.get('openInterestAmount')
                
                if open_interest is not None:
                    return float(open_interest)
                    
            # If we get here, we couldn't find the open interest value
            logger.error(f"Could not find open interest value in response for {symbol}. Response: {open_interest_data}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting open interest for {symbol}: {str(e)}")
            return None

    async def get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """Get the current order book for a trading pair.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of orders to fetch for each side
            
        Returns:
            Optional[Dict]: Order book data or None if error
        """
        try:
            if not self._is_initialized:
                logger.error("Binance service not initialized")
                return None
            
            # Get order book from Binance
            order_book = await self._make_request(
                self.exchange.fetch_order_book,
                symbol,
                limit=limit
            )
            
            if not order_book:
                logger.error(f"Failed to get order book for {symbol}")
                return None
            
            # Format order book data
            formatted_book = {
                'bids': order_book['bids'][:limit],
                'asks': order_book['asks'][:limit],
                'timestamp': order_book['timestamp'],
                'datetime': order_book['datetime']
            }
            
            return formatted_book
            
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {str(e)}")
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
            if cache_key in self._cache:
                cache_data = self._cache[cache_key]
                if time.time() - cache_data['timestamp'] < self._cache_ttl['orders']:
                    return cache_data['data']
                    
            # Use REST API with retry mechanism
            if symbol:
                orders = await self._make_request(self.exchange.fetch_open_orders, symbol)
            else:
                orders = await self._make_request(self.exchange.fetch_open_orders)
                
            if orders:
                self._cache[cache_key] = {
                    'data': orders,
                    'timestamp': time.time()
                }
            return orders
            
        except Exception as e:
            logger.error(f"Error getting open orders: {str(e)}")
            return None

    async def get_stop_price(self, symbol: str, position_side: str, order_type: str) -> Optional[float]:
        """Get stop price of open position for specific symbol and side.
        
        Args:
            symbol: Trading pair symbol
            position_side: Position side (LONG/SHORT)
            
        Returns:
            Optional[float]: Stop price if found, None otherwise
        """
        try:
            if not self._is_initialized:
                logger.error("Binance service not initialized")
                return None
                
            if self._is_closed:
                logger.error("Binance service is closed") 
                return None

            existing_orders = await self.get_open_orders(symbol)
            open_position_side = 'SELL' if is_long_side(position_side) else 'BUY'
            if existing_orders:
                existing_sl = await self.get_existing_order(symbol, order_type, open_position_side)
                if existing_sl:
                    return float(existing_sl.get('stopPrice', 0))
                    
            return None

        except Exception as e:
            logger.error(f"Error getting stop price for {symbol} {position_side}: {str(e)}")
            return None
    async def get_existing_order(self, symbol: str, order_type: str, side: str) -> Optional[Dict]:
        """Get existing order for a symbol with specific type and side.
        
        Args:
            symbol: Trading pair symbol
            order_type: Order type (e.g. 'stop', 'limit', 'market')
            side: Order side (e.g. 'buy', 'sell')
            
        Returns:
            Optional[Dict]: Order details if found, None otherwise
        """
        try:
            if not self._is_initialized:
                logger.error("Binance service not initialized")
                return None
                
            if self._is_closed:
                logger.error("Binance service is closed")
                return None
                
            # Get open orders
            orders = await self.get_open_orders(symbol)
            if not orders:
                return None
                
            # Find matching order
            for order in orders:
                if (order['type'].lower() == order_type.lower() and 
                    order['side'].lower() == side.lower()):
                    return order
                    
            return None
            
        except Exception as e:
            logger.error(f"Error getting existing order for {symbol}: {str(e)}")
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
                
            # Get position amount and side
            position_amt = float(position.get('info', {}).get('positionAmt', 0))
            if position_amt == 0:
                logger.warning(f"No position amount for {symbol} {position_side}")
                return False
                
            # Determine side based on position amount
            side = 'SELL' if position_amt > 0 else 'BUY'
            position_side_value = position.get('info', {}).get('positionSide', position_side)
            
            # Use direct API call to close position
            try:
                result = await self._make_request(
                    self.exchange.create_order,
                    symbol=symbol,
                    type='MARKET',
                    side=side,
                    amount=abs(position_amt),
                    params={
                        'positionSide': position_side_value
                    }
                )
                
                if result:
                    logger.info(f"Position closed for {symbol} {position_side_value}: {result}")
                    # Clear position cache
                    self.clear_cache('position')
                    return True
                else:
                    logger.error(f"Failed to close position for {symbol} {position_side_value}")
                    return False
            except Exception as e:
                logger.error(f"Error in API call to close position: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position for {symbol} {position_side}: {str(e)}")
            return False

    async def get_all_future_symbols(self) -> Optional[List[str]]:
        """Get all available future trading pairs from Binance.
        
        Returns:
            Optional[List[str]]: List of all future trading pairs or None if error
        """
        try:
            if not self._is_initialized:
                logger.error("Binance service not initialized")
                return None
                
            if self._is_closed:
                logger.error("Binance service is closed")
                return None

            # Load markets if not already loaded
            if not self.exchange.markets:
                await self.exchange.load_markets()

            # Filter for future markets only
            future_symbols = []
            for symbol, market in self.exchange.markets.items():
                # Check if it's a future market and active
                if (market.get('future', False) or 
                    market.get('swap', False) or 
                    market.get('linear', False) or 
                    market.get('inverse', False)) and market.get('active', False):
                    # Remove any exchange-specific prefix if present
                    clean_symbol = symbol.split(':')[0]
                    if clean_symbol not in future_symbols:
                        future_symbols.append(clean_symbol)

            # Sort symbols alphabetically
            future_symbols.sort()

            logger.info(f"Found {len(future_symbols)} future trading pairs")
            return future_symbols

        except Exception as e:
            logger.error(f"Error getting future symbols: {str(e)}")
            return None
