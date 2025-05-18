"""
Service for managing technical indicators and data caching.
"""
import logging
from typing import Dict, Optional
import pandas as pd
from datetime import timedelta
import ccxt.async_support as ccxt
import asyncio
import platform
import numpy as np
import time
import sys

logger = logging.getLogger(__name__)

class IndicatorService:
    """Service for managing technical indicators and data caching."""
    
    def __init__(self, config: Dict):
        """Initialize the service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_cache = {}
        self.indicator_cache = {}
        self.cache_expiry = timedelta(minutes=5)
        self._is_initialized = False
        self._is_closed = False
        self.exchange = None
        self.max_retries = 3
        self.retry_delay = 1
        self._cache = {}
        self._cache_ttl = {
            "1m": 30,  # 30 seconds for 1m data
            "5m": 60,  # 1 minute for 5m data
            "15m": 300,  # 5 minutes for 15m data
            "1h": 900,  # 15 minutes for 1h data
            "4h": 3600,  # 1 hour for 4h data
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
        self._rate_limit = 1000  # Reduced from 2000 to avoid throttle queue issues
        self._rate_limit_window = 60  # 60 seconds
        self._rate_limit_queue = asyncio.Queue(maxsize=500)  # Limit queue size
        self._rate_limit_task = None
        self._ws_connections = {}
        self._ws_subscriptions = {}
        self._ws_data = {}
        self._last_error_time = 0
        self._error_count = 0
        self._max_errors = 5
        self._error_window = 300  # 5 minutes
        self._throttle_backoff = 1  # Initial backoff time in seconds
        self._max_throttle_backoff = 32  # Maximum backoff time in seconds
        
    async def initialize(self) -> bool:
        """Initialize the indicator service.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            if self._is_initialized:
                logger.warning("Indicator service already initialized")
                return True
                
            # Get API credentials based on mode
            use_testnet = self.config['api']['binance']['use_testnet']
            api_config = self.config['api']['binance']['testnet' if use_testnet else 'mainnet']
            
            # Configure exchange options for Windows
            exchange_options = {
                'apiKey': api_config['api_key'],
                'secret': api_config['api_secret'],
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000,  # 60 seconds
                    'defaultTimeInForce': 'GTC',
                    'createMarketBuyOrderRequiresPrice': False,
                    'warnOnFetchOpenOrdersWithoutSymbol': False,
                    'defaultPositionMode': 'hedge'  # Enable hedge mode
                }
            }
            
            # Add Windows-specific options
            if platform.system() == 'Windows':
                exchange_options['options'].update({
                    'asyncio_loop': asyncio.get_event_loop(),
                    'timeout': 30000
                })
                
            # Initialize exchange
            self.exchange = ccxt.binance(exchange_options)
            
            # Sync time with Binance server
            await self._sync_time()
            
            self._is_initialized = True
            logger.info("Indicator service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize indicator service: {str(e)}")
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
                        # logger.info(f"Time synchronized. Offset: {self._time_offset}ms, recvWindow: 60s")
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

    async def _setup_websocket(self, symbol: str, channel: str) -> None:
        """Setup websocket connection for a symbol and channel."""
        try:
            if symbol not in self._ws_connections:
                self._ws_connections[symbol] = {}
                
            if channel not in self._ws_connections[symbol]:
                # Check if websocket method is supported
                try:
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
                    
                    # Start background task to process websocket data
                    asyncio.create_task(self._process_ws_data(symbol, channel))
                    
                except Exception as e:
                    if "not supported" in str(e).lower():
                        # Fallback to REST API for unsupported methods
                        await self._fallback_to_rest_api(symbol, channel)
                    else:
                        raise e
                        
        except Exception as e:
            logger.error(f"Error setting up websocket for {symbol} {channel}: {str(e)}")
            await self._fallback_to_rest_api(symbol, channel)
            
    async def _fallback_to_rest_api(self, symbol: str, channel: str) -> None:
        """Fallback to REST API when websocket fails or is not supported."""
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
                    
        except Exception as e:
            logger.error(f"Error cleaning up websocket for {symbol} {channel}: {str(e)}")
            
    async def _rate_limit_handler(self) -> None:
        """Handle rate limiting by queueing requests."""
        while True:
            try:
                current_time = time.time()
                
                # Check error rate
                if self._error_count >= self._max_errors:
                    if current_time - self._last_error_time < self._error_window:
                        # Too many errors, wait longer
                        await asyncio.sleep(self._error_window)
                        self._error_count = 0
                        self._last_error_time = current_time
                    else:
                        self._error_count = 0
                
                # Check rate limit
                if self._request_count >= self._rate_limit:
                    # Wait until rate limit window resets
                    wait_time = self._rate_limit_window - (current_time - self._last_request_time)
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    self._request_count = 0
                    self._last_request_time = current_time
                    
                # Process queued requests
                while not self._rate_limit_queue.empty():
                    request = await self._rate_limit_queue.get()
                    try:
                        await request
                    except Exception as e:
                        logger.error(f"Error processing queued request: {str(e)}")
                        self._error_count += 1
                        self._last_error_time = current_time
                    finally:
                        self._rate_limit_queue.task_done()
                        
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in rate limit handler: {str(e)}")
                await asyncio.sleep(1)
                
    async def _make_request(self, func, *args, **kwargs):
        """Make a request with rate limiting and retry mechanism."""
        retries = 0
        last_error = None
        throttle_backoff = self._throttle_backoff

        while retries < self._max_retries:
            try:
                # Check rate limit
                current_time = time.time()
                if current_time - self._last_request_time >= self._rate_limit_window:
                    self._request_count = 0
                    self._last_request_time = current_time
                    
                if self._request_count >= self._rate_limit:
                    # Wait until rate limit window resets
                    wait_time = self._rate_limit_window - (current_time - self._last_request_time)
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    self._request_count = 0
                    self._last_request_time = current_time
                    
                # Make request
                self._request_count += 1
                result = await func(*args, **kwargs)
                # Reset throttle backoff on successful request
                throttle_backoff = self._throttle_backoff
                return result
                
            except Exception as e:
                last_error = e
                retries += 1
                
                if '-1021' in str(e):  # Timestamp error
                    logger.warning(f"Timestamp error detected: {str(e)}. Forcing time sync...")
                    self._last_sync_time = 0
                    await self._sync_time()
                    continue
                    
                if 'throttle queue is over maxCapacity' in str(e):
                    # Exponential backoff for throttle errors
                    wait_time = min(throttle_backoff, self._max_throttle_backoff)
                    logger.warning(f"Throttle queue full, waiting {wait_time}s before retry (attempt {retries}/{self._max_retries})")
                    await asyncio.sleep(wait_time)
                    throttle_backoff *= 2  # Double the backoff time
                    continue
                    
                if retries < self._max_retries:
                    delay = min(self._retry_delay * (2 ** (retries - 1)), self._max_retry_delay)
                    logger.warning(f"Request failed, retrying in {delay}s (attempt {retries}/{self._max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Request failed after {retries} attempts: {str(last_error)}")
                    self._error_count += 1
                    self._last_error_time = time.time()
                    raise last_error

    async def get_historical_data(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> Optional[pd.DataFrame]:
        """Get historical data with improved caching and rate limiting."""
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_{limit}"
            if cache_key in self._cache:
                cache_data = self._cache[cache_key]
                if time.time() - cache_data['timestamp'] < self._cache_ttl.get(timeframe, 60):
                    return cache_data['data']
                    
            # Get data from exchange
            data = await self._make_request(
                self.exchange.fetch_ohlcv,
                symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            if data:
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Update cache
                self._cache[cache_key] = {
                    'data': df,
                    'timestamp': time.time()
                }
                
                return df
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return None
            
    async def calculate_indicators(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> Optional[pd.DataFrame]:
        """Calculate technical indicators for a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            limit: Number of candles to fetch
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with indicators or None if error
        """
        try:
            if not self._is_initialized:
                logger.error("Indicator service not initialized")
                return None
                
            if self._is_closed:
                logger.error("Indicator service is closed")
                return None
                
            # Check cache
            cache_key = f"{symbol}_{timeframe}"
            current_time = time.time()
            
            if cache_key in self._cache:
                cached_data, timestamp = self._cache[cache_key]
                # Check if cache is still valid
                if current_time - timestamp < self._cache_ttl[timeframe]:
                    return cached_data
            
            # Fetch new data if cache is invalid or missing
            try:
                ohlcv = await self._make_request(self.exchange.fetch_ohlcv, symbol, timeframe, limit)
                if not ohlcv:
                    logger.error(f"Failed to fetch OHLCV data for {symbol}")
                    return None
                    
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Calculate indicators
                df = await self._calculate_technical_indicators(df)
                
                # Cache the result
                self._cache[cache_key] = (df, current_time)
                self._last_update[cache_key] = current_time
                
                return df
                
            except Exception as e:
                if 'throttle queue is over maxCapacity' in str(e):
                    logger.warning(f"Throttle queue full for {symbol}, using cached data if available")
                    if cache_key in self._cache:
                        return self._cache[cache_key][0]
                raise e
                
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return None
            
    def clear_cache(self):
        """Clear the indicator cache."""
        self._cache.clear()
        self._last_update.clear()
        
    async def close(self):
        """Close the service and clear cache."""
        try:
            self.clear_cache()
            self._is_closed = True
            logger.info("Indicator service closed")
        except Exception as e:
            logger.error(f"Error closing indicator service: {str(e)}")
            
    def _calculate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud indicators.
        
        Args:
            df: Price data DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with Ichimoku indicators
        """
        try:
            # Tenkan-sen (Conversion Line)
            high_9 = df["high"].rolling(window=9, min_periods=1).max()
            low_9 = df["low"].rolling(window=9, min_periods=1).min()
            df["tenkan_sen"] = (high_9 + low_9) / 2
            
            # Kijun-sen (Base Line)
            high_26 = df["high"].rolling(window=26, min_periods=1).max()
            low_26 = df["low"].rolling(window=26, min_periods=1).min()
            df["kijun_sen"] = (high_26 + low_26) / 2
            
            # Senkou Span A (Leading Span A)
            df["senkou_span_a"] = ((df["tenkan_sen"] + df["kijun_sen"]) / 2).shift(26)
            
            # Senkou Span B (Leading Span B)
            high_52 = df["high"].rolling(window=52, min_periods=1).max()
            low_52 = df["low"].rolling(window=52, min_periods=1).min()
            df["senkou_span_b"] = ((high_52 + low_52) / 2).shift(26)
            
            # Chikou Span (Lagging Span)
            df["chikou_span"] = df["close"].shift(-26)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating Ichimoku Cloud: {str(e)}")
            return df
            
    def _calculate_fibonacci(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Fibonacci retracement levels.
        
        Args:
            df: Price data DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with Fibonacci levels
        """
        try:
            # Find swing high and low
            window = min(20, len(df))
            df["swing_high"] = df["high"].rolling(window=window, min_periods=1).max()
            df["swing_low"] = df["low"].rolling(window=window, min_periods=1).min()
            
            # Calculate Fibonacci levels
            high = df["swing_high"].iloc[-1]
            low = df["swing_low"].iloc[-1]
            diff = high - low
            
            if diff > 0:  # Only calculate if there is a price difference
                df["fib_0"] = high
                df["fib_0.236"] = high - diff * 0.236
                df["fib_0.382"] = high - diff * 0.382
                df["fib_0.5"] = high - diff * 0.5
                df["fib_0.618"] = high - diff * 0.618
                df["fib_0.786"] = high - diff * 0.786
                df["fib_1"] = low
            else:
                # Set all levels to current price if no difference
                current_price = df["close"].iloc[-1]
                for level in ["0", "0.236", "0.382", "0.5", "0.618", "0.786", "1"]:
                    df[f"fib_{level}"] = current_price
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {str(e)}")
            return df
            
    def _calculate_volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Profile.
        
        Args:
            df: Price data DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with Volume Profile
        """
        try:
            if len(df) < 2:
                return df
                
            # Calculate price levels
            price_range = df["high"].max() - df["low"].min()
            if price_range > 0:
                num_levels = min(20, len(df))
                price_step = price_range / num_levels
                
                # Calculate volume at each price level
                volume_profile = {}
                for i in range(num_levels):
                    price_level = df["low"].min() + i * price_step
                    volume = df[(df["low"] <= price_level) & (df["high"] >= price_level)]["volume"].sum()
                    volume_profile[price_level] = volume
                    
                # Add volume profile to DataFrame
                df["volume_profile"] = df["close"].apply(
                    lambda x: volume_profile.get(min(volume_profile.keys(), key=lambda k: abs(k - x)), 0)
                )
            else:
                df["volume_profile"] = df["volume"]
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating Volume Profile: {str(e)}")
            return df
            
    async def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators in parallel."""
        try:
            # Calculate each indicator sequentially
            df = self._calculate_rsi(df)
            df = self._calculate_macd(df)
            df = self._calculate_bollinger_bands(df)
            df = self._calculate_atr(df)
            df = self._calculate_ema(df)
            df = self._calculate_roc(df)
            df = self._calculate_adx(df)
            df = self._calculate_ichimoku(df)
            
            # Add advanced momentum indicators
            df = self._calculate_momentum(df)
            df = self._calculate_stochastic(df)
            df = self._calculate_mfi(df)
            df = self._calculate_obv(df)
            df = self._calculate_cci(df)
            df = self._calculate_williams_r(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return df
            
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI indicator."""
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            # Handle division by zero
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            df['RSI'] = rsi.fillna(50)  # Fill NaN with neutral value
            return df
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return df
            
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicator."""
        try:
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            df['MACD'] = macd
            df['MACD_SIGNAL'] = signal
            return df
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return df
            
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        try:
            sma = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            
            # Handle NaN values
            std = std.fillna(0)
            
            df['BB_upper'] = sma + (std * 2)
            df['BB_lower'] = sma - (std * 2)
            df['BB_middle'] = sma
            return df
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return df
            
    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range."""
        try:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            df['ATR'] = true_range.rolling(window=14).mean().fillna(0)
            return df
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return df
            
    def _calculate_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Exponential Moving Average."""
        try:
            df['EMA_FAST'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA_SLOW'] = df['close'].ewm(span=26, adjust=False).mean()
            return df
        except Exception as e:
            logger.error(f"Error calculating EMA: {str(e)}")
            return df
            
    def _calculate_roc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Rate of Change (ROC) indicator."""
        try:
            # Calculate ROC with 14-period lookback
            df['ROC'] = ((df['close'] - df['close'].shift(14)) / df['close'].shift(14)) * 100
            return df
        except Exception as e:
            logger.error(f"Error calculating ROC: {str(e)}")
            return df
            
    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average Directional Index (ADX)."""
        try:
            # Calculate True Range
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            
            # Convert to pandas Series for rolling operations
            high_low = pd.Series(high_low)
            high_close = pd.Series(high_close)
            low_close = pd.Series(low_close)
            
            # Calculate True Range
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Calculate +DM and -DM
            up_move = df['high'] - df['high'].shift()
            down_move = df['low'].shift() - df['low']
            
            # Convert to pandas Series
            up_move = pd.Series(up_move)
            down_move = pd.Series(down_move)
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Convert back to pandas Series
            plus_dm = pd.Series(plus_dm)
            minus_dm = pd.Series(minus_dm)
            
            # Calculate smoothed values
            tr14 = true_range.rolling(window=14).mean()
            plus_di14 = 100 * (plus_dm.rolling(window=14).mean() / tr14)
            minus_di14 = 100 * (minus_dm.rolling(window=14).mean() / tr14)
            
            # Calculate ADX
            dx = 100 * abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
            df['ADX'] = dx.rolling(window=14).mean()
            
            # Fill NaN values
            df['ADX'] = df['ADX'].fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            # Return original DataFrame with ADX column set to 0
            df['ADX'] = 0
            return df 

    def get_cache_stats(self) -> Dict:
        """Get cache statistics.
        
        Returns:
            Dict: Cache statistics including hit rate and memory usage
        """
        try:
            total_requests = len(self._last_update)
            cache_hits = sum(1 for key in self._cache if time.time() - self._last_update.get(key, 0) < self._cache_ttl.get(key.split('_')[1], 300))
            hit_rate = cache_hits / total_requests if total_requests > 0 else 0
            
            return {
                'total_requests': total_requests,
                'cache_hits': cache_hits,
                'hit_rate': hit_rate,
                'cache_size': len(self._cache),
                'memory_usage': sys.getsizeof(self._cache)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {} 

    async def calculate_atr(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> Optional[float]:
        """Calculate Average True Range for a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for calculation
            limit: Number of candles to use
            
        Returns:
            Optional[float]: ATR value or None if error
        """
        try:
            # Get historical data
            df = await self.get_historical_data(symbol, timeframe, limit)
            if df is None or df.empty:
                logger.error(f"Failed to get historical data for {symbol}")
                return None
                
            # Calculate ATR
            df = self._calculate_atr(df)
            
            # Return the latest ATR value
            if 'ATR' in df.columns:
                return float(df['ATR'].iloc[-1])
            else:
                logger.error(f"ATR not calculated for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating ATR for {symbol}: {str(e)}")
            return None 

    def _calculate_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators."""
        try:
            # Rate of Change (ROC)
            df['ROC'] = ((df['close'] - df['close'].shift(14)) / df['close'].shift(14)) * 100.0
            
            # Momentum
            df['MOM'] = df['close'] - df['close'].shift(10)
            
            # Relative Momentum Index (RMI)
            df['RMI'] = self._calculate_rmi(df)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {str(e)}")
            return df

    def _calculate_rmi(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Relative Momentum Index."""
        try:
            # Calculate price changes
            price_changes = df['close'].diff()
            
            # Calculate upward and downward movements
            up_moves = price_changes.where(price_changes > 0, 0.0)
            down_moves = -price_changes.where(price_changes < 0, 0.0)
            
            # Calculate smoothed averages
            up_avg = up_moves.rolling(window=14).mean()
            down_avg = down_moves.rolling(window=14).mean()
            
            # Calculate RMI
            rmi = 100.0 - (100.0 / (1.0 + (up_avg / down_avg)))
            return rmi
        except Exception as e:
            logger.error(f"Error calculating RMI: {str(e)}")
            return pd.Series(0.0, index=df.index)

    def _calculate_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        try:
            # Calculate %K
            low_min = df['low'].rolling(window=14).min()
            high_max = df['high'].rolling(window=14).max()
            df['STOCH_K'] = 100.0 * ((df['close'] - low_min) / (high_max - low_min))
            
            # Calculate %D (3-period moving average of %K)
            df['STOCH_D'] = df['STOCH_K'].rolling(window=3).mean()
            
            return df
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {str(e)}")
            return df

    def _calculate_mfi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Money Flow Index."""
        try:
            # Calculate typical price
            typical_price = (df['high'] + df['low'] + df['close']) / 3.0
            
            # Calculate money flow
            money_flow = typical_price * df['volume']
            
            # Calculate positive and negative money flow
            price_diff = typical_price.diff()
            positive_flow = money_flow.where(price_diff > 0, 0.0)
            negative_flow = money_flow.where(price_diff < 0, 0.0)
            
            # Calculate money ratio
            positive_flow_sum = positive_flow.rolling(window=14).sum()
            negative_flow_sum = negative_flow.rolling(window=14).sum()
            money_ratio = positive_flow_sum / negative_flow_sum
            
            # Calculate MFI
            df['MFI'] = 100.0 - (100.0 / (1.0 + money_ratio))
            
            return df
        except Exception as e:
            logger.error(f"Error calculating MFI: {str(e)}")
            return df

    def _calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume."""
        try:
            # Calculate price changes
            price_changes = df['close'].diff()
            
            # Calculate OBV
            obv = pd.Series(0.0, index=df.index)
            obv[price_changes > 0] = df['volume']
            obv[price_changes < 0] = -df['volume']
            df['OBV'] = obv.cumsum()
            
            return df
        except Exception as e:
            logger.error(f"Error calculating OBV: {str(e)}")
            return df

    def _calculate_cci(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Commodity Channel Index."""
        try:
            # Calculate typical price
            typical_price = (df['high'] + df['low'] + df['close']) / 3.0
            
            # Calculate moving average
            ma = typical_price.rolling(window=20).mean()
            
            # Calculate mean deviation
            mean_deviation = abs(typical_price - ma).rolling(window=20).mean()
            
            # Calculate CCI
            df['CCI'] = (typical_price - ma) / (0.015 * mean_deviation)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating CCI: {str(e)}")
            return df

    def _calculate_williams_r(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Williams %R."""
        try:
            # Calculate highest high and lowest low
            highest_high = df['high'].rolling(window=14).max()
            lowest_low = df['low'].rolling(window=14).min()
            
            # Calculate Williams %R
            df['WILLIAMS_R'] = -100.0 * (highest_high - df['close']) / (highest_high - lowest_low)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {str(e)}")
            return df 