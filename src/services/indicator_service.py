"""
Service for managing technical indicators and data caching.
"""
import logging
from typing import Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
import ccxt.async_support as ccxt
import asyncio
import platform
import numpy as np

import time

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
        self._cache_ttl = 300  # 5 minutes
        self._last_update = {}
        
    async def initialize(self) -> bool:
        """Initialize the indicator service.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            if self._is_initialized:
                logger.warning("Indicator service already initialized")
                return True
                
            # Configure exchange options for Windows
            exchange_options = {
                'apiKey': self.config['api']['binance']['api_key'],
                'secret': self.config['api']['binance']['api_secret'],
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True
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
            
            self._is_initialized = True
            logger.info("Indicator service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize indicator service: {str(e)}")
            return False
            
    async def get_historical_data(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> Optional[pd.DataFrame]:
        """Get historical price data with caching and retry mechanism.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            limit: Number of candles to fetch
            
        Returns:
            Optional[pd.DataFrame]: Historical price data
        """
        try:
            if not self._is_initialized:
                logger.error("Indicator service not initialized")
                return None
                
            if self._is_closed:
                logger.error("Indicator service is closed")
                return None
                
            cache_key = f"{symbol}_{timeframe}_{limit}"
            current_time = datetime.now()
            
            # Check cache
            if cache_key in self.data_cache:
                cached_data, timestamp = self.data_cache[cache_key]
                if current_time - timestamp < self.cache_expiry:
                    return cached_data
                    
            # Fetch new data with retry mechanism
            for attempt in range(self.max_retries):
                try:
                    # Use the same event loop for all operations
                    ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Validate data
                    if df.empty or len(df) < 2:
                        logger.warning(f"Not enough data for {symbol} {timeframe}")
                        return None
                        
                    # Check for missing values
                    if df.isnull().any().any():
                        logger.warning(f"Missing values in data for {symbol} {timeframe}")
                        df = df.ffill().bfill()
                        
                    # Update cache
                    self.data_cache[cache_key] = (df, current_time)
                    
                    return df
                    
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                        await asyncio.sleep(self.retry_delay)
                    else:
                        logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error in get_historical_data for {symbol}: {str(e)}")
            return None
            
    async def calculate_indicators(self, symbol: str, timeframe: str = "5m") -> Optional[pd.DataFrame]:
        """Calculate indicators with caching."""
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}"
            current_time = time.time()
            
            if cache_key in self._cache:
                cached_data, timestamp = self._cache[cache_key]
                if current_time - timestamp < self._cache_ttl:
                    return cached_data
            
            # Calculate indicators if not in cache or cache expired
            df = await self.get_historical_data(symbol, timeframe)
            if df is None or df.empty:
                return None
                
            # Calculate indicators
            df = await self._calculate_technical_indicators(df)
            
            # Update cache
            self._cache[cache_key] = (df, current_time)
            self._last_update[cache_key] = current_time
            
            return df
            
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