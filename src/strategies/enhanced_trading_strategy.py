"""
Enhanced trading strategy implementation.
"""

import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np
import time

from src.services.indicator_service import IndicatorService
from src.services.sentiment_service import SentimentService

logger = logging.getLogger(__name__)

class EnhancedTradingStrategy:
    """Enhanced trading strategy with multiple indicators and risk management."""
    
    def __init__(self, config: Dict):
        """Initialize the strategy.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.indicator_service = None
        self.sentiment_service = None
        self.binance_service = None
        
        # Trading parameters
        self.min_volume_ratio = 1.2  # Tỷ lệ volume tối thiểu so với trung bình
        self.max_volatility_ratio = 2.0  # Tỷ lệ biến động tối đa so với trung bình
        self.min_adx = 25  # ADX tối thiểu cho xu hướng mạnh
        self.max_bb_width = 0.1  # Độ rộng tối đa của dải Bollinger
        
        # BTC correlation parameters
        self.min_btc_correlation = 0.5  # Tương quan tối thiểu với BTC
        self.max_btc_lag = 2  # Độ trễ tối đa cho phép (số nến)
        self.btc_volatility_threshold = 0.004  # Ngưỡng biến động BTC (0.4%)
        
        # Timeframe weights
        self.timeframe_weights = {
            "5m": 0.2,
            "15m": 0.3,
            "1h": 0.3,
            "4h": 0.2
        }
        
        # Signal score weights
        self.score_weights = {
            "technical": 0.3,
            "market": 0.2,
            "timeframe": 0.2,
            "btc": 0.15,
            "sentiment": 0.15
        }
        
        self._is_initialized = False
        self._is_closed = False
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_update = {}
        
    async def initialize(self) -> bool:
        """Initialize the strategy.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            if self._is_initialized:
                logger.warning("Strategy already initialized")
                return True
                
            # Initialize services
            self.indicator_service = IndicatorService(self.config)
            self.sentiment_service = SentimentService(self.config)
            
            # Initialize services
            if not await self.indicator_service.initialize():
                logger.error("Failed to initialize indicator service")
                return False
                
            if not await self.sentiment_service.initialize():
                logger.error("Failed to initialize sentiment service")
                return False
                
            self._is_initialized = True
            logger.info("Strategy initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize strategy: {str(e)}")
            return False
            
    def set_binance_service(self, binance_service):
        """Set the Binance service for trading operations.
        
        Args:
            binance_service: Binance service instance
        """
        self.binance_service = binance_service
        
    async def _calculate_position_size(self, symbol: str, risk_per_trade: float, current_price: float) -> Optional[float]:
        """Calculate position size based on risk management."""
        try:
            if not self.binance_service:
                logger.error("Binance service not set")
                return None
                
            # Validate inputs
            if current_price <= 0:
                logger.error(f"Invalid current price: {current_price}")
                return None
                
            if risk_per_trade <= 0:
                logger.error(f"Invalid risk per trade: {risk_per_trade}")
                return None
                
            # Get account balance
            balance = await self.binance_service.get_account_balance()
            if not balance:
                logger.error(f"Failed to get balance for {symbol}")
                return None
                
            # Get USDT balance
            usdt_balance = balance.get('USDT', {}).get('total', 0)
            if not usdt_balance or float(usdt_balance) <= 0:
                logger.error(f"Invalid USDT balance: {usdt_balance}")
                return None
                
            # Get leverage from config
            leverage = self.config['trading'].get('leverage', 10)
            if leverage <= 0:
                logger.error(f"Invalid leverage: {leverage}")
                return None
                
            # Calculate risk amount in USDT
            risk_amount = float(usdt_balance) * risk_per_trade
            
            # Calculate position size with leverage
            position_size = (risk_amount * leverage) / current_price
            
            # Calculate position notional value
            position_notional = position_size * current_price
            
            # Check minimum notional value (5 USDT)
            min_notional = 5.0  # Binance minimum notional value
            if position_notional < min_notional:
                logger.warning(f"Position notional value {position_notional} below minimum {min_notional} USDT")
                return None
                
            # Calculate maximum position size based on available balance and leverage
            max_position_size = (float(usdt_balance) * leverage) / current_price
            
            # Check if position size exceeds maximum
            if position_size > max_position_size:
                logger.warning(f"Position size {position_size} exceeds available balance with leverage")
                return None
                
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return None
            
    async def generate_signals(self, symbol: str, indicator_service: IndicatorService) -> Optional[Dict]:
        """Generate trading signals based on technical indicators and BTC correlation.
        
        Args:
            symbol: Trading pair symbol
            indicator_service: Indicator service instance
            
        Returns:
            Optional[Dict]: Trading signals or None if no signal
        """
        try:
            # Calculate indicators using indicator service
            df = await indicator_service.calculate_indicators(symbol)
            if df is None:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Calculate position size
            position_size = await self._calculate_position_size(symbol, self.config['trading']['risk_per_trade'], current_price)
            if not position_size:
                logger.warning(f"Invalid position size for {symbol}")
                return None
                
            # Calculate stop loss and take profit
            stop_loss = self._calculate_stop_loss(df)
            take_profit = self._calculate_take_profit(df, current_price)
            
            # Analyze BTC volatility
            btc_volatility = await self.analyze_btc_volatility()
            
            # Analyze altcoin correlation
            altcoin_correlation = await self.analyze_altcoin_correlation(symbol, btc_volatility)
            
            # Analyze multiple timeframes
            timeframe_analysis = await self.analyze_multiple_timeframes(symbol)
            
            # Check volume condition
            volume_ma = df['volume'].rolling(20).mean()
            volume_condition = df['volume'].iloc[-1] > volume_ma.iloc[-1] * 1.2
            
            # Check ADX condition
            adx_condition = df['ADX'].iloc[-1] > 20
            
            # print(f"BTC volatility: {btc_volatility}")
            # print(f"Altcoin correlation: {altcoin_correlation}")
            # print(f"Volume condition: {volume_condition}")
            # print(f"ADX condition: {adx_condition}")
            # logger.info(f"{symbol} - BTC volatility: {btc_volatility} - Altcoin correlation: {altcoin_correlation} - Volume condition: {volume_condition} - ADX condition: {adx_condition}")
            
            # Generate signals with BTC correlation check
            if (df['RSI'].iloc[-1] < 35 and  # Giảm từ 30 xuống 35
                df['MACD'].iloc[-1] > df['MACD_SIGNAL'].iloc[-1] and
                btc_volatility["trend"] == "UP" and
                btc_volatility["is_volatile"] and
                altcoin_correlation["reaction"] in ["STRONG", "MODERATE", "WEAK"] and  # Thêm WEAK
                altcoin_correlation["correlation"] > 0 and
                altcoin_correlation["lag"] <= self.max_btc_lag + 1 and  # Tăng độ trễ cho phép
                volume_condition and  # Thêm điều kiện volume
                adx_condition and  # Thêm điều kiện ADX
                self.check_trend_conflicts(timeframe_analysis)):  # Thêm điều kiện multiple timeframe
                
                return {
                    'symbol': symbol,
                    'side': 'buy',
                    'type': 'market',
                    'amount': position_size,
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'position_size': position_size,
                    'btc_volatility': btc_volatility,
                    'altcoin_correlation': altcoin_correlation,
                    'timeframe_analysis': timeframe_analysis
                }
                
            elif (df['RSI'].iloc[-1] > 65 and  # Giảm từ 70 xuống 65
                  df['MACD'].iloc[-1] < df['MACD_SIGNAL'].iloc[-1] and
                  btc_volatility["trend"] == "DOWN" and
                  btc_volatility["is_volatile"] and
                  altcoin_correlation["reaction"] in ["STRONG", "MODERATE", "WEAK"] and  # Thêm WEAK
                  altcoin_correlation["correlation"] < 0 and
                  altcoin_correlation["lag"] <= self.max_btc_lag + 1 and  # Tăng độ trễ cho phép
                  volume_condition and  # Thêm điều kiện volume
                  adx_condition and  # Thêm điều kiện ADX
                  self.check_trend_conflicts(timeframe_analysis)):  # Thêm điều kiện multiple timeframe
                
                return {
                    'symbol': symbol,
                    'side': 'sell',
                    'type': 'market',
                    'amount': position_size,
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'position_size': position_size,
                    'btc_volatility': btc_volatility,
                    'altcoin_correlation': altcoin_correlation,
                    'timeframe_analysis': timeframe_analysis
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return None
            
    def _check_trend_following(self, df: pd.DataFrame) -> bool:
        """Check for trend following signals.
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            bool: True if trend following signal is present
        """
        try:
            # Check for EMA crossover
            ema_crossover = (
                df['EMA_FAST'].iloc[-2] < df['EMA_SLOW'].iloc[-2] and
                df['EMA_FAST'].iloc[-1] > df['EMA_SLOW'].iloc[-1]
            )
            
            # Check for MACD crossover
            macd_crossover = (
                df['MACD'].iloc[-2] < df['MACD_SIGNAL'].iloc[-2] and
                df['MACD'].iloc[-1] > df['MACD_SIGNAL'].iloc[-1]
            )
            
            # Check for ADX strength
            adx_strong = df['ADX'].iloc[-1] > 25
            
            return (ema_crossover or macd_crossover) and adx_strong
            
        except Exception as e:
            logger.error(f"Error checking trend following: {str(e)}")
            return False
            
    def _check_mean_reversion(self, df: pd.DataFrame) -> bool:
        """Check for mean reversion signals.
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            bool: True if mean reversion signal is present
        """
        try:
            # Check for RSI extremes
            rsi_extreme = df['RSI'].iloc[-1] < 30 or df['RSI'].iloc[-1] > 70
            
            # Check for price near Bollinger Bands
            price_near_bands = (
                df['close'].iloc[-1] < df['BB_lower'].iloc[-1] or
                df['close'].iloc[-1] > df['BB_upper'].iloc[-1]
            )
            
            return rsi_extreme and price_near_bands
            
        except Exception as e:
            logger.error(f"Error checking mean reversion: {str(e)}")
            return False
            
    def _check_breakout(self, df: pd.DataFrame) -> bool:
        """Check for breakout signals.
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            bool: True if breakout signal is present
        """
        try:
            # Check for price breaking Bollinger Bands
            price_breakout = (
                df['close'].iloc[-1] > df['BB_upper'].iloc[-1] or
                df['close'].iloc[-1] < df['BB_lower'].iloc[-1]
            )
            
            # Check for volume confirmation
            volume_confirmation = df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1]
            
            return price_breakout and volume_confirmation
            
        except Exception as e:
            logger.error(f"Error checking breakout: {str(e)}")
            return False
            
    def _calculate_stop_loss(self, df: pd.DataFrame, position_type: str = 'buy') -> float:
        """Calculate dynamic stop loss level based on market conditions.
        
        Args:
            df: DataFrame with indicators
            position_type: Position type (buy/sell)
            
        Returns:
            float: Stop loss level
        """
        try:
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Calculate stop loss with fixed 2% distance
            if position_type == 'buy':
                stop_loss = current_price * 0.98  # 2% below current price
            else:
                stop_loss = current_price * 1.02  # 2% above current price
                
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            return current_price
            
    def _calculate_take_profit(self, df: pd.DataFrame, position_type: str = 'buy') -> float:
        """Calculate dynamic take profit level based on market conditions.
        
        Args:
            df: DataFrame with indicators
            position_type: Position type (buy/sell)
            
        Returns:
            float: Take profit level
        """
        try:
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Calculate take profit with fixed 4% distance
            if position_type == 'buy':
                take_profit = current_price * 1.04  # 4% above current price
            else:
                take_profit = current_price * 0.96  # 4% below current price
                
            return take_profit
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {str(e)}")
            return current_price
            
    async def close(self):
        """Close the strategy and clear cache."""
        try:
            self.clear_cache()
            self._is_closed = True
            logger.info("Strategy closed")
        except Exception as e:
            logger.error(f"Error closing strategy: {str(e)}")
            
    async def should_close_position(self, position: Dict, current_price: float) -> bool:
        """Check if a position should be closed.
        
        Args:
            position: Position information
            current_price: Current market price
            
        Returns:
            bool: True if position should be closed, False otherwise
        """
        try:
            # Get historical data and indicators
            df = await self.indicator_service.calculate_indicators(position["symbol"])
            if df is None or df.empty:
                return False
                
            # Check stop loss
            if position["side"] == "BUY":
                if current_price <= position["stop_loss"]:
                    return True
            else:
                if current_price >= position["stop_loss"]:
                    return True
                    
            # Check take profit
            if position["side"] == "BUY":
                if current_price >= position["take_profit"]:
                    return True
            else:
                if current_price <= position["take_profit"]:
                    return True
                    
            # Check trend reversal
            current_trend = self.get_trend(df)
            if position["side"] == "BUY" and current_trend == "DOWN":
                return True
            elif position["side"] == "SELL" and current_trend == "UP":
                return True
                
            # Check sentiment change
            sentiment = await self.sentiment_service.analyze_sentiment(position["symbol"])
            if position["side"] == "BUY" and sentiment["trend"] == "BEARISH":
                return True
            elif position["side"] == "SELL" and sentiment["trend"] == "BULLISH":
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking position close: {str(e)}")
            return False
            
    async def should_update_stops(self, position: Dict, current_price: float) -> bool:
        """Check if stop loss and take profit should be updated.
        
        Args:
            position: Position information
            current_price: Current market price
            
        Returns:
            bool: True if stops should be updated, False otherwise
        """
        try:
            # Get historical data and indicators
            df = await self.indicator_service.calculate_indicators(position["symbol"])
            if df is None or df.empty:
                return False
                
            # Check if price moved significantly
            if position["side"] == "BUY":
                price_change = (current_price - position["entry_price"]) / position["entry_price"]
                if price_change > 0.02:  # 2% move
                    return True
            else:
                price_change = (position["entry_price"] - current_price) / position["entry_price"]
                if price_change > 0.02:  # 2% move
                    return True
                    
            # Check if trend is strengthening
            current_trend = self.get_trend(df)
            if position["side"] == "BUY" and current_trend == "UP":
                return True
            elif position["side"] == "SELL" and current_trend == "DOWN":
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking stop update: {str(e)}")
            return False
            
    async def calculate_new_stops(self, position: Dict, current_price: float) -> Dict:
        """Calculate new stop loss and take profit levels.
        
        Args:
            position: Position information
            current_price: Current market price
            
        Returns:
            Dict: New stop loss and take profit levels
        """
        try:
            # Get historical data and indicators
            df = await self.indicator_service.calculate_indicators(position["symbol"])
            if df is None or df.empty:
                return position
                
            atr = df["ATR"].iloc[-1]
            
            # Calculate new stops
            if position["side"] == "BUY":
                new_stop_loss = self.calculate_trailing_stop(current_price, atr, "BUY")
                new_take_profit = self.calculate_take_profit(df, current_price)
            else:
                new_stop_loss = self.calculate_trailing_stop(current_price, atr, "SELL")
                new_take_profit = self.calculate_take_profit(df, current_price)
                
            return {
                "stop_loss": new_stop_loss,
                "take_profit": new_take_profit
            }
            
        except Exception as e:
            logger.error(f"Error calculating new stops: {str(e)}")
            return position
            
    def get_trend(self, df: pd.DataFrame) -> str:
        """Get current trend direction.
        
        Args:
            df: Price data DataFrame
            
        Returns:
            str: Trend direction (UP/DOWN)
        """
        try:
            # Calculate EMAs
            ema_20 = df["close"].ewm(span=20).mean()
            ema_50 = df["close"].ewm(span=50).mean()
            
            # Determine trend
            if ema_20.iloc[-1] > ema_50.iloc[-1]:
                return "UP"
            else:
                return "DOWN"
                
        except Exception as e:
            logger.error(f"Error getting trend: {str(e)}")
            return "NEUTRAL"
            
    async def analyze_multiple_timeframes(self, symbol: str) -> Dict:
        """Analyze market on multiple timeframes.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict: Analysis results for each timeframe
        """
        try:
            results = {}
            for timeframe, weight in self.timeframe_weights.items():
                # Get data for timeframe
                df = await self.indicator_service.calculate_indicators(symbol, timeframe)
                if df is None or df.empty:
                    continue
                    
                # Analyze trend
                ema_20 = df["close"].ewm(span=20).mean()
                ema_50 = df["close"].ewm(span=50).mean()
                ema_200 = df["close"].ewm(span=200).mean()
                
                trend = "UP" if ema_20.iloc[-1] > ema_50.iloc[-1] and ema_50.iloc[-1] > ema_200.iloc[-1] else "DOWN"
                strength = abs(ema_20.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1]
                
                # Analyze momentum
                rsi = df["RSI"].iloc[-1]
                macd = df["MACD"].iloc[-1]
                macd_signal = df["MACD_SIGNAL"].iloc[-1]
                
                # Analyze Ichimoku
                tenkan_sen = df["tenkan_sen"].iloc[-1]
                kijun_sen = df["kijun_sen"].iloc[-1]
                senkou_span_a = df["senkou_span_a"].iloc[-1]
                senkou_span_b = df["senkou_span_b"].iloc[-1]
                
                results[timeframe] = {
                    "trend": trend,
                    "strength": strength,
                    "rsi": rsi,
                    "macd_hist": macd - macd_signal,
                    "ichimoku": {
                        "tenkan_sen": tenkan_sen,
                        "kijun_sen": kijun_sen,
                        "senkou_span_a": senkou_span_a,
                        "senkou_span_b": senkou_span_b
                    },
                    "weight": weight
                }
                
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing multiple timeframes: {str(e)}")
            return {}
            
    async def analyze_btc_volatility(self) -> Dict:
        """Analyze BTC volatility with caching."""
        try:
            current_time = time.time()
            cache_key = "btc_volatility"
            
            if cache_key in self._cache:
                cached_data, timestamp = self._cache[cache_key]
                if current_time - timestamp < self._cache_ttl:
                    return cached_data
            
            # Calculate fresh analysis
            analysis = await self._calculate_btc_volatility()
            if analysis:
                self._cache[cache_key] = (analysis, current_time)
                self._last_update[cache_key] = current_time
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing BTC volatility: {str(e)}")
            return None
            
    async def analyze_altcoin_correlation(self, symbol: str, btc_volatility: Dict) -> Dict:
        """Analyze altcoin correlation with caching."""
        try:
            current_time = time.time()
            cache_key = f"correlation_{symbol}"
            
            if cache_key in self._cache:
                cached_data, timestamp = self._cache[cache_key]
                if current_time - timestamp < self._cache_ttl:
                    return cached_data
            
            # Calculate fresh analysis
            analysis = await self._calculate_altcoin_correlation(symbol)
            if analysis:
                self._cache[cache_key] = (analysis, current_time)
                self._last_update[cache_key] = current_time
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing altcoin correlation: {str(e)}")
            return None
            
    def clear_cache(self):
        """Clear the strategy cache."""
        self._cache.clear()
        self._last_update.clear()
        
    def calculate_trailing_stop(self, current_price: float, atr: float, position_type: str) -> float:
        """Calculate trailing stop level.
        
        Args:
            current_price: Current market price
            atr: Average True Range
            position_type: Position type (BUY/SELL)
            
        Returns:
            float: Trailing stop level
        """
        try:
            # Calculate base trailing stop
            base_stop = atr * 1.5  # Tighter than initial stop
            
            # Adjust for position type
            if position_type == "BUY":
                trailing_stop = current_price - base_stop
            else:
                trailing_stop = current_price + base_stop
                
            return trailing_stop
            
        except Exception as e:
            logger.error(f"Error calculating trailing stop: {str(e)}")
            return current_price
            
    async def calculate_dynamic_correlation(self, symbol: str, window: int = 20) -> float:
        """Calculate dynamic correlation between altcoin and BTC.
        
        Args:
            symbol: Altcoin symbol
            window: Correlation window
            
        Returns:
            float: Correlation coefficient
        """
        try:
            # Get BTC and altcoin data
            btc_df = await self.indicator_service.calculate_indicators("BTC/USDT", "5m", window)
            alt_df = await self.indicator_service.calculate_indicators(symbol, "5m", window)
            
            if btc_df is None or alt_df is None or btc_df.empty or alt_df.empty:
                return 0.0
                
            # Calculate returns
            btc_returns = btc_df["close"].pct_change().dropna()
            alt_returns = alt_df["close"].pct_change().dropna()
            
            # Calculate correlation
            correlation = btc_returns.corr(alt_returns)
            
            return correlation
            
        except Exception as e:
            logger.error(f"Error calculating dynamic correlation: {str(e)}")
            return 0.0
            
    async def calculate_signal_score(self, df: pd.DataFrame, timeframe_analysis: Dict, 
                                   btc_volatility: Dict, altcoin_correlation: Dict,
                                   sentiment: Dict) -> float:
        """Calculate signal score.
        
        Args:
            df: Price data DataFrame
            timeframe_analysis: Multi-timeframe analysis
            btc_volatility: BTC volatility analysis
            altcoin_correlation: Altcoin correlation analysis
            sentiment: Sentiment analysis
            
        Returns:
            float: Signal score
        """
        try:
            score = 0.0
            
            # 1. Technical Analysis Score (0-3 points)
            if "RSI" in df and "MACD" in df and "MACD_SIGNAL" in df:
                rsi = df["RSI"].iloc[-1]
                macd = df["MACD"].iloc[-1]
                macd_signal = df["MACD_SIGNAL"].iloc[-1]
                
                # RSI conditions
                if rsi < 30 or rsi > 70:
                    score += 1.5
                elif (rsi < 35 and macd > macd_signal) or (rsi > 65 and macd < macd_signal):
                    score += 1
                elif macd > macd_signal or macd < macd_signal:
                    score += 0.5
                    
            # 2. Market Conditions Score (0-2 points)
            if df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 1.5:
                score += 1
            if df["ATR"].iloc[-1] < df["ATR"].rolling(20).mean().iloc[-1] * 1.5:
                score += 0.5
            if df["ADX"].iloc[-1] > 25:
                score += 0.5
                
            # 3. Multi-timeframe Analysis Score (0-2 points)
            if self.check_trend_conflicts(timeframe_analysis):
                score += 1
            if sum(1 for data in timeframe_analysis.values() if data["strength"] > 0.02) >= len(timeframe_analysis) * 0.5:
                score += 1
                
            # 4. BTC Volatility Score (0-1.5 points)
            if btc_volatility["is_volatile"]:
                score += 0.5
            if btc_volatility["is_accelerating"]:
                score += 0.5
            if btc_volatility["strength"] > 0.02:
                score += 0.5
                
            # 5. Altcoin Correlation Score (0-1.5 points)
            if altcoin_correlation["is_reacting"]:
                score += 0.5
            if altcoin_correlation["correlation"] > 0.7:
                score += 0.5
            if altcoin_correlation["strength"] > 0.02:
                score += 0.5
                
            # 6. Sentiment Score (0-1.5 points)
            if sentiment["confidence"] > 0.5:
                if sentiment["trend"] in ["BULLISH", "BEARISH"]:
                    score += 1
                elif sentiment["trend"] == "NEUTRAL":
                    score += 0.5
                    
            # Apply weights
            score = (
                score * self.score_weights["technical"] +
                score * self.score_weights["market"] +
                score * self.score_weights["timeframe"] +
                score * self.score_weights["btc"] +
                score * self.score_weights["sentiment"]
            )
                
            return score
            
        except Exception as e:
            logger.error(f"Error calculating signal score: {str(e)}")
            return 0.0
            
    def check_trend_conflicts(self, timeframe_analysis: Dict) -> bool:
        """Check for trend conflicts between timeframes.
        
        Args:
            timeframe_analysis: Multi-timeframe analysis
            
        Returns:
            bool: True if trends are aligned, False otherwise
        """
        try:
            if not timeframe_analysis:
                return False
                
            trends = [data["trend"] for data in timeframe_analysis.values()]
            weights = [data["weight"] for data in timeframe_analysis.values()]
            
            weighted_up = sum(w for t, w in zip(trends, weights) if t == "UP")
            weighted_down = sum(w for t, w in zip(trends, weights) if t == "DOWN")
            
            return weighted_up >= 0.67 or weighted_down >= 0.67
            
        except Exception as e:
            logger.error(f"Error checking trend conflicts: {str(e)}")
            return False
            
    def check_volume_condition(self, df: pd.DataFrame) -> bool:
        """Check volume condition.
        
        Args:
            df: Price data DataFrame
            
        Returns:
            bool: True if volume condition is met, False otherwise
        """
        try:
            volume_ma = df["volume"].rolling(20).mean()
            if volume_ma.iloc[-1] == 0:
                return False
            volume_ratio = df["volume"].iloc[-1] / volume_ma.iloc[-1]
            return volume_ratio >= self.min_volume_ratio
            
        except Exception as e:
            logger.error(f"Error checking volume condition: {str(e)}")
            return False
            
    def check_volatility_condition(self, df: pd.DataFrame) -> bool:
        """Check volatility condition.
        
        Args:
            df: Price data DataFrame
            
        Returns:
            bool: True if volatility condition is met, False otherwise
        """
        try:
            atr = df["ATR"].iloc[-1]
            atr_ma = df["ATR"].rolling(20).mean().iloc[-1]
            if atr_ma == 0:
                return False
            volatility_ratio = atr / atr_ma
            return volatility_ratio <= self.max_volatility_ratio
            
        except Exception as e:
            logger.error(f"Error checking volatility condition: {str(e)}")
            return False
            
    def check_adx_condition(self, df: pd.DataFrame) -> bool:
        """Check ADX condition.
        
        Args:
            df: Price data DataFrame
            
        Returns:
            bool: True if ADX condition is met, False otherwise
        """
        try:
            return df["ADX"].iloc[-1] >= self.min_adx
            
        except Exception as e:
            logger.error(f"Error checking ADX condition: {str(e)}")
            return False
            
    def check_bollinger_condition(self, df: pd.DataFrame) -> bool:
        """Check Bollinger Bands condition.
        
        Args:
            df: Price data DataFrame
            
        Returns:
            bool: True if Bollinger condition is met, False otherwise
        """
        try:
            bb_width = (df["BB_upper"].iloc[-1] - df["BB_lower"].iloc[-1]) / df["BB_middle"].iloc[-1]
            return bb_width <= self.max_bb_width
            
        except Exception as e:
            logger.error(f"Error checking Bollinger condition: {str(e)}")
            return False
            
    async def _calculate_btc_volatility(self) -> Dict:
        """Calculate BTC volatility metrics."""
        try:
            # Get BTC data
            btc_data = await self.indicator_service.calculate_indicators("BTCUSDT")
            if btc_data is None or btc_data.empty:
                return None
                
            # Calculate volatility metrics
            atr = btc_data['ATR'].iloc[-1]
            roc = btc_data['ROC'].iloc[-1]
            ema = btc_data['EMA_FAST'].iloc[-1]
            current_price = btc_data['close'].iloc[-1]
            
            # Calculate volatility score (0-100)
            volatility_score = min(100, (atr / current_price) * 1000)
            
            # Determine volatility state
            is_volatile = volatility_score > 50
            is_accelerating = abs(roc) > 1.0
            
            # Determine trend based on EMA and current price
            trend = "UP" if current_price > ema else "DOWN"
            
            return {
                'volatility_score': volatility_score,
                'is_volatile': is_volatile,
                'is_accelerating': is_accelerating,
                'atr': atr,
                'roc': roc,
                'ema': ema,
                'current_price': current_price,
                'trend': trend
            }
            
        except Exception as e:
            logger.error(f"Error calculating BTC volatility: {str(e)}")
            return None
            
    async def _calculate_altcoin_correlation(self, symbol: str) -> Dict:
        """Calculate correlation between altcoin and BTC."""
        try:
            # Get altcoin and BTC data
            altcoin_data = await self.indicator_service.calculate_indicators(symbol)
            btc_data = await self.indicator_service.calculate_indicators("BTCUSDT")
            
            if altcoin_data is None or btc_data is None or altcoin_data.empty or btc_data.empty:
                return None
                
            # Clean data - remove NaN and infinite values
            altcoin_returns = altcoin_data['close'].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            btc_returns = btc_data['close'].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            
            # Ensure we have enough data points and no NaN values
            if len(altcoin_returns) < 2 or len(btc_returns) < 2:
                return None
                
            # Align the data
            common_index = altcoin_returns.index.intersection(btc_returns.index)
            if len(common_index) < 2:
                return None
                
            altcoin_returns = altcoin_returns[common_index]
            btc_returns = btc_returns[common_index]
            
            # Calculate correlation coefficient with error handling
            try:
                # Calculate standard deviations first
                alt_std = altcoin_returns.std()
                btc_std = btc_returns.std()
                
                # Check for zero standard deviation
                if alt_std == 0 or btc_std == 0:
                    correlation = 0.0
                else:
                    # Calculate covariance
                    covariance = ((altcoin_returns - altcoin_returns.mean()) * 
                                (btc_returns - btc_returns.mean())).mean()
                    # Calculate correlation
                    correlation = covariance / (alt_std * btc_std)
                    
                if np.isnan(correlation):
                    correlation = 0.0
            except Exception as e:
                logger.warning(f"Error calculating correlation: {str(e)}")
                correlation = 0.0
            
            # Calculate price change correlation with error handling
            try:
                # Calculate standard deviations for returns
                alt_returns_std = altcoin_returns.std()
                btc_returns_std = btc_returns.std()
                
                # Check for zero standard deviation
                if alt_returns_std == 0 or btc_returns_std == 0:
                    returns_correlation = 0.0
                else:
                    # Calculate covariance for returns
                    returns_covariance = ((altcoin_returns - altcoin_returns.mean()) * 
                                        (btc_returns - btc_returns.mean())).mean()
                    # Calculate correlation for returns
                    returns_correlation = returns_covariance / (alt_returns_std * btc_returns_std)
                    
                if np.isnan(returns_correlation):
                    returns_correlation = 0.0
            except Exception as e:
                logger.warning(f"Error calculating returns correlation: {str(e)}")
                returns_correlation = 0.0
            
            # Calculate reaction strength with bounds
            reaction_strength = min(100, max(0, abs(returns_correlation) * 100))
            
            # Determine correlation state
            is_strongly_correlated = abs(correlation) > 0.7
            is_reacting = abs(returns_correlation) > 0.5
            
            return {
                'correlation': correlation,
                'returns_correlation': returns_correlation,
                'reaction_strength': reaction_strength,
                'is_strongly_correlated': is_strongly_correlated,
                'is_reacting': is_reacting
            }
            
        except Exception as e:
            logger.error(f"Error calculating altcoin correlation: {str(e)}")
            return None 