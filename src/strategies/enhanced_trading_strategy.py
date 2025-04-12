"""
Enhanced trading strategy implementation.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd

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
        
    async def _calculate_position_size(self, symbol: str, risk_per_trade: float) -> Optional[float]:
        """Calculate position size based on risk management.
        
        Args:
            symbol: Trading pair symbol
            risk_per_trade: Risk per trade as a percentage
            
        Returns:
            Optional[float]: Position size or None if calculation fails
        """
        try:
            if not self.binance_service:
                logger.error("Binance service not set")
                return None
                
            # Get account balance
            balance = await self.binance_service.get_account_balance()
            if not balance:
                logger.error(f"Failed to get balance for {symbol}")
                return None
                
            # Get current price
            ticker = await self.binance_service.get_ticker(symbol)
            if not ticker:
                logger.error(f"Failed to get ticker for {symbol}")
                return None
                
            current_price = float(ticker['last'])
            
            # Calculate position size
            risk_amount = balance * risk_per_trade
            position_size = risk_amount / current_price
            
            # Apply position size limits
            min_position = self.config['trading']['min_position_size']
            max_position = self.config['trading']['max_position_size']
            
            if position_size < min_position:
                logger.warning(f"Position size {position_size} below minimum {min_position}")
                return None
                
            if position_size > max_position:
                logger.warning(f"Position size {position_size} above maximum {max_position}")
                return None
                
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return None
            
    async def generate_signals(self, symbol: str, indicator_service: IndicatorService) -> List[Dict]:
        """Generate trading signals based on technical indicators."""
        try:
            if not self._is_initialized:
                logger.error("Strategy not initialized")
                return []
                
            if self._is_closed:
                logger.error("Strategy is closed")
                return []
                
            # Get historical data
            df = await indicator_service.get_historical_data(symbol)
            if df is None or df.empty:
                logger.warning(f"No data available for {symbol}")
                return []
                
            # Calculate indicators
            df = await indicator_service.calculate_indicators(symbol)
            if df is None or df.empty:
                logger.warning(f"No indicators available for {symbol}")
                return []
                
            signals = []
            current_price = df['close'].iloc[-1]
            
            # Calculate position size
            position_size = await self._calculate_position_size(symbol, self.config['trading']['risk_per_trade'])
            if position_size is None:
                logger.warning(f"Invalid position size for {symbol}")
                return []
                
            # Check for trend following signals
            if await self._check_trend_following(df):
                signals.append({
                    'symbol': symbol,
                    'side': 'buy',
                    'type': 'market',
                    'amount': position_size,
                    'stop_loss': await self._calculate_stop_loss(df, 'buy'),
                    'take_profit': await self._calculate_take_profit(df, 'buy')
                })
                
            # Check for mean reversion signals
            if await self._check_mean_reversion(df):
                signals.append({
                    'symbol': symbol,
                    'side': 'sell',
                    'type': 'market',
                    'amount': position_size,
                    'stop_loss': await self._calculate_stop_loss(df, 'sell'),
                    'take_profit': await self._calculate_take_profit(df, 'sell')
                })
                
            # Check for breakout signals
            if await self._check_breakout(df):
                signals.append({
                    'symbol': symbol,
                    'side': 'buy',
                    'type': 'market',
                    'amount': position_size,
                    'stop_loss': await self._calculate_stop_loss(df, 'buy'),
                    'take_profit': await self._calculate_take_profit(df, 'buy')
                })
                
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {str(e)}")
            return []
            
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
            
    def _calculate_stop_loss(self, df: pd.DataFrame) -> float:
        """Calculate stop loss level.
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            float: Stop loss level
        """
        try:
            # Use ATR to calculate stop loss
            atr = df['ATR'].iloc[-1]
            return atr * self.config['trading']['stop_loss_atr_multiplier']
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            return 0.0
            
    def _calculate_take_profit(self, df: pd.DataFrame) -> float:
        """Calculate take profit level.
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            float: Take profit level
        """
        try:
            # Use ATR to calculate take profit
            atr = df['ATR'].iloc[-1]
            return atr * self.config['trading']['take_profit_atr_multiplier']
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {str(e)}")
            return 0.0
            
    async def close(self):
        """Close the strategy."""
        try:
            if not self._is_initialized:
                logger.warning("Strategy was not initialized")
                return
                
            if self._is_closed:
                logger.warning("Strategy already closed")
                return
                
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
                new_take_profit = self.calculate_take_profit(current_price, new_stop_loss, "BUY")
            else:
                new_stop_loss = self.calculate_trailing_stop(current_price, atr, "SELL")
                new_take_profit = self.calculate_take_profit(current_price, new_stop_loss, "SELL")
                
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
            
    async def analyze_btc_volatility(self, timeframe: str = "5m") -> Dict:
        """Analyze BTC volatility.
        
        Args:
            timeframe: Timeframe for analysis
            
        Returns:
            Dict: BTC volatility analysis
        """
        try:
            # Get BTC data
            btc_df = await self.indicator_service.calculate_indicators("BTC/USDT", timeframe)
            if btc_df is None or btc_df.empty:
                return {
                    "volatility": 0.0,
                    "trend": "NEUTRAL",
                    "strength": 0.0,
                    "roc": 0.0,
                    "acceleration": 0.0,
                    "is_volatile": False,
                    "is_accelerating": False
                }
            
            # Calculate volatility
            current_price = btc_df["close"].iloc[-1]
            atr = btc_df.get("ATR", pd.Series([0.0])).iloc[-1]
            volatility = atr / current_price if current_price > 0 else 0.0
            
            # Calculate rate of change
            roc = (current_price - btc_df["close"].iloc[-5]) / btc_df["close"].iloc[-5] if len(btc_df) >= 5 else 0.0
            
            # Calculate acceleration
            prev_roc = (btc_df["close"].iloc[-6] - btc_df["close"].iloc[-10]) / btc_df["close"].iloc[-10] if len(btc_df) >= 10 else 0.0
            acceleration = roc - prev_roc
            
            # Determine trend
            ema_20 = btc_df["close"].ewm(span=20).mean()
            ema_50 = btc_df["close"].ewm(span=50).mean()
            trend = "UP" if ema_20.iloc[-1] > ema_50.iloc[-1] else "DOWN"
            strength = abs(ema_20.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1] if ema_50.iloc[-1] > 0 else 0.0
            
            return {
                "volatility": volatility,
                "trend": trend,
                "strength": strength,
                "roc": roc,
                "acceleration": acceleration,
                "is_volatile": volatility > 0.004,  # > 0.5%
                "is_accelerating": abs(acceleration) > 0.005  # > 0.5%
            }
            
        except Exception as e:
            logger.error(f"Error analyzing BTC volatility: {str(e)}")
            return {
                "volatility": 0.0,
                "trend": "NEUTRAL",
                "strength": 0.0,
                "roc": 0.0,
                "acceleration": 0.0,
                "is_volatile": False,
                "is_accelerating": False
            }
            
    async def analyze_altcoin_correlation(self, symbol: str, btc_volatility: Dict) -> Dict:
        """Analyze altcoin correlation with BTC.
        
        Args:
            symbol: Altcoin symbol
            btc_volatility: BTC volatility analysis
            
        Returns:
            Dict: Altcoin correlation analysis
        """
        try:
            # Get altcoin data
            alt_df = await self.indicator_service.calculate_indicators(symbol, "5m")
            if alt_df is None or alt_df.empty:
                return {
                    "correlation": 0.0,
                    "reaction": "NEUTRAL",
                    "strength": 0.0,
                    "volatility": 0.0,
                    "roc": 0.0,
                    "is_reacting": False
                }
            
            # Calculate altcoin volatility
            atr = alt_df.get("ATR", pd.Series([0.0])).iloc[-1]
            current_price = alt_df["close"].iloc[-1]
            alt_volatility = atr / current_price if current_price > 0 else 0.0
            
            # Calculate rate of change
            alt_roc = (current_price - alt_df["close"].iloc[-5]) / alt_df["close"].iloc[-5] if len(alt_df) >= 5 else 0.0
            
            # Calculate correlation
            correlation = await self.calculate_dynamic_correlation(symbol)
            
            # Analyze reaction
            reaction = "NEUTRAL"
            reaction_strength = 0.0
            is_reacting = False
            
            if btc_volatility.get("is_volatile", False):
                if abs(alt_roc) > abs(btc_volatility.get("roc", 0.0)) * 0.8:
                    reaction = "STRONG"
                    reaction_strength = alt_volatility / btc_volatility.get("volatility", 1.0)
                    is_reacting = True
                elif abs(alt_roc) > abs(btc_volatility.get("roc", 0.0)) * 0.5:
                    reaction = "MODERATE"
                    reaction_strength = alt_volatility / btc_volatility.get("volatility", 1.0)
                    is_reacting = True
                else:
                    reaction = "WEAK"
                    reaction_strength = alt_volatility / btc_volatility.get("volatility", 1.0)
                    is_reacting = True
            
            return {
                "correlation": float(correlation),
                "reaction": reaction,
                "strength": float(reaction_strength),
                "volatility": float(alt_volatility),
                "roc": float(alt_roc),
                "is_reacting": is_reacting
            }
            
        except Exception as e:
            logger.error(f"Error analyzing altcoin correlation: {str(e)}")
            return {
                "correlation": 0.0,
                "reaction": "NEUTRAL",
                "strength": 0.0,
                "volatility": 0.0,
                "roc": 0.0,
                "is_reacting": False
            }
            
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
            
    def calculate_stop_loss(self, entry_price: float, atr: float, position_type: str) -> float:
        """Calculate stop loss level.
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            position_type: Position type (BUY/SELL)
            
        Returns:
            float: Stop loss level
        """
        try:
            # Calculate base stop based on ATR
            base_stop = atr * 2
            
            # Adjust for coin price
            min_stop_percent = 0.02  # Minimum 2% for low-priced coins
            price_based_stop = entry_price * min_stop_percent
            
            # Use larger of ATR-based and price-based
            base_stop = max(base_stop, price_based_stop)
            
            # Adjust for position type
            if position_type == "BUY":
                stop_loss = entry_price - base_stop
                max_stop_distance = entry_price * 0.05  # Max 5%
                stop_loss = max(stop_loss, entry_price - max_stop_distance)
            else:
                stop_loss = entry_price + base_stop
                max_stop_distance = entry_price * 0.05  # Max 5%
                stop_loss = min(stop_loss, entry_price + max_stop_distance)
                
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            return entry_price
            
    def calculate_take_profit(self, entry_price: float, stop_loss: float, position_type: str) -> float:
        """Calculate take profit level.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss level
            position_type: Position type (BUY/SELL)
            
        Returns:
            float: Take profit level
        """
        try:
            # Calculate risk
            risk = abs(entry_price - stop_loss)
            
            # Ensure minimum take profit distance
            min_tp_percent = 0.04  # Minimum 4% for low-priced coins
            min_tp_distance = entry_price * min_tp_percent
            
            # Calculate take profit with 2:1 R:R
            if position_type == "BUY":
                take_profit = entry_price + max(risk * 2, min_tp_distance)
            else:
                take_profit = entry_price - max(risk * 2, min_tp_distance)
                
            # Ensure take profit is not too far
            max_profit_distance = entry_price * 0.1  # Max 10%
            if position_type == "BUY":
                take_profit = min(take_profit, entry_price + max_profit_distance)
            else:
                take_profit = max(take_profit, entry_price - max_profit_distance)
                
            return take_profit
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {str(e)}")
            return entry_price
            
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