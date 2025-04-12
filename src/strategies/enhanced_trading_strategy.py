"""
Enhanced trading strategy implementation.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import ccxt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

class EnhancedTradingStrategy:
    """Enhanced trading strategy with multiple indicators and risk management."""
    
    def __init__(self, config: Dict):
        """Initialize the strategy.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.timeframes = ["5m", "15m", "1h", "4h"]  # Multi-timeframe analysis
        
        # Trading parameters
        self.min_volume_ratio = 1.2  # Tỷ lệ volume tối thiểu so với trung bình
        self.max_volatility_ratio = 2.0  # Tỷ lệ biến động tối đa so với trung bình
        self.min_adx = 25  # ADX tối thiểu cho xu hướng mạnh
        self.max_bb_width = 0.1  # Độ rộng tối đa của dải Bollinger
        
        # Initialize Binance client
        self.exchange = ccxt.binance({
            'apiKey': config['api']['binance']['api_key'],
            'secret': config['api']['binance']['api_secret'],
            'enableRateLimit': True
        })
        
    async def generate_signals(self, symbol: str) -> List[Dict]:
        """Generate trading signals for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            List[Dict]: List of trading signals
        """
        try:
            signal = await self.generate_signal(symbol)
            if signal:
                return [signal]
            return []
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {str(e)}")
            return []
            
    async def generate_signal(self, symbol: str) -> Optional[Dict]:
        """Generate trading signal for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Optional[Dict]: Trading signal if generated, None otherwise
        """
        try:
            # Get historical data
            df = await self.get_historical_data(symbol)
            if df is None or df.empty:
                return None
                
            # Calculate indicators
            df = await self.calculate_indicators(df)
            
            # Check basic conditions
            if not all([
                self.check_volume_condition(df),
                self.check_volatility_condition(df),
                self.check_adx_condition(df),
                self.check_bollinger_condition(df)
            ]):
                return None
                
            # Analyze multiple timeframes
            timeframe_analysis = await self.analyze_multiple_timeframes(symbol)
            
            # Analyze BTC volatility
            btc_volatility = await self.analyze_btc_volatility()
            
            # Analyze altcoin correlation
            altcoin_correlation = await self.analyze_altcoin_correlation(symbol, btc_volatility)
            
            # Calculate signal score
            signal_score = await self.calculate_signal_score(df, timeframe_analysis, btc_volatility, altcoin_correlation)
            
            self.logger.info(f"{symbol} Signal score: {signal_score}")
            self.logger.info(f"BTC volatility: {btc_volatility}")
            
            # Require minimum score of 6
            if signal_score >= 6:
                current_price = df["close"].iloc[-1]
                atr = df["ATR"].iloc[-1]
                
                # Check buy conditions
                if (btc_volatility["trend"] == "UP" and btc_volatility["is_volatile"] and 
                    altcoin_correlation["reaction"] in ["STRONG", "MODERATE"]):
                    stop_loss = self.calculate_stop_loss(current_price, atr, "BUY")
                    take_profit = self.calculate_take_profit(current_price, stop_loss, "BUY")
                    return {
                        "symbol": symbol,
                        "side": "BUY",
                        "price": current_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "atr": atr,
                        "score": signal_score
                    }
                    
                # Check sell conditions
                elif (btc_volatility["trend"] == "DOWN" and btc_volatility["is_volatile"] and 
                      altcoin_correlation["reaction"] in ["STRONG", "MODERATE"]):
                    stop_loss = self.calculate_stop_loss(current_price, atr, "SELL")
                    take_profit = self.calculate_take_profit(current_price, stop_loss, "SELL")
                    return {
                        "symbol": symbol,
                        "side": "SELL",
                        "price": current_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "atr": atr,
                        "score": signal_score
                    }
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {str(e)}")
            return None
            
    async def should_close_position(self, position: Dict, current_price: float) -> bool:
        """Check if a position should be closed.
        
        Args:
            position: Position information
            current_price: Current market price
            
        Returns:
            bool: True if position should be closed, False otherwise
        """
        try:
            # Get historical data
            df = await self.get_historical_data(position["symbol"])
            if df is None or df.empty:
                return False
                
            # Calculate indicators
            df = await self.calculate_indicators(df)
            
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
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking position close: {str(e)}")
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
            # Get historical data
            df = await self.get_historical_data(position["symbol"])
            if df is None or df.empty:
                return False
                
            # Calculate indicators
            df = await self.calculate_indicators(df)
            
            # Check if price moved significantly
            if position["side"] == "BUY":
                price_change = (current_price - position["entry_price"]) / position["entry_price"]
                if price_change > 0.02:  # 2% move
                    return True
            else:
                price_change = (position["entry_price"] - current_price) / position["entry_price"]
                if price_change > 0.02:  # 2% move
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking stop update: {str(e)}")
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
            # Get historical data
            df = await self.get_historical_data(position["symbol"])
            if df is None or df.empty:
                return position
                
            # Calculate indicators
            df = await self.calculate_indicators(df)
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
            self.logger.error(f"Error calculating new stops: {str(e)}")
            return position
            
    async def get_historical_data(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> Optional[pd.DataFrame]:
        """Get historical price data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            limit: Number of candles to fetch
            
        Returns:
            Optional[pd.DataFrame]: Historical price data
        """
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return None
            
    async def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators.
        
        Args:
            df: Price data DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with indicators
        """
        try:
            # Calculate RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = df["close"].ewm(span=12, adjust=False).mean()
            exp2 = df["close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = exp1 - exp2
            df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
            
            # Calculate ADX
            tr1 = df["high"] - df["low"]
            tr2 = abs(df["high"] - df["close"].shift())
            tr3 = abs(df["low"] - df["close"].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            plus_dm = df["high"].diff()
            minus_dm = df["low"].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            
            tr14 = tr.rolling(window=14).sum()
            plus_di14 = 100 * (plus_dm.rolling(window=14).sum() / tr14)
            minus_di14 = 100 * (minus_dm.rolling(window=14).sum() / tr14)
            
            dx = 100 * abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
            df["ADX"] = dx.rolling(window=14).mean()
            
            # Calculate ATR and ensure it's added to the DataFrame
            df["ATR"] = atr
            
            # Calculate Bollinger Bands
            df["BB_middle"] = df["close"].rolling(window=20).mean()
            df["BB_upper"] = df["BB_middle"] + 2 * df["close"].rolling(window=20).std()
            df["BB_lower"] = df["BB_middle"] - 2 * df["close"].rolling(window=20).std()
            df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return df
            
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
            self.logger.error(f"Error calculating stop loss: {str(e)}")
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
            self.logger.error(f"Error calculating take profit: {str(e)}")
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
            self.logger.error(f"Error calculating trailing stop: {str(e)}")
            return current_price
            
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
            self.logger.error(f"Error getting trend: {str(e)}")
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
            for timeframe in self.timeframes:
                # Get data for timeframe
                df = await self.get_historical_data(symbol, timeframe)
                if df is None or df.empty:
                    continue
                    
                # Calculate indicators
                df = await self.calculate_indicators(df)
                
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
                
                results[timeframe] = {
                    "trend": trend,
                    "strength": strength,
                    "rsi": rsi,
                    "macd_hist": macd - macd_signal
                }
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing multiple timeframes: {str(e)}")
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
            btc_df = await self.get_historical_data("BTC/USDT", timeframe)
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
            atr = btc_df.get("ATR", pd.Series([0.0])).iloc[-1]  # Safely get ATR with default
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
            self.logger.error(f"Error analyzing BTC volatility: {str(e)}")
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
            alt_df = await self.get_historical_data(symbol, "5m")
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
            atr = alt_df.get("ATR", pd.Series([0.0])).iloc[-1]  # Safely get ATR with default
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
            self.logger.error(f"Error analyzing altcoin correlation: {str(e)}")
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
            btc_df = await self.get_historical_data("BTC/USDT", "5m", window)
            alt_df = await self.get_historical_data(symbol, "5m", window)
            
            if btc_df is None or alt_df is None or btc_df.empty or alt_df.empty:
                return 0.0
                
            # Calculate returns
            btc_returns = btc_df["close"].pct_change().dropna()
            alt_returns = alt_df["close"].pct_change().dropna()
            
            # Calculate correlation
            correlation = btc_returns.corr(alt_returns)
            
            return correlation
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic correlation: {str(e)}")
            return 0.0
            
    async def calculate_signal_score(self, df: pd.DataFrame, timeframe_analysis: Dict, 
                                   btc_volatility: Dict, altcoin_correlation: Dict) -> float:
        """Calculate signal score.
        
        Args:
            df: Price data DataFrame
            timeframe_analysis: Multi-timeframe analysis
            btc_volatility: BTC volatility analysis
            altcoin_correlation: Altcoin correlation analysis
            
        Returns:
            float: Signal score
        """
        try:
            score = 0.0
            
            # 1. Technical Analysis Score (0-4 points)
            if "RSI" in df and "MACD" in df and "MACD_SIGNAL" in df:
                rsi = df["RSI"].iloc[-1]
                macd = df["MACD"].iloc[-1]
                macd_signal = df["MACD_SIGNAL"].iloc[-1]
                
                # RSI conditions
                if rsi < 30 or rsi > 70:
                    score += 2
                elif (rsi < 35 and macd > macd_signal) or (rsi > 65 and macd < macd_signal):
                    score += 1.5
                elif macd > macd_signal or macd < macd_signal:
                    score += 1
                    
            # 2. Market Conditions Score (0-3 points)
            if df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 1.5:
                score += 1.5
            if df["ATR"].iloc[-1] < df["ATR"].rolling(20).mean().iloc[-1] * 1.5:
                score += 1
            if df["ADX"].iloc[-1] > 25:
                score += 0.5
                
            # 3. Multi-timeframe Analysis Score (0-3 points)
            if self.check_trend_conflicts(timeframe_analysis):
                score += 2
            if sum(1 for data in timeframe_analysis.values() if data["strength"] > 0.02) >= len(timeframe_analysis) * 0.5:
                score += 1
                
            # 4. BTC Volatility Score (0-2.5 points)
            if btc_volatility["is_volatile"]:
                score += 1
            if btc_volatility["is_accelerating"]:
                score += 1
            if btc_volatility["strength"] > 0.02:
                score += 0.5
                
            # 5. Altcoin Correlation Score (0-2.5 points)
            if altcoin_correlation["is_reacting"]:
                score += 1
            if altcoin_correlation["correlation"] > 0.7:
                score += 1
            if altcoin_correlation["strength"] > 0.02:
                score += 0.5
                
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating signal score: {str(e)}")
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
            up_trends = sum(1 for trend in trends if trend == "UP")
            down_trends = sum(1 for trend in trends if trend == "DOWN")
            
            total_timeframes = len(trends)
            return up_trends >= total_timeframes * 0.67 or down_trends >= total_timeframes * 0.67
            
        except Exception as e:
            self.logger.error(f"Error checking trend conflicts: {str(e)}")
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
            self.logger.error(f"Error checking volume condition: {str(e)}")
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
            self.logger.error(f"Error checking volatility condition: {str(e)}")
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
            self.logger.error(f"Error checking ADX condition: {str(e)}")
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
            self.logger.error(f"Error checking Bollinger condition: {str(e)}")
            return False 