"""
Enhanced trading strategy implementation.
"""

import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np
import time
import asyncio
from datetime import datetime

from src.services.indicator_service import IndicatorService
from src.services.sentiment_service import SentimentService
from src.services.binance_service import BinanceService
from src.services.telegram_service import TelegramService
from src.services.notification_service import NotificationService
from src.utils.helpers import is_long_side, is_short_side

logger = logging.getLogger(__name__)

class EnhancedTradingStrategy:
    """Enhanced trading strategy with multiple indicators and risk management."""
    
    def __init__(self, config: Dict, binance_service: BinanceService, indicator_service: IndicatorService,
                 notification_service: NotificationService, telegram_service: TelegramService):
        """Initialize the trading strategy.
        
        Args:
            config: Configuration dictionary
            binance_service: Binance service instance
            indicator_service: Indicator service instance
            notification_service: Notification service instance
            telegram_service: Telegram service instance
        """
        self.config = config
        self.binance_service = binance_service
        self.indicator_service = indicator_service
        self.notification_service = notification_service
        self.telegram_service = telegram_service
        self._is_running = False
        self._monitoring_tasks = []
        self._last_dca_time = {}
        self._dca_history = {}
        self._last_trailing_stop_update = {}  # Track last trailing stop update time per symbol
        self._trailing_stop_debounce = 60  # Debounce time in seconds
        
        # Trading parameters
        self.min_volume_ratio = 1.2  # Tỷ lệ volume tối thiểu so với trung bình
        self.max_volatility_ratio = 2.0  # Tỷ lệ biến động tối đa so với trung bình
        self.min_adx = 20  # ADX tối thiểu cho xu hướng mạnh
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
        
        # DCA settings with dynamic levels
        self.dca_settings = {
            'base_levels': [
                {'drop': 0.02, 'multiplier': 1.0, 'max_uses': 2},  # 2% drop - same size
                {'drop': 0.04, 'multiplier': 1.5, 'max_uses': 1},  # 4% drop - 1.5x size
                {'drop': 0.06, 'multiplier': 2.0, 'max_uses': 1}   # 6% drop - 2x size
            ],
            'dynamic_adjustment': {
                'volatility_factor': 1.2,  # Increase levels in high volatility
                'trend_factor': 1.1,       # Increase levels in strong trend
                'volume_factor': 1.15      # Increase levels in high volume
            },
            'max_total_multiplier': 3.0,   # Maximum total position size multiplier
            'min_time_between_dca': 3600,  # Minimum time between DCA orders (1 hour)
            'max_dca_attempts': 3          # Maximum number of DCA attempts per position
        }
        
        # Trailing stop settings
        self.trailing_settings = [
            {'profit': 0.01, 'trail_percent': 0.005, 'atr_multiplier': 1.0},  # 1% profit - 0.5% trail
            {'profit': 0.02, 'trail_percent': 0.008, 'atr_multiplier': 1.2},  # 2% profit - 0.8% trail
            {'profit': 0.03, 'trail_percent': 0.01, 'atr_multiplier': 1.5}    # 3% profit - 1.0% trail
        ]
        
        # Trade tracking variables
        self._last_trade_time = {}
        self._last_trade_price = {}
        self._last_trade_side = {}
        self._last_trade_amount = {}
        self._last_trade_stop_loss = {}
        self._last_trade_take_profit = {}
        self._position_entry_time = {}  # Track entry time for each position
        self._order_history = {}  # Track order history per symbol
        self._min_order_interval = 120  # Minimum 5 minutes between orders
        self._max_orders_per_hour = 10  # Maximum 3 orders per hour
        self._max_risk_per_symbol = 0.3  # Maximum 30% risk per symbol
        
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
            self.sentiment_service = SentimentService(self.config)
            
            # Initialize services
            if not await self.sentiment_service.initialize():
                logger.error("Failed to initialize sentiment service")
                return False
                
            self._is_initialized = True
            logger.info("Strategy initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize strategy: {str(e)}")
            return False
            
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
        """Generate trading signals for a symbol."""
        try:
            # Get historical data
            df = await indicator_service.get_historical_data(symbol, timeframe='5m', limit=100)
            if df is None or df.empty:
                logger.warning(f"No data available for {symbol}")
                return None
                
            # Ensure data is valid
            if not self._validate_data(df):
                logger.warning(f"Invalid data for {symbol}")
                return None
                
            # Calculate indicators
            df = await indicator_service.calculate_indicators(symbol)
            if df is None or df.empty:
                logger.warning(f"Failed to calculate indicators for {symbol}")
                return None
                
            # Analyze multiple timeframes
            timeframe_analysis = await self.analyze_multiple_timeframes(symbol)
            if not timeframe_analysis:
                logger.warning(f"Failed to analyze timeframes for {symbol}")
                return None
                
            # Analyze BTC volatility
            btc_volatility = await self.analyze_btc_volatility()
            if not btc_volatility:
                logger.warning("Failed to analyze BTC volatility")
                return None
                
            # Analyze altcoin correlation
            altcoin_correlation = await self.analyze_altcoin_correlation(symbol, btc_volatility)
            if not altcoin_correlation:
                logger.warning(f"Failed to analyze correlation for {symbol}")
                return None
                
            # Analyze market sentiment
            sentiment = await self.analyze_market_sentiment(symbol)
            if not sentiment:
                logger.warning(f"Failed to analyze sentiment for {symbol}")
                return None
                
            # Calculate signal score
            signal_score = await self.calculate_signal_score(
                symbol=symbol,
                df=df,
                timeframe_analysis=timeframe_analysis,
                btc_volatility=btc_volatility,
                altcoin_correlation=altcoin_correlation,
                sentiment=sentiment
            )
            
            # Validate signal conditions
            # if not self.check_volume_condition(df):
            #     logger.debug(f"Volume condition not met for {symbol}")
            #     return None
                
            # if not self.check_volatility_condition(df):
            #     logger.debug(f"Volatility condition not met for {symbol}")
            #     return None
                
            # if not self.check_adx_condition(df):
            #     print(f"ADX condition not met for {symbol}")
            #     return None
                
            # Determine signal type
            logger.debug(f"Signal score: {signal_score}")
            if signal_score >= self.config['trading']['buy_threshold']:
                signal_type = "LONG"
            elif signal_score <= self.config['trading']['sell_threshold']:
                signal_type = "SHORT"
            else:
                logger.debug(f"Signal score {signal_score:.2f} below threshold for {symbol}")
                return None
                
            # Log signal details
            logger.info(f"Generated {signal_type} signal for {symbol} with score {signal_score:.2f}")
            logger.debug(f"Signal components for {symbol}:")
            logger.debug(f"- Trend: {timeframe_analysis.get('trend', 'NEUTRAL')}")
            logger.debug(f"- Volume: {'OK' if self.check_volume_condition(df) else 'LOW'}")
            logger.debug(f"- Volatility: {btc_volatility.get('volatility_level', 'LOW')}")
            logger.debug(f"- Correlation: {altcoin_correlation.get('correlation', 0):.2f}")
            logger.debug(f"- Sentiment: {sentiment.get('sentiment', 0):.2f}")
            
            return {
                'symbol': symbol,
                'position_type': signal_type,
                'score': signal_score,
                'price': df['close'].iloc[-1],
                'timestamp': datetime.now().isoformat(),
                'analysis': {
                    'trend': timeframe_analysis,
                    'volume': self.check_volume_condition(df),
                    'volatility': btc_volatility,
                    'correlation': altcoin_correlation,
                    'sentiment': sentiment
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {str(e)}")
            return None
            
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the input data before processing.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        try:
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error("Missing required columns")
                return False
                
            # Check for NaN values
            if df[required_columns].isnull().any().any():
                logger.error("Data contains NaN values")
                return False
                
            # Check for zero or negative values
            if (df[required_columns] <= 0).any().any():
                logger.error("Data contains zero or negative values")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return False

    async def analyze_market_sentiment(self, symbol: str) -> Dict:
        """Analyze market sentiment using multiple indicators.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict: Market sentiment analysis
        """
        try:
            # Get market data
            df = await self.indicator_service.calculate_indicators(symbol)
            if df is None or df.empty:
                return None
                
            # Calculate sentiment indicators
            rsi = df['RSI'].iloc[-1]
            mfi = df['MFI'].iloc[-1] if 'MFI' in df.columns else None
            obv = df['OBV'].iloc[-1] if 'OBV' in df.columns else None
            adx = df['ADX'].iloc[-1]
            
            # Analyze sentiment
            sentiment = {
                'rsi_sentiment': 'bullish' if rsi < 30 else 'bearish' if rsi > 70 else 'neutral',
                'mfi_sentiment': 'bullish' if mfi and mfi < 20 else 'bearish' if mfi and mfi > 80 else 'neutral',
                'obv_sentiment': 'bullish' if obv and obv > 0 else 'bearish' if obv and obv < 0 else 'neutral',
                'trend_strength': 'strong' if adx > 25 else 'weak',
                'overall_sentiment': 'bullish' if (
                    (rsi < 30 and (not mfi or mfi < 20)) or
                    (obv and obv > 0 and adx > 25)
                ) else 'bearish' if (
                    (rsi > 70 and (not mfi or mfi > 80)) or
                    (obv and obv < 0 and adx > 25)
                ) else 'neutral'
            }
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {str(e)}")
            return None
            
         
    async def _calculate_stop_loss(self, symbol: str, position_type: str, current_price: float, atr: float) -> float:
        """Calculate stop loss price based on ATR and market conditions."""
        try:
            # Get stop loss multiplier from config
            stop_loss_multiplier = float(self.config['risk_management']['stop_loss_atr_multiplier'])
            
            # Calculate base stop loss using ATR
            # logger.info(f"Calculating stop loss for {symbol} {position_type} {position_type.upper()}")
            if is_long_side(position_type):
                stop_loss = float(current_price) - (float(atr) * stop_loss_multiplier)
            else:
                stop_loss = float(current_price) + (float(atr) * stop_loss_multiplier)
            
            # Get market conditions
            market_conditions = await self._get_market_conditions(symbol)
            
            # Adjust stop loss based on volatility
            volatility = market_conditions.get('volatility', 'LOW')
            if volatility == 'HIGH':
                # Increase stop loss distance in high volatility
                if is_long_side(position_type):
                    stop_loss = float(current_price) - (float(atr) * stop_loss_multiplier * 1.5)
                else:
                    stop_loss = float(current_price) + (float(atr) * stop_loss_multiplier * 1.5)
            
            # Ensure minimum distance from current price
            min_distance = float(self.config['risk_management']['min_stop_distance'])
            if is_long_side(position_type):
                # For LONG positions, ensure stop loss is below current price
                stop_loss = min(stop_loss, float(current_price) * (1 - min_distance))
            else:
                # For SHORT positions, ensure stop loss is above current price
                stop_loss = max(stop_loss, float(current_price) * (1 + min_distance))
            
            # logger.info(f"Calculated stop loss for {symbol} {position_type.lower()}: {stop_loss} (current price: {current_price})")
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating stop loss for {symbol}: {str(e)}")
            return None

    async def _calculate_take_profit(self, symbol: str, position_type: str, current_price: float, stop_loss: float) -> float:
        """Calculate take profit price based on risk-reward ratio."""
        try:
            # Get risk-reward ratio from config
            risk_reward_ratio = self.config['risk_management']['take_profit_multiplier']
            
            # Calculate price difference between current price and stop loss
            price_diff = abs(current_price - stop_loss)
            
            # Calculate take profit based on risk-reward ratio
            # logger.info(f"Calculating take profit for {symbol} {position_type} {position_type.upper()}")
            if is_long_side(position_type):
                take_profit = current_price + (price_diff * risk_reward_ratio)
            else:
                take_profit = current_price - (price_diff * risk_reward_ratio)
            
            # Ensure minimum distance from current price
            min_distance = float(self.config['risk_management']['min_tp_distance'])
            if is_long_side(position_type):
                # For LONG positions, ensure take profit is above current price
                take_profit = max(take_profit, current_price * (1 + min_distance))
            else:
                # For SHORT positions, ensure take profit is below current price
                take_profit = min(take_profit, current_price * (1 - min_distance))
            
            # logger.info(f"Calculated take profit for {symbol} {position_type.lower()}: {take_profit} (current price: {current_price})")
            return take_profit
            
        except Exception as e:
            logger.error(f"Error calculating take profit for {symbol}: {str(e)}")
            return None

    async def close(self):
        """Close the strategy and clear cache."""
        try:
            self.clear_cache()
            self._is_closed = True
            logger.info("Strategy closed")
        except Exception as e:
            logger.error(f"Error closing strategy: {str(e)}")
            
    async def should_close_position(self, position: dict) -> bool:
        """Check if position should be closed based on various conditions.
        
        Args:
            position: Position details
            
        Returns:
            bool: True if position should be closed, False otherwise
        """
        try:
            # symbol = position['symbol']
            entry_price = float(position['entryPrice'])
            current_price = float(position['markPrice'])
            position_amt = float(position['info']['positionAmt'])
            
            # Calculate price change
            price_change = (current_price - entry_price) / entry_price
            if position_amt < 0:  # For short positions, invert the price change
                price_change = -price_change
                
            # Check stop loss
            stop_loss = float(position.get('stopLoss', 0))
            if stop_loss > 0:
                if (position_amt > 0 and current_price <= stop_loss) or \
                   (position_amt < 0 and current_price >= stop_loss):
                    # logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                    return True
                    
            # Check take profit
            take_profit = float(position.get('takeProfit', 0))
            if take_profit > 0:
                if (position_amt > 0 and current_price >= take_profit) or \
                   (position_amt < 0 and current_price <= take_profit):
                    # logger.info(f"Take profit triggered for {symbol} at {current_price}")
                    return True
                    
            # Check trend reversal
            # try:
            #     # Get current trend
            #     trend = await self._get_market_trend(symbol)
            #     if trend:
            #         # If trend has reversed from our position
            #         if (position_amt > 0 and trend == 'bearish') or \
            #            (position_amt < 0 and trend == 'bullish'):
            #             logger.info(f"Trend reversal detected for {symbol}: {trend}")
            #             return True
            # except Exception as e:
            #     logger.error(f"Error checking trend for {symbol}: {str(e)}")
                
            # Check market sentiment
            # try:
            #     sentiment = await self._get_market_sentiment(symbol)
            #     if sentiment:
            #         # If sentiment is strongly against our position
            #         if (position_amt > 0 and sentiment < -0.7) or \
            #            (position_amt < 0 and sentiment > 0.7):
            #             logger.info(f"Strong market sentiment against position for {symbol}: {sentiment}")
            #             return True
            # except Exception as e:
            #     logger.error(f"Error checking sentiment for {symbol}: {str(e)}")
                
            return False
            
        except Exception as e:
            logger.error(f"Error in should_close_position: {str(e)}")
            return False
            
    async def should_update_stops(self, position: Dict, current_price: float) -> bool:
        """Check if stop loss and take profit should be updated.
        
        Args:
            position: Position details
            current_price: Current market price
            
        Returns:
            bool: True if stops should be updated, False otherwise
        """
        try:
            # Check minimum update interval (5 minutes)
            position_key = f"{position['symbol']}_{position['side']}"
            current_time = time.time()
            cache_key = f"last_stop_update_{position_key}"
            # Get last update time from cache
            last_update = await self.binance_service._get_cached_data(cache_key)
            if last_update and current_time - last_update < 300:  # 5 minutes
                return False
                
            # Get historical data and indicators
            df = await self.indicator_service.calculate_indicators(position["symbol"])
            if df is None or df.empty:
                return False
                
            # Check if price moved significantly
            if is_long_side(position["side"]):
                price_change = (current_price - position["entryPrice"]) / position["entryPrice"]
                if price_change > 0.02:  # 2% move
                    # Update last update time
                    self.binance_service._set_cached_data(cache_key, current_time)
                    return True
            else:
                price_change = (position["entryPrice"] - current_price) / position["entryPrice"]
                if price_change > 0.02:  # 2% move
                    # Update last update time
                    self.binance_service._set_cached_data(cache_key, current_time)
                    return True
                    
            # # Check if trend is strengthening with confirmation
            # current_trend = self.get_trend(df)
            # if is_long_side(position["side"]):
            #     if current_trend == "UP":
            #         # Check if price is above EMA and volume is increasing
            #         if (current_price > df['EMA_FAST'].iloc[-1] and 
            #             df['volume'].iloc[-1] > df['volume'].iloc[-2] * 1.2):
            #             # Update last update time
            #             self.binance_service._set_cached_data(cache_key, current_time)
            #             return True
            # else:
            #     if current_trend == "DOWN":
            #         # Check if price is below EMA and volume is increasing
            #         if (current_price < df['EMA_FAST'].iloc[-1] and 
            #             df['volume'].iloc[-1] > df['volume'].iloc[-2] * 1.2):
            #             # Update last update time
            #             self.binance_service._set_cached_data(cache_key, current_time)
            #             return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking stop update: {str(e)}")
            return False
            
    async def calculate_new_stops(self, position: Dict, current_price: float) -> Dict[str, float]:
        """Calculate new stop loss and take profit levels.
        
        Args:
            position: Position details
            current_price: Current market price
            
        Returns:
            Dict[str, float]: Dictionary containing new stop loss and take profit levels
        """
        try:
            if 'info' not in position:
                logger.error("Invalid position structure")
                return {}
                
            # Get position details
            position_amt = float(position['info'].get('positionAmt', '0'))
            if position_amt == 0:
                return {}
                
            entry_price = float(position['info'].get('entryPrice', '0'))
            if not entry_price:
                return {}
                
            # Get market conditions
            market_conditions = await self._get_market_conditions(position['info']['symbol'])
            if not market_conditions:
                return {}
                
            # Calculate ATR-based stop distance
            atr = market_conditions.get('atr', 0)
            if not atr:
                return {}
                
            # Calculate stop distance based on volatility
            volatility = market_conditions.get('volatility', 'LOW')
            base_stop_distance = self.config['risk_management']['base_stop_distance']
            
            # Adjust volatility multiplier based on volatility level
            if volatility == 'HIGH':
                volatility_multiplier = 1.5
            elif volatility == 'MEDIUM':
                volatility_multiplier = 1.0
            else:  # LOW
                volatility_multiplier = 0.5
                
            stop_distance = base_stop_distance * volatility_multiplier
            
            # Adjust stop distance based on trend
            trend = market_conditions.get('trend', 'UP')
            # If trend is in our favor, we can use a tighter stop
            if (position_amt > 0 and trend == 'UP') or (position_amt < 0 and trend == 'DOWN'):
                trend_multiplier = 0.8  # Tighter stop in trend direction
            else:
                trend_multiplier = 1.2  # Wider stop against trend
                
            stop_distance *= trend_multiplier
            
            # Calculate new stop loss
            if position_amt > 0:  # Long position
                new_stop_loss = current_price - (stop_distance * current_price)
                new_take_profit = current_price + (stop_distance * current_price * self.config['risk_management']['take_profit_multiplier'])
            else:  # Short position
                new_stop_loss = current_price + (stop_distance * current_price)
                new_take_profit = current_price - (stop_distance * current_price * self.config['risk_management']['take_profit_multiplier'])
                
            return {
                'stop_loss': new_stop_loss,
                'take_profit': new_take_profit
            }
            
        except Exception as e:
            logger.error(f"Error calculating new stops: {str(e)}")
            return {}

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
        """Analyze altcoin correlation with BTC and adjust based on BTC volatility.
        
        Args:
            symbol: Trading pair symbol
            btc_volatility: BTC volatility information
            
        Returns:
            Dict: Correlation analysis results
        """
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
                # Adjust correlation based on BTC volatility
                if btc_volatility and 'volatility_level' in btc_volatility:
                    btc_vol = btc_volatility['volatility_level']
                    if btc_vol == 'HIGH':  # High BTC volatility
                        analysis['correlation'] *= 1.2  # Increase correlation during high volatility
                        analysis['volatility_adjusted'] = True
                    elif btc_vol == 'LOW':  # Low BTC volatility
                        analysis['correlation'] *= 0.8  # Decrease correlation during low volatility
                        analysis['volatility_adjusted'] = True
                    else:
                        analysis['volatility_adjusted'] = False
                
                self._cache[cache_key] = (analysis, current_time)
                self._last_update[cache_key] = current_time
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing altcoin correlation for {symbol}: {str(e)}")
            return None
            
    def clear_cache(self):
        """Clear the strategy cache."""
        self._cache.clear()
        self._last_update.clear()
        
    def calculate_trailing_stop(self, symbol: str, current_price: float, atr: float, position_type: str, 
                              market_conditions: Dict = None) -> float:
        """Calculate trailing stop price based on current market conditions."""
        try:
            # Get trailing stop configuration
            trailing_config = self.config['risk_management']['trailing_stop']
            dynamic_config = trailing_config['dynamic']

            # Calculate base ATR multiplier
            atr_multiplier = dynamic_config['atr_multiplier']

            # Adjust ATR multiplier based on market conditions
            if market_conditions:
                # Adjust for volatility
                volatility = market_conditions.get('volatility', 0)
                if volatility > 0.05:  # High volatility
                    atr_multiplier *= 1.5
                elif volatility < 0.02:  # Low volatility
                    atr_multiplier *= 0.8

                # Adjust for trend strength
                trend_strength = market_conditions.get('trend_strength', 0)
                if trend_strength > 0.7:  # Strong trend
                    atr_multiplier *= 1.2
                elif trend_strength < 0.3:  # Weak trend
                    atr_multiplier *= 0.9

            # Calculate final trailing stop distance
            trailing_distance = atr * atr_multiplier

            # Calculate trailing stop price
            if is_long_side(position_type):
                return current_price - trailing_distance
            else:
                return current_price + trailing_distance

        except Exception as e:
            logger.error(f"Error calculating trailing stop for {symbol}: {str(e)}")
            # Return a safe default value
            if is_long_side(position_type):
                return current_price * 0.99  # 1% below current price
            else:
                return current_price * 1.01  # 1% above current price

    async def calculate_signal_score(self, symbol: str, df: pd.DataFrame, timeframe_analysis: Dict, 
                                   btc_volatility: Dict, altcoin_correlation: Dict,
                                   sentiment: Dict) -> float:
        """Calculate signal score based on multiple factors."""
        try:
            # Get trend score (30% weight)
            trend_score = 0.3 * self._calculate_trend_score(timeframe_analysis)
            
            # Get volume score (20% weight)
            volume_score = 0.2 * self._calculate_volume_score(df)
            
            # Get volatility score (20% weight)
            volatility_score = 0.2 * self._calculate_volatility_score(btc_volatility)
            
            # Get correlation score (20% weight)
            correlation_score = 0.2 * self._calculate_correlation_score(altcoin_correlation)
            
            # Get sentiment score (10% weight)
            sentiment_score = 0.1 * self._calculate_sentiment_score(sentiment)
                
            # Calculate final score
            final_score = (
                trend_score +
                volume_score +
                volatility_score +
                correlation_score +
                sentiment_score
            )
            
            # Ensure score is between -1 and 1
            final_score = max(min(final_score, 1.0), -1.0)
            
            # Log score components
            logger.debug(f"{symbol} Signal score components: trend={trend_score:.2f}, "
                      f"volume={volume_score:.2f}, volatility={volatility_score:.2f}, "
                      f"correlation={correlation_score:.2f}, sentiment={sentiment_score:.2f}, "
                      f"final={final_score:.2f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating signal score: {str(e)}")
            return 0.0

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
            logger.debug(f"Volume ratio: {volume_ratio}")
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
        """Calculate BTC volatility metrics using both 1m and 5m timeframes."""
        try:
            # Get BTC data for both timeframes with caching
            btc_data_1m = await self.indicator_service.calculate_indicators("BTCUSDT", "1m")
            btc_data_5m = await self.indicator_service.calculate_indicators("BTCUSDT", "5m")
            
            if btc_data_1m is None or btc_data_5m is None or btc_data_1m.empty or btc_data_5m.empty:
                return None
                
            # Calculate volatility metrics for 1m
            atr_1m = btc_data_1m['ATR'].iloc[-1]
            roc_1m = btc_data_1m['ROC'].iloc[-1]
            current_price_1m = btc_data_1m['close'].iloc[-1]
            
            # Calculate volatility metrics for 5m
            atr_5m = btc_data_5m['ATR'].iloc[-1]
            roc_5m = btc_data_5m['ROC'].iloc[-1]
            current_price_5m = btc_data_5m['close'].iloc[-1]
            ema = btc_data_5m['EMA_FAST'].iloc[-1]
            
            # Calculate volatility scores
            volatility_score_1m = min(100, (atr_1m / current_price_1m) * 1000)
            volatility_score_5m = min(100, (atr_5m / current_price_5m) * 1000)
            
            # Combined volatility score with more weight on 5m
            volatility_score = (volatility_score_1m * 0.3) + (volatility_score_5m * 0.7)
            
            # More appropriate thresholds:
            # - Low volatility: < 7 (< 0.7% movement)
            # - Medium volatility: 7-8 (0.7-0.8% movement)
            # - High volatility: > 8 (> 0.8% movement)
            is_volatile = volatility_score > 8
            
            # Consider acceleration when price changes more than 0.2% in 1m or 0.3% in 5m
            is_accelerating = abs(roc_1m) > 0.2 or abs(roc_5m) > 0.3
            
            # Determine trend based on EMA and current price
            trend = "UP" if current_price_5m > ema else "DOWN"
            
            # Add volatility level for more granular decision making
            volatility_level = "LOW" if volatility_score < 7 else "MEDIUM" if volatility_score < 8 else "HIGH"
            
            return {
                'volatility_score': volatility_score,
                'volatility_score_1m': volatility_score_1m,
                'volatility_score_5m': volatility_score_5m,
                'volatility_level': volatility_level,
                'is_volatile': is_volatile,
                'is_accelerating': is_accelerating,
                'atr_1m': atr_1m,
                'atr_5m': atr_5m,
                'roc_1m': roc_1m,
                'roc_5m': roc_5m,
                'ema': ema,
                'current_price': current_price_5m,
                'trend': trend
            }
            
        except Exception as e:
            logger.error(f"Error calculating BTC volatility: {str(e)}")
            return None
            
    async def _calculate_altcoin_correlation(self, symbol: str) -> Dict:
        """Calculate correlation between BTC and altcoin price movements."""
        if not self.binance_service or not hasattr(self.binance_service, 'get_klines'):
            logging.error("Binance service not properly initialized")
            return self._get_default_correlation()

        try:
            btc_symbol = 'BTCUSDT'
            # Format symbol for Binance if needed
            if not symbol.endswith('USDT'):
                symbol = f"{symbol}USDT"

            logging.debug(f"Starting correlation calculation for {symbol} with BTC")
            
            # Fetch data with timeout
            async with asyncio.timeout(10):  # 10 second timeout
                btc_data, alt_data = await asyncio.gather(
                    self.binance_service.get_klines(btc_symbol, '5m', limit=100),
                    self.binance_service.get_klines(symbol, '5m', limit=100)
                )

            if not btc_data or not alt_data:
                logging.error(f"Failed to fetch data for correlation calculation: BTC data exists: {bool(btc_data)}, Alt data exists: {bool(alt_data)}")
                return self._get_default_correlation()

            # Convert to DataFrames with numeric close prices
            try:
                btc_df = pd.DataFrame(btc_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                alt_df = pd.DataFrame(alt_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Convert timestamps
                btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
                alt_df['timestamp'] = pd.to_datetime(alt_df['timestamp'], unit='ms')
                
                # Set index and ensure close prices are numeric
                btc_df.set_index('timestamp', inplace=True)
                alt_df.set_index('timestamp', inplace=True)
                
                btc_df['close'] = pd.to_numeric(btc_df['close'], errors='coerce')
                alt_df['close'] = pd.to_numeric(alt_df['close'], errors='coerce')
                
                # Remove any rows with zero or invalid prices
                btc_df = btc_df[btc_df['close'] > 0]
                alt_df = alt_df[alt_df['close'] > 0]
                
                if len(btc_df) < 10 or len(alt_df) < 10:
                    logging.warning("Insufficient price data points")
                    return self._get_default_correlation()
                
                # Calculate returns with explicit handling of edge cases
                btc_returns = btc_df['close'].pct_change()
                alt_returns = alt_df['close'].pct_change()
                
                # Handle inf/nan values
                btc_returns = btc_returns.replace([np.inf, -np.inf], np.nan)
                alt_returns = alt_returns.replace([np.inf, -np.inf], np.nan)
                
                # Calculate trend alignment
                btc_trend = 'UP' if btc_df['close'].iloc[-1] > btc_df['close'].iloc[-20] else 'DOWN'
                alt_trend = 'UP' if alt_df['close'].iloc[-1] > alt_df['close'].iloc[-20] else 'DOWN'
                trend_alignment = 1.0 if btc_trend == alt_trend else -1.0
                
                # Calculate correlation with different lags
                max_lag = 3  # Maximum lag to consider
                correlations = []
                for lag in range(max_lag + 1):
                    # Shift altcoin returns by lag
                    alt_shifted = alt_returns.shift(lag)
                    
                    # Create clean DataFrame for correlation
                    corr_df = pd.DataFrame({
                        'btc': btc_returns,
                        'alt': alt_shifted
                    })
                    
                    # Drop any NaN values
                    corr_df = corr_df.dropna()
                    
                    if len(corr_df) < 10:
                        continue
                    
                    # Calculate standard deviations
                    btc_std = corr_df['btc'].std()
                    alt_std = corr_df['alt'].std()
                    
                    # Check for very small or zero standard deviations
                    min_std = 1e-8
                    if btc_std < min_std or alt_std < min_std:
                        continue
                    
                    # Calculate correlation manually
                    btc_norm = (corr_df['btc'] - corr_df['btc'].mean()) / btc_std
                    alt_norm = (corr_df['alt'] - corr_df['alt'].mean()) / alt_std
                    correlation = (btc_norm * alt_norm).mean()
                    
                    if np.isfinite(correlation):
                        correlations.append((lag, correlation))
                
                if not correlations:
                    logging.warning("No valid correlations found")
                    return self._get_default_correlation()
                
                # Find best correlation and its lag
                best_lag, best_correlation = max(correlations, key=lambda x: abs(x[1]))
                
                # Calculate reaction metrics
                reaction_strength = abs(best_correlation)
                is_strongly_correlated = reaction_strength > 0.7
                is_reacting = reaction_strength > 0.5
                
                # Adjust correlation based on trend alignment
                adjusted_correlation = best_correlation * trend_alignment
                
                result = {
                    'correlation': float(adjusted_correlation),
                    'returns_correlation': float(best_correlation),
                    'reaction_strength': float(reaction_strength),
                    'is_strongly_correlated': is_strongly_correlated,
                    'is_reacting': is_reacting,
                    'reaction': 'STRONG' if is_strongly_correlated else 'MODERATE' if is_reacting else 'WEAK',
                    'data_points': len(corr_df),
                    'btc_std': float(btc_std),
                    'alt_std': float(alt_std),
                    'lag': best_lag,
                    'trend_alignment': float(trend_alignment),
                    'btc_trend': btc_trend,
                    'alt_trend': alt_trend
                }
                
                logging.debug(f"Correlation calculation complete for {symbol}:")
                logging.debug(f"Data points: {result['data_points']}")
                logging.debug(f"BTC std: {result['btc_std']:.6f}")
                logging.debug(f"ALT std: {result['alt_std']:.6f}")
                logging.debug(f"Correlation: {result['correlation']:.4f}")
                logging.debug(f"Reaction: {result['reaction']}")
                logging.debug(f"Best lag: {result['lag']}")
                logging.debug(f"Trend alignment: {result['trend_alignment']}")
                
                return result
                
            except Exception as e:
                logging.error(f"Error in data processing: {str(e)}")
                return self._get_default_correlation()

        except asyncio.TimeoutError:
            logging.error(f"Timeout while calculating correlation for {symbol}")
            return self._get_default_correlation()
        except Exception as e:
            logging.error(f"Error calculating correlation for {symbol}: {str(e)}")
            return self._get_default_correlation()
            
    def _get_default_correlation(self) -> Dict:
        """Return default correlation values when calculation fails."""
        return {
            'correlation': 0.0,
            'returns_correlation': 0.0,
            'reaction_strength': 0.0,
            'is_strongly_correlated': False,
            'is_reacting': False,
            'reaction': 'WEAK',
            'data_points': 0
        }

    async def _handle_dca(self, symbol: str, position: Dict) -> Optional[Dict]:
        """Handle DCA for a position.
        
        Args:
            symbol: Trading pair symbol
            position: Position details
            
        Returns:
            Optional[Dict]: DCA details if successful, None otherwise
        """
        try:
            # Get current price
            current_price = await self.binance_service.get_current_price(symbol)
            if not current_price:
                logger.error(f"Failed to get current price for {symbol}")
                return None
                
            # Validate position details
            if not position or float(position.get('contracts', 0)) <= 0:
                logger.error(f"Invalid position for {symbol}")
                return None
                
            # Calculate price drop percentage
            entry_price = float(position.get('entryPrice', 0))
            if entry_price <= 0:
                logger.error(f"Invalid entry price for {symbol}")
                return None
                
            position_type = position.get('side', 'LONG')
            # Calculate price drop based on position type
            if is_long_side(position_type):
                # For LONG: price drop = (entry_price - current_price) / entry_price
                price_drop = (entry_price - current_price) / entry_price * 100
            else:
                # For SHORT: price drop = (current_price - entry_price) / entry_price
                price_drop = (current_price - entry_price) / entry_price * 100
                
            # Get market conditions
            market_conditions = await self._get_market_conditions(symbol)
            if not market_conditions:
                logger.error(f"Failed to get market conditions for {symbol}")
                return None
                
            # Check if DCA is favorable
            if not self._is_dca_favorable(price_drop, market_conditions, position_type):
                return None
                
            # Calculate DCA size
            current_size = float(position.get('contracts', 0))
            dca_size = await self._calculate_dca_size(current_size, price_drop, position_type)
            if not dca_size or dca_size <= 0:
                logger.error(f"Invalid DCA size calculated for {symbol}: {dca_size}")
                return None
                
            # Place DCA order
            order_params = {
                'symbol': symbol,
                'side': 'BUY' if is_long_side(position_type) else 'SELL',
                'type': 'MARKET',
                'amount': dca_size,
                'reduceOnly': False
            }
            
            # Place order with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    order = await self.binance_service.place_order(order_params)
                    if order:
                        # Calculate new stop loss and take profit
                        new_stop_loss = await self._calculate_stop_loss(symbol, position_type, current_price, 
                                                                      await self.indicator_service.calculate_atr(symbol))
                        new_take_profit = await self._calculate_take_profit(symbol, position_type, current_price, new_stop_loss)
                        
                        # Update orders
                        await self._update_stop_loss(symbol, new_stop_loss, position_type)
                        await self._update_take_profit(symbol, new_take_profit, position_type)
                        
                        # Update DCA information
                        if is_long_side(position_type):
                            market_conditions['long_dca_attempts'] += 1
                            market_conditions['long_active_dca_positions'].append({
                                'entry_price': current_price,
                                'size': dca_size
                            })
                        else:
                            market_conditions['short_dca_attempts'] += 1
                            market_conditions['short_active_dca_positions'].append({
                                'entry_price': current_price,
                                'size': dca_size
                            })
                            
                        # Send notification
                        await self.telegram_service.send_dca_notification({
                            'symbol': symbol,
                            'dca_amount': dca_size,
                            'new_entry_price': current_price,
                            'price_drop': price_drop,
                            'order_id': order.get('id', 'N/A'),
                            'position_type': position_type
                        })
                        
                        return {
                            'order_id': order.get('id'),
                            'dca_amount': dca_size,
                            'new_entry_price': current_price,
                            'price_drop': price_drop,
                            'position_type': position_type
                        }
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"DCA order attempt {attempt + 1} failed for {symbol}: {str(e)}")
                        await asyncio.sleep(1)
                    else:
                        logger.error(f"Failed to place DCA order for {symbol}: {str(e)}")
                        return None
                        
            return None
            
        except Exception as e:
            logger.error(f"Error handling DCA for {symbol}: {str(e)}")
            return None

    def _is_dca_favorable(self, price_drop: float, market_conditions: Dict, position_type: str) -> bool:
        """Check if DCA is favorable based on multiple conditions.
        
        Args:
            price_drop: Percentage price drop from entry
            market_conditions: Current market conditions
            position_type: Type of position (LONG/SHORT)
            
        Returns:
            bool: True if DCA is favorable, False otherwise
        """
        try:
            if not market_conditions:
                logger.error("No market conditions provided")
                return False
                
            if position_type.upper() not in ['LONG', 'SHORT']:
                logger.error(f"Invalid position type: {position_type}")
                return False
                
            # Get DCA configuration
            dca_config = self.config.get('risk_management', {}).get('dca', {})
            if not dca_config:
                logger.error("No DCA configuration found")
                return False
                
            # Convert all values to float for comparison
            price_drop = float(price_drop)
            min_price_drop = float(dca_config['price_drop_thresholds'][0])
            volume_threshold = float(dca_config['volume_threshold'])
            
            # Check price drop threshold
            if price_drop < min_price_drop:
                return False
                
            # Check volume condition
            volume_ratio = float(market_conditions.get('volume_ratio', 1.0))
            if volume_ratio < volume_threshold:
                return False
                
            # Check volatility condition
            volatility = market_conditions.get('volatility', 'LOW')
            if volatility == 'HIGH':
                return False
                
            # Check trend condition based on position type
            trend = market_conditions.get('trend', 'NEUTRAL')
            if is_long_side(position_type) and trend == 'DOWN':
                return False
            if is_short_side(position_type) and trend == 'UP':
                return False
                
            # Check ATR condition
            atr = float(market_conditions.get('atr', 0))
            if atr <= 0:
                return False
                
            # Check DCA attempts for specific position type
            dca_attempts = market_conditions.get(f'{position_type.lower()}_dca_attempts', 0)
            max_dca_attempts = int(dca_config.get('max_attempts', 3))
            if dca_attempts >= max_dca_attempts:
                return False
                
            # Check active DCA positions for specific position type
            active_positions = len(market_conditions.get(f'{position_type.lower()}_active_dca_positions', []))
            max_active_positions = int(dca_config.get('max_active_positions', 2))
            if active_positions >= max_active_positions:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking DCA favorability for {position_type}: {str(e)}")
            return False

    async def _calculate_dca_size(self, current_size: float, price_drop: float, position_type: str) -> Optional[float]:
        """Calculate DCA size based on current position size and price movement.
        
        Args:
            current_size: Current position size
            price_drop: Price movement percentage (positive for LONG, negative for SHORT)
            position_type: Position type (LONG/SHORT)
            
        Returns:
            Optional[float]: DCA size or None if invalid
        """
        try:
            # Validate inputs
            if current_size <= 0:
                logger.error(f"Invalid current size for DCA: {current_size}")
                return None
                
            if position_type.upper() not in ["LONG", "SHORT"]:
                logger.error(f"Invalid position type for DCA: {position_type}")
                return None
                
            # Get DCA configuration
            dca_config = self.config['trading']['dca']
            max_dca_size = dca_config['max_dca_size']
            min_dca_size = dca_config['min_dca_size']
            max_dca_percentage = dca_config['max_dca_percentage']
            
            # Calculate base DCA size based on price movement
            # For LONG: price_drop is positive (price decreased)
            # For SHORT: price_drop is negative (price increased)
            price_movement = abs(price_drop)  # Use absolute value for calculation
            base_dca_size = current_size * (price_movement / 100.0) * 0.5
            
            # Ensure DCA size is within limits
            dca_size = min(
                max(base_dca_size, min_dca_size),  # Minimum DCA size
                max_dca_size,  # Maximum absolute DCA size
                current_size * (max_dca_percentage / 100.0)  # Maximum percentage of current size
            )
            
            # Final validation
            if dca_size <= 0:
                logger.error(f"Calculated invalid DCA size: {dca_size}")
                return None
                
            logger.debug(f"DCA size calculation: current_size={current_size}, "
                        f"price_drop={price_drop}, position_type={position_type}, "
                        f"base_dca={base_dca_size}, final_dca={dca_size}")
                        
            return dca_size
            
        except Exception as e:
            logger.error(f"Error calculating DCA size: {str(e)}")
            return None

    async def _get_market_conditions(self, symbol: str) -> Dict:
        """Get current market conditions for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict: Market conditions including:
                - df: DataFrame with price data
                - trend: Current trend
                - volatility: Market volatility
                - volume: Volume status
                - atr: Average True Range
                - volume_ratio: Volume ratio
                - long_dca_attempts: Number of DCA attempts for LONG position
                - short_dca_attempts: Number of DCA attempts for SHORT position
                - long_active_dca_positions: List of active DCA positions for LONG
                - short_active_dca_positions: List of active DCA positions for SHORT
        """
        try:
            # Get historical data
            df = await self.indicator_service.get_historical_data(symbol)
            if df is None or df.empty:
                logger.error(f"Failed to get historical data for {symbol}")
                return None
                
            # Get current trend
            trend = self.get_trend(df)
            
            # Calculate volatility
            volatility = await self.analyze_btc_volatility()
            
            # Get volume status
            volume_status = self.check_volume_condition(df)
            
            # Calculate ATR
            atr = await self.indicator_service.calculate_atr(symbol)
            
            # Calculate volume ratio
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(window=20).mean().iloc[-1]
            
            # Initialize DCA information for both sides
            long_dca_attempts = 0
            short_dca_attempts = 0
            long_active_dca_positions = []
            short_active_dca_positions = []
            
            # Get current position for both LONG and SHORT sides
            long_position = await self.binance_service.get_position(symbol, position_side='LONG')
            short_position = await self.binance_service.get_position(symbol, position_side='SHORT')
            
            # Check LONG position
            if long_position and float(long_position.get('contracts', 0)) > 0:
                long_dca_attempts = int(long_position.get('info', {}).get('dca_attempts', 0))
                long_active_dca_positions = long_position.get('info', {}).get('active_dca_positions', [])
                
            # Check SHORT position
            if short_position and float(short_position.get('contracts', 0)) > 0:
                short_dca_attempts = int(short_position.get('info', {}).get('dca_attempts', 0))
                short_active_dca_positions = short_position.get('info', {}).get('active_dca_positions', [])
            
            return {
                'df': df,
                'trend': trend,
                'volatility': volatility,
                'volume': volume_status,
                'atr': atr,
                'volume_ratio': volume_ratio,
                'long_dca_attempts': long_dca_attempts,
                'short_dca_attempts': short_dca_attempts,
                'long_active_dca_positions': long_active_dca_positions,
                'short_active_dca_positions': short_active_dca_positions
            }
            
        except Exception as e:
            logger.error(f"Error getting market conditions for {symbol}: {str(e)}")
            return None

    async def _update_trailing_stop(self, symbol: str, position_type: str) -> None:
        """Update trailing stop for a position with advanced features.
        
        Args:
            symbol: Trading pair symbol
            position_type: Position type (BUY/SELL)
        """
        try:
            # Check if we've updated recently
            if symbol in self._last_update:
                last_update = self._last_update[symbol]
                if time.time() - last_update < self.config['risk_management']['trailing_stop']['update_interval']:
                    return
                    
            position_side = "LONG" if is_long_side(position_type) else "SHORT"
            # Get position details
            position = await self.binance_service.get_position(symbol, position_side)
            if not position:
                return

            # Get current market data
            current_price = float(position['markPrice'])
            entry_price = float(position['entryPrice'])
            unrealized_pnl = float(position.get('unrealizedPnl', 0))
            position_size = float(position.get('info', {}).get('positionAmt', 0))
            
            # Only proceed if we have unrealized profit
            if unrealized_pnl <= 0:
                return
            
            # Calculate position age
            position_age = time.time() - self._position_entry_time.get(symbol, time.time())
            
            # Get market analysis
            market_conditions = await self._analyze_market_conditions(symbol)
            
            # Calculate new stop loss
            atr = await self.indicator_service.calculate_atr(symbol)
            new_stop_loss = self.calculate_trailing_stop(
                symbol, current_price, atr, position_type, market_conditions
            )
            
            # Get current stop loss
            # Check for existing SL/TP orders
            open_position_side = "SELL" if is_long_side(position_type) else "BUY"
            
            existing_orders = await self.binance_service.get_open_orders(symbol)
            if existing_orders:
                existing_sl = await self.binance_service.get_existing_order(symbol, 'STOP_MARKET', open_position_side)
                if not existing_sl:
                    current_stop_loss = 0
                else:
                    current_stop_loss = float(existing_sl.get('stopPrice', 0))
            else:
                current_stop_loss = 0

            # Check if we should move to break-even
            if self._should_move_to_break_even(
                current_price, unrealized_pnl, position_age, position_size
            ):
                new_stop_loss = entry_price
                # logger.info(f"Moving to break-even for {symbol} at {new_stop_loss}")
                
            # Check if we should take partial profit
            if self._should_take_partial_profit(
                current_price, unrealized_pnl, position_age, position_size
            ):
                await self._take_partial_profit(symbol, position_size)
                # logger.info(f"Taking partial profit for {symbol}")
                
            # Check for emergency stop
            if self._should_emergency_stop(market_conditions):
                new_stop_loss = self._calculate_emergency_stop(
                    current_price, position_type
                )
                # logger.warning(f"Emergency stop triggered for {symbol} at {new_stop_loss}")
                
            # Only update if new stop is more favorable and we have unrealized profit
            if is_long_side(position_type):
                # For LONG positions, only move stop loss up
                if new_stop_loss > current_stop_loss and new_stop_loss < current_price:
                    await self._update_stop_loss(symbol, new_stop_loss, position_type)
                    # logger.info(f"Updated trailing stop for {symbol} LONG to {new_stop_loss}")
            else:
                # For SHORT positions, only move stop loss down
                if new_stop_loss < current_stop_loss and new_stop_loss > current_price:
                    await self._update_stop_loss(symbol, new_stop_loss, position_type)
                    # logger.info(f"Updated trailing stop for {symbol} SHORT to {new_stop_loss}")
                
            # Update last update time
            self._last_update[symbol] = time.time()

        except Exception as e:
            logger.error(f"Error updating trailing stop for {symbol}: {str(e)}")

    def _should_move_to_break_even(self, current_price: float, unrealized_pnl: float, position_age: float,
                                 position_size: float) -> bool:
        """Check if we should move stop loss to break-even.
        
        Args:
            current_price: Current market price
            entry_price: Entry price
            unrealized_pnl: Unrealized profit/loss
            position_age: Time since position was opened
            position_size: Current position size
            
        Returns:
            bool: True if should move to break-even
        """
        min_profit_ratio = self.config['risk_management']['trailing_stop']['break_even']['min_profit']
        min_time = self.config['risk_management']['trailing_stop']['break_even']['min_time']
        
        profit_ratio = abs(unrealized_pnl / (current_price * abs(position_size)))
        return profit_ratio >= min_profit_ratio and position_age >= min_time
        
    def _should_take_partial_profit(self, current_price: float, unrealized_pnl: float, position_age: float,
                                  position_size: float) -> bool:
        """Check if we should take partial profit.
        
        Args:
            current_price: Current market price
            entry_price: Entry price
            unrealized_pnl: Unrealized profit/loss
            position_age: Time since position was opened
            position_size: Current position size
            
        Returns:
            bool: True if should take partial profit
        """
        min_profit_ratio = self.config['risk_management']['trailing_stop']['partial_profit']['min_profit']
        min_time = self.config['risk_management']['trailing_stop']['partial_profit']['min_time']
        
        profit_ratio = abs(unrealized_pnl / (current_price * abs(position_size)))
        return profit_ratio >= min_profit_ratio and position_age >= min_time
        
    async def _take_partial_profit(self, symbol: str, position_size: float) -> None:
        """Take partial profit by closing a portion of the position.
        
        Args:
            symbol: Trading pair symbol
            position_size: Current position size
        """
        try:
            # Calculate size to close
            close_ratio = self.config['risk_management']['trailing_stop']['partial_profit']['close_ratio']
            close_size = abs(position_size) * close_ratio
            
            # Create order parameters
            order_params = {
                'symbol': symbol,
                'side': "SELL" if position_size > 0 else "BUY",
                'type': "MARKET",
                'amount': close_size
            }
            
            # Place market order to close portion
            await self.binance_service.place_order(order_params)
            
        except Exception as e:
            logger.error(f"Error taking partial profit for {symbol}: {str(e)}")
            
    def _should_emergency_stop(self, market_conditions: Dict) -> bool:
        """Check if emergency stop should be triggered.
        
        Args:
            market_conditions: Dictionary containing market analysis data
            
        Returns:
            bool: True if emergency stop should be triggered
        """
        volatility = market_conditions.get('volatility', 0)
        volume_ratio = market_conditions.get('volume_ratio', 1.0)
        
        return (volatility > self.config['risk_management']['emergency_stop']['volatility_threshold'] or
                volume_ratio > self.config['risk_management']['emergency_stop']['volume_threshold'])
                
    def _calculate_emergency_stop(self, current_price: float, position_type: str) -> float:
        """Calculate emergency stop level based on current market price and position type."""
        emergency_distance = self.config['risk_management']['emergency_stop']['distance']
        
        if is_long_side(position_type):
            return current_price * (1 - emergency_distance)
        else:
            return current_price * (1 + emergency_distance)

    async def _update_stop_loss(self, symbol: str, new_stop_loss: float,
                              position_type: str) -> None:
        """
        Update stop loss for a position.
        
        Args:
            symbol: Trading pair symbol
            new_stop_loss: New stop loss price
            position_type: Position type (BUY/SELL/LONG/SHORT)
        """
        try:
            # Convert position_type to position_side for HEDGE mode
            position_side = "LONG" if is_long_side(position_type) else "SHORT"
            
            # Get current position
            position = await self.binance_service.get_position(symbol, position_side)
            if not position or float(position.get('contracts', 0)) == 0:
                logger.warning(f"No active {position_side} position found for {symbol}")
                return
                
            existing_orders = await self.binance_service.get_open_orders(symbol)
            open_position_side = "SELL" if is_long_side(position_type) else "BUY"
            if existing_orders:
                existing_sl = await self.binance_service.get_existing_order(symbol, 'STOP_MARKET', open_position_side)
                if not existing_sl:
                    current_stop_loss = 0
                else:
                    current_stop_loss = float(existing_sl.get('stopPrice', 0))
            else:
                current_stop_loss = 0
                
            if not (is_long_side(position_type) and new_stop_loss > current_stop_loss * 1.01) and \
                not (is_short_side(position_type) and new_stop_loss < current_stop_loss * 0.99):
                return
            
            # Update stop loss using binance_service
            success = await self.binance_service._update_stop_loss(
                symbol=symbol,
                position=position,
                new_stop_loss=new_stop_loss
            )
            
            if success:
                logger.info(f"Updated stop loss for {symbol} {position_side} from {current_stop_loss} to {new_stop_loss}")
                await self.telegram_service.send_stop_loss_notification(
                    symbol=symbol,
                    position_side=position_side,
                    entry_price=float(position.get('entryPrice', 0)),
                    stop_price=new_stop_loss,
                    pnl_percent=float(position.get('unrealizedPnl', 0))
                )
            else:
                logger.error(f"Failed to update stop loss for {symbol} {position_side}")
                
        except Exception as e:
            logger.error(f"Error updating stop loss for {symbol}: {str(e)}")

    async def _update_take_profit(self, symbol: str, new_take_profit: float,
                               position_type: str) -> None:
        """
        Update take profit for a position.
        
        Args:
            symbol: Trading pair symbol
            new_take_profit: New take profit price
            position_type: Position type (BUY/SELL/LONG/SHORT)
        """
        try:
            # Convert position_type to position_side for HEDGE mode
            position_side = "LONG" if is_long_side(position_type) else "SHORT"
            
            # Get current position
            position = await self.binance_service.get_position(symbol, position_side)
            if not position or float(position.get('contracts', 0)) == 0:
                logger.warning(f"No active {position_side} position found for {symbol}")
                return
                
            existing_orders = await self.binance_service.get_open_orders(symbol)
            open_position_side = "SELL" if is_long_side(position_type) else "BUY"
            if existing_orders:
                existing_tp = await self.binance_service.get_existing_order(symbol, 'TAKE_PROFIT_MARKET', open_position_side)
                if not existing_tp:
                    current_take_profit = 0
                else:
                    current_take_profit = float(existing_tp.get('stopPrice', 0))
            else:
                current_take_profit = 0
                
            if not (is_long_side(position_type) and new_take_profit > current_take_profit * 1.01) and \
                not (is_short_side(position_type) and new_take_profit < current_take_profit * 0.99):
                return

            # Update take profit using binance_service
            success = await self.binance_service._update_take_profit(
                symbol=symbol,
                position=position,
                new_take_profit=new_take_profit
            )
            
            if success:
                logger.info(f"Updated take profit for {symbol} {position_side} from {current_take_profit} to {new_take_profit}")
                await self.telegram_service.send_take_profit_notification(
                    symbol=symbol,
                    position_side=position_side,
                    entry_price=float(position.get('entryPrice', 0)),
                    tp_price=new_take_profit,
                    pnl_percent=float(position.get('unrealizedPnl', 0))
                )
            else:
                logger.error(f"Failed to update take profit for {symbol} {position_side}")
                
        except Exception as e:
            logger.error(f"Error updating take profit for {symbol}: {str(e)}")

    async def _analyze_market_conditions(self, symbol: str) -> Dict:
        """Analyze current market conditions.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict: Market analysis data
        """
        try:
            # Get recent candles
            candles = await self.binance_service.get_klines(symbol, timeframe="1h", limit=24)
            
            # Calculate volatility
            high_prices = [float(c[2]) for c in candles]
            low_prices = [float(c[3]) for c in candles]
            volatility = (max(high_prices) - min(low_prices)) / min(low_prices)
            
            # Calculate volume ratio
            current_volume = float(candles[-1][5])
            avg_volume = sum(float(c[5]) for c in candles[:-1]) / (len(candles) - 1)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate trend strength using ADX
            df = await self.indicator_service.get_historical_data(symbol, timeframe="1h", limit=24)
            if df is not None and not df.empty:
                df = self.indicator_service._calculate_adx(df)
                adx = df['ADX'].iloc[-1] if 'ADX' in df.columns else 0
            else:
                adx = 0
            trend_strength = adx / 100.0
            
            return {
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'trend_strength': trend_strength
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions for {symbol}: {str(e)}")
            return {}

    async def process_trading_signals(self, signals: Dict) -> None:
        """Process trading signals and execute trades."""
        try:
            symbol = signals.get('symbol')
            if not symbol:
                logger.error("No symbol in signals")
                return

            # Get current position
            position_side = "LONG" if is_long_side(signals['position_type']) else "SHORT"
            position = await self.binance_service.get_position(symbol, position_side)
            
            if position and float(position.get('info').get('positionAmt', 0)) != 0:
                # We have an existing position, manage it
                await self._manage_existing_position(symbol, position)
            else:
                # No existing position, check if we should open a new one
                if not self.telegram_service.is_trading_paused():
                    await self._execute_trade(signals)
                else:
                    logger.info(f"Trading paused - Skipping new trade for {symbol}")
                    
        except Exception as e:
            logger.error(f"Error processing trading signals: {str(e)}")
            
    async def _execute_trade(self, signal: Dict) -> None:
        """Execute a trade based on the signal."""
        try:
            symbol = signal.get('symbol')
            if not symbol:
                logger.error("No symbol in signal")
                return

            # Get current price
            current_price = await self.binance_service.get_current_price(symbol)
            if not current_price:
                logger.error(f"Failed to get current price for {symbol}")
                return

            # Calculate position size
            position_size = await self._calculate_position_size(
                symbol=symbol,
                risk_per_trade=float(self.config['trading']['risk_per_trade']),
                current_price=current_price
            )
            if not position_size:
                logger.error(f"Failed to calculate position size for {symbol}")
                return

            # Get ATR for stop loss calculation
            atr = await self.indicator_service.calculate_atr(symbol)
            if not atr:
                logger.error(f"Failed to get ATR for {symbol}")
                return

            # Calculate stop loss and take profit
            stop_loss = await self._calculate_stop_loss(
                symbol=symbol,
                position_type=signal['position_type'],
                current_price=current_price,
                atr=atr
            )
            if not stop_loss:
                logger.error(f"Failed to calculate stop loss for {symbol}")
                return

            take_profit = await self._calculate_take_profit(
                symbol=symbol,
                position_type=signal['position_type'],
                current_price=current_price,
                stop_loss=stop_loss
            )
            if not take_profit:
                logger.error(f"Failed to calculate take profit for {symbol}")
                return

            # Get current position
            position_side = "LONG" if is_long_side(signal['position_type']) else "SHORT"
            current_position = await self.binance_service.get_position(symbol, position_side)
            if current_position:
                # Calculate new total position size
                current_size = float(current_position.get('info').get('positionAmt', 0))
                new_total_size = current_size + position_size if is_long_side(signal['position_type']) else current_size - position_size
                
                # Calculate new SL/TP based on average entry price
                entry_price = float(current_position.get('entryPrice', 0))
                new_entry_price = (entry_price * abs(current_size) + current_price * position_size) / abs(new_total_size)
                
                # Recalculate SL/TP with new entry price
                stop_loss = await self._calculate_stop_loss(
                    symbol=symbol,
                    position_type=signal['position_type'],
                    current_price=new_entry_price,
                    atr=atr
                )
                take_profit = await self._calculate_take_profit(
                    symbol=symbol,
                    position_type=signal['position_type'],
                    current_price=new_entry_price,
                    stop_loss=stop_loss
                )

            # Place the order
            order_params = {
                'symbol': symbol,
                'side': 'buy' if is_long_side(signal['position_type']) else 'sell',
                'type': 'market',
                'amount': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }

            order = await self.binance_service.place_order(order_params)
            if not order:
                logger.error(f"Failed to place order for {symbol}")
                return

            # Send notification
            order['stop_loss'] = stop_loss
            order['take_profit'] = take_profit
            await self.telegram_service.send_order_notification(order)

            # Update tracking variables
            self._last_trade_time[symbol] = time.time()
            self._last_trade_price[symbol] = current_price
            self._last_trade_side[symbol] = signal['position_type']
            self._last_trade_amount[symbol] = position_size
            self._last_trade_stop_loss[symbol] = stop_loss
            self._last_trade_take_profit[symbol] = take_profit

            logger.info(f"Trade executed for {symbol}: {order}")

        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {str(e)}")
            return

    async def start_monitoring_tasks(self) -> None:
        """Start all monitoring tasks."""
        try:
            self._is_running = True
            self._monitoring_tasks = [
                asyncio.create_task(self._monitor_positions()),
                asyncio.create_task(self._monitor_dca()),
                asyncio.create_task(self._monitor_trailing_stops())
            ]
            logger.info("Monitoring tasks started successfully")
        except Exception as e:
            logger.error(f"Error starting monitoring tasks: {str(e)}")
            
    async def stop_monitoring_tasks(self) -> None:
        """Stop all monitoring tasks."""
        try:
            self._is_running = False
            for task in self._monitoring_tasks:
                task.cancel()
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
            logger.info("Monitoring tasks stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping monitoring tasks: {str(e)}")
            
    async def _monitor_positions(self) -> None:
        """Monitor and manage open positions."""
        try:
            positions = await self.binance_service.get_positions()
            if not positions:
                return
                
            for position in positions:
                if position and float(position.get('contracts', 0)) > 0:
                    symbol = position.get('symbol')
                    if symbol:
                        # Check if we should close the position
                        if await self.should_close_position(position):
                            await self.binance_service.close_position(symbol)
                            continue
                            
                        # Check if we should update stops
                        current_price = await self.binance_service.get_current_price(symbol)
                        if current_price and await self.should_update_stops(position, current_price):
                            new_stops = await self.calculate_new_stops(position, current_price)
                            if new_stops:
                                position_side = position.get('side', 'LONG')
                                if 'stop_loss' in new_stops:
                                    await self._update_stop_loss(symbol, new_stops['stop_loss'], position_side)
                                if 'take_profit' in new_stops:
                                    await self._update_take_profit(symbol, new_stops['take_profit'], position_side)
                                    
        except Exception as e:
            logger.error(f"Error monitoring positions: {str(e)}")
            
    async def _monitor_dca(self) -> None:
        """Monitor positions for DCA opportunities."""
        try:
            positions = await self.binance_service.get_positions()
            if not positions:
                return
                
            for position in positions:
                if position and float(position.get('contracts', 0)) > 0:
                    symbol = position.get('symbol')
                    if symbol:
                        # Check if the position is in loss (for DCA, we want to DCA when in loss)
                        unrealized_pnl = float(position.get('unrealizedPnl', 0))
                        if unrealized_pnl > 0:
                            continue  # Skip this position but continue with others
                            
                        dca_result = await self._handle_dca(symbol, position)
                        if dca_result:
                            await self.telegram_service.send_dca_notification(dca_result)
                        
        except Exception as e:
            logger.error(f"Error monitoring DCA: {str(e)}")
            
    async def _monitor_trailing_stops(self) -> None:
        """Monitor and update trailing stops for all positions."""
        try:
            positions = await self.binance_service.get_positions()
            if not positions:
                return
                
            for position in positions:
                if position and float(position.get('contracts', 0)) > 0:
                    symbol = position.get('symbol')
                    if symbol:
                        # Check if the position is profitable
                        unrealized_pnl = float(position.get('unrealizedPnl', 0))
                        if unrealized_pnl <= 0:
                            continue  # Skip this position but continue with others
                            
                        # Determine position side based on position amount
                        position_amt = float(position.get('info').get('positionAmt', 0))
                        position_side = 'LONG' if position_amt > 0 else 'SHORT'
                        
                        await self._update_trailing_stop(symbol, position_side)
                    
        except Exception as e:
            logger.error(f"Error monitoring trailing stops: {str(e)}")
            
    def _check_trend_following_signal(self, symbol: str, market_conditions: Dict) -> Optional[Dict]:
        """
        Check for trend following signals based on EMAs, MACD, and RSI.
        
        Args:
            symbol: Trading pair symbol
            market_conditions: Dictionary containing market data and indicators
            
        Returns:
            Optional[Dict]: Signal dictionary if conditions met, None otherwise
        """
        try:
            if not market_conditions or 'df' not in market_conditions:
                logger.warning(f"No market data available for {symbol}")
                return None
                
            df = market_conditions['df']
            if df is None or df.empty:
                logger.warning(f"Empty DataFrame for {symbol}")
                return None
                
            # Get current values
            current_price = df['close'].iloc[-1]
            ema_fast = df['EMA_FAST'].iloc[-1]
            ema_slow = df['EMA_SLOW'].iloc[-1]
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_SIGNAL'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            volume = df['volume'].iloc[-1]
            volume_ma = df['volume'].rolling(window=20).mean().iloc[-1]
            atr = df['ATR'].iloc[-1]
            
            # Initialize signal
            signal = {
                'type': 'trend_following',
                'side': None,
                'strength': 0.0,
                'confidence': 0.0,
                'entry_price': current_price,
                'stop_loss': None,
                'take_profit': None,
                'conditions': {}
            }
            
            # Check bullish trend conditions
            if (current_price > ema_fast > ema_slow and 
                macd > macd_signal and 
                rsi < 70 and 
                volume > volume_ma * 1.2):
                
                signal['side'] = 'long'
                signal['strength'] = 0.8
                signal['confidence'] = 0.7
                signal['stop_loss'] = ema_slow * 0.99  # 1% buffer
                signal['take_profit'] = current_price + (atr * 2)
                signal['conditions'] = {
                    'price_above_emas': True,
                    'macd_bullish': True,
                    'rsi_ok': True,
                    'volume_confirmation': True
                }
                
            # Check bearish trend conditions
            elif (current_price < ema_fast < ema_slow and 
                  macd < macd_signal and 
                  rsi > 30 and 
                  volume > volume_ma * 1.2):
                
                signal['side'] = 'short'
                signal['strength'] = -0.8
                signal['confidence'] = 0.7
                signal['stop_loss'] = ema_slow * 1.01  # 1% buffer
                signal['take_profit'] = current_price - (atr * 2)
                signal['conditions'] = {
                    'price_below_emas': True,
                    'macd_bearish': True,
                    'rsi_ok': True,
                    'volume_confirmation': True
                }
            
            return signal if signal['side'] is not None else None
            
        except Exception as e:
            logger.error(f"Error checking trend following signals for {symbol}: {str(e)}")
            return None

    def _check_breakout_signal(self, symbol: str, market_conditions: Dict) -> Optional[Dict]:
        """
        Check for breakout signals based on Bollinger Bands, volume, and RSI.
        
        Args:
            symbol: Trading pair symbol
            market_conditions: Dictionary containing market data and indicators
            
        Returns:
            Optional[Dict]: Signal dictionary if conditions met, None otherwise
        """
        try:
            if not market_conditions or 'df' not in market_conditions:
                logger.warning(f"No market data available for {symbol}")
                return None
                
            df = market_conditions['df']
            if df is None or df.empty:
                logger.warning(f"Empty DataFrame for {symbol}")
                return None
                
            # Get current values
            current_price = df['close'].iloc[-1]
            bb_upper = df['BB_upper'].iloc[-1]
            bb_lower = df['BB_lower'].iloc[-1]
            bb_middle = df['BB_middle'].iloc[-1]
            volume = df['volume'].iloc[-1]
            volume_ma = df['volume'].rolling(window=20).mean().iloc[-1]
            rsi = df['RSI'].iloc[-1]
            atr = df['ATR'].iloc[-1]
            
            # Initialize signal
            signal = {
                'type': 'breakout',
                'side': None,
                'strength': 0.0,
                'confidence': 0.0,
                'entry_price': current_price,
                'stop_loss': None,
                'take_profit': None,
                'conditions': {}
            }
            
            # Check bullish breakout conditions
            if (current_price > bb_upper and 
                volume > volume_ma * 1.5 and 
                rsi < 70 and 
                atr / current_price < 0.05):
                
                signal['side'] = 'long'
                signal['strength'] = 0.8
                signal['confidence'] = 0.7
                signal['stop_loss'] = bb_middle * 0.99  # 1% buffer
                signal['take_profit'] = current_price + (atr * 2)
                signal['conditions'] = {
                    'price_above_bb_upper': True,
                    'high_volume': True,
                    'rsi_ok': True,
                    'atr_ratio_ok': True
                }
                
            # Check bearish breakout conditions
            elif (current_price < bb_lower and 
                  volume > volume_ma * 1.5 and 
                  rsi > 30 and 
                  atr / current_price < 0.05):
                
                signal['side'] = 'short'
                signal['strength'] = -0.8
                signal['confidence'] = 0.7
                signal['stop_loss'] = bb_middle * 1.01  # 1% buffer
                signal['take_profit'] = current_price - (atr * 2)
                signal['conditions'] = {
                    'price_below_bb_lower': True,
                    'high_volume': True,
                    'rsi_ok': True,
                    'atr_ratio_ok': True
                }
            
            return signal if signal['side'] is not None else None
            
        except Exception as e:
            logger.error(f"Error checking breakout signals for {symbol}: {str(e)}")
            return None

    async def _get_market_trend(self, symbol: str) -> str:
        """Get the current market trend for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            str: 'bullish', 'bearish', or None if unable to determine
        """
        try:
            # Get historical data
            df = await self.indicator_service.calculate_indicators(symbol)
            if df is None or df.empty:
                return None
                
            # Calculate trend using multiple indicators
            rsi = df['RSI'].iloc[-1]
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_SIGNAL'].iloc[-1]
            ema20 = df['EMA_FAST'].iloc[-1]
            ema50 = df['EMA_SLOW'].iloc[-1]
            close = df['close'].iloc[-1]
            
            # Count bullish signals
            bullish_signals = 0
            if rsi > 50:
                bullish_signals += 1
            if macd > macd_signal:
                bullish_signals += 1
            if close > ema20:
                bullish_signals += 1
            if ema20 > ema50:
                bullish_signals += 1
                
            # Determine trend based on majority of signals
            if bullish_signals >= 3:
                return 'bullish'
            elif bullish_signals <= 1:
                return 'bearish'
            return None
            
        except Exception as e:
            logger.error(f"Error getting market trend for {symbol}: {str(e)}")
            return None
            
    async def _get_market_sentiment(self, symbol: str) -> float:
        """Get the current market sentiment for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            float: Sentiment score between -1 (bearish) and 1 (bullish), or None if unable to determine
        """
        try:
            # Get historical data
            df = await self.indicator_service.calculate_indicators(symbol)
            if df is None or df.empty:
                return None
                
            # Calculate sentiment using multiple factors
            rsi = df['RSI'].iloc[-1]
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_SIGNAL'].iloc[-1]
            volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            
            # Normalize RSI to -1 to 1 range
            rsi_sentiment = (rsi - 50) / 50
            
            # Calculate MACD sentiment
            macd_sentiment = 1 if macd > macd_signal else -1
            
            # Calculate volume sentiment
            volume_sentiment = 1 if volume > avg_volume else -1
            
            # Combine sentiments with weights
            sentiment = (
                0.4 * rsi_sentiment +
                0.4 * macd_sentiment +
                0.2 * volume_sentiment
            )
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error getting market sentiment for {symbol}: {str(e)}")
            return None

   
    async def _manage_existing_position(self, symbol: str, position: Dict) -> None:
        """Manage an existing position.
        
        Args:
            symbol: Trading pair symbol
            position: Current position details
        """
        try:
            # Check if we should close the position
            if await self.should_close_position(position):
                await self.binance_service.close_position(symbol, position['info']['positionSide'])
                return
                
            # Get current price and market conditions
            current_price = await self.binance_service.get_current_price(symbol)
            if not current_price:
                logger.error(f"Failed to get current price for {symbol}")
                return
                
            market_conditions = await self._get_market_conditions(symbol)
            if not market_conditions:
                logger.error(f"Failed to get market conditions for {symbol}")
                return
                
            # Check if we should update stops
            if await self.should_update_stops(position, current_price):
                new_stops = await self.calculate_new_stops(position, current_price)
                if new_stops:
                    # Update stop loss
                    await self._update_stop_loss(
                        symbol=symbol,
                        new_stop_loss=new_stops['stop_loss'],
                        position_type=position['info']['positionSide']
                    )
                    
                    # Update take profit
                    await self._update_take_profit(
                        symbol=symbol,
                        new_take_profit=new_stops['take_profit'],
                        position_type=position['info']['positionSide']
                    )
                    
            # Check for DCA opportunity
            dca_result = await self._handle_dca(symbol, position)
            if dca_result:
                # DCA was executed, update position tracking
                self._last_dca_time[symbol] = time.time()
                self._dca_history[symbol] = dca_result.get('dca_history', [])
                
            # Update trailing stop if needed
            if await self._update_trailing_stop(symbol, position['info']['positionSide']):
                # Trailing stop was updated, no need for further action
                return
                
        except Exception as e:
            logger.error(f"Error managing existing position for {symbol}: {str(e)}")
                
    def _calculate_trend_score(self, timeframe_analysis: Dict) -> float:
        """Calculate trend score based on multiple timeframe analysis.
        
        Args:
            df: DataFrame with price data
            timeframe_analysis: Analysis of multiple timeframes
            
        Returns:
            float: Trend score between -1 and 1
        """
        try:
            if not timeframe_analysis:
                return 0.0
                
            trend = timeframe_analysis.get('trend', 'NEUTRAL')
            if trend == 'UP':
                return 1.0
            elif trend == 'DOWN':
                return -1.0
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating trend score: {str(e)}")
            return 0.0
            
    def _calculate_volume_score(self, df: pd.DataFrame) -> float:
        """Calculate volume score based on volume conditions.
        
        Args:
            df: DataFrame with price and volume data
            
        Returns:
            float: Volume score between -1.0 and 1.0
        """
        try:
            if df is None or df.empty:
                logger.warning("Empty DataFrame provided for volume score calculation")
                return 0.0
                
            # Calculate volume moving average
            volume_ma = df["volume"].rolling(window=20).mean()
            
            # Calculate volume ratio
            volume_ratio = df["volume"].iloc[-1] / volume_ma.iloc[-1]
            
            # Calculate score based on volume ratio
            if volume_ratio >= 2.0:  # Very high volume
                return 1.0
            elif volume_ratio >= 1.5:  # High volume
                return 0.8
            elif volume_ratio >= 1.2:  # Above average volume
                return 0.6
            elif volume_ratio >= 1.0:  # Average volume
                return 0.4
            elif volume_ratio >= 0.8:  # Below average volume
                return 0.2
            elif volume_ratio >= 0.5:  # Low volume
                return -0.2
            else:  # Very low volume
                return -0.4
                
        except Exception as e:
            logger.error(f"Error calculating volume score: {str(e)}")
            return 0.0
            
    def _calculate_volatility_score(self, btc_volatility: Dict) -> float:
        """Calculate volatility score based on BTC volatility and trend.
        
        Args:
            df: DataFrame with price data
            btc_volatility: BTC volatility analysis
            
        Returns:
            float: Volatility score between -1 and 1
        """
        try:
            if not btc_volatility:
                return 0.0
                
            volatility = btc_volatility.get('volatility_level', 'LOW')
            btc_trend = btc_volatility.get('trend', 'NEUTRAL')
            
            # Base volatility score
            if volatility == 'HIGH':
                base_score = 1.0
            elif volatility == 'MEDIUM':
                base_score = 0.5
            else:
                base_score = 0.0
                
            # Adjust score based on BTC trend
            if btc_trend == 'UP':
                return base_score  # Keep positive score for LONG signals
            elif btc_trend == 'DOWN':
                return -base_score  # Negative score for SHORT signals
            else:
                return 0.0  # Neutral trend, no adjustment
                
        except Exception as e:
            logger.error(f"Error calculating volatility score: {str(e)}")
            return 0.0
            
    def _calculate_correlation_score(self, altcoin_correlation: Dict) -> float:
        """Calculate correlation score based on altcoin correlation with BTC.
        
        Args:
            altcoin_correlation: Altcoin correlation analysis
            
        Returns:
            float: Correlation score between -1 and 1
        """
        try:
            if not altcoin_correlation:
                return 0.0
                
            correlation = altcoin_correlation.get('correlation', 0)
            btc_trend = altcoin_correlation.get('btc_trend', 'NEUTRAL')
            
            # Base correlation score
            base_score = correlation
            
            # Adjust score based on BTC trend and correlation
            if btc_trend == 'UP':
                if correlation > 0:
                    return base_score  # Positive correlation with uptrend
                else:
                    return -base_score  # Negative correlation with uptrend
            elif btc_trend == 'DOWN':
                if correlation > 0:
                    return -base_score  # Positive correlation with downtrend
                else:
                    return base_score  # Negative correlation with downtrend
            else:
                return base_score  # Neutral trend, use raw correlation
                
        except Exception as e:
            logger.error(f"Error calculating correlation score: {str(e)}")
            return 0.0
            
    def _calculate_sentiment_score(self, sentiment: Dict) -> float:
        """Calculate sentiment score based on market sentiment.
        
        Args:
            sentiment: Market sentiment analysis
            
        Returns:
            float: Sentiment score between -1 and 1
        """
        try:
            if not sentiment:
                return 0.0
                
            sentiment_value = sentiment.get('sentiment', 0)
            return sentiment_value  # Use raw sentiment value
            
        except Exception as e:
            logger.error(f"Error calculating sentiment score: {str(e)}")
            return 0.0

