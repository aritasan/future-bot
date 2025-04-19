"""
Enhanced trading strategy implementation.
"""

import logging
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import time
import asyncio

from src.services.indicator_service import IndicatorService
from src.services.sentiment_service import SentimentService
from src.services.binance_service import BinanceService
from src.services.telegram_service import TelegramService

logger = logging.getLogger(__name__)

class EnhancedTradingStrategy:
    """Enhanced trading strategy with multiple indicators and risk management."""
    
    def __init__(self, config: Dict, binance_service: BinanceService, indicator_service: IndicatorService, telegram_service: TelegramService):
        """Initialize the trading strategy.
        
        Args:
            config: Configuration dictionary
            binance_service: Binance service instance
            indicator_service: Indicator service instance
            telegram_service: Telegram service instance
        """
        self.config = config
        self.binance_service = binance_service
        self.indicator_service = indicator_service
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
        """Generate trading signals with enhanced market analysis.
        
        Args:
            symbol: Trading pair symbol
            indicator_service: Service for calculating indicators
            
        Returns:
            Optional[Dict]: Trading signals if valid, None otherwise
        """
        try:
            # Get market data
            df = await indicator_service.calculate_indicators(symbol)
            if df is None or df.empty:
                return None
                
            # Multi-timeframe analysis
            timeframe_analysis = await self.analyze_multiple_timeframes(symbol)
            if not timeframe_analysis:
                return None
                
            # BTC volatility analysis
            btc_volatility = await self.analyze_btc_volatility()
            if not btc_volatility:
                return None
                
            # Altcoin correlation analysis
            altcoin_correlation = await self.analyze_altcoin_correlation(symbol, btc_volatility)
            if not altcoin_correlation:
                return None
                
            # Market sentiment analysis
            sentiment = await self.analyze_market_sentiment(symbol)
            if not sentiment:
                return None
                
            # Calculate signal score
            signal_score = await self.calculate_signal_score(
                df, timeframe_analysis, btc_volatility, 
                altcoin_correlation, sentiment
            )
            
            # Prepare market conditions for strategy-specific checks
            market_conditions = {
                'trend': timeframe_analysis.get('trend', ''),
                'trend_strength': timeframe_analysis.get('trend_strength', 0),
                'volatility': btc_volatility.get('volatility', 0),
                'volume': df['volume'].iloc[-1] / df['volume'].mean(),
                'price_action': {
                    'breakout_direction': 'up' if df['close'].iloc[-1] > df['BB_upper'].iloc[-1] else 'down' if df['close'].iloc[-1] < df['BB_lower'].iloc[-1] else ''
                }
            }
            
            # Check strategy-specific signals
            trend_signal = self._check_trend_following_signal(symbol, market_conditions)
            breakout_signal = self._check_breakout_signal(symbol, market_conditions)
            
            # If either strategy generates a valid signal, use it
            strategy_signal = trend_signal or breakout_signal
            if strategy_signal:
                signal_score = strategy_signal['signal_score']
                position_type = strategy_signal['side']
                should_trade = strategy_signal['should_trade']
            else:
                # Determine position type based on signal score
                if signal_score > 0.6 or signal_score < -0.6:
                    print(f"{symbol} signal_score: {signal_score}")
                if signal_score > 0.6:  # Strong buy signal
                    position_type = 'buy'
                    should_trade = True
                elif signal_score < -0.6:  # Strong sell signal
                    position_type = 'sell'
                    should_trade = True
                else:
                    should_trade = False
                    return None
                
            # Check conditions
            if not self.check_volume_condition(df):
                print(f"Volume condition not met for {symbol}")
                return None
                
            if not self.check_volatility_condition(df):
                print(f"Volatility condition not met for {symbol}")
                return None
                
            # if not self.check_adx_condition(df):
            #     print(f"ADX condition not met for {symbol}")
            #     return None
                
            if not self.check_bollinger_condition(df):
                print(f"Bollinger condition not met for {symbol}")
                return None
                
            # Calculate position size
            current_price = float(df['close'].iloc[-1])
            position_size = await self._calculate_position_size(symbol, self.config['trading']['risk_per_trade'], current_price)
            if not position_size:
                return None
                
            # Calculate stop loss and take profit
            stop_loss = await self._calculate_stop_loss(symbol, df, position_type, current_price, df['ATR'].iloc[-1])
            take_profit = await self._calculate_take_profit(symbol, df, position_type, current_price, stop_loss)
            
            return {
                'symbol': symbol,
                'side': position_type,
                'amount': position_size,
                'type': 'market',
                'price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'signal_score': signal_score,
                'timeframe_analysis': timeframe_analysis,
                'btc_volatility': btc_volatility,
                'altcoin_correlation': altcoin_correlation,
                'sentiment': sentiment,
                'should_trade': should_trade
            }
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return None
            
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
            
    async def _calculate_stop_loss(self, symbol: str, df: pd.DataFrame, position_type: str, current_price: float, atr: float) -> float:
        """Calculate stop loss based on market conditions and risk parameters."""
        try:
            # Get market conditions
            market_conditions = await self._get_market_conditions(symbol)
            volatility = market_conditions.get('volatility', 'LOW')
            trend = market_conditions.get('trend', 'DOWN')
            
            # Calculate base stop distance using ATR
            base_stop = atr * self.config['risk_management']['atr_multiplier']
            
            # Adjust stop distance based on market conditions
            if volatility == 'HIGH':  # High volatility
                base_stop *= self.config['risk_management']['volatility_multiplier']
            if trend == 'UP' and position_type == 'LONG' or trend == 'DOWN' and position_type == 'SHORT':  # Strong trend
                base_stop *= self.config['risk_management']['trend_multiplier']
                
            # Calculate stop loss price
            if position_type == 'LONG':
                stop_loss = current_price - base_stop
                # Ensure stop loss is below current price with minimum distance
                min_distance = current_price * self.config['risk_management']['min_stop_distance']
                stop_loss = min(stop_loss, current_price - min_distance)
                
                # Additional validation for minimum distance
                min_allowed_distance = current_price * self.config['risk_management']['min_stop_distance']
                if current_price - stop_loss < min_allowed_distance:
                    stop_loss = current_price - min_allowed_distance
            else:  # SHORT
                stop_loss = current_price + base_stop
                # Ensure stop loss is above current price with minimum distance
                min_distance = current_price * self.config['risk_management']['min_stop_distance']
                stop_loss = max(stop_loss, current_price + min_distance)
                
                # Additional validation for minimum distance
                min_allowed_distance = current_price * self.config['risk_management']['min_stop_distance']
                if stop_loss - current_price < min_allowed_distance:
                    stop_loss = current_price + min_allowed_distance
                
            # Round to appropriate precision
            stop_loss = round(stop_loss, self.config['trading']['price_precision'])
            
            logger.info(f"Calculated stop loss for {symbol} {position_type}: {stop_loss} (current price: {current_price})")
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            return None

    async def _calculate_take_profit(self, symbol: str, df: pd.DataFrame, position_type: str, current_price: float, stop_loss: float) -> float:
        """Calculate take profit based on risk-reward ratio and market conditions."""
        try:
            # Get market conditions
            market_conditions = await self._get_market_conditions(symbol)
            volatility = market_conditions.get('volatility', 'LOW')
            trend = market_conditions.get('trend', 'DOWN')
            
            # Calculate base take profit using risk-reward ratio
            if position_type == 'LONG':
                risk_distance = current_price - stop_loss
                base_tp = current_price + (risk_distance * self.config['risk_management']['take_profit_multiplier'])
                
                # Ensure take profit is above current price with minimum distance
                min_distance = current_price * self.config['risk_management']['min_tp_distance']
                base_tp = max(base_tp, current_price + min_distance)
            else:  # SHORT
                risk_distance = stop_loss - current_price
                base_tp = current_price - (risk_distance * self.config['risk_management']['take_profit_multiplier'])
                
                # Ensure take profit is below current price with minimum distance
                min_distance = current_price * self.config['risk_management']['min_tp_distance']
                base_tp = min(base_tp, current_price - min_distance)
            
            # Adjust take profit based on market conditions
            if volatility == 'HIGH':  # High volatility
                base_tp *= self.config['risk_management']['volatility_multiplier']
            if trend == 'UP' and position_type == 'LONG' or trend == 'DOWN' and position_type == 'SHORT':  # Strong trend
                base_tp *= self.config['risk_management']['trend_multiplier']
            
            # Round to appropriate precision
            take_profit = round(base_tp, self.config['trading']['price_precision'])
            
            logger.info(f"Calculated take profit for {symbol} {position_type}: {take_profit} (current price: {current_price})")
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
            symbol = position['symbol']
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
                    logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                    return True
                    
            # Check take profit
            take_profit = float(position.get('takeProfit', 0))
            if take_profit > 0:
                if (position_amt > 0 and current_price >= take_profit) or \
                   (position_amt < 0 and current_price <= take_profit):
                    logger.info(f"Take profit triggered for {symbol} at {current_price}")
                    return True
                    
            # Check trend reversal
            try:
                # Get current trend
                trend = await self._get_market_trend(symbol)
                if trend:
                    # If trend has reversed from our position
                    if (position_amt > 0 and trend == 'bearish') or \
                       (position_amt < 0 and trend == 'bullish'):
                        logger.info(f"Trend reversal detected for {symbol}: {trend}")
                        return True
            except Exception as e:
                logger.error(f"Error checking trend for {symbol}: {str(e)}")
                
            # Check market sentiment
            try:
                sentiment = await self._get_market_sentiment(symbol)
                if sentiment:
                    # If sentiment is strongly against our position
                    if (position_amt > 0 and sentiment < -0.7) or \
                       (position_amt < 0 and sentiment > 0.7):
                        logger.info(f"Strong market sentiment against position for {symbol}: {sentiment}")
                        return True
            except Exception as e:
                logger.error(f"Error checking sentiment for {symbol}: {str(e)}")
                
            return False
            
        except Exception as e:
            logger.error(f"Error in should_close_position: {str(e)}")
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
            volatility = market_conditions['volatility']
            base_stop_distance = self.config['risk_management']['base_stop_distance']
            volatility_multiplier = 1 + (volatility * self.config['risk_management']['volatility_multiplier'])
            stop_distance = base_stop_distance * volatility_multiplier
            
            # Adjust stop distance based on trend strength
            trend_strength = market_conditions['trend_strength']
            trend_multiplier = 1 + (trend_strength * self.config['risk_management']['trend_multiplier'])
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
            base_stop = atr * self.config['risk_management']['atr_multiplier']  # Use ATR multiplier from config
            
            # Adjust for position type
            if position_type == "BUY":
                trailing_stop = current_price - base_stop
            else:
                trailing_stop = current_price + base_stop
                
            return trailing_stop
            
        except Exception as e:
            logger.error(f"Error calculating trailing stop: {str(e)}")
            return current_price
            
    async def calculate_signal_score(self, df: pd.DataFrame, timeframe_analysis: Dict, 
                                   btc_volatility: Dict, altcoin_correlation: Dict,
                                   sentiment: Dict) -> float:
        """Calculate a signal score based on multiple factors.
        
        Args:
            df: DataFrame with price data
            timeframe_analysis: Analysis results from multiple timeframes
            btc_volatility: BTC volatility analysis
            altcoin_correlation: Altcoin correlation analysis
            sentiment: Market sentiment analysis
            
        Returns:
            float: Signal score between 0 and 1
        """
        try:
            # Initialize score components
            trend_score = 0.0
            volume_score = 0.0
            volatility_score = 0.0
            correlation_score = 0.0
            sentiment_score = 0.0
            
            # Calculate trend score
            if timeframe_analysis:
                trend_strength = timeframe_analysis.get('strength', 0)
                trend_direction = timeframe_analysis.get('trend', 0)
                trend_score = (trend_strength * trend_direction) / 100.0
                
            # Calculate volume score
            if self.check_volume_condition(df):
                volume_score = 0.2
                
            # Calculate volatility score
            if btc_volatility:
                volatility_level = btc_volatility.get('volatility_level', 'medium')
                if volatility_level == 'low':
                    volatility_score = 0.3
                elif volatility_level == 'medium':
                    volatility_score = 0.2
                else:
                    volatility_score = 0.1
                    
            # Calculate correlation score
            if altcoin_correlation:
                correlation = altcoin_correlation.get('correlation', 0)
                correlation_score = abs(correlation) * 0.2
                
            # Calculate sentiment score
            if sentiment:
                sentiment_value = sentiment.get('sentiment', 0)
                sentiment_score = (sentiment_value + 1) * 0.15  # Normalize to 0-0.3
                
            # Calculate final score
            final_score = (
                trend_score +
                volume_score +
                volatility_score +
                correlation_score +
                sentiment_score
            )
            
            # Ensure score is between 0 and 1
            final_score = max(0.0, min(1.0, final_score))
            
            logger.debug(f"Signal score components: trend={trend_score:.2f}, "
                        f"volume={volume_score:.2f}, volatility={volatility_score:.2f}, "
                        f"correlation={correlation_score:.2f}, sentiment={sentiment_score:.2f}, "
                        f"final={final_score:.2f}")
                        
            return final_score
            
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
                
                # Create clean DataFrames for correlation
                corr_df = pd.DataFrame({
                    'btc': btc_returns,
                    'alt': alt_returns
                })
                
                # Drop any NaN values
                corr_df = corr_df.dropna()
                
                if len(corr_df) < 10:
                    logging.warning(f"Insufficient data points after cleaning: {len(corr_df)}")
                    return self._get_default_correlation()
                
                # Calculate standard deviations with minimum threshold
                btc_std = corr_df['btc'].std()
                alt_std = corr_df['alt'].std()
                
                # Check for very small or zero standard deviations
                min_std = 1e-8
                if btc_std < min_std or alt_std < min_std:
                    logging.warning(f"Near-zero standard deviation detected: BTC={btc_std}, ALT={alt_std}")
                    return self._get_default_correlation()
                
                # Calculate correlation manually to avoid numpy warning
                btc_norm = (corr_df['btc'] - corr_df['btc'].mean()) / btc_std
                alt_norm = (corr_df['alt'] - corr_df['alt'].mean()) / alt_std
                correlation = (btc_norm * alt_norm).mean()
                
                if not np.isfinite(correlation):
                    logging.warning("Non-finite correlation value detected")
                    return self._get_default_correlation()
                
                # Calculate reaction metrics
                reaction_strength = abs(correlation)
                is_strongly_correlated = reaction_strength > 0.7
                is_reacting = reaction_strength > 0.5
                
                result = {
                    'correlation': float(correlation),
                    'returns_correlation': float(correlation),
                    'reaction_strength': float(reaction_strength),
                    'is_strongly_correlated': is_strongly_correlated,
                    'is_reacting': is_reacting,
                    'reaction': 'STRONG' if is_strongly_correlated else 'MODERATE' if is_reacting else 'WEAK',
                    'data_points': len(corr_df),
                    'btc_std': float(btc_std),
                    'alt_std': float(alt_std)
                }
                
                logging.debug(f"Correlation calculation complete for {symbol}:")
                logging.debug(f"Data points: {result['data_points']}")
                logging.debug(f"BTC std: {result['btc_std']:.6f}")
                logging.debug(f"ALT std: {result['alt_std']:.6f}")
                logging.debug(f"Correlation: {result['correlation']:.4f}")
                logging.debug(f"Reaction: {result['reaction']}")
                
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
        """Handle Dollar Cost Averaging (DCA) for a position.
        
        Args:
            symbol: Trading pair symbol
            position: Position details
            
        Returns:
            Dict with DCA execution details or None if no DCA needed
        """
        try:
            # Get position details from the position structure
            if 'info' not in position:
                logger.error(f"Invalid position structure for {symbol}")
                return None
                
            position_amt = float(position['info'].get('positionAmt', '0'))
            if position_amt == 0:
                logger.info(f"No position found for {symbol}")
                return None
                
            mark_price = float(position['info'].get('markPrice', '0'))
            entry_price = float(position['info'].get('entryPrice', '0'))
            if not mark_price or not entry_price:
                logger.error(f"Invalid price data for {symbol}")
                return None
                
            # Check if the position is in loss (for DCA, we want to DCA when in loss)
            unrealized_pnl = float(position.get('unrealizedPnl', 0))
            if unrealized_pnl > 0:
                logger.info(f"Skipping DCA for {symbol} - position is profitable")
                return None
                
            # Determine position side
            position_side = position['info'].get('positionSide', 'LONG')
            is_long = position_side == 'LONG'
            
            # Calculate price drop percentage
            price_drop = ((entry_price - mark_price) / entry_price) * 100 if is_long else \
                        ((mark_price - entry_price) / entry_price) * 100
                        
            # Get market conditions
            market_conditions = await self._get_market_conditions(symbol)
            if not market_conditions:
                logger.error(f"Could not get market conditions for {symbol}")
                return None
                
            # Check if DCA is favorable
            if not self._is_dca_favorable(price_drop, market_conditions):
                logger.info(f"DCA not favorable for {symbol} - Price drop: {price_drop:.2f}%")
                return None
                
            # Calculate DCA amount based on position size and risk
            base_amount = abs(position_amt)
            dca_amount = base_amount * self.config['risk_management']['dca_multiplier']
            
            # Round to appropriate precision
            precision = self.config['trading']['price_precision']
            dca_amount = round(dca_amount, precision)
            
            # Execute DCA order
            order_params = {
                'symbol': symbol,
                'side': 'BUY' if is_long else 'SELL',
                'type': 'MARKET',
                'amount': dca_amount,
                'position_side': position_side
            }
            order = await self.binance_service.place_order(order_params)
            
            if not order:
                logger.error(f"Failed to execute DCA order for {symbol}")
                return None
                
            # Calculate new average entry price
            total_position = base_amount + dca_amount
            new_entry = ((entry_price * base_amount) + (mark_price * dca_amount)) / total_position
            
            logger.info(f"Executed DCA for {symbol} - Amount: {dca_amount}, New Entry: {new_entry:.8f}")
            
            return {
                'symbol': symbol,
                'dca_amount': dca_amount,
                'new_entry_price': new_entry,
                'price_drop': price_drop,
                'order_id': order.get('orderId')
            }
            
        except Exception as e:
            logger.error(f"Error handling DCA: {str(e)}")
            return None

    async def _get_market_conditions(self, symbol: str) -> Dict:
        """Get current market conditions for DCA decision making.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict: Market conditions including trend, volatility, and volume
        """
        try:
            # Get indicators for the symbol
            df = await self.indicator_service.calculate_indicators(symbol)
            if df is None or df.empty:
                return {}
                
            # Analyze trend
            trend = "UP" if df['close'].iloc[-1] > df['EMA_FAST'].iloc[-1] else "DOWN"
            
            # Analyze volatility
            atr = df['ATR'].iloc[-1]
            atr_ma = df['ATR'].rolling(20).mean().iloc[-1]
            volatility = "HIGH" if atr > atr_ma * 1.5 else "MEDIUM" if atr > atr_ma else "LOW"
            
            # Analyze volume
            volume = df['volume'].iloc[-1]
            volume_ma = df['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = volume / volume_ma if volume_ma > 0 else 0
            volume_status = "HIGH" if volume_ratio > 1.5 else "MEDIUM" if volume_ratio > 1 else "LOW"
            
            return {
                'trend': trend,
                'volatility': volatility,
                'volume': volume_status,
                'atr': atr,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            logger.error(f"Error getting market conditions: {str(e)}")
            return {}
            
    def _is_dca_favorable(self, price_drop: float, market_conditions: Dict) -> bool:
        """Check if market conditions are favorable for DCA.
        
        Args:
            price_drop: Current price drop percentage
            market_conditions: Current market conditions
            
        Returns:
            bool: True if conditions are favorable for DCA
        """
        try:
            # Check if we have valid market conditions
            if not market_conditions:
                return False
                
            # Get conditions
            trend = market_conditions.get('trend')
            volatility = market_conditions.get('volatility')
            volume = market_conditions.get('volume', '')
            atr = market_conditions.get('atr', 0)
            volume_ratio = market_conditions.get('volume_ratio', 0)
            
            # Favorable conditions:
            # 1. Price drop is significant but not too extreme (2% - 10%)
            # 2. Volatility is not extremely high
            # 3. Volume is not extremely low
            # 4. Trend is not strongly against the position
            # 5. ATR is within reasonable range
            return (
                price_drop >= 0.02 and  # Minimum 2% drop
                price_drop <= 0.10 and  # Maximum 10% drop
                volatility != "HIGH" and  # Avoid high volatility
                volume != "LOW" and  # Avoid low volume
                volume_ratio > 0.8 and  # Volume above 80% of average
                atr > 0 and  # Valid ATR
                trend != "DOWN"  # Avoid DCA in strong downtrend
            )
            
        except Exception as e:
            logger.error(f"Error checking DCA favorability: {str(e)}")
            return False

    async def _update_trailing_stop(self, symbol: str, position: Dict) -> bool:
        """Update trailing stop for a position based on market analysis."""
        try:
            # Check if we've updated this symbol's trailing stop recently
            current_time = time.time()
            if symbol in self._last_trailing_stop_update:
                last_update = self._last_trailing_stop_update[symbol]
                if current_time - last_update < self._trailing_stop_debounce:
                    logger.info(f"Skipping trailing stop update for {symbol} - too soon since last update")
                    return False
            
            # Get position details from the new structure
            position_amt = float(position.get('info', {}).get('positionAmt', 0))
            if position_amt == 0:
                logger.info(f"No position found for {symbol}")
                return False
                
            mark_price = float(position.get('markPrice', 0))
            entry_price = float(position.get('entryPrice', 0))
            if not mark_price or not entry_price:
                logger.error(f"Invalid price data for {symbol}")
                return False
                
            # Check if the position is profitable
            unrealized_pnl = float(position.get('unrealizedPnl', 0))
            if unrealized_pnl <= 0:
                logger.info(f"Skipping trailing stop update for {symbol} - position is not profitable")
                return False
                
            # Determine position side
            position_side = position.get('positionSide', 'LONG')
            is_long = position_side == 'LONG'
            
            # Get current market conditions
            market_data = await self.binance_service.get_market_data(symbol)
            if not market_data:
                logger.error(f"Could not get market data for {symbol}")
                return False
                
            # Calculate ATR for dynamic stop loss
            atr = await self.indicator_service.calculate_atr(symbol)
            if not atr:
                logger.error(f"Could not calculate ATR for {symbol}")
                return False
                
            # Calculate new stop loss based on market conditions
            volatility = market_data.get('volatility', 0)
            trend_strength = market_data.get('trend_strength', 0)
            
            # Adjust ATR multiplier based on volatility and trend strength
            atr_multiplier = self.config['risk_management']['atr_multiplier']
            if volatility > 0.02:  # High volatility
                atr_multiplier *= 1.2
            if trend_strength > 0.7:  # Strong trend
                atr_multiplier *= 1.1
                
            # Calculate new stop loss
            if is_long:
                new_stop = mark_price - (atr * atr_multiplier)
                # Only update if new stop is higher than current stop
                current_stop = await self._get_current_stop_loss(symbol)
                if current_stop and new_stop <= current_stop:
                    logger.info(f"New stop loss {new_stop} not more favorable than current {current_stop}")
                    return False
            else:
                new_stop = mark_price + (atr * atr_multiplier)
                # Only update if new stop is lower than current stop
                current_stop = await self._get_current_stop_loss(symbol)
                if current_stop and new_stop >= current_stop:
                    logger.info(f"New stop loss {new_stop} not more favorable than current {current_stop}")
                    return False
                    
            # Round to appropriate precision
            precision = self.config['trading']['price_precision']
            new_stop = round(new_stop, precision)
            
            # Cancel existing stop loss orders before creating a new one
            existing_stop_orders = await self._get_stop_loss_orders(symbol)
            if existing_stop_orders:
                logger.info(f"Canceling {len(existing_stop_orders)} existing stop loss orders for {symbol}")
                for order in existing_stop_orders:
                    order_id = order.get('id')
                    if order_id:
                        success = await self.binance_service.cancel_order(symbol, order_id)
                        if not success:
                            logger.warning(f"Failed to cancel stop loss order {order_id} for {symbol}")
            
            # Update stop loss order
            success = await self.binance_service.update_stop_loss(
                symbol=symbol,
                stop_price=new_stop,
                side='SELL' if is_long else 'BUY',
                position_side=position_side,
                amount=abs(position_amt)  # Add the position amount
            )
            
            if success:
                logger.info(f"Updated trailing stop for {symbol} to {new_stop}")
                # Update the last update time
                self._last_trailing_stop_update[symbol] = current_time
                # Send Telegram notification
                await self.telegram_service.send_trailing_stop_notification(
                    symbol=symbol,
                    new_stop=new_stop,
                    position_side=position_side
                )
            else:
                logger.error(f"Failed to update trailing stop for {symbol}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error updating trailing stop: {str(e)}")
            return False

    async def _get_current_stop_loss(self, symbol: str) -> float:
        """Get current stop loss price for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            float: Current stop loss price or None if not found
        """
        try:
            # Get open orders
            orders = await self.binance_service.get_open_orders(symbol)
            if not orders:
                return None
                
            # Find stop loss order
            for order in orders:
                if order['type'].lower() == 'stop_loss':
                    return float(order['stopPrice'])
                    
            return None
        except Exception as e:
            logger.error(f"Error getting current stop loss: {str(e)}")
            return None
            
    async def _get_stop_loss_orders(self, symbol: str) -> List[Dict]:
        """Get all stop loss orders for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            List[Dict]: List of stop loss orders or empty list if none found
        """
        try:
            # Get open orders
            orders = await self.binance_service.get_open_orders(symbol)
            if not orders:
                return []
                
            # Filter stop loss orders
            stop_orders = []
            for order in orders:
                if order['type'].lower() in ['stop_loss', 'stop_market']:
                    stop_orders.append(order)
                    
            return stop_orders
        except Exception as e:
            logger.error(f"Error getting stop loss orders: {str(e)}")
            return []

    async def process_trading_signals(self, signals: Dict) -> None:
        """Process trading signals and execute trades with enhanced risk management.
        
        Args:
            signals: Dictionary containing trading signals and analysis
        """
        try:
            symbol = signals.get('symbol')
            if not symbol:
                return
                
            # Get current position
            position = await self.binance_service.get_position(symbol)
            
            # Case 1: No existing position - check if we should open a new position
            if not position:
                # Check if we should trade based on signals
                if signals.get('should_trade', False):
                    # Get market conditions
                    market_conditions = await self._get_market_conditions(symbol)
                    if not market_conditions:
                        return
                        
                    # Execute new trade
                    await self._execute_trade(symbol, signals, market_conditions)
                return
                
            # Case 2: Existing position - manage the position
            # Check if we should close the position
            if await self.should_close_position(position):
                # Position will be closed by should_close_position method
                # and appropriate notifications will be sent
                await self.binance_service.close_position(symbol)
                return
                
            # Check if we should update stops
            if await self.should_update_stops(position, position['entryPrice']):
                # Calculate new stops
                new_stops = await self.calculate_new_stops(position, position['entryPrice'])
                if new_stops:
                    # Update stop loss and take profit
                    await self.binance_service.update_stop_loss(
                        symbol=symbol,
                        stop_price=new_stops['stop_loss'],
                        side='sell' if position['positionAmt'] > 0 else 'buy',
                        amount=abs(float(position['positionAmt']))
                    )
                    await self.binance_service.update_take_profit(
                        symbol=symbol,
                        take_profit=new_stops['take_profit'],
                        side='sell' if position['positionAmt'] > 0 else 'buy',
                        amount=abs(float(position['positionAmt']))
                    )
                    
            # Check for DCA opportunity
            dca_result = await self._handle_dca(symbol, position)
            if dca_result:
                # DCA was executed, update position tracking
                self._last_dca_time[symbol] = time.time()
                self._dca_history[symbol] = dca_result.get('dca_history', [])
                
            # Update trailing stop if needed
            if await self._update_trailing_stop(symbol, position):
                # Trailing stop was updated, no need for further action
                return
                
        except Exception as e:
            logger.error(f"Error processing trading signals: {str(e)}")
            
    async def _execute_trade(self, symbol: str, signal: Dict, market_conditions: Dict) -> None:
        """Execute a trade based on the signal and market conditions."""
        try:
            # Get current price
            current_price = await self.binance_service.get_current_price(symbol)
            if not current_price:
                logger.error(f"Could not get current price for {symbol}")
                return
                
            # Calculate position size
            position_size = await self._calculate_position_size(symbol, self.config['trading']['risk_per_trade'], current_price)
            if not position_size:
                logger.error(f"Invalid position size calculated for {symbol}")
                return
                
            # Get historical data for stop loss and take profit calculation
            df = await self.indicator_service.calculate_indicators(symbol)
            if df is None or df.empty:
                logger.error(f"Could not get historical data for {symbol}")
                return
                
            # Convert signal side to position type
            position_type = 'LONG' if signal['side'].upper() == 'BUY' else 'SHORT'
                
            # Calculate stop loss and take profit
            stop_loss = await self._calculate_stop_loss(symbol, df, position_type, current_price, df['ATR'].iloc[-1])
            take_profit = await self._calculate_take_profit(symbol, df, position_type, current_price, stop_loss)
            
            if not stop_loss or not take_profit:
                logger.error(f"Invalid stop loss or take profit calculated for {symbol}")
                return
                
            # Check if we need to use reduceOnly
            position = await self.binance_service.get_position(symbol)
            use_reduce_only = False
            position_side = None
            
            if position and float(position.get('positionAmt', 0)) != 0:
                position_side = position.get('positionSide', 'LONG')
                is_long_position = position_side == 'LONG'
                is_long_signal = signal['side'].upper() == 'BUY'
                
                # If we're trying to open a position in the opposite direction, use reduceOnly
                if is_long_position != is_long_signal:
                    use_reduce_only = True
                    logger.info(f"Using reduceOnly for {symbol} - opening position in opposite direction")
            else:
                # If no existing position, set position_side based on signal
                position_side = 'LONG' if signal['side'].upper() == 'BUY' else 'SHORT'
                
            # Place the order
            order_params = {
                'symbol': symbol,
                'side': signal['side'].upper(),  # Ensure side is uppercase
                'type': 'MARKET',  # Ensure type is uppercase
                'amount': position_size,
                'price': signal.get('price'),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_side': position_side
            }
            
            # Add reduceOnly if needed
            if use_reduce_only:
                order_params['reduceOnly'] = True
                
            order = await self.binance_service.place_order(order_params)
            
            if order:
                # Send notification
                await self.telegram_service.send_order_notification(order)
                
                # Update position tracking
                self._last_trade_time[symbol] = time.time()
                self._last_trade_price[symbol] = current_price
                self._last_trade_side[symbol] = signal['side']
                
                logger.info(f"Trade executed for {symbol}: {order}")
            else:
                logger.error(f"Failed to execute trade for {symbol}")
                
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            
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
        """Monitor open positions and update trailing stops."""
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
                            # Position will be closed by should_close_position method
                            # and appropriate notifications will be sent
                            await self.binance_service.close_position(symbol)
                        
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
                            logger.info(f"Skipping DCA for {symbol} - position is profitable")
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
                            logger.info(f"Skipping trailing stop update for {symbol} - position is not profitable")
                            continue  # Skip this position but continue with others
                        await self._update_trailing_stop(symbol, position)
                    
        except Exception as e:
            logger.error(f"Error monitoring trailing stops: {str(e)}")
            
    def _check_trend_following_signal(self, symbol: str, market_conditions: Dict) -> Optional[Dict]:
        """Check if trend following conditions are met and generate a trading signal.
        
        Args:
            symbol: Trading pair symbol
            market_conditions: Current market conditions
            
        Returns:
            Optional[Dict]: Trading signal if conditions are met, None otherwise
        """
        try:
            # Get trend direction
            trend = market_conditions.get('trend', '')
            if not trend:
                return None
                
            # Get trend strength
            trend_strength = market_conditions.get('trend_strength', 0)
            if trend_strength < 0.6:  # Require strong trend
                return None
                
            # Get volatility
            volatility = market_conditions.get('volatility', 0)
            if volatility > 0.8:  # Avoid extremely volatile markets
                return None
                
            # Create trading signal
            signal = {
                'symbol': symbol,
                'strategy': 'trend_following',
                'side': 'buy' if trend == 'uptrend' else 'sell',
                'signal_score': 0.8 if trend == 'uptrend' else -0.8,
                'should_trade': True
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error checking trend following conditions: {str(e)}")
            return None
            
    def _check_breakout_signal(self, symbol: str, market_conditions: Dict) -> Optional[Dict]:
        """Check if breakout conditions are met and generate a trading signal.
        
        Args:
            symbol: Trading pair symbol
            market_conditions: Current market conditions
            
        Returns:
            Optional[Dict]: Trading signal if conditions are met, None otherwise
        """
        try:
            # Get volatility
            volatility = market_conditions.get('volatility', 0)
            if volatility < 0.4:  # Require some volatility for breakouts
                return None
                
            # Get volume
            volume = market_conditions.get('volume', 0)
            if volume < 0.6:  # Require decent volume for breakouts
                return None
                
            # Get price action
            price_action = market_conditions.get('price_action', {})
            if not price_action:
                return None
                
            # Check for breakout
            breakout_direction = price_action.get('breakout_direction', '')
            if not breakout_direction:
                return None
                
            # Create trading signal
            signal = {
                'symbol': symbol,
                'strategy': 'breakout',
                'side': 'buy' if breakout_direction == 'up' else 'sell',
                'signal_score': 0.7 if breakout_direction == 'up' else -0.7,
                'should_trade': True
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error checking breakout conditions: {str(e)}")
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
