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
from src.services.notification_service import NotificationService

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
                print(f"No data for {symbol}")
                return None
                
            # Multi-timeframe analysis
            timeframe_analysis = await self.analyze_multiple_timeframes(symbol)
            if not timeframe_analysis:
                print(f"No timeframe analysis for {symbol}")
                return None
                
            # BTC volatility analysis
            btc_volatility = await self.analyze_btc_volatility()
            if not btc_volatility:
                print(f"No BTC volatility analysis for {symbol}")
                return None
                
            # Altcoin correlation analysis
            altcoin_correlation = await self.analyze_altcoin_correlation(symbol, btc_volatility)
            if not altcoin_correlation:
                print(f"No altcoin correlation analysis for {symbol}")
                return None
                
            # Market sentiment analysis
            sentiment = await self.analyze_market_sentiment(symbol)
            if not sentiment:
                print(f"No sentiment analysis for {symbol}")
                return None
                
            # Calculate signal score
            signal_score = await self.calculate_signal_score(
                df, timeframe_analysis, btc_volatility, 
                altcoin_correlation, sentiment
            )
            
            # Prepare market conditions for strategy-specific checks
            market_conditions = {
                'df': df,
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
            if strategy_signal and strategy_signal.get('should_trade', False):
                signal_score = strategy_signal['signal_score']
                position_type = strategy_signal['side']
                
                # Add additional signal information
                signal_info = {
                    'strength': strategy_signal.get('strength', 0),
                    'confidence': strategy_signal.get('confidence', 0),
                    'entry_price': strategy_signal.get('entry_price', 0),
                    'stop_loss': strategy_signal.get('stop_loss', 0),
                    'take_profit': strategy_signal.get('take_profit', 0),
                    'conditions': strategy_signal.get('conditions', {})
                }
            else:
                # Determine position type based on signal score
                if signal_score > 0.6 or signal_score < -0.6:
                    print(f"{symbol} signal_score: {signal_score}")
                if signal_score > 0.6:  # Strong buy signal
                    position_type = 'long'
                elif signal_score < -0.6:  # Strong sell signal
                    position_type = 'short'
                else:
                    return None
                
                # Default signal information
                signal_info = {
                    'strength': abs(signal_score),
                    'confidence': 0.7 if abs(signal_score) > 0.8 else 0.5,
                    'entry_price': float(df['close'].iloc[-1]),
                    'stop_loss': 0,
                    'take_profit': 0,
                    'conditions': {}
                }
                
            # Check conditions
            if not self.check_volume_condition(df):
                print(f"Volume condition not met for {symbol}")
                return None
                
            if not self.check_volatility_condition(df):
                print(f"Volatility condition not met for {symbol}")
                return None
                
            # if not self.check_bollinger_condition(df):
            #     print(f"Bollinger condition not met for {symbol}")
            #     return None
                
            # Calculate position size
            current_price = float(df['close'].iloc[-1])
            position_size = await self._calculate_position_size(symbol, self.config['trading']['risk_per_trade'], current_price)
            if not position_size:
                print(f"Could not calculate position size for {symbol}")
                return None
                
            # Return final signal
            return {
                'symbol': symbol,
                'side': position_type,
                'position_type': position_type,
                'signal_score': signal_score,
                'should_trade': True,
                'position_size': position_size,
                'current_price': current_price,
                'market_analysis': {
                    'trend': timeframe_analysis.get('trend', ''),
                    'trend_strength': timeframe_analysis.get('trend_strength', 0),
                    'volatility': btc_volatility.get('volatility', 0),
                    'volume_ratio': df['volume'].iloc[-1] / df['volume'].mean(),
                    'price_action': market_conditions['price_action']
                },
                'risk_metrics': {
                    'atr': float(df['ATR'].iloc[-1]),
                    'rsi': float(df['RSI'].iloc[-1]),
                    'adx': float(df['ADX'].iloc[-1]) if 'ADX' in df.columns else 0,
                    'correlation': altcoin_correlation.get('correlation', 0)
                },
                'timeframe_analysis': timeframe_analysis,
                'sentiment_analysis': sentiment,
                **signal_info
            }
            
        except Exception as e:
            print(f"Error generating signals for {symbol}: {str(e)}")
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
            
    async def _calculate_stop_loss(self, symbol: str, position_type: str, current_price: float, atr: float) -> float:
        """Calculate stop loss price based on ATR and market conditions."""
        try:
            # Get stop loss multiplier from config
            stop_loss_multiplier = float(self.config['risk_management']['stop_loss_atr_multiplier'])
            
            # Calculate base stop loss using ATR
            logger.info(f"Calculating stop loss for {symbol} {position_type} {position_type.upper()}")
            if position_type.upper() == "LONG":
                stop_loss = float(current_price) - (float(atr) * stop_loss_multiplier)
            else:
                stop_loss = float(current_price) + (float(atr) * stop_loss_multiplier)
            
            # Get market conditions
            market_conditions = await self._get_market_conditions(symbol)
            
            # Adjust stop loss based on volatility
            volatility = market_conditions.get('volatility', 0)
            try:
                volatility = float(volatility)
                if volatility > float(self.config['risk_management']['high_volatility_threshold']):
                    # Increase stop loss distance in high volatility
                    if position_type.upper() == "LONG":
                        stop_loss = float(current_price) - (float(atr) * stop_loss_multiplier * 1.5)
                    else:
                        stop_loss = float(current_price) + (float(atr) * stop_loss_multiplier * 1.5)
            except (ValueError, TypeError):
                # If volatility is not a number, use default multiplier
                logger.warning(f"Invalid volatility value for {symbol}, using default stop loss")
            
            # Ensure minimum distance from current price
            min_distance = float(self.config['risk_management']['min_stop_distance'])
            if position_type.upper() == "LONG":
                # For LONG positions, ensure stop loss is below current price
                stop_loss = min(stop_loss, float(current_price) * (1 - min_distance))
            else:
                # For SHORT positions, ensure stop loss is above current price
                stop_loss = max(stop_loss, float(current_price) * (1 + min_distance))
            
            logger.info(f"Calculated stop loss for {symbol} {position_type.lower()}: {stop_loss} (current price: {current_price})")
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
            logger.info(f"Calculating take profit for {symbol} {position_type} {position_type.upper()}")
            if position_type.upper() == "LONG":
                take_profit = current_price + (price_diff * risk_reward_ratio)
            else:
                take_profit = current_price - (price_diff * risk_reward_ratio)
            
            # Ensure minimum distance from current price
            min_distance = float(self.config['risk_management']['min_tp_distance'])
            if position_type.upper() == "LONG":
                # For LONG positions, ensure take profit is above current price
                take_profit = max(take_profit, current_price * (1 + min_distance))
            else:
                # For SHORT positions, ensure take profit is below current price
                take_profit = min(take_profit, current_price * (1 - min_distance))
            
            logger.info(f"Calculated take profit for {symbol} {position_type.lower()}: {take_profit} (current price: {current_price})")
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
                if btc_volatility and 'volatility' in btc_volatility:
                    btc_vol = btc_volatility['volatility']
                    if btc_vol > 0.02:  # High BTC volatility
                        analysis['correlation'] *= 1.2  # Increase correlation during high volatility
                        analysis['volatility_adjusted'] = True
                    elif btc_vol < 0.01:  # Low BTC volatility
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
            time_based = trailing_config['time_based']

            # Calculate base trailing stop distance
            base_distance = atr * dynamic_config['atr_multiplier']

            # Adjust distance based on market conditions
            if market_conditions and dynamic_config['volatility_adjustment']:
                volatility = market_conditions.get('volatility', 0)
                base_distance *= (1 + volatility)

            if market_conditions and dynamic_config['trend_adjustment']:
                trend_strength = market_conditions.get('trend_strength', 0)
                base_distance *= (1 + trend_strength)

            # Adjust distance based on time
            if time_based['enabled'] and market_conditions:
                position_age = market_conditions.get('position_age', 0)
                for i, window in enumerate(time_based['time_windows']):
                    if position_age <= window:
                        base_distance *= time_based['distance_multipliers'][i]
                        break

            # Calculate trailing stop price
            if position_type.upper() == 'LONG':
                trailing_stop = current_price - base_distance
            else:
                trailing_stop = current_price + base_distance

            # Ensure minimum distance
            min_distance = self.config['risk_management']['min_stop_distance']
            if position_type.upper() == 'LONG':
                trailing_stop = max(trailing_stop, current_price * (1 - min_distance))
            else:
                trailing_stop = min(trailing_stop, current_price * (1 + min_distance))

            logger.info(f"Calculated trailing stop for {symbol} {position_type}: {trailing_stop}")
            return trailing_stop

        except Exception as e:
            logger.error(f"Error calculating trailing stop: {str(e)}")
            return None

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
        """Handle DCA (Dollar Cost Averaging) for a position with enhanced features.
        
        Args:
            symbol: Trading pair symbol
            position: Current position details
            
        Returns:
            Optional[Dict]: DCA execution details if successful, None otherwise
        """
        try:
            # Get current price and position details
            current_price = await self.binance_service.get_current_price(symbol)
            if not current_price:
                logger.error(f"Failed to get current price for {symbol}")
                return None
                
            entry_price = float(position.get('entryPrice', 0))
            position_size = float(position.get('info').get('positionAmt', 0))
            position_type = position.get('positionSide', 'LONG')
            
            if not entry_price or not position_size:
                logger.error(f"Invalid position details for {symbol}")
                return None
                
            # Calculate price drop percentage
            price_drop = abs(current_price - entry_price) / entry_price
            
            # Get market conditions
            market_conditions = await self._get_market_conditions(symbol)
            if not market_conditions:
                logger.error(f"Failed to get market conditions for {symbol}")
                return None
                
            # Check if DCA is favorable based on multiple conditions
            if not self._is_dca_favorable(price_drop, market_conditions):
                logger.info(f"DCA not favorable for {symbol} at {current_price}")
                return None
                
            # Calculate DCA size based on risk management
            dca_size = await self._calculate_dca_size(position_size, price_drop)
            if not dca_size:
                logger.error(f"Failed to calculate DCA size for {symbol}")
                return None
                
            # Place DCA order
            order_params = {
                'symbol': symbol,
                'side': 'BUY' if position_type.upper() == 'LONG' else 'SELL',
                'type': 'MARKET',
                'amount': dca_size,
                'reduceOnly': False
            }
            
            order = await self.binance_service.place_order(order_params)
            if not order:
                logger.error(f"Failed to place DCA order for {symbol}")
                return None
                
            # Update stop loss and take profit
            new_stop_loss = await self._calculate_stop_loss(
                symbol=symbol,
                position_type=position_type,
                current_price=current_price,
                atr=market_conditions['atr']
            )
            
            new_take_profit = await self._calculate_take_profit(
                symbol=symbol,
                position_type=position_type,
                current_price=current_price,
                stop_loss=new_stop_loss
            )
            
            # Update orders
            await self._update_stop_loss(symbol, new_stop_loss, position_type, position_size + dca_size)
            await self._update_take_profit(symbol, new_take_profit, position_type, position_size + dca_size)
            
            # Update DCA information
            market_conditions['dca_attempts'] = market_conditions.get('dca_attempts', 0) + 1
            market_conditions['last_dca_time'] = time.time()
            market_conditions['active_dca_positions'] = market_conditions.get('active_dca_positions', 0) + 1
            
            # Send notification
            dca_details = {
                'symbol': symbol,
                'dca_amount': dca_size,
                'new_entry_price': current_price,
                'price_drop': price_drop * 100,
                'order_id': order.get('orderId'),
                'stop_loss': new_stop_loss,
                'take_profit': new_take_profit,
                'dca_attempt': market_conditions['dca_attempts'],
                'active_dca_positions': market_conditions['active_dca_positions']
            }
            
            await self.telegram_service.send_dca_notification(dca_details)
            
            return dca_details
            
        except Exception as e:
            logger.error(f"Error handling DCA for {symbol}: {str(e)}")
            return None

    def _is_dca_favorable(self, price_drop: float, market_conditions: Dict) -> bool:
        """Check if DCA conditions are favorable."""
        try:
            # Get DCA configuration
            dca_config = self.config['risk_management']['dca']
            risk_control = dca_config['risk_control']

            # Check price drop threshold
            if price_drop < dca_config['price_drop_thresholds'][0]:
                logger.info("Price drop below minimum threshold")
                return False

            # Check volume condition
            volume_ratio = market_conditions.get('volume_ratio', 1.0)
            if volume_ratio < dca_config['volume_threshold']:
                logger.info("Volume below threshold")
                return False

            # Check volatility condition
            volatility = market_conditions.get('volatility', 0)
            if volatility > dca_config['volatility_threshold']:
                logger.info("Volatility above threshold")
                return False

            # Check RSI condition
            rsi = market_conditions.get('rsi', 50)
            if dca_config['rsi_thresholds']['oversold'] < rsi < dca_config['rsi_thresholds']['overbought']:
                logger.info("RSI not in favorable range")
                return False

            # Check BTC correlation
            btc_correlation = market_conditions.get('btc_correlation', 0)
            if abs(btc_correlation) < dca_config['btc_correlation_threshold']:
                logger.info("BTC correlation below threshold")
                return False

            # Check time since last DCA
            last_dca_time = market_conditions.get('last_dca_time', 0)
            current_time = time.time()
            if current_time - last_dca_time < dca_config['min_time_between_attempts']:
                logger.info("Not enough time since last DCA")
                return False

            # Check max drawdown
            current_drawdown = market_conditions.get('current_drawdown', 0)
            if current_drawdown > risk_control['max_drawdown']:
                logger.info("Max drawdown exceeded")
                return False

            # Check position size
            position_size = market_conditions.get('position_size', 0)
            account_balance = market_conditions.get('account_balance', 0)
            if position_size / account_balance > risk_control['max_position_size']:
                logger.info("Max position size exceeded")
                return False

            # Check profit target
            min_profit_target = risk_control['min_profit_target']
            if market_conditions.get('unrealized_pnl', 0) < min_profit_target:
                logger.info("Profit target not reached")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking DCA conditions: {str(e)}")
            return False

    async def _calculate_dca_size(self, current_size: float, price_drop: float) -> Optional[float]:
        """Calculate DCA size based on risk management rules.
        
        Args:
            current_size: Current position size
            price_drop: Current price drop percentage
            
        Returns:
            Optional[float]: Calculated DCA size if successful, None otherwise
        """
        try:
            # Get account balance
            balance = await self.binance_service.get_account_balance()
            if not balance:
                return None
                
            # Calculate base risk amount
            base_risk = float(balance.get('total', 0)) * self.config['risk_management']['max_risk_per_trade']
            
            # Adjust risk based on price drop
            risk_multiplier = 1.0
            price_drop_thresholds = self.config['risk_management']['dca']['price_drop_thresholds']
            for threshold in price_drop_thresholds:
                if price_drop >= threshold:
                    risk_multiplier *= self.config['risk_management']['dca']['risk_reduction']
                    
            # Calculate final DCA size
            dca_size = (base_risk * risk_multiplier) / self.config['risk_management']['min_stop_distance']
            
            # Ensure DCA size is not too large
            max_position_size = float(balance.get('total', 0)) * self.config['risk_management']['max_risk_per_position']
            if current_size + dca_size > max_position_size:
                dca_size = max_position_size - current_size
                
            return dca_size
            
        except Exception as e:
            logger.error(f"Error calculating DCA size: {str(e)}")
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
                'df': df,
                'trend': trend,
                'volatility': volatility,
                'volume': volume_status,
                'atr': atr,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            logger.error(f"Error getting market conditions: {str(e)}")
            return {}
            
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
                    
            # Get position details
            position = await self.binance_service.get_position(symbol)
            if not position:
                return
                
            # Get current market data
            current_price = float(position['markPrice'])
            entry_price = float(position['entryPrice'])
            unrealized_pnl = float(position.get('unrealizedPnl', 0))
            position_size = float(position.get('info', {}).get('positionAmt', 0))
            
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
            current_stop_loss = float(position.get('stopLoss', 0))
            
            # Check if we should move to break-even
            if self._should_move_to_break_even(
                current_price, unrealized_pnl, position_age, position_size
            ):
                new_stop_loss = entry_price
                logger.info(f"Moving to break-even for {symbol} at {new_stop_loss}")
                
            # Check if we should take partial profit
            if self._should_take_partial_profit(
                current_price, unrealized_pnl, position_age, position_size
            ):
                await self._take_partial_profit(symbol, position_size)
                logger.info(f"Taking partial profit for {symbol}")
                
            # Check for emergency stop
            if self._should_emergency_stop(market_conditions):
                new_stop_loss = self._calculate_emergency_stop(
                    current_price, position_type
                )
                logger.warning(f"Emergency stop triggered for {symbol} at {new_stop_loss}")
                
            # Only update if new stop is more favorable
            if position_type == "BUY" and new_stop_loss > current_stop_loss:
                await self._update_stop_loss(symbol, new_stop_loss, position_type, position_size)
            elif position_type == "SELL" and new_stop_loss < current_stop_loss:
                await self._update_stop_loss(symbol, new_stop_loss, position_type, position_size)
                
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
            
            # Place market order to close portion
            await self.binance_service.place_order(
                symbol=symbol,
                side="SELL" if position_size > 0 else "BUY",
                order_type="MARKET",
                quantity=close_size
            )
            
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
        
        if position_type == "BUY":
            return current_price * (1 - emergency_distance)
        else:
            return current_price * (1 + emergency_distance)
            
    async def _update_stop_loss(self, symbol: str, new_stop_loss: float,
                              position_type: str, position_size: float) -> None:
        """Update stop loss order.
        
        Args:
            symbol: Trading pair symbol
            new_stop_loss: New stop loss level
            position_type: Position type (BUY/SELL)
            position_size: Current position size
        """
        try:
            # Get current price
            current_price = await self.binance_service.get_current_price(symbol)
            if not current_price:
                logger.error(f"Failed to get current price for {symbol}")
                return

            # Check minimum distance
            min_distance = current_price * self.config['risk_management']['min_stop_distance']
            if position_type == "BUY":
                if current_price - new_stop_loss < min_distance:
                    logger.warning(f"Stop loss too close to current price for {symbol}. Adjusting...")
                    new_stop_loss = current_price - min_distance
            else:  # SELL
                if new_stop_loss - current_price < min_distance:
                    logger.warning(f"Stop loss too close to current price for {symbol}. Adjusting...")
                    new_stop_loss = current_price + min_distance

            # Cancel existing stop loss
            await self.binance_service.cancel_all_orders(symbol)
            
            # Place new stop loss
            await self.binance_service.place_order(
                symbol=symbol,
                side="SELL" if position_type == "BUY" else "BUY",
                order_type="STOP_MARKET",
                quantity=abs(position_size),
                stop_price=new_stop_loss
            )
            
            # Send notification
            await self.notification_service.send_message(
                f"Updated trailing stop for {symbol} to {new_stop_loss}"
            )
            
        except Exception as e:
            logger.error(f"Error updating stop loss for {symbol}: {str(e)}")
            
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
            position = await self.binance_service.get_position(symbol)
            
            if position and float(position.get('info').get('positionAmt', 0)) != 0:
                # We have an existing position, manage it
                await self._manage_existing_position(symbol, position)
            else:
                # No existing position, check if we should open a new one
                if not self.telegram_service.is_trading_paused():
                    await self._execute_trade(symbol, signals)
                else:
                    logger.info(f"Trading paused - Skipping new trade for {symbol}")
                    
        except Exception as e:
            logger.error(f"Error processing trading signals: {str(e)}")
            
    async def _execute_trade(self, symbol: str, signal: Dict) -> None:
        """Execute a trade based on the signal."""
        try:
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
            current_position = await self.binance_service.get_position(symbol)
            if current_position:
                # Cancel existing SL/TP orders
                open_orders = await self.binance_service.get_open_orders(symbol)
                if open_orders:
                    for order in open_orders:
                        if order['type'] in ['STOP_MARKET', 'TAKE_PROFIT_MARKET']:
                            await self.binance_service.cancel_order(symbol, order['id'])
                            logger.info(f"Cancelled existing {order['type']} order for {symbol}")

                # Calculate new total position size
                current_size = float(current_position.get('info').get('positionAmt', 0))
                new_total_size = current_size + position_size if signal['position_type'].upper() == 'LONG' else current_size - position_size
                
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
                'side': 'buy' if signal['position_type'].upper() == 'LONG' else 'sell',
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

    def _get_support_resistance(self, market_conditions: Dict) -> Optional[Dict]:
        """Identify support and resistance levels from price data.
        
        Args:
            market_conditions: Dictionary containing market data
            
        Returns:
            Optional[Dict]: Dictionary with support and resistance levels
        """
        try:
            if not market_conditions or 'df' not in market_conditions:
                return None
                
            df = market_conditions['df']
            if df.empty:
                return None
                
            # Get recent price data
            recent_data = df.tail(50)  # Look at last 50 candles
            
            # Find local minima and maxima
            local_min = []
            local_max = []
            
            for i in range(1, len(recent_data)-1):
                if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and 
                    recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1]):
                    local_min.append(recent_data['low'].iloc[i])
                if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and 
                    recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1]):
                    local_max.append(recent_data['high'].iloc[i])
                    
            # Get current price
            current_price = recent_data['close'].iloc[-1]
            
            # Find nearest support and resistance
            support = max([p for p in local_min if p < current_price], default=None)
            resistance = min([p for p in local_max if p > current_price], default=None)
            
            if support is None or resistance is None:
                return None
                
            return {
                'support': support,
                'resistance': resistance
            }
            
        except Exception as e:
            logger.error(f"Error identifying support/resistance levels: {str(e)}")
            return None

    def _analyze_trend_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze market trend structure with enhanced flexibility."""
        try:
            if df is None or df.empty:
                logger.warning("Empty or None DataFrame provided for trend analysis")
                return {
                    'has_clear_structure': False,
                    'trend_type': 'unknown',
                    'strength': 0.0,
                    'support_levels': [],
                    'resistance_levels': []
                }

            # Calculate key levels
            high = df['high'].max()
            low = df['low'].min()
            current_price = df['close'].iloc[-1]
            
            # Calculate trend strength using multiple EMAs
            ema_8 = df['close'].ewm(span=8, adjust=False).mean()
            ema_21 = df['close'].ewm(span=21, adjust=False).mean()
            ema_55 = df['close'].ewm(span=55, adjust=False).mean()
            
            # Calculate trend alignment score
            trend_alignment = 0
            if ema_8.iloc[-1] > ema_21.iloc[-1] > ema_55.iloc[-1]:
                trend_alignment = 1  # Strong uptrend
            elif ema_8.iloc[-1] < ema_21.iloc[-1] < ema_55.iloc[-1]:
                trend_alignment = -1  # Strong downtrend
            
            # Calculate trend strength using multiple factors
            price_distance = abs(ema_8.iloc[-1] - ema_55.iloc[-1]) / current_price
            ema_angles = [
                (ema_8.iloc[-1] - ema_8.iloc[-5]) / ema_8.iloc[-5],
                (ema_21.iloc[-1] - ema_21.iloc[-5]) / ema_21.iloc[-5],
                (ema_55.iloc[-1] - ema_55.iloc[-5]) / ema_55.iloc[-5]
            ]
            trend_strength = (price_distance + abs(sum(ema_angles))) / 2
            
            # Determine trend type with more nuance
            if trend_alignment > 0:
                trend_type = 'strong_uptrend' if trend_strength > 0.02 else 'weak_uptrend'
            elif trend_alignment < 0:
                trend_type = 'strong_downtrend' if trend_strength > 0.02 else 'weak_downtrend'
            else:
                # Check for potential reversal or consolidation
                recent_trend = 'up' if ema_8.iloc[-1] > ema_8.iloc[-5] else 'down'
                trend_type = f'consolidation_{recent_trend}_bias'
                
            # Find support and resistance levels with volume confirmation
            support_levels = []
            resistance_levels = []
            
            # Use volume profile for level confirmation
            volume_profile = df.groupby(pd.cut(df['close'], bins=20))['volume'].sum()
            high_volume_prices = volume_profile[volume_profile > volume_profile.mean()].index
            
            # Use Fibonacci levels as potential support/resistance
            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            for level in fib_levels:
                price_level = low + (high - low) * level
                # Check if level is near high volume area
                for vol_price in high_volume_prices:
                    if abs(price_level - vol_price.mid) / current_price < 0.01:
                        if price_level < current_price:
                            support_levels.append(price_level)
                        else:
                            resistance_levels.append(price_level)
                    
            # Add recent swing highs and lows
            window = 20
            df['swing_high'] = df['high'].rolling(window=window, center=True).apply(
                lambda x: x[len(x)//2] == max(x)
            )
            df['swing_low'] = df['low'].rolling(window=window, center=True).apply(
                lambda x: x[len(x)//2] == min(x)
            )
            
            recent_swing_highs = df[df['swing_high']]['high'].dropna()
            recent_swing_lows = df[df['swing_low']]['low'].dropna()
            
            for high_price in recent_swing_highs:
                if high_price > current_price:
                    resistance_levels.append(high_price)
                    
            for low_price in recent_swing_lows:
                if low_price < current_price:
                    support_levels.append(low_price)
                    
            # Remove duplicates and sort
            support_levels = sorted(list(set(support_levels)))
            resistance_levels = sorted(list(set(resistance_levels)))
            
            # Determine if structure is clear with more flexible conditions
            has_clear_structure = (
                len(support_levels) >= 2 and  # At least 2 support levels
                len(resistance_levels) >= 2 and  # At least 2 resistance levels
                trend_strength > 0.005 and  # Reduced from 0.01
                abs(trend_alignment) >= 0.5  # Allow for developing trends
            )
            
            return {
                'has_clear_structure': has_clear_structure,
                'trend_type': trend_type,
                'strength': trend_strength,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'trend_alignment': trend_alignment
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend structure: {str(e)}")
            return {
                'has_clear_structure': False,
                'trend_type': 'unknown',
                'strength': 0.0,
                'support_levels': [],
                'resistance_levels': [],
                'trend_alignment': 0
            }

    async def select_trading_pairs(self, timeframe: str = '1h') -> List[str]:
        """
        Select trading pairs based on enhanced trend analysis and market conditions.
        
        Args:
            timeframe: The timeframe to analyze, default is 1h
            
        Returns:
            List of selected trading pairs
        """
        try:
            selected_pairs = []
            all_pairs = await self.exchange_service.get_trading_pairs()
            
            for symbol in all_pairs:
                try:
                    # Get historical data with increased lookback
                    df = await self.indicator_service.get_historical_data(symbol, timeframe, limit=100)
                    if df is None or df.empty:
                        continue
                        
                    # Calculate indicators
                    adx = self.indicator_service.calculate_adx(df)
                    volume_ma = df['volume'].rolling(window=20).mean()
                    current_volume = df['volume'].iloc[-1]
                    
                    # Analyze trend structure
                    trend_analysis = self._analyze_trend_structure(df)
                    
                    # Enhanced selection criteria
                    meets_criteria = (
                        trend_analysis['has_clear_structure'] and
                        adx.iloc[-1] > 15 and  # Reduced from 20
                        current_volume > volume_ma.iloc[-1] * 1.2 and  # 20% above average volume
                        trend_analysis['strength'] > 0.01  # Minimum trend strength
                    )
                    
                    if meets_criteria:
                        # Calculate selection score
                        score = (
                            trend_analysis['strength'] * 0.4 +  # Weight trend strength
                            (adx.iloc[-1] / 100) * 0.3 +  # Weight ADX
                            (current_volume / volume_ma.iloc[-1]) * 0.3  # Weight volume
                        )
                        
                        selected_pairs.append({
                            'symbol': symbol,
                            'score': score,
                            'trend_type': trend_analysis['trend_type'],
                            'strength': trend_analysis['strength'],
                            'adx': adx.iloc[-1],
                            'volume_ratio': current_volume / volume_ma.iloc[-1]
                        })
                        
                except Exception as e:
                    logger.warning(f"Error analyzing {symbol}: {str(e)}")
                    continue
            
            # Sort pairs by score and take top 10
            selected_pairs.sort(key=lambda x: x['score'], reverse=True)
            top_pairs = selected_pairs[:10]
            
            # Log selection results
            for pair in top_pairs:
                logger.info(
                    f"Selected {pair['symbol']}: "
                    f"Score={pair['score']:.2f}, "
                    f"Trend={pair['trend_type']}, "
                    f"Strength={pair['strength']:.2f}, "
                    f"ADX={pair['adx']:.2f}, "
                    f"Volume Ratio={pair['volume_ratio']:.2f}"
                )
            
            return [pair['symbol'] for pair in top_pairs]
            
        except Exception as e:
            logger.error(f"Error in select_trading_pairs: {str(e)}")
            return []

    async def _update_take_profit(self, symbol: str, new_take_profit: float,
                               position_type: str, position_size: float) -> None:
        """Update take profit order.
        
        Args:
            symbol: Trading pair symbol
            new_take_profit: New take profit level
            position_type: Position type (BUY/SELL)
            position_size: Current position size
        """
        try:
            # Get current price
            current_price = await self.binance_service.get_current_price(symbol)
            if not current_price:
                logger.error(f"Failed to get current price for {symbol}")
                return

            # Check minimum distance
            min_distance = current_price * self.config['risk_management']['min_take_profit_distance']
            if position_type == "BUY":
                if new_take_profit - current_price < min_distance:
                    logger.warning(f"Take profit too close to current price for {symbol}. Adjusting...")
                    new_take_profit = current_price + min_distance
            else:  # SELL
                if current_price - new_take_profit < min_distance:
                    logger.warning(f"Take profit too close to current price for {symbol}. Adjusting...")
                    new_take_profit = current_price - min_distance

            # Cancel existing take profit
            await self.binance_service.cancel_all_orders(symbol)
            
            # Place new take profit
            await self.binance_service.place_order(
                symbol=symbol,
                side="SELL" if position_type == "BUY" else "BUY",
                order_type="TAKE_PROFIT_MARKET",
                quantity=abs(position_size),
                stop_price=new_take_profit
            )
            
            # Send notification
            await self.notification_service.send_message(
                f"Updated take profit for {symbol} to {new_take_profit}"
            )
            
        except Exception as e:
            logger.error(f"Error updating take profit for {symbol}: {str(e)}")

    async def _check_order_frequency(self, symbol: str) -> bool:
        """Check if we can place a new order based on frequency limits.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            bool: True if we can place a new order
        """
        try:
            current_time = time.time()
            
            # Initialize order history if not exists
            if symbol not in self._order_history:
                self._order_history[symbol] = []
            
            # Remove orders older than 1 hour
            self._order_history[symbol] = [
                order_time for order_time in self._order_history[symbol]
                if current_time - order_time < 3600
            ]
            
            # Check if we have reached max orders per hour
            if len(self._order_history[symbol]) >= self._max_orders_per_hour:
                logger.warning(f"Maximum orders per hour reached for {symbol}")
                return False
            
            # Check minimum interval since last order
            if self._order_history[symbol]:
                last_order_time = self._order_history[symbol][-1]
                if current_time - last_order_time < self._min_order_interval:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking order frequency: {str(e)}")
            return False

    async def _check_risk_accumulation(self, symbol: str, new_position_size: float) -> bool:
        """Check if adding new position would exceed risk limits.
        
        Args:
            symbol: Trading pair symbol
            new_position_size: Size of new position to add
            
        Returns:
            bool: True if risk limit is not exceeded
        """
        try:
            # Get current positions
            positions = await self.binance_service.get_positions()
            if not positions:
                return True
                
            # Calculate current risk for symbol
            current_risk = 0
            for position in positions:
                if position['symbol'] == symbol:
                    position_amt = float(position.get('info', {}).get('positionAmt', 0))
                    entry_price = float(position.get('entryPrice', 0))
                    current_price = float(position.get('markPrice', 0))
                    
                    # Calculate risk as percentage of account
                    risk = abs(position_amt * (current_price - entry_price))
                    current_risk += risk
                    
            # Calculate new risk
            new_risk = new_position_size * self.config['risk_management']['max_risk_per_trade']
            
            # Check if total risk would exceed limit
            if current_risk + new_risk > self._max_risk_per_symbol:
                logger.warning(f"Risk limit would be exceeded for {symbol}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk accumulation: {str(e)}")
            return False

    async def _manage_existing_position(self, symbol: str, position: Dict) -> None:
        """Manage an existing position.
        
        Args:
            symbol: Trading pair symbol
            position: Current position details
        """
        try:
            # Check if we should close the position
            if await self.should_close_position(position):
                await self.binance_service.close_position(symbol)
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
                        position_type=position['positionSide'],
                        position_size=abs(float(position['info']['positionAmt']))
                    )
                    
                    # Update take profit
                    await self._update_take_profit(
                        symbol=symbol,
                        new_take_profit=new_stops['take_profit'],
                        position_type=position['positionSide'],
                        position_size=abs(float(position['info']['positionAmt']))
                    )
                    
            # Check for DCA opportunity
            dca_result = await self._handle_dca(symbol, position)
            if dca_result:
                # DCA was executed, update position tracking
                self._last_dca_time[symbol] = time.time()
                self._dca_history[symbol] = dca_result.get('dca_history', [])
                
            # Update trailing stop if needed
            if await self._update_trailing_stop(symbol, position['positionSide']):
                # Trailing stop was updated, no need for further action
                return
                
        except Exception as e:
            logger.error(f"Error managing existing position for {symbol}: {str(e)}")
