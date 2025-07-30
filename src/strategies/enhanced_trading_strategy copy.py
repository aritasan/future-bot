"""
Enhanced trading strategy implementation.
"""

import logging
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import time
import asyncio
from datetime import datetime
import json
import os

from src.services.indicator_service import IndicatorService
from src.services.sentiment_service import SentimentService
from src.services.binance_service import BinanceService
from src.services.telegram_service import TelegramService
from src.services.notification_service import NotificationService
from src.utils.helpers import is_long_side, is_short_side, is_trending_down,\
    is_trending_up

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
        
        
        # Trade tracking variables
        self._last_trade_time = {}
        self._last_trade_price = {}
        self._last_trade_side = {}
        self._last_trade_amount = {}
        self._last_trade_stop_loss = {}
        self._last_trade_take_profit = {}
        self._position_entry_time = {}  # Track entry time for each position
        
        # Thêm các tham số cấu hình mới
        self.base_thresholds = {
            'funding_rate': 0.01,
            'open_interest': 1000000,
            'volume_ratio': 1.0,
            'max_correlation': 0.8,
            'max_portfolio_exposure': 0.5
        }
        
        self.base_weights = {
            'trend': 0.15,
            'volume': 0.1,
            'volatility': 0.1,
            'correlation': 0.1,
            'sentiment': 0.1,
            'structure': 0.15,
            'volume_profile': 0.1,
            'funding_rate': 0.05,
            'open_interest': 0.05,
            'order_book': 0.1
        }
        
        self.adjustment_factors = {
            'volatility_threshold': 0.05,
            'weight_adjustment': 0.05,
            'score_adjustment': 0.2
        }
        
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
                
            return await self._adjust_position_size_by_volatility(symbol, position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return None

    async def generate_signals(self, symbol: str, indicator_service: IndicatorService) -> Optional[Dict]:
        """Generate trading signals for a symbol."""
        try:
            # Get data for multiple timeframes
            timeframes = ['5m', '15m', '1h', '4h']
            timeframe_data = {}
            for tf in timeframes:
                df = await indicator_service.calculate_indicators(symbol, timeframe=tf)
                if df is not None and not df.empty:
                    timeframe_data[tf] = df

            if not timeframe_data:
                logger.warning(f"Failed to get data for any timeframe for {symbol}")
                return None

            # Analyze market structure
            market_structure = self._analyze_market_structure(timeframe_data)
            
            # Analyze volume profile
            volume_profile = await self._analyze_volume_profile(symbol, timeframe='1h')
            
            # Get funding rate
            funding_rate = await self.binance_service.get_funding_rate(symbol)
            
            # Get open interest
            open_interest = await self.binance_service.get_open_interest(symbol)
            
            # Analyze order book
            order_book = await self.binance_service.get_order_book(symbol)
            
            # Calculate indicators for main timeframe
            df = timeframe_data['5m']
            
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

            # Check for trend following and breakout signals
            market_conditions = {
                'df': df,
                'timeframe_data': timeframe_data,
                'market_structure': market_structure,
                'volume_profile': volume_profile,
                'funding_rate': funding_rate,
                'open_interest': open_interest,
                'order_book': order_book,
                'timeframe_analysis': timeframe_analysis,
                'btc_volatility': btc_volatility,
                'altcoin_correlation': altcoin_correlation,
                'sentiment': sentiment
            }

            # Check trend following signal
            trend_signal = self._check_trend_following_signal(symbol, market_conditions)
            if trend_signal:
                logger.info(f"Trend following signal detected for {symbol}: {trend_signal}")
                signal_score = trend_signal['confidence'] * 100  # Convert confidence to score
                position_type = "LONG" if is_long_side(trend_signal['side']) else "SHORT"
            else:
                # Check breakout signal
                breakout_signal = self._check_breakout_signal(symbol, market_conditions)
                if breakout_signal:
                    logger.info(f"Breakout signal detected for {symbol}: {breakout_signal}")
                    signal_score = breakout_signal['confidence'] * 100  # Convert confidence to score
                    position_type = "LONG" if is_long_side(breakout_signal['side']) else "SHORT"
                else:
                    # Calculate signal score with new components
                    signal_score = await self.calculate_signal_score(
                        symbol=symbol,
                        df=df,
                        timeframe_analysis=timeframe_analysis,
                        btc_volatility=btc_volatility,
                        altcoin_correlation=altcoin_correlation,
                        sentiment=sentiment,
                        market_structure=market_structure,
                        volume_profile=volume_profile,
                        funding_rate=funding_rate,
                        open_interest=open_interest,
                        order_book=order_book
                    )
                    
                    # Determine position type based on signal score
                    if signal_score >= self.config['trading']['signal_thresholds']['long_entry']:
                        position_type = "LONG"
                        logger.info(f"Signal score {signal_score:.2f} above threshold for {symbol}")
                    elif signal_score <= self.config['trading']['signal_thresholds']['short_entry']:
                        position_type = "SHORT"
                        logger.info(f"Signal score {signal_score:.2f} below threshold for {symbol}")
                    else:
                        logger.info(f"Signal score {signal_score:.2f} below threshold for {symbol}")
                        return None

            # Check additional entry conditions
            if not await self._check_entry_conditions(
                symbol= symbol,
                df=df,
                volume_profile=volume_profile,
                funding_rate=funding_rate,
                open_interest=open_interest,
                order_book=order_book,
                position_type=position_type
            ):
                logger.info(f"Entry conditions not met for {symbol}")
                return None

            # Log signal details
            logger.info(f"{symbol} Signal score: {signal_score}")
            logger.info(f"Generated {position_type} signal for {symbol} with score {signal_score:.2f}")
            logger.info(f"Signal components for {symbol}:")
            logger.info(f"- Trend: {timeframe_analysis.get('trend', 'NEUTRAL')}")
            logger.info(f"- Volume: {'OK' if self.check_volume_condition(df) else 'LOW'}")
            logger.info(f"- Volatility: {btc_volatility.get('volatility_level', 'LOW')}")
            logger.info(f"- Correlation: {altcoin_correlation.get('correlation', 0):.2f}")
            logger.info(f"- Sentiment: {sentiment.get('sentiment', 0):.2f}")
            logger.info(f"- Market Structure: {market_structure.get('structure', 'NEUTRAL')}")
            logger.info(f"- Funding Rate: {funding_rate:.4f}")
            logger.info(f"- Open Interest: {open_interest}")

            # Prepare signal response
            signal = {
                'symbol': symbol,
                'position_type': position_type,
                'score': signal_score,
                'price': df['close'].iloc[-1],
                'timestamp': datetime.now().isoformat(),
                'analysis': {
                    'trend': timeframe_analysis,
                    'volume': self.check_volume_condition(df),
                    'volatility': btc_volatility,
                    'correlation': altcoin_correlation,
                    'sentiment': sentiment,
                    'market_structure': market_structure,
                    'volume_profile': volume_profile,
                    'funding_rate': funding_rate,
                    'open_interest': open_interest,
                    'order_book': order_book
                }
            }

            # Add trend following or breakout signal details if available
            if trend_signal:
                signal['signal_type'] = 'trend_following'
                signal['stop_loss'] = trend_signal['stop_loss']
                signal['take_profit'] = trend_signal['take_profit']
                signal['conditions'] = trend_signal['conditions']
            elif breakout_signal:
                signal['signal_type'] = 'breakout'
                signal['stop_loss'] = breakout_signal['stop_loss']
                signal['take_profit'] = breakout_signal['take_profit']
                signal['conditions'] = breakout_signal['conditions']

            return signal

        except Exception as e:
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            logger.error(f"Error generating signals for {symbol}: {str(e)}")
            return None

    def _analyze_market_structure(self, timeframe_data: Dict) -> Dict:
        """Analyze market structure for multiple timeframes.
        
        Args:
            timeframe_data: Dictionary containing data for different timeframes
            
        Returns:
            Dict: Market structure analysis results
        """
        try:
            if not timeframe_data:
                logger.error("No timeframe data provided")
                return {}
                
            structure_analysis = {}
            
            for timeframe, df in timeframe_data.items():
                if df is None or df.empty:
                    logger.warning(f"No data available for timeframe {timeframe}")
                    continue
                    
                try:
                    # Calculate support and resistance levels
                    levels = self._calculate_support_resistance(df)
                    
                    # Get current price
                    current_price = df['close'].iloc[-1]
                    
                    # Determine market structure
                    if levels['nearest_support'] is not None and levels['nearest_resistance'] is not None:
                        # Calculate distance to nearest levels
                        support_distance = (current_price - levels['nearest_support']) / current_price
                        resistance_distance = (levels['nearest_resistance'] - current_price) / current_price
                        
                        # Determine structure type
                        if support_distance < 0.01:  # Near support
                            structure = 'support'
                        elif resistance_distance < 0.01:  # Near resistance
                            structure = 'resistance'
                        else:
                            structure = 'neutral'
                            
                        structure_analysis[timeframe] = {
                            'structure': structure,
                            'support_levels': levels['support_levels'],
                            'resistance_levels': levels['resistance_levels'],
                            'nearest_support': levels['nearest_support'],
                            'nearest_resistance': levels['nearest_resistance'],
                            'current_price': current_price
                        }
                except Exception as e:
                    logger.error(f"Error analyzing market structure for timeframe {timeframe}: {str(e)}")
                    continue
                    
            return structure_analysis
            
        except Exception as e:
            logger.error(f"Error in market structure analysis: {str(e)}")
            return {}
            
    async def _analyze_volume_profile(self, symbol: str, timeframe: str = '1h') -> Dict:
        """Analyze volume profile for a symbol"""
        try:
            # Get historical klines
            klines = await self.binance_service.get_klines(
                symbol=symbol,
                timeframe=timeframe,
                limit=100
            )
            
            if not klines:
                logger.warning(f"No klines data available for {symbol}")
                return {}
                
            # Convert to DataFrame with correct columns
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            
            # Get current price
            current_price = await self.binance_service.get_current_price(symbol)
            if not current_price:
                current_price = df['close'].iloc[-1]
            
            # Calculate volume profile
            price_range = df['high'].max() - df['low'].min()
            num_bins = 10
            bin_size = price_range / num_bins
            
            # Create price bins
            bins = np.linspace(df['low'].min(), df['high'].max(), num_bins + 1)
            df['price_bin'] = pd.cut(df['close'], bins=bins)
            
            # Calculate volume per bin with observed=True to avoid warning
            volume_profile = df.groupby('price_bin', observed=True)['volume'].sum()
            
            # Convert volume profile to dict with string keys
            volume_profile_dict = {}
            for bin_price, volume in volume_profile.items():
                if isinstance(bin_price, pd.Interval):
                    key = f"{bin_price.left:.8f}_{bin_price.right:.8f}"
                    volume_profile_dict[key] = float(volume)
            
            # Find POC (Point of Control)
            poc_price = volume_profile.idxmax()
            if isinstance(poc_price, pd.Interval):
                poc_price = (poc_price.left + poc_price.right) / 2
            
            # Calculate value area
            total_volume = volume_profile.sum()
            value_area_volume = total_volume * 0.68  # 68% of total volume
            
            # Sort bins by volume
            sorted_bins = volume_profile.sort_values(ascending=False)
            cumulative_volume = 0
            value_area_bins = []
            
            for bin_price, volume in sorted_bins.items():
                cumulative_volume += volume
                value_area_bins.append(bin_price)
                if cumulative_volume >= value_area_volume:
                    break
            
            # Calculate value area boundaries
            if value_area_bins:
                value_area_high = max(v_bin.right for v_bin in value_area_bins if isinstance(v_bin, pd.Interval))
                value_area_low = min(v_bin.left for v_bin in value_area_bins if isinstance(v_bin, pd.Interval))
            else:
                value_area_high = df['high'].max()
                value_area_low = df['low'].min()
            
            # Identify high volume nodes (bins with volume > 1.5x average)
            avg_volume = volume_profile.mean()
            high_volume_nodes = []
            for bin_price, volume in volume_profile.items():
                if volume > avg_volume * 1.5:
                    if isinstance(bin_price, pd.Interval):
                        node_price = (bin_price.left + bin_price.right) / 2
                        high_volume_nodes.append({
                            'price': float(node_price),
                            'volume': float(volume),
                            'strength': float(volume / avg_volume)
                        })
            
            return {
                'poc_price': float(poc_price),
                'value_area_high': float(value_area_high),
                'value_area_low': float(value_area_low),
                'volume_profile': volume_profile_dict,
                'high_volume_nodes': high_volume_nodes,
                'current_price': float(current_price)
            }
            
        except Exception as e:
            logger.error(f"Error in volume profile analysis: {str(e)}")
            return {}

    async def _check_entry_conditions(self, symbol: str, df: pd.DataFrame, 
                          volume_profile: Dict, funding_rate: float,
                          open_interest: float, order_book: Dict,
                          position_type: str) -> bool:
        """Check if entry conditions are met based on market structure and other factors."""
        try:
            # Calculate dynamic thresholds
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            volatility = df['close'].pct_change().std()
            current_price = df['close'].iloc[-1]
            
            # 1. Check trend confirmation
            trend = self.get_trend(df)
            if (is_long_side(position_type) and trend != 'UP') or \
               (is_short_side(position_type) and trend != 'DOWN'):
                logger.info(f"{symbol} Trend confirmation not met for {position_type}: {trend}")
                return False
                
            # 2. Check RSI conditions
            indicators = await self.indicator_service.calculate_indicators(symbol, timeframe='5m')
            if indicators is not None and not indicators.empty:
                rsi = indicators['RSI'].iloc[-1] if 'RSI' in indicators.columns else 50
                if is_long_side(position_type):
                    if rsi > 70:  # Overbought
                        logger.info(f"{symbol} RSI overbought for {position_type}: {rsi}")
                        return False
                else:
                    if rsi < 30:  # Oversold
                        logger.info(f"{symbol} RSI oversold for {position_type}: {rsi}")
                        return False
                    
            # 3. Check volume profile with enhanced conditions
            if not self._check_volume_profile_conditions(volume_profile, position_type, min_volume_ratio=0.2):
                logger.info(f"{symbol} Volume profile conditions not met for {position_type}")
                return False
                
            # 4. Check order book with dynamic conditions
            if not self._check_order_book_conditions(order_book, position_type, min_depth=avg_volume * 0.1):
                logger.info(f"{symbol} Order book conditions not met for {position_type}")
                return False
                
            # 5. Check candlestick patterns with enhanced scoring
            pattern_score = self._check_candlestick_patterns(df, position_type)
            if is_short_side(position_type):
                if pattern_score > -0.5:  # Require at least 50% bearish confirmation
                    logger.info(f"{symbol} Candlestick patterns not bearish enough for {position_type}: {pattern_score}")
                    return False
            else:
                if pattern_score < 0.5:  # Require at least 50% bullish confirmation
                    logger.info(f"{symbol} Candlestick patterns not bullish enough for {position_type}: {pattern_score}")
                    return False
                    
            # 6. Check funding rate with dynamic threshold
            funding_threshold = 0.0002 * (1 + volatility)  # Increase threshold with volatility
            if is_long_side(position_type) and funding_rate > funding_threshold:
                logger.info(f"{symbol} Funding rate too high for {position_type}: {funding_rate}")
                return False
            elif is_short_side(position_type) and funding_rate < -funding_threshold:
                logger.info(f"{symbol} Funding rate too low for {position_type}: {funding_rate}")
                return False
                
            # 7. Check open interest with dynamic thresholds
            min_oi = max(50000, avg_volume * 0.1)  # Base minimum open interest
            
            # Check if open interest is valid
            if open_interest is None or open_interest <= 0:
                logger.info(f"{symbol} Invalid open interest value: {open_interest}")
                return False
                
            # Calculate position-specific threshold (20% higher than base)
            min_oi_position = min_oi * 1.2
            if open_interest < min_oi_position:
                logger.info(f"{symbol} Open interest too low for {position_type} position: {open_interest} < {min_oi_position}")
                return False
                
            # 8. Check market structure
            support_resistance = self._calculate_support_resistance(df)
            
            if is_long_side(position_type):
                # Check if price is near support
                nearest_support = max([s for s in support_resistance['support_levels'] if s < current_price], default=0)
                if nearest_support and (current_price - nearest_support) / current_price > 0.02:  # Within 2% of support
                    logger.info(f"{symbol} Price not near support for {position_type}")
                    return False
            else:
                # Check if price is near resistance
                nearest_resistance = min([r for r in support_resistance['resistance_levels'] if r > current_price], default=float('inf'))
                if nearest_resistance and (nearest_resistance - current_price) / current_price > 0.02:  # Within 2% of resistance
                    logger.info(f"{symbol} Price not near resistance for {position_type}")
                    return False
                    
            # 9. Check value area
            value_area = self._calculate_value_area(df)
            if is_long_side(position_type):
                if current_price < value_area['value_area_low']:
                    logger.info(f"{symbol} Price below value area for {position_type} {value_area['value_area_low']}")
                    return False
            else:
                if current_price > value_area['value_area_high']:
                    logger.info(f"{symbol} Price above value area for {position_type} {value_area['value_area_high']}")
                    return False
                    
            # All conditions met
            logger.info(f"{symbol} All entry conditions met for {position_type}")
            return True
            
        except Exception as e:
            logger.error(f"{symbol} Error checking entry conditions: {str(e)}")
            return False

    async def _check_correlation_conditions(self, symbol: str, position_type: str) -> bool:
        """Check correlation conditions with existing positions."""
        try:
            # Lấy các vị thế hiện có
            current_positions = await self._get_current_positions()
            
            # Chỉ kiểm tra correlation với các vị thế cùng loại
            relevant_positions = [p for p in current_positions if p['position_side'] == position_type]
            
            # Tính correlation với các vị thế hiện có
            for position in relevant_positions:
                correlation = await self._calculate_correlation(symbol, position['symbol'])
                if abs(correlation) > self.base_thresholds['max_correlation']:
                    logger.warning(f"High correlation detected between {symbol} and {position['symbol']}: {correlation}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking correlation conditions: {str(e)}")
            return False

    async def _get_current_positions(self) -> List[Dict]:
        """Get current open positions."""
        try:
            return await self.binance_service.get_positions()
        except Exception as e:
            logger.error(f"Error getting current positions: {str(e)}")
            return []

    async def _calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols with timeout handling and caching."""
        try:
            # Check cache first
            cache_key = f"correlation_{symbol1}_{symbol2}"
            if cache_key in self._cache:
                cached_data, timestamp = self._cache[cache_key]
                if time.time() - timestamp < self._cache_ttl:
                    return cached_data
                    
            # Set timeout and retry parameters
            timeout = 5  # Reduced timeout to 5 seconds
            max_retries = 3
            retry_delay = 2  # Increased delay between retries
            
            for attempt in range(max_retries):
                try:
                    # Get historical data for both symbols with timeout handling
                    try:
                        df1 = await asyncio.wait_for(
                            self.indicator_service.get_historical_data(symbol1, timeframe="5m", limit=50),
                            timeout=timeout
                        )
                        df2 = await asyncio.wait_for(
                            self.indicator_service.get_historical_data(symbol2, timeframe="5m", limit=50),
                            timeout=timeout
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout while fetching historical data for correlation between {symbol1} and {symbol2}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                        return 0.0
                    except Exception as e:
                        logger.error(f"Error fetching historical data for correlation between {symbol1} and {symbol2}: {str(e)}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                        return 0.0
                    
                    if df1 is None or df2 is None or df1.empty or df2.empty:
                        logger.warning(f"Missing historical data for correlation calculation between {symbol1} and {symbol2}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                        return 0.0
                        
                    # Ensure both DataFrames have the same length
                    min_length = min(len(df1), len(df2))
                    if min_length < 10:  # Minimum required data points
                        logger.warning(f"Insufficient data points for correlation calculation between {symbol1} and {symbol2}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                        return 0.0
                        
                    df1 = df1.tail(min_length)
                    df2 = df2.tail(min_length)
                    # Calculate correlation with timeout handling
                    try:
                        correlation = df1['close'].corr(df2['close'])
                        
                        if pd.isna(correlation):
                            logger.warning(f"Invalid correlation value between {symbol1} and {symbol2}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_delay)
                                continue
                            return 0.0
                            
                        # Cache the result
                        self._cache[cache_key] = (correlation, time.time())
                        return correlation
                        
                    except Exception as e:
                        logger.error(f"Error calculating correlation between {symbol1} and {symbol2}: {str(e)}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                        return 0.0
                        
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        logger.warning(f"Timeout while calculating correlation between {symbol1} and {symbol2}, retrying...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        logger.warning(f"Timeout while calculating correlation between {symbol1} and {symbol2} after {max_retries} attempts")
                        return 0.0
                except Exception as e:
                    logger.error(f"Error calculating correlation between {symbol1} and {symbol2}: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    return 0.0
                    
            return 0.0
            
        except Exception as e:
            logger.error(f"Unexpected error calculating correlation between {symbol1} and {symbol2}: {str(e)}")
            return 0.0

    def _check_volume_profile_conditions(self, volume_profile: Dict, position_type: str, min_volume_ratio: float = 0.3) -> bool:
        """Check volume profile conditions for entry.
        
        Args:
            volume_profile: Volume profile data
            position_type: Position type (LONG/SHORT)
            min_volume_ratio: Minimum volume ratio threshold
            
        Returns:
            bool: True if conditions are met, False otherwise
        """
        try:
            if not volume_profile:
                logger.error("No volume profile data available")
                return False
                
            # Get current price
            current_price = float(volume_profile.get('current_price', 0))
            if current_price <= 0:
                logger.error("Invalid current price in volume profile")
                return False
                
            # Get profile data
            profile_data = volume_profile.get('volume_profile', {})
            if not profile_data:
                logger.error("No profile data available")
                return False
                
            # Convert string price ranges to float using average of range
            converted_profile = {}
            for price_range, vol in profile_data.items():
                try:
                    # Split price range and convert to float
                    if '_' in price_range:
                        low, high = map(float, price_range.split('_'))
                        avg_price = (low + high) / 2
                        converted_profile[avg_price] = float(vol)
                    else:
                        # If single price, convert directly
                        converted_profile[float(price_range)] = float(vol)
                except (ValueError, TypeError) as e:
                    logger.error(f"Error converting price range {price_range}: {str(e)}")
                    continue
                    
            if not converted_profile:
                logger.error("No valid price data after conversion")
                return False
                
            # Calculate total volume
            total_volume = sum(converted_profile.values())
            if total_volume <= 0:
                logger.error("Invalid total volume")
                return False
                
            if is_long_side(position_type):
                # For LONG: check volume in lower price levels
                lower_volume = sum(vol for price, vol in converted_profile.items() 
                                 if price < current_price)
                volume_ratio = lower_volume / total_volume
                
                # Check if there's significant volume below current price
                if volume_ratio < min_volume_ratio:
                    logger.info(f"LONG: Insufficient volume below current price: {volume_ratio:.2%} < {min_volume_ratio:.2%}")
                    return False
                    
                # Check if POC is below current price
                poc = float(volume_profile.get('poc', 0))
                if poc >= current_price:
                    logger.info(f"LONG: POC {poc} is not below current price {current_price}")
                    return False
                    
            else:  # SHORT
                # For SHORT: check volume in higher price levels
                higher_volume = sum(vol for price, vol in converted_profile.items() 
                                  if price > current_price)
                volume_ratio = higher_volume / total_volume
                
                # Check if there's significant volume above current price
                if volume_ratio < min_volume_ratio:
                    logger.info(f"SHORT: Insufficient volume above current price: {volume_ratio:.2%} < {min_volume_ratio:.2%}")
                    return False
                    
                # Check if POC is above current price
                poc = float(volume_profile.get('poc', 0))
                if poc <= current_price:
                    logger.info(f"SHORT: POC {poc} is not above current price {current_price}")
                    return False
                    
            # Check value area conditions
            value_area = volume_profile.get('value_area', {})
            if value_area:
                upper = float(value_area.get('upper', 0))
                lower = float(value_area.get('lower', 0))
                
                if is_long_side(position_type):
                    # For LONG: check if current price is above value area
                    if current_price <= upper:
                        logger.info(f"LONG: Current price {current_price} is not above value area upper {upper}")
                        return False
                else:  # SHORT
                    # For SHORT: check if current price is below value area
                    if current_price >= lower:
                        logger.info(f"SHORT: Current price {current_price} is not below value area lower {lower}")
                        return False
                        
            logger.info(f"Volume profile conditions met for {position_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking volume profile conditions: {str(e)}")
            return False

    def _check_order_book_conditions(self, order_book: Dict, position_type: str, min_depth: float = 100000.0) -> bool:
        """Check order book conditions for a specific position type.
        
        Args:
            order_book: Dictionary containing order book data
            position_type: Type of position (LONG/SHORT)
            min_depth: Minimum required depth for the order book
            
        Returns:
            bool: True if conditions are met, False otherwise
        """
        try:
            if not order_book or 'bids' not in order_book or 'asks' not in order_book:
                logger.warning("Invalid order book data")
                return False
                
            # Get current price
            current_price = (float(order_book['bids'][0][0]) + float(order_book['asks'][0][0])) / 2
            
            if is_long_side(position_type):
                # For LONG: check depth on bid side
                bid_depth = sum(float(bid[1]) for bid in order_book['bids'])
                ask_depth = sum(float(ask[1]) for ask in order_book['asks'])
                
                # Check if bid depth meets minimum requirement
                if bid_depth < min_depth:
                    logger.info(f"Bid depth {bid_depth} below minimum requirement {min_depth} for LONG position")
                    return False
                    
                # Check bid/ask depth ratio
                if bid_depth / ask_depth < 1.2:
                    logger.info(f"Bid/ask depth ratio {bid_depth/ask_depth:.2f} below 1.2 for LONG position")
                    return False
                    
                # Check spread
                spread = (float(order_book['asks'][0][0]) - float(order_book['bids'][0][0])) / current_price
                if spread > 0.001:  # Spread too large
                    logger.info(f"Spread {spread:.4f} too large for LONG position")
                    return False
                    
            else:  # SHORT position
                # For SHORT: check depth on ask side
                bid_depth = sum(float(bid[1]) for bid in order_book['bids'])
                ask_depth = sum(float(ask[1]) for ask in order_book['asks'])
                
                # Check if ask depth meets minimum requirement
                if ask_depth < min_depth:
                    logger.info(f"Ask depth {ask_depth} below minimum requirement {min_depth} for SHORT position")
                    return False
                    
                # Check ask/bid depth ratio
                if ask_depth / bid_depth < 1.2:
                    logger.info(f"Ask/bid depth ratio {ask_depth/bid_depth:.2f} below 1.2 for SHORT position")
                    return False
                    
                # Check spread
                spread = (float(order_book['asks'][0][0]) - float(order_book['bids'][0][0])) / current_price
                if spread > 0.001:  # Spread too large
                    logger.info(f"Spread {spread:.4f} too large for SHORT position")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking order book conditions: {str(e)}")
            return False

    def _check_candlestick_patterns(self, df: pd.DataFrame, position_type: str) -> float:
        """Check candlestick patterns for entry signals.
        
        Args:
            df: DataFrame with OHLCV data
            position_type: Position type (LONG/SHORT)
            
        Returns:
            float: Score between -1 and 1 indicating pattern strength
                  Positive for LONG, negative for SHORT
        """
        try:
            if df is None or df.empty:
                return 0.0
                
            score = 0.0
            
            # Get last 3 candles for pattern analysis
            last_3_candles = df.tail(3)
            if len(last_3_candles) < 3:
                return 0.0
                
            last_candle = last_3_candles.iloc[-1]
            
            if is_long_side(position_type):
                # Check bullish patterns
                # 1. Hammer or Inverted Hammer
                body_size = abs(last_candle['close'] - last_candle['open'])
                lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
                upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
                
                is_hammer = (lower_wick > 2 * body_size and upper_wick < body_size)
                is_inverted_hammer = (upper_wick > 2 * body_size and lower_wick < body_size)
                
                if is_hammer or is_inverted_hammer:
                    score += 0.5
                    
                # 2. Check previous downtrend
                prev_2_candles = last_3_candles.iloc[:-1]
                if all(prev_2_candles['close'] < prev_2_candles['open']):
                    score += 0.5
                    
            else:  # SHORT position
                # Check bearish patterns
                # 1. Shooting Star or Hanging Man
                body_size = abs(last_candle['close'] - last_candle['open'])
                lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
                upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
                
                is_shooting_star = (upper_wick > 2 * body_size and lower_wick < body_size)
                is_hanging_man = (lower_wick > 2 * body_size and upper_wick < body_size)
                
                if is_shooting_star or is_hanging_man:
                    score += 0.5
                    
                # 2. Check previous uptrend
                prev_2_candles = last_3_candles.iloc[:-1]
                if all(prev_2_candles['close'] > prev_2_candles['open']):
                    score += 0.5
                    
                # Convert score to negative for SHORT positions
                score = -score
                    
            return score
            
        except Exception as e:
            logger.error(f"Error checking candlestick patterns: {str(e)}")
            return 0.0

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
                # For long positions, ensure stop loss is always positive
                k = stop_loss_multiplier
                while stop_loss <= 0 and k > 0:  # Add k > 0 check to avoid infinite loop
                    k = k * 0.8  # Reduce multiplier by 20% each time for smoother adjustment
                    stop_loss = float(current_price) - (float(atr) * k)
                    
                # Set minimum stop loss if still <= 0
                if stop_loss <= 0:
                    stop_loss = float(current_price) * 0.02  # Set to 2% of current price as minimum
            else:
                stop_loss = float(current_price) + (float(atr) * stop_loss_multiplier)
            
            # Get market conditions
            market_conditions = await self._get_market_conditions(symbol)
            
            # Adjust stop loss based on volatility
            volatility = market_conditions.get('volatility', 0.0)  # Default to 0.0 if not present
            if volatility > 0.02:  # Consider high volatility if > 2%
                # Increase stop loss distance in high volatility
                if is_long_side(position_type):
                    stop_loss = float(current_price) - (float(atr) * stop_loss_multiplier * 1.5)
                    k = stop_loss_multiplier
                    while stop_loss <= 0 and k > 0:  # Add k > 0 check to avoid infinite loop
                        k = k * 0.8  # Reduce multiplier by 20% each time for smoother adjustment
                        stop_loss = float(current_price) - (float(atr) * k)
                    
                    # Set minimum stop loss if still <= 0
                    if stop_loss <= 0:
                        stop_loss = float(current_price) * 0.01  # Set to 1% of current price as minimum
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
                
                # Additional validation for SHORT positions
                if take_profit <= 0:
                    # If take profit is negative, set it to a reasonable percentage below current price
                    take_profit = current_price * 0.5  # 50% below current price
                elif take_profit >= current_price:
                    # If take profit is above current price, set it to a reasonable percentage below
                    take_profit = current_price * 0.8  # 20% below current price
            
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
        try:
            symbol = position.get('symbol')
            position_side = position.get('info', {}).get('positionSide')
            position_amt = float(position.get('info', {}).get('positionAmt', 0))
            unrealized_pnl = float(position.get('unrealizedPnl', 0))
            if position_amt == 0:
                return False
                
            # Get current market data
            df = await self.indicator_service.get_historical_data(symbol, '1h', limit=100)
            if df is None or df.empty:
                return False
                
            # Check momentum-based exit
            if self._should_exit_by_momentum(df, position):
                logger.info(f"Closing position for {symbol} {position_side.upper()} due to momentum signal")
                await self.telegram_service.send_message(
                    f"Closing position for {symbol} {position_side.upper()} due to momentum signal\n"
                    f"Current price: {df['close'].iloc[-1]}\n"
                    f"Position size: {position_amt}\n"
                    f"Unrealized PnL: {unrealized_pnl}"
                )
                return True
                
            # Continue with existing exit conditions
            current_price = float(df['close'].iloc[-1])
            entry_price = float(position.get('entryPrice', 0))
            margin = float(position.get('initialMargin', 0))
            
            # Calculate price change
            price_change = (current_price - entry_price) / entry_price
            if position_amt < 0:  # For short positions, invert the price change
                price_change = -price_change
                
            # Check stop loss
            stop_loss = await self.binance_service.get_stop_price(symbol, position_side, 'STOP_MARKET')
            if stop_loss:
                if (position_amt > 0 and current_price <= stop_loss) or \
                   (position_amt < 0 and current_price >= stop_loss):
                    logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                    await self.telegram_service.send_message(
                        f"Stop loss triggered for {symbol} at {current_price}\n"
                        f"Current price: {current_price}\n"
                        f"Price change: {price_change:.2%}\n" 
                        f"Unrealized PnL: {unrealized_pnl}"
                    )
                    return True
                    
            # Check take profit
            take_profit = await self.binance_service.get_stop_price(symbol, position_side, 'TAKE_PROFIT_MARKET')
            if take_profit:
                if (position_amt > 0 and current_price >= take_profit) or \
                   (position_amt < 0 and current_price <= take_profit):
                    logger.info(f"Take profit triggered for {symbol} at {current_price}")
                    await self.telegram_service.send_message(
                        f"Take profit triggered for {symbol} at {current_price}\n"
                        f"Current price: {current_price}\n"
                        f"Price change: {price_change:.2%}\n" 
                        f"Unrealized PnL: {unrealized_pnl}"
                    )
                    return True
                    
            # Check if unrealized PnL is below -400% margin (emergency stop)
            if unrealized_pnl < -4*margin:
                logger.info(f"Emergency stop triggered for {symbol} {position_side.upper()} - PnL below -400% margin")
                await self.telegram_service.send_message(
                    f"Emergency stop triggered for {symbol} {position_side.upper()} - PnL below -400% margin\n"
                    f"Current price: {current_price}\n"
                    f"Price change: {price_change:.2%}\n" 
                    f"Unrealized PnL: {unrealized_pnl}"
                )
                return True
                
            # Check market conditions
            # TODO: Add market conditions check
            market_conditions = await self._get_market_conditions(symbol)
            if market_conditions:
                # Check if market volatility is too high
                if market_conditions.get('volatility') > 0.05 and abs(price_change) > 0.1:  # 10% move
                    logger.info(f"Closing position for {symbol} {position_side.upper()} due to high volatility")
                    await self.telegram_service.send_message(
                        f"Closing position for {symbol} {position_side.upper()} due to high volatility\n"
                        f"Current price: {current_price}\n"
                        f"Price change: {price_change:.2%}\n" 
                        f"Volatility: {market_conditions.get('volatility'):.2%}\n"
                        f"Unrealized PnL: {unrealized_pnl}"
                    )
                    return True
                    
                # Check if market trend has reversed
                trend = market_conditions.get('btc_trend')
                if trend:
                    # Get BTC volatility
                    btc_volatility = await self.analyze_btc_volatility()
                    if not btc_volatility:
                        logger.warning("Failed to get BTC volatility for correlation analysis")
                        return False
                        
                    # Get correlation analysis
                    correlation_analysis = await self.analyze_altcoin_correlation(symbol, btc_volatility)
                    
                #     result = {
                #     'correlation': float(adjusted_correlation),
                #     'returns_correlation': float(best_correlation),
                #     'reaction_strength': float(reaction_strength),
                #     'is_strongly_correlated': is_strongly_correlated,
                #     'is_reacting': is_reacting,
                #     'reaction': 'STRONG' if is_strongly_correlated else 'MODERATE' if is_reacting else 'WEAK',
                #     'data_points': len(corr_df),
                #     'btc_std': float(btc_std),
                #     'alt_std': float(alt_std),
                #     'lag': best_lag,
                #     'trend_alignment': float(trend_alignment),
                #     'btc_trend': btc_trend,
                #     'alt_trend': alt_trend
                # }
                    
                    if correlation_analysis:
                        correlation = correlation_analysis.get('correlation', 0)
                        reaction_strength = correlation_analysis.get('reaction_strength', 0)
                        trend_alignment = correlation_analysis.get('trend_alignment', 0)
                        btc_trend = correlation_analysis.get('btc_trend', 'NEUTRAL')
                        alt_trend = correlation_analysis.get('alt_trend', 'NEUTRAL')
                        
                        # Get trend strength
                        df = await self.indicator_service.calculate_indicators(symbol)
                        if df is not None and not df.empty:
                            adx = df['ADX'].iloc[-1] if 'ADX' in df.columns else 0
                            trend_strength = adx / 100.0
                            
                            # Define thresholds
                            CORRELATION_THRESHOLD = 0.7  # Strong correlation threshold
                            TREND_STRENGTH_THRESHOLD = 0.25  # Strong trend threshold
                            
                            # Check for trend reversal with correlation consideration
                            if position_amt > 0:  # LONG position
                                if (is_trending_down(trend) and 
                                    ((correlation > CORRELATION_THRESHOLD and trend_alignment < 0) or  # Strong negative correlation
                                     (trend_strength > TREND_STRENGTH_THRESHOLD and is_trending_down(btc_trend) and is_trending_down(alt_trend)))):  # Strong trend alignment
                                    logger.info(f"Closing LONG position for {symbol} due to BTC trend reversal and correlation")
                                    await self.telegram_service.send_message(
                                        f"Closing LONG position for {symbol} due to BTC trend reversal and correlation\n"
                                        f"Current price: {current_price}\n"
                                        f"Price change: {price_change:.2%}\n"
                                        f"Unrealized PnL: {unrealized_pnl}\n"
                                        f"BTC Trend: {btc_trend}\n"
                                        f"Alt Trend: {alt_trend}\n"
                                        f"Correlation: {correlation:.2f}\n"
                                        f"Reaction Strength: {reaction_strength:.2f}\n"
                                        f"Trend Alignment: {trend_alignment:.2f}\n"
                                        f"Trend Strength: {trend_strength:.2f}"
                                    )
                                    return True
                                    
                            elif position_amt < 0:  # SHORT position
                                if (is_trending_up(trend) and 
                                    ((correlation > CORRELATION_THRESHOLD and trend_alignment > 0) or  # Strong positive correlation
                                     (trend_strength > TREND_STRENGTH_THRESHOLD and is_trending_up(btc_trend) and is_trending_up(alt_trend)))):  # Strong trend alignment
                                    logger.info(f"Closing SHORT position for {symbol} due to BTC trend reversal and correlation")
                                    await self.telegram_service.send_message(
                                        f"Closing SHORT position for {symbol} due to BTC trend reversal and correlation\n"
                                        f"Current price: {current_price}\n"
                                        f"Price change: {price_change:.2%}\n"
                                        f"Unrealized PnL: {unrealized_pnl}\n"
                                        f"BTC Trend: {btc_trend}\n"
                                        f"Alt Trend: {alt_trend}\n"
                                        f"Correlation: {correlation:.2f}\n"
                                        f"Reaction Strength: {reaction_strength:.2f}\n"
                                        f"Trend Alignment: {trend_alignment:.2f}\n"
                                        f"Trend Strength: {trend_strength:.2f}"
                                    )
                                    return True
                                    
                            # Check for divergence between BTC and altcoin
                            if position_amt > 0:  # LONG position
                                if (is_trending_down(btc_trend) and is_trending_up(alt_trend) and 
                                    correlation > CORRELATION_THRESHOLD and 
                                    trend_strength > TREND_STRENGTH_THRESHOLD):
                                    logger.info(f"Closing LONG position for {symbol} due to trend divergence")
                                    await self.telegram_service.send_message(
                                        f"Closing LONG position for {symbol} due to trend divergence\n"
                                        f"Current price: {current_price}\n"
                                        f"Price change: {price_change:.2%}\n"
                                        f"Unrealized PnL: {unrealized_pnl}\n"
                                        f"BTC Trend: {btc_trend}\n"
                                        f"Alt Trend: {alt_trend}\n"
                                        f"Correlation: {correlation:.2f}\n"
                                        f"Trend Strength: {trend_strength:.2f}"
                                    )
                                    return True
                                    
                            elif position_amt < 0:  # SHORT position
                                if (is_trending_up(btc_trend) and is_trending_down(alt_trend) and 
                                    correlation > CORRELATION_THRESHOLD and 
                                    trend_strength > TREND_STRENGTH_THRESHOLD):
                                    logger.info(f"Closing SHORT position for {symbol} due to trend divergence")
                                    await self.telegram_service.send_message(
                                        f"Closing SHORT position for {symbol} due to trend divergence\n"
                                        f"Current price: {current_price}\n"
                                        f"Price change: {price_change:.2%}\n"
                                        f"Unrealized PnL: {unrealized_pnl}\n"
                                        f"BTC Trend: {btc_trend}\n"
                                        f"Alt Trend: {alt_trend}\n"
                                        f"Correlation: {correlation:.2f}\n"
                                        f"Trend Strength: {trend_strength:.2f}"
                                    )
                                    return True
                        
                # Check if market sentiment is strongly against position
                sentiment_analysis = await self.analyze_market_sentiment(symbol)
                if sentiment_analysis:
                    # Get individual sentiment components
                    rsi_sentiment = sentiment_analysis.get('rsi_sentiment', 'neutral')
                    mfi_sentiment = sentiment_analysis.get('mfi_sentiment', 'neutral')
                    obv_sentiment = sentiment_analysis.get('obv_sentiment', 'neutral')
                    trend_strength = sentiment_analysis.get('trend_strength', 'weak')
                    overall_sentiment = sentiment_analysis.get('overall_sentiment', 'neutral')
                    
                    # Calculate sentiment score
                    sentiment_score = self._calculate_sentiment_score(sentiment_analysis, position_side)
                    
                    # Check for strong sentiment reversal
                    sentiment_threshold = 0.6  # Threshold for strong sentiment
                    sentiment_duration = 3  # Number of candles to confirm sentiment
                    
                    # Get historical sentiment data
                    df = await self.indicator_service.calculate_indicators(symbol)
                    if df is not None and not df.empty:
                        # Calculate historical sentiment scores
                        historical_sentiments = []
                        for i in range(min(sentiment_duration, len(df))):
                            historical_data = {
                                'rsi_sentiment': 'bullish' if df['RSI'].iloc[-(i+1)] < 30 else 'bearish' if df['RSI'].iloc[-(i+1)] > 70 else 'neutral',
                                'mfi_sentiment': 'bullish' if df['MFI'].iloc[-(i+1)] < 20 else 'bearish' if df['MFI'].iloc[-(i+1)] > 80 else 'neutral',
                                'obv_sentiment': 'bullish' if df['OBV'].iloc[-(i+1)] > 0 else 'bearish' if df['OBV'].iloc[-(i+1)] < 0 else 'neutral',
                                'trend_strength': 'strong' if df['ADX'].iloc[-(i+1)] > 25 else 'weak',
                                'overall_sentiment': 'bullish' if (
                                    (df['RSI'].iloc[-(i+1)] < 30 and df['MFI'].iloc[-(i+1)] < 20) or
                                    (df['OBV'].iloc[-(i+1)] > 0 and df['ADX'].iloc[-(i+1)] > 25)
                                ) else 'bearish' if (
                                    (df['RSI'].iloc[-(i+1)] > 70 and df['MFI'].iloc[-(i+1)] > 80) or
                                    (df['OBV'].iloc[-(i+1)] < 0 and df['ADX'].iloc[-(i+1)] > 25)
                                ) else 'neutral'
                            }
                            historical_sentiments.append(self._calculate_sentiment_score(historical_data, position_side))
                        
                        # Check if sentiment has been consistently against position
                        if len(historical_sentiments) == sentiment_duration:
                            if position_amt > 0:  # LONG position
                                if all(score < -sentiment_threshold for score in historical_sentiments):
                                    logger.info(f"Closing LONG position for {symbol} due to strong bearish sentiment")
                                    await self.telegram_service.send_message(
                                        f"Closing LONG position for {symbol} due to strong bearish sentiment\n"
                                        f"Current price: {current_price}\n"
                                        f"Price change: {price_change:.2%}\n"
                                        f"Unrealized PnL: {unrealized_pnl}\n"
                                        f"Sentiment score: {sentiment_score:.2f}\n"
                                        f"RSI sentiment: {rsi_sentiment}\n"
                                        f"MFI sentiment: {mfi_sentiment}\n"
                                        f"OBV sentiment: {obv_sentiment}\n"
                                        f"Trend strength: {trend_strength}"
                                    )
                                    return True
                            elif position_amt < 0:  # SHORT position
                                if all(score > sentiment_threshold for score in historical_sentiments):
                                    logger.info(f"Closing SHORT position for {symbol} due to strong bullish sentiment")
                                    await self.telegram_service.send_message(
                                        f"Closing SHORT position for {symbol} due to strong bullish sentiment\n"
                                        f"Current price: {current_price}\n"
                                        f"Price change: {price_change:.2%}\n"
                                        f"Unrealized PnL: {unrealized_pnl}\n"
                                        f"Sentiment score: {sentiment_score:.2f}\n"
                                        f"RSI sentiment: {rsi_sentiment}\n"
                                        f"MFI sentiment: {mfi_sentiment}\n"
                                        f"OBV sentiment: {obv_sentiment}\n"
                                        f"Trend strength: {trend_strength}"
                                    )
                                    return True
                    
                    # Check for extreme sentiment conditions
                    if position_amt > 0:  # LONG position
                        if (rsi_sentiment == 'bearish' and mfi_sentiment == 'bearish' and 
                            obv_sentiment == 'bearish' and trend_strength == 'strong'):
                            logger.info(f"Closing LONG position for {symbol} due to extreme bearish sentiment")
                            await self.telegram_service.send_message(
                                f"Closing LONG position for {symbol} due to extreme bearish sentiment\n"
                                f"Current price: {current_price}\n"
                                f"Price change: {price_change:.2%}\n"
                                f"Unrealized PnL: {unrealized_pnl}\n"
                                f"RSI sentiment: {rsi_sentiment}\n"
                                f"MFI sentiment: {mfi_sentiment}\n"
                                f"OBV sentiment: {obv_sentiment}\n"
                                f"Trend strength: {trend_strength}"
                            )
                            return True
                    elif position_amt < 0:  # SHORT position
                        if (rsi_sentiment == 'bullish' and mfi_sentiment == 'bullish' and 
                            obv_sentiment == 'bullish' and trend_strength == 'strong'):
                            logger.info(f"Closing SHORT position for {symbol} due to extreme bullish sentiment")
                            await self.telegram_service.send_message(
                                f"Closing SHORT position for {symbol} due to extreme bullish sentiment\n"
                                f"Current price: {current_price}\n"
                                f"Price change: {price_change:.2%}\n"
                                f"Unrealized PnL: {unrealized_pnl}\n"
                                f"RSI sentiment: {rsi_sentiment}\n"
                                f"MFI sentiment: {mfi_sentiment}\n"
                                f"OBV sentiment: {obv_sentiment}\n"
                                f"Trend strength: {trend_strength}"
                            )
                            return True
            
            return False
            
        except Exception as e:
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
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
                
            # Ensure all numeric values are properly converted to float
            try:
                logger.info(f"position: {position}")
                logger.info(f"current_price: {current_price}")
                entry_price = float(position.get("entryPrice", 0))
                current_price = float(current_price)
                
                if entry_price <= 0:
                    logger.error(f"Invalid entry price for {position['symbol']}: {entry_price}")
                    return False
                    
                # Check if price moved significantly
                if is_long_side(position["side"]):
                    price_change = (current_price - entry_price) / entry_price
                    if price_change > 0.04:  # 4% move
                        # Update last update time
                        self.binance_service._set_cached_data(cache_key, current_time)
                        return True
                else:
                    price_change = (entry_price - current_price) / entry_price
                    if price_change > 0.04:  # 4% move
                        # Update last update time
                        self.binance_service._set_cached_data(cache_key, current_time)
                        return True
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting numeric values for {position['symbol']}: {str(e)}")
                return False
                    
            return False
            
        except Exception as e:
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
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
                
            # Ensure current_price is a float
            try:
                current_price = float(current_price)
            except (TypeError, ValueError):
                logger.error(f"Invalid current_price type: {type(current_price)}")
                return {}
                
            # Get market conditions
            market_conditions = await self._get_market_conditions(position['info']['symbol'])
            if not market_conditions:
                return {}
            
            # Calculate stop distance based on volatility
            volatility = market_conditions.get('volatility', 0.0)  # Default to 0.0 if not present
            base_stop_distance = self.config['risk_management']['base_stop_distance']
            
            # Adjust volatility multiplier based on volatility level
            if volatility > 0.02:  # High volatility > 2%
                volatility_multiplier = 1.5
            elif volatility > 0.01:  # Medium volatility > 1%
                volatility_multiplier = 1.0
            else:  # Low volatility <= 1%
                volatility_multiplier = 0.5
                
            stop_distance = base_stop_distance * volatility_multiplier
            
            # Adjust stop distance based on trend
            trend = market_conditions.get('btc_trend', 'UP')
            # If trend is in our favor, we can use a tighter stop
            if (position_amt > 0 and is_trending_up(trend)) or (position_amt < 0 and is_trending_down(trend)):
                trend_multiplier = 0.8  # Tighter stop in trend direction
            else:
                trend_multiplier = 1.2  # Wider stop against trend
                
            stop_distance *= trend_multiplier
            
            # Ensure minimum stop distance (at least 2% for Binance)
            min_stop_distance = 0.02  # 2% minimum distance
            stop_distance = max(stop_distance, min_stop_distance)
            
            # Calculate new stop loss and take profit
            take_profit_multiplier = self.config['risk_management'].get('take_profit_multiplier', 2.0)
            
            if position_amt > 0:  # Long position
                new_stop_loss = current_price * (1 - stop_distance)
                new_take_profit = current_price * (1 + (stop_distance * take_profit_multiplier))
            else:  # Short position
                new_stop_loss = current_price * (1 + stop_distance)
                new_take_profit = current_price * (1 - (stop_distance * take_profit_multiplier))
            
            # Validate the calculated prices
            if new_stop_loss <= 0 or new_take_profit <= 0:
                logger.error(f"Invalid calculated prices - Stop Loss: {new_stop_loss}, Take Profit: {new_take_profit}")
                return {}
                
            # For short positions, ensure stop loss is above current price and take profit is below
            if position_amt < 0:
                if new_stop_loss <= current_price * 1.02:  # Ensure at least 2% above current price
                    new_stop_loss = current_price * 1.02
                if new_take_profit >= current_price:
                    new_take_profit = current_price * 0.98  # Set take profit 2% below current price
            # For long positions, ensure stop loss is below current price and take profit is above
            else:
                if new_stop_loss >= current_price * 0.98:  # Ensure at least 2% below current price
                    new_stop_loss = current_price * 0.98
                if new_take_profit <= current_price:
                    new_take_profit = current_price * 1.02  # Set take profit 2% above current price
            
            # Round prices to appropriate decimal places
            price_precision = self._get_price_precision(position['info']['symbol'])
            new_stop_loss = round(new_stop_loss, price_precision)
            new_take_profit = round(new_take_profit, price_precision)
            
            logger.info(f"Calculated stops for {position['info']['symbol']} - Current Price: {current_price}, Stop Loss: {new_stop_loss}, Take Profit: {new_take_profit}")
            
            return {
                'stop_loss': new_stop_loss,
                'take_profit': new_take_profit
            }
            
        except Exception as e:
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            logger.error(f"Error calculating new stops: {str(e)}")
            return {}
            
    def _get_price_precision(self, symbol: str) -> int:
        """Get price precision for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            int: Number of decimal places for price
        """
        try:
            # Get market info
            market = self.binance_service.exchange.markets.get(symbol)
            if not market:
                return 8  # Default precision
                
            # Get price precision from market info
            precision = market.get('precision', {}).get('price', 8)
            return precision
            
        except Exception as e:
            logger.error(f"Error getting price precision for {symbol}: {str(e)}")
            return 8  # Default precision

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
                volatility = market_conditions.get('volatility', 0.0)
                if volatility > 0.05:  # High volatility > 5%
                    atr_multiplier *= 1.5
                elif volatility < 0.02:  # Low volatility < 2%
                    atr_multiplier *= 0.8

                # Adjust for trend
                trend = market_conditions.get('btc_trend', 'UP')
                if (is_long_side(position_type) and is_trending_up(trend)) or \
                   (is_short_side(position_type) and is_trending_down(trend)):
                    atr_multiplier *= 0.9  # Tighter stop in trend direction
                else:
                    atr_multiplier *= 1.1  # Wider stop against trend

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
                                   sentiment: Dict, market_structure: Dict,
                                   volume_profile: Dict, funding_rate: float,
                                   open_interest: float, order_book: Dict) -> float:
        """Calculate signal score with dynamic weights and logging."""
        try:
            # Xác định position type dựa trên trend
            position_type = 'LONG' if is_trending_up(self.get_trend(df)) else 'SHORT'
            
            # Lấy trọng số động
            weights = await self._get_dynamic_weights(position_type)
            
            # Tính điểm cho từng yếu tố
            scores = {
                'trend': self._calculate_timeframe_score(timeframe_analysis, position_type),
                'volume': self._calculate_volume_score(df, position_type),
                'volatility': self._calculate_volatility_score(btc_volatility, position_type),
                'correlation': self._calculate_correlation_score(altcoin_correlation, position_type),
                'sentiment': self._calculate_sentiment_score(sentiment, position_type),
                'structure': self._calculate_structure_score(market_structure, position_type),
                'volume_profile': self._calculate_volume_profile_score(volume_profile, position_type),
                'funding_rate': self._calculate_funding_rate_score(funding_rate, position_type),
                'open_interest': self._calculate_open_interest_score(open_interest, position_type),
                'order_book': self._calculate_order_book_score(order_book, position_type)
            }
            
            # Tính điểm tổng hợp
            total_score = sum(score * weights[factor] for factor, score in scores.items())
            
            # Log kết quả phân tích
            await self._log_signal_analysis(symbol, position_type, scores, total_score, {
                'trend': timeframe_analysis,
                'volume': df['volume'].iloc[-1],
                'volatility': btc_volatility,
                'correlation': altcoin_correlation,
                'sentiment': sentiment,
                'structure': market_structure,
                'volume_profile': volume_profile,
                'funding_rate': funding_rate,
                'open_interest': open_interest,
                'order_book': order_book
            })
            
            return total_score
            
        except Exception as e:
            logger.error(f"Error calculating signal score for {symbol}: {str(e)}")
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
            
            # Calculate average volatility from historical ATR
            average_volatility_1m = btc_data_1m['ATR'].mean() / current_price_1m
            average_volatility_5m = btc_data_5m['ATR'].mean() / current_price_5m
            average_volatility = (average_volatility_1m * 0.3) + (average_volatility_5m * 0.7)
            
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
                'trend': trend,
                'average_volatility': average_volatility
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
            logger.info(f"_handle_dca: Starting DCA process for {symbol}")
            
            # Get current price and position details
            current_price = await self.binance_service.get_current_price(symbol)
            if not current_price:
                logger.error(f"_handle_dca: Failed to get current price for {symbol}")
                return None
                
            position_type = position.get('side', 'LONG')
            entry_price = float(position.get('entryPrice', 0))
            
            # Calculate price drop
            price_drop = (current_price - entry_price) / entry_price * 100
                
            # Get market conditions
            market_conditions = await self._get_market_conditions(symbol)
            if not market_conditions:
                logger.error(f"_handle_dca: Failed to get market conditions for {symbol}")
                return None
                
            # Check DCA limits
            if not self._check_dca_limits(symbol, position_type):
                logger.info(f"_handle_dca: DCA limits reached for {symbol}")
                return None
                
            # Check if DCA is favorable
            if not self._is_dca_favorable(symbol, price_drop, market_conditions, position_type):
                logger.info(f"_handle_dca: DCA conditions not favorable for {symbol} {position_type}")
                return None
                
            # Calculate DCA size
            current_size = abs(float(position.get('info', {}).get('positionAmt', 0)))
            dca_size = await self._calculate_dca_size(symbol, current_size, price_drop, position_type)
            if not dca_size:
                logger.error(f"_handle_dca: Failed to calculate DCA size for {symbol}")
                return None
                
            logger.info(f"_handle_dca: Calculated DCA size {dca_size} for {symbol}")
            
            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'side': 'BUY' if is_long_side(position_type) else 'SELL',
                'type': 'MARKET',
                'amount': dca_size
            }
            
            # Place DCA order
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"_handle_dca: Attempting to place DCA order for {symbol} (attempt {attempt + 1})")
                    order = await self.binance_service.place_order(order_params)
                    if not order:
                        logger.error(f"_handle_dca: Failed to place DCA order for {symbol} - no order returned")
                        continue
                        
                    logger.info(f"_handle_dca: DCA order placed successfully for {symbol}: {order.get('id')}")
                    
                    # Calculate new stop loss and take profit
                    logger.info(f"_handle_dca: Calculating new stop loss and take profit for {symbol}")
                    atr = await self.indicator_service.calculate_atr(symbol)
                    if not atr:
                        logger.error(f"_handle_dca: Failed to calculate ATR for {symbol}")
                        
                    new_stop_loss = await self._calculate_stop_loss(symbol, position_type, current_price, atr)
                    new_take_profit = await self._calculate_take_profit(symbol, position_type, current_price, new_stop_loss)
                        
                    logger.info(f"_handle_dca: New stop loss: {new_stop_loss}, take profit: {new_take_profit} for {symbol}")
                    
                    # Update stop loss
                    logger.info(f"_handle_dca: Updating stop loss for {symbol} to {new_stop_loss}")
                    if await self._update_stop_loss(symbol, new_stop_loss, position_type):
                        logger.info(f"_handle_dca: Stop loss updated successfully for {symbol}")
                    else:
                        logger.error(f"_handle_dca: Failed to update stop loss for {symbol}")
                        
                    # Update take profit
                    logger.info(f"_handle_dca: Updating take profit for {symbol} to {new_take_profit}")
                    if await self._update_take_profit(symbol, new_take_profit, position_type):
                        logger.info(f"_handle_dca: Take profit updated successfully for {symbol}")
                    else:
                        logger.error(f"_handle_dca: Failed to update take profit for {symbol}")
                    
                    # Send notification
                    logger.info(f"_handle_dca: Sending DCA notification for {symbol}")
                    try:
                        await self.telegram_service.send_dca_notification({
                            'symbol': symbol,
                            'dca_amount': dca_size,
                            'new_entry_price': current_price,
                            'price_drop': price_drop,
                            'order_id': order.get('id', 'N/A'),
                            'position_type': position_type
                        })
                        logger.info(f"_handle_dca: DCA notification sent successfully for {symbol}")
                    except Exception as e:
                        logger.error(f"_handle_dca: Failed to send DCA notification for {symbol}: {str(e)}")
                    
                    # Create DCA history entry
                    dca_history_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'order_id': order.get('id'),
                        'dca_amount': dca_size,
                        'entry_price': current_price,
                        'price_drop': price_drop,
                        'position_type': position_type,
                        'stop_loss': new_stop_loss,
                        'take_profit': new_take_profit,
                        'market_conditions': {
                            'volatility': market_conditions.get('volatility', 0),
                            'trend': market_conditions.get('btc_trend', 'NEUTRAL'),
                            'volume': market_conditions.get('volume', 0)
                        }
                    }
                    
                    # Update DCA history
                    if symbol not in self._dca_history:
                        self._dca_history[symbol] = []
                    self._dca_history[symbol].append(dca_history_entry)
                    
                    return {
                        'order_id': order.get('id'),
                        'dca_amount': dca_size,
                        'new_entry_price': current_price,
                        'price_drop': price_drop,
                        'position_type': position_type,
                        'dca_history': self._dca_history[symbol]
                    }
                    
                except Exception as e:
                    logger.error(f"_handle_dca: Error in DCA process for {symbol} (attempt {attempt + 1}): {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                    else:
                        logger.error(f"_handle_dca: All DCA attempts failed for {symbol}")
                        return None
                        
            return None
            
        except Exception as e:
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            logger.error(f"_handle_dca: Unexpected error in DCA process for {symbol}: {str(e)}")
            return None

    def _is_dca_favorable(self, symbol: str, price_change: float, market_conditions: Dict, position_type: str) -> bool:
        """Check if DCA conditions are favorable.
        
        Args:
            symbol: Symbol of the position
            price_change: Price change percentage from entry (positive for increase, negative for decrease)
            market_conditions: Current market conditions
            position_type: Position type (LONG/SHORT)
            
        Returns:
            bool: True if DCA conditions are favorable, False otherwise
        """
        try:
            # Get current market conditions
            volatility = market_conditions.get('volatility', 0)
            trend = market_conditions.get('btc_trend', 'NEUTRAL')
            volume = market_conditions.get('volume', 0)
            
            # Check if market conditions are favorable
            if volatility > 0.05:  # High volatility
                logger.info(f"DCA not favorable: High volatility {volatility}")
                return False
                
            # Check trend conditions
            if is_trending_down(trend) and is_long_side(position_type):
                logger.info(f"{symbol} DCA not favorable: Bearish trend for LONG position")
                return False
                
            if is_trending_up(trend) and is_short_side(position_type):
                logger.info(f"{symbol} DCA not favorable: Bullish trend for SHORT position")
                return False
                
            # Define minimum price change thresholds for DCA
            MIN_PRICE_CHANGE = 4  # 4% minimum price change
            MAX_PRICE_CHANGE = 10  # 10% maximum price change for DCA
            
            if is_long_side(position_type):
                # For LONG positions, we want price to drop by at least MIN_PRICE_CHANGE
                # but not more than MAX_PRICE_CHANGE
                if price_change > -MIN_PRICE_CHANGE:
                    logger.info(f"{symbol} DCA not favorable for LONG: Price drop {price_change:.2%} less than minimum {-MIN_PRICE_CHANGE:.2%}")
                    return False
                if price_change < -MAX_PRICE_CHANGE:
                    logger.info(f"{symbol} DCA not favorable for LONG: Price drop {price_change:.2%} exceeds maximum {-MAX_PRICE_CHANGE:.2%}")
                    return False
                    
            elif is_short_side(position_type):
                # For SHORT positions, we want price to increase by at least MIN_PRICE_CHANGE
                # but not more than MAX_PRICE_CHANGE
                if price_change < MIN_PRICE_CHANGE:
                    logger.info(f"{symbol} DCA not favorable for SHORT: Price increase {price_change:.2%} less than minimum {MIN_PRICE_CHANGE:.2%}")
                    return False
                if price_change > MAX_PRICE_CHANGE:
                    logger.info(f"{symbol} DCA not favorable for SHORT: Price increase {price_change:.2%} exceeds maximum {MAX_PRICE_CHANGE:.2%}")
                    return False
            
            logger.info(f"{symbol} DCA conditions favorable for {position_type} with price change {price_change:.2%}")
            return True
            
        except Exception as e:
            logger.error(f"{symbol} Error checking DCA conditions: {str(e)}")
            return False

    async def _calculate_dca_size(self, symbol: str, current_size: float, price_change: float, position_type: str) -> Optional[float]:
        """Calculate DCA size based on current position size and price change."""
        try:
            # Get DCA configuration
            dca_config = self.config.get('risk_management', {}).get('dca', {})
            
            # Calculate base DCA size
            base_dca_size = current_size * float(dca_config.get('dca_size_multiplier', 0.5))
            
            # Adjust DCA size based on price change
            price_change_factor = abs(price_change) / 100  # Convert percentage to decimal
            dca_size = base_dca_size * (1 + price_change_factor)
            
            # Get current price and position
            current_price = await self.binance_service.get_current_price(symbol)
            position = await self.binance_service.get_position(symbol)
            
            if not current_price or not position:
                logger.error(f"{symbol} Failed to get current price or position for DCA calculation")
                return None
                
            # Calculate max DCA size based on current position value
            current_position_size = float(position.get('info', {}).get('positionAmt', 0))
            current_position_value = abs(current_position_size * current_price)
            max_dca_percentage = float(dca_config.get('max_dca_percentage', 0.5))  # Default 50% of current position
            max_dca_size = current_position_value * max_dca_percentage / current_price
            
            # Apply minimum and maximum constraints
            min_dca_size = float(dca_config.get('min_dca_size', 0.001))  # Minimum size in base currency
            max_dca_size = min(max_dca_size, float(dca_config.get('max_dca_size_multiplier', 2.0)) * current_position_size) # Maximum size in base currency
            
            if dca_size < min_dca_size:
                logger.info(f"{symbol} DCA size {dca_size} below minimum {min_dca_size}")
                dca_size = min_dca_size
            elif dca_size > max_dca_size:
                logger.info(f"{symbol} DCA size {dca_size} exceeds maximum {max_dca_size}")
                dca_size = max_dca_size
            
            logger.info(f"{symbol} Calculated DCA size {dca_size} for {position_type} with price change {price_change:.2%}")
            return dca_size
            
        except Exception as e:
            logger.error(f"{symbol} Error calculating DCA size: {str(e)}")
            return None

    async def _get_market_conditions(self, symbol: str = None) -> Dict:
        """Get current market conditions.
        
        Args:
            symbol: Optional symbol to get specific market conditions
            
        Returns:
            Dict: Market conditions or empty dict if error
        """
        try:
            # Get BTC volatility
            btc_volatility = await self.analyze_btc_volatility()
            if not btc_volatility:
                logger.error("Failed to get BTC volatility")
                return {}
                
            # Get market trend
            if symbol:
                trend = await self._get_market_trend(symbol)
            else:
                trend = 'NEUTRAL'
                
            # Get market sentiment
            if symbol:
                sentiment = await self._get_market_sentiment(symbol)
            else:
                sentiment = 0.0
                
            # Get market liquidity
            liquidity = await self._calculate_market_liquidity()
            
            # Get risk environment
            risk_on = await self._is_risk_on_environment()
            
            return {
                'btc_trend': trend,
                'volatility': btc_volatility.get('volatility', 0.0),
                'sentiment': sentiment,
                'liquidity': liquidity,
                'risk_on': risk_on
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
                    logger.info(f"Skipping trailing stop update for {symbol} because it was updated recently")
                    return
                    
            position_side = "LONG" if is_long_side(position_type) else "SHORT"
            # Get position details
            position = await self.binance_service.get_position(symbol, position_side)
            if not position:
                logger.warning(f"No active {position_side} position found for {symbol}")
                return

            # Get current market data
            current_price = float(position['markPrice'])
            entry_price = float(position['entryPrice'])
            unrealized_pnl = float(position.get('unrealizedPnl', 0))
            position_size = float(position.get('info', {}).get('positionAmt', 0))
            
            # Only proceed if we have unrealized profit
            if unrealized_pnl <= 0:
                logger.warning(f"No unrealized profit for {symbol} {position_side}")
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
            current_stop_loss = await self.binance_service.get_stop_price(symbol, position_type, 'STOP_MARKET')
            logger.info(f"_update_trailing_stop: Current stop loss for {symbol}: {current_stop_loss}")

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
                if (not current_stop_loss or new_stop_loss > current_stop_loss) and new_stop_loss < current_price:
                    await self._update_stop_loss(symbol, new_stop_loss, position_type)
                    logger.info(f"_update_trailing_stop: Updated trailing stop for {symbol} LONG to {new_stop_loss}")
                    # logger.info(f"Updated trailing stop for {symbol} LONG to {new_stop_loss}")
            else:
                # For SHORT positions, only move stop loss down
                if (not current_stop_loss or new_stop_loss < current_stop_loss) and new_stop_loss > current_price:
                    await self._update_stop_loss(symbol, new_stop_loss, position_type)
                    logger.info(f"_update_trailing_stop: Updated trailing stop for {symbol} SHORT to {new_stop_loss}")
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
            
    def _should_emergency_stop(self, analysis_market_conditions: Dict) -> bool:
        """Check if emergency stop should be triggered.
        
        Args:
            market_conditions: Dictionary containing market analysis data
            
        Returns:
            bool: True if emergency stop should be triggered
        """
        volatility = analysis_market_conditions.get('volatility', 0.0)
        volume_ratio = analysis_market_conditions.get('volume_ratio', 1.0)
        
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
                              position_type: str) -> bool:
        """
        Update stop loss for a position.
        
        Args:
            symbol: Trading pair symbol
            new_stop_loss: New stop loss price
            position_type: Position type (BUY/SELL/LONG/SHORT)
        Returns:
            bool: True if stop loss updated successfully, False otherwise
        """
        try:
            # Convert position_type to position_side for HEDGE mode
            position_side = "LONG" if is_long_side(position_type) else "SHORT"
            
            # Get current position
            position = await self.binance_service.get_position(symbol, position_side)
            if not position or float(position.get('contracts', 0)) == 0:
                logger.warning(f"No active {position_side} position found for {symbol}")
                return False
                
            current_stop_loss = await self.binance_service.get_stop_price(symbol, position_type, 'STOP_MARKET')
            logger.info(f"_update_stop_loss: Current stop loss for {symbol}: {current_stop_loss}")

            if not (is_long_side(position_type) and (not current_stop_loss or new_stop_loss > current_stop_loss * 1.02)):
                logger.info(f"_update_stop_loss: New stop loss for {symbol} LONG: {new_stop_loss} to minimium 2% = {current_stop_loss * 1.02}")
                new_stop_loss = current_stop_loss * 1.02
            
            if not (is_short_side(position_type) and (not current_stop_loss or new_stop_loss < current_stop_loss * 0.98)):
                logger.info(f"_update_stop_loss: New stop loss for {symbol} SHORT: {new_stop_loss} to minimium 2% = {current_stop_loss * 0.98}")
                new_stop_loss = current_stop_loss * 0.98
            
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
                    position_size=float(position.get('info', {}).get('positionAmt', 0)),
                    entry_price=float(position.get('entryPrice', 0)),
                    stop_price=new_stop_loss,
                    pnl_usd=float(position.get('unrealizedPnl', 0))
                )
                return True
            else:
                logger.error(f"Failed to update stop loss for {symbol} {position_side}")
                return False
        except Exception as e:
            logger.error(f"Error updating stop loss for {symbol}: {str(e)}")
            return False

    async def _update_take_profit(self, symbol: str, new_take_profit: float,
                               position_type: str) -> bool:
        """
        Update take profit for a position.
        
        Args:
            symbol: Trading pair symbol
            new_take_profit: New take profit price
            position_type: Position type (BUY/SELL/LONG/SHORT)
        Returns:
            bool: True if take profit updated successfully, False otherwise
        """
        try:
            # Convert position_type to position_side for HEDGE mode
            position_side = "LONG" if is_long_side(position_type) else "SHORT"
            
            # Get current position
            position = await self.binance_service.get_position(symbol, position_side)
            if not position or float(position.get('contracts', 0)) == 0:
                logger.warning(f"No active {position_side} position found for {symbol}")
                return False
                
            current_take_profit = await self.binance_service.get_stop_price(symbol, position_type, 'TAKE_PROFIT_MARKET')
            logger.info(f"_update_take_profit: Current take profit for {symbol}: {current_take_profit}")
            
            if not (is_long_side(position_type) and (not current_take_profit or new_take_profit > current_take_profit * 1.02)):
                logger.info(f"_update_take_profit: New take profit for {symbol} LONG: {new_take_profit} to minimium 2% = {current_take_profit * 1.02}")
                new_take_profit = current_take_profit * 1.02
                
            if not (is_short_side(position_type) and (not current_take_profit or new_take_profit < current_take_profit * 0.98)):
                logger.info(f"_update_take_profit: New take profit for {symbol} SHORT: {new_take_profit} to minimium 2% = {current_take_profit * 0.98}")
                new_take_profit = current_take_profit * 0.98

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
                    pnl_usd=float(position.get('unrealizedPnl', 0))
                )
                return True
            else:
                logger.error(f"Failed to update take profit for {symbol} {position_side}")
                return False
        except Exception as e:
            logger.error(f"Error updating take profit for {symbol}: {str(e)}")
            return False
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
                            logger.info(f"_monitor_positions: Closing position for {symbol} with side {position['info']['positionSide']}")
                            if await self.binance_service.close_position(symbol, position['info']['positionSide']):
                                logger.info(f"_monitor_positions: Position closed for {symbol}")
                                await self.telegram_service.send_message(
                                    f"Position closed for {symbol} {position['info']['positionSide']}\n"
                                    f"Amount: {position['info']['positionAmt']}\n"
                                    f"PnL: {position['unrealizedPnl']}")
                            continue
                            
                        # Check if we should update stops
                        current_price = await self.binance_service.get_current_price(symbol)
                        if current_price and await self.should_update_stops(position, current_price):
                            new_stops = await self.calculate_new_stops(position, current_price)
                            if new_stops:
                                position_side = position.get('side', 'LONG')
                                if 'stop_loss' in new_stops:
                                    unrealized_pnl = float(position.get('unrealizedPnl', 0))
                                    margin = float(position.get('initialMargin', 0))
                                    if unrealized_pnl > margin:
                                        if await self._update_stop_loss(symbol, new_stops['stop_loss'], position_side):
                                            logger.info(f"_monitor_positions: Updated stop loss for {symbol} {position_side} to {new_stops['stop_loss']}")
                                        else:
                                            logger.error(f"_monitor_positions: Failed to update stop loss for {symbol} {position_side}")
                                if 'take_profit' in new_stops:
                                    if await self._update_take_profit(symbol, new_stops['take_profit'], position_side):
                                        logger.info(f"_monitor_positions: Updated take profit for {symbol} {position_side} to {new_stops['take_profit']}")
                                    else:
                                        logger.error(f"_monitor_positions: Failed to update take profit for {symbol} {position_side}")
                                    
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
                            
                        await self._handle_dca(symbol, position)
                        
                        
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
                        # Get position details
                        unrealized_pnl = float(position.get('unrealizedPnl', 0))
                        margin = float(position.get('initialMargin', 0))
                        
                        # Only update trailing stop if PnL >= 100% margin
                        if unrealized_pnl < margin:
                            continue
                            
                        # Determine position side based on position amount
                        position_amt = float(position.get('info').get('positionAmt', 0))
                        position_side = 'LONG' if position_amt > 0 else 'SHORT'
                        logger.info(f"_monitor_trailing_stops: Updating trailing stop for {symbol} with side {position_side}")
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
                logger.info(f"_manage_existing_position: Closing position for {symbol} with side {position['info']['positionSide']}")
                # Close the position
                if await self.binance_service.close_position(symbol, position['info']['positionSide']):
                    logger.info(f"_manage_existing_position: Position closed for {symbol}")
                    await self.telegram_service.send_message(
                        f"Position closed for {symbol} {position['info']['positionSide']}\n"
                        f"Amount: {position['info']['positionAmt']}\n"
                        f"PnL: {position['unrealizedPnl']}")
                return
                
            # Get position details
            unrealized_pnl = float(position.get('unrealizedPnl', 0))
            margin = float(position.get('initialMargin', 0))
            
            # Only proceed with management if PnL >= 100% margin
            if unrealized_pnl < margin:
                return
                
            # Check if we should update stop loss and take profit
            if await self.should_update_stops(position, position['info']['markPrice']):
                # Calculate new stops
                new_stops = await self.calculate_new_stops(position, position['info']['markPrice'])
                if new_stops:
                    # Update stop loss and take profit
                    logger.info(f"_manage_existing_position: Updating stop loss for {symbol} to {new_stops['stop_loss']}")
                    if await self._update_stop_loss(
                        symbol=symbol,
                        new_stop_loss=new_stops['stop_loss'],
                        position_type=position['info']['positionSide']
                    ):
                        logger.info(f"_manage_existing_position: Stop loss updated for {symbol} {position['info']['positionSide']}")
                    else:
                        logger.error(f"_manage_existing_position: Failed to update stop loss for {symbol} {position['info']['positionSide']}")
                    
                    if await self._update_take_profit(
                        symbol=symbol,
                        new_take_profit=new_stops['take_profit'],
                        position_type=position['info']['positionSide']
                    ):
                        logger.info(f"_manage_existing_position: Take profit updated for {symbol} {position['info']['positionSide']}")
                    else:
                        logger.error(f"_manage_existing_position: Failed to update take profit for {symbol} {position['info']['positionSide']}")
                    
                    
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
                
    def _calculate_volume_score(self, df: pd.DataFrame, position_type: str) -> float:
        """Calculate volume score based on volume conditions and position type."""
        try:
            if df is None or df.empty:
                logger.warning("Empty DataFrame provided for volume score calculation")
                return 0.0
                
            # Calculate volume metrics
            volume_ma = df["volume"].rolling(window=20).mean()
            volume_std = df["volume"].rolling(window=20).std()
            volume_ratio = df["volume"].iloc[-1] / volume_ma.iloc[-1]
            
            # Calculate volume trend (3-period)
            volume_trend = (df["volume"].iloc[-1] / df["volume"].iloc[-3]) - 1
            
            if is_long_side(position_type):
                # For LONG: higher volume is better, but need to consider trend
                if volume_ratio >= 2.0 and volume_trend > 0:  # Very high volume with uptrend
                    return 1.0
                elif volume_ratio >= 1.5 and volume_trend > 0:  # High volume with uptrend
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
            elif is_short_side(position_type):  # SHORT
                # For SHORT: moderate volume is better, avoid extreme low/high
                if 0.8 <= volume_ratio <= 1.2:  # Moderate volume
                    return -1.0  # Negative score for SHORT
                elif 0.6 <= volume_ratio <= 1.4:  # Slightly below/above moderate
                    return -0.8
                elif 0.4 <= volume_ratio <= 1.6:  # Below/above moderate
                    return -0.6
                elif 0.3 <= volume_ratio <= 2.0:  # Low/high volume
                    return -0.4
                elif 0.2 <= volume_ratio <= 2.5:  # Very low/high volume
                    return -0.2
                else:  # Extreme volume
                    return 0.4  # Positive score for extreme volume (unfavorable for SHORT)
                
        except Exception as e:
            logger.error(f"Error calculating volume score: {str(e)}")
            return 0.0

    def _calculate_volatility_score(self, btc_volatility: Dict, position_type: str) -> float:
        """Calculate volatility score based on BTC volatility and position type.
        
        Args:
            btc_volatility: Dictionary containing volatility data with fields:
                - volatility_score: Overall volatility score
                - volatility_score_1m: 1-minute timeframe volatility score
                - volatility_score_5m: 5-minute timeframe volatility score
                - volatility_level: Current volatility level
                - is_volatile: Boolean indicating if market is volatile
                - is_accelerating: Boolean indicating if volatility is accelerating
                - atr_1m: ATR value for 1-minute timeframe
                - atr_5m: ATR value for 5-minute timeframe
                - roc_1m: Rate of Change for 1-minute timeframe
                - roc_5m: Rate of Change for 5-minute timeframe
                - ema: EMA value
                - current_price: Current BTC price
                - trend: Current market trend
            position_type: Position type (LONG/SHORT)
            
        Returns:
            float: Volatility score between -1 and 1
        """
        try:
            if not btc_volatility:
                return 0.0
                
            # Get base volatility score
            volatility_score = btc_volatility.get('volatility_score', 0.0)
            is_volatile = btc_volatility.get('is_volatile', False)
            is_accelerating = btc_volatility.get('is_accelerating', False)
            
            if is_long_side(position_type):
                # For LONG: lower volatility is better
                if not is_volatile:
                    base_score = 1.0
                else:
                    # Calculate decreasing score as volatility increases
                    base_score = max(0, 1 - volatility_score)
                    
                # Adjust score based on volatility acceleration
                if is_accelerating:
                    base_score *= 0.5  # Reduce score if volatility is accelerating
                    
                return base_score
                
            elif is_short_side(position_type):  # SHORT
                # For SHORT: higher volatility is better
                if is_volatile:
                    base_score = -1.0  # Negative score for SHORT
                else:
                    # Calculate increasing negative score as volatility decreases
                    base_score = -max(0, volatility_score)
                    
                # Adjust score based on volatility acceleration
                if is_accelerating:
                    base_score *= 1.5  # Increase negative score if volatility is accelerating
                    
                return base_score
                
        except Exception as e:
            logger.error(f"Error calculating volatility score: {str(e)}")
            return 0.0
            
    def _calculate_correlation_score(self, altcoin_correlation: Dict, position_type: str) -> float:
        """Calculate correlation score based on altcoin correlation and position type."""
        try:
            if not altcoin_correlation or 'correlation' not in altcoin_correlation:
                return 0.0
                
            correlation = altcoin_correlation['correlation']
            avg_correlation = altcoin_correlation.get('avg_correlation', correlation)
            
            # Handle edge cases
            if correlation is None or avg_correlation is None:
                return 0.0
                
            # Handle division by zero
            if avg_correlation == 0:
                return 0.0
                
            if is_long_side(position_type):
                # For LONG: higher correlation is better
                if correlation > avg_correlation:
                    return 1.0
                else:
                    # Calculate decreasing score as correlation decreases
                    correlation_ratio = correlation / avg_correlation
                    return max(0, min(1, correlation_ratio))  # Ensure score is between 0 and 1
                    
            elif is_short_side(position_type):  # SHORT
                # For SHORT: lower correlation is better
                if correlation < avg_correlation:
                    return -1.0  # Negative score for SHORT
                else:
                    # Calculate increasing negative score as correlation increases
                    correlation_ratio = avg_correlation / correlation
                    return -max(0, min(1, correlation_ratio))  # Negative score between -1 and 0
                    
        except Exception as e:
            logger.error(f"Error calculating correlation score: {str(e)}")
            return 0.0
            
    def _calculate_sentiment_score(self, sentiment: Dict, position_type: str) -> float:
        """Calculate sentiment score based on market sentiment and position type.
        
        Args:
            sentiment: Dictionary containing sentiment indicators
            position_type: Position type (LONG/SHORT)
            
        Returns:
            float: Sentiment score between -1 and 1
        """
        try:
            if not sentiment:
                return 0.0
                
            # Get individual sentiment indicators
            rsi_sentiment = sentiment.get('rsi_sentiment', 'neutral')
            mfi_sentiment = sentiment.get('mfi_sentiment', 'neutral')
            obv_sentiment = sentiment.get('obv_sentiment', 'neutral')
            trend_strength = sentiment.get('trend_strength', 'weak')
            overall_sentiment = sentiment.get('overall_sentiment', 'neutral')
            
            # Calculate score based on position type
            if is_long_side(position_type):
                # For LONG: bullish sentiment is positive
                score = 0.0
                
                # Weight the indicators
                if rsi_sentiment == 'bullish':
                    score += 0.3
                elif rsi_sentiment == 'bearish':
                    score -= 0.3
                    
                if mfi_sentiment == 'bullish':
                    score += 0.2
                elif mfi_sentiment == 'bearish':
                    score -= 0.2
                    
                if obv_sentiment == 'bullish':
                    score += 0.2
                elif obv_sentiment == 'bearish':
                    score -= 0.2
                    
                # Adjust score based on trend strength
                if trend_strength == 'strong':
                    if overall_sentiment == 'bullish':
                        score *= 1.5
                    elif overall_sentiment == 'bearish':
                        score *= 0.5
                        
                return max(-1.0, min(1.0, score))
                
            elif is_short_side(position_type):  # SHORT
                # For SHORT: bearish sentiment is positive (negative score)
                score = 0.0
                
                # Weight the indicators
                if rsi_sentiment == 'bearish':
                    score += 0.3
                elif rsi_sentiment == 'bullish':
                    score -= 0.3
                    
                if mfi_sentiment == 'bearish':
                    score += 0.2
                elif mfi_sentiment == 'bullish':
                    score -= 0.2
                    
                if obv_sentiment == 'bearish':
                    score += 0.2
                elif obv_sentiment == 'bullish':
                    score -= 0.2
                    
                # Adjust score based on trend strength
                if trend_strength == 'strong':
                    if overall_sentiment == 'bearish':
                        score *= 1.5
                    elif overall_sentiment == 'bullish':
                        score *= 0.5
                        
                return -max(-1.0, min(1.0, score))  # Negative score for SHORT
                
        except Exception as e:
            logger.error(f"Error calculating sentiment score: {str(e)}")
            return 0.0

    def _calculate_structure_score(self, market_structure: Dict, position_type: str) -> float:
        """Calculate structure score based on market structure and position type."""
        try:
            if not market_structure:
                return 0.0
                
            # Get the main timeframe (5m) structure
            main_timeframe = market_structure.get('5m', {})
            if not main_timeframe:
                return 0.0
                
            structure = main_timeframe.get('structure', 'neutral')
            current_price = main_timeframe.get('current_price', 0)
            nearest_support = main_timeframe.get('nearest_support', 0)
            nearest_resistance = main_timeframe.get('nearest_resistance', 0)
            
            if is_long_side(position_type):
                # For LONG: being near support is good
                if structure == "support":
                    # Calculate strength based on distance to support
                    if nearest_support > 0:
                        distance = (current_price - nearest_support) / current_price
                        return max(0.5, 1.0 - distance)  # Positive score for LONG
                    return 0.5
                elif structure == "resistance":
                    return 0.0
                else:  # neutral
                    return 0.3
                    
            elif is_short_side(position_type):  # SHORT
                # For SHORT: being near resistance is good
                if structure == "resistance":
                    # Calculate strength based on distance to resistance
                    if nearest_resistance > 0:
                        distance = (nearest_resistance - current_price) / current_price
                        return -max(0.5, 1.0 - distance)  # Negative score for SHORT
                    return -0.5
                elif structure == "support":
                    return 0.0
                else:  # neutral
                    return -0.3  # Negative score for neutral structure in SHORT
                    
        except Exception as e:
            logger.error(f"Error calculating structure score: {str(e)}")
            return 0.0

    def _calculate_volume_profile_score(self, volume_profile: Dict, position_type: str) -> float:
        """Calculate volume profile score based on volume distribution and position type."""
        try:
            if not volume_profile or 'high_volume_nodes' not in volume_profile:
                return 0.0
                
            high_volume_nodes = volume_profile['high_volume_nodes']
            current_price = volume_profile.get('current_price', 0)
            total_volume = sum(node['volume'] for node in high_volume_nodes)
            
            if is_long_side(position_type):
                # For LONG: check volume at lower price levels
                low_price_nodes = [node for node in high_volume_nodes if node['price'] < current_price]
                if not low_price_nodes:
                    logger.info(f"No low price nodes found")
                    return 0.0
                    
                # Calculate volume ratio at lower prices
                low_price_volume = sum(node['volume'] for node in low_price_nodes)
                volume_ratio = low_price_volume / total_volume
                
                # Calculate score based on volume ratio
                if volume_ratio > 0.5:  # More than 50% volume at lower prices
                    logger.info(f"50% volume at lower prices: {volume_ratio}")
                    return 1.0
                else:
                    logger.info(f"LONG Volume ratio: {volume_ratio}")
                    return volume_ratio * 2  # Proportional to volume ratio
                    
            elif is_short_side(position_type):  # SHORT
                # For SHORT: check volume at higher price levels
                high_price_nodes = [node for node in high_volume_nodes if node['price'] > current_price]
                if not high_price_nodes:
                    logger.info(f"No high price nodes found")
                    return 0.0
                    
                # Calculate volume ratio at higher prices
                high_price_volume = sum(node['volume'] for node in high_price_nodes)
                volume_ratio = high_price_volume / total_volume
                
                # Calculate negative score based on volume ratio
                if volume_ratio > 0.5:  # More than 50% volume at higher prices
                    logger.info(f"50% volume at higher prices: {volume_ratio}")
                    return -1.0  # Negative score for SHORT
                else:
                    logger.info(f"SHORT Volume ratio: {volume_ratio}")
                    return -volume_ratio * 2  # Negative score proportional to volume ratio
                    
        except Exception as e:
            logger.error(f"Error calculating volume profile score: {str(e)}")
            return 0.0

    def _calculate_funding_rate_score(self, funding_rate: float, position_type: str) -> float:
        """Calculate funding rate score based on funding rate and position type."""
        try:
            if funding_rate == 0:
                return 0.0
                
            if is_long_side(position_type):
                # For LONG: negative funding rate is good
                if funding_rate < -0.0001:  # Significant negative funding rate
                    return 1.0
                elif funding_rate < 0:  # Slight negative funding rate
                    return 0.5
                else:  # Positive funding rate
                    return -0.5
            elif is_short_side(position_type):  # SHORT
                # For SHORT: positive funding rate is good
                if funding_rate > 0.0001:  # Significant positive funding rate
                    return -1.0  # Negative score for SHORT
                elif funding_rate > 0:  # Slight positive funding rate
                    return -0.5  # Negative score for SHORT
                else:  # Negative funding rate
                    return 0.5  # Positive score for unfavorable SHORT conditions
                    
        except Exception as e:
            logger.error(f"Error calculating funding rate score: {str(e)}")
            return 0.0

    def _calculate_open_interest_score(self, open_interest: float, position_type: str) -> float:
        """Calculate open interest score based on OI value and position type."""
        try:
            if open_interest == 0:
                return 0.0
                
            # Normalize open interest to a score between -1 and 1
            normalized_oi = min(max(open_interest / 1000000, 0.1), 1.0)  # Assuming 1M as max OI
            
            if is_long_side(position_type):
                # For LONG: Higher OI is better
                return normalized_oi  # Positive score (0.1 to 1.0)
            elif is_short_side(position_type):  # SHORT
                # For SHORT: Lower OI is better
                return -(1.0 - normalized_oi)  # Negative score (-0.9 to 0.0)
                    
        except Exception as e:
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            logger.error(f"Error calculating open interest score: {str(e)}")
            return 0.0

    def _calculate_order_book_score(self, order_book: Dict, position_type: str) -> float:
        """Calculate order book score based on order book depth and position type."""
        try:
            if not isinstance(order_book, dict) or not order_book:
                return 0.0
                
            bid_depth = order_book.get('bid_depth', 0)
            ask_depth = order_book.get('ask_depth', 0)
            spread = order_book.get('spread', 0)
            
            if bid_depth == 0 or ask_depth == 0:
                return 0.0
                
            depth_ratio = bid_depth / ask_depth
            
            if is_long_side(position_type):
                # For LONG: higher bid depth is better
                if depth_ratio > 1.5:  # Significantly higher bid depth
                    return 1.0
                elif depth_ratio > 1.2:  # Moderately higher bid depth
                    return 0.5
                else:  # Lower bid depth
                    return -0.5
            elif is_short_side(position_type):  # SHORT
                # For SHORT: higher ask depth is better (negative score)
                if depth_ratio < 0.67:  # Significantly higher ask depth
                    return -1.0  # Negative score for SHORT
                elif depth_ratio < 0.83:  # Moderately higher ask depth
                    return -0.5  # Negative score for SHORT
                else:  # Lower ask depth
                    return 0.5  # Positive score for unfavorable SHORT conditions
                    
        except Exception as e:
            logger.error(f"Error calculating order book score: {str(e)}")
            return 0.0

    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels.
        
        Args:
            df: Price data DataFrame
            
        Returns:
            Dict: Support and resistance levels
        """
        try:
            if df is None or df.empty:
                logger.error("No data provided for support/resistance calculation")
                return {
                    'support_levels': [],
                    'resistance_levels': [],
                    'nearest_support': None,
                    'nearest_resistance': None
                }
                
            # Get high and low prices
            highs = df['high']
            lows = df['low']
            
            # Initialize lists for support and resistance levels
            support_levels = []
            resistance_levels = []
            
            # Find swing highs and lows
            for i in range(1, len(df)-1):
                # Check for swing low (support)
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                    support_levels.append(lows.iloc[i])
                    
                # Check for swing high (resistance)
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    resistance_levels.append(highs.iloc[i])
                    
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Find nearest support and resistance
            nearest_support = None
            nearest_resistance = None
            
            # Find nearest support (highest support level below current price)
            valid_supports = [s for s in support_levels if s < current_price]
            if valid_supports:
                nearest_support = max(valid_supports)
            elif support_levels:  # If no support below current price, use lowest support
                nearest_support = min(support_levels)
                
            # Find nearest resistance (lowest resistance level above current price)
            valid_resistances = [r for r in resistance_levels if r > current_price]
            if valid_resistances:
                nearest_resistance = min(valid_resistances)
            elif resistance_levels:  # If no resistance above current price, use highest resistance
                nearest_resistance = max(resistance_levels)
                
            # Log debug information
            logger.debug(f"Found {len(support_levels)} support levels and {len(resistance_levels)} resistance levels")
            logger.debug(f"Current price: {current_price}, Nearest support: {nearest_support}, Nearest resistance: {nearest_resistance}")
                
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance levels: {str(e)}")
            return {
                'support_levels': [],
                'resistance_levels': [],
                'nearest_support': None,
                'nearest_resistance': None
            }

    def _calculate_value_area(self, df: pd.DataFrame) -> Dict:
        """Calculate volume profile value area.
        
        Args:
            df: Price data DataFrame
            
        Returns:
            Dict: Dictionary containing value area information
        """
        try:
            # Calculate price levels
            price_range = df['high'].max() - df['low'].min()
            num_levels = 20
            price_step = price_range / num_levels
            
            # Calculate volume at each price level
            volume_profile = {}
            for i in range(num_levels):
                price_level = df['low'].min() + i * price_step
                volume = df[(df['low'] <= price_level) & (df['high'] >= price_level)]['volume'].sum()
                volume_profile[price_level] = volume
            
            # Sort by volume
            sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
            
            # Calculate value area (70% of total volume)
            total_volume = sum(volume_profile.values())
            target_volume = total_volume * 0.7
            current_volume = 0
            value_area_levels = []
            
            for level, volume in sorted_levels:
                if current_volume < target_volume:
                    value_area_levels.append(level)
                    current_volume += volume
                else:
                    break
            
            # Get value area high and low
            value_area_high = max(value_area_levels)
            value_area_low = min(value_area_levels)
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Calculate value area position
            if value_area_high != value_area_low:
                value_area_position = (current_price - value_area_low) / (value_area_high - value_area_low)
            else:
                value_area_position = 0.5
                
            return {
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'value_area_position': value_area_position,
                'current_price': current_price,
                'volume_profile': volume_profile
            }
            
        except Exception as e:
            logger.error(f"Error calculating value area: {str(e)}")
            return {
                'value_area_high': None,
                'value_area_low': None,
                'value_area_position': 0.5,
                'current_price': None,
                'volume_profile': {}
            }

    async def _get_dynamic_threshold(self, threshold_type: str, position_type: str) -> float:
        """Calculate dynamic threshold based on market conditions."""
        try:
            base_threshold = self.base_thresholds.get(threshold_type, 0.0)
            
            # Lấy volatility hiện tại
            volatility = await self._get_current_volatility()
            
            # Điều chỉnh theo volatility
            adjustment_factor = 1.0
            if volatility > self.adjustment_factors['volatility_threshold']:
                adjustment_factor = 1.2
            elif volatility < 0.02:
                adjustment_factor = 0.8
                
            # Điều chỉnh theo position type
            if is_short_side(position_type):
                adjustment_factor *= 1.1  # Thắt chặt hơn cho SHORT
                
            return base_threshold * adjustment_factor
            
        except Exception as e:
            logger.error(f"Error calculating dynamic threshold: {str(e)}")
            return self.base_thresholds.get(threshold_type, 0.0)

    async def _get_dynamic_weights(self, position_type: str) -> Dict[str, float]:
        """Calculate dynamic weights based on market conditions."""
        try:
            weights = self.base_weights.copy()
            
            # Lấy market conditions
            market_conditions = await self._get_market_conditions()
            
            # Điều chỉnh theo volatility
            if market_conditions.get('volatility', 0) > self.adjustment_factors['volatility_threshold']:
                weights['volatility'] += self.adjustment_factors['weight_adjustment']
                weights['trend'] -= self.adjustment_factors['weight_adjustment']
                
            # Điều chỉnh theo position type
            if is_short_side(position_type):
                weights['funding_rate'] += self.adjustment_factors['weight_adjustment']
                weights['trend'] -= self.adjustment_factors['weight_adjustment']
                
            # Điều chỉnh theo market sentiment
            if market_conditions.get('sentiment', 0) < 0:
                weights['sentiment'] += self.adjustment_factors['weight_adjustment']
                weights['trend'] -= self.adjustment_factors['weight_adjustment']
                
            # Chuẩn hóa weights
            total = sum(weights.values())
            return {k: v/total for k, v in weights.items()}
            
        except Exception as e:
            logger.error(f"Error calculating dynamic weights: {str(e)}")
            return self.base_weights.copy()

    async def _get_current_volatility(self) -> float:
        """Calculate current market volatility."""
        try:
            # Lấy dữ liệu BTC
            btc_data = await self._get_btc_data()
            if btc_data is None or btc_data.empty:
                return 0.0
                
            # Tính volatility
            returns = btc_data['close'].pct_change()
            return returns.std()
            
        except Exception as e:
            logger.error(f"Error calculating current volatility: {str(e)}")
            return 0.0

    async def _get_btc_data(self) -> Optional[pd.DataFrame]:
        """Get BTC historical data."""
        try:
            return await self.indicator_service.get_historical_data('BTCUSDT', timeframe='1h', limit=100)
        except Exception as e:
            logger.error(f"Error getting BTC data: {str(e)}")
            return None

    async def _log_signal_analysis(self, symbol: str, position_type: str, scores: Dict, 
                           total_score: float, conditions: Dict) -> None:
        """Log detailed signal analysis."""
        try:
            # Convert numpy.bool_ to Python bool
            def convert_numpy_types(obj):
                if isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj

            log_data = {
                'symbol': symbol,
                'position_type': position_type,
                'timestamp': datetime.now().isoformat(),
                'scores': convert_numpy_types(scores),
                'total_score': total_score,
                'conditions': convert_numpy_types(conditions)
            }
            
            logger.debug(f"Signal analysis: {json.dumps(log_data, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error logging signal analysis: {str(e)}")

    async def _monitor_signal_quality(self, symbol: str, position_type: str, 
                              signal_data: Dict) -> None:
        """Monitor signal quality and send alerts if needed."""
        try:
            # Get historical signals
            historical_signals = await self._get_historical_signals(symbol, position_type)
            
            # Calculate metrics
            win_rate = self._calculate_win_rate(historical_signals)
            avg_profit = self._calculate_avg_profit(historical_signals)
            avg_loss = self._calculate_avg_loss(historical_signals)
            
            # Analyze current signal data
            current_score = signal_data.get('total_score', 0)
            current_conditions = signal_data.get('conditions', {})
            
            # Check signal quality based on historical data and current signal
            if (win_rate < 0.5 or avg_profit/avg_loss < 1.5) and current_score < 0.7:
                await self._send_alert(f"Low quality signal detected for {symbol} {position_type}")
                
            # Update metrics with current signal data
            self._update_signal_metrics(symbol, position_type, {
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'total_signals': len(historical_signals),
                'current_score': current_score,
                'current_conditions': current_conditions
            })
            
        except Exception as e:
            logger.error(f"Error monitoring signal quality: {str(e)}")

    async def _get_historical_signals(self, symbol: str, position_type: str) -> List[Dict]:
        """Get historical signals for a symbol and position type."""
        try:
            symbol = symbol.split(':')[0].replace('/', '')
            # Kiểm tra cache trước
            cache_key = f"historical_signals_{symbol}_{position_type}"
            cached_signals = await self.binance_service._get_cached_data(cache_key)
            if cached_signals is not None:
                return cached_signals

            # Lấy dữ liệu từ database hoặc file
            signals = []
            try:
                # Thử đọc từ file JSON
                file_path = f"data/signals/{symbol}_{position_type}_signals.json"
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        signals = json.load(f)
            except Exception as e:
                logger.warning(f"Error reading signals from file: {str(e)}")

            # Lọc tín hiệu theo thời gian (chỉ lấy tín hiệu trong 30 ngày gần nhất)
            current_time = datetime.now()
            signals = [
                s for s in signals 
                if (current_time - datetime.fromisoformat(s['timestamp'])).days <= 30
            ]

            # Lưu vào cache
            self.binance_service._set_cached_data(cache_key, signals)
            return signals

        except Exception as e:
            logger.error(f"Error getting historical signals: {str(e)}")
            return []

    def _calculate_win_rate(self, signals: List[Dict]) -> float:
        """Calculate win rate from historical signals."""
        try:
            if not signals:
                return 0.0
                
            winning_signals = [s for s in signals if s.get('profit', 0) > 0]
            return len(winning_signals) / len(signals)
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {str(e)}")
            return 0.0

    def _calculate_avg_profit(self, signals: List[Dict]) -> float:
        """Calculate average profit from historical signals."""
        try:
            if not signals:
                return 0.0
                
            profits = [s.get('profit', 0) for s in signals if s.get('profit', 0) > 0]
            return sum(profits) / len(profits) if profits else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating average profit: {str(e)}")
            return 0.0

    def _calculate_avg_loss(self, signals: List[Dict]) -> float:
        """Calculate average loss from historical signals."""
        try:
            if not signals:
                return 0.0
                
            losses = [s.get('profit', 0) for s in signals if s.get('profit', 0) < 0]
            return sum(losses) / len(losses) if losses else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating average loss: {str(e)}")
            return 0.0

    def _update_signal_metrics(self, symbol: str, position_type: str, metrics: Dict) -> None:
        """Update signal metrics for a symbol and position type."""
        try:
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs("data/metrics", exist_ok=True)
            
            symbol = symbol.split(':')[0].replace('/', '')
            # Đọc metrics hiện tại
            file_path = f"data/metrics/{symbol}_{position_type}_metrics.json"
            current_metrics = {}
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    current_metrics = json.load(f)
            
            # Cập nhật metrics
            current_metrics.update({
                'last_updated': datetime.now().isoformat(),
                'total_trades': current_metrics.get('total_trades', 0) + 1,
                'win_rate': metrics.get('win_rate', current_metrics.get('win_rate', 0)),
                'avg_profit': metrics.get('avg_profit', current_metrics.get('avg_profit', 0)),
                'avg_loss': metrics.get('avg_loss', current_metrics.get('avg_loss', 0)),
                'total_signals': metrics.get('total_signals', current_metrics.get('total_signals', 0))
            })
            
            # Lưu metrics mới
            with open(file_path, 'w') as f:
                json.dump(current_metrics, f, indent=4)
                
            # Cập nhật cache
            cache_key = f"signal_metrics_{symbol}_{position_type}"
            self.binance_service._set_cached_data(cache_key, current_metrics)
            
        except Exception as e:
            logger.error(f"Error updating signal metrics: {str(e)}")

    async def _send_alert(self, message: str) -> None:
        """Send alert through notification service."""
        try:
            await self.notification_service.send_notification('error', message)
        except Exception as e:
            logger.error(f"Error sending alert: {str(e)}")

    async def _calculate_market_liquidity(self) -> float:
        """Calculate current market liquidity score."""
        try:
            # Lấy dữ liệu order book cho BTC
            btc_order_book = await self.binance_service.get_order_book("BTCUSDT", limit=100)
            if not btc_order_book:
                return 0.0
                
            # Tính toán spread
            best_bid = float(btc_order_book['bids'][0][0])
            best_ask = float(btc_order_book['asks'][0][0])
            spread = (best_ask - best_bid) / best_bid
            
            # Tính toán depth
            bid_depth = sum(float(bid[1]) for bid in btc_order_book['bids'])
            ask_depth = sum(float(ask[1]) for ask in btc_order_book['asks'])
            avg_depth = (bid_depth + ask_depth) / 2
            
            # Tính toán volume 24h
            ticker = await self.binance_service.get_ticker("BTCUSDT")
            volume_24h = float(ticker.get('volume', 0))
            
            # Tính điểm thanh khoản (0-1)
            spread_score = max(0, 1 - spread * 100)  # Spread càng nhỏ điểm càng cao
            depth_score = min(1, avg_depth / 100)    # Depth càng lớn điểm càng cao
            volume_score = min(1, volume_24h / 10000) # Volume càng lớn điểm càng cao
            
            # Tính điểm tổng hợp
            liquidity_score = (spread_score * 0.4 + depth_score * 0.3 + volume_score * 0.3)
            
            # Lưu vào cache
            self.binance_service._set_cached_data("market_liquidity", liquidity_score)
            
            return liquidity_score
            
        except Exception as e:
            logger.error(f"Error calculating market liquidity: {str(e)}")
            return 0.0

    async def _is_risk_on_environment(self) -> bool:
        """Check if current market environment is risk-on."""
        try:
            # Lấy dữ liệu BTC
            btc_data = await self._get_btc_data()
            if btc_data is None or btc_data.empty:
                return False
                
            # Kiểm tra trend BTC
            btc_trend = self.get_trend(btc_data)
            if btc_trend != "UP":
                return False
                
            # Kiểm tra volume BTC
            current_volume = btc_data['volume'].iloc[-1]
            avg_volume = btc_data['volume'].rolling(20).mean().iloc[-1]
            if current_volume < avg_volume:
                return False
                
            # Kiểm tra funding rate
            funding_rate = await self.binance_service.get_funding_rate("BTCUSDT")
            if funding_rate is None or funding_rate < 0:
                return False
                
            # Kiểm tra open interest
            open_interest = await self.binance_service.get_open_interest("BTCUSDT")
            if open_interest is None or open_interest < 1000:  # Ngưỡng tối thiểu
                return False
                
            # Kiểm tra thanh khoản
            liquidity = await self._calculate_market_liquidity()
            if liquidity < 0.5:  # Ngưỡng thanh khoản tối thiểu
                return False
                
            # Kiểm tra volatility
            volatility = await self._get_current_volatility()
            if volatility > 0.05:  # Volatility quá cao
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk-on environment: {str(e)}")
            return False

    def _calculate_timeframe_score(self, timeframe_data: Dict, position_type: str) -> float:
        """Calculate timeframe score based on trend and strength."""
        try:
            if not timeframe_data:
                return 0.0
                
            # Get trend and strength from timeframes
            trends = {
                '1h': timeframe_data.get('1h', {}).get('trend', ''),
                '4h': timeframe_data.get('4h', {}).get('trend', ''),
                '1d': timeframe_data.get('1d', {}).get('trend', '')
            }
            
            strengths = {
                '1h': timeframe_data.get('1h', {}).get('strength', 0),
                '4h': timeframe_data.get('4h', {}).get('strength', 0),
                '1d': timeframe_data.get('1d', {}).get('strength', 0)
            }
            
            if is_long_side(position_type):
                # For LONG: count bullish timeframes and average their strengths
                bullish_count = sum(1 for trend in trends.values() if is_trending_up(trend))
                bullish_strength = sum(strengths[tf] for tf, trend in trends.items() if is_trending_up(trend))
                
                if bullish_count == 3:  # All timeframes bullish
                    return 1.0
                elif bullish_count == 2:  # Two timeframes bullish
                    return 0.7 + (bullish_strength / 200)  # Add up to 0.3 based on strength
                elif bullish_count == 1:  # One timeframe bullish
                    return 0.3 + (bullish_strength / 300)  # Add up to 0.2 based on strength
                else:  # No bullish timeframes
                    return -0.5
            elif is_short_side(position_type):  # SHORT
                # For SHORT: count bearish timeframes and average their strengths (negative scores)
                bearish_count = sum(1 for trend in trends.values() if is_trending_down(trend))
                bearish_strength = sum(strengths[tf] for tf, trend in trends.items() if is_trending_down(trend))
                
                if bearish_count == 3:  # All timeframes bearish
                    return -1.0  # Negative score for SHORT
                elif bearish_count == 2:  # Two timeframes bearish
                    return -(0.7 + (bearish_strength / 200))  # Negative score with strength bonus
                elif bearish_count == 1:  # One timeframe bearish
                    return -(0.3 + (bearish_strength / 300))  # Negative score with strength bonus
                else:  # No bearish timeframes
                    return 0.5  # Positive score for unfavorable SHORT conditions
                    
        except Exception as e:
            logger.error(f"Error calculating timeframe score: {str(e)}")
            return 0.0

    def _calculate_momentum(self, df: pd.DataFrame, lookback_period: int = 14) -> float:
        """
        Calculate momentum using rate of change (ROC)
        """
        try:
            # Calculate rate of change
            roc = ((df['close'].iloc[-1] - df['close'].iloc[-lookback_period]) / 
                   df['close'].iloc[-lookback_period]) * 100
            
            # Calculate momentum strength
            momentum_strength = abs(roc)
            
            # Calculate momentum direction
            momentum_direction = 1 if roc > 0 else -1
            
            return {
                'strength': momentum_strength,
                'direction': momentum_direction,
                'roc': roc
            }
        except Exception as e:
            logger.error(f"Error calculating momentum: {str(e)}")
            return {'strength': 0, 'direction': 0, 'roc': 0}

    async def _adjust_position_size_by_volatility(self, symbol: str, base_size: float) -> float:
        """Adjust position size based on market volatility"""
        try:
            # Get historical klines
            klines = await self.binance_service.get_klines(
                symbol=symbol,
                timeframe='1h',
                limit=24  # Last 24 hours
            )
            
            if not klines:
                logger.warning(f"No klines data available for {symbol}")
                return base_size
                
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            
            # Calculate volatility
            df['returns'] = df['close'].pct_change()
            volatility = df['returns'].std() * np.sqrt(24)  # Annualized volatility
            
            # Get market volatility
            market_volatility = await self._get_market_volatility()
            if market_volatility is None:
                return base_size
                
            # Calculate relative volatility
            relative_vol = volatility / market_volatility
            
            # Adjust position size
            if relative_vol > 2.0:  # Very high volatility
                adjusted_size = base_size * 0.5
            elif relative_vol > 1.5:  # High volatility
                adjusted_size = base_size * 0.75
            elif relative_vol < 0.5:  # Low volatility
                adjusted_size = base_size * 1.25
            else:  # Normal volatility
                adjusted_size = base_size
                
            return max(adjusted_size, base_size * 0.5)  # Never go below 50% of base size
            
        except Exception as e:
            logger.error(f"Error adjusting position size by volatility: {str(e)}")
            return base_size

    def _should_exit_by_momentum(self, df: pd.DataFrame, position: Dict, 
                           momentum_threshold: float = 2.0) -> bool:
        """
        Determine if position should be closed based on momentum with enhanced conditions
        """
        try:
            # Calculate momentum
            momentum = self._calculate_momentum(df)
            
            # Get additional market data
            volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            
            # Calculate trend using EMA
            ema20 = df['close'].ewm(span=20).mean().iloc[-1]
            ema50 = df['close'].ewm(span=50).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Determine trend strength
            trend_strength = abs((ema20 - ema50) / ema50) * 100
            
            # Get position information
            position_age = (datetime.now() - position.get('updateTime', datetime.now())).total_seconds() / 3600  # hours
            position_type = position.get('info', {}).get('positionSide', 'LONG')
            unrealized_pnl = position.get('unrealizedPnl', 0)
            entry_price = position.get('entryPrice', current_price)
            
            # Calculate price change percentage
            price_change_pct = ((current_price - entry_price) / entry_price) * 100
            
            # Dynamic momentum threshold based on market conditions
            dynamic_threshold = momentum_threshold
            
            # Adjust threshold based on volume
            if volume_ratio > 2.0:  # High volume
                dynamic_threshold *= 1.2  # Require stronger momentum
            elif volume_ratio < 0.5:  # Low volume
                dynamic_threshold *= 0.8  # Lower threshold
                
            # Adjust threshold based on trend strength
            if trend_strength > 5.0:  # Strong trend
                dynamic_threshold *= 1.3  # Require stronger momentum
                
            # Adjust threshold based on position age
            if position_age < 1:  # Less than 1 hour
                dynamic_threshold *= 1.5  # Require stronger momentum
            elif position_age > 24:  # More than 24 hours
                dynamic_threshold *= 0.8  # Lower threshold
                
            # For long positions
            if is_long_side(position_type):
                # Exit conditions for long positions
                if momentum['direction'] == -1 and momentum['strength'] > dynamic_threshold:
                    # Additional checks for long positions
                    if price_change_pct < -2.0:  # Price dropped more than 2%
                        if volume_ratio > 1.5:  # High volume
                            return True
                    elif price_change_pct < -5.0:  # Price dropped more than 5%
                        return True
                        
            # For short positions
            elif is_short_side(position_type):
                # Exit conditions for short positions
                if momentum['direction'] == 1 and momentum['strength'] > dynamic_threshold:
                    # Additional checks for short positions
                    if price_change_pct > 2.0:  # Price increased more than 2%
                        if volume_ratio > 1.5:  # High volume
                            return True
                    elif price_change_pct > 5.0:  # Price increased more than 5%
                        return True
            
            # Log decision details
            logger.info(f"Momentum exit check for {position_type}: "
                    f"momentum_strength={momentum['strength']:.2f}, "
                    f"dynamic_threshold={dynamic_threshold:.2f}, "
                    f"volume_ratio={volume_ratio:.2f}, "
                    f"trend_strength={trend_strength:.2f}, "
                    f"position_age={position_age:.1f}h, "
                    f"price_change={price_change_pct:.2f}%")
            
            return False
            
        except Exception as e:
            logger.error(f"Error in momentum exit check: {str(e)}")
            return False

    async def _get_market_volatility(self) -> Optional[float]:
        """
        Calculate market volatility based on BTC data.
        Returns the annualized volatility of BTC.
        """
        try:
            # Get BTC data for the last 24 hours
            btc_data = await self._get_btc_data()
            if btc_data is None or btc_data.empty:
                logger.warning("No BTC data available for market volatility calculation")
                return None

            # Calculate returns
            btc_data['returns'] = btc_data['close'].pct_change()
            
            # Calculate annualized volatility (24 hours)
            market_volatility = btc_data['returns'].std() * np.sqrt(24)
            
            logger.info(f"Market volatility: {market_volatility:.4f}")
            return market_volatility
            
        except Exception as e:
            logger.error(f"Error calculating market volatility: {str(e)}")
            return None

    def _check_dca_limits(self, symbol: str, position_type: str) -> bool:
        """Check if DCA is allowed based on history and limits.
        
        Args:
            symbol: Trading pair symbol
            position_type: Position type (LONG/SHORT)
            
        Returns:
            bool: True if DCA is allowed, False otherwise
        """
        try:
            if symbol not in self._dca_history:
                return True
                
            dca_history = self._dca_history[symbol]
            if not dca_history:
                return True
                
            # Get DCA configuration
            dca_config = self.config.get('risk_management', {}).get('dca', {})
            max_dca_attempts = dca_config.get('max_attempts', 3)
            min_dca_interval = dca_config.get('min_interval', 3600)  # 1 hour in seconds
            
            # Check number of DCA attempts
            if len(dca_history) >= max_dca_attempts:
                logger.info(f"DCA limit reached: {len(dca_history)} attempts for {symbol}")
                return False
                
            # Check time interval between DCAs
            last_dca_time = datetime.fromisoformat(dca_history[-1]['timestamp'])
            time_since_last_dca = (datetime.now() - last_dca_time).total_seconds()
            if time_since_last_dca < min_dca_interval:
                logger.info(f"DCA too soon: {time_since_last_dca} seconds since last DCA for {symbol}")
                return False
                
            # Check effectiveness of previous DCAs
            if len(dca_history) >= 2:
                last_two_dcas = dca_history[-2:]
                price_trend = last_two_dcas[-1]['entry_price'] - last_two_dcas[0]['entry_price']
                
                # If price is still dropping after DCA, be more cautious
                if (is_long_side(position_type) and price_trend < 0) or \
                   (is_short_side(position_type) and price_trend > 0):
                    logger.info(f"Previous DCAs not effective: price still moving against position for {symbol}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking DCA limits for {symbol}: {str(e)}")
            return False

