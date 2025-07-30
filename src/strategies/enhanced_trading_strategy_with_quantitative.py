"""
Enhanced Trading Strategy with Quantitative Trading Integration.
Extends the original strategy with quantitative analysis capabilities.
"""

import logging
from typing import Dict, Optional, List, Any
import pandas as pd
import numpy as np
import time
import asyncio
from datetime import datetime
import json
import os
import sys
import psutil
import gc
from collections import OrderedDict

# Set event loop policy for Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from src.core.config import load_config
from src.services.indicator_service import IndicatorService
from src.services.sentiment_service import SentimentService
from src.services.binance_service import BinanceService
from src.services.notification_service import NotificationService
from src.utils.helpers import is_long_side, is_short_side, is_trending_down, is_trending_up
from src.quantitative.integration import QuantitativeIntegration
from src.quantitative.quantitative_trading_system import QuantitativeTradingSystem
from src.quantitative.statistical_validator import StatisticalValidator

logger = logging.getLogger(__name__)

class EnhancedTradingStrategyWithQuantitative:
    """
    Enhanced trading strategy with WorldQuant-level quantitative analysis integration.
    """
    
    def __init__(self, config: Dict, binance_service: BinanceService, 
                 indicator_service: IndicatorService, notification_service: NotificationService,
                 cache_service: Optional['CacheService'] = None):
        """
        Initialize enhanced trading strategy with quantitative integration.
        
        Args:
            config: Configuration dictionary
            binance_service: Binance service instance
            indicator_service: Indicator service instance
            notification_service: Notification service instance
            cache_service: Cache service instance (optional)
        """
        self.config = config
        self.binance_service = binance_service
        self.indicator_service = indicator_service
        self.notification_service = notification_service
        self.cache_service = cache_service
        
        # Initialize quantitative components
        self.quantitative_integration = QuantitativeIntegration(config)
        self.quantitative_system = QuantitativeTradingSystem(config)
        
        # Initialize statistical validator
        significance_level = config.get('trading', {}).get('statistical_significance_level', 0.05)
        min_sample_size = config.get('trading', {}).get('min_sample_size', 100)
        self.statistical_validator = StatisticalValidator(significance_level, min_sample_size)
        
        # Performance tracking
        self.signal_history = {}
        self.quantitative_analysis_history = {}
        self.data_cache = {}
        
        # Confidence performance tracking
        self.confidence_performance = {
            'buy': {'executions': 0, 'successes': 0, 'thresholds': []},
            'sell': {'executions': 0, 'successes': 0, 'thresholds': []},
            'thresholds': {
                'buy': {'avg_threshold': 0.0, 'count': 0},
                'sell': {'avg_threshold': 0.0, 'count': 0}
            }
        }
        
        # Initialize cache service if provided
        if self.cache_service:
            logger.info("Cache service initialized for enhanced trading strategy")
        
        logger.info("Enhanced Trading Strategy with Quantitative Analysis initialized")
    
    async def initialize(self) -> bool:
        """Initialize the strategy and quantitative components."""
        try:
            # Initialize quantitative integration
            await self.quantitative_integration.initialize()
            
            # Initialize quantitative trading system
            await self.quantitative_system.initialize()
            
            logger.info("Enhanced Trading Strategy with Quantitative Analysis initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing enhanced trading strategy: {str(e)}")
            return False
    
    async def generate_signals(self, symbol: str, indicator_service: IndicatorService) -> Optional[Dict]:
        """Generate trading signals with WorldQuant-level quantitative analysis."""
        try:
            # Get comprehensive market data
            market_data = await self._get_comprehensive_market_data(symbol)
            
            # Generate advanced signal with quantitative analysis
            signal = await self._generate_advanced_signal(symbol, indicator_service, market_data)
            
            if signal:
                # Store signal history
                self._store_signal_history(symbol, signal)
                
                # Log quantitative analysis
                validation = await self.quantitative_system.validate_signal(signal, market_data)
                await self._log_quantitative_analysis(symbol, signal, validation)
                
                logger.info(f"Generated signal for {symbol}: {signal.get('action', 'HOLD')} "
                          f"(confidence: {signal.get('confidence', 0):.3f})")
                
                return signal
            
            return None
            
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            logger.error(f"Error generating signals for {symbol}: {str(e)}")
            return None
    
    async def _generate_advanced_signal(self, symbol: str, indicator_service: IndicatorService, market_data: Dict) -> Optional[Dict]:
        """Generate advanced signal with quantitative analysis."""
        try:
            # Get market data for different timeframes
            klines_1h = await indicator_service.get_klines(symbol, '1h', limit=100)
            klines_4h = await indicator_service.get_klines(symbol, '4h', limit=100)
            klines_1d = await indicator_service.get_klines(symbol, '1d', limit=100)
            
            if klines_1h is None or klines_4h is None or klines_1d is None:
                logger.warning(f"Missing market data for {symbol}")
                return None
            
            # Convert klines to DataFrames
            df_1h = self._convert_klines_to_dataframe(klines_1h)
            df_4h = self._convert_klines_to_dataframe(klines_4h)
            df_1d = self._convert_klines_to_dataframe(klines_1d)
            
            # Calculate advanced indicators
            df_1h = await self._calculate_advanced_indicators(df_1h)
            df_4h = await self._calculate_advanced_indicators(df_4h)
            df_1d = await self._calculate_advanced_indicators(df_1d)
            
            # Create advanced signal
            signal = self._create_advanced_signal(symbol, df_1h, df_4h, df_1d, market_data)
            
            # Apply quantitative analysis
            signal = await self._apply_quantitative_analysis(symbol, signal, market_data)
            
            # Apply statistical validation
            signal = await self._apply_statistical_validation(symbol, signal, market_data)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating advanced signal for {symbol}: {str(e)}")
            return None
    
    def _convert_klines_to_dataframe(self, klines: Dict) -> pd.DataFrame:
        """Convert klines dictionary to pandas DataFrame."""
        try:
            if not klines or 'close' not in klines:
                return pd.DataFrame()
            
            # Create DataFrame from klines data
            df = pd.DataFrame({
                'open': klines['open'],
                'high': klines['high'],
                'low': klines['low'],
                'close': klines['close'],
                'volume': klines.get('volume', [0] * len(klines['close']))
            })
            
            return df
            
        except Exception as e:
            logger.error(f"Error converting klines to DataFrame: {str(e)}")
            return pd.DataFrame()
    
    async def _apply_quantitative_analysis(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """Apply quantitative analysis to signal."""
        try:
            # Apply market microstructure analysis
            signal = await self._apply_market_microstructure_analysis(symbol, signal, market_data)
            
            # Apply advanced risk management
            signal = await self._apply_advanced_risk_management(symbol, signal, market_data)
            
            # Apply statistical arbitrage
            signal = await self._apply_statistical_arbitrage(symbol, signal, market_data)
            
            # Apply momentum mean reversion analysis
            signal = await self._apply_momentum_mean_reversion_analysis(symbol, signal, market_data)
            
            # Apply volatility regime analysis
            signal = await self._apply_volatility_regime_analysis(symbol, signal, market_data)
            
            # Apply correlation analysis
            signal = await self._apply_correlation_analysis(symbol, signal, market_data)
            
            # Apply factor model analysis
            signal = await self._apply_factor_model_analysis(symbol, signal, market_data)
            
            # Optimize final signal
            signal = await self._optimize_final_signal(symbol, signal, market_data)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error applying quantitative analysis for {symbol}: {str(e)}")
            return signal
    
    async def _apply_factor_model_analysis(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """Apply factor model analysis to signal."""
        try:
            # Get comprehensive market data for factor analysis
            comprehensive_data = await self._get_comprehensive_market_data(symbol)
            
            if not comprehensive_data:
                logger.warning(f"No comprehensive data available for factor analysis on {symbol}")
                return signal
            
            # Calculate factor exposures for the symbol
            symbols = [symbol]
            factor_exposures = await self.quantitative_system.factor_model.calculate_factor_exposures(
                symbols, comprehensive_data
            )
            
            if symbol in factor_exposures:
                symbol_factors = factor_exposures[symbol]
                
                # Add factor exposures to signal
                signal['factor_exposures'] = symbol_factors
                
                # Calculate factor-adjusted confidence
                factor_adjusted_confidence = self._calculate_factor_adjusted_confidence(
                    signal.get('confidence', 0), symbol_factors
                )
                signal['factor_adjusted_confidence'] = factor_adjusted_confidence
                
                # Apply factor-based signal adjustment
                signal = self._adjust_signal_by_factors(signal, symbol_factors)
                
                logger.info(f"Factor analysis applied to {symbol}: {len(symbol_factors)} factors")
            else:
                logger.warning(f"No factor exposures calculated for {symbol}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error applying factor model analysis for {symbol}: {str(e)}")
            return signal
    
    def _calculate_factor_adjusted_confidence(self, base_confidence: float, factor_exposures: Dict[str, float]) -> float:
        """Calculate factor-adjusted confidence score."""
        try:
            if not factor_exposures:
                return base_confidence
            
            # Define factor weights for confidence adjustment
            factor_weights = {
                'market': 0.2,      # Market factor weight
                'size': 0.15,       # Size factor weight
                'value': 0.15,      # Value factor weight
                'momentum': 0.2,    # Momentum factor weight
                'volatility': 0.15, # Volatility factor weight
                'liquidity': 0.15  # Liquidity factor weight
            }
            
            # Calculate factor adjustment
            factor_adjustment = 0.0
            
            for factor_name, exposure in factor_exposures.items():
                if factor_name in factor_weights:
                    weight = factor_weights[factor_name]
                    # Normalize exposure to [-1, 1] and apply weight
                    normalized_exposure = np.clip(exposure, -1, 1)
                    factor_adjustment += weight * normalized_exposure
            
            # Apply factor adjustment to base confidence
            adjusted_confidence = base_confidence + (factor_adjustment * 0.1)  # 10% adjustment max
            
            # Ensure confidence is within [0, 1] range
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
            
            return adjusted_confidence
            
        except Exception as e:
            logger.error(f"Error calculating factor-adjusted confidence: {str(e)}")
            return base_confidence
    
    def _adjust_signal_by_factors(self, signal: Dict, factor_exposures: Dict[str, float]) -> Dict:
        """Adjust signal based on factor exposures."""
        try:
            if not factor_exposures:
                return signal
            
            # Get current signal action
            current_action = signal.get('action', 'hold')
            
            # Factor-based action adjustments
            action_adjustments = {
                'market': {
                    'positive': 'buy',    # High market exposure -> buy
                    'negative': 'sell'    # Low market exposure -> sell
                },
                'momentum': {
                    'positive': 'buy',    # High momentum -> buy
                    'negative': 'sell'    # Low momentum -> sell
                },
                'value': {
                    'positive': 'buy',    # High value -> buy
                    'negative': 'sell'    # Low value -> sell
                },
                'volatility': {
                    'positive': 'sell',   # High volatility -> sell
                    'negative': 'buy'     # Low volatility -> buy
                }
            }
            
            # Calculate factor-based action score
            action_scores = {'buy': 0, 'sell': 0, 'hold': 0}
            
            for factor_name, exposure in factor_exposures.items():
                if factor_name in action_adjustments:
                    if exposure > 0.1:  # Positive exposure threshold
                        action = action_adjustments[factor_name]['positive']
                        action_scores[action] += abs(exposure)
                    elif exposure < -0.1:  # Negative exposure threshold
                        action = action_adjustments[factor_name]['negative']
                        action_scores[action] += abs(exposure)
                    else:
                        action_scores['hold'] += 1
            
            # Determine factor-adjusted action
            if action_scores['buy'] > action_scores['sell'] and action_scores['buy'] > action_scores['hold']:
                signal['factor_adjusted_action'] = 'buy'
            elif action_scores['sell'] > action_scores['buy'] and action_scores['sell'] > action_scores['hold']:
                signal['factor_adjusted_action'] = 'sell'
            else:
                signal['factor_adjusted_action'] = 'hold'
            
            # Add factor analysis summary
            signal['factor_analysis'] = {
                'factor_exposures': factor_exposures,
                'action_scores': action_scores,
                'factor_adjusted_action': signal.get('factor_adjusted_action', current_action)
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error adjusting signal by factors: {str(e)}")
            return signal
    
    async def analyze_portfolio_factor_exposures(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze portfolio factor exposures."""
        try:
            logger.info(f"Analyzing portfolio factor exposures for {len(symbols)} symbols")
            
            # Get comprehensive market data for all symbols
            all_market_data = {}
            
            for symbol in symbols:
                market_data = await self._get_comprehensive_market_data(symbol)
                if market_data:
                    all_market_data[symbol] = market_data
            
            if not all_market_data:
                logger.warning("No market data available for factor analysis")
                return {}
            
            # Calculate factor exposures
            factor_exposures = await self.quantitative_system.factor_model.calculate_factor_exposures(
                symbols, all_market_data
            )
            
            # Perform risk attribution analysis
            risk_attribution = await self.quantitative_system.factor_model.perform_risk_attribution_analysis(
                symbols, all_market_data
            )
            
            # Analyze sector risk exposure
            sector_analysis = await self.quantitative_system.factor_model.analyze_sector_risk_exposure(symbols)
            
            # Analyze geographic risk exposure
            geographic_analysis = await self.quantitative_system.factor_model.analyze_geographic_risk_exposure(symbols)
            
            # Compile comprehensive analysis
            portfolio_analysis = {
                'factor_exposures': factor_exposures,
                'risk_attribution': risk_attribution,
                'sector_analysis': sector_analysis,
                'geographic_analysis': geographic_analysis,
                'summary': {
                    'total_symbols': len(symbols),
                    'total_factors': len(self.quantitative_system.factor_model.factors),
                    'diversification_score': risk_attribution.get('diversification_score', 0.0),
                    'total_factor_risk': risk_attribution.get('total_factor_risk', 0.0)
                }
            }
            
            logger.info("Portfolio factor analysis completed")
            return portfolio_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio factor exposures: {str(e)}")
            return {}
    
    async def get_factor_model_summary(self) -> Dict[str, Any]:
        """Get factor model summary."""
        try:
            return await self.quantitative_system.factor_model.get_factor_summary()
        except Exception as e:
            logger.error(f"Error getting factor model summary: {str(e)}")
            return {}
    
    async def _calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators."""
        try:
            # Basic indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df['atr'] = true_range.rolling(window=14).mean()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Momentum indicators
            df['momentum'] = df['close'] / df['close'].shift(10) - 1
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating advanced indicators: {str(e)}")
            return df
    
    def _create_advanced_signal(self, symbol: str, df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame, market_data: Dict) -> Dict:
        """Create advanced trading signal with multi-timeframe analysis."""
        try:
            # Analyze each timeframe
            timeframe_1h = self._analyze_timeframe(df_1h, '1h')
            timeframe_4h = self._analyze_timeframe(df_4h, '4h')
            timeframe_1d = self._analyze_timeframe(df_1d, '1d')
            
            # Combine timeframe signals
            timeframes = {
                '1h': timeframe_1h,
                '4h': timeframe_4h,
                '1d': timeframe_1d
            }
            
            combined_signal = self._combine_timeframe_signals(timeframes)
            
            # Add market data context
            combined_signal['symbol'] = symbol
            combined_signal['timestamp'] = datetime.now().isoformat()
            combined_signal['market_data'] = market_data
            
            # Ensure current_price is available
            if 'current_price' not in combined_signal or combined_signal['current_price'] <= 0:
                # Get current price from 1h timeframe as fallback
                if len(df_1h) > 0:
                    combined_signal['current_price'] = float(df_1h['close'].iloc[-1])
                else:
                    combined_signal['current_price'] = 0.0
            
            return combined_signal
            
        except Exception as e:
            logger.error(f"Error creating advanced signal: {str(e)}")
            return {'signal': 'hold', 'strength': 0.0, 'confidence': 0.0, 'current_price': 0.0}
    
    def _analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Analyze single timeframe for trading signals."""
        try:
            if len(df) < 50:
                return {'signal': 'hold', 'strength': 0.0, 'confidence': 0.0, 'current_price': 0.0}
            
            current = df.iloc[-1]
            prev = df.iloc[-2]
            current_price = float(current['close'])
            
            signal_strength = 0.0
            signal_reasons = []
            
            # Trend analysis
            if current['close'] > current['sma_20'] > current['sma_50']:
                signal_strength += 0.2
                signal_reasons.append(f'{timeframe}_uptrend')
            elif current['close'] < current['sma_20'] < current['sma_50']:
                signal_strength -= 0.2
                signal_reasons.append(f'{timeframe}_downtrend')
            
            # MACD analysis
            if current['macd'] > current['macd_signal'] and current['macd_histogram'] > prev['macd_histogram']:
                signal_strength += 0.15
                signal_reasons.append(f'{timeframe}_macd_bullish')
            elif current['macd'] < current['macd_signal'] and current['macd_histogram'] < prev['macd_histogram']:
                signal_strength -= 0.15
                signal_reasons.append(f'{timeframe}_macd_bearish')
            
            # RSI analysis
            if current['rsi'] < 30:
                signal_strength += 0.1
                signal_reasons.append(f'{timeframe}_rsi_oversold')
            elif current['rsi'] > 70:
                signal_strength -= 0.1
                signal_reasons.append(f'{timeframe}_rsi_overbought')
            
            # Bollinger Bands analysis
            if current['close'] < current['bb_lower']:
                signal_strength += 0.1
                signal_reasons.append(f'{timeframe}_bb_oversold')
            elif current['close'] > current['bb_upper']:
                signal_strength -= 0.1
                signal_reasons.append(f'{timeframe}_bb_overbought')
            
            # Volume analysis
            if current['volume_ratio'] > 1.5:
                signal_strength += 0.05
                signal_reasons.append(f'{timeframe}_high_volume')
            
            # Momentum analysis
            if current['momentum'] > 0.02:
                signal_strength += 0.1
                signal_reasons.append(f'{timeframe}_positive_momentum')
            elif current['momentum'] < -0.02:
                signal_strength -= 0.1
                signal_reasons.append(f'{timeframe}_negative_momentum')
            
            # Determine action
            if signal_strength > 0.3:
                action = 'buy'
            elif signal_strength < -0.3:
                action = 'sell'
            else:
                action = 'hold'
            
            return {
                'signal': action,
                'strength': signal_strength,
                'confidence': min(abs(signal_strength), 1.0),
                'reasons': signal_reasons,
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"Error analyzing timeframe {timeframe}: {str(e)}")
            return {'signal': 'hold', 'strength': 0.0, 'confidence': 0.0}
    
    def _combine_timeframe_signals(self, timeframes: Dict) -> Dict:
        """Combine signals from multiple timeframes using weighted approach."""
        try:
            # Weight factors (higher timeframe has more weight)
            weights = {'1h': 0.2, '4h': 0.3, '1d': 0.5}
            
            combined_strength = 0.0
            weighted_confidence = 0.0
            all_reasons = []
            
            # Get current price from 1h timeframe (most recent)
            current_price = 0.0
            if '1h' in timeframes and 'current_price' in timeframes['1h']:
                current_price = timeframes['1h']['current_price']
            elif '4h' in timeframes and 'current_price' in timeframes['4h']:
                current_price = timeframes['4h']['current_price']
            elif '1d' in timeframes and 'current_price' in timeframes['1d']:
                current_price = timeframes['1d']['current_price']
            
            for timeframe, signal in timeframes.items():
                weight = weights.get(timeframe, 0.2)
                combined_strength += signal['strength'] * weight
                weighted_confidence += signal['confidence'] * weight
                all_reasons.extend(signal.get('reasons', []))
            
            # Calculate dynamic thresholds based on market conditions
            # Note: This is a simplified version - in production, you'd pass actual market data
            thresholds = self._calculate_dynamic_thresholds(
                pd.DataFrame(),  # Placeholder - would need actual market data
                volatility_regime=None,
                risk_metrics=None
            )
            
            # WorldQuant-level decision logic with dynamic thresholds
            buy_threshold = thresholds['buy_threshold']
            sell_threshold = thresholds['sell_threshold']
            
            # Determine final action and confidence with WorldQuant-level asymmetric logic
            if combined_strength > buy_threshold:
                action = 'buy'
                confidence = min(combined_strength, 1.0)
            elif combined_strength < sell_threshold:
                action = 'sell'
                confidence = min(abs(combined_strength), 1.0)
            else:
                action = 'hold'
                confidence = 0.0
            
            return {
                'action': action,
                'strength': combined_strength,
                'confidence': confidence,
                'reasons': all_reasons,
                'timeframes': timeframes,
                'thresholds': thresholds,
                'position_size': 0.01,  # Default position size
                'current_price': current_price  # Add current price to signal
            }
            
        except Exception as e:
            logger.error(f"Error combining timeframe signals: {str(e)}")
            return {'action': 'hold', 'strength': 0.0, 'confidence': 0.0, 'current_price': 0.0}
    
    async def _get_comprehensive_market_data(self, symbol: str) -> Dict:
        """Get comprehensive market data for quantitative analysis."""
        try:
            market_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'returns': [],
                'volatility': 0.0,
                'market_regime': 'normal'
            }
            
            # Get historical data for returns calculation
            klines = await self.indicator_service.get_klines(symbol, '1h', limit=100)
            if klines is not None and isinstance(klines, dict) and 'close' in klines and len(klines['close']) > 1:
                prices = np.array(klines['close'])
                returns = np.diff(np.log(prices))
                market_data['returns'] = returns.tolist()
                market_data['volatility'] = float(np.std(returns) * np.sqrt(252))
            
            # Get additional market data if available
            try:
                # Get funding rate
                funding_rate = await self.binance_service.get_funding_rate(symbol)
                if funding_rate is not None:
                    market_data['funding_rate'] = float(funding_rate)
                
                # Get 24h ticker
                ticker = await self.binance_service.get_ticker(symbol)
                if ticker and isinstance(ticker, dict):
                    market_data['volume_24h'] = float(ticker.get('volume', 0))
                    market_data['price_change_24h'] = float(ticker.get('percentage', 0))
                
            except Exception as e:
                logger.warning(f"Could not fetch additional market data for {symbol}: {str(e)}")
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting comprehensive market data for {symbol}: {str(e)}")
            return {'symbol': symbol, 'returns': [], 'volatility': 0.0}
    
    def _create_base_signal(self, symbol: str, df: pd.DataFrame, conditions: Dict) -> Dict:
        """Create base trading signal."""
        try:
            current_price = float(df['close'].iloc[-1])
            
            signal = {
                'symbol': symbol,
                'action': 'hold',
                'strength': 0.0,
                'confidence': 0.0,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat(),
                'conditions': conditions,
                'position_size': 0.01  # Default position size
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error creating base signal: {str(e)}")
            return {'action': 'hold', 'strength': 0.0, 'confidence': 0.0}
    
    def _calculate_portfolio_returns(self, positions: List[Dict]) -> Dict:
        """Calculate portfolio returns for quantitative analysis."""
        try:
            if not positions:
                return {'total_return': 0.0, 'positions': 0}
            
            total_pnl = 0.0
            for position in positions:
                unrealized_pnl = float(position.get('unrealizedPnl', 0))
                total_pnl += unrealized_pnl
            
            return {
                'total_return': total_pnl,
                'positions': len(positions)
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio returns: {str(e)}")
            return {'total_return': 0.0, 'positions': 0}
    
    def _get_trend(self, df: pd.DataFrame) -> str:
        """Get trend direction from dataframe."""
        try:
            if len(df) < 20:
                return 'neutral'
            
            sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            if current_price > sma_20 * 1.02:
                return 'uptrend'
            elif current_price < sma_20 * 0.98:
                return 'downtrend'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error getting trend: {str(e)}")
            return 'neutral'
    
    def _store_signal_history(self, symbol: str, signal: Dict) -> None:
        """Store signal history for analysis."""
        try:
            if symbol not in self.signal_history:
                self.signal_history[symbol] = []
            
            self.signal_history[symbol].append({
                'timestamp': datetime.now().isoformat(),
                'signal': signal
            })
            
            # Keep only last 100 signals per symbol
            if len(self.signal_history[symbol]) > 100:
                self.signal_history[symbol] = self.signal_history[symbol][-100:]
                
        except Exception as e:
            logger.error(f"Error storing signal history: {str(e)}")
    
    async def _log_quantitative_analysis(self, symbol: str, signal: Dict, validation: Dict) -> None:
        """Log quantitative analysis results."""
        try:
            analysis_entry = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'signal': signal,
                'validation': validation
            }
            
            self.quantitative_analysis_history[symbol] = analysis_entry
            
            logger.info(f"Quantitative analysis for {symbol}: "
                       f"Action={signal.get('action', 'HOLD')}, "
                       f"Confidence={signal.get('confidence', 0):.3f}, "
                       f"Validation={validation.get('is_valid', False)}")
                       
        except Exception as e:
            logger.error(f"Error logging quantitative analysis: {str(e)}")
    
    async def get_quantitative_recommendations(self, symbol: str) -> Dict:
        """Get quantitative trading recommendations."""
        try:
            recommendations = await self.quantitative_system.get_recommendations(symbol)
            return recommendations
            
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            logger.error(f"Error getting quantitative recommendations for {symbol}: {str(e)}")
            return {}
    
    async def analyze_portfolio_optimization(self, symbols: List[str]) -> Dict:
        """Analyze portfolio optimization opportunities."""
        try:
            # Get historical data for all symbols
            returns_data = {}
            for symbol in symbols[:10]:  # Limit to first 10 symbols to avoid overload
                try:
                    # Get historical data
                    klines = await self.indicator_service.get_klines(symbol, '1d', limit=100)
                    if klines and 'close' in klines:
                        # Calculate returns
                        prices = pd.Series(klines['close'])
                        returns = prices.pct_change().dropna()
                        if len(returns) > 0:
                            returns_data[symbol] = returns
                except Exception as e:
                    logger.warning(f"Could not get data for {symbol}: {str(e)}")
                    continue
            
            if len(returns_data) < 2:
                return {'error': 'Insufficient data for portfolio optimization'}
            
            # Convert to DataFrame
            returns_df = pd.DataFrame(returns_data)
            
            # Call optimize_portfolio with proper data
            optimization = self.quantitative_system.optimize_portfolio(returns_df)
            return optimization
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio optimization: {str(e)}")
            return {'error': str(e)}
    
    async def analyze_factor_exposures(self, symbols: List[str]) -> Dict:
        """Analyze factor exposures for portfolio."""
        try:
            # Get historical data for factor analysis
            returns_data = {}
            for symbol in symbols[:10]:  # Limit to first 10 symbols
                try:
                    klines = await self.indicator_service.get_klines(symbol, '1d', limit=100)
                    if klines and 'close' in klines:
                        prices = pd.Series(klines['close'])
                        returns = prices.pct_change().dropna()
                        if len(returns) > 0:
                            returns_data[symbol] = returns
                except Exception as e:
                    logger.warning(f"Could not get data for {symbol}: {str(e)}")
                    continue
            
            if len(returns_data) < 2:
                return {'error': 'Insufficient data for factor analysis'}
            
            # Convert to DataFrame
            returns_df = pd.DataFrame(returns_data)
            
            # Use factor model directly
            factor_results = self.quantitative_system.factor_model.build_factor_model(returns_df)
            return factor_results
            
        except Exception as e:
            logger.error(f"Error analyzing factor exposures: {str(e)}")
            return {'error': str(e)}
    
    async def check_profit_target(self) -> bool:
        """Check if profit target has been reached."""  
        try:
            # Check if profit target is enabled
            if not self.config.get('trading', {}).get('enable_check_profit_target', False):
                return False

            # Get current positions
            positions = await self.binance_service.get_positions()
            
            if not positions:
                return False
            
            total_pnl = 0.0
            for position in positions:
                unrealized_pnl = float(position.get('unrealizedPnl', 0))
                total_pnl += unrealized_pnl
            
            # Check against profit target from config
            profit_target = float(self.config.get('trading', {}).get('profit_target', 0.05))  # 5% default
            
            if total_pnl > profit_target:
                logger.info(f"Profit target reached: {total_pnl:.2%}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking profit target: {str(e)}")
            return False
    
    async def process_trading_signals(self, signals: Dict) -> None:
        """Process trading signals and execute trades for futures trading with HEDGING mode."""
        try:
            if not signals or signals.get('action') == 'hold':
                logger.info(f"Signal for {signals.get('symbol')} is hold")
                return
            
            symbol = signals.get('symbol')
            action = signals.get('action', 'hold')
            
            # Apply quantitative validation
            market_data = await self._get_comprehensive_market_data(symbol)
            validation = await self.quantitative_system.validate_signal(signals, market_data)
            
            # if not validation.get('is_valid', False):
            #     logger.info(f"Signal for {symbol} failed quantitative validation")
            #     return
            
            logger.info(f"Signal for {symbol} is valid")
            
            # Check confidence threshold
            confidence = signals.get('confidence', 0)
            threshold = self._calculate_dynamic_confidence_threshold(action, market_data)
            
            # Execute trade based on signal action
            if action == 'buy':
                # Open LONG position
                await self._execute_buy_order(symbol, signals)
                
            elif action == 'sell':
                # Open SHORT position
                await self._execute_sell_order(symbol, signals)
                
            elif action == 'close_long':
                # Close LONG position
                await self._close_long_position(symbol, signals)
                
            elif action == 'close_short':
                # Close SHORT position
                await self._close_short_position(symbol, signals)
                
            elif action == 'close_all':
                # Close all positions for this symbol
                await self._close_long_position(symbol, signals)
                await self._close_short_position(symbol, signals)
                
            else:
                logger.warning(f"Unknown action '{action}' for {symbol}")
                return
            
            # Track confidence performance
            await self._track_confidence_performance(
                action,
                confidence,
                threshold,
                market_data,
                validation.get('risk_metrics', {})
            )
                
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            logger.error(f"Error processing trading signals: {str(e)}")
    
    async def _execute_buy_order(self, symbol: str, signals: Dict) -> None:
        """Execute LONG position order (futures trading with HEDGING mode)."""
        try:
            current_price = signals.get('current_price', 0.0)
            
            if current_price <= 0:
                logger.error(f"Invalid current price for {symbol}: {current_price}")
                return
            
            # Calculate position size using risk management
            risk_per_trade = self.config.get('risk_management', {}).get('risk_per_trade', 0.02)  # 2% risk per trade
            position_size = await self._calculate_position_size(symbol, risk_per_trade, current_price)
            
            if position_size is None:
                logger.warning(f"Could not calculate position size for {symbol}")
                return
            
            # Calculate stop loss and take profit for LONG position
            atr = signals.get('atr', current_price * 0.02)  # Default ATR
            stop_loss = await self._calculate_stop_loss(symbol, "LONG", current_price, atr)
            take_profit = await self._calculate_take_profit(symbol, "LONG", current_price, stop_loss)
            
            # Prepare order parameters for LONG position
            order_params = {
                'symbol': symbol,
                'side': 'BUY',
                'type': 'MARKET',
                'positionSide': 'LONG',  # Specify position side for HEDGING mode
                'amount': position_size
            }
            
            # Add stop loss and take profit if calculated
            if stop_loss and stop_loss > 0:
                order_params['stop_loss'] = stop_loss
                logger.info(f"Stop loss calculated for {symbol} LONG: {stop_loss}")
            
            if take_profit and take_profit > current_price:
                order_params['take_profit'] = take_profit
                logger.info(f"Take profit calculated for {symbol} LONG: {take_profit}")
            
            # Place LONG position order
            order = await self.binance_service.place_order(order_params)
            
            if order:
                logger.info(f"LONG position opened for {symbol} with size {position_size} and SL/TP: {order}")
            else:
                logger.error(f"Failed to place LONG order for {symbol}")
            
        except Exception as e:
            logger.error(f"Error executing LONG order for {symbol}: {str(e)}")
    
    async def _execute_sell_order(self, symbol: str, signals: Dict) -> None:
        """Execute SHORT position order (futures trading with HEDGING mode)."""
        try:
            current_price = signals.get('current_price', 0.0)
            
            if current_price <= 0:
                logger.error(f"Invalid current price for {symbol}: {current_price}")
                return
            
            # Calculate position size using risk management
            risk_per_trade = self.config.get('risk_management', {}).get('risk_per_trade', 0.02)  # 2% risk per trade
            position_size = await self._calculate_position_size(symbol, risk_per_trade, current_price)
            
            if position_size is None:
                logger.warning(f"Could not calculate position size for {symbol}")
                return
            
            # Calculate stop loss and take profit for SHORT position
            atr = signals.get('atr', current_price * 0.02)  # Default ATR
            stop_loss = await self._calculate_stop_loss(symbol, "SHORT", current_price, atr)
            take_profit = await self._calculate_take_profit(symbol, "SHORT", current_price, stop_loss)
            
            # Prepare order parameters for SHORT position
            order_params = {
                'symbol': symbol,
                'side': 'SELL',
                'type': 'MARKET',
                'positionSide': 'SHORT',  # Specify position side for HEDGING mode
                'amount': position_size
            }
            
            # Add stop loss and take profit if calculated
            if stop_loss and stop_loss > current_price:
                order_params['stop_loss'] = stop_loss
                logger.info(f"Stop loss calculated for {symbol} SHORT: {stop_loss}")
            
            if take_profit and take_profit < current_price:
                order_params['take_profit'] = take_profit
                logger.info(f"Take profit calculated for {symbol} SHORT: {take_profit}")
            
            # Place SHORT position order
            order = await self.binance_service.place_order(order_params)
            
            if order:
                logger.info(f"SHORT position opened for {symbol} with size {position_size} and SL/TP: {order}")
            else:
                logger.error(f"Failed to place SHORT order for {symbol}")
            
        except Exception as e:
            logger.error(f"Error executing SHORT order for {symbol}: {str(e)}")
    
    async def _close_long_position(self, symbol: str, signals: Dict) -> None:
        """Close LONG position using binance_service.close_position()."""
        try:
            # Use the existing close_position method from binance_service
            success = await self.binance_service.close_position(symbol, 'LONG')
            
            if success:
                logger.info(f"LONG position closed successfully for {symbol}")
            else:
                logger.warning(f"Failed to close LONG position for {symbol}")
            
        except Exception as e:
            logger.error(f"Error closing LONG position for {symbol}: {str(e)}")
    
    async def _close_short_position(self, symbol: str, signals: Dict) -> None:
        """Close SHORT position using binance_service.close_position()."""
        try:
            # Use the existing close_position method from binance_service
            success = await self.binance_service.close_position(symbol, 'SHORT')
            
            if success:
                logger.info(f"SHORT position closed successfully for {symbol}")
            else:
                logger.warning(f"Failed to close SHORT position for {symbol}")
            
        except Exception as e:
            logger.error(f"Error closing SHORT position for {symbol}: {str(e)}")
    
    async def _calculate_stop_loss(self, symbol: str, position_type: str, current_price: float, atr: float) -> float:
        """Calculate stop loss price based on ATR and market conditions."""
        try:
            # Get stop loss multiplier from config
            stop_loss_multiplier = float(self.config.get('risk_management', {}).get('stop_loss_atr_multiplier', 2.0))
            
            # Calculate base stop loss using ATR
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
                stop_loss = float(current_price) + (float(atr) * stop_loss_multiplier/2)
            
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
                    stop_loss = float(current_price) + (float(atr) * stop_loss_multiplier * 1.5/2)
            

            # Ensure minimum distance from current price
            min_distance = float(self.config.get('risk_management', {}).get('min_stop_distance', 0.01))
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
            risk_reward_ratio = self.config.get('risk_management', {}).get('take_profit_multiplier', 2.0)
            
            # Calculate price difference between current price and stop loss
            price_diff = abs(current_price - stop_loss)
            
            # Calculate take profit based on risk-reward ratio
            if is_long_side(position_type):
                take_profit = current_price + (price_diff * risk_reward_ratio)
            else:
                take_profit = current_price - (price_diff * risk_reward_ratio/8)
            
            # Ensure minimum distance from current price
            min_distance = float(self.config.get('risk_management', {}).get('min_tp_distance', 0.01))
            if is_long_side(position_type):
                # For LONG positions, ensure take profit is above current price
                take_profit = max(take_profit, current_price * (1 + min_distance))
            else:
                # For SHORT positions, ensure take profit is below current price
                take_profit = min(take_profit, current_price * (1.03 - min_distance))
                
                # Additional validation for SHORT positions
                if take_profit <= 0:
                    # If take profit is negative, set it to a reasonable percentage below current price
                    take_profit = current_price * 0.5  # 50% below current price
                elif take_profit >= current_price:
                    # If take profit is above current price, set it to a reasonable percentage below
                    take_profit = current_price * 0.9  # 10% below current price
            
            logger.info(f"Calculated take profit for {symbol} {position_type.lower()}: {take_profit} (current price: {current_price})")
            return take_profit
            
        except Exception as e:
            logger.error(f"Error calculating take profit for {symbol}: {str(e)}")
            return None

    async def _get_market_conditions(self, symbol: str) -> Dict:
        """Get market conditions for stop loss adjustment."""
        try:
            # Get recent price data for volatility calculation
            klines = await self.indicator_service.get_klines(symbol, '1h', limit=24)
            if klines is not None and isinstance(klines, dict) and 'close' in klines and len(klines['close']) > 1:
                # Handle dictionary format with list
                prices = np.array(klines['close'])
                
                if len(prices) > 1:
                    returns = np.diff(np.log(prices))
                    volatility = float(np.std(returns) * np.sqrt(252))
                    
                    return {
                        'volatility': volatility,
                        'price_change_24h': float((prices[-1] / prices[0] - 1) * 100)
                    }
            elif klines is not None and hasattr(klines, 'values'):
                # Handle pandas DataFrame format
                if len(klines['close']) > 1:
                    prices = klines['close'].values
                    returns = np.diff(np.log(prices))
                    volatility = float(np.std(returns) * np.sqrt(252))
                    
                    return {
                        'volatility': volatility,
                        'price_change_24h': float((prices[-1] / prices[0] - 1) * 100)
                    }
            else:
                # Fallback to default values
                logger.warning(f"Unexpected klines format for {symbol}: {type(klines)}")
            
            return {'volatility': 0.02, 'price_change_24h': 0.0}
            
        except Exception as e:
            logger.error(f"Error getting market conditions for {symbol}: {str(e)}")
            return {'volatility': 0.02, 'price_change_24h': 0.0}
    
    async def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics with WorldQuant-level confidence analytics."""
        try:
            metrics = {
                'quantitative_integration_status': self.quantitative_integration.get_integration_status(),
                'signal_history_count': len(self.signal_history),
                'quantitative_analysis_count': len(self.quantitative_analysis_history),
                'cache_size': len(self.data_cache)
            }
            
            # Calculate signal success rate
            total_signals = 0
            successful_signals = 0
            
            for symbol_signals in self.signal_history.values():
                for signal_entry in symbol_signals:
                    total_signals += 1
                    signal = signal_entry['signal']
                    if signal.get('quantitative_confidence', 0) > 0.5:
                        successful_signals += 1
            
            if total_signals > 0:
                metrics['signal_success_rate'] = successful_signals / total_signals
            else:
                metrics['signal_success_rate'] = 0.0
            
            # Add confidence performance analytics
            if hasattr(self, 'confidence_performance'):
                metrics['confidence_analytics'] = {
                    'buy_executions': self.confidence_performance['buy']['executions'],
                    'sell_executions': self.confidence_performance['sell']['executions'],
                    'buy_avg_threshold': self.confidence_performance['thresholds']['buy']['avg_threshold'],
                    'sell_avg_threshold': self.confidence_performance['thresholds']['sell']['avg_threshold'],
                    'buy_threshold_count': self.confidence_performance['thresholds']['buy']['count'],
                    'sell_threshold_count': self.confidence_performance['thresholds']['sell']['count']
                }
                
                # Calculate confidence efficiency metrics
                if self.confidence_performance['buy']['executions'] > 0:
                    metrics['confidence_analytics']['buy_success_rate'] = (
                        self.confidence_performance['buy']['successes'] / 
                        self.confidence_performance['buy']['executions']
                    )
                else:
                    metrics['confidence_analytics']['buy_success_rate'] = 0.0
                
                if self.confidence_performance['sell']['executions'] > 0:
                    metrics['confidence_analytics']['sell_success_rate'] = (
                        self.confidence_performance['sell']['successes'] / 
                        self.confidence_performance['sell']['executions']
                    )
                else:
                    metrics['confidence_analytics']['sell_success_rate'] = 0.0
            
            # Add quantitative system metrics
            if hasattr(self, 'quantitative_system'):
                qs_metrics = self.quantitative_system.get_performance_metrics()
                metrics['quantitative_system'] = qs_metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {}
    
    async def close(self):
        """Close the strategy and cleanup resources."""
        try:
            # Close quantitative components
            if hasattr(self, 'quantitative_integration'):
                await self.quantitative_integration.close()
            
            if hasattr(self, 'quantitative_system'):
                await self.quantitative_system.close()
            
            # Clear caches
            self.signal_history.clear()
            self.quantitative_analysis_history.clear()
            self.data_cache.clear()
            
            logger.info("Enhanced Trading Strategy with Quantitative Analysis closed")
            
        except Exception as e:
            logger.error(f"Error closing enhanced trading strategy: {str(e)}") 
    
    async def _apply_market_microstructure_analysis(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """Apply market microstructure analysis to signal."""
        try:
            enhanced_signal = signal.copy()
            
            # Order flow analysis
            if 'orderbook' in market_data:
                orderbook = market_data['orderbook']
                bid_ask_spread = self._calculate_bid_ask_spread(orderbook)
                order_imbalance = self._calculate_order_imbalance(orderbook)
                
                # Adjust signal based on microstructure
                if bid_ask_spread < 0.001:  # Tight spread
                    enhanced_signal['strength'] += 0.1
                    enhanced_signal['reasons'].append('tight_spread')
                
                if order_imbalance > 0.2:  # Strong buy pressure
                    enhanced_signal['strength'] += 0.15
                    enhanced_signal['reasons'].append('buy_imbalance')
                elif order_imbalance < -0.2:  # Strong sell pressure
                    enhanced_signal['strength'] -= 0.15
                    enhanced_signal['reasons'].append('sell_imbalance')
            
            # Volume profile analysis
            if 'trades' in market_data and isinstance(market_data['trades'], pd.DataFrame):
                volume_profile = self._analyze_volume_profile(market_data['trades'])
                enhanced_signal['volume_profile'] = volume_profile
                
                # Adjust based on volume profile
                if volume_profile.get('high_volume_nodes', []):
                    enhanced_signal['strength'] += 0.05
                    enhanced_signal['reasons'].append('volume_support')
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error applying market microstructure analysis: {str(e)}")
            return signal
    
    def _calculate_bid_ask_spread(self, orderbook: Dict) -> float:
        """Calculate bid-ask spread."""
        try:
            if 'bids' in orderbook and 'asks' in orderbook and orderbook['bids'] and orderbook['asks']:
                best_bid = float(orderbook['bids'][0][0])
                best_ask = float(orderbook['asks'][0][0])
                return (best_ask - best_bid) / best_bid
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating bid-ask spread: {str(e)}")
            return 0.0
    
    def _calculate_order_imbalance(self, orderbook: Dict) -> float:
        """Calculate order imbalance."""
        try:
            if 'bids' in orderbook and 'asks' in orderbook:
                bid_volume = sum(float(bid[1]) for bid in orderbook['bids'][:5])
                ask_volume = sum(float(ask[1]) for ask in orderbook['asks'][:5])
                total_volume = bid_volume + ask_volume
                
                if total_volume > 0:
                    return (bid_volume - ask_volume) / total_volume
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating order imbalance: {str(e)}")
            return 0.0
    
    def _analyze_volume_profile(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze volume profile from trades data."""
        try:
            if trades_df.empty:
                return {}
            
            # Calculate volume-weighted average price (VWAP)
            if 'price' in trades_df.columns and 'qty' in trades_df.columns:
                vwap = (trades_df['price'] * trades_df['qty']).sum() / trades_df['qty'].sum()
                
                # Find high volume nodes
                price_bins = pd.cut(trades_df['price'], bins=20)
                volume_by_price = trades_df.groupby(price_bins)['qty'].sum()
                high_volume_nodes = volume_by_price[volume_by_price > volume_by_price.quantile(0.8)].index.tolist()
                
                return {
                    'vwap': vwap,
                    'high_volume_nodes': high_volume_nodes,
                    'volume_distribution': volume_by_price.to_dict()
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error analyzing volume profile: {str(e)}")
            return {}
    
    async def _apply_advanced_risk_management(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """Apply advanced risk management to signal."""
        try:
            risk_adjusted_signal = signal.copy()
            
            # Calculate dynamic VaR
            if 'returns' in market_data and len(market_data['returns']) > 30:
                var_95 = np.percentile(market_data['returns'], 5)
                var_99 = np.percentile(market_data['returns'], 1)
                
                risk_adjusted_signal['var_95'] = var_95
                risk_adjusted_signal['var_99'] = var_99
                
                # Adjust position size based on VaR
                if abs(var_95) > 0.05:  # High volatility
                    risk_adjusted_signal['position_size'] *= 0.5
                    risk_adjusted_signal['reasons'].append('high_volatility_reduction')
            
            # Correlation analysis with portfolio
            if 'portfolio_returns' in market_data:
                correlation = self._calculate_portfolio_correlation(symbol, market_data)
                risk_adjusted_signal['portfolio_correlation'] = correlation
                
                # Reduce position if high correlation
                if abs(correlation) > 0.7:
                    risk_adjusted_signal['position_size'] *= 0.7
                    risk_adjusted_signal['reasons'].append('high_correlation_reduction')
            
            # Maximum drawdown protection
            max_dd = self._calculate_max_drawdown(market_data.get('returns', []))
            risk_adjusted_signal['max_drawdown'] = max_dd
            
            if max_dd > 0.2:  # High drawdown
                risk_adjusted_signal['position_size'] *= 0.6
                risk_adjusted_signal['reasons'].append('drawdown_protection')
            
            return risk_adjusted_signal
            
        except Exception as e:
            logger.error(f"Error applying advanced risk management: {str(e)}")
            return signal
    
    def _calculate_portfolio_correlation(self, symbol: str, market_data: Dict) -> float:
        """Calculate correlation with existing portfolio."""
        try:
            if 'returns' in market_data and 'portfolio_returns' in market_data:
                symbol_returns = pd.Series(market_data['returns'])
                portfolio_returns = pd.Series(market_data['portfolio_returns'].get(symbol, []))
                
                if len(symbol_returns) > 10 and len(portfolio_returns) > 10:
                    # Align series
                    min_length = min(len(symbol_returns), len(portfolio_returns))
                    correlation = symbol_returns.iloc[-min_length:].corr(portfolio_returns.iloc[-min_length:])
                    return correlation if not pd.isna(correlation) else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating portfolio correlation: {str(e)}")
            return 0.0
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns."""
        try:
            if not returns or len(returns) < 2:
                return 0.0
            
            returns_array = np.array(returns)
            cumulative = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return float(abs(drawdown.min()))
            
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0
    
    async def _apply_statistical_arbitrage(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """Apply statistical arbitrage analysis."""
        try:
            stat_arb_signal = signal.copy()
            
            # Cointegration analysis with major pairs
            cointegration_signals = await self._analyze_cointegration(symbol, market_data)
            stat_arb_signal['cointegration'] = cointegration_signals
            
            # Pairs trading opportunities
            if cointegration_signals.get('cointegrated_pairs'):
                stat_arb_signal['strength'] += 0.1
                stat_arb_signal['reasons'].append('statistical_arbitrage')
            
            # Mean reversion analysis
            mean_reversion = self._analyze_mean_reversion(market_data.get('returns', []))
            stat_arb_signal['mean_reversion'] = mean_reversion
            
            if mean_reversion.get('is_mean_reverting'):
                if mean_reversion.get('deviation') > 2:  # Strong deviation
                    stat_arb_signal['strength'] += 0.15
                    stat_arb_signal['reasons'].append('mean_reversion_opportunity')
            
            return stat_arb_signal
            
        except Exception as e:
            logger.error(f"Error applying statistical arbitrage: {str(e)}")
            return signal
    
    async def _analyze_cointegration(self, symbol: str, market_data: Dict) -> Dict:
        """Analyze cointegration with other symbols."""
        try:
            # This would require data from multiple symbols
            # For now, return basic structure
            return {
                'cointegrated_pairs': [],
                'cointegration_score': 0.0,
                'spread_zscore': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cointegration: {str(e)}")
            return {}
    
    def _analyze_mean_reversion(self, returns: List[float]) -> Dict:
        """Analyze mean reversion characteristics."""
        try:
            if not returns or len(returns) < 30:
                return {'is_mean_reverting': False, 'deviation': 0.0}
            
            returns_array = np.array(returns)
            mean = float(np.mean(returns_array))
            std = float(np.std(returns_array))
            current_return = float(returns_array[-1])
            
            deviation = (current_return - mean) / std
            
            return {
                'is_mean_reverting': abs(deviation) > 1.5,
                'deviation': float(deviation),
                'mean': mean,
                'std': std
            }
            
        except Exception as e:
            import traceback
            logger.error(f"Error analyzing mean reversion: {str(e)}")
            logger.error(traceback.format_exc())
            return {'is_mean_reverting': False, 'deviation': 0.0}
    
    async def _apply_momentum_mean_reversion_analysis(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """Apply momentum and mean reversion analysis."""
        try:
            momentum_signal = signal.copy()
            
            # Momentum analysis
            if 'returns' in market_data and len(market_data['returns']) > 20:
                returns = np.array(market_data['returns'])
                
                # Short-term momentum (5 periods)
                short_momentum = float(np.mean(returns[-5:]))
                
                # Medium-term momentum (20 periods)
                medium_momentum = float(np.mean(returns[-20:]))
                
                # Long-term momentum (60 periods)
                long_momentum = float(np.mean(returns[-60:])) if len(returns) >= 60 else medium_momentum
                
                momentum_signal['momentum'] = {
                    'short_term': short_momentum,
                    'medium_term': medium_momentum,
                    'long_term': long_momentum
                }
                
                # Momentum signal adjustment
                if short_momentum > 0.01 and medium_momentum > 0.005:
                    momentum_signal['strength'] += 0.1
                    momentum_signal['reasons'].append('positive_momentum')
                elif short_momentum < -0.01 and medium_momentum < -0.005:
                    momentum_signal['strength'] -= 0.1
                    momentum_signal['reasons'].append('negative_momentum')
                
                # Mean reversion signal
                if abs(short_momentum) > 0.02 and abs(short_momentum - medium_momentum) > 0.01:
                    if short_momentum > medium_momentum:
                        momentum_signal['strength'] -= 0.05  # Revert from high
                        momentum_signal['reasons'].append('momentum_reversion')
                    else:
                        momentum_signal['strength'] += 0.05  # Revert from low
                        momentum_signal['reasons'].append('momentum_reversion')
            
            return momentum_signal
            
        except Exception as e:
            logger.error(f"Error applying momentum analysis: {str(e)}")
            return signal
    
    async def _apply_volatility_regime_analysis(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """Apply volatility regime analysis."""
        try:
            volatility_signal = signal.copy()
            
            if 'returns' in market_data and len(market_data['returns']) > 50:
                returns = np.array(market_data['returns'])
                
                # Calculate rolling volatility
                rolling_vol = pd.Series(returns).rolling(20).std()
                current_vol = float(rolling_vol.iloc[-1])
                avg_vol = float(rolling_vol.mean())
                
                # Volatility regime classification
                if current_vol > avg_vol * 1.5:
                    regime = 'high_volatility'
                    volatility_signal['position_size'] *= 0.7
                    volatility_signal['reasons'].append('high_volatility_regime')
                elif current_vol < avg_vol * 0.7:
                    regime = 'low_volatility'
                    volatility_signal['position_size'] *= 1.2
                    volatility_signal['reasons'].append('low_volatility_regime')
                else:
                    regime = 'normal_volatility'
                
                volatility_signal['volatility_regime'] = {
                    'regime': regime,
                    'current_volatility': current_vol,
                    'average_volatility': avg_vol,
                    'volatility_ratio': current_vol / avg_vol
                }
            
            return volatility_signal
            
        except Exception as e:
            logger.error(f"Error applying volatility regime analysis: {str(e)}")
            return signal
    
    async def _apply_correlation_analysis(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """Apply correlation analysis with market indices."""
        try:
            correlation_signal = signal.copy()
            
            # This would require market index data
            # For now, implement basic correlation logic
            correlation_signal['market_correlation'] = {
                'btc_correlation': 0.0,
                'eth_correlation': 0.0,
                'market_beta': 1.0
            }
            
            return correlation_signal
            
        except Exception as e:
            logger.error(f"Error applying correlation analysis: {str(e)}")
            return signal
    
    async def _optimize_final_signal(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """Optimize final signal using machine learning techniques."""
        try:
            optimized_signal = signal.copy()
            
            # Preserve current_price
            current_price = signal.get('current_price', 0.0)
            
            # Signal strength normalization - use 'strength' key instead of 'signal_strength'
            signal_strength = optimized_signal.get('strength', 0.0)
            optimized_signal['signal_strength'] = np.clip(signal_strength, -1.0, 1.0)
            
            # Confidence calculation
            confidence_factors = [
                abs(optimized_signal['signal_strength']),
                optimized_signal.get('confidence', 0.0),
                len(optimized_signal.get('reasons', [])) / 10.0  # More reasons = higher confidence
            ]
            
            optimized_signal['final_confidence'] = float(np.mean(confidence_factors))
            
            # Position size optimization
            base_size = optimized_signal.get('position_size', 0.01)
            confidence_multiplier = optimized_signal['final_confidence']
            volatility_adjustment = 1.0 / (1.0 + optimized_signal.get('volatility_regime', {}).get('volatility_ratio', 1.0))
            
            optimized_signal['optimized_position_size'] = base_size * confidence_multiplier * volatility_adjustment
            
            # Risk-adjusted signal strength
            risk_adjustment = 1.0 - abs(optimized_signal.get('var_95', 0.0)) * 10
            optimized_signal['risk_adjusted_strength'] = optimized_signal['signal_strength'] * risk_adjustment
            
            # Ensure current_price is preserved
            optimized_signal['current_price'] = current_price
            
            return optimized_signal
            
        except Exception as e:
            logger.error(f"Error optimizing final signal: {str(e)}")
            return signal
    
    async def _optimize_position_size_advanced(self, symbol: str, base_size: float, market_data: Dict, signal: Dict) -> float:
        """Advanced position size optimization using Kelly Criterion and risk metrics."""
        try:
            # Kelly Criterion calculation
            if 'returns' in market_data and len(market_data['returns']) > 30:
                returns = np.array(market_data['returns'])
                positive_returns_mask = returns > 0
                negative_returns_mask = returns < 0
                positive_count = float(np.sum(positive_returns_mask))
                negative_count = float(np.sum(negative_returns_mask))
                
                win_rate = positive_count / len(returns)
                avg_win = float(np.mean(returns[positive_returns_mask])) if positive_count > 0 else 0.001
                avg_loss = abs(float(np.mean(returns[negative_returns_mask]))) if negative_count > 0 else 0.001
                
                # Kelly fraction
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = float(np.clip(kelly_fraction, 0.0, 0.25))  # Cap at 25%
            else:
                kelly_fraction = 0.02  # Default 2%
            
            # Volatility adjustment
            returns_array = market_data.get('returns', [0.01])
            volatility = float(np.std(returns_array)) if returns_array else 0.01
            volatility_adjustment = 1.0 / (1.0 + volatility * 10)
            
            # Correlation adjustment
            correlation = signal.get('portfolio_correlation', 0.0)
            correlation_adjustment = 1.0 - abs(correlation) * 0.5
            
            # Final position size
            final_size = base_size * kelly_fraction * volatility_adjustment * correlation_adjustment
            
            # Safety limits
            final_size = float(np.clip(final_size, 0.001, 0.1))  # Between 0.1% and 10%
            
            return final_size
            
        except Exception as e:
            import traceback
            logger.error(f"Error optimizing position size: {str(e)}")
            logger.error(traceback.format_exc())
            return base_size 
    
    def _calculate_dynamic_thresholds(self, market_data: pd.DataFrame, 
                                    volatility_regime: str = None,
                                    risk_metrics: Dict = None) -> Dict[str, float]:
        """
        Calculate dynamic thresholds based on WorldQuant-level analysis.
        
        Args:
            market_data: Historical market data
            volatility_regime: Current volatility regime
            risk_metrics: Risk metrics including VaR, Sharpe ratio
            
        Returns:
            Dict with dynamic thresholds
        """
        try:
            # Base thresholds
            base_buy_threshold = 0.15
            base_sell_threshold = -0.15
            
            # Volatility adjustment
            if volatility_regime:
                if volatility_regime == 'high':
                    # Higher thresholds in high volatility
                    vol_adjustment = 0.1
                elif volatility_regime == 'low':
                    # Lower thresholds in low volatility
                    vol_adjustment = -0.05
                else:
                    vol_adjustment = 0.0
            else:
                vol_adjustment = 0.0
            
            # Risk-adjusted thresholds
            if risk_metrics:
                sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
                var_95 = risk_metrics.get('var_95', 0)
                max_drawdown = risk_metrics.get('max_drawdown', 0)
                
                # Adjust based on Sharpe ratio
                if sharpe_ratio > 1.0:
                    sharpe_adjustment = -0.05  # Lower threshold for good performance
                elif sharpe_ratio < 0.5:
                    sharpe_adjustment = 0.05   # Higher threshold for poor performance
                else:
                    sharpe_adjustment = 0.0
                
                # Adjust based on VaR
                if var_95 < -0.02:  # High risk
                    var_adjustment = 0.03
                elif var_95 > -0.01:  # Low risk
                    var_adjustment = -0.02
                else:
                    var_adjustment = 0.0
                
                # Adjust based on drawdown
                if max_drawdown > 0.1:  # High drawdown
                    drawdown_adjustment = 0.05
                else:
                    drawdown_adjustment = 0.0
            else:
                sharpe_adjustment = 0.0
                var_adjustment = 0.0
                drawdown_adjustment = 0.0
            
            # Market regime adjustment
            market_regime = self._detect_market_regime(market_data)
            if market_regime == 'trending':
                regime_adjustment = -0.03  # Lower threshold in trending markets
            elif market_regime == 'mean_reverting':
                regime_adjustment = 0.03   # Higher threshold in mean-reverting markets
            else:
                regime_adjustment = 0.0
            
            # Calculate final thresholds
            buy_threshold = base_buy_threshold + vol_adjustment + sharpe_adjustment + \
                          var_adjustment + drawdown_adjustment + regime_adjustment
            
            sell_threshold = base_sell_threshold - vol_adjustment - sharpe_adjustment - \
                           var_adjustment - drawdown_adjustment - regime_adjustment
            
            # Ensure reasonable bounds
            buy_threshold = max(0.05, min(0.4, buy_threshold))
            sell_threshold = max(-0.4, min(-0.05, sell_threshold))
            
            return {
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold,
                'volatility_regime': volatility_regime,
                'market_regime': market_regime,
                'risk_metrics': risk_metrics
            }
            
        except Exception as e:
            logger.error(f"Error calculating dynamic thresholds: {str(e)}")
            return {
                'buy_threshold': 0.25,
                'sell_threshold': -0.25,
                'volatility_regime': 'unknown',
                'market_regime': 'unknown',
                'risk_metrics': None
            }
    
    def _detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """
        Detect current market regime using advanced statistical methods.
        
        Args:
            market_data: Historical market data
            
        Returns:
            Market regime: 'trending', 'mean_reverting', 'volatile', 'stable'
        """
        try:
            if len(market_data) < 50:
                return 'unknown'
            
            # Calculate returns
            returns = market_data['close'].pct_change().dropna()
            
            # Augmented Dickey-Fuller test for stationarity
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(returns)
            is_stationary = adf_result[1] < 0.05
            
            # Hurst exponent for trend detection
            hurst_exponent = self._calculate_hurst_exponent(returns)
            
            # Volatility clustering
            volatility = returns.rolling(window=20).std()
            vol_clustering = volatility.autocorr()
            
            # Regime classification
            if hurst_exponent > 0.6 and not is_stationary:
                return 'trending'
            elif hurst_exponent < 0.4 and is_stationary:
                return 'mean_reverting'
            elif vol_clustering > 0.3:
                return 'volatile'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return 'unknown'
    
    def _calculate_hurst_exponent(self, returns: pd.Series) -> float:
        """
        Calculate Hurst exponent for trend detection.
        
        Args:
            returns: Price returns series
            
        Returns:
            Hurst exponent (H > 0.5: trending, H < 0.5: mean-reverting)
        """
        try:
            if len(returns) < 20:
                return 0.5
            
            # Calculate price range
            prices = (1 + returns).cumprod()
            price_range = prices.max() - prices.min()
            
            # Calculate time range
            time_range = len(returns)
            
            # Calculate Hurst exponent
            if price_range > 0 and time_range > 0:
                hurst = float(np.log(price_range) / np.log(time_range))
                return max(0.1, min(0.9, hurst))  # Bound between 0.1 and 0.9
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating Hurst exponent: {str(e)}")
            return 0.5
    
    def _calculate_risk_metrics(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics for threshold adjustment.
        
        Args:
            market_data: Historical market data
            
        Returns:
            Dict with risk metrics
        """
        try:
            returns = market_data['close'].pct_change().dropna()
            
            if len(returns) < 30:
                return {}
            
            # Calculate risk metrics
            returns_std = float(np.std(returns))
            sharpe_ratio = float(np.mean(returns) / returns_std * np.sqrt(252)) if returns_std > 0 else 0
            var_95 = float(np.percentile(returns, 5))
            max_drawdown = float(self._calculate_max_drawdown(returns))
            volatility = float(returns_std * np.sqrt(252))
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_std = float(np.std(downside_returns)) if len(downside_returns) > 0 else 0
            sortino_ratio = float(np.mean(returns) / downside_std * np.sqrt(252)) if len(downside_returns) > 0 and downside_std > 0 else 0
            
            # Calmar ratio
            cumulative_return = float((1 + returns).prod() - 1)
            calmar_ratio = float(cumulative_return / abs(max_drawdown)) if max_drawdown != 0 else 0
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'var_95': var_95,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'cumulative_return': cumulative_return
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {} 
    
    def _calculate_dynamic_confidence_threshold(self, action: str, market_data: Dict, risk_metrics: Dict = None) -> float:
        """
        Calculate dynamic confidence threshold based on action type, market conditions, and risk metrics.
        WorldQuant-level implementation with asymmetric thresholds for BUY/SELL.
        """
        try:
            # Load configuration
            from src.core.config import load_config
            config = load_config()
            confidence_config = config.get('trading', {}).get('confidence_thresholds', {})
            
            # Base asymmetric thresholds from config
            base_thresholds = {
                'buy': confidence_config.get('buy_base', 0.45),
                'sell': confidence_config.get('sell_base', 0.65),
                'hold': confidence_config.get('hold_base', 0.35)
            }
            
            base_threshold = base_thresholds.get(action, 0.6)
            
            # Volatility adjustment from config
            volatility = market_data.get('volatility', 0.0)
            vol_adjustment = 0.0
            
            vol_adjustments = confidence_config.get('volatility_adjustments', {})
            if volatility > 0.05:  # High volatility
                vol_adjustment = vol_adjustments.get('high_volatility', 0.1)
            elif volatility < 0.01:  # Low volatility
                vol_adjustment = vol_adjustments.get('low_volatility', -0.05)
            
            # Market regime adjustment from config
            market_regime = market_data.get('market_regime', 'normal')
            regime_adjustment = 0.0
            
            regime_adjustments = confidence_config.get('regime_adjustments', {})
            if market_regime == 'trending':
                regime_adjustment = regime_adjustments.get('trending', -0.05)
            elif market_regime == 'mean_reverting':
                regime_adjustment = regime_adjustments.get('mean_reverting', 0.05)
            elif market_regime == 'high_volatility':
                regime_adjustment = regime_adjustments.get('high_volatility', 0.08)
            
            # Risk metrics adjustment from config
            risk_adjustment = 0.0
            if risk_metrics:
                sharpe_ratio = risk_metrics.get('sharpe_ratio', 0.0)
                var_95 = risk_metrics.get('var_95', -0.02)
                max_drawdown = risk_metrics.get('max_drawdown', 0.0)
                
                risk_adjustments = confidence_config.get('risk_adjustments', {})
                
                # Sharpe ratio adjustment
                if sharpe_ratio > 1.0:
                    risk_adjustment += risk_adjustments.get('sharpe_ratio_good', -0.05)
                elif sharpe_ratio < 0.5:
                    risk_adjustment += risk_adjustments.get('sharpe_ratio_poor', 0.05)
                
                # VaR adjustment
                if var_95 < -0.03:  # High risk
                    risk_adjustment += risk_adjustments.get('var_high_risk', 0.03)
                elif var_95 > -0.01:  # Low risk
                    risk_adjustment += risk_adjustments.get('var_low_risk', -0.02)
                
                # Drawdown adjustment
                if max_drawdown > 0.05:
                    risk_adjustment += risk_adjustments.get('drawdown_high', 0.02)
            
            # Calculate final threshold
            final_threshold = base_threshold + vol_adjustment + regime_adjustment + risk_adjustment
            
            # Ensure bounds from config
            bounds = confidence_config.get('bounds', {})
            min_threshold = bounds.get('min_threshold', 0.25)
            max_threshold = bounds.get('max_threshold', 0.85)
            final_threshold = max(min_threshold, min(max_threshold, final_threshold))
            
            logger.info(f"Dynamic confidence threshold for {action}: "
                       f"base={base_threshold:.3f}, vol_adj={vol_adjustment:.3f}, "
                       f"regime_adj={regime_adjustment:.3f}, risk_adj={risk_adjustment:.3f}, "
                       f"final={final_threshold:.3f}")
            
            return final_threshold
            
        except Exception as e:
            logger.error(f"Error calculating dynamic confidence threshold: {str(e)}")
            return 0.6  # Fallback to original threshold
    
    def _calculate_risk_adjusted_confidence(self, signal: Dict, risk_metrics: Dict) -> float:
        """
        Calculate risk-adjusted confidence score based on signal strength and risk metrics.
        WorldQuant-level implementation with comprehensive risk consideration.
        """
        try:
            # Load configuration
            from src.core.config import load_config
            config = load_config()
            confidence_config = config.get('trading', {}).get('confidence_thresholds', {})
            risk_adjusted_config = confidence_config.get('risk_adjusted_confidence', {})
            
            base_confidence = signal.get('confidence', 0.0)
            signal_strength = signal.get('strength', 0.0)
            
            # Base confidence boost from signal strength using config
            strength_multiplier = risk_adjusted_config.get('strength_boost_multiplier', 0.2)
            max_strength_boost = risk_adjusted_config.get('max_strength_boost', 0.1)
            strength_boost = min(max_strength_boost, abs(signal_strength) * strength_multiplier)
            
            # Risk metrics adjustment using config
            risk_boost = 0.0
            
            if risk_metrics:
                sharpe_ratio = risk_metrics.get('sharpe_ratio', 0.0)
                var_95 = risk_metrics.get('var_95', -0.02)
                max_drawdown = risk_metrics.get('max_drawdown', 0.0)
                volatility = risk_metrics.get('volatility', 0.02)
                
                # Sharpe ratio boost
                if sharpe_ratio > 1.0:
                    risk_boost += risk_adjusted_config.get('sharpe_ratio_boost', 0.05)
                elif sharpe_ratio < 0.5:
                    risk_boost += risk_adjusted_config.get('sharpe_ratio_penalty', -0.05)
                
                # VaR boost
                if var_95 > -0.01:  # Low risk
                    risk_boost += risk_adjusted_config.get('var_low_risk_boost', 0.03)
                elif var_95 < -0.03:  # High risk
                    risk_boost += risk_adjusted_config.get('var_high_risk_penalty', -0.03)
                
                # Drawdown penalty
                if max_drawdown > 0.05:
                    risk_boost += risk_adjusted_config.get('drawdown_penalty', -0.02)
                
                # Volatility adjustment
                if volatility < 0.01:  # Low volatility
                    risk_boost += risk_adjusted_config.get('low_volatility_boost', 0.02)
                elif volatility > 0.05:  # High volatility
                    risk_boost += risk_adjusted_config.get('high_volatility_penalty', -0.02)
            
            # Market microstructure boost using config
            microstructure_boost = 0.0
            if 'market_microstructure' in signal:
                microstructure = signal['market_microstructure']
                
                # Volume profile boost
                if microstructure.get('volume_profile_valid', False):
                    microstructure_boost += risk_adjusted_config.get('volume_profile_boost', 0.02)
                
                # Order flow boost
                if microstructure.get('order_flow_bullish', False):
                    microstructure_boost += risk_adjusted_config.get('order_flow_bullish_boost', 0.03)
                elif microstructure.get('order_flow_bearish', False):
                    microstructure_boost += risk_adjusted_config.get('order_flow_bearish_penalty', -0.03)
                
                # Liquidity boost
                if microstructure.get('liquidity_adequate', False):
                    microstructure_boost += risk_adjusted_config.get('liquidity_boost', 0.01)
            
            # Calculate final confidence
            final_confidence = base_confidence + strength_boost + risk_boost + microstructure_boost
            
            # Ensure bounds from config
            confidence_bounds = risk_adjusted_config.get('confidence_bounds', {})
            min_confidence = confidence_bounds.get('min_confidence', 0.05)
            max_confidence = confidence_bounds.get('max_confidence', 0.95)
            final_confidence = max(min_confidence, min(max_confidence, final_confidence))
            
            logger.info(f"Risk-adjusted confidence: base={base_confidence:.3f}, "
                       f"strength_boost={strength_boost:.3f}, risk_boost={risk_boost:.3f}, "
                       f"microstructure_boost={microstructure_boost:.3f}, "
                       f"final={final_confidence:.3f}")
            
            return final_confidence
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted confidence: {str(e)}")
            return signal.get('confidence', 0.0)
    
    async def _track_confidence_performance(self, action: str, confidence: float, threshold: float, 
                                          market_data: Dict, risk_metrics: Dict) -> None:
        """Track confidence performance for optimization."""
        try:
            # Store performance metrics
            performance_key = f"confidence_performance_{action}"
            if performance_key not in self.confidence_performance:
                self.confidence_performance[performance_key] = {
                    'total_signals': 0,
                    'successful_signals': 0,
                    'confidence_scores': [],
                    'thresholds_used': []
                }
            
            self.confidence_performance[performance_key]['total_signals'] += 1
            self.confidence_performance[performance_key]['confidence_scores'].append(confidence)
            self.confidence_performance[performance_key]['thresholds_used'].append(threshold)
            
            # Track successful signals (you can implement your own success criteria)
            if confidence > threshold:
                self.confidence_performance[performance_key]['successful_signals'] += 1
            
            logger.debug(f"Tracked confidence performance for {action}: {confidence:.3f} vs {threshold:.3f}")
            
        except Exception as e:
            logger.error(f"Error tracking confidence performance: {str(e)}")
    
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
            leverage = self.config.get('trading', {}).get('leverage', 10)
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
                
            # Convert to DataFrame - handle different klines formats
            if isinstance(klines[0], list):
                # Binance API format: [timestamp, open, high, low, close, volume, ...]
                # Check actual number of columns in the data
                if len(klines[0]) >= 6:
                    # Use only the first 6 columns to avoid column mismatch
                    df = pd.DataFrame([row[:6] for row in klines], columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume'
                    ])
                else:
                    logger.warning(f"Unexpected klines format for {symbol}")
                    return base_size
            else:
                # Dictionary format
                df = pd.DataFrame(klines)
            
            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate volatility
            if 'close' in df.columns and len(df) > 1:
                df['returns'] = df['close'].pct_change()
                volatility = df['returns'].std() * np.sqrt(24)  # Annualized volatility
            else:
                logger.warning(f"Insufficient data for volatility calculation for {symbol}")
                return base_size
            
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
    
    async def _get_market_volatility(self) -> Optional[float]:
        """Get market volatility for comparison."""
        try:
            # Get BTC volatility as market benchmark
            btc_klines = await self.binance_service.get_klines(
                symbol='BTCUSDT',
                timeframe='1h',
                limit=24
            )
            
            if not btc_klines:
                return None
                
            # Convert to DataFrame - handle different klines formats
            if isinstance(btc_klines[0], list):
                # Binance API format: [timestamp, open, high, low, close, volume, ...]
                # Check actual number of columns in the data
                if len(btc_klines[0]) >= 6:
                    # Use only the first 6 columns to avoid column mismatch
                    df = pd.DataFrame([row[:6] for row in btc_klines], columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume'
                    ])
                else:
                    logger.warning("Unexpected BTC klines format")
                    return None
            else:
                # Dictionary format
                df = pd.DataFrame(btc_klines)
            
            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate BTC volatility
            if 'close' in df.columns and len(df) > 1:
                df['returns'] = df['close'].pct_change()
                btc_volatility = df['returns'].std() * np.sqrt(24)  # Annualized volatility
                return btc_volatility
            else:
                logger.warning("Insufficient data for BTC volatility calculation")
                return None
                
        except Exception as e:
            logger.error(f"Error getting market volatility: {str(e)}")
            return None
    
    async def _apply_statistical_validation(self, symbol: str, signal: Dict, market_data: Dict) -> Optional[Dict]:
        """Apply statistical validation to signal."""
        try:
            # Validate signal quality
            quality_validation = self.statistical_validator.validate_signal_quality(signal)
            
            if not quality_validation['is_valid']:
                logger.warning(f"Signal for {symbol} failed quality validation: {quality_validation['warnings']}")
                return None
            
            # Get benchmark returns for significance testing
            benchmark_returns = await self._get_benchmark_returns(symbol)
            
            if benchmark_returns is not None:
                # Test signal significance
                significance_result = self.statistical_validator.test_signal_significance(
                    self.signal_history.get(symbol, []), 
                    benchmark_returns
                )
                
                # Only proceed if signal is statistically significant
                if not significance_result.get('significant', False):
                    logger.warning(f"Signal for {symbol} not statistically significant (p_value={significance_result.get('p_value', 1.0):.4f})")
                    return None
                
                # Add statistical validation results to signal
                signal['statistical_validation'] = {
                    'quality_validation': quality_validation,
                    'significance_test': significance_result,
                    'confidence_score': quality_validation['confidence_score']
                }
            
            # Validate market regime stability
            if 'returns' in market_data and len(market_data['returns']) > 60:
                regime_validation = self.statistical_validator.validate_market_regime_stability(
                    np.array(market_data['returns'])
                )
                signal['statistical_validation']['regime_stability'] = regime_validation
            
            # Store validation in history
            self.statistical_validator.validation_history[symbol] = signal.get('statistical_validation', {})
            
            logger.info(f"Statistical validation passed for {symbol}: confidence_score={quality_validation['confidence_score']:.3f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error applying statistical validation for {symbol}: {str(e)}")
            return signal
    
    async def _get_benchmark_returns(self, symbol: str) -> Optional[np.ndarray]:
        """Get benchmark returns for statistical testing."""
        try:
            # Use BTC as benchmark for crypto trading
            btc_klines = await self.indicator_service.get_klines('BTCUSDT', '1h', limit=100)
            
            if btc_klines and 'close' in btc_klines and len(btc_klines['close']) > 1:
                # Calculate BTC returns
                btc_prices = np.array(btc_klines['close'])
                btc_returns = np.diff(np.log(btc_prices))
                return btc_returns
            
            # Fallback: use market average returns
            market_returns = await self._get_market_average_returns()
            return market_returns
            
        except Exception as e:
            logger.error(f"Error getting benchmark returns for {symbol}: {str(e)}")
            return None
    
    async def _get_market_average_returns(self) -> Optional[np.ndarray]:
        """Get market average returns as benchmark."""
        try:
            # Get returns for major pairs
            major_pairs = ['ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
            all_returns = []
            
            for pair in major_pairs:
                try:
                    klines = await self.indicator_service.get_klines(pair, '1h', limit=100)
                    if klines and 'close' in klines and len(klines['close']) > 1:
                        prices = np.array(klines['close'])
                        returns = np.diff(np.log(prices))
                        all_returns.append(returns)
                except Exception as e:
                    logger.warning(f"Could not get returns for {pair}: {str(e)}")
                    continue
            
            if all_returns:
                # Calculate average returns across all pairs
                min_length = min(len(returns) for returns in all_returns)
                aligned_returns = [returns[:min_length] for returns in all_returns]
                average_returns = np.mean(aligned_returns, axis=0)
                return average_returns
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting market average returns: {str(e)}")
            return None
    
    async def perform_walk_forward_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Perform walk-forward analysis for strategy validation."""
        try:
            logger.info(f"Starting walk-forward analysis for {len(symbols)} symbols")
            
            # Get historical data for all symbols
            all_data = {}
            
            for symbol in symbols[:10]:  # Limit to first 10 symbols for performance
                try:
                    klines = await self.indicator_service.get_klines(symbol, '1d', limit=500)
                    if klines and 'close' in klines:
                        prices = pd.Series(klines['close'])
                        returns = prices.pct_change().dropna()
                        all_data[symbol] = returns
                except Exception as e:
                    logger.warning(f"Could not get data for {symbol}: {str(e)}")
                    continue
            
            if len(all_data) < 2:
                logger.warning("Insufficient data for walk-forward analysis")
                return {'success': False, 'error': 'Insufficient data'}
            
            # Convert to DataFrame
            returns_df = pd.DataFrame(all_data)
            
            # Perform walk-forward analysis
            walk_forward_result = self.statistical_validator.perform_walk_forward_analysis(
                self, returns_df
            )
            
            if walk_forward_result['success']:
                logger.info(f"Walk-forward analysis completed successfully")
                return walk_forward_result
            else:
                logger.warning(f"Walk-forward analysis failed: {walk_forward_result.get('error', 'Unknown error')}")
                return walk_forward_result
                
        except Exception as e:
            logger.error(f"Error performing walk-forward analysis: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_statistical_validation_summary(self) -> Dict[str, Any]:
        """Get summary of statistical validation results."""
        try:
            summary = self.statistical_validator.get_validation_summary()
            
            # Add strategy-specific metrics
            summary.update({
                'total_signals_generated': len(self.signal_history),
                'signals_passed_validation': sum(
                    1 for signals in self.signal_history.values() 
                    for signal in signals 
                    if signal.get('statistical_validation', {}).get('quality_validation', {}).get('is_valid', False)
                ),
                'average_confidence_score': np.mean([
                    signal.get('statistical_validation', {}).get('confidence_score', 0)
                    for signals in self.signal_history.values() 
                    for signal in signals
                    if signal.get('statistical_validation')
                ]) if self.signal_history else 0.0
            })
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting statistical validation summary: {str(e)}")
            return {}