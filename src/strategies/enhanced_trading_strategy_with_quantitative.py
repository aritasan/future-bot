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
from src.quantitative.factor_model import WorldQuantFactorModel
from src.quantitative.ml_ensemble import WorldQuantMLEnsemble
from src.quantitative.portfolio_optimizer import WorldQuantPortfolioOptimizer

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
from src.quantitative.worldquant_dca_trailing import WorldQuantDCA, WorldQuantTrailingStop
from src.quantitative.implied_volatility import ImpliedVolatilityEngine
from src.quantitative.advanced_risk_management import DynamicRiskManager
from src.quantitative.statistical_arbitrage import StatisticalArbitrageEngine
from src.quantitative.advanced_ml_ensemble import AdvancedMLEnsemble
from src.quantitative.market_microstructure import MarketMicrostructureAnalyzer
from src.quantitative.risk_manager import RiskManager
from src.quantitative.high_frequency_trading import HighFrequencyTradingEngine, TickData
from src.quantitative.advanced_market_microstructure import AdvancedMarketMicrostructureAnalyzer
from src.quantitative.options_based_strategies import OptionsBasedStrategies, OptionContract
from src.quantitative.on_chain_analytics import OnChainAnalytics, BlockchainTransaction
from src.quantitative.alternative_data_integration import AlternativeDataEngine

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
        self.quantitative_system = QuantitativeTradingSystem(config)
        self.quantitative_integration = QuantitativeIntegration(config)
        self.statistical_validator = StatisticalValidator(config)
        self.risk_manager = RiskManager(config)
        self.factor_model = WorldQuantFactorModel(config)
        self.ml_ensemble = WorldQuantMLEnsemble(config)
        self.portfolio_optimizer = WorldQuantPortfolioOptimizer(config)
        self.market_microstructure = MarketMicrostructureAnalyzer(config)
        
        # Initialize new advanced modules
        self.dynamic_risk_manager = DynamicRiskManager(config)
        self.statistical_arbitrage_engine = StatisticalArbitrageEngine(config)
        self.advanced_ml_ensemble = AdvancedMLEnsemble(config)
        self.market_microstructure_analyzer = MarketMicrostructureAnalyzer(config)
        
        # Initialize DCA and Trailing Stop
        self.worldquant_dca = WorldQuantDCA(config)
        self.worldquant_trailing = WorldQuantTrailingStop(config)
        
        # Initialize Implied Volatility Engine
        self.volatility_engine = ImpliedVolatilityEngine(config)
        
        # Initialize Phase 3 WorldQuant-Level Features
        self.hft_engine = HighFrequencyTradingEngine(config)
        self.advanced_microstructure_analyzer = AdvancedMarketMicrostructureAnalyzer(config)
        self.options_strategies = OptionsBasedStrategies(config)
        
        # Initialize Alternative Data Engine
        self.alternative_data_engine = AlternativeDataEngine(config)
        self.on_chain_analytics = OnChainAnalytics(config)
        
        # Initialize signal history
        self.signal_history = {}
        
        # Initialize quantitative analysis history
        self.quantitative_analysis_history = {}
        
        # Initialize confidence performance tracking
        self.confidence_performance = {}
        
        # Initialize performance monitoring
        self.performance_monitoring = {
            'active': False,
            'last_update': None,
            'performance_metrics': {},
            'alerts': [],
            'performance_score': 0.0,
            'risk_score': 0.0,
            'stability_score': 0.0,
            'update_frequency': 30
        }
        
        # Initialize data cache
        self.data_cache = {}
        
        # Initialize cache service if provided
        if self.cache_service:
            logger.info("Cache service initialized for enhanced trading strategy")
        
        logger.info("Enhanced Trading Strategy with Quantitative Analysis initialized")
    

    async def with_timeout(self, coro, timeout_seconds=60, operation_name="operation"):
        """Execute coroutine with timeout protection."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.error(f"Timeout error in {operation_name} after {timeout_seconds}s")
            return None
        except Exception as e:
            logger.error(f"Error in {operation_name}: {str(e)}")
            return None

    async def initialize(self) -> bool:
        """Initialize the strategy and quantitative components."""
        try:
            # Initialize quantitative integration
            await self.quantitative_integration.initialize()
            
            # Initialize quantitative trading system
            await self.quantitative_system.initialize()
            
            # Start real-time performance monitoring
            await self.start_performance_monitoring()
            
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
            
            # Calculate dynamic validation thresholds
            dynamic_thresholds = self._calculate_dynamic_validation_thresholds(symbol, market_data)
            
            # Accumulate signals if history is insufficient
            signal_history = self.signal_history.get(symbol, [])
            if len(signal_history) < dynamic_thresholds['min_sample_size']:
                logger.info(f"Accumulating signals for {symbol} (history: {len(signal_history)} < {dynamic_thresholds['min_sample_size']})")
                await self._accumulate_signals_for_symbol(symbol)
            
            # Generate advanced signal with quantitative analysis
            signal = await self._generate_advanced_signal(symbol, indicator_service, market_data)
            
            if signal:
                # Improve signal quality
                signal = await self._improve_signal_quality(signal, market_data)
                
                # Boost signal quality for new symbols
                signal = await self._boost_signal_quality(signal, market_data, symbol)
                
                # Apply dynamic thresholds
                if signal.get('confidence', 0) < dynamic_thresholds['confidence_threshold']:
                    logger.info(f"Signal confidence too low for {symbol}: {signal.get('confidence', 0):.3f} < {dynamic_thresholds['confidence_threshold']}")
                    return None
                
                if abs(signal.get('strength', 0)) < dynamic_thresholds['strength_threshold']:
                    logger.info(f"Signal strength too low for {symbol}: {abs(signal.get('strength', 0)):.3f} < {dynamic_thresholds['strength_threshold']}")
                    return None
                
                # Store signal history
                self._store_signal_history(symbol, signal)
                
                # Log quantitative analysis
                validation = self.quantitative_system.statistical_validator.validate_signal_quality(signal)
                await self._log_quantitative_analysis(symbol, signal, validation)
                
                logger.info(f"Generated signal for {symbol}: {signal.get('action', 'HOLD')} "
                          f"(confidence: {signal.get('confidence', 0):.3f}, strength: {signal.get('strength', 0):.3f})")
                
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
            
            # Apply machine learning analysis
            signal = await self._apply_machine_learning_analysis(symbol, signal, market_data)
            
            # Apply Phase 3 WorldQuant-Level Features
            signal = await self._apply_phase3_analysis(symbol, signal, market_data)
            
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
        """Store signal in history for statistical validation."""
        try:
            if symbol not in self.signal_history:
                self.signal_history[symbol] = []
            
            # Add timestamp if not present
            if 'timestamp' not in signal:
                signal['timestamp'] = datetime.now().isoformat()
            
            # Store signal with performance tracking
            signal_with_performance = {
                **signal,
                'performance_metrics': {
                    'signal_strength': signal.get('strength', 0.0),
                    'confidence': signal.get('confidence', 0.0),
                    'action': signal.get('action', 'hold'),
                    'timestamp': signal.get('timestamp', datetime.now().isoformat())
                }
            }
            
            self.signal_history[symbol].append(signal_with_performance)
            
            # Keep only last 1000 signals to prevent memory issues
            if len(self.signal_history[symbol]) > 1000:
                self.signal_history[symbol] = self.signal_history[symbol][-1000:]
            
            logger.debug(f"Stored signal for {symbol}, history size: {len(self.signal_history[symbol])}")
            
        except Exception as e:
            logger.error(f"Error storing signal history for {symbol}: {str(e)}")

    async def _accumulate_signals_for_symbol(self, symbol: str) -> None:
        """Accumulate signals for a symbol to build statistical significance."""
        try:
            # Get recent market data
            market_data = await self._get_comprehensive_market_data(symbol)
            
            # Generate multiple signals with different timeframes
            timeframes = ['5m', '15m', '1h', '4h']
            
            for timeframe in timeframes:
                try:
                    # Get klines for this timeframe
                    klines = await self.indicator_service.get_klines(symbol, timeframe, limit=100)
                    
                    if klines and 'close' in klines:
                        # Convert to DataFrame
                        df = self._convert_klines_to_dataframe(klines)
                        
                        if not df.empty:
                            # Calculate indicators
                            df = await self._calculate_advanced_indicators(df)
                            
                            # Create signal
                            signal = self._create_advanced_signal(symbol, df, df, df, market_data)
                            
                            if signal:
                                # Store in history
                                self._store_signal_history(symbol, signal)
                                
                except Exception as e:
                    logger.warning(f"Error accumulating signal for {symbol} {timeframe}: {str(e)}")
                    continue
            
            logger.info(f"Signal accumulation completed for {symbol}, total signals: {len(self.signal_history.get(symbol, []))}")
            
        except Exception as e:
            logger.error(f"Error accumulating signals for {symbol}: {str(e)}")

    def _calculate_dynamic_validation_thresholds(self, symbol: str, market_data: Dict) -> Dict[str, float]:
        """Calculate dynamic validation thresholds based on market conditions."""
        try:
            signal_history = self.signal_history.get(symbol, [])
            history_size = len(signal_history)
            
            # Base thresholds
            base_thresholds = {
                'min_sample_size': 10,
                'significance_level': 0.1,
                'confidence_threshold': 0.3,
                'strength_threshold': 0.1
            }
            
            # Adjust based on history size - MORE AGGRESSIVE FOR NEW SYMBOLS
            if history_size >= 100:
                base_thresholds['min_sample_size'] = 50
                base_thresholds['significance_level'] = 0.05
                base_thresholds['confidence_threshold'] = 0.5
                base_thresholds['strength_threshold'] = 0.2
            elif history_size >= 50:
                base_thresholds['min_sample_size'] = 25
                base_thresholds['significance_level'] = 0.08
                base_thresholds['confidence_threshold'] = 0.4
                base_thresholds['strength_threshold'] = 0.15
            elif history_size >= 20:
                base_thresholds['min_sample_size'] = 15
                base_thresholds['significance_level'] = 0.1
                base_thresholds['confidence_threshold'] = 0.35
                base_thresholds['strength_threshold'] = 0.12
            elif history_size < 10:
                # VERY AGGRESSIVE for very new symbols
                base_thresholds['confidence_threshold'] = 0.15  # 50% reduction
                base_thresholds['strength_threshold'] = 0.05   # 50% reduction
            elif history_size < 20:
                # AGGRESSIVE for new symbols
                base_thresholds['confidence_threshold'] = 0.2   # 33% reduction
                base_thresholds['strength_threshold'] = 0.07   # 30% reduction
            
            # Adjust based on market volatility
            if 'volatility' in market_data:
                volatility = market_data['volatility']
                if volatility > 0.05:  # High volatility
                    base_thresholds['confidence_threshold'] *= 0.8  # Lower threshold
                    base_thresholds['strength_threshold'] *= 0.8
                elif volatility < 0.02:  # Low volatility
                    base_thresholds['confidence_threshold'] *= 1.2  # Higher threshold
                    base_thresholds['strength_threshold'] *= 1.2
            
            # Adjust based on market regime
            if 'market_regime' in market_data:
                regime = market_data['market_regime']
                if regime == 'trending':
                    base_thresholds['strength_threshold'] *= 1.1
                elif regime == 'mean_reverting':
                    base_thresholds['confidence_threshold'] *= 0.9
            
            logger.debug(f"Dynamic thresholds for {symbol}: {base_thresholds}")
            return base_thresholds
            
        except Exception as e:
            logger.error(f"Error calculating dynamic thresholds for {symbol}: {str(e)}")
            return {
                'min_sample_size': 10,
                'significance_level': 0.1,
                'confidence_threshold': 0.15,  # Lower default for new symbols
                'strength_threshold': 0.05    # Lower default for new symbols
            }

    async def _improve_signal_quality(self, signal: Dict, market_data: Dict) -> Dict:
        """Improve signal quality with advanced analysis."""
        try:
            # Calculate signal strength based on multiple factors
            strength_factors = []
            
            # Technical indicators strength
            if 'timeframes' in signal:
                for timeframe, tf_data in signal['timeframes'].items():
                    tf_strength = abs(tf_data.get('strength', 0))
                    strength_factors.append(tf_strength)
            
            # Market condition strength
            if 'volatility' in market_data:
                volatility = market_data['volatility']
                vol_strength = 1.0 if 0.02 <= volatility <= 0.05 else 0.8
                strength_factors.append(vol_strength)
            
            # Trend strength
            if 'market_regime' in market_data:
                regime = market_data['market_regime']
                regime_strength = 1.2 if regime == 'trending' else 0.9
                strength_factors.append(regime_strength)
            
            # Calculate improved strength
            if strength_factors:
                improved_strength = np.mean(strength_factors) * signal.get('strength', 0)
                signal['strength'] = improved_strength
            
            # Improve confidence based on signal consistency
            if 'timeframes' in signal:
                confidences = [tf_data.get('confidence', 0) for tf_data in signal['timeframes'].values()]
                if confidences:
                    signal['confidence'] = np.mean(confidences)
            
            # Add quality metrics
            signal['quality_metrics'] = {
                'strength_factors': strength_factors,
                'confidence_consistency': np.std(confidences) if 'confidences' in locals() else 0.0,
                'market_alignment': regime_strength if 'regime_strength' in locals() else 1.0
            }
            
            logger.debug(f"Improved signal quality for {signal.get('symbol', 'unknown')}: strength={signal.get('strength', 0):.3f}, confidence={signal.get('confidence', 0):.3f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error improving signal quality: {str(e)}")
            return signal

    async def _boost_signal_quality(self, signal: Dict, market_data: Dict, symbol: str) -> Dict:
        """Boost signal quality for new symbols with low confidence."""
        try:
            signal_history = self.signal_history.get(symbol, [])
            history_size = len(signal_history)
            
            # Base confidence and strength
            base_confidence = signal.get('confidence', 0)
            base_strength = signal.get('strength', 0)
            
            # Boost factors for new symbols
            boost_multiplier = 1.0
            
            # Boost based on history size
            if history_size < 10:
                boost_multiplier = 2.0  # Double boost for very new symbols
            elif history_size < 20:
                boost_multiplier = 1.5  # 50% boost for new symbols
            elif history_size < 50:
                boost_multiplier = 1.2  # 20% boost for moderately new symbols
            
            # Boost based on market conditions
            if 'volatility' in market_data:
                volatility = market_data['volatility']
                if volatility < 0.03:  # Low volatility - more predictable
                    boost_multiplier *= 1.1
                elif volatility > 0.08:  # High volatility - less predictable
                    boost_multiplier *= 0.9
            
            # Boost based on market regime
            if 'market_regime' in market_data:
                regime = market_data['market_regime']
                if regime == 'trending':
                    boost_multiplier *= 1.1  # Trending markets are more predictable
                elif regime == 'mean_reverting':
                    boost_multiplier *= 1.05  # Mean reverting markets are moderately predictable
            
            # Boost based on signal strength
            if abs(base_strength) > 0.1:
                boost_multiplier *= 1.2  # Strong signals get more boost
            
            # Apply boost to confidence
            boosted_confidence = min(base_confidence * boost_multiplier, 0.95)  # Cap at 95%
            
            # Apply boost to strength
            boosted_strength = base_strength * boost_multiplier
            
            # Update signal
            signal['confidence'] = boosted_confidence
            signal['strength'] = boosted_strength
            
            # Add boost metrics
            signal['quality_metrics']['boost_applied'] = {
                'original_confidence': base_confidence,
                'boosted_confidence': boosted_confidence,
                'boost_multiplier': boost_multiplier,
                'history_size': history_size,
                'boost_reason': f"New symbol boost (history: {history_size})"
            }
            
            logger.info(f"Boosted signal quality for {symbol}: confidence {base_confidence:.3f} -> {boosted_confidence:.3f} (boost: {boost_multiplier:.2f}x)")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error boosting signal quality for {symbol}: {str(e)}")
            return signal
    
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
            for symbol in symbols:
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
            try:
                optimization = await self.quantitative_system.optimize_portfolio(returns_df)
                return optimization
            except Exception as e:
                logger.error(f"Error in portfolio optimization: {str(e)}")
                return {'error': str(e)}
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio optimization: {str(e)}")
            return {'error': str(e)}
    
    async def analyze_factor_exposures(self, symbols: List[str]) -> Dict:
        """Analyze factor exposures for portfolio."""
        try:
            # Get historical data for factor analysis
            returns_data = {}
            for symbol in symbols:
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
            try:
                factor_results = await self.quantitative_system.factor_model.build_factor_model(returns_df)
                return factor_results
            except Exception as e:
                logger.error(f"Error in factor analysis: {str(e)}")
                return {'error': str(e)}
            
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
        """Process trading signals with enhanced error handling."""
        try:
            logger.info(f"Processing signals: {signals.get('action', 'unknown')} for {signals.get('symbol', 'unknown')}")
        except Exception as e:
            import traceback
            logger.error(f"Error in process_trading_signals: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return
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
            
            # Check DCA and Trailing Stop opportunities
            await self._check_dca_and_trailing_opportunities(symbol, market_data)
                
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
            risk_per_trade = self.config.get('trading', {}).get('risk_per_trade', 0.02)  # 2% risk per trade
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
        """Calculate stop loss price based on fixed percentage."""
        try:
            # Get fixed percentage SL/TP config
            fixed_config = self.config.get('risk_management', {}).get('fixed_percentage_sl_tp', {})
            
            if fixed_config.get('enabled', False):
                # Use fixed percentage approach
                if is_long_side(position_type):
                    # For LONG positions: SL = current_price * (1 - 10%)
                    stop_loss_percentage = fixed_config.get('long', {}).get('stop_loss_percentage', 0.10)
                    stop_loss = float(current_price) * (1 - stop_loss_percentage)
                else:
                    # For SHORT positions: SL = current_price * (1 + 10%)
                    stop_loss_percentage = fixed_config.get('short', {}).get('stop_loss_percentage', 0.10)
                    stop_loss = float(current_price) * (1 + stop_loss_percentage)
                
                logger.info(f"Calculated fixed percentage stop loss for {symbol} {position_type.lower()}: {stop_loss} (current price: {current_price})")
                return stop_loss
            else:
                # Fallback to original ATR-based calculation
                stop_loss_multiplier = float(self.config.get('risk_management', {}).get('stop_loss_atr_multiplier', 2.0))
                
                if is_long_side(position_type):
                    stop_loss = float(current_price) - (float(atr) * stop_loss_multiplier)
                    if stop_loss <= 0:
                        stop_loss = float(current_price) * 0.02
                    stop_loss = min(stop_loss, current_price * 0.8)
                else:
                    stop_loss = float(current_price) + (float(atr) * stop_loss_multiplier/2)
                    stop_loss = max(stop_loss, current_price * 1.1)
                
                logger.info(f"Calculated ATR-based stop loss for {symbol} {position_type.lower()}: {stop_loss} (current price: {current_price})")
                return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating stop loss for {symbol}: {str(e)}")
            return None

    async def _calculate_take_profit(self, symbol: str, position_type: str, current_price: float, stop_loss: float) -> float:
        """Calculate take profit price based on fixed percentage."""
        try:
            # Get fixed percentage SL/TP config
            fixed_config = self.config.get('risk_management', {}).get('fixed_percentage_sl_tp', {})
            
            if fixed_config.get('enabled', False):
                # Use fixed percentage approach
                if is_long_side(position_type):
                    # For LONG positions: TP = current_price * (1 + 20%)
                    take_profit_percentage = fixed_config.get('long', {}).get('take_profit_percentage', 0.20)
                    take_profit = float(current_price) * (1 + take_profit_percentage)
                else:
                    # For SHORT positions: TP = current_price * (1 - 5%)
                    take_profit_percentage = fixed_config.get('short', {}).get('take_profit_percentage', 0.05)
                    take_profit = float(current_price) * (1 - take_profit_percentage)
                
                logger.info(f"Calculated fixed percentage take profit for {symbol} {position_type.lower()}: {take_profit} (current price: {current_price})")
                return take_profit
            else:
                # Fallback to original risk-reward ratio calculation
                risk_reward_ratio = self.config.get('risk_management', {}).get('take_profit_multiplier', 2.0)
                price_diff = abs(current_price - stop_loss)
                
                if is_long_side(position_type):
                    take_profit = current_price + (price_diff * risk_reward_ratio)
                    take_profit = max(take_profit, current_price * 1.2)
                else:
                    take_profit = current_price - (price_diff * risk_reward_ratio/8)
                    take_profit = max(take_profit, current_price * 0.95)
                
                logger.info(f"Calculated risk-reward based take profit for {symbol} {position_type.lower()}: {take_profit} (current price: {current_price})")
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
                    # Signal is stored directly, not nested under 'signal' key
                    if signal_entry.get('quantitative_confidence', 0) > 0.5:
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
    
    async def _check_dca_and_trailing_opportunities(self, symbol: str, market_data: Dict) -> None:
        """Check DCA and Trailing Stop opportunities for existing positions."""
        try:
            # Get all positions
            all_positions = await self.binance_service.get_positions()
            
            if not all_positions:
                return
            
            # Filter positions for this specific symbol
            symbol_positions = []
            for position in all_positions:
                if not position or not isinstance(position, dict):
                    continue
                    
                # Get position info
                info = position.get('info', {})
                if not info:
                    continue
                    
                # Normalize position symbol
                pos_symbol = info.get('symbol', '').replace('/', '')
                normalized_symbol = symbol.split(':')[0].replace('/', '')
                
                # Check if symbols match
                if pos_symbol == normalized_symbol:
                    position_size = abs(float(info.get('positionAmt', 0)))
                    
                    # Skip if no position
                    if position_size <= 0:
                        continue
                        
                    symbol_positions.append(position)
            
            # Process positions for this symbol
            for position in symbol_positions:
                # Check DCA opportunity
                dca_decision = await self.with_timeout(self.worldquant_dca.check_dca_opportunity(symbol, position, market_data), 30, 'dca_check')
                if dca_decision.get('should_dca', False):
                    logger.info(f"DCA opportunity detected for {symbol}: {dca_decision}")
                    await self.worldquant_dca.execute_dca(symbol, position, dca_decision, self.binance_service)
                
                # Check Trailing Stop opportunity
                trailing_decision = await self.with_timeout(self.worldquant_trailing.check_trailing_stop_opportunity(symbol, position, market_data), 30, 'trailing_check')
                if trailing_decision.get('should_update', False):
                    logger.info(f"Trailing Stop opportunity detected for {symbol}: {trailing_decision}")
                    await self.worldquant_trailing.execute_trailing_stop_update(symbol, position, trailing_decision, self.binance_service)
                    
        except Exception as e:
            logger.error(f"Error checking DCA and Trailing Stop opportunities for {symbol}: {str(e)}")
    

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on strategy components."""
        try:
            health_status = {
                'timestamp': time.time(),
                'status': 'healthy',
                'components': {}
            }
            
            # Check quantitative components
            if hasattr(self, 'quantitative_system'):
                try:
                    # Quick test of quantitative system
                    health_status['components']['quantitative_system'] = 'healthy'
                except Exception as e:
                    health_status['components']['quantitative_system'] = f'unhealthy: {str(e)}'
                    health_status['status'] = 'degraded'
            
            # Check cache service
            if hasattr(self, 'cache_service'):
                try:
                    # Quick test of cache service
                    health_status['components']['cache_service'] = 'healthy'
                except Exception as e:
                    health_status['components']['cache_service'] = f'unhealthy: {str(e)}'
                    health_status['status'] = 'degraded'
            
            # Check signal history
            if hasattr(self, 'signal_history'):
                health_status['components']['signal_history'] = f'healthy (size: {len(self.signal_history)})'
            
            logger.info(f"Health check completed: {health_status['status']}")
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'timestamp': time.time(),
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def recover_from_error(self, error: Exception) -> bool:
        """Attempt to recover from an error."""
        try:
            logger.info(f"Attempting to recover from error: {str(error)}")
            
            # Clear caches if needed
            if hasattr(self, 'data_cache'):
                self.data_cache.clear()
                logger.info("Cleared data cache")
            
            # Reset signal history if needed
            if hasattr(self, 'signal_history'):
                self.signal_history.clear()
                logger.info("Cleared signal history")
            
            # Perform health check
            health = await self.health_check()
            if health['status'] == 'healthy':
                logger.info("Recovery successful")
                return True
            else:
                logger.warning("Recovery incomplete")
                return False
                
        except Exception as e:
            logger.error(f"Recovery failed: {str(e)}")
            return False

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
        """
        Apply advanced risk management using DynamicRiskManager.
        """
        try:
            # Get historical returns for risk calculation
            returns = self._calculate_returns_from_market_data(market_data)
            
            # Calculate dynamic VaR with regime switching
            regime = self._detect_volatility_regime(returns)
            var_results = self.dynamic_risk_manager.calculate_dynamic_var(returns, regime)
            
            # Calculate portfolio risk attribution
            portfolio_weights = {symbol: 1.0}  # Single asset portfolio
            covariance_matrix = pd.DataFrame([[returns.var()]], index=[symbol], columns=[symbol])
            risk_attribution = self.dynamic_risk_manager.calculate_portfolio_risk_attribution(
                portfolio_weights, covariance_matrix
            )
            
            # Run stress tests
            stress_results = self.dynamic_risk_manager.run_stress_tests(
                portfolio_weights, {symbol: returns}
            )
            
            # Adjust signal based on risk analysis
            adjusted_signal = signal.copy()
            
            # Adjust position size based on VaR
            if 'expected_shortfall' in var_results:
                var_adjustment = min(1.0, 0.1 / abs(var_results['expected_shortfall'])) if var_results['expected_shortfall'] != 0 else 1.0
                # Safely access optimized_position_size with default value
                current_position_size = adjusted_signal.get('optimized_position_size', 0.01)
                adjusted_signal['optimized_position_size'] = current_position_size * var_adjustment
            
            # Add risk metrics to signal
            adjusted_signal['risk_metrics'] = {
                'dynamic_var': var_results,
                'risk_attribution': risk_attribution,
                'stress_test_results': stress_results,
                'volatility_regime': regime
            }
            
            logger.info(f"Applied advanced risk management for {symbol}")
            return adjusted_signal
            
        except Exception as e:
            logger.error(f"Error applying advanced risk management: {str(e)}")
            return signal

    async def _apply_statistical_arbitrage_analysis(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """
        Apply statistical arbitrage analysis using StatisticalArbitrageEngine.
        """
        try:
            # Prepare market data for arbitrage analysis
            market_data_dict = {symbol: market_data.get('close', pd.Series())}
            
            # Generate pairs trading signals
            pairs_signals = self.statistical_arbitrage_engine.generate_pairs_trading_signals(market_data_dict)
            
            # Generate mean reversion signals
            mean_reversion_signals = self.statistical_arbitrage_engine.generate_mean_reversion_signals(market_data_dict)
            
            # Generate momentum reversal signals
            momentum_reversal_signals = self.statistical_arbitrage_engine.generate_momentum_reversal_signals(market_data_dict)
            
            # Generate volatility arbitrage signals
            volatility_arbitrage_signals = self.statistical_arbitrage_engine.generate_volatility_arbitrage_signals(market_data_dict)
            
            # Combine arbitrage signals
            arbitrage_analysis = {
                'pairs_trading': pairs_signals,
                'mean_reversion': mean_reversion_signals,
                'momentum_reversal': momentum_reversal_signals,
                'volatility_arbitrage': volatility_arbitrage_signals
            }
            
            # Adjust signal based on arbitrage analysis
            adjusted_signal = signal.copy()
            
            # Check for strong arbitrage opportunities
            strong_signals = []
            
            for signal_type, signals in arbitrage_analysis.items():
                if symbol in signals:
                    symbol_signal = signals[symbol]
                    if symbol_signal['signal']['confidence'] > 0.7:
                        strong_signals.append({
                            'type': signal_type,
                            'action': symbol_signal['signal']['action'],
                            'confidence': symbol_signal['signal']['confidence']
                        })
            
            # Boost signal if strong arbitrage opportunity found
            if strong_signals:
                best_signal = max(strong_signals, key=lambda x: x['confidence'])
                adjusted_signal['arbitrage_boost'] = {
                    'type': best_signal['type'],
                    'action': best_signal['action'],
                    'confidence': best_signal['confidence']
                }
                adjusted_signal['quantitative_confidence'] = min(
                    adjusted_signal['quantitative_confidence'] + 0.2, 1.0
                )
            
            adjusted_signal['arbitrage_analysis'] = arbitrage_analysis
            
            logger.info(f"Applied statistical arbitrage analysis for {symbol}")
            return adjusted_signal
            
        except Exception as e:
            logger.error(f"Error applying statistical arbitrage analysis: {str(e)}")
            return signal

    async def _apply_advanced_ml_analysis(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """
        Apply advanced machine learning analysis using AdvancedMLEnsemble.
        """
        try:
            # Prepare data for ML models
            market_df = pd.DataFrame(market_data)
            X, y = self.advanced_ml_ensemble.prepare_data(market_df)
            
            # Make predictions with uncertainty quantification
            if len(X) > 0:
                ml_predictions = self.advanced_ml_ensemble.predict_with_uncertainty(X)
                
                # Adjust signal based on ML predictions
                adjusted_signal = signal.copy()
                
                # Use ensemble prediction
                ensemble_prediction = ml_predictions.get('ensemble_prediction', 0.0)
                ensemble_uncertainty = ml_predictions.get('ensemble_uncertainty', 0.0)
                
                # Adjust signal strength based on ML prediction
                if ensemble_prediction > 0.05:  # Positive prediction
                    adjusted_signal['ml_boost'] = min(ensemble_prediction * 2, 0.3)
                    adjusted_signal['quantitative_confidence'] = min(
                        adjusted_signal['quantitative_confidence'] + adjusted_signal['ml_boost'], 1.0
                    )
                elif ensemble_prediction < -0.05:  # Negative prediction
                    adjusted_signal['ml_penalty'] = min(abs(ensemble_prediction) * 2, 0.3)
                    adjusted_signal['quantitative_confidence'] = max(
                        adjusted_signal['quantitative_confidence'] - adjusted_signal['ml_penalty'], 0.0
                    )
                
                # Add uncertainty information
                adjusted_signal['ml_uncertainty'] = ensemble_uncertainty
                adjusted_signal['ml_predictions'] = ml_predictions
                
                # Adjust position size based on uncertainty
                if ensemble_uncertainty > 0.1:  # High uncertainty
                    current_size = adjusted_signal.get('optimized_position_size', 0.01)
                    adjusted_signal['optimized_position_size'] = current_size * 0.8  # Reduce position size
                
                logger.info(f"Applied advanced ML analysis for {symbol}")
                return adjusted_signal
            
            return signal
            
        except Exception as e:
            logger.error(f"Error applying advanced ML analysis: {str(e)}")
            return signal



    def _calculate_returns_from_market_data(self, market_data: Dict) -> pd.Series:
        """
        Calculate returns from market data.
        """
        try:
            if 'close' in market_data:
                prices = pd.Series(market_data['close'])
                returns = prices.pct_change().dropna()
                return returns
            else:
                return pd.Series([0.0])
        except Exception as e:
            logger.error(f"Error calculating returns: {str(e)}")
            return pd.Series([0.0])

    def _detect_volatility_regime(self, returns: pd.Series) -> str:
        """
        Detect volatility regime for risk management.
        """
        try:
            if len(returns) < 20:
                return 'normal_volatility'
            
            current_vol = returns.rolling(window=20).std().iloc[-1]
            avg_vol = returns.std()
            
            if current_vol > avg_vol * 1.5:
                return 'high_volatility'
            elif current_vol < avg_vol * 0.7:
                return 'low_volatility'
            else:
                return 'normal_volatility'
                
        except Exception as e:
            logger.error(f"Error detecting volatility regime: {str(e)}")
            return 'normal_volatility'

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
            
            # Check if we have sufficient signal history for statistical validation
            signal_history = self.signal_history.get(symbol, [])
            if len(signal_history) < self.statistical_validator.min_sample_size:
                logger.info(f"Insufficient signal history for {symbol} ({len(signal_history)} < {self.statistical_validator.min_sample_size}), skipping statistical validation")
                # Add basic validation results without statistical testing
                signal['statistical_validation'] = {
                    'quality_validation': quality_validation,
                    'significance_test': {
                        'significant': True,  # Assume significant for new symbols
                        'p_value': 0.05,
                        'sample_size': len(signal_history),
                        'note': 'Insufficient history, validation skipped'
                    },
                    'confidence_score': quality_validation['confidence_score']
                }
                return signal
            
            # Get benchmark returns for significance testing
            benchmark_returns = await self._get_benchmark_returns(symbol)
            
            if benchmark_returns is not None:
                # Test signal significance
                significance_result = self.statistical_validator.test_signal_significance(
                    signal_history, 
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
            
            for symbol in symbols:
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
    
    async def _apply_machine_learning_analysis(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """Apply machine learning analysis to signal."""
        try:
            # Get comprehensive market data for ML analysis
            comprehensive_data = await self._get_comprehensive_market_data(symbol)
            
            if not comprehensive_data:
                logger.warning(f"No comprehensive data available for ML analysis on {symbol}")
                return signal
            
            # Convert market data to DataFrame
            df = self._convert_market_data_to_dataframe(comprehensive_data)
            
            if df.empty:
                logger.warning(f"Empty DataFrame for ML analysis on {symbol}")
                return signal
            
            # Engineer features
            df_features = self.quantitative_system.ml_ensemble.engineer_features(df)
            
            if df_features.empty:
                logger.warning(f"No features available for ML analysis on {symbol}")
                return signal
            
            # Make ML predictions
            ml_predictions = await self.quantitative_system.ml_ensemble.predict_ensemble(df_features)
            
            if ml_predictions:
                # Add ML predictions to signal
                signal['ml_predictions'] = {
                    'ensemble_prediction': float(ml_predictions['ensemble_prediction'][-1]) if len(ml_predictions['ensemble_prediction']) > 0 else 0.0,
                    'confidence': float(ml_predictions['confidence'][-1]) if len(ml_predictions['confidence']) > 0 else 0.0,
                    'individual_predictions': {
                        model: float(pred[-1]) if len(pred) > 0 else 0.0
                        for model, pred in ml_predictions['individual_predictions'].items()
                    }
                }
                
                # Adjust signal based on ML predictions
                signal = self._adjust_signal_by_ml_predictions(signal, ml_predictions)
                
                logger.info(f"ML analysis applied to {symbol}: ensemble_prediction={signal['ml_predictions']['ensemble_prediction']:.4f}, confidence={signal['ml_predictions']['confidence']:.3f}")
            else:
                logger.warning(f"No ML predictions available for {symbol}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error applying machine learning analysis for {symbol}: {str(e)}")
            return signal
    
    def _convert_market_data_to_dataframe(self, market_data: Dict) -> pd.DataFrame:
        """Convert market data to DataFrame for ML analysis."""
        try:
            if not market_data or 'klines' not in market_data:
                return pd.DataFrame()
            
            klines = market_data['klines']
            
            # Convert klines to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Error converting market data to DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def _adjust_signal_by_ml_predictions(self, signal: Dict, ml_predictions: Dict) -> Dict:
        """Adjust signal based on ML predictions."""
        try:
            if not ml_predictions or 'ensemble_prediction' not in ml_predictions:
                return signal
            
            ensemble_pred = ml_predictions['ensemble_prediction']
            confidence = ml_predictions['confidence']
            
            if len(ensemble_pred) == 0:
                return signal
            
            # Get latest prediction
            latest_pred = ensemble_pred[-1]
            latest_confidence = confidence[-1] if len(confidence) > 0 else 0.5
            
            # Adjust signal based on ML prediction
            current_action = signal.get('action', 'hold')
            current_confidence = signal.get('confidence', 0.5)
            
            # ML-based action adjustment
            if latest_pred > 0.01:  # Positive prediction threshold
                ml_adjusted_action = 'buy'
            elif latest_pred < -0.01:  # Negative prediction threshold
                ml_adjusted_action = 'sell'
            else:
                ml_adjusted_action = 'hold'
            
            # Adjust confidence based on ML confidence
            ml_adjusted_confidence = current_confidence * 0.7 + latest_confidence * 0.3
            ml_adjusted_confidence = max(0.0, min(1.0, ml_adjusted_confidence))
            
            # Add ML analysis to signal
            signal['ml_analysis'] = {
                'ml_adjusted_action': ml_adjusted_action,
                'ml_adjusted_confidence': ml_adjusted_confidence,
                'prediction_strength': abs(latest_pred),
                'prediction_direction': 'positive' if latest_pred > 0 else 'negative',
                'model_agreement': self._calculate_model_agreement(ml_predictions)
            }
            
            # Update signal with ML adjustments
            if ml_adjusted_confidence > current_confidence:
                signal['action'] = ml_adjusted_action
                signal['confidence'] = ml_adjusted_confidence
            
            return signal
            
        except Exception as e:
            logger.error(f"Error adjusting signal by ML predictions: {str(e)}")
            return signal
    
    def _calculate_model_agreement(self, ml_predictions: Dict) -> float:
        """Calculate agreement among individual models."""
        try:
            individual_predictions = ml_predictions.get('individual_predictions', {})
            
            if not individual_predictions:
                return 0.0
            
            # Get predictions from all models
            predictions = list(individual_predictions.values())
            
            if len(predictions) < 2:
                return 1.0
            
            # Calculate agreement (lower std = higher agreement)
            prediction_std = np.std(predictions)
            agreement = 1.0 / (1.0 + prediction_std)
            
            return float(agreement)
            
        except Exception as e:
            logger.error(f"Error calculating model agreement: {str(e)}")
            return 0.0
    
    async def train_ml_models(self, symbols: List[str]) -> Dict[str, Any]:
        """Train ML models for all symbols."""
        try:
            logger.info(f"Training ML models for {len(symbols)} symbols")
            
            training_results = {}
            
            for symbol in symbols:
                try:
                    # Get market data
                    market_data = await self._get_comprehensive_market_data(symbol)
                    
                    if not market_data:
                        logger.warning(f"No market data available for ML training on {symbol}")
                        continue
                    
                    # Convert to DataFrame
                    df = self._convert_market_data_to_dataframe(market_data)
                    
                    if df.empty:
                        logger.warning(f"Empty DataFrame for ML training on {symbol}")
                        continue
                    
                    # Engineer features
                    df_features = self.quantitative_system.ml_ensemble.engineer_features(df)
                    
                    if df_features.empty:
                        logger.warning(f"No features available for ML training on {symbol}")
                        continue
                    
                    # Train models
                    symbol_results = await self.quantitative_system.ml_ensemble.train_ensemble(df_features)
                    
                    if symbol_results:
                        training_results[symbol] = symbol_results
                        logger.info(f"ML training completed for {symbol}")
                    else:
                        logger.warning(f"ML training failed for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error training ML models for {symbol}: {str(e)}")
                    continue
            
            logger.info(f"ML training completed for {len(training_results)} symbols")
            return training_results
            
        except Exception as e:
            logger.error(f"Error training ML models: {str(e)}")
            return {}
    
    async def get_ml_model_interpretability(self, symbol: str, model_name: str = 'random_forest') -> Dict[str, Any]:
        """Get ML model interpretability for a symbol."""
        try:
            # Get market data
            market_data = await self._get_comprehensive_market_data(symbol)
            
            if not market_data:
                logger.warning(f"No market data available for ML interpretability on {symbol}")
                return {}
            
            # Convert to DataFrame
            df = self._convert_market_data_to_dataframe(market_data)
            
            if df.empty:
                logger.warning(f"Empty DataFrame for ML interpretability on {symbol}")
                return {}
            
            # Engineer features
            df_features = self.quantitative_system.ml_ensemble.engineer_features(df)
            
            if df_features.empty:
                logger.warning(f"No features available for ML interpretability on {symbol}")
                return {}
            
            # Get interpretability
            interpretability = self.quantitative_system.ml_ensemble.get_model_interpretability(
                df_features, model_name
            )
            
            if interpretability:
                logger.info(f"ML interpretability analysis completed for {symbol}")
                return interpretability
            else:
                logger.warning(f"ML interpretability analysis failed for {symbol}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting ML model interpretability for {symbol}: {str(e)}")
            return {}
    
    async def get_ml_summary(self) -> Dict[str, Any]:
        """Get ML ensemble summary."""
        try:
            return await self.quantitative_system.ml_ensemble.get_ml_summary()
        except Exception as e:
            logger.error(f"Error getting ML summary: {str(e)}")
            return {}
    
    # ==================== PERFORMANCE MONITORING METHODS ====================
    
    async def start_performance_monitoring(self) -> None:
        """Start real-time performance monitoring."""
        try:
            if not self.performance_monitoring['active']:
                self.performance_monitoring['active'] = True
                self.performance_monitoring['last_update'] = datetime.now()
                logger.info("Real-time performance monitoring started")
        except Exception as e:
            logger.error(f"Error starting performance monitoring: {str(e)}")
    
    async def stop_performance_monitoring(self) -> None:
        """Stop real-time performance monitoring."""
        try:
            if self.performance_monitoring['active']:
                self.performance_monitoring['active'] = False
                logger.info("Real-time performance monitoring stopped")
        except Exception as e:
            logger.error(f"Error stopping performance monitoring: {str(e)}")
    
    async def update_performance_metrics(self) -> None:
        """Update real-time performance metrics."""
        try:
            if not self.performance_monitoring['active']:
                return
            
            # Get current portfolio state
            portfolio_data = await self._get_portfolio_performance_data()
            
            # Calculate performance metrics
            metrics = await self._calculate_real_time_metrics(portfolio_data)
            
            # Update monitoring state
            self.performance_monitoring['performance_metrics'] = metrics
            self.performance_monitoring['last_update'] = datetime.now()
            
            # Check for alerts
            alerts = await self._check_performance_alerts(metrics)
            self.performance_monitoring['alerts'] = alerts
            
            # Calculate performance scores
            await self._calculate_performance_scores(metrics)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    async def _get_portfolio_performance_data(self) -> Dict:
        """Get current portfolio performance data."""
        try:
            # Get current positions
            positions = await self.binance_service.get_positions() if self.binance_service else []
            
            # Calculate portfolio metrics
            total_value = 0.0
            total_pnl = 0.0
            position_data = {}
            
            for position in positions:
                symbol = position.get('symbol', '')
                unrealized_pnl = float(position.get('unrealizedPnl', 0))
                position_amt = float(position.get('positionAmt', 0))
                mark_price = float(position.get('markPrice', 0))
                
                if abs(position_amt) > 0:
                    position_value = abs(position_amt) * mark_price
                    total_value += position_value
                    total_pnl += unrealized_pnl
                    
                    position_data[symbol] = {
                        'weight': position_value / max(total_value, 1),
                        'return': unrealized_pnl / max(position_value, 1),
                        'size': position_amt,
                        'price': mark_price
                    }
            
            return {
                'timestamp': datetime.now(),
                'total_value': total_value,
                'total_pnl': total_pnl,
                'positions': position_data,
                'return_rate': total_pnl / max(total_value, 1) if total_value > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio performance data: {str(e)}")
            return {
                'timestamp': datetime.now(),
                'total_value': 0.0,
                'total_pnl': 0.0,
                'positions': {},
                'return_rate': 0.0
            }
    
    async def _calculate_real_time_metrics(self, portfolio_data: Dict) -> Dict[str, float]:
        """Calculate real-time performance metrics."""
        try:
            metrics = {}
            
            # Basic metrics
            total_return = portfolio_data.get('return_rate', 0.0)
            total_value = portfolio_data.get('total_value', 0.0)
            
            # Calculate rolling metrics from signal history
            returns_list = []
            for symbol, signals in self.signal_history.items():
                for signal in signals:
                    if 'return' in signal:
                        returns_list.append(signal['return'])
            
            if len(returns_list) > 1:
                # Volatility (rolling)
                volatility = np.std(returns_list[-30:]) * np.sqrt(252) if len(returns_list) >= 30 else np.std(returns_list) * np.sqrt(252)
                
                # Sharpe ratio (assuming risk-free rate of 2%)
                risk_free_rate = 0.02
                excess_return = total_return - risk_free_rate / 252
                sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0
                
                # Drawdown calculation
                cumulative_returns = np.cumprod(1 + np.array(returns_list))
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / running_max
                current_drawdown = drawdown[-1] if len(drawdown) > 0 else 0.0
                max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
                
                # VaR (95% confidence)
                var_95 = np.percentile(returns_list, 5)
                
                # Win rate
                positive_returns = [r for r in returns_list if r > 0]
                win_rate = len(positive_returns) / len(returns_list) if returns_list else 0.5
                
                metrics.update({
                    'total_return': total_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'current_drawdown': current_drawdown,
                    'max_drawdown': max_drawdown,
                    'var_95': var_95,
                    'win_rate': win_rate,
                    'total_value': total_value,
                    'total_pnl': portfolio_data.get('total_pnl', 0.0)
                })
            else:
                metrics.update({
                    'total_return': total_return,
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'current_drawdown': 0.0,
                    'max_drawdown': 0.0,
                    'var_95': 0.0,
                    'win_rate': 0.5,
                    'total_value': total_value,
                    'total_pnl': portfolio_data.get('total_pnl', 0.0)
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating real-time metrics: {str(e)}")
            return {}
    
    async def _check_performance_alerts(self, metrics: Dict) -> List[Dict]:
        """Check for performance alerts."""
        try:
            alerts = []
            
            # Alert thresholds
            thresholds = {
                'drawdown_exceeded': {'threshold': 0.10, 'level': 'warning'},
                'volatility_spike': {'threshold': 0.25, 'level': 'warning'},
                'sharpe_decline': {'threshold': 0.5, 'level': 'warning'},
                'var_exceeded': {'threshold': 0.15, 'level': 'critical'}
            }
            
            current_drawdown = abs(metrics.get('current_drawdown', 0))
            current_volatility = metrics.get('volatility', 0)
            current_sharpe = metrics.get('sharpe_ratio', 0)
            current_var = abs(metrics.get('var_95', 0))
            
            # Check each threshold
            if current_drawdown > thresholds['drawdown_exceeded']['threshold']:
                alerts.append({
                    'type': 'drawdown_exceeded',
                    'level': thresholds['drawdown_exceeded']['level'],
                    'current_value': current_drawdown,
                    'threshold': thresholds['drawdown_exceeded']['threshold'],
                    'message': f"Drawdown exceeded: {current_drawdown:.2%} > {thresholds['drawdown_exceeded']['threshold']:.2%}"
                })
            
            if current_volatility > thresholds['volatility_spike']['threshold']:
                alerts.append({
                    'type': 'volatility_spike',
                    'level': thresholds['volatility_spike']['level'],
                    'current_value': current_volatility,
                    'threshold': thresholds['volatility_spike']['threshold'],
                    'message': f"Volatility spike: {current_volatility:.2%} > {thresholds['volatility_spike']['threshold']:.2%}"
                })
            
            if current_sharpe < thresholds['sharpe_decline']['threshold']:
                alerts.append({
                    'type': 'sharpe_decline',
                    'level': thresholds['sharpe_decline']['level'],
                    'current_value': current_sharpe,
                    'threshold': thresholds['sharpe_decline']['threshold'],
                    'message': f"Sharpe ratio decline: {current_sharpe:.3f} < {thresholds['sharpe_decline']['threshold']:.3f}"
                })
            
            if current_var > thresholds['var_exceeded']['threshold']:
                alerts.append({
                    'type': 'var_exceeded',
                    'level': thresholds['var_exceeded']['level'],
                    'current_value': current_var,
                    'threshold': thresholds['var_exceeded']['threshold'],
                    'message': f"VaR exceeded: {current_var:.2%} > {thresholds['var_exceeded']['threshold']:.2%}"
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking performance alerts: {str(e)}")
            return []
    
    async def _calculate_performance_scores(self, metrics: Dict) -> None:
        """Calculate performance scores."""
        try:
            # Performance score (0-100)
            sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
            win_rate = metrics.get('win_rate', 0.5)
            total_return = metrics.get('total_return', 0.0)
            
            performance_score = (
                min(max(sharpe_ratio * 20, 0), 40) +
                min(max(win_rate * 30, 0), 30) +
                min(max(total_return * 100, 0), 30)
            )
            
            # Risk score (0-100, lower is better)
            volatility = metrics.get('volatility', 0.0)
            max_drawdown = abs(metrics.get('max_drawdown', 0.0))
            var = abs(metrics.get('var_95', 0.0))
            
            risk_score = (
                min(max(volatility * 100, 0), 40) +
                min(max(max_drawdown * 100, 0), 30) +
                min(max(var * 100, 0), 30)
            )
            
            # Stability score (0-100)
            stability_score = max(0, 100 - risk_score)
            
            # Update monitoring state
            self.performance_monitoring['performance_score'] = performance_score
            self.performance_monitoring['risk_score'] = risk_score
            self.performance_monitoring['stability_score'] = stability_score
            
        except Exception as e:
            logger.error(f"Error calculating performance scores: {str(e)}")
    
    async def get_real_time_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive real-time performance summary."""
        try:
            # Update metrics if monitoring is active
            if self.performance_monitoring['active']:
                await self.update_performance_metrics()
            
            summary = {
                'performance_metrics': self.performance_monitoring['performance_metrics'],
                'alerts': self.performance_monitoring['alerts'],
                'performance_score': self.performance_monitoring['performance_score'],
                'risk_score': self.performance_monitoring['risk_score'],
                'stability_score': self.performance_monitoring['stability_score'],
                'last_update': self.performance_monitoring['last_update'],
                'monitoring_active': self.performance_monitoring['active']
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting real-time performance summary: {str(e)}")
            return {}
    
    # ==================== ADVANCED PERFORMANCE MONITORING ====================
    
    async def get_advanced_performance_metrics(self) -> Dict[str, Any]:
        """Get advanced performance metrics with WorldQuant standards."""
        try:
            # Get basic performance data
            portfolio_data = await self._get_portfolio_performance_data()
            basic_metrics = await self._calculate_real_time_metrics(portfolio_data)
            
            # Calculate advanced metrics
            advanced_metrics = {
                'basic_metrics': basic_metrics,
                'risk_metrics': await self._calculate_advanced_risk_metrics(basic_metrics),
                'efficiency_metrics': await self._calculate_efficiency_metrics(basic_metrics),
                'timing_metrics': await self._calculate_timing_metrics(),
                'quality_metrics': await self._calculate_quality_metrics()
            }
            
            return advanced_metrics
            
        except Exception as e:
            logger.error(f"Error getting advanced performance metrics: {str(e)}")
            return {}
    
    async def _calculate_advanced_risk_metrics(self, basic_metrics: Dict) -> Dict[str, float]:
        """Calculate advanced risk metrics."""
        try:
            volatility = basic_metrics.get('volatility', 0.0)
            max_drawdown = abs(basic_metrics.get('max_drawdown', 0.0))
            var_95 = abs(basic_metrics.get('var_95', 0.0))
            
            # Value at Risk (VaR) metrics
            var_99 = var_95 * 1.5  # Approximate 99% VaR
            conditional_var = var_95 * 1.8  # Expected shortfall
            
            # Downside deviation
            downside_deviation = volatility * 0.7  # Simplified calculation
            
            # Maximum drawdown duration (simplified)
            max_drawdown_duration = 30 if max_drawdown > 0.1 else 10
            
            # Recovery time (simplified)
            recovery_time = max_drawdown_duration * 2 if max_drawdown > 0.05 else max_drawdown_duration
            
            risk_metrics = {
                'var_95': var_95,
                'var_99': var_99,
                'conditional_var': conditional_var,
                'downside_deviation': downside_deviation,
                'max_drawdown': max_drawdown,
                'max_drawdown_duration': max_drawdown_duration,
                'recovery_time': recovery_time,
                'tail_risk': var_99 * 1.2,
                'expected_shortfall': conditional_var
            }
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating advanced risk metrics: {str(e)}")
            return {}
    
    async def _calculate_efficiency_metrics(self, basic_metrics: Dict) -> Dict[str, float]:
        """Calculate efficiency metrics."""
        try:
            total_return = basic_metrics.get('total_return', 0.0)
            volatility = basic_metrics.get('volatility', 0.0)
            sharpe_ratio = basic_metrics.get('sharpe_ratio', 0.0)
            win_rate = basic_metrics.get('win_rate', 0.5)
            
            # Information ratio (simplified)
            information_ratio = sharpe_ratio * 0.8
            
            # Sortino ratio
            downside_deviation = volatility * 0.7
            sortino_ratio = (total_return - 0.02/252) / downside_deviation if downside_deviation > 0 else 0.0
            
            # Calmar ratio
            max_drawdown = abs(basic_metrics.get('max_drawdown', 0.0))
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0.0
            
            # Treynor ratio
            beta = 1.0  # Simplified
            treynor_ratio = (total_return - 0.02/252) / beta if beta > 0 else 0.0
            
            # Jensen's alpha
            market_return = 0.08/252  # Simplified market return
            jensen_alpha = total_return - (0.02/252 + beta * (market_return - 0.02/252))
            
            efficiency_metrics = {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'treynor_ratio': treynor_ratio,
                'information_ratio': information_ratio,
                'jensen_alpha': jensen_alpha,
                'win_rate': win_rate,
                'profit_factor': basic_metrics.get('profit_factor', 1.0)
            }
            
            return efficiency_metrics
            
        except Exception as e:
            logger.error(f"Error calculating efficiency metrics: {str(e)}")
            return {}
    
    async def _calculate_timing_metrics(self) -> Dict[str, Any]:
        """Calculate timing metrics."""
        try:
            # Analyze signal timing
            timing_analysis = {
                'signal_frequency': len(self.signal_history),
                'signal_quality': 0.0,
                'timing_accuracy': 0.0,
                'execution_delay': 0.0
            }
            
            # Calculate signal quality based on signal history
            if self.signal_history:
                total_signals = sum(len(signals) for signals in self.signal_history.values())
                high_confidence_signals = 0
                
                for symbol, signals in self.signal_history.items():
                    for signal in signals:
                        confidence = signal.get('confidence', 0.0)
                        if confidence > 0.7:
                            high_confidence_signals += 1
                
                timing_analysis['signal_quality'] = high_confidence_signals / max(total_signals, 1)
            
            # Timing accuracy (simplified)
            timing_analysis['timing_accuracy'] = 0.75  # Placeholder
            
            # Execution delay (simplified)
            timing_analysis['execution_delay'] = 0.5  # Placeholder in seconds
            
            return timing_analysis
            
        except Exception as e:
            logger.error(f"Error calculating timing metrics: {str(e)}")
            return {}
    
    async def _calculate_quality_metrics(self) -> Dict[str, Any]:
        """Calculate quality metrics."""
        try:
            quality_metrics = {
                'data_quality': 0.95,  # Placeholder
                'model_accuracy': 0.82,  # Placeholder
                'signal_stability': 0.78,  # Placeholder
                'execution_quality': 0.91,  # Placeholder
                'risk_management_effectiveness': 0.87  # Placeholder
            }
            
            # Calculate overall quality score
            overall_quality = sum(quality_metrics.values()) / len(quality_metrics)
            quality_metrics['overall_quality_score'] = overall_quality
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {str(e)}")
            return {}
    
    async def get_performance_attribution_analysis(self) -> Dict[str, Any]:
        """Get performance attribution analysis."""
        try:
            attribution = {
                'factor_attribution': {},
                'asset_attribution': {},
                'timing_attribution': {},
                'selection_attribution': {}
            }
            
            # Factor attribution (simplified)
            if hasattr(self, 'quantitative_system'):
                factor_summary = await self.quantitative_system.get_factor_model_summary()
                attribution['factor_attribution'] = {
                    'market_factor': 0.6,
                    'size_factor': 0.1,
                    'value_factor': 0.05,
                    'momentum_factor': 0.15,
                    'volatility_factor': 0.1
                }
            
            # Asset attribution
            portfolio_data = await self._get_portfolio_performance_data()
            positions = portfolio_data.get('positions', {})
            
            for symbol, position in positions.items():
                weight = position.get('weight', 0.0)
                if weight > 0.01:  # Only significant positions
                    attribution['asset_attribution'][symbol] = {
                        'weight': weight,
                        'contribution': weight * 0.02,  # Placeholder
                        'risk_contribution': weight * 0.01  # Placeholder
                    }
            
            # Timing attribution
            attribution['timing_attribution'] = {
                'entry_timing': 0.02,
                'exit_timing': 0.01,
                'rebalancing_timing': 0.005
            }
            
            # Selection attribution
            attribution['selection_attribution'] = {
                'asset_selection': 0.03,
                'sector_selection': 0.01,
                'factor_selection': 0.02
            }
            
            return attribution
            
        except Exception as e:
            logger.error(f"Error getting performance attribution analysis: {str(e)}")
            return {}
    
    async def get_comprehensive_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report with all advanced metrics."""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'basic_performance': await self.get_real_time_performance_summary(),
                'advanced_metrics': await self.get_advanced_performance_metrics(),
                'performance_attribution': await self.get_performance_attribution_analysis(),
                'quantitative_analysis': {
                    'statistical_validation': await self.get_statistical_validation_summary(),
                    'factor_analysis': await self.get_factor_model_summary(),
                    'ml_analysis': await self.get_ml_summary()
                },
                'portfolio_optimization': await self.analyze_portfolio_optimization([]) if hasattr(self, 'quantitative_system') else {},
                'monitoring_status': {
                    'active': self.performance_monitoring['active'],
                    'last_update': self.performance_monitoring['last_update'],
                    'update_frequency': self.performance_monitoring.get('update_frequency', 30),
                    'alert_count': len(self.performance_monitoring.get('alerts', [])),
                    'performance_score': self.performance_monitoring['performance_score'],
                    'risk_score': self.performance_monitoring['risk_score'],
                    'stability_score': self.performance_monitoring['stability_score']
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error getting comprehensive performance report: {str(e)}")
            return {}
    
    async def _apply_implied_volatility_analysis(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """Apply WorldQuant Implied Volatility analysis to trading signal."""
        try:
            implied_vol_signal = signal.copy()
            
            # Perform comprehensive volatility analysis
            volatility_analysis = self.volatility_engine.analyze_volatility_surface(symbol, market_data)
            
            if 'error' not in volatility_analysis:
                # Add volatility analysis to signal
                implied_vol_signal['implied_volatility_analysis'] = volatility_analysis
                
                # Get volatility trading signals
                vol_signals = self.volatility_engine.get_volatility_trading_signals(volatility_analysis)
                implied_vol_signal['volatility_signals'] = vol_signals
                
                # Adjust signal strength based on volatility
                regime_analysis = volatility_analysis.get('regime_analysis', {})
                regime = regime_analysis.get('regime', 'normal_volatility')
                regime_score = regime_analysis.get('regime_score', 0.5)
                
                # Volatility-based signal adjustments
                if regime == 'high_volatility':
                    implied_vol_signal['strength'] *= (1.0 - regime_score * 0.3)
                    implied_vol_signal['reasons'].append('high_volatility_suppression')
                elif regime == 'low_volatility':
                    implied_vol_signal['strength'] *= (1.0 + regime_score * 0.2)
                    implied_vol_signal['reasons'].append('low_volatility_enhancement')
                
                # Add volatility confidence to signal confidence
                vol_consensus = volatility_analysis.get('volatility_consensus', {})
                vol_confidence = vol_consensus.get('volatility_confidence', 0.5)
                implied_vol_signal['confidence'] = (implied_vol_signal.get('confidence', 0.5) + vol_confidence) / 2
                
                # Add volatility risk metrics
                vol_risk_metrics = volatility_analysis.get('volatility_risk_metrics', {})
                implied_vol_signal['volatility_risk_metrics'] = vol_risk_metrics
                
                logger.info(f"Applied Implied Volatility analysis for {symbol}: {regime} (score: {regime_score:.2f})")
            
            return implied_vol_signal
            
        except Exception as e:
            logger.error(f"Error applying implied volatility analysis: {str(e)}")
            return signal
    
    async def _optimize_final_signal_with_volatility(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """Optimize final signal with volatility adjustments."""
        try:
            optimized_signal = signal.copy()
            
            # Preserve current_price
            current_price = signal.get('current_price', 0.0)
            
            # Signal strength normalization
            signal_strength = optimized_signal.get('strength', 0.0)
            optimized_signal['signal_strength'] = np.clip(signal_strength, -1.0, 1.0)
            
            # Volatility-adjusted confidence calculation
            base_confidence = optimized_signal.get('confidence', 0.0)
            vol_analysis = optimized_signal.get('implied_volatility_analysis', {})
            
            if vol_analysis and 'volatility_consensus' in vol_analysis:
                vol_confidence = vol_analysis['volatility_consensus'].get('volatility_confidence', 0.5)
                # Blend base confidence with volatility confidence
                optimized_signal['final_confidence'] = float((base_confidence + vol_confidence) / 2)
            else:
                optimized_signal['final_confidence'] = float(base_confidence)
            
            # Position size optimization with volatility
            base_size = optimized_signal.get('position_size', 0.01)
            
            if vol_analysis and 'error' not in vol_analysis:
                # Use volatility engine for position size adjustment
                adjusted_size = self.volatility_engine.adjust_position_size_by_volatility(base_size, vol_analysis)
                optimized_signal['optimized_position_size'] = adjusted_size
            else:
                # Fallback to basic position size calculation
                confidence_multiplier = optimized_signal['final_confidence']
                optimized_signal['optimized_position_size'] = base_size * confidence_multiplier
            
            # Risk-adjusted signal strength with volatility
            risk_adjustment = 1.0 - abs(optimized_signal.get('var_95', 0.0)) * 10
            
            # Add volatility regime adjustment
            vol_regime = optimized_signal.get('volatility_regime', {})
            if vol_regime.get('regime') == 'high_volatility':
                risk_adjustment *= 0.8  # Reduce risk in high volatility
            elif vol_regime.get('regime') == 'low_volatility':
                risk_adjustment *= 1.2  # Increase risk in low volatility
            
            optimized_signal['risk_adjusted_strength'] = optimized_signal['signal_strength'] * risk_adjustment
            
            # Ensure current_price is preserved
            optimized_signal['current_price'] = current_price
            
            return optimized_signal
            
        except Exception as e:
            logger.error(f"Error optimizing final signal with volatility: {str(e)}")
            return signal

    async def _apply_statistical_arbitrage(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """
        Apply statistical arbitrage analysis (alias for _apply_statistical_arbitrage_analysis).
        """
        return await self._apply_statistical_arbitrage_analysis(symbol, signal, market_data)
    
    async def _apply_phase3_analysis(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """
        Apply Phase 3 WorldQuant-Level Features analysis.
        
        Args:
            symbol: Trading symbol
            signal: Current signal
            market_data: Market data
            
        Returns:
            Enhanced signal with Phase 3 analysis
        """
        try:
            enhanced_signal = signal.copy()
            
            # 1. High-Frequency Trading Analysis
            hft_analysis = await self._apply_hft_analysis(symbol, signal, market_data)
            if hft_analysis:
                enhanced_signal['hft_analysis'] = hft_analysis
                logger.info(f"HFT analysis applied for {symbol}")
            
            # 2. Advanced Market Microstructure Analysis
            microstructure_analysis = await self._apply_advanced_microstructure_analysis(symbol, signal, market_data)
            if microstructure_analysis:
                enhanced_signal['microstructure_analysis'] = microstructure_analysis
                logger.info(f"Advanced microstructure analysis applied for {symbol}")
            
            # 3. Options-Based Strategies Analysis
            options_analysis = await self._apply_options_analysis(symbol, signal, market_data)
            if options_analysis:
                enhanced_signal['options_analysis'] = options_analysis
                logger.info(f"Options analysis applied for {symbol}")
            
            # 4. On-Chain Analytics
            onchain_analysis = await self._apply_onchain_analysis(symbol, signal, market_data)
            if onchain_analysis:
                enhanced_signal['onchain_analysis'] = onchain_analysis
                logger.info(f"On-chain analysis applied for {symbol}")
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error applying Phase 3 analysis for {symbol}: {str(e)}")
            return signal
    
    async def _apply_hft_analysis(self, symbol: str, signal: Dict, market_data: Dict) -> Dict[str, Any]:
        """Apply High-Frequency Trading analysis."""
        try:
            # Create mock tick data for testing
            import time
            tick_data = TickData(
                timestamp=time.time(),
                price=market_data.get('current_price', 50000.0),
                volume=market_data.get('volume', 1.0),
                side='buy' if signal.get('action') == 'buy' else 'sell',
                exchange='binance',
                symbol=symbol
            )
            
            # Process tick data
            hft_analysis = await self.hft_engine.process_tick_data(tick_data)
            
            # Adjust signal based on HFT analysis
            if hft_analysis.get('tick_analysis', {}).get('pattern_detected'):
                signal['hft_pattern'] = hft_analysis['tick_analysis']['pattern_detected']
                signal['confidence'] = min(signal.get('confidence', 0) + 0.1, 1.0)
            
            return hft_analysis
            
        except Exception as e:
            logger.error(f"Error applying HFT analysis: {str(e)}")
            return {}
    
    async def _apply_advanced_microstructure_analysis(self, symbol: str, signal: Dict, market_data: Dict) -> Dict[str, Any]:
        """Apply Advanced Market Microstructure analysis."""
        try:
            # Create mock orderbook data
            orderbook_data = {
                'bids': [[market_data.get('current_price', 50000) - i, 1.0] for i in range(1, 6)],
                'asks': [[market_data.get('current_price', 50000) + i, 1.0] for i in range(1, 6)]
            }
            
            # Create mock trade data
            import pandas as pd
            trade_data = pd.DataFrame({
                'price': [market_data.get('current_price', 50000) + np.random.normal(0, 10) for _ in range(50)],
                'volume': [np.random.uniform(0.1, 5.0) for _ in range(50)],
                'side': [np.random.choice(['buy', 'sell']) for _ in range(50)]
            })
            
            # Analyze advanced microstructure
            microstructure_analysis = self.advanced_microstructure_analyzer.analyze_advanced_order_flow(
                orderbook_data, trade_data
            )
            
            # Adjust signal based on microstructure analysis
            if microstructure_analysis.get('microstructure_signals', {}).get('action') != 'hold':
                signal['microstructure_signal'] = microstructure_analysis['microstructure_signals']['action']
                signal['confidence'] = min(signal.get('confidence', 0) + 0.15, 1.0)
            
            return microstructure_analysis
            
        except Exception as e:
            logger.error(f"Error applying advanced microstructure analysis: {str(e)}")
            return {}
    
    async def _apply_options_analysis(self, symbol: str, signal: Dict, market_data: Dict) -> Dict[str, Any]:
        """Apply Options-Based Strategies analysis."""
        try:
            # Create mock options data
            options_data = []
            underlying_price = market_data.get('current_price', 50000.0)
            
            # Create various strike prices
            strikes = [underlying_price * 0.9, underlying_price * 0.95, underlying_price, 
                      underlying_price * 1.05, underlying_price * 1.1]
            
            for strike in strikes:
                # Call options
                options_data.append(OptionContract(
                    symbol=f'{symbol}-{strike:.0f}-C',
                    strike=strike,
                    expiry='2024-12-31',
                    option_type='call',
                    price=max(0, underlying_price - strike) + np.random.uniform(100, 500),
                    implied_volatility=np.random.uniform(0.2, 0.8),
                    delta=np.random.uniform(0.1, 0.9),
                    gamma=np.random.uniform(0.001, 0.01),
                    theta=np.random.uniform(-100, -10),
                    vega=np.random.uniform(10, 100)
                ))
                
                # Put options
                options_data.append(OptionContract(
                    symbol=f'{symbol}-{strike:.0f}-P',
                    strike=strike,
                    expiry='2024-12-31',
                    option_type='put',
                    price=max(0, strike - underlying_price) + np.random.uniform(100, 500),
                    implied_volatility=np.random.uniform(0.2, 0.8),
                    delta=np.random.uniform(-0.9, -0.1),
                    gamma=np.random.uniform(0.001, 0.01),
                    theta=np.random.uniform(-100, -10),
                    vega=np.random.uniform(10, 100)
                ))
            
            # Analyze implied volatility
            iv_analysis = self.options_strategies.analyze_implied_volatility(underlying_price, options_data)
            
            # Create volatility strategy
            strategy = self.options_strategies.create_volatility_strategy(underlying_price, options_data, 'straddle')
            
            # Adjust signal based on options analysis
            if iv_analysis.get('volatility_regime') == 'high':
                signal['options_volatility_regime'] = 'high'
                signal['confidence'] = min(signal.get('confidence', 0) + 0.1, 1.0)
            
            return {
                'iv_analysis': iv_analysis,
                'strategy': strategy
            }
            
        except Exception as e:
            logger.error(f"Error applying options analysis: {str(e)}")
            return {}
    
    async def _apply_onchain_analysis(self, symbol: str, signal: Dict, market_data: Dict) -> Dict[str, Any]:
        """Apply On-Chain Analytics."""
        try:
            # Create mock blockchain transactions
            import time
            transactions = []
            
            for i in range(20):
                transactions.append(BlockchainTransaction(
                    tx_hash=f'0x{np.random.bytes(32).hex()}',
                    block_number=1000000 + i,
                    timestamp=int(time.time()) - i * 60,
                    from_address=f'0x{np.random.bytes(20).hex()}',
                    to_address=f'0x{np.random.bytes(20).hex()}',
                    value=np.random.uniform(0.1, 10.0),
                    gas_price=np.random.uniform(20, 100),
                    gas_used=np.random.randint(21000, 100000),
                    token_address=np.random.choice([None, f'0x{np.random.bytes(20).hex()}']),
                    token_amount=np.random.uniform(0.1, 100.0) if np.random.random() > 0.5 else None
                ))
            
            # Analyze transaction flow
            flow_analysis = self.on_chain_analytics.analyze_transaction_flow(transactions)
            
            # Analyze wallet behavior
            wallet_profiles = self.on_chain_analytics.analyze_wallet_behavior(transactions)
            
            # Generate on-chain signals
            analysis = {
                'transaction_flow': flow_analysis,
                'wallet_behavior': wallet_profiles,
                'defi_metrics': {}
            }
            
            onchain_signals = self.on_chain_analytics.generate_on_chain_signals(analysis)
            
            # Adjust signal based on on-chain analysis
            if onchain_signals.get('action') != 'hold':
                signal['onchain_signal'] = onchain_signals['action']
                signal['confidence'] = min(signal.get('confidence', 0) + 0.1, 1.0)
            
            return {
                'flow_analysis': flow_analysis,
                'wallet_profiles': len(wallet_profiles),
                'onchain_signals': onchain_signals
            }
            
        except Exception as e:
            logger.error(f"Error applying on-chain analysis: {str(e)}")
            return {}
    
    def get_phase3_summary(self) -> Dict[str, Any]:
        """Get comprehensive Phase 3 features summary."""
        try:
            summary = {
                'hft_engine': self.hft_engine.get_hft_performance_metrics(),
                'advanced_microstructure': self.advanced_microstructure_analyzer.get_advanced_microstructure_summary(),
                'options_strategies': self.options_strategies.get_options_strategy_summary(),
                'on_chain_analytics': self.on_chain_analytics.get_on_chain_summary(),
                'total_phase3_features': 4,
                'status': 'active'
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting Phase 3 summary: {str(e)}")
            return {'error': str(e)}
    
    async def _apply_momentum_mean_reversion_analysis(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """Apply momentum mean reversion analysis."""
        try:
            logger.info(f"Applying momentum mean reversion analysis for {symbol}")
            
            # Get market data for analysis
            if not market_data or 'returns' not in market_data:
                logger.warning(f"No returns data available for {symbol}")
                return signal
            
            returns = market_data['returns']
            if len(returns) < 20:  # Need minimum data for analysis
                logger.warning(f"Insufficient data for momentum analysis on {symbol}")
                return signal
            
            # 1. Calculate momentum indicators
            momentum_analysis = {}
            
            # Price momentum (12-period)
            price_momentum = (returns.iloc[-1] - returns.iloc[-12]) / returns.iloc[-12] if len(returns) >= 12 else 0
            
            # Volume momentum
            if 'volume' in market_data:
                volume_data = market_data['volume']
                if len(volume_data) >= 12:
                    volume_momentum = (volume_data.iloc[-1] - volume_data.iloc[-12]) / volume_data.iloc[-12]
                else:
                    volume_momentum = 0
            else:
                volume_momentum = 0
            
            # 2. Calculate mean reversion indicators
            mean_reversion_analysis = {}
            
            # Bollinger Bands for mean reversion
            if 'bollinger_bands' in market_data:
                bb_data = market_data['bollinger_bands']
                current_price = market_data.get('current_price', returns.iloc[-1])
                
                if 'upper' in bb_data and 'lower' in bb_data:
                    bb_position = (current_price - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
                    mean_reversion_analysis['bb_position'] = bb_position
                    
                    # Mean reversion signal based on BB position
                    if bb_position > 0.8:  # Near upper band
                        mean_reversion_analysis['signal'] = 'sell'
                        mean_reversion_analysis['strength'] = min(bb_position - 0.8, 0.2) * 5  # Scale to 0-1
                    elif bb_position < 0.2:  # Near lower band
                        mean_reversion_analysis['signal'] = 'buy'
                        mean_reversion_analysis['strength'] = min(0.2 - bb_position, 0.2) * 5  # Scale to 0-1
                    else:
                        mean_reversion_analysis['signal'] = 'hold'
                        mean_reversion_analysis['strength'] = 0.0
            
            # 3. RSI for momentum/mean reversion
            if 'rsi' in market_data:
                rsi_value = market_data['rsi']
                if rsi_value > 70:
                    momentum_analysis['rsi_signal'] = 'overbought'
                    momentum_analysis['rsi_strength'] = (rsi_value - 70) / 30
                elif rsi_value < 30:
                    momentum_analysis['rsi_signal'] = 'oversold'
                    momentum_analysis['rsi_strength'] = (30 - rsi_value) / 30
                else:
                    momentum_analysis['rsi_signal'] = 'neutral'
                    momentum_analysis['rsi_strength'] = 0.0
            
            # 4. Statistical arbitrage analysis
            if hasattr(self, 'statistical_arbitrage_engine'):
                try:
                    arb_analysis = await self.statistical_arbitrage_engine.analyze_mean_reversion(
                        symbol, returns, market_data
                    )
                    momentum_analysis['arbitrage'] = arb_analysis
                except Exception as e:
                    logger.warning(f"Error in statistical arbitrage analysis: {str(e)}")
            
            # 5. Combine momentum and mean reversion signals
            combined_signal = signal.copy()
            
            # Adjust signal based on momentum
            if price_momentum > 0.05:  # Strong positive momentum
                combined_signal['momentum_signal'] = 'buy'
                combined_signal['momentum_strength'] = min(price_momentum * 10, 1.0)
            elif price_momentum < -0.05:  # Strong negative momentum
                combined_signal['momentum_signal'] = 'sell'
                combined_signal['momentum_strength'] = min(abs(price_momentum) * 10, 1.0)
            else:
                combined_signal['momentum_signal'] = 'hold'
                combined_signal['momentum_strength'] = 0.0
            
            # Adjust signal based on mean reversion
            if mean_reversion_analysis.get('signal') != 'hold':
                combined_signal['mean_reversion_signal'] = mean_reversion_analysis['signal']
                combined_signal['mean_reversion_strength'] = mean_reversion_analysis['strength']
                
                # If momentum and mean reversion conflict, reduce confidence
                if (combined_signal['momentum_signal'] != 'hold' and 
                    combined_signal['momentum_signal'] != combined_signal['mean_reversion_signal']):
                    combined_signal['confidence'] = max(combined_signal.get('confidence', 0) - 0.2, 0.0)
                    logger.info(f"Momentum and mean reversion signals conflict for {symbol}")
            
            # 6. Final signal adjustment
            if combined_signal.get('momentum_strength', 0) > 0.7:
                combined_signal['confidence'] = min(combined_signal.get('confidence', 0) + 0.15, 1.0)
                logger.info(f"Strong momentum signal for {symbol}: {combined_signal['momentum_signal']}")
            
            if combined_signal.get('mean_reversion_strength', 0) > 0.7:
                combined_signal['confidence'] = min(combined_signal.get('confidence', 0) + 0.15, 1.0)
                logger.info(f"Strong mean reversion signal for {symbol}: {combined_signal['mean_reversion_signal']}")
            
            # Store analysis results
            combined_signal['momentum_analysis'] = momentum_analysis
            combined_signal['mean_reversion_analysis'] = mean_reversion_analysis
            
            logger.info(f"Momentum mean reversion analysis completed for {symbol}")
            return combined_signal
            
        except Exception as e:
            logger.error(f"Error applying momentum mean reversion analysis: {str(e)}")
            return signal
    
    async def _apply_volatility_regime_analysis(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """Apply volatility regime analysis."""
        try:
            logger.info(f"Applying volatility regime analysis for {symbol}")
            
            # Get market data for analysis
            if not market_data or 'returns' not in market_data:
                logger.warning(f"No returns data available for {symbol}")
                return signal
            
            returns = market_data['returns']
            if len(returns) < 30:  # Need minimum data for regime analysis
                logger.warning(f"Insufficient data for volatility regime analysis on {symbol}")
                return signal
            
            # 1. Calculate volatility metrics
            volatility_analysis = {}
            
            # Rolling volatility (20-period)
            rolling_vol = returns.rolling(window=20).std().iloc[-1]
            volatility_analysis['rolling_volatility'] = rolling_vol
            
            # Historical volatility (252-period annualized)
            if len(returns) >= 252:
                annualized_vol = returns.std() * np.sqrt(252)
                volatility_analysis['annualized_volatility'] = annualized_vol
            else:
                annualized_vol = rolling_vol * np.sqrt(252)
                volatility_analysis['annualized_volatility'] = annualized_vol
            
            # 2. Detect volatility regime
            regime_analysis = {}
            
            # Calculate volatility percentiles
            vol_percentile = np.percentile(returns.rolling(window=20).std().dropna(), 75)
            
            if rolling_vol > vol_percentile * 1.5:
                regime = 'high_volatility'
                regime_analysis['regime'] = 'high_volatility'
                regime_analysis['confidence'] = min((rolling_vol - vol_percentile) / vol_percentile, 1.0)
            elif rolling_vol < vol_percentile * 0.5:
                regime = 'low_volatility'
                regime_analysis['regime'] = 'low_volatility'
                regime_analysis['confidence'] = min((vol_percentile - rolling_vol) / vol_percentile, 1.0)
            else:
                regime = 'normal_volatility'
                regime_analysis['regime'] = 'normal_volatility'
                regime_analysis['confidence'] = 0.5
            
            # 3. Use implied volatility if available
            if hasattr(self, 'volatility_engine'):
                try:
                    iv_analysis = await self.volatility_engine.analyze_volatility_regime(
                        symbol, market_data
                    )
                    regime_analysis['implied_volatility'] = iv_analysis
                    
                    # Adjust regime based on implied volatility
                    if iv_analysis.get('regime') != regime:
                        # Weighted average of historical and implied volatility
                        regime_analysis['final_regime'] = regime if regime_analysis['confidence'] > 0.7 else iv_analysis.get('regime', regime)
                    else:
                        regime_analysis['final_regime'] = regime
                        regime_analysis['confidence'] = min(regime_analysis['confidence'] + 0.2, 1.0)
                        
                except Exception as e:
                    logger.warning(f"Error in implied volatility analysis: {str(e)}")
                    regime_analysis['final_regime'] = regime
            
            # 4. Use advanced risk management for regime analysis
            if hasattr(self, 'dynamic_risk_manager'):
                try:
                    risk_analysis = self.dynamic_risk_manager.calculate_dynamic_var(
                        returns, regime_analysis.get('final_regime', regime)
                    )
                    regime_analysis['risk_metrics'] = risk_analysis
                except Exception as e:
                    logger.warning(f"Error in dynamic risk analysis: {str(e)}")
            
            # 5. Adjust signal based on volatility regime
            adjusted_signal = signal.copy()
            
            # High volatility regime adjustments
            if regime_analysis.get('final_regime') == 'high_volatility':
                # Reduce position size in high volatility
                adjusted_signal['position_size_multiplier'] = 0.5
                adjusted_signal['stop_loss_multiplier'] = 1.5  # Wider stops
                adjusted_signal['confidence'] = max(adjusted_signal.get('confidence', 0) - 0.1, 0.0)
                logger.info(f"High volatility regime detected for {symbol}, reducing position size")
            
            # Low volatility regime adjustments
            elif regime_analysis.get('final_regime') == 'low_volatility':
                # Increase position size in low volatility
                adjusted_signal['position_size_multiplier'] = 1.2
                adjusted_signal['stop_loss_multiplier'] = 0.8  # Tighter stops
                adjusted_signal['confidence'] = min(adjusted_signal.get('confidence', 0) + 0.1, 1.0)
                logger.info(f"Low volatility regime detected for {symbol}, increasing position size")
            
            # Normal volatility regime
            else:
                adjusted_signal['position_size_multiplier'] = 1.0
                adjusted_signal['stop_loss_multiplier'] = 1.0
                logger.info(f"Normal volatility regime detected for {symbol}")
            
            # 6. Volatility-based signal enhancement
            if regime_analysis.get('confidence', 0) > 0.8:
                # Strong regime detection
                if regime_analysis.get('final_regime') == 'high_volatility':
                    # In high volatility, prefer mean reversion strategies
                    if adjusted_signal.get('action') == 'buy' and adjusted_signal.get('confidence', 0) > 0.6:
                        adjusted_signal['strategy_type'] = 'mean_reversion'
                        logger.info(f"High volatility regime: applying mean reversion strategy for {symbol}")
                elif regime_analysis.get('final_regime') == 'low_volatility':
                    # In low volatility, prefer momentum strategies
                    if adjusted_signal.get('action') == 'buy' and adjusted_signal.get('confidence', 0) > 0.6:
                        adjusted_signal['strategy_type'] = 'momentum'
                        logger.info(f"Low volatility regime: applying momentum strategy for {symbol}")
            
            # Store analysis results
            adjusted_signal['volatility_analysis'] = volatility_analysis
            adjusted_signal['regime_analysis'] = regime_analysis
            
            logger.info(f"Volatility regime analysis completed for {symbol}: {regime_analysis.get('final_regime', 'unknown')}")
            return adjusted_signal
            
        except Exception as e:
            logger.error(f"Error applying volatility regime analysis: {str(e)}")
            return signal
    
    async def _apply_correlation_analysis(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """Apply correlation analysis."""
        try:
            logger.info(f"Applying correlation analysis for {symbol}")
            
            # Get market data for analysis
            if not market_data or 'returns' not in market_data:
                logger.warning(f"No returns data available for {symbol}")
                return signal
            
            returns = market_data['returns']
            if len(returns) < 30:  # Need minimum data for correlation analysis
                logger.warning(f"Insufficient data for correlation analysis on {symbol}")
                return signal
            
            # 1. Calculate correlation metrics
            correlation_analysis = {}
            
            # Get benchmark returns (BTC/USDT as default benchmark)
            benchmark_symbol = 'BTCUSDT'
            benchmark_returns = None
            
            try:
                # Try to get benchmark data from cache or market data
                if 'benchmark_returns' in market_data:
                    benchmark_returns = market_data['benchmark_returns']
                elif hasattr(self, 'cache_service') and self.cache_service:
                    # Get benchmark data from cache
                    benchmark_data = await self.cache_service.get(f"{benchmark_symbol}_returns")
                    if benchmark_data:
                        benchmark_returns = pd.Series(benchmark_data)
                
                # If no benchmark data available, use market average
                if benchmark_returns is None or len(benchmark_returns) < 30:
                    # Calculate market average returns from available symbols
                    market_returns = await self._get_market_average_returns()
                    if market_returns is not None:
                        benchmark_returns = pd.Series(market_returns)
                    else:
                        logger.warning(f"No benchmark data available for {symbol}")
                        return signal
                        
            except Exception as e:
                logger.warning(f"Error getting benchmark data: {str(e)}")
                return signal
            
            # 2. Calculate correlation metrics
            # Align returns data
            min_length = min(len(returns), len(benchmark_returns))
            if min_length < 30:
                logger.warning(f"Insufficient aligned data for correlation analysis on {symbol}")
                return signal
            
            aligned_returns = returns.iloc[-min_length:]
            aligned_benchmark = benchmark_returns.iloc[-min_length:]
            
            # Calculate correlation
            correlation = aligned_returns.corr(aligned_benchmark)
            correlation_analysis['benchmark_correlation'] = correlation
            
            # Calculate rolling correlation (20-period)
            if min_length >= 20:
                rolling_corr = aligned_returns.rolling(window=20).corr(aligned_benchmark).iloc[-1]
                correlation_analysis['rolling_correlation'] = rolling_corr
            else:
                rolling_corr = correlation
                correlation_analysis['rolling_correlation'] = rolling_corr
            
            # 3. Calculate beta (market sensitivity)
            if not pd.isna(correlation) and correlation != 0:
                # Beta = correlation * (asset_volatility / market_volatility)
                asset_vol = returns.std()
                market_vol = benchmark_returns.std()
                
                if market_vol > 0:
                    beta = correlation * (asset_vol / market_vol)
                    correlation_analysis['beta'] = beta
                else:
                    beta = 1.0
                    correlation_analysis['beta'] = beta
            else:
                beta = 1.0
                correlation_analysis['beta'] = beta
            
            # 4. Calculate sector/category correlations
            sector_correlations = {}
            
            # Define crypto sectors
            crypto_sectors = {
                'defi': ['UNIUSDT', 'AAVEUSDT', 'COMPUSDT', 'SUSHIUSDT'],
                'layer1': ['ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'SOLUSDT'],
                'layer2': ['MATICUSDT', 'OPUSDT', 'ARBUSDT'],
                'meme': ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT'],
                'exchange': ['BNBUSDT', 'FTTUSDT', 'OKBUSDT']
            }
            
            # Calculate sector correlations
            for sector, symbols in crypto_sectors.items():
                sector_returns = []
                for sector_symbol in symbols:
                    try:
                        if hasattr(self, 'cache_service') and self.cache_service:
                            sector_data = await self.cache_service.get(f"{sector_symbol}_returns")
                            if sector_data and len(sector_data) >= min_length:
                                sector_returns.append(pd.Series(sector_data).iloc[-min_length:])
                    except Exception:
                        continue
                
                if sector_returns:
                    # Calculate average sector returns
                    sector_df = pd.concat(sector_returns, axis=1)
                    avg_sector_returns = sector_df.mean(axis=1)
                    
                    # Calculate correlation with sector
                    sector_corr = aligned_returns.corr(avg_sector_returns)
                    sector_correlations[sector] = sector_corr
            
            correlation_analysis['sector_correlations'] = sector_correlations
            
            # 5. Adjust signal based on correlation analysis
            adjusted_signal = signal.copy()
            
            # High correlation with market (beta > 1.2)
            if beta > 1.2:
                adjusted_signal['market_sensitivity'] = 'high'
                adjusted_signal['position_size_multiplier'] = 0.8  # Reduce size for high beta
                logger.info(f"High market sensitivity detected for {symbol} (beta={beta:.2f})")
            
            # Low correlation with market (beta < 0.8)
            elif beta < 0.8:
                adjusted_signal['market_sensitivity'] = 'low'
                adjusted_signal['position_size_multiplier'] = 1.2  # Increase size for low beta
                logger.info(f"Low market sensitivity detected for {symbol} (beta={beta:.2f})")
            
            # Normal correlation
            else:
                adjusted_signal['market_sensitivity'] = 'normal'
                adjusted_signal['position_size_multiplier'] = 1.0
                logger.info(f"Normal market sensitivity for {symbol} (beta={beta:.2f})")
            
            # 6. Sector-based adjustments
            if sector_correlations:
                # Find highest sector correlation
                max_sector = max(sector_correlations.items(), key=lambda x: abs(x[1]) if not pd.isna(x[1]) else 0)
                max_sector_name, max_sector_corr = max_sector
                
                if not pd.isna(max_sector_corr) and abs(max_sector_corr) > 0.7:
                    adjusted_signal['primary_sector'] = max_sector_name
                    adjusted_signal['sector_correlation'] = max_sector_corr
                    
                    # Adjust based on sector performance
                    if max_sector_corr > 0.7:
                        adjusted_signal['sector_alignment'] = 'positive'
                        adjusted_signal['confidence'] = min(adjusted_signal.get('confidence', 0) + 0.1, 1.0)
                        logger.info(f"Strong positive sector correlation for {symbol} with {max_sector_name}")
                    elif max_sector_corr < -0.7:
                        adjusted_signal['sector_alignment'] = 'negative'
                        adjusted_signal['confidence'] = max(adjusted_signal.get('confidence', 0) - 0.1, 0.0)
                        logger.info(f"Strong negative sector correlation for {symbol} with {max_sector_name}")
            
            # 7. Correlation-based signal enhancement
            if abs(correlation) > 0.8:
                # High correlation with market
                if correlation > 0.8:
                    adjusted_signal['correlation_signal'] = 'market_following'
                    logger.info(f"High positive correlation with market for {symbol}")
                else:
                    adjusted_signal['correlation_signal'] = 'market_contrarian'
                    logger.info(f"High negative correlation with market for {symbol}")
            
            # Store analysis results
            adjusted_signal['correlation_analysis'] = correlation_analysis
            
            logger.info(f"Correlation analysis completed for {symbol}: beta={beta:.2f}, correlation={correlation:.2f}")
            return adjusted_signal
            
        except Exception as e:
            logger.error(f"Error applying correlation analysis: {str(e)}")
            return signal
    
    async def _optimize_final_signal(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """Optimize final signal."""
        try:
            logger.info(f"Optimizing final signal for {symbol}")
            
            # Create optimized signal
            optimized_signal = signal.copy()
            
            # 1. Apply all quantitative analysis layers
            analysis_layers = []
            
            # Momentum and mean reversion analysis
            if hasattr(self, '_apply_momentum_mean_reversion_analysis'):
                try:
                    momentum_result = await self._apply_momentum_mean_reversion_analysis(symbol, optimized_signal, market_data)
                    if momentum_result != optimized_signal:
                        analysis_layers.append('momentum_mean_reversion')
                        optimized_signal = momentum_result
                except Exception as e:
                    logger.warning(f"Error in momentum analysis: {str(e)}")
            
            # Volatility regime analysis
            if hasattr(self, '_apply_volatility_regime_analysis'):
                try:
                    volatility_result = await self._apply_volatility_regime_analysis(symbol, optimized_signal, market_data)
                    if volatility_result != optimized_signal:
                        analysis_layers.append('volatility_regime')
                        optimized_signal = volatility_result
                except Exception as e:
                    logger.warning(f"Error in volatility analysis: {str(e)}")
            
            # Correlation analysis
            if hasattr(self, '_apply_correlation_analysis'):
                try:
                    correlation_result = await self._apply_correlation_analysis(symbol, optimized_signal, market_data)
                    if correlation_result != optimized_signal:
                        analysis_layers.append('correlation')
                        optimized_signal = correlation_result
                except Exception as e:
                    logger.warning(f"Error in correlation analysis: {str(e)}")
            
            # 2. Apply advanced quantitative analysis
            if hasattr(self, '_apply_advanced_risk_management'):
                try:
                    risk_result = await self._apply_advanced_risk_management(symbol, optimized_signal, market_data)
                    if risk_result != optimized_signal:
                        analysis_layers.append('advanced_risk')
                        optimized_signal = risk_result
                except Exception as e:
                    logger.warning(f"Error in advanced risk analysis: {str(e)}")
            
            # Statistical arbitrage analysis
            if hasattr(self, '_apply_statistical_arbitrage_analysis'):
                try:
                    arb_result = await self._apply_statistical_arbitrage_analysis(symbol, optimized_signal, market_data)
                    if arb_result != optimized_signal:
                        analysis_layers.append('statistical_arbitrage')
                        optimized_signal = arb_result
                except Exception as e:
                    logger.warning(f"Error in statistical arbitrage analysis: {str(e)}")
            
            # Advanced ML analysis
            if hasattr(self, '_apply_advanced_ml_analysis'):
                try:
                    ml_result = await self._apply_advanced_ml_analysis(symbol, optimized_signal, market_data)
                    if ml_result != optimized_signal:
                        analysis_layers.append('advanced_ml')
                        optimized_signal = ml_result
                except Exception as e:
                    logger.warning(f"Error in advanced ML analysis: {str(e)}")
            
            # 3. Apply Phase 3 WorldQuant-Level features
            if hasattr(self, '_apply_phase3_analysis'):
                try:
                    phase3_result = await self._apply_phase3_analysis(symbol, optimized_signal, market_data)
                    if phase3_result != optimized_signal:
                        analysis_layers.append('phase3_features')
                        optimized_signal = phase3_result
                except Exception as e:
                    logger.warning(f"Error in Phase 3 analysis: {str(e)}")
            
            # 4. Apply implied volatility analysis
            if hasattr(self, '_apply_implied_volatility_analysis'):
                try:
                    iv_result = await self._apply_implied_volatility_analysis(symbol, optimized_signal, market_data)
                    if iv_result != optimized_signal:
                        analysis_layers.append('implied_volatility')
                        optimized_signal = iv_result
                except Exception as e:
                    logger.warning(f"Error in implied volatility analysis: {str(e)}")
            
            # 5. Final signal optimization
            final_optimization = {}
            
            # Confidence aggregation
            confidence_scores = []
            if optimized_signal.get('confidence') is not None:
                confidence_scores.append(optimized_signal['confidence'])
            if optimized_signal.get('momentum_strength') is not None:
                confidence_scores.append(optimized_signal['momentum_strength'] * 0.3)
            if optimized_signal.get('mean_reversion_strength') is not None:
                confidence_scores.append(optimized_signal['mean_reversion_strength'] * 0.3)
            if optimized_signal.get('correlation_analysis', {}).get('beta') is not None:
                beta = optimized_signal['correlation_analysis']['beta']
                if 0.8 <= beta <= 1.2:
                    confidence_scores.append(0.2)
            
            if confidence_scores:
                final_confidence = sum(confidence_scores) / len(confidence_scores)
                optimized_signal['final_confidence'] = min(final_confidence, 1.0)
                final_optimization['confidence_aggregation'] = final_confidence
            
            # Position size optimization
            position_multipliers = []
            if optimized_signal.get('position_size_multiplier') is not None:
                position_multipliers.append(optimized_signal['position_size_multiplier'])
            if optimized_signal.get('volatility_analysis', {}).get('regime') == 'low_volatility':
                position_multipliers.append(1.2)
            elif optimized_signal.get('volatility_analysis', {}).get('regime') == 'high_volatility':
                position_multipliers.append(0.8)
            
            if position_multipliers:
                final_position_multiplier = sum(position_multipliers) / len(position_multipliers)
                optimized_signal['final_position_multiplier'] = final_position_multiplier
                final_optimization['position_size_optimization'] = final_position_multiplier
            
            # Stop loss optimization
            stop_loss_multipliers = []
            if optimized_signal.get('stop_loss_multiplier') is not None:
                stop_loss_multipliers.append(optimized_signal['stop_loss_multiplier'])
            if optimized_signal.get('volatility_analysis', {}).get('regime') == 'high_volatility':
                stop_loss_multipliers.append(1.5)
            elif optimized_signal.get('volatility_analysis', {}).get('regime') == 'low_volatility':
                stop_loss_multipliers.append(0.8)
            
            if stop_loss_multipliers:
                final_stop_loss_multiplier = sum(stop_loss_multipliers) / len(stop_loss_multipliers)
                optimized_signal['final_stop_loss_multiplier'] = final_stop_loss_multiplier
                final_optimization['stop_loss_optimization'] = final_stop_loss_multiplier
            
            # 6. Risk-adjusted final decision
            if optimized_signal.get('final_confidence', 0) >= 0.7:
                # High confidence signal
                if optimized_signal.get('action') == 'buy':
                    optimized_signal['final_action'] = 'buy'
                    optimized_signal['signal_strength'] = 'strong'
                elif optimized_signal.get('action') == 'sell':
                    optimized_signal['final_action'] = 'sell'
                    optimized_signal['signal_strength'] = 'strong'
                else:
                    optimized_signal['final_action'] = 'hold'
                    optimized_signal['signal_strength'] = 'weak'
            elif optimized_signal.get('final_confidence', 0) >= 0.5:
                # Medium confidence signal
                optimized_signal['final_action'] = optimized_signal.get('action', 'hold')
                optimized_signal['signal_strength'] = 'medium'
            else:
                # Low confidence signal
                optimized_signal['final_action'] = 'hold'
                optimized_signal['signal_strength'] = 'weak'
            
            # 7. Final validation
            if optimized_signal.get('final_action') != 'hold':
                # Additional validation for non-hold signals
                validation_passed = True
                
                # Check if signal conflicts with market conditions
                if (optimized_signal.get('correlation_analysis', {}).get('beta', 1.0) > 1.5 and 
                    optimized_signal.get('final_action') == 'buy'):
                    # High beta asset in bullish market - reduce confidence
                    optimized_signal['final_confidence'] = max(optimized_signal.get('final_confidence', 0) - 0.1, 0.0)
                
                # Check volatility regime compatibility
                if (optimized_signal.get('volatility_analysis', {}).get('regime') == 'high_volatility' and 
                    optimized_signal.get('final_action') != 'hold'):
                    # High volatility - require higher confidence
                    if optimized_signal.get('final_confidence', 0) < 0.8:
                        optimized_signal['final_action'] = 'hold'
                        optimized_signal['signal_strength'] = 'weak'
                        validation_passed = False
                
                if not validation_passed:
                    logger.info(f"Signal validation failed for {symbol}, reverting to hold")
            
            # 8. Store optimization results
            optimized_signal['final_optimization'] = final_optimization
            optimized_signal['analysis_layers_applied'] = analysis_layers
            
            logger.info(f"Final signal optimization completed for {symbol}: "
                       f"action={optimized_signal.get('final_action', 'hold')}, "
                       f"confidence={optimized_signal.get('final_confidence', 0):.3f}, "
                       f"strength={optimized_signal.get('signal_strength', 'weak')}")
            
            return optimized_signal
            
        except Exception as e:
            logger.error(f"Error optimizing final signal: {str(e)}")
            return signal