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

logger = logging.getLogger(__name__)

class EnhancedTradingStrategyWithQuantitative:
    """
    Enhanced trading strategy with quantitative analysis integration.
    Extends the original strategy with quantitative trading capabilities.
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
        
        # Initialize quantitative integration
        quantitative_config = {
            'confidence_level': config.get('quantitative_confidence_level', 0.95),
            'max_position_size': config.get('quantitative_max_position_size', 0.02),
            'risk_free_rate': config.get('quantitative_risk_free_rate', 0.02),
            'optimization_method': config.get('quantitative_optimization_method', 'markowitz'),
            'n_factors': config.get('quantitative_n_factors', 5),
            'var_limit': config.get('quantitative_var_limit', 0.02),
            'quantitative_integration_enabled': config.get('quantitative_integration_enabled', True)
        }
        
        self.quantitative_integration = QuantitativeIntegration(quantitative_config)
        
        # Initialize other services
        self.sentiment_service = SentimentService(config)
        
        # Performance tracking
        self.performance_metrics = {}
        self.signal_history = {}
        self.quantitative_analysis_history = {}
        
        # Cache for optimization
        self.data_cache = {}
        self.last_analysis_time = {}
        
        logger.info("Enhanced Trading Strategy with Quantitative Integration initialized")
    
    async def initialize(self) -> bool:
        """Initialize the strategy."""
        try:
            logger.info("Initializing Enhanced Trading Strategy with Quantitative Integration...")
            
            # Initialize quantitative integration
            if self.quantitative_integration:
                logger.info("Quantitative integration initialized")
            
            # Initialize sentiment service
            if self.sentiment_service:
                logger.info("Sentiment service initialized")
            
            logger.info("Enhanced Trading Strategy with Quantitative Integration initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing strategy: {str(e)}")
            return False
    
    async def generate_signals(self, symbol: str, indicator_service: IndicatorService) -> Optional[Dict]:
        """
        Generate trading signals with WorldQuant-level quantitative enhancement.
        
        Args:
            symbol: Trading symbol
            indicator_service: Indicator service
            
        Returns:
            Optional[Dict]: Enhanced trading signal with sophisticated analysis
        """
        try:
            # Get comprehensive market data
            market_data = await self._get_comprehensive_market_data(symbol)
            
            # Generate base signal using advanced strategy
            base_signal = await self._generate_advanced_signal(symbol, indicator_service, market_data)
            
            if not base_signal:
                return None
            
            # Apply market microstructure analysis
            microstructure_signal = await self._apply_market_microstructure_analysis(symbol, base_signal, market_data)
            
            # Apply advanced risk management
            risk_adjusted_signal = await self._apply_advanced_risk_management(symbol, microstructure_signal, market_data)
            
            # Apply statistical arbitrage signals
            stat_arb_signal = await self._apply_statistical_arbitrage(symbol, risk_adjusted_signal, market_data)
            
            # Apply momentum and mean reversion analysis
            momentum_signal = await self._apply_momentum_mean_reversion_analysis(symbol, stat_arb_signal, market_data)
            
            # Apply volatility regime analysis
            volatility_signal = await self._apply_volatility_regime_analysis(symbol, momentum_signal, market_data)
            
            # Apply correlation analysis
            correlation_signal = await self._apply_correlation_analysis(symbol, volatility_signal, market_data)
            
            # Final signal optimization
            final_signal = await self._optimize_final_signal(symbol, correlation_signal, market_data)
            
            # Validate signal quantitatively
            validation_results = await self.quantitative_integration.validate_signal_quantitatively(
                symbol, final_signal, market_data
            )
            
            # Add validation results to signal
            final_signal['quantitative_validation'] = validation_results
            
            # Optimize position size with advanced methods
            optimized_size = await self._optimize_position_size_advanced(
                symbol, final_signal.get('position_size', 0.01), market_data, final_signal
            )
            final_signal['optimized_position_size'] = optimized_size
            
            # Store signal history
            self._store_signal_history(symbol, final_signal)
            
            # Log quantitative analysis
            await self._log_quantitative_analysis(symbol, final_signal, validation_results)
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {str(e)}")
            return None
    
    async def _generate_advanced_signal(self, symbol: str, indicator_service: IndicatorService, market_data: Dict) -> Optional[Dict]:
        """Generate advanced signal using WorldQuant-level analysis."""
        try:
            # Get multi-timeframe data
            df_1h = await indicator_service.get_historical_data(symbol, '1h', limit=500)
            df_4h = await indicator_service.get_historical_data(symbol, '4h', limit=200)
            df_1d = await indicator_service.get_historical_data(symbol, '1d', limit=100)
            
            if df_1h is None or df_1h.empty:
                return None
            
            # Calculate advanced indicators
            df_1h = await self._calculate_advanced_indicators(df_1h)
            df_4h = await self._calculate_advanced_indicators(df_4h)
            df_1d = await self._calculate_advanced_indicators(df_1d)
            
            # Multi-timeframe analysis
            signal = self._create_advanced_signal(symbol, df_1h, df_4h, df_1d, market_data)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating advanced signal for {symbol}: {str(e)}")
            return None
    
    async def _calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators."""
        try:
            # Basic indicators
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
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
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Advanced indicators
            # Stochastic RSI
            df['stoch_rsi'] = (df['rsi'] - df['rsi'].rolling(14).min()) / (df['rsi'].rolling(14).max() - df['rsi'].rolling(14).min())
            
            # Williams %R
            df['williams_r'] = (df['high'].rolling(14).max() - df['close']) / (df['high'].rolling(14).max() - df['low'].rolling(14).min()) * -100
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(14).mean()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price momentum
            df['momentum'] = df['close'] / df['close'].shift(10) - 1
            df['rate_of_change'] = df['close'].pct_change(10)
            
            # Volatility indicators
            df['volatility'] = df['close'].pct_change().rolling(20).std()
            df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating advanced indicators: {str(e)}")
            return df
    
    def _create_advanced_signal(self, symbol: str, df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame, market_data: Dict) -> Dict:
        """Create advanced trading signal using multi-timeframe analysis."""
        try:
            signal = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'signal_strength': 0.0,
                'position_size': 0.01,
                'action': 'hold',
                'confidence': 0.0,
                'timeframes': {}
            }
            
            # 1-hour timeframe analysis
            h1_signal = self._analyze_timeframe(df_1h, '1h')
            signal['timeframes']['1h'] = h1_signal
            
            # 4-hour timeframe analysis
            h4_signal = self._analyze_timeframe(df_4h, '4h')
            signal['timeframes']['4h'] = h4_signal
            
            # 1-day timeframe analysis
            d1_signal = self._analyze_timeframe(df_1d, '1d')
            signal['timeframes']['1d'] = d1_signal
            
            # Combine signals using weighted approach
            combined_signal = self._combine_timeframe_signals(signal['timeframes'])
            
            signal.update(combined_signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error creating advanced signal for {symbol}: {str(e)}")
            return {'symbol': symbol, 'action': 'hold', 'signal_strength': 0.0}
    
    def _analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Analyze single timeframe for signals."""
        try:
            if df.empty or len(df) < 50:
                return {'signal': 'hold', 'strength': 0.0, 'confidence': 0.0}
            
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
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
                'reasons': signal_reasons
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
                # Asymmetric confidence calculation for BUY (lower base, higher potential)
                base_confidence = weighted_confidence * 0.8  # Lower base for BUY
                strength_boost = min(0.15, (combined_strength - buy_threshold) / (0.4 - buy_threshold) * 0.2)
                confidence = min(0.95, base_confidence + strength_boost)
            elif combined_strength < sell_threshold:
                action = 'sell'
                # Asymmetric confidence calculation for SELL (higher base, stricter requirements)
                base_confidence = weighted_confidence * 1.2  # Higher base for SELL
                strength_boost = min(0.10, (sell_threshold - combined_strength) / (sell_threshold + 0.4) * 0.15)
                confidence = min(0.95, base_confidence + strength_boost)
            else:
                action = 'hold'
                confidence = weighted_confidence  # Use the weighted confidence for hold actions
            
            # Log threshold information for monitoring
            logger.info(f"Dynamic thresholds - Buy: {buy_threshold:.3f}, Sell: {sell_threshold:.3f}, "
                       f"Combined strength: {combined_strength:.3f}, Weighted confidence: {weighted_confidence:.3f}, "
                       f"Final confidence: {confidence:.3f}, Action: {action}, "
                       f"Market regime: {thresholds['market_regime']}")
            
            return {
                'action': action,
                'signal_strength': combined_strength,
                'confidence': confidence,
                'reasons': all_reasons
            }
            
        except Exception as e:
            logger.error(f"Error combining timeframe signals: {str(e)}")
            return {'action': 'hold', 'signal_strength': 0.0, 'confidence': 0.0}
    
    async def _get_comprehensive_market_data(self, symbol: str) -> Dict:
        """Get comprehensive market data for quantitative analysis."""
        try:
            market_data = {}
            
            # Get historical data
            df = await self.indicator_service.get_historical_data(symbol, '1h', limit=252)
            if df is not None and not df.empty:
                # Calculate returns
                df['returns'] = df['close'].pct_change()
                market_data['returns'] = df['returns'].dropna().values
                
                # Get current price
                market_data['current_price'] = df['close'].iloc[-1]
            
            # Get orderbook data
            try:
                orderbook = await self.binance_service.get_order_book(symbol)
                market_data['orderbook'] = orderbook
            except Exception as e:
                logger.warning(f"Could not get orderbook for {symbol}: {str(e)}")
            
            # Get recent trades
            try:
                if hasattr(self.binance_service, 'get_recent_trades'):
                    trades = await self.binance_service.get_recent_trades(symbol)
                else:
                    trades = await self.binance_service.get_trades(symbol)
                market_data['trades'] = pd.DataFrame(trades)
            except Exception as e:
                logger.warning(f"Could not get trades for {symbol}: {str(e)}")
            
            # Get portfolio data (if available)
            try:
                positions = await self.binance_service.get_positions()
                portfolio_returns = self._calculate_portfolio_returns(positions)
                market_data['portfolio_returns'] = portfolio_returns
            except Exception as e:
                logger.warning(f"Could not get portfolio data: {str(e)}")
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting comprehensive market data for {symbol}: {str(e)}")
            return {}
    
    def _create_base_signal(self, symbol: str, df: pd.DataFrame, conditions: Dict) -> Dict:
        """Create base trading signal."""
        try:
            signal = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'signal_strength': 0.5,
                'position_size': 0.01,
                'action': 'hold'
            }
            
            # Determine signal based on conditions
            if conditions.get('trend') == 'up' and conditions.get('macd_bullish', False):
                signal['action'] = 'buy'
                signal['signal_strength'] = 0.7
            elif conditions.get('trend') == 'down' and conditions.get('macd_bearish', False):
                signal['action'] = 'sell'
                signal['signal_strength'] = 0.7
            
            # Adjust signal strength based on conditions
            if conditions.get('rsi_oversold', False) and signal['action'] == 'buy':
                signal['signal_strength'] += 0.1
            elif conditions.get('rsi_overbought', False) and signal['action'] == 'sell':
                signal['signal_strength'] += 0.1
            
            if conditions.get('volume_trend', False):
                signal['signal_strength'] += 0.1
            
            # Cap signal strength
            signal['signal_strength'] = min(signal['signal_strength'], 1.0)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error creating base signal for {symbol}: {str(e)}")
            return {'symbol': symbol, 'action': 'hold', 'signal_strength': 0.0}
    
    def _calculate_portfolio_returns(self, positions: List[Dict]) -> Dict:
        """Calculate portfolio returns from positions."""
        try:
            portfolio_returns = {}
            
            for position in positions:
                symbol = position.get('symbol', '')
                if symbol:
                    # Calculate position return (simplified)
                    unrealized_pnl = position.get('unrealizedPnl', 0)
                    position_value = position.get('positionAmt', 0)
                    
                    if position_value != 0:
                        return_rate = unrealized_pnl / abs(position_value)
                        portfolio_returns[symbol] = return_rate
            
            return portfolio_returns
            
        except Exception as e:
            logger.error(f"Error calculating portfolio returns: {str(e)}")
            return {}
    
    def _get_trend(self, df: pd.DataFrame) -> str:
        """Determine market trend."""
        try:
            if len(df) < 20:
                return 'neutral'
            
            # Simple trend calculation
            short_ma = df['close'].rolling(10).mean().iloc[-1]
            long_ma = df['close'].rolling(20).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            if current_price > short_ma > long_ma:
                return 'up'
            elif current_price < short_ma < long_ma:
                return 'down'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error determining trend: {str(e)}")
            return 'neutral'
    
    def _store_signal_history(self, symbol: str, signal: Dict) -> None:
        """Store signal history for analysis."""
        try:
            if symbol not in self.signal_history:
                self.signal_history[symbol] = []
            
            self.signal_history[symbol].append({
                'timestamp': datetime.now(),
                'signal': signal
            })
            
            # Keep only last 100 signals
            if len(self.signal_history[symbol]) > 100:
                self.signal_history[symbol] = self.signal_history[symbol][-100:]
                
        except Exception as e:
            logger.error(f"Error storing signal history: {str(e)}")
    
    async def _log_quantitative_analysis(self, symbol: str, signal: Dict, validation: Dict) -> None:
        """Log quantitative analysis results."""
        try:
            analysis_log = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'signal': signal,
                'validation': validation,
                'quantitative_confidence': signal.get('quantitative_confidence', 0.0),
                'statistical_valid': validation.get('is_valid', False),
                'sharpe_ratio': validation.get('sharpe_ratio', 0.0),
                'var_estimate': signal.get('var_estimate', 0.0)
            }
            
            # Store in history
            if symbol not in self.quantitative_analysis_history:
                self.quantitative_analysis_history[symbol] = []
            
            self.quantitative_analysis_history[symbol].append(analysis_log)
            
            # Keep only last 50 analyses
            if len(self.quantitative_analysis_history[symbol]) > 50:
                self.quantitative_analysis_history[symbol] = self.quantitative_analysis_history[symbol][-50:]
            
            # Log to notification service
            # if self.notification_service:
            #     message = f"Quantitative Analysis for {symbol}:\n"
            #     message += f"Confidence: {signal.get('quantitative_confidence', 0.0):.2f}\n"
            #     message += f"Valid: {validation.get('is_valid', False)}\n"
            #     message += f"Sharpe: {validation.get('sharpe_ratio', 0.0):.2f}\n"
            #     message += f"VaR: {signal.get('var_estimate', 0.0):.4f}"
                
            #     await self.notification_service.send_message(message)
                
        except Exception as e:
            logger.error(f"Error logging quantitative analysis: {str(e)}")
    
    async def get_quantitative_recommendations(self, symbol: str) -> Dict:
        """Get quantitative recommendations for a symbol."""
        try:
            # Get current market data
            market_data = await self._get_comprehensive_market_data(symbol)
            
            # Get current positions
            positions = await self.binance_service.get_positions()
            
            # Get recommendations
            recommendations = await self.quantitative_integration.get_quantitative_recommendations(
                symbol, market_data, positions
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting quantitative recommendations for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    async def analyze_portfolio_optimization(self, symbols: List[str]) -> Dict:
        """Analyze portfolio optimization."""
        try:
            # Get returns data for all symbols
            returns_data = {}
            
            for symbol in symbols:
                df = await self.indicator_service.get_historical_data(symbol, '1h', limit=252)
                if df is not None and not df.empty:
                    returns = df['close'].pct_change().dropna()
                    returns_data[symbol] = returns.values
            
            if returns_data:
                optimization_results = await self.quantitative_integration.get_portfolio_optimization(
                    symbols, returns_data
                )
                return optimization_results
            else:
                return {'error': 'No returns data available'}
                
        except Exception as e:
            logger.error(f"Error analyzing portfolio optimization: {str(e)}")
            return {'error': str(e)}
    
    async def analyze_factor_exposures(self, symbols: List[str]) -> Dict:
        """Analyze factor exposures for portfolio."""
        try:
            # Get returns data for all symbols
            returns_data = {}
            
            for symbol in symbols:
                df = await self.indicator_service.get_historical_data(symbol, '1h', limit=252)
                if df is not None and not df.empty:
                    returns = df['close'].pct_change().dropna()
                    returns_data[symbol] = returns.values
            
            if returns_data:
                factor_results = await self.quantitative_integration.analyze_factor_exposures(
                    symbols, returns_data
                )
                return factor_results
            else:
                return {'error': 'No returns data available'}
                
        except Exception as e:
            logger.error(f"Error analyzing factor exposures: {str(e)}")
            return {'error': str(e)}
    
    async def check_profit_target(self) -> bool:
        """Check if profit target has been reached."""
        try:
            # Get current positions
            positions = await self.binance_service.get_positions()
            
            total_pnl = 0.0
            for position in positions:
                unrealized_pnl = float(position.get('unrealizedPnl', 0))
                total_pnl += unrealized_pnl
            
            # Check if total PnL exceeds profit target
            profit_target = self.config.get('profit_target', 1000000)  # Default $100
            return total_pnl >= profit_target
            
        except Exception as e:
            logger.error(f"Error checking profit target: {str(e)}")
            return False
    
    async def process_trading_signals(self, signals: Dict) -> None:
        """Process trading signals with WorldQuant-level dynamic confidence thresholds."""
        try:
            if not signals:
                return
            
            symbol = signals.get('symbol', '')
            action = signals.get('action', 'hold')
            base_confidence = signals.get('quantitative_confidence', 0.0)
            
            # Get market data and risk metrics for dynamic threshold calculation
            market_data = signals.get('market_data', {})
            risk_metrics = signals.get('risk_metrics', {})
            
            # Calculate risk-adjusted confidence
            adjusted_confidence = self._calculate_risk_adjusted_confidence(signals, risk_metrics)
            
            # Calculate dynamic confidence threshold based on action type
            dynamic_threshold = self._calculate_dynamic_confidence_threshold(action, market_data, risk_metrics)
            
            logger.info(f"Processing signal for {symbol}: {action} with "
                       f"base_confidence={base_confidence:.3f}, "
                       f"adjusted_confidence={adjusted_confidence:.3f}, "
                       f"dynamic_threshold={dynamic_threshold:.3f}")
            
            # Track performance for optimization
            self._track_confidence_performance(action, adjusted_confidence, dynamic_threshold, market_data, risk_metrics)
            
            # Execute only if adjusted confidence meets dynamic threshold
            if adjusted_confidence < dynamic_threshold:
                logger.info(f"Signal confidence too low for {symbol}: {adjusted_confidence:.3f} < {dynamic_threshold:.3f}")
                return
            
            # Execute trade based on signal
            if action == 'buy':
                await self._execute_buy_order(symbol, signals)
            elif action == 'sell':
                await self._execute_sell_order(symbol, signals)
                
        except Exception as e:
            logger.error(f"Error processing trading signals: {str(e)}")
    
    async def _execute_buy_order(self, symbol: str, signals: Dict) -> None:
        """Execute buy order."""
        try:
            position_size = signals.get('optimized_position_size', 0.01)
            current_price = signals.get('current_price', 0.0)
            
            if current_price > 0:
                # Calculate quantity
                account_info = await self.binance_service.get_account_info()
                balance = float(account_info.get('totalWalletBalance', 0))
                quantity = (balance * position_size) / current_price
                
                # Place order
                order = await self.binance_service.place_order(
                    symbol=symbol,
                    side='BUY',
                    order_type='MARKET',
                    amount=quantity
                )
                
                logger.info(f"Buy order placed for {symbol}: {order}")
                
        except Exception as e:
            logger.error(f"Error executing buy order for {symbol}: {str(e)}")
    
    async def _execute_sell_order(self, symbol: str, signals: Dict) -> None:
        """Execute sell order."""
        try:
            # Get current position
            positions = await self.binance_service.get_positions()
            position = None
            
            for pos in positions:
                if pos.get('symbol') == symbol:
                    position = pos
                    break
            
            if position and float(position.get('positionAmt', 0)) > 0:
                quantity = abs(float(position.get('positionAmt', 0)))
                
                # Place order
                order = await self.binance_service.place_order(
                    symbol=symbol,
                    side='SELL',
                    order_type='MARKET',
                    amount=quantity
                )
                
                logger.info(f"Sell order placed for {symbol}: {order}")
                
        except Exception as e:
            logger.error(f"Error executing sell order for {symbol}: {str(e)}")
    
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
            else:
                metrics['confidence_analytics'] = {
                    'status': 'not_initialized',
                    'message': 'Confidence performance tracking not yet started'
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {'error': str(e)}
    
    async def close(self):
        """Clean up resources."""
        try:
            # Clear caches
            self.data_cache.clear()
            self.signal_history.clear()
            self.quantitative_analysis_history.clear()
            
            logger.info("Enhanced Trading Strategy with Quantitative Integration closed")
            
        except Exception as e:
            logger.error(f"Error closing strategy: {str(e)}") 
    
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
                    enhanced_signal['signal_strength'] += 0.1
                    enhanced_signal['reasons'].append('tight_spread')
                
                if order_imbalance > 0.2:  # Strong buy pressure
                    enhanced_signal['signal_strength'] += 0.15
                    enhanced_signal['reasons'].append('buy_imbalance')
                elif order_imbalance < -0.2:  # Strong sell pressure
                    enhanced_signal['signal_strength'] -= 0.15
                    enhanced_signal['reasons'].append('sell_imbalance')
            
            # Volume profile analysis
            if 'trades' in market_data and isinstance(market_data['trades'], pd.DataFrame):
                volume_profile = self._analyze_volume_profile(market_data['trades'])
                enhanced_signal['volume_profile'] = volume_profile
                
                # Adjust based on volume profile
                if volume_profile.get('high_volume_nodes', []):
                    enhanced_signal['signal_strength'] += 0.05
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
                stat_arb_signal['signal_strength'] += 0.1
                stat_arb_signal['reasons'].append('statistical_arbitrage')
            
            # Mean reversion analysis
            mean_reversion = self._analyze_mean_reversion(market_data.get('returns', []))
            stat_arb_signal['mean_reversion'] = mean_reversion
            
            if mean_reversion.get('is_mean_reverting'):
                if mean_reversion.get('deviation') > 2:  # Strong deviation
                    stat_arb_signal['signal_strength'] += 0.15
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
            logger.error(f"Error analyzing mean reversion: {str(e)}")
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
                    momentum_signal['signal_strength'] += 0.1
                    momentum_signal['reasons'].append('positive_momentum')
                elif short_momentum < -0.01 and medium_momentum < -0.005:
                    momentum_signal['signal_strength'] -= 0.1
                    momentum_signal['reasons'].append('negative_momentum')
                
                # Mean reversion signal
                if abs(short_momentum) > 0.02 and abs(short_momentum - medium_momentum) > 0.01:
                    if short_momentum > medium_momentum:
                        momentum_signal['signal_strength'] -= 0.05  # Revert from high
                        momentum_signal['reasons'].append('momentum_reversion')
                    else:
                        momentum_signal['signal_strength'] += 0.05  # Revert from low
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
            
            # Signal strength normalization
            optimized_signal['signal_strength'] = np.clip(optimized_signal['signal_strength'], -1.0, 1.0)
            
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
                win_rate = float(np.sum(returns > 0)) / len(returns)
                avg_win = float(np.mean(returns[returns > 0])) if float(np.sum(returns > 0)) > 0 else 0.001
                avg_loss = abs(float(np.mean(returns[returns < 0]))) if float(np.sum(returns < 0)) > 0 else 0.001
                
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
            logger.error(f"Error optimizing position size: {str(e)}")
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
                hurst = np.log(price_range) / np.log(time_range)
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
            sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if float(np.std(returns)) > 0 else 0
            var_95 = float(np.percentile(returns, 5))
            max_drawdown = float(self._calculate_max_drawdown(returns))
            volatility = float(np.std(returns) * np.sqrt(252))
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            sortino_ratio = float(np.mean(returns) / np.std(downside_returns) * np.sqrt(252)) if len(downside_returns) > 0 and float(np.std(downside_returns)) > 0 else 0
            
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
            volatility = market_data.get('volatility', 0.02)
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
            signal_strength = signal.get('signal_strength', 0.0)
            
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
    
    def _track_confidence_performance(self, action: str, confidence: float, threshold: float, 
                                    market_data: Dict, risk_metrics: Dict) -> None:
        """
        Track confidence threshold performance for optimization.
        WorldQuant-level performance analytics.
        """
        try:
            if not hasattr(self, 'confidence_performance'):
                self.confidence_performance = {
                    'buy': {'executions': 0, 'successes': 0, 'total_pnl': 0.0},
                    'sell': {'executions': 0, 'successes': 0, 'total_pnl': 0.0},
                    'thresholds': {
                        'buy': {'avg_threshold': 0.0, 'count': 0},
                        'sell': {'avg_threshold': 0.0, 'count': 0}
                    }
                }
            
            # Track threshold usage
            if action in ['buy', 'sell']:
                self.confidence_performance['thresholds'][action]['avg_threshold'] = (
                    (self.confidence_performance['thresholds'][action]['avg_threshold'] * 
                     self.confidence_performance['thresholds'][action]['count'] + threshold) /
                    (self.confidence_performance['thresholds'][action]['count'] + 1)
                )
                self.confidence_performance['thresholds'][action]['count'] += 1
            
            # Log performance metrics
            logger.info(f"Confidence performance tracking - Action: {action}, "
                       f"Confidence: {confidence:.3f}, Threshold: {threshold:.3f}, "
                       f"Market regime: {market_data.get('market_regime', 'unknown')}, "
                       f"Volatility: {market_data.get('volatility', 0.0):.4f}")
            
        except Exception as e:
            logger.error(f"Error tracking confidence performance: {str(e)}") 