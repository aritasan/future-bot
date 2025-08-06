#!/usr/bin/env python3
"""
Statistical Arbitrage Module
WorldQuant Standards Implementation

Implements:
- Pairs Trading with cointegration analysis
- Mean Reversion strategies
- Momentum Reversal signals
- Volatility Arbitrage
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class StatisticalArbitrageEngine:
    """
    Advanced Statistical Arbitrage System following WorldQuant standards.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize Statistical Arbitrage Engine."""
        self.config = config or {}
        self.pairs_data = {}
        self.cointegration_results = {}
        self.mean_reversion_signals = {}
        self.momentum_reversal_signals = {}
        self.volatility_arbitrage_signals = {}
        
        # Arbitrage parameters
        self.z_score_threshold = self.config.get('z_score_threshold', 2.0)
        self.cointegration_p_value = self.config.get('cointegration_p_value', 0.05)
        self.lookback_period = self.config.get('lookback_period', 252)
        self.volatility_window = self.config.get('volatility_window', 30)
        
        logger.info("Statistical Arbitrage Engine initialized")
    
    def find_cointegrated_pairs(self, market_data: Dict[str, pd.Series], 
                               min_correlation: float = 0.7) -> List[Tuple[str, str]]:
        """
        Find cointegrated pairs using Engle-Granger test.
        
        Args:
            market_data: Dictionary of price series
            min_correlation: Minimum correlation threshold
            
        Returns:
            List of cointegrated pairs
        """
        try:
            cointegrated_pairs = []
            symbols = list(market_data.keys())
            
            logger.info(f"Searching for cointegrated pairs among {len(symbols)} symbols...")
            
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    symbol1, symbol2 = symbols[i], symbols[j]
                    
                    # Get price series
                    price1 = market_data[symbol1]
                    price2 = market_data[symbol2]
                    
                    # Check correlation first
                    correlation = price1.corr(price2)
                    if correlation < min_correlation:
                        continue
                    
                    # Test for cointegration
                    is_cointegrated = self._test_cointegration(price1, price2)
                    
                    if is_cointegrated:
                        cointegrated_pairs.append((symbol1, symbol2))
                        logger.info(f"Found cointegrated pair: {symbol1} - {symbol2} (correlation: {correlation:.3f})")
                        
                        # Store cointegration results
                        self.cointegration_results[f"{symbol1}_{symbol2}"] = {
                            'correlation': correlation,
                            'cointegration_test': True,
                            'spread_series': self._calculate_spread(price1, price2)
                        }
            
            logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs")
            return cointegrated_pairs
            
        except Exception as e:
            logger.error(f"Error finding cointegrated pairs: {str(e)}")
            return []
    
    def _test_cointegration(self, price1: pd.Series, price2: pd.Series) -> bool:
        """
        Test for cointegration using Engle-Granger test.
        
        Args:
            price1: First price series
            price2: Second price series
            
        Returns:
            True if cointegrated, False otherwise
        """
        try:
            # Calculate spread
            spread = self._calculate_spread(price1, price2)
            
            # Test for stationarity using ADF test
            from statsmodels.tsa.stattools import adfuller
            
            adf_result = adfuller(spread.dropna())
            p_value = adf_result[1]
            
            # If p-value < 0.05, spread is stationary (cointegrated)
            return p_value < self.cointegration_p_value
            
        except Exception as e:
            logger.error(f"Error testing cointegration: {str(e)}")
            return False
    
    def _calculate_spread(self, price1: pd.Series, price2: pd.Series) -> pd.Series:
        """
        Calculate spread between two price series.
        
        Args:
            price1: First price series
            price2: Second price series
            
        Returns:
            Spread series
        """
        try:
            # Log prices
            log_price1 = np.log(price1)
            log_price2 = np.log(price2)
            
            # Calculate hedge ratio using OLS
            hedge_ratio = self._calculate_hedge_ratio(log_price1, log_price2)
            
            # Calculate spread
            spread = log_price1 - hedge_ratio * log_price2
            
            return spread
            
        except Exception as e:
            logger.error(f"Error calculating spread: {str(e)}")
            return pd.Series([0.0])
    
    def _calculate_hedge_ratio(self, price1: pd.Series, price2: pd.Series) -> float:
        """
        Calculate hedge ratio using OLS regression.
        
        Args:
            price1: First price series
            price2: Second price series
            
        Returns:
            Hedge ratio
        """
        try:
            # Remove NaN values
            valid_data = pd.DataFrame({'price1': price1, 'price2': price2}).dropna()
            
            if len(valid_data) < 30:  # Need minimum data points
                return 1.0
            
            # OLS regression: price1 = alpha + beta * price2
            # Use numpy polyfit instead of sklearn LinearRegression
            X = valid_data['price2'].values
            y = valid_data['price1'].values
            
            # Fit linear regression using numpy
            coeffs = np.polyfit(X, y, 1)
            beta = coeffs[0]  # slope coefficient
            
            return float(beta)
            
        except Exception as e:
            logger.error(f"Error calculating hedge ratio: {str(e)}")
            return 1.0
    
    def generate_pairs_trading_signals(self, market_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Generate pairs trading signals based on cointegration.
        
        Args:
            market_data: Dictionary of price series
            
        Returns:
            Dictionary with pairs trading signals
        """
        try:
            # Find cointegrated pairs
            cointegrated_pairs = self.find_cointegrated_pairs(market_data)
            
            pairs_signals = {}
            
            for symbol1, symbol2 in cointegrated_pairs:
                pair_key = f"{symbol1}_{symbol2}"
                
                if pair_key in self.cointegration_results:
                    spread_series = self.cointegration_results[pair_key]['spread_series']
                    
                    # Calculate z-score of spread
                    z_score = self._calculate_z_score(spread_series)
                    
                    # Generate trading signals
                    signal = self._generate_pairs_signal(z_score, symbol1, symbol2)
                    
                    pairs_signals[pair_key] = {
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'z_score': z_score,
                        'signal': signal,
                        'correlation': self.cointegration_results[pair_key]['correlation'],
                        'spread_mean': float(spread_series.mean()),
                        'spread_std': float(spread_series.std())
                    }
            
            return pairs_signals
            
        except Exception as e:
            logger.error(f"Error generating pairs trading signals: {str(e)}")
            return {}
    
    def _calculate_z_score(self, series: pd.Series, window: int = None) -> float:
        """
        Calculate z-score of a series.
        
        Args:
            series: Time series
            window: Rolling window (if None, use full series)
            
        Returns:
            Z-score
        """
        try:
            if len(series) == 0:
                return 0.0
                
            if window is None:
                mean = series.mean()
                std = series.std()
            else:
                if len(series) < window:
                    return 0.0
                mean = series.rolling(window=window).mean().iloc[-1]
                std = series.rolling(window=window).std().iloc[-1]
            
            if std == 0 or pd.isna(std):
                return 0.0
            
            return float((series.iloc[-1] - mean) / std)
            
        except Exception as e:
            logger.error(f"Error calculating z-score: {str(e)}")
            return 0.0
    
    def _generate_pairs_signal(self, z_score: float, symbol1: str, symbol2: str) -> Dict[str, Any]:
        """
        Generate pairs trading signal based on z-score.
        
        Args:
            z_score: Z-score of spread
            symbol1: First symbol
            symbol2: Second symbol
            
        Returns:
            Trading signal
        """
        try:
            signal = {
                'action': 'hold',
                'symbol1_action': 'hold',
                'symbol2_action': 'hold',
                'confidence': 0.0,
                'reasoning': []
            }
            
            # Generate signals based on z-score thresholds
            if z_score > self.z_score_threshold:
                # Spread is high, expect mean reversion
                signal['action'] = 'sell_spread'
                signal['symbol1_action'] = 'sell'
                signal['symbol2_action'] = 'buy'
                signal['confidence'] = min(abs(z_score) / 3.0, 1.0)
                signal['reasoning'].append(f'High z-score ({z_score:.2f}) indicates overvalued spread')
                
            elif z_score < -self.z_score_threshold:
                # Spread is low, expect mean reversion
                signal['action'] = 'buy_spread'
                signal['symbol1_action'] = 'buy'
                signal['symbol2_action'] = 'sell'
                signal['confidence'] = min(abs(z_score) / 3.0, 1.0)
                signal['reasoning'].append(f'Low z-score ({z_score:.2f}) indicates undervalued spread')
                
            else:
                # Spread is within normal range
                signal['reasoning'].append(f'Z-score ({z_score:.2f}) within normal range')
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating pairs signal: {str(e)}")
            return {'action': 'hold', 'confidence': 0.0, 'reasoning': ['Error in signal generation']}
    
    def generate_mean_reversion_signals(self, market_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Generate mean reversion signals for individual assets.
        
        Args:
            market_data: Dictionary of price series
            
        Returns:
            Dictionary with mean reversion signals
        """
        try:
            mean_reversion_signals = {}
            
            for symbol, price_series in market_data.items():
                # Calculate returns
                returns = price_series.pct_change().dropna()
                
                # Calculate various mean reversion indicators
                indicators = self._calculate_mean_reversion_indicators(returns)
                
                # Generate signal
                signal = self._generate_mean_reversion_signal(indicators, symbol)
                
                mean_reversion_signals[symbol] = {
                    'symbol': symbol,
                    'indicators': indicators,
                    'signal': signal
                }
            
            return mean_reversion_signals
            
        except Exception as e:
            logger.error(f"Error generating mean reversion signals: {str(e)}")
            return {}
    
    def _calculate_mean_reversion_indicators(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate mean reversion indicators.
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary of mean reversion indicators
        """
        try:
            indicators = {}
            
            # 1. Hurst Exponent
            indicators['hurst_exponent'] = self._calculate_hurst_exponent(returns)
            
            # 2. Augmented Dickey-Fuller test
            indicators['adf_p_value'] = self._calculate_adf_test(returns)
            
            # 3. Variance ratio test
            indicators['variance_ratio'] = self._calculate_variance_ratio(returns)
            
            # 4. Z-score of returns
            indicators['returns_z_score'] = self._calculate_z_score(returns)
            
            # 5. Mean reversion strength
            indicators['mean_reversion_strength'] = self._calculate_mean_reversion_strength(returns)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating mean reversion indicators: {str(e)}")
            return {}
    
    def _calculate_hurst_exponent(self, returns: pd.Series) -> float:
        """
        Calculate Hurst exponent to test for mean reversion.
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        try:
            # Simplified Hurst exponent calculation
            lags = range(2, min(20, len(returns) // 4))
            tau = [np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag]))) for lag in lags]
            
            if len(tau) < 2:
                return 0.5
            
            # Linear fit to double-log graph
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            return float(reg[0])
            
        except Exception as e:
            logger.error(f"Error calculating Hurst exponent: {str(e)}")
            return 0.5
    
    def _calculate_adf_test(self, returns: pd.Series) -> float:
        """
        Calculate Augmented Dickey-Fuller test p-value.
        """
        try:
            if len(returns) < 10:
                return 1.0
                
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(returns.dropna())
            return float(adf_result[1])
        except Exception as e:
            logger.error(f"Error calculating ADF test: {str(e)}")
            return 1.0
    
    def _calculate_variance_ratio(self, returns: pd.Series) -> float:
        """
        Calculate variance ratio test statistic.
        """
        try:
            if len(returns) < 20:
                return 1.0
            
            # Calculate variance ratio for different lags
            lag = 5
            var_ratio = returns.rolling(lag).var().iloc[-1] / (lag * returns.var())
            return float(var_ratio)
            
        except Exception as e:
            logger.error(f"Error calculating variance ratio: {str(e)}")
            return 1.0
    
    def _calculate_mean_reversion_strength(self, returns: pd.Series) -> float:
        """
        Calculate mean reversion strength based on multiple indicators.
        """
        try:
            # Combine multiple indicators
            hurst = self._calculate_hurst_exponent(returns)
            adf_p = self._calculate_adf_test(returns)
            var_ratio = self._calculate_variance_ratio(returns)
            
            # Calculate strength (0 = no mean reversion, 1 = strong mean reversion)
            strength = 0.0
            
            # Hurst exponent contribution
            if hurst < 0.5:
                strength += (0.5 - hurst) * 2  # Normalize to [0, 1]
            
            # ADF test contribution
            if adf_p < 0.05:
                strength += 0.3
            
            # Variance ratio contribution
            if var_ratio < 1.0:
                strength += (1.0 - var_ratio) * 0.2
            
            return min(strength, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating mean reversion strength: {str(e)}")
            return 0.0
    
    def _generate_mean_reversion_signal(self, indicators: Dict[str, float], symbol: str) -> Dict[str, Any]:
        """
        Generate mean reversion signal based on indicators.
        """
        try:
            signal = {
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': []
            }
            
            hurst = indicators.get('hurst_exponent', 0.5)
            adf_p = indicators.get('adf_p_value', 1.0)
            var_ratio = indicators.get('variance_ratio', 1.0)
            z_score = indicators.get('returns_z_score', 0.0)
            strength = indicators.get('mean_reversion_strength', 0.0)
            
            # Determine if mean reverting
            is_mean_reverting = (hurst < 0.5 and adf_p < 0.05 and var_ratio < 1.0)
            
            if is_mean_reverting:
                signal['confidence'] = strength
                
                # Generate signal based on z-score
                if z_score > self.z_score_threshold:
                    signal['action'] = 'sell'
                    signal['reasoning'].append(f'Mean reverting asset with high z-score ({z_score:.2f})')
                elif z_score < -self.z_score_threshold:
                    signal['action'] = 'buy'
                    signal['reasoning'].append(f'Mean reverting asset with low z-score ({z_score:.2f})')
                else:
                    signal['reasoning'].append(f'Mean reverting asset within normal range (z-score: {z_score:.2f})')
            else:
                signal['reasoning'].append('Asset does not exhibit mean reversion characteristics')
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating mean reversion signal: {str(e)}")
            return {'action': 'hold', 'confidence': 0.0, 'reasoning': ['Error in signal generation']}
    
    def generate_momentum_reversal_signals(self, market_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Generate momentum reversal signals.
        
        Args:
            market_data: Dictionary of price series
            
        Returns:
            Dictionary with momentum reversal signals
        """
        try:
            momentum_reversal_signals = {}
            
            for symbol, price_series in market_data.items():
                # Calculate momentum indicators
                indicators = self._calculate_momentum_indicators(price_series)
                
                # Generate signal
                signal = self._generate_momentum_reversal_signal(indicators, symbol)
                
                momentum_reversal_signals[symbol] = {
                    'symbol': symbol,
                    'indicators': indicators,
                    'signal': signal
                }
            
            return momentum_reversal_signals
            
        except Exception as e:
            logger.error(f"Error generating momentum reversal signals: {str(e)}")
            return {}
    
    def _calculate_momentum_indicators(self, price_series: pd.Series) -> Dict[str, float]:
        """
        Calculate momentum indicators.
        """
        try:
            indicators = {}
            
            # 1. RSI
            indicators['rsi'] = self._calculate_rsi(price_series)
            
            # 2. MACD
            indicators['macd'] = self._calculate_macd(price_series)
            
            # 3. Bollinger Bands
            indicators['bb_position'] = self._calculate_bollinger_position(price_series)
            
            # 4. Momentum strength
            indicators['momentum_strength'] = self._calculate_momentum_strength(price_series)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {str(e)}")
            return {}
    
    def _calculate_rsi(self, price_series: pd.Series, period: int = 14) -> float:
        """
        Calculate Relative Strength Index.
        """
        try:
            if len(price_series) < period + 1:
                return 50.0
                
            delta = price_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            if len(rsi) == 0 or pd.isna(rsi.iloc[-1]):
                return 50.0
                
            return float(rsi.iloc[-1])
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return 50.0
    
    def _calculate_macd(self, price_series: pd.Series) -> float:
        """
        Calculate MACD.
        """
        try:
            if len(price_series) < 26:
                return 0.0
                
            ema12 = price_series.ewm(span=12).mean()
            ema26 = price_series.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            
            if len(macd) == 0 or len(signal) == 0:
                return 0.0
                
            return float(macd.iloc[-1] - signal.iloc[-1])
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return 0.0
    
    def _calculate_bollinger_position(self, price_series: pd.Series, period: int = 20) -> float:
        """
        Calculate position within Bollinger Bands.
        """
        try:
            if len(price_series) < period:
                return 0.5
                
            sma = price_series.rolling(window=period).mean()
            std = price_series.rolling(window=period).std()
            
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            
            if len(sma) == 0 or len(std) == 0:
                return 0.5
                
            current_price = price_series.iloc[-1]
            bb_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
            
            return float(bb_position)
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger position: {str(e)}")
            return 0.5
    
    def _calculate_momentum_strength(self, price_series: pd.Series) -> float:
        """
        Calculate momentum strength.
        """
        try:
            if len(price_series) < 20:
                return 0.0
                
            # Calculate momentum over different periods
            momentum_5 = price_series.pct_change(5).iloc[-1] if len(price_series) >= 5 else 0.0
            momentum_10 = price_series.pct_change(10).iloc[-1] if len(price_series) >= 10 else 0.0
            momentum_20 = price_series.pct_change(20).iloc[-1] if len(price_series) >= 20 else 0.0
            
            # Weighted average
            strength = (0.5 * momentum_5 + 0.3 * momentum_10 + 0.2 * momentum_20)
            
            return float(strength)
            
        except Exception as e:
            logger.error(f"Error calculating momentum strength: {str(e)}")
            return 0.0
    
    def _generate_momentum_reversal_signal(self, indicators: Dict[str, float], symbol: str) -> Dict[str, Any]:
        """
        Generate momentum reversal signal.
        """
        try:
            signal = {
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': []
            }
            
            rsi = indicators.get('rsi', 50.0)
            macd = indicators.get('macd', 0.0)
            bb_position = indicators.get('bb_position', 0.5)
            momentum_strength = indicators.get('momentum_strength', 0.0)
            
            # Overbought conditions (potential reversal down)
            if rsi > 70 and bb_position > 0.8 and momentum_strength > 0.05:
                signal['action'] = 'sell'
                signal['confidence'] = min((rsi - 70) / 30, 1.0)
                signal['reasoning'].append(f'Overbought conditions: RSI={rsi:.1f}, BB={bb_position:.2f}')
                
            # Oversold conditions (potential reversal up)
            elif rsi < 30 and bb_position < 0.2 and momentum_strength < -0.05:
                signal['action'] = 'buy'
                signal['confidence'] = min((30 - rsi) / 30, 1.0)
                signal['reasoning'].append(f'Oversold conditions: RSI={rsi:.1f}, BB={bb_position:.2f}')
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating momentum reversal signal: {str(e)}")
            return {'action': 'hold', 'confidence': 0.0, 'reasoning': ['Error in signal generation']}
    
    def generate_volatility_arbitrage_signals(self, market_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Generate volatility arbitrage signals.
        
        Args:
            market_data: Dictionary of price series
            
        Returns:
            Dictionary with volatility arbitrage signals
        """
        try:
            volatility_signals = {}
            
            for symbol, price_series in market_data.items():
                # Calculate volatility indicators
                indicators = self._calculate_volatility_indicators(price_series)
                
                # Generate signal
                signal = self._generate_volatility_arbitrage_signal(indicators, symbol)
                
                volatility_signals[symbol] = {
                    'symbol': symbol,
                    'indicators': indicators,
                    'signal': signal
                }
            
            return volatility_signals
            
        except Exception as e:
            logger.error(f"Error generating volatility arbitrage signals: {str(e)}")
            return {}
    
    def _calculate_volatility_indicators(self, price_series: pd.Series) -> Dict[str, float]:
        """
        Calculate volatility indicators.
        """
        try:
            if len(price_series) < self.volatility_window:
                return {}
                
            indicators = {}
            
            # Calculate returns
            returns = price_series.pct_change().dropna()
            
            if len(returns) < self.volatility_window:
                return {}
            
            # 1. Historical volatility
            vol_rolling = returns.rolling(window=self.volatility_window).std()
            if len(vol_rolling) == 0 or pd.isna(vol_rolling.iloc[-1]):
                indicators['historical_volatility'] = 0.0
            else:
                indicators['historical_volatility'] = float(vol_rolling.iloc[-1])
            
            # 2. Volatility of volatility
            vol_series = returns.rolling(window=10).std()
            if len(vol_series) >= 20:
                vol_of_vol_rolling = vol_series.rolling(window=20).std()
                if len(vol_of_vol_rolling) > 0 and not pd.isna(vol_of_vol_rolling.iloc[-1]):
                    indicators['vol_of_vol'] = float(vol_of_vol_rolling.iloc[-1])
                else:
                    indicators['vol_of_vol'] = 0.0
            else:
                indicators['vol_of_vol'] = 0.0
            
            # 3. Volatility skewness
            if len(vol_series) > 0:
                indicators['vol_skewness'] = float(vol_series.skew())
            else:
                indicators['vol_skewness'] = 0.0
            
            # 4. Volatility regime
            current_vol = indicators.get('historical_volatility', 0.0)
            avg_vol = vol_series.mean() if len(vol_series) > 0 else 0.0
            indicators['vol_regime'] = 'high' if current_vol > avg_vol * 1.5 else 'low' if current_vol < avg_vol * 0.7 else 'normal'
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {str(e)}")
            return {}
    
    def _generate_volatility_arbitrage_signal(self, indicators: Dict[str, float], symbol: str) -> Dict[str, Any]:
        """
        Generate volatility arbitrage signal.
        """
        try:
            signal = {
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': []
            }
            
            current_vol = indicators.get('historical_volatility', 0.0)
            vol_of_vol = indicators.get('vol_of_vol', 0.0)
            vol_skewness = indicators.get('vol_skewness', 0.0)
            vol_regime = indicators.get('vol_regime', 'normal')
            
            # Volatility mean reversion strategy
            if vol_regime == 'high' and vol_skewness > 0:
                signal['action'] = 'sell_volatility'
                signal['confidence'] = min(current_vol / 0.1, 1.0)  # Normalize
                signal['reasoning'].append(f'High volatility regime with positive skewness')
                
            elif vol_regime == 'low' and vol_skewness < 0:
                signal['action'] = 'buy_volatility'
                signal['confidence'] = min(0.05 / current_vol, 1.0)  # Normalize
                signal['reasoning'].append(f'Low volatility regime with negative skewness')
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating volatility arbitrage signal: {str(e)}")
            return {'action': 'hold', 'confidence': 0.0, 'reasoning': ['Error in signal generation']}
    
    def get_arbitrage_summary(self) -> Dict[str, Any]:
        """Get comprehensive arbitrage summary."""
        try:
            summary = {
                'arbitrage_engine_status': 'active',
                'pairs_found': len(self.cointegration_results),
                'mean_reversion_signals': len(self.mean_reversion_signals),
                'momentum_reversal_signals': len(self.momentum_reversal_signals),
                'volatility_arbitrage_signals': len(self.volatility_arbitrage_signals),
                'total_signals_generated': (
                    len(self.cointegration_results) +
                    len(self.mean_reversion_signals) +
                    len(self.momentum_reversal_signals) +
                    len(self.volatility_arbitrage_signals)
                )
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting arbitrage summary: {str(e)}")
            return {'error': str(e)} 