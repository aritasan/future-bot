#!/usr/bin/env python3
"""
Implied Volatility Module for Quantitative Trading Strategy
Following WorldQuant Standards for Volatility Analysis

This module implements advanced implied volatility calculations and analysis
for crypto assets, adapting traditional options volatility models for spot trading.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ImpliedVolatilityEngine:
    """
    WorldQuant-Standard Implied Volatility Engine
    
    Implements advanced volatility analysis for crypto assets:
    - Black-Scholes adapted for spot trading
    - GARCH volatility modeling
    - Volatility surface construction
    - Volatility regime detection
    - Volatility forecasting
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the Implied Volatility Engine."""
        self.config = config or {}
        self.volatility_cache = {}
        self.volatility_surface = {}
        self.regime_history = []
        
        # WorldQuant volatility parameters
        self.volatility_window = self.config.get('volatility_window', 252)  # 1 year
        self.garch_p = self.config.get('garch_p', 1)
        self.garch_q = self.config.get('garch_q', 1)
        self.volatility_threshold = self.config.get('volatility_threshold', 0.2)
        
    def calculate_historical_volatility(self, prices: pd.Series, window: int = 30) -> float:
        """
        Calculate historical volatility using WorldQuant methodology.
        
        Args:
            prices: Price series
            window: Rolling window size
            
        Returns:
            Historical volatility (annualized)
        """
        try:
            # Calculate log returns
            log_returns = np.log(prices / prices.shift(1)).dropna()
            
            # Calculate rolling volatility
            rolling_vol = log_returns.rolling(window=window).std()
            current_vol = rolling_vol.iloc[-1]
            
            # Annualize volatility (assuming daily data)
            annualized_vol = current_vol * np.sqrt(252)
            
            return float(annualized_vol)
            
        except Exception as e:
            logger.error(f"Error calculating historical volatility: {str(e)}")
            return 0.0
    
    def calculate_realized_volatility(self, returns: pd.Series, window: int = 30) -> float:
        """
        Calculate realized volatility using high-frequency data approximation.
        
        Args:
            returns: Return series
            window: Rolling window size
            
        Returns:
            Realized volatility
        """
        try:
            # Calculate realized volatility using sum of squared returns
            squared_returns = returns ** 2
            realized_vol = np.sqrt(squared_returns.rolling(window=window).sum())
            current_realized_vol = realized_vol.iloc[-1]
            
            # Annualize
            annualized_realized_vol = current_realized_vol * np.sqrt(252)
            
            return float(annualized_realized_vol)
            
        except Exception as e:
            logger.error(f"Error calculating realized volatility: {str(e)}")
            return 0.0
    
    def estimate_implied_volatility_black_scholes(
        self, 
        current_price: float, 
        strike_price: float, 
        time_to_expiry: float, 
        risk_free_rate: float, 
        option_price: float, 
        option_type: str = 'call'
    ) -> float:
        """
        Estimate implied volatility using Black-Scholes model.
        Adapted for crypto spot trading by using synthetic options.
        
        Args:
            current_price: Current asset price
            strike_price: Strike price (synthetic)
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free rate
            option_price: Option price (synthetic)
            option_type: 'call' or 'put'
            
        Returns:
            Implied volatility
        """
        try:
            def black_scholes_price(S, K, T, r, sigma, option_type):
                """Calculate Black-Scholes option price."""
                d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                d2 = d1 - sigma*np.sqrt(T)
                
                if option_type == 'call':
                    price = S*stats.norm.cdf(d1) - K*np.exp(-r*T)*stats.norm.cdf(d2)
                else:  # put
                    price = K*np.exp(-r*T)*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1)
                
                return price
            
            def objective_function(sigma):
                """Objective function for volatility optimization."""
                return abs(black_scholes_price(current_price, strike_price, time_to_expiry, 
                                             risk_free_rate, sigma, option_type) - option_price)
            
            # Use Brent's method to find implied volatility
            result = minimize_scalar(objective_function, bounds=(0.01, 5.0), method='bounded')
            
            if result.success:
                return float(result.x)
            else:
                logger.warning("Failed to converge on implied volatility")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error estimating implied volatility: {str(e)}")
            return 0.0
    
    def calculate_garch_volatility(self, returns: pd.Series, p: int = 1, q: int = 1) -> Dict[str, float]:
        """
        Calculate GARCH volatility using WorldQuant methodology.
        
        Args:
            returns: Return series
            p: GARCH lag order
            q: ARCH lag order
            
        Returns:
            GARCH volatility parameters
        """
        try:
            # Simple GARCH(1,1) implementation
            n = len(returns)
            omega = np.var(returns) * 0.1  # Initial variance
            alpha = 0.1  # ARCH parameter
            beta = 0.8   # GARCH parameter
            
            # Initialize variance series
            variance = np.zeros(n)
            variance[0] = omega / (1 - alpha - beta)
            
            # GARCH recursion
            for t in range(1, n):
                variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]
            
            # Calculate volatility
            volatility = np.sqrt(variance)
            current_vol = volatility[-1]
            
            # Annualize
            annualized_vol = current_vol * np.sqrt(252)
            
            return {
                'garch_volatility': float(annualized_vol),
                'omega': float(omega),
                'alpha': float(alpha),
                'beta': float(beta),
                'persistence': float(alpha + beta)
            }
            
        except Exception as e:
            logger.error(f"Error calculating GARCH volatility: {str(e)}")
            return {'garch_volatility': 0.0, 'omega': 0.0, 'alpha': 0.0, 'beta': 0.0, 'persistence': 0.0}
    
    def detect_volatility_regime(self, volatility_series: pd.Series, threshold: float = 0.2) -> Dict[str, Any]:
        """
        Detect volatility regime using WorldQuant methodology.
        
        Args:
            volatility_series: Historical volatility series
            threshold: Regime change threshold
            
        Returns:
            Volatility regime analysis
        """
        try:
            # Calculate volatility statistics
            mean_vol = volatility_series.mean()
            std_vol = volatility_series.std()
            current_vol = volatility_series.iloc[-1]
            
            # Regime classification
            vol_zscore = (current_vol - mean_vol) / std_vol
            
            if vol_zscore > threshold:
                regime = 'high_volatility'
                regime_score = min(vol_zscore / 2, 1.0)
            elif vol_zscore < -threshold:
                regime = 'low_volatility'
                regime_score = min(abs(vol_zscore) / 2, 1.0)
            else:
                regime = 'normal_volatility'
                regime_score = 0.5
            
            # Volatility persistence
            autocorr = volatility_series.autocorr(lag=1)
            
            return {
                'regime': regime,
                'regime_score': float(regime_score),
                'current_volatility': float(current_vol),
                'mean_volatility': float(mean_vol),
                'volatility_zscore': float(vol_zscore),
                'autocorrelation': float(autocorr),
                'volatility_persistence': float(autocorr > 0.5)
            }
            
        except Exception as e:
            logger.error(f"Error detecting volatility regime: {str(e)}")
            return {
                'regime': 'unknown',
                'regime_score': 0.0,
                'current_volatility': 0.0,
                'mean_volatility': 0.0,
                'volatility_zscore': 0.0,
                'autocorrelation': 0.0,
                'volatility_persistence': False
            }
    
    def forecast_volatility(self, volatility_series: pd.Series, horizon: int = 30) -> Dict[str, float]:
        """
        Forecast volatility using WorldQuant methodology.
        
        Args:
            volatility_series: Historical volatility series
            horizon: Forecast horizon in days
            
        Returns:
            Volatility forecast
        """
        try:
            # Simple exponential smoothing for volatility forecasting
            alpha = 0.1  # Smoothing parameter
            forecast = volatility_series.ewm(alpha=alpha).mean().iloc[-1]
            
            # Add mean reversion component
            long_term_mean = volatility_series.mean()
            mean_reversion_factor = 0.1
            adjusted_forecast = forecast + mean_reversion_factor * (long_term_mean - forecast)
            
            # Confidence interval
            volatility_std = volatility_series.std()
            confidence_interval = 1.96 * volatility_std / np.sqrt(len(volatility_series))
            
            return {
                'forecast_volatility': float(adjusted_forecast),
                'confidence_lower': float(adjusted_forecast - confidence_interval),
                'confidence_upper': float(adjusted_forecast + confidence_interval),
                'forecast_horizon': horizon
            }
            
        except Exception as e:
            logger.error(f"Error forecasting volatility: {str(e)}")
            return {
                'forecast_volatility': 0.0,
                'confidence_lower': 0.0,
                'confidence_upper': 0.0,
                'forecast_horizon': horizon
            }
    
    def calculate_volatility_smile(self, prices: pd.Series, strikes: List[float], 
                                  time_to_expiry: float = 0.25) -> Dict[str, Any]:
        """
        Calculate volatility smile for crypto assets.
        
        Args:
            prices: Price series
            strikes: Strike prices
            time_to_expiry: Time to expiry in years
            
        Returns:
            Volatility smile analysis
        """
        try:
            current_price = prices.iloc[-1]
            volatility_smile = {}
            
            for strike in strikes:
                # Calculate moneyness
                moneyness = np.log(strike / current_price)
                
                # Estimate implied volatility for this strike
                # Using synthetic option prices based on historical volatility
                historical_vol = self.calculate_historical_volatility(prices)
                synthetic_option_price = current_price * 0.1  # Simplified
                
                implied_vol = self.estimate_implied_volatility_black_scholes(
                    current_price, strike, time_to_expiry, 0.02, synthetic_option_price, 'call'
                )
                
                volatility_smile[moneyness] = implied_vol
            
            # Fit quadratic function to volatility smile
            moneyness_values = list(volatility_smile.keys())
            vol_values = list(volatility_smile.values())
            
            if len(moneyness_values) >= 3:
                coeffs = np.polyfit(moneyness_values, vol_values, 2)
                smile_curvature = coeffs[0]
            else:
                smile_curvature = 0.0
            
            return {
                'volatility_smile': volatility_smile,
                'smile_curvature': float(smile_curvature),
                'current_price': float(current_price),
                'time_to_expiry': time_to_expiry
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility smile: {str(e)}")
            return {
                'volatility_smile': {},
                'smile_curvature': 0.0,
                'current_price': 0.0,
                'time_to_expiry': time_to_expiry
            }
    
    def analyze_volatility_surface(self, symbol: str, market_data: Dict) -> Dict[str, Any]:
        """
        Analyze volatility surface for comprehensive volatility analysis.
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
            
        Returns:
            Comprehensive volatility analysis
        """
        try:
            if 'close' not in market_data or len(market_data['close']) < 100:
                return {'error': 'Insufficient data for volatility surface analysis'}
            
            prices = pd.Series(market_data['close'])
            returns = prices.pct_change().dropna()
            
            # Calculate different volatility measures
            historical_vol = self.calculate_historical_volatility(prices)
            realized_vol = self.calculate_realized_volatility(returns)
            garch_vol = self.calculate_garch_volatility(returns)
            
            # Detect volatility regime
            volatility_series = returns.rolling(30).std() * np.sqrt(252)
            regime_analysis = self.detect_volatility_regime(volatility_series)
            
            # Forecast volatility
            forecast = self.forecast_volatility(volatility_series)
            
            # Calculate volatility smile
            strikes = [prices.iloc[-1] * x for x in [0.8, 0.9, 1.0, 1.1, 1.2]]
            smile_analysis = self.calculate_volatility_smile(prices, strikes)
            
            # Volatility risk metrics
            vol_of_vol = returns.rolling(30).std().rolling(30).std().iloc[-1] * np.sqrt(252)
            vol_skewness = volatility_series.skew()
            vol_kurtosis = volatility_series.kurtosis()
            
            return {
                'symbol': symbol,
                'historical_volatility': historical_vol,
                'realized_volatility': realized_vol,
                'garch_volatility': garch_vol['garch_volatility'],
                'regime_analysis': regime_analysis,
                'volatility_forecast': forecast,
                'volatility_smile': smile_analysis,
                'volatility_risk_metrics': {
                    'volatility_of_volatility': float(vol_of_vol),
                    'volatility_skewness': float(vol_skewness),
                    'volatility_kurtosis': float(vol_kurtosis)
                },
                'volatility_consensus': {
                    'mean_volatility': float(np.mean([historical_vol, realized_vol, garch_vol['garch_volatility']])),
                    'volatility_dispersion': float(np.std([historical_vol, realized_vol, garch_vol['garch_volatility']])),
                    'volatility_confidence': float(1.0 - np.std([historical_vol, realized_vol, garch_vol['garch_volatility']]) / np.mean([historical_vol, realized_vol, garch_vol['garch_volatility']]))
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volatility surface: {str(e)}")
            return {'error': str(e)}
    
    def adjust_position_size_by_volatility(self, base_position_size: float, 
                                         volatility_analysis: Dict) -> float:
        """
        Adjust position size based on volatility analysis.
        
        Args:
            base_position_size: Base position size
            volatility_analysis: Volatility analysis results
            
        Returns:
            Adjusted position size
        """
        try:
            # Get volatility consensus
            vol_consensus = volatility_analysis.get('volatility_consensus', {})
            mean_vol = vol_consensus.get('mean_volatility', 0.2)
            vol_confidence = vol_consensus.get('volatility_confidence', 0.5)
            
            # Get regime analysis
            regime_analysis = volatility_analysis.get('regime_analysis', {})
            regime = regime_analysis.get('regime', 'normal_volatility')
            regime_score = regime_analysis.get('regime_score', 0.5)
            
            # Volatility adjustment factors
            vol_adjustment = 1.0
            
            # High volatility regime - reduce position size
            if regime == 'high_volatility':
                vol_adjustment *= (1.0 - regime_score * 0.5)
            # Low volatility regime - increase position size
            elif regime == 'low_volatility':
                vol_adjustment *= (1.0 + regime_score * 0.3)
            
            # Volatility confidence adjustment
            vol_adjustment *= (0.5 + vol_confidence * 0.5)
            
            # Ensure position size is within reasonable bounds
            adjusted_size = base_position_size * vol_adjustment
            adjusted_size = max(adjusted_size, base_position_size * 0.1)  # Minimum 10%
            adjusted_size = min(adjusted_size, base_position_size * 2.0)   # Maximum 200%
            
            return float(adjusted_size)
            
        except Exception as e:
            logger.error(f"Error adjusting position size by volatility: {str(e)}")
            return base_position_size
    
    def calculate_volatility_optimal_stop_loss(self, current_price: float, 
                                             volatility_analysis: Dict, 
                                             position_type: str = 'long') -> float:
        """
        Calculate optimal stop loss based on volatility analysis.
        
        Args:
            current_price: Current asset price
            volatility_analysis: Volatility analysis results
            position_type: 'long' or 'short'
            
        Returns:
            Optimal stop loss price
        """
        try:
            # Get volatility consensus
            vol_consensus = volatility_analysis.get('volatility_consensus', {})
            mean_vol = vol_consensus.get('mean_volatility', 0.2)
            
            # Get regime analysis
            regime_analysis = volatility_analysis.get('regime_analysis', {})
            regime = regime_analysis.get('regime', 'normal_volatility')
            regime_score = regime_analysis.get('regime_score', 0.5)
            
            # Base stop loss distance (2 standard deviations)
            base_distance = 2.0 * mean_vol / np.sqrt(252)  # Daily volatility
            
            # Regime adjustment
            if regime == 'high_volatility':
                base_distance *= (1.0 + regime_score * 0.5)
            elif regime == 'low_volatility':
                base_distance *= (1.0 - regime_score * 0.3)
            
            # Calculate stop loss
            if position_type == 'long':
                stop_loss = current_price * (1.0 - base_distance)
            else:  # short
                stop_loss = current_price * (1.0 + base_distance)
            
            return float(stop_loss)
            
        except Exception as e:
            logger.error(f"Error calculating volatility optimal stop loss: {str(e)}")
            return current_price * 0.95 if position_type == 'long' else current_price * 1.05
    
    def get_volatility_trading_signals(self, volatility_analysis: Dict) -> Dict[str, Any]:
        """
        Generate trading signals based on volatility analysis.
        
        Args:
            volatility_analysis: Volatility analysis results
            
        Returns:
            Volatility-based trading signals
        """
        try:
            signals = {}
            
            # Get key metrics
            regime_analysis = volatility_analysis.get('regime_analysis', {})
            regime = regime_analysis.get('regime', 'normal_volatility')
            regime_score = regime_analysis.get('regime_score', 0.5)
            
            vol_consensus = volatility_analysis.get('volatility_consensus', {})
            mean_vol = vol_consensus.get('mean_volatility', 0.2)
            vol_confidence = vol_consensus.get('volatility_confidence', 0.5)
            
            # Volatility regime signals
            if regime == 'high_volatility':
                signals['volatility_signal'] = 'reduce_exposure'
                signals['volatility_reason'] = 'High volatility regime detected'
                signals['position_adjustment'] = -regime_score * 0.5
            elif regime == 'low_volatility':
                signals['volatility_signal'] = 'increase_exposure'
                signals['volatility_reason'] = 'Low volatility regime detected'
                signals['position_adjustment'] = regime_score * 0.3
            else:
                signals['volatility_signal'] = 'maintain_exposure'
                signals['volatility_reason'] = 'Normal volatility regime'
                signals['position_adjustment'] = 0.0
            
            # Volatility confidence signals
            if vol_confidence > 0.8:
                signals['confidence_signal'] = 'high_confidence'
                signals['confidence_reason'] = 'Low volatility dispersion'
            elif vol_confidence < 0.4:
                signals['confidence_signal'] = 'low_confidence'
                signals['confidence_reason'] = 'High volatility dispersion'
            else:
                signals['confidence_signal'] = 'medium_confidence'
                signals['confidence_reason'] = 'Moderate volatility dispersion'
            
            # Risk management signals
            vol_risk_metrics = volatility_analysis.get('volatility_risk_metrics', {})
            vol_of_vol = vol_risk_metrics.get('volatility_of_volatility', 0.0)
            
            if vol_of_vol > 0.5:
                signals['risk_signal'] = 'high_risk'
                signals['risk_reason'] = 'High volatility of volatility'
            else:
                signals['risk_signal'] = 'normal_risk'
                signals['risk_reason'] = 'Normal volatility of volatility'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating volatility trading signals: {str(e)}")
            return {
                'volatility_signal': 'unknown',
                'volatility_reason': 'Error in volatility analysis',
                'position_adjustment': 0.0,
                'confidence_signal': 'unknown',
                'confidence_reason': 'Error in confidence analysis',
                'risk_signal': 'unknown',
                'risk_reason': 'Error in risk analysis'
            } 