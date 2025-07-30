"""
Statistical Validator for WorldQuant-level trading strategy validation.
Implements hypothesis testing, bootstrap confidence intervals, and statistical validation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from sklearn.utils import resample
import warnings

# Suppress divide by zero warnings from scipy
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')

logger = logging.getLogger(__name__)

class StatisticalValidator:
    """
    WorldQuant-level statistical validator for trading signals.
    Implements hypothesis testing, bootstrap confidence intervals, and walk-forward backtesting.
    """
    
    def __init__(self, significance_level: float = 0.05, min_sample_size: int = 100):
        """
        Initialize statistical validator.
        
        Args:
            significance_level: Significance level for hypothesis testing (default: 0.05)
            min_sample_size: Minimum sample size required for statistical tests (default: 100)
        """
        self.significance_level = significance_level
        self.min_sample_size = min_sample_size
        self.validation_history = {}
        
        logger.info(f"StatisticalValidator initialized with significance_level={significance_level}, min_sample_size={min_sample_size}")
    
    def test_signal_significance(self, signal_history: List[Dict], benchmark_returns: np.ndarray) -> Dict[str, Any]:
        """
        Test if trading signals are statistically significant compared to benchmark.
        
        Args:
            signal_history: List of signal dictionaries with returns
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            Dictionary with statistical test results
        """
        try:
            # Extract signal returns
            signal_returns = self._extract_signal_returns(signal_history)
            
            if len(signal_returns) < self.min_sample_size:
                logger.warning(f"Insufficient signal history: {len(signal_returns)} < {self.min_sample_size}")
                return {
                    'significant': False,
                    't_statistic': 0.0,
                    'p_value': 1.0,
                    'confidence_interval': (0.0, 0.0),
                    'sample_size': len(signal_returns),
                    'error': 'Insufficient sample size'
                }
            
            # Ensure benchmark returns has same length
            if len(benchmark_returns) != len(signal_returns):
                # Align lengths by taking the minimum
                min_length = min(len(signal_returns), len(benchmark_returns))
                signal_returns = signal_returns[:min_length]
                benchmark_returns = benchmark_returns[:min_length]
            
            # Perform t-test for mean return difference
            t_stat, p_value = stats.ttest_ind(signal_returns, benchmark_returns)
            
            # Calculate bootstrap confidence interval
            ci_lower, ci_upper = self._bootstrap_confidence_interval(signal_returns)
            
            # Calculate effect size (Cohen's d)
            effect_size = self._calculate_effect_size(signal_returns, benchmark_returns)
            
            # Determine if significant
            is_significant = p_value < self.significance_level
            
            result = {
                'significant': is_significant,
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'confidence_interval': (float(ci_lower), float(ci_upper)),
                'effect_size': float(effect_size),
                'sample_size': len(signal_returns),
                'mean_signal_return': float(np.mean(signal_returns)),
                'mean_benchmark_return': float(np.mean(benchmark_returns)),
                'signal_volatility': float(np.std(signal_returns)),
                'benchmark_volatility': float(np.std(benchmark_returns))
            }
            
            logger.info(f"Signal significance test: significant={is_significant}, p_value={p_value:.4f}, effect_size={effect_size:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in signal significance test: {str(e)}")
            return {
                'significant': False,
                't_statistic': 0.0,
                'p_value': 1.0,
                'confidence_interval': (0.0, 0.0),
                'error': str(e)
            }
    
    def _extract_signal_returns(self, signal_history: List[Dict]) -> np.ndarray:
        """
        Extract returns from signal history.
        
        Args:
            signal_history: List of signal dictionaries
            
        Returns:
            Array of signal returns
        """
        try:
            returns = []
            
            for signal in signal_history:
                # Extract return from signal
                if 'return' in signal:
                    returns.append(float(signal['return']))
                elif 'pnl' in signal:
                    returns.append(float(signal['pnl']))
                elif 'performance' in signal:
                    returns.append(float(signal['performance']))
                else:
                    # Calculate return from price change if available
                    if 'entry_price' in signal and 'exit_price' in signal:
                        entry_price = float(signal['entry_price'])
                        exit_price = float(signal['exit_price'])
                        if entry_price > 0:
                            returns.append((exit_price - entry_price) / entry_price)
            
            return np.array(returns)
            
        except Exception as e:
            logger.error(f"Error extracting signal returns: {str(e)}")
            return np.array([])
    
    def _bootstrap_confidence_interval(self, data: np.ndarray, confidence_level: float = 0.95, n_bootstrap: int = 10000) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for mean return.
        
        Args:
            data: Array of returns
            confidence_level: Confidence level (default: 0.95)
            n_bootstrap: Number of bootstrap samples (default: 10000)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        try:
            if len(data) == 0:
                return (0.0, 0.0)
            
            # Generate bootstrap samples
            bootstrap_means = []
            
            for _ in range(n_bootstrap):
                # Resample with replacement
                bootstrap_sample = resample(data, n_samples=len(data))
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            # Calculate confidence interval
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_means, lower_percentile)
            ci_upper = np.percentile(bootstrap_means, upper_percentile)
            
            return float(ci_lower), float(ci_upper)
            
        except Exception as e:
            logger.error(f"Error calculating bootstrap confidence interval: {str(e)}")
            return (0.0, 0.0)
    
    def _calculate_effect_size(self, signal_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size.
        
        Args:
            signal_returns: Signal returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Effect size (Cohen's d)
        """
        try:
            if len(signal_returns) == 0 or len(benchmark_returns) == 0:
                return 0.0
            
            # Calculate pooled standard deviation
            n1, n2 = len(signal_returns), len(benchmark_returns)
            var1, var2 = np.var(signal_returns, ddof=1), np.var(benchmark_returns, ddof=1)
            
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            
            if pooled_std == 0:
                return 0.0
            
            # Calculate effect size
            effect_size = (np.mean(signal_returns) - np.mean(benchmark_returns)) / pooled_std
            
            return float(effect_size)
            
        except Exception as e:
            logger.error(f"Error calculating effect size: {str(e)}")
            return 0.0
    
    def validate_signal_quality(self, signal: Dict) -> Dict[str, Any]:
        """
        Validate individual signal quality using statistical measures.
        
        Args:
            signal: Signal dictionary
            
        Returns:
            Dictionary with validation results
        """
        try:
            validation_result = {
                'is_valid': True,
                'confidence_score': 0.0,
                'statistical_measures': {},
                'warnings': []
            }
            
            # Check signal strength
            signal_strength = abs(signal.get('strength', 0))
            if signal_strength < 0.1:
                validation_result['warnings'].append('Low signal strength')
                validation_result['confidence_score'] -= 0.2
            
            # Check confidence level
            confidence = signal.get('confidence', 0)
            if confidence < 0.3:
                validation_result['warnings'].append('Low confidence level')
                validation_result['confidence_score'] -= 0.3
            
            # Check current price validity
            current_price = signal.get('current_price', 0)
            if current_price <= 0:
                validation_result['warnings'].append('Invalid current price')
                validation_result['is_valid'] = False
            
            # Check for sufficient reasons
            reasons = signal.get('reasons', [])
            if len(reasons) < 2:
                validation_result['warnings'].append('Insufficient signal reasons')
                validation_result['confidence_score'] -= 0.1
            
            # Calculate statistical measures
            validation_result['statistical_measures'] = {
                'signal_strength': signal_strength,
                'confidence_level': confidence,
                'reason_count': len(reasons),
                'price_validity': current_price > 0
            }
            
            # Adjust confidence score based on positive factors
            if signal_strength > 0.5:
                validation_result['confidence_score'] += 0.3
            if confidence > 0.7:
                validation_result['confidence_score'] += 0.3
            if len(reasons) > 3:
                validation_result['confidence_score'] += 0.2
            
            # Ensure confidence score is between 0 and 1
            validation_result['confidence_score'] = max(0.0, min(1.0, validation_result['confidence_score']))
            
            # Final validation
            if len(validation_result['warnings']) > 2:
                validation_result['is_valid'] = False
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating signal quality: {str(e)}")
            return {
                'is_valid': False,
                'confidence_score': 0.0,
                'statistical_measures': {},
                'warnings': [f'Validation error: {str(e)}']
            }
    
    def perform_walk_forward_analysis(self, strategy, data: pd.DataFrame, 
                                    train_window: int = 252, test_window: int = 63) -> Dict[str, Any]:
        """
        Perform walk-forward analysis with proper out-of-sample testing.
        
        Args:
            strategy: Trading strategy object
            data: Historical market data
            train_window: Training window size (default: 252 days)
            test_window: Test window size (default: 63 days)
            
        Returns:
            Dictionary with walk-forward analysis results
        """
        try:
            results = []
            total_periods = len(data)
            
            if total_periods < train_window + test_window:
                logger.warning(f"Insufficient data for walk-forward analysis: {total_periods} < {train_window + test_window}")
                return {
                    'success': False,
                    'error': 'Insufficient data for walk-forward analysis'
                }
            
            # Perform walk-forward analysis
            for i in range(0, total_periods - train_window - test_window + 1, test_window):
                # Training period
                train_start = i
                train_end = i + train_window
                train_data = data.iloc[train_start:train_end]
                
                # Test period
                test_start = train_end
                test_end = min(test_start + test_window, total_periods)
                test_data = data.iloc[test_start:test_end]
                
                # Train strategy on training data
                strategy.train(train_data)
                
                # Test strategy on out-of-sample data
                test_results = strategy.test(test_data)
                
                results.append({
                    'train_period': (train_start, train_end),
                    'test_period': (test_start, test_end),
                    'results': test_results
                })
            
            # Aggregate results
            aggregated_results = self._aggregate_walk_forward_results(results)
            
            logger.info(f"Walk-forward analysis completed: {len(results)} periods analyzed")
            
            return {
                'success': True,
                'total_periods': len(results),
                'aggregated_results': aggregated_results,
                'period_results': results
            }
            
        except Exception as e:
            logger.error(f"Error in walk-forward analysis: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate walk-forward analysis results.
        
        Args:
            results: List of period results
            
        Returns:
            Aggregated results dictionary
        """
        try:
            if not results:
                return {}
            
            # Extract performance metrics
            returns = []
            sharpe_ratios = []
            max_drawdowns = []
            hit_rates = []
            
            for result in results:
                period_results = result['results']
                
                if 'returns' in period_results:
                    returns.extend(period_results['returns'])
                if 'sharpe_ratio' in period_results:
                    sharpe_ratios.append(period_results['sharpe_ratio'])
                if 'max_drawdown' in period_results:
                    max_drawdowns.append(period_results['max_drawdown'])
                if 'hit_rate' in period_results:
                    hit_rates.append(period_results['hit_rate'])
            
            # Calculate aggregated metrics
            aggregated = {
                'total_return': np.sum(returns) if returns else 0.0,
                'mean_return': np.mean(returns) if returns else 0.0,
                'return_volatility': np.std(returns) if returns else 0.0,
                'mean_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0.0,
                'mean_max_drawdown': np.mean(max_drawdowns) if max_drawdowns else 0.0,
                'mean_hit_rate': np.mean(hit_rates) if hit_rates else 0.0,
                'consistency_score': self._calculate_consistency_score(results)
            }
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating walk-forward results: {str(e)}")
            return {}
    
    def _calculate_consistency_score(self, results: List[Dict]) -> float:
        """
        Calculate consistency score across walk-forward periods.
        
        Args:
            results: List of period results
            
        Returns:
            Consistency score (0-1)
        """
        try:
            if not results:
                return 0.0
            
            # Extract performance metrics for consistency calculation
            sharpe_ratios = []
            hit_rates = []
            
            for result in results:
                period_results = result['results']
                if 'sharpe_ratio' in period_results:
                    sharpe_ratios.append(period_results['sharpe_ratio'])
                if 'hit_rate' in period_results:
                    hit_rates.append(period_results['hit_rate'])
            
            # Calculate consistency based on coefficient of variation
            consistency_scores = []
            
            if sharpe_ratios:
                sharpe_cv = np.std(sharpe_ratios) / (np.mean(sharpe_ratios) + 1e-8)
                consistency_scores.append(1.0 / (1.0 + sharpe_cv))
            
            if hit_rates:
                hit_rate_cv = np.std(hit_rates) / (np.mean(hit_rates) + 1e-8)
                consistency_scores.append(1.0 / (1.0 + hit_rate_cv))
            
            # Return average consistency score
            return np.mean(consistency_scores) if consistency_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating consistency score: {str(e)}")
            return 0.0
    
    def validate_market_regime_stability(self, returns: np.ndarray, window: int = 60) -> Dict[str, Any]:
        """
        Validate market regime stability using statistical tests.
        
        Args:
            returns: Array of returns
            window: Rolling window size (default: 60)
            
        Returns:
            Dictionary with regime stability validation results
        """
        try:
            if len(returns) < window * 2:
                return {
                    'is_stable': False,
                    'stability_score': 0.0,
                    'regime_changes': 0,
                    'error': 'Insufficient data for regime analysis'
                }
            
            # Calculate rolling volatility
            rolling_vol = pd.Series(returns).rolling(window).std()
            
            # Detect regime changes using structural break tests
            regime_changes = self._detect_regime_changes(rolling_vol.dropna())
            
            # Calculate stability score
            volatility_cv = np.std(rolling_vol.dropna()) / (np.mean(rolling_vol.dropna()) + 1e-8)
            stability_score = 1.0 / (1.0 + volatility_cv)
            
            # Determine if regime is stable
            is_stable = stability_score > 0.7 and len(regime_changes) < 3
            
            return {
                'is_stable': is_stable,
                'stability_score': float(stability_score),
                'regime_changes': len(regime_changes),
                'volatility_cv': float(volatility_cv),
                'mean_volatility': float(np.mean(rolling_vol.dropna())),
                'volatility_std': float(np.std(rolling_vol.dropna()))
            }
            
        except Exception as e:
            logger.error(f"Error validating market regime stability: {str(e)}")
            return {
                'is_stable': False,
                'stability_score': 0.0,
                'regime_changes': 0,
                'error': str(e)
            }
    
    def _detect_regime_changes(self, data: pd.Series) -> List[int]:
        """
        Detect structural breaks in time series data.
        
        Args:
            data: Time series data
            
        Returns:
            List of change point indices
        """
        try:
            if len(data) < 20:
                return []
            
            # Simple regime change detection using rolling mean comparison
            change_points = []
            window_size = min(20, len(data) // 4)
            
            rolling_mean = data.rolling(window=window_size).mean()
            rolling_std = data.rolling(window=window_size).std()
            
            for i in range(window_size * 2, len(data) - window_size):
                # Compare current window with previous window
                current_mean = rolling_mean.iloc[i]
                previous_mean = rolling_mean.iloc[i - window_size]
                current_std = rolling_std.iloc[i]
                
                # Detect significant change
                if current_std > 0:
                    z_score = abs(current_mean - previous_mean) / current_std
                    if z_score > 2.0:  # Significant change
                        change_points.append(i)
            
            return change_points
            
        except Exception as e:
            logger.error(f"Error detecting regime changes: {str(e)}")
            return []
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all validation results.
        
        Returns:
            Dictionary with validation summary
        """
        try:
            summary = {
                'total_validations': len(self.validation_history),
                'successful_validations': 0,
                'failed_validations': 0,
                'average_confidence_score': 0.0,
                'statistical_significance_rate': 0.0,
                'regime_stability_rate': 0.0
            }
            
            if self.validation_history:
                successful_count = sum(1 for v in self.validation_history.values() if v.get('is_valid', False))
                confidence_scores = [v.get('confidence_score', 0) for v in self.validation_history.values()]
                significance_count = sum(1 for v in self.validation_history.values() if v.get('significant', False))
                stability_count = sum(1 for v in self.validation_history.values() if v.get('is_stable', False))
                
                summary.update({
                    'successful_validations': successful_count,
                    'failed_validations': len(self.validation_history) - successful_count,
                    'average_confidence_score': np.mean(confidence_scores) if confidence_scores else 0.0,
                    'statistical_significance_rate': significance_count / len(self.validation_history),
                    'regime_stability_rate': stability_count / len(self.validation_history)
                })
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting validation summary: {str(e)}")
            return {} 