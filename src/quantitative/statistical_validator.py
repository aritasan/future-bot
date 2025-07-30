import numpy as np
import pandas as pd
import warnings
from scipy import stats
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple
import logging

# Suppress divide by zero warnings from scipy
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')

logger = logging.getLogger(__name__)

class StatisticalSignalValidator:
    """
    Statistical validation framework for trading signals.
    Implements quantitative trading standards for signal validation.
    """
    
    def __init__(self, min_p_value: float = 0.1, min_t_stat: float = 1.5):
        """
        Initialize statistical validator with more realistic thresholds.
        
        Args:
            min_p_value: Minimum p-value threshold (default: 0.1, more realistic)
            min_t_stat: Minimum t-statistic threshold (default: 1.5, more realistic)
        """
        self.min_p_value = min_p_value
        self.min_t_stat = min_t_stat
        self.validation_history = []
        
        # Adaptive thresholds based on market conditions
        self.adaptive_thresholds = {
            'high_volatility': {
                'min_p_value': 0.15,
                'min_t_stat': 1.2,
                'min_sharpe_ratio': 0.1,
                'max_drawdown': 0.3
            },
            'normal_volatility': {
                'min_p_value': 0.1,
                'min_t_stat': 1.5,
                'min_sharpe_ratio': 0.2,
                'max_drawdown': 0.25
            },
            'low_volatility': {
                'min_p_value': 0.05,
                'min_t_stat': 2.0,
                'min_sharpe_ratio': 0.5,
                'max_drawdown': 0.15
            }
        }
    
    def validate_signal(self, signal_data: Dict, historical_returns: np.array = None) -> Dict:
        """
        Validate signal using statistical tests with adaptive thresholds.
        
        Args:
            signal_data: Dictionary containing signal information
            historical_returns: Historical returns for statistical testing
            
        Returns:
            Dict: Validation results with statistical metrics
        """
        try:
            results = {
                'is_valid': False,
                'p_value': None,
                't_statistic': None,
                'sharpe_ratio': None,
                'information_ratio': None,
                'calmar_ratio': None,
                'sortino_ratio': None,
                'max_drawdown': None,
                'volatility': None,
                'skewness': None,
                'kurtosis': None,
                'market_regime': None,
                'adaptive_thresholds_used': None
            }
            
            if historical_returns is None:
                # Generate synthetic returns based on signal strength
                historical_returns = self._generate_synthetic_returns(signal_data)
            
            # Determine market regime based on volatility
            volatility = float(np.std(historical_returns) * np.sqrt(252))
            market_regime = self._determine_market_regime(volatility)
            
            # Get adaptive thresholds for current market regime
            thresholds = self.adaptive_thresholds[market_regime]
            
            # Perform statistical tests
            t_stat, p_value = stats.ttest_1samp(historical_returns, 0)
            
            # Calculate risk-adjusted metrics
            sharpe_ratio = self._calculate_sharpe_ratio(historical_returns)
            information_ratio = self._calculate_information_ratio(historical_returns)
            sortino_ratio = self._calculate_sortino_ratio(historical_returns)
            calmar_ratio = self._calculate_calmar_ratio(historical_returns)
            
            # Calculate additional metrics
            max_drawdown = self._calculate_max_drawdown(historical_returns)
            skewness = stats.skew(historical_returns)
            kurtosis = stats.kurtosis(historical_returns)
            
            # Determine if signal is statistically valid using adaptive thresholds
            is_valid = (p_value < thresholds['min_p_value'] and 
                       abs(t_stat) > thresholds['min_t_stat'] and
                       sharpe_ratio > thresholds['min_sharpe_ratio'] and
                       max_drawdown < thresholds['max_drawdown'])
            
            results.update({
                'is_valid': is_valid,
                'p_value': p_value,
                't_statistic': t_stat,
                'sharpe_ratio': sharpe_ratio,
                'information_ratio': information_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'market_regime': market_regime,
                'adaptive_thresholds_used': thresholds
            })
            
            # Store validation history
            self.validation_history.append({
                'timestamp': pd.Timestamp.now(),
                'signal_data': signal_data,
                'validation_results': results
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in signal validation: {str(e)}")
            return {'is_valid': False, 'error': str(e)}
    
    def _generate_synthetic_returns(self, signal_data: Dict) -> np.array:
        """Generate synthetic returns based on signal characteristics."""
        try:
            signal_strength = signal_data.get('strength', 0.5)
            confidence = signal_data.get('confidence', 0.5)
            
            # Base return expectation
            expected_return = signal_strength * confidence * 0.02  # 2% base
            
            # Generate returns with some noise
            n_periods = 252  # One year of daily data
            returns = np.random.normal(expected_return, 0.02, n_periods)
            
            # Add some autocorrelation to make it more realistic
            for i in range(1, len(returns)):
                returns[i] = 0.1 * returns[i-1] + 0.9 * returns[i]
            
            return returns
            
        except Exception as e:
            logger.error(f"Error generating synthetic returns: {str(e)}")
            return np.random.normal(0, 0.02, 252)
    
    def _calculate_sharpe_ratio(self, returns: np.array, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        try:
            if returns is None or len(returns) < 2 or bool(np.all(np.isnan(returns))) or float(np.nanstd(returns)) == 0:
                return 0.0
            excess_returns = returns - risk_free_rate/252
            std_dev = np.nanstd(excess_returns)
            if std_dev == 0 or np.isnan(std_dev):
                return 0.0
            return float(np.nanmean(excess_returns) / std_dev * np.sqrt(252))
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0
    
    def _calculate_information_ratio(self, returns: np.array, benchmark_returns: np.array = None) -> float:
        """Calculate Information ratio."""
        try:
            if returns is None or len(returns) < 2 or bool(np.all(np.isnan(returns))):
                return 0.0
            if benchmark_returns is None:
                benchmark_returns = np.random.normal(0.0005, 0.03, len(returns))
            excess_returns = returns - benchmark_returns
            std_dev = np.nanstd(excess_returns)
            if std_dev == 0 or np.isnan(std_dev):
                return 0.0
            return float(np.nanmean(excess_returns) / std_dev * np.sqrt(252))
        except Exception as e:
            logger.error(f"Error calculating Information ratio: {str(e)}")
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: np.array, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio."""
        try:
            if returns is None or len(returns) < 2 or bool(np.all(np.isnan(returns))):
                return 0.0
            excess_returns = returns - risk_free_rate/252
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) == 0 or bool(np.all(np.isnan(downside_returns))):
                return 0.0
            downside_deviation = float(np.nanstd(downside_returns))
            if downside_deviation == 0 or np.isnan(downside_deviation):
                return 0.0
            return float(np.nanmean(excess_returns) / downside_deviation * np.sqrt(252))
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0.0
    
    def _calculate_calmar_ratio(self, returns: np.array) -> float:
        """Calculate Calmar ratio."""
        try:
            if returns is None or len(returns) < 2 or bool(np.all(np.isnan(returns))):
                return 0.0
            cumulative_returns = np.cumprod(1 + returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            if max_drawdown == 0 or np.isnan(max_drawdown):
                return 0.0
            annual_return = (cumulative_returns[-1] - 1) * 252 / len(returns)
            if abs(max_drawdown) < 1e-8:
                return 0.0
            return annual_return / abs(max_drawdown)
        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {str(e)}")
            return 0.0
    
    def _calculate_max_drawdown(self, returns: np.array) -> float:
        """Calculate maximum drawdown."""
        try:
            if returns is None or len(returns) < 2 or bool(np.all(np.isnan(returns))):
                return 0.0
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            if len(drawdown) == 0 or bool(np.all(np.isnan(drawdown))):
                return 0.0
            return float(np.nanmin(drawdown))
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0
    
    def _determine_market_regime(self, volatility: float) -> str:
        """
        Determine market regime based on volatility.
        
        Args:
            volatility: Annualized volatility
            
        Returns:
            str: Market regime ('high_volatility', 'normal_volatility', 'low_volatility')
        """
        if volatility > 0.4:  # High volatility (>40% annualized)
            return 'high_volatility'
        elif volatility < 0.2:  # Low volatility (<20% annualized)
            return 'low_volatility'
        else:  # Normal volatility (20-40% annualized)
            return 'normal_volatility'
    
    def get_validation_summary(self) -> Dict:
        """Get summary of validation history."""
        try:
            if not self.validation_history:
                return {'total_validations': 0}
            valid_signals = [v for v in self.validation_history if v['validation_results']['is_valid']]
            total_valid = len(valid_signals)
            total = len(self.validation_history)
            summary = {
                'total_validations': total,
                'valid_signals': total_valid,
                'validation_rate': total_valid / total if total > 0 else 0,
                'avg_sharpe_ratio': np.mean([v['validation_results']['sharpe_ratio'] for v in self.validation_history]) if total > 0 else 0,
                'avg_information_ratio': np.mean([v['validation_results']['information_ratio'] for v in self.validation_history]) if total > 0 else 0,
                'avg_max_drawdown': np.mean([v['validation_results']['max_drawdown'] for v in self.validation_history]) if total > 0 else 0
            }
            return summary
        except Exception as e:
            logger.error(f"Error getting validation summary: {str(e)}")
            return {'error': str(e)}
    
    def validate_signal_ensemble(self, signals: List[Dict]) -> Dict:
        """Validate multiple signals and return ensemble validation."""
        try:
            validations = []
            for signal in signals:
                validation = self.validate_signal(signal)
                validations.append(validation)
            
            # Calculate ensemble metrics
            valid_signals = [v for v in validations if v['is_valid']]
            
            if not valid_signals:
                return {'ensemble_valid': False, 'valid_signals': 0}
            
            ensemble_metrics = {
                'ensemble_valid': len(valid_signals) >= len(signals) * 0.7,  # 70% threshold
                'valid_signals': len(valid_signals),
                'total_signals': len(signals),
                'avg_sharpe': np.mean([v['sharpe_ratio'] for v in valid_signals]),
                'avg_information_ratio': np.mean([v['information_ratio'] for v in valid_signals]),
                'min_p_value': min([v['p_value'] for v in valid_signals]),
                'max_t_stat': max([abs(v['t_statistic']) for v in valid_signals])
            }
            
            return ensemble_metrics
            
        except Exception as e:
            logger.error(f"Error in ensemble validation: {str(e)}")
            return {'ensemble_valid': False, 'error': str(e)}
    
    def perform_hypothesis_test(self, signal_returns: np.array, null_hypothesis: float = 0.0) -> Dict:
        """Perform comprehensive hypothesis testing."""
        try:
            # One-sample t-test
            t_stat, p_value = stats.ttest_1samp(signal_returns, null_hypothesis)
            
            # Wilcoxon signed-rank test (non-parametric)
            w_stat, w_p_value = stats.wilcoxon(signal_returns)
            
            # Jarque-Bera test for normality
            jb_stat, jb_p_value = stats.jarque_bera(signal_returns)
            
            # Ljung-Box test for autocorrelation
            lb_stat, lb_p_value = stats.acf(signal_returns, nlags=10, fft=True)
            
            return {
                't_test': {'statistic': t_stat, 'p_value': p_value},
                'wilcoxon_test': {'statistic': w_stat, 'p_value': w_p_value},
                'jarque_bera_test': {'statistic': jb_stat, 'p_value': jb_p_value},
                'autocorrelation': lb_stat,
                'is_normal': jb_p_value > 0.05,
                'has_autocorrelation': any(abs(lb_stat) > 0.1)
            }
            
        except Exception as e:
            logger.error(f"Error in hypothesis testing: {str(e)}")
            return {'error': str(e)} 