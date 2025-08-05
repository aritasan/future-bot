#!/usr/bin/env python3
"""
Advanced Risk Management Module
WorldQuant Standards Implementation

Implements:
- Dynamic VaR with regime switching
- Expected Shortfall (Conditional VaR)
- Portfolio-Level Risk Attribution
- Stress Testing with historical scenarios
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

class DynamicRiskManager:
    """
    Advanced Risk Management System following WorldQuant standards.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize Dynamic Risk Manager."""
        self.config = config or {}
        self.risk_history = {}
        self.stress_scenarios = {}
        self.regime_models = {}
        
        # Risk parameters
        self.var_confidence_levels = {
            'high_volatility': 0.99,
            'normal_volatility': 0.975,
            'low_volatility': 0.95
        }
        
        # Stress test scenarios
        self._initialize_stress_scenarios()
        
        logger.info("Dynamic Risk Manager initialized")
    
    def _initialize_stress_scenarios(self):
        """Initialize historical stress scenarios."""
        self.stress_scenarios = {
            'crypto_winter_2018': {
                'btc_drop': -0.85,
                'eth_drop': -0.92,
                'market_correlation': 0.95,
                'volatility_multiplier': 3.0
            },
            'covid_crash_2020': {
                'btc_drop': -0.50,
                'eth_drop': -0.60,
                'market_correlation': 0.90,
                'volatility_multiplier': 2.5
            },
            'ftx_collapse_2022': {
                'btc_drop': -0.25,
                'eth_drop': -0.30,
                'market_correlation': 0.85,
                'volatility_multiplier': 2.0
            },
            'silvergate_collapse_2023': {
                'btc_drop': -0.15,
                'eth_drop': -0.20,
                'market_correlation': 0.80,
                'volatility_multiplier': 1.8
            }
        }
    
    def calculate_dynamic_var(self, returns: pd.Series, regime: str = 'normal_volatility') -> Dict[str, float]:
        """
        Calculate Dynamic VaR with regime switching.
        
        Args:
            returns: Return series
            regime: Volatility regime ('high_volatility', 'normal_volatility', 'low_volatility')
            
        Returns:
            Dictionary with VaR results
        """
        try:
            confidence_level = self.var_confidence_levels.get(regime, 0.975)
            
            # Calculate VaR using different methods
            var_results = {}
            
            # 1. Historical VaR
            var_results['historical_var'] = self._calculate_historical_var(returns, confidence_level)
            
            # 2. Parametric VaR (assuming normal distribution)
            var_results['parametric_var'] = self._calculate_parametric_var(returns, confidence_level)
            
            # 3. Modified VaR (using Cornish-Fisher expansion)
            var_results['modified_var'] = self._calculate_modified_var(returns, confidence_level)
            
            # 4. Regime-adjusted VaR
            regime_multiplier = self._get_regime_multiplier(regime)
            var_results['regime_adjusted_var'] = var_results['historical_var'] * regime_multiplier
            
            # 5. Expected Shortfall (Conditional VaR)
            var_results['expected_shortfall'] = self._calculate_expected_shortfall(returns, confidence_level)
            
            # 6. Dynamic confidence interval
            var_results['confidence_interval'] = self._calculate_var_confidence_interval(returns, confidence_level)
            
            return var_results
            
        except Exception as e:
            logger.error(f"Error calculating dynamic VaR: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_historical_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate historical VaR."""
        try:
            return float(np.percentile(returns, (1 - confidence_level) * 100))
        except Exception as e:
            logger.error(f"Error calculating historical VaR: {str(e)}")
            return 0.0
    
    def _calculate_parametric_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        try:
            mean_return = returns.mean()
            std_return = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            return float(mean_return + z_score * std_return)
        except Exception as e:
            logger.error(f"Error calculating parametric VaR: {str(e)}")
            return 0.0
    
    def _calculate_modified_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate modified VaR using Cornish-Fisher expansion."""
        try:
            mean_return = returns.mean()
            std_return = returns.std()
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Cornish-Fisher expansion
            z_score = stats.norm.ppf(1 - confidence_level)
            modified_z = z_score + (z_score**2 - 1) * skewness / 6 + (z_score**3 - 3*z_score) * (kurtosis - 3) / 24
            
            return float(mean_return + modified_z * std_return)
        except Exception as e:
            logger.error(f"Error calculating modified VaR: {str(e)}")
            return 0.0
    
    def _calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Expected Shortfall = E[X | X > VaR]
        """
        try:
            var_threshold = self._calculate_historical_var(returns, confidence_level)
            tail_returns = returns[returns <= var_threshold]
            
            if len(tail_returns) > 0:
                return float(tail_returns.mean())
            else:
                return float(var_threshold)
        except Exception as e:
            logger.error(f"Error calculating expected shortfall: {str(e)}")
            return 0.0
    
    def _get_regime_multiplier(self, regime: str) -> float:
        """Get regime-specific multiplier for VaR adjustment."""
        multipliers = {
            'high_volatility': 1.5,
            'normal_volatility': 1.0,
            'low_volatility': 0.8
        }
        return multipliers.get(regime, 1.0)
    
    def _calculate_var_confidence_interval(self, returns: pd.Series, confidence_level: float) -> Dict[str, float]:
        """Calculate confidence interval for VaR estimate."""
        try:
            # Bootstrap confidence interval
            n_bootstrap = 1000
            var_bootstrap = []
            
            for _ in range(n_bootstrap):
                bootstrap_sample = returns.sample(n=len(returns), replace=True)
                var_bootstrap.append(self._calculate_historical_var(bootstrap_sample, confidence_level))
            
            var_bootstrap = np.array(var_bootstrap)
            
            return {
                'lower_bound': float(np.percentile(var_bootstrap, 5)),
                'upper_bound': float(np.percentile(var_bootstrap, 95)),
                'standard_error': float(var_bootstrap.std())
            }
        except Exception as e:
            logger.error(f"Error calculating VaR confidence interval: {str(e)}")
            return {'lower_bound': 0.0, 'upper_bound': 0.0, 'standard_error': 0.0}
    
    def calculate_portfolio_risk_attribution(self, portfolio_weights: Dict[str, float], 
                                          covariance_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Portfolio-Level Risk Attribution.
        
        Args:
            portfolio_weights: Dictionary of asset weights
            covariance_matrix: Covariance matrix of returns
            
        Returns:
            Risk attribution analysis
        """
        try:
            # Convert weights to numpy array
            assets = list(portfolio_weights.keys())
            weights = np.array([portfolio_weights[asset] for asset in assets])
            
            # Calculate portfolio variance
            portfolio_variance = weights.T @ covariance_matrix.values @ weights
            
            # Calculate marginal contribution to risk
            marginal_contribution = covariance_matrix.values @ weights / np.sqrt(portfolio_variance)
            
            # Calculate component contribution to risk
            component_contribution = weights * marginal_contribution
            
            # Calculate percentage contribution
            percentage_contribution = component_contribution / np.sqrt(portfolio_variance) * 100
            
            # Risk attribution results
            risk_attribution = {
                'portfolio_variance': float(portfolio_variance),
                'portfolio_volatility': float(np.sqrt(portfolio_variance)),
                'marginal_contribution': {asset: float(mc) for asset, mc in zip(assets, marginal_contribution)},
                'component_contribution': {asset: float(cc) for asset, cc in zip(assets, component_contribution)},
                'percentage_contribution': {asset: float(pc) for asset, pc in zip(assets, percentage_contribution)}
            }
            
            # Factor risk attribution (if factor model is available)
            if hasattr(self, 'factor_model'):
                factor_attribution = self._calculate_factor_risk_attribution(portfolio_weights)
                risk_attribution['factor_attribution'] = factor_attribution
            
            return risk_attribution
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk attribution: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_factor_risk_attribution(self, portfolio_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate factor risk attribution."""
        try:
            # This would require factor model data
            # For now, return placeholder
            factors = ['market', 'size', 'value', 'momentum', 'volatility']
            factor_attribution = {}
            
            for factor in factors:
                factor_attribution[factor] = 0.0  # Placeholder
            
            return factor_attribution
            
        except Exception as e:
            logger.error(f"Error calculating factor risk attribution: {str(e)}")
            return {}
    
    def run_stress_tests(self, portfolio: Dict[str, float], 
                        market_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Run comprehensive stress tests with historical scenarios.
        
        Args:
            portfolio: Portfolio weights
            market_data: Market data for assets
            
        Returns:
            Stress test results
        """
        try:
            stress_results = {}
            
            for scenario_name, scenario_params in self.stress_scenarios.items():
                logger.info(f"Running stress test: {scenario_name}")
                
                # Apply stress scenario to market data
                stressed_data = self._apply_stress_scenario(market_data, scenario_params)
                
                # Calculate portfolio returns under stress
                portfolio_returns = self._calculate_stressed_portfolio_returns(portfolio, stressed_data)
                
                # Calculate risk metrics under stress
                stress_risk_metrics = self._calculate_stress_risk_metrics(portfolio_returns)
                
                stress_results[scenario_name] = {
                    'scenario_params': scenario_params,
                    'portfolio_return': float(portfolio_returns.mean()),
                    'portfolio_volatility': float(portfolio_returns.std()),
                    'max_drawdown': float(self._calculate_max_drawdown(portfolio_returns)),
                    'var_95': float(self._calculate_historical_var(portfolio_returns, 0.95)),
                    'var_99': float(self._calculate_historical_var(portfolio_returns, 0.99)),
                    'expected_shortfall': float(self._calculate_expected_shortfall(portfolio_returns, 0.95)),
                    'sharpe_ratio': float(self._calculate_sharpe_ratio(portfolio_returns)),
                    'worst_day_return': float(portfolio_returns.min()),
                    'best_day_return': float(portfolio_returns.max())
                }
            
            # Aggregate stress test results
            stress_results['aggregate'] = self._aggregate_stress_results(stress_results)
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error running stress tests: {str(e)}")
            return {'error': str(e)}
    
    def _apply_stress_scenario(self, market_data: Dict[str, pd.Series], 
                              scenario_params: Dict) -> Dict[str, pd.Series]:
        """Apply stress scenario to market data."""
        try:
            stressed_data = {}
            
            for asset, returns in market_data.items():
                if asset == 'BTCUSDT':
                    stress_multiplier = scenario_params['btc_drop']
                elif asset == 'ETHUSDT':
                    stress_multiplier = scenario_params['eth_drop']
                else:
                    # Apply correlation-based stress
                    stress_multiplier = scenario_params['btc_drop'] * scenario_params['market_correlation']
                
                # Apply stress to returns
                stressed_returns = returns * (1 + stress_multiplier)
                
                # Increase volatility
                volatility_multiplier = scenario_params['volatility_multiplier']
                stressed_returns = stressed_returns * volatility_multiplier
                
                stressed_data[asset] = stressed_returns
            
            return stressed_data
            
        except Exception as e:
            logger.error(f"Error applying stress scenario: {str(e)}")
            return market_data
    
    def _calculate_stressed_portfolio_returns(self, portfolio: Dict[str, float], 
                                           stressed_data: Dict[str, pd.Series]) -> pd.Series:
        """Calculate portfolio returns under stress scenario."""
        try:
            portfolio_returns = pd.Series(0.0, index=next(iter(stressed_data.values())).index)
            
            for asset, weight in portfolio.items():
                if asset in stressed_data:
                    portfolio_returns += weight * stressed_data[asset]
            
            return portfolio_returns
            
        except Exception as e:
            logger.error(f"Error calculating stressed portfolio returns: {str(e)}")
            return pd.Series([0.0])
    
    def _calculate_stress_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive risk metrics for stress scenario."""
        try:
            metrics = {
                'mean_return': float(returns.mean()),
                'volatility': float(returns.std()),
                'skewness': float(returns.skew()),
                'kurtosis': float(returns.kurtosis()),
                'var_95': float(self._calculate_historical_var(returns, 0.95)),
                'var_99': float(self._calculate_historical_var(returns, 0.99)),
                'expected_shortfall': float(self._calculate_expected_shortfall(returns, 0.95)),
                'max_drawdown': float(self._calculate_max_drawdown(returns)),
                'sharpe_ratio': float(self._calculate_sharpe_ratio(returns))
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating stress risk metrics: {str(e)}")
            return {}
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            return float(drawdown.min())
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        try:
            if returns.std() == 0:
                return 0.0
            return float(returns.mean() / returns.std() * np.sqrt(252))
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0
    
    def _aggregate_stress_results(self, stress_results: Dict) -> Dict[str, Any]:
        """Aggregate results across all stress scenarios."""
        try:
            aggregate = {
                'worst_case_scenario': None,
                'average_portfolio_return': 0.0,
                'average_portfolio_volatility': 0.0,
                'worst_var_95': float('inf'),
                'worst_expected_shortfall': float('inf'),
                'worst_max_drawdown': float('inf')
            }
            
            worst_return = float('inf')
            total_return = 0.0
            total_volatility = 0.0
            count = 0
            
            for scenario_name, results in stress_results.items():
                if scenario_name == 'aggregate':
                    continue
                
                count += 1
                total_return += results['portfolio_return']
                total_volatility += results['portfolio_volatility']
                
                if results['portfolio_return'] < worst_return:
                    worst_return = results['portfolio_return']
                    aggregate['worst_case_scenario'] = scenario_name
                
                if results['var_95'] < aggregate['worst_var_95']:
                    aggregate['worst_var_95'] = results['var_95']
                
                if results['expected_shortfall'] < aggregate['worst_expected_shortfall']:
                    aggregate['worst_expected_shortfall'] = results['expected_shortfall']
                
                if results['max_drawdown'] < aggregate['worst_max_drawdown']:
                    aggregate['worst_max_drawdown'] = results['max_drawdown']
            
            if count > 0:
                aggregate['average_portfolio_return'] = total_return / count
                aggregate['average_portfolio_volatility'] = total_volatility / count
            
            return aggregate
            
        except Exception as e:
            logger.error(f"Error aggregating stress results: {str(e)}")
            return {}
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        try:
            summary = {
                'risk_manager_status': 'active',
                'stress_scenarios_available': len(self.stress_scenarios),
                'regime_models_available': len(self.regime_models),
                'risk_history_count': len(self.risk_history),
                'last_risk_calculation': None,
                'risk_metrics': {
                    'total_risk_calculations': len(self.risk_history),
                    'stress_tests_run': 0,
                    'var_calculations': 0,
                    'expected_shortfall_calculations': 0
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {str(e)}")
            return {'error': str(e)} 