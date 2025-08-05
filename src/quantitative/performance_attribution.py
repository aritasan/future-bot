#!/usr/bin/env python3
"""
Performance Attribution Module
WorldQuant Standards Implementation

Implements:
- Brinson Attribution
- Factor Attribution
- Risk Attribution
- Timing Attribution
- Performance Decomposition
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PerformanceAttribution:
    """
    Advanced Performance Attribution System following WorldQuant standards.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize Performance Attribution."""
        self.config = config or {}
        self.attribution_methods = {
            'brinson': BrinsonAttribution(config),
            'factor': FactorAttribution(config),
            'risk': RiskAttribution(config),
            'timing': TimingAttribution(config)
        }
        
        # Attribution parameters
        self.attribution_period = self.config.get('attribution_period', 'monthly')
        self.factor_model = self.config.get('factor_model', 'fama_french')
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        
        logger.info("Performance Attribution initialized")
    
    async def attribute_performance(self, portfolio_returns: pd.Series, 
                                  benchmark_returns: pd.Series,
                                  portfolio_weights: Dict[str, float],
                                  benchmark_weights: Dict[str, float],
                                  method: str = 'comprehensive') -> Dict[str, Any]:
        """
        Perform comprehensive performance attribution.
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            portfolio_weights: Portfolio weights
            benchmark_weights: Benchmark weights
            method: Attribution method ('brinson', 'factor', 'risk', 'timing', 'comprehensive')
            
        Returns:
            Performance attribution results
        """
        try:
            attribution_results = {}
            
            if method == 'comprehensive':
                # Perform all attribution methods
                logger.info("Performing comprehensive performance attribution...")
                
                # 1. Brinson Attribution
                brinson_attribution = await self.attribution_methods['brinson'].attribute_brinson(
                    portfolio_returns, benchmark_returns, portfolio_weights, benchmark_weights
                )
                attribution_results['brinson_attribution'] = brinson_attribution
                
                # 2. Factor Attribution
                factor_attribution = await self.attribution_methods['factor'].attribute_factors(
                    portfolio_returns, benchmark_returns, portfolio_weights, benchmark_weights
                )
                attribution_results['factor_attribution'] = factor_attribution
                
                # 3. Risk Attribution
                risk_attribution = await self.attribution_methods['risk'].attribute_risk(
                    portfolio_returns, benchmark_returns, portfolio_weights, benchmark_weights
                )
                attribution_results['risk_attribution'] = risk_attribution
                
                # 4. Timing Attribution
                timing_attribution = await self.attribution_methods['timing'].attribute_timing(
                    portfolio_returns, benchmark_returns, portfolio_weights, benchmark_weights
                )
                attribution_results['timing_attribution'] = timing_attribution
                
                # 5. Combined Attribution
                combined_attribution = self._combine_attribution_results(attribution_results)
                attribution_results['combined_attribution'] = combined_attribution
                
            else:
                # Perform specific attribution method
                if method in self.attribution_methods:
                    attribution_results[method] = await self.attribution_methods[method].attribute(
                        portfolio_returns, benchmark_returns, portfolio_weights, benchmark_weights
                    )
                else:
                    raise ValueError(f"Unknown attribution method: {method}")
            
            return attribution_results
            
        except Exception as e:
            logger.error(f"Error in performance attribution: {str(e)}")
            return {'error': str(e)}
    
    def _combine_attribution_results(self, attribution_results: Dict) -> Dict[str, Any]:
        """
        Combine all attribution results into a unified analysis.
        """
        try:
            combined = {
                'total_attribution': 0.0,
                'allocation_effect': 0.0,
                'selection_effect': 0.0,
                'interaction_effect': 0.0,
                'factor_effect': 0.0,
                'risk_effect': 0.0,
                'timing_effect': 0.0,
                'residual_effect': 0.0
            }
            
            # Extract effects from each attribution method
            if 'brinson_attribution' in attribution_results:
                brinson = attribution_results['brinson_attribution']
                combined['allocation_effect'] = brinson.get('allocation_effect', 0.0)
                combined['selection_effect'] = brinson.get('selection_effect', 0.0)
                combined['interaction_effect'] = brinson.get('interaction_effect', 0.0)
            
            if 'factor_attribution' in attribution_results:
                factor = attribution_results['factor_attribution']
                combined['factor_effect'] = factor.get('total_factor_effect', 0.0)
            
            if 'risk_attribution' in attribution_results:
                risk = attribution_results['risk_attribution']
                combined['risk_effect'] = risk.get('total_risk_effect', 0.0)
            
            if 'timing_attribution' in attribution_results:
                timing = attribution_results['timing_attribution']
                combined['timing_effect'] = timing.get('total_timing_effect', 0.0)
            
            # Calculate total attribution
            combined['total_attribution'] = (
                combined['allocation_effect'] +
                combined['selection_effect'] +
                combined['interaction_effect'] +
                combined['factor_effect'] +
                combined['risk_effect'] +
                combined['timing_effect']
            )
            
            # Calculate residual
            portfolio_return = attribution_results.get('portfolio_return', 0.0)
            benchmark_return = attribution_results.get('benchmark_return', 0.0)
            combined['residual_effect'] = (portfolio_return - benchmark_return) - combined['total_attribution']
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining attribution results: {str(e)}")
            return {'error': str(e)}

class BrinsonAttribution:
    """Brinson Attribution Analysis."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        logger.info("Brinson Attribution initialized")
    
    async def attribute_brinson(self, portfolio_returns: pd.Series, 
                              benchmark_returns: pd.Series,
                              portfolio_weights: Dict[str, float],
                              benchmark_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Perform Brinson attribution analysis.
        """
        try:
            # Calculate attribution effects
            allocation_effect = 0.0
            selection_effect = 0.0
            interaction_effect = 0.0
            
            # Get common assets
            common_assets = set(portfolio_weights.keys()) & set(benchmark_weights.keys())
            
            for asset in common_assets:
                portfolio_weight = portfolio_weights.get(asset, 0.0)
                benchmark_weight = benchmark_weights.get(asset, 0.0)
                portfolio_return = portfolio_returns.get(asset, 0.0)
                benchmark_return = benchmark_returns.get(asset, 0.0)
                
                # Allocation effect: (portfolio_weight - benchmark_weight) * benchmark_return
                allocation_effect += (portfolio_weight - benchmark_weight) * benchmark_return
                
                # Selection effect: benchmark_weight * (portfolio_return - benchmark_return)
                selection_effect += benchmark_weight * (portfolio_return - benchmark_return)
                
                # Interaction effect: (portfolio_weight - benchmark_weight) * (portfolio_return - benchmark_return)
                interaction_effect += (portfolio_weight - benchmark_weight) * (portfolio_return - benchmark_return)
            
            # Calculate total effects
            total_effect = allocation_effect + selection_effect + interaction_effect
            
            return {
                'allocation_effect': float(allocation_effect),
                'selection_effect': float(selection_effect),
                'interaction_effect': float(interaction_effect),
                'total_effect': float(total_effect),
                'attribution_method': 'brinson'
            }
            
        except Exception as e:
            logger.error(f"Error in Brinson attribution: {str(e)}")
            return {'error': str(e)}

class FactorAttribution:
    """Factor Attribution Analysis."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        logger.info("Factor Attribution initialized")
    
    async def attribute_factors(self, portfolio_returns: pd.Series, 
                              benchmark_returns: pd.Series,
                              portfolio_weights: Dict[str, float],
                              benchmark_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Perform factor attribution analysis.
        """
        try:
            # Define factors (market, size, value, momentum, etc.)
            factors = ['market', 'size', 'value', 'momentum', 'volatility', 'quality']
            
            factor_effects = {}
            total_factor_effect = 0.0
            
            for factor in factors:
                # Calculate factor exposure difference
                portfolio_exposure = self._calculate_factor_exposure(portfolio_weights, factor)
                benchmark_exposure = self._calculate_factor_exposure(benchmark_weights, factor)
                
                # Calculate factor return
                factor_return = self._calculate_factor_return(factor)
                
                # Factor effect: (portfolio_exposure - benchmark_exposure) * factor_return
                factor_effect = (portfolio_exposure - benchmark_exposure) * factor_return
                factor_effects[factor] = float(factor_effect)
                total_factor_effect += factor_effect
            
            return {
                'factor_effects': factor_effects,
                'total_factor_effect': float(total_factor_effect),
                'attribution_method': 'factor'
            }
            
        except Exception as e:
            logger.error(f"Error in factor attribution: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_factor_exposure(self, weights: Dict[str, float], factor: str) -> float:
        """
        Calculate factor exposure for given weights.
        """
        try:
            # Placeholder for factor exposure calculation
            # In production, this would use actual factor data
            
            # Mock factor exposures
            factor_exposures = {
                'market': 1.0,
                'size': 0.5,
                'value': 0.3,
                'momentum': 0.4,
                'volatility': 0.2,
                'quality': 0.6
            }
            
            # Weighted average factor exposure
            total_weight = sum(weights.values())
            if total_weight > 0:
                return factor_exposures.get(factor, 0.0)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating factor exposure: {str(e)}")
            return 0.0
    
    def _calculate_factor_return(self, factor: str) -> float:
        """
        Calculate factor return.
        """
        try:
            # Placeholder for factor return calculation
            # In production, this would use actual factor return data
            
            # Mock factor returns
            factor_returns = {
                'market': 0.05,
                'size': 0.02,
                'value': 0.03,
                'momentum': 0.04,
                'volatility': -0.01,
                'quality': 0.02
            }
            
            return factor_returns.get(factor, 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating factor return: {str(e)}")
            return 0.0

class RiskAttribution:
    """Risk Attribution Analysis."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        logger.info("Risk Attribution initialized")
    
    async def attribute_risk(self, portfolio_returns: pd.Series, 
                           benchmark_returns: pd.Series,
                           portfolio_weights: Dict[str, float],
                           benchmark_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Perform risk attribution analysis.
        """
        try:
            # Calculate risk metrics
            portfolio_volatility = portfolio_returns.std()
            benchmark_volatility = benchmark_returns.std()
            
            # Calculate VaR
            portfolio_var = self._calculate_var(portfolio_returns)
            benchmark_var = self._calculate_var(benchmark_returns)
            
            # Calculate beta
            portfolio_beta = self._calculate_beta(portfolio_returns, benchmark_returns)
            
            # Risk attribution effects
            volatility_effect = portfolio_volatility - benchmark_volatility
            var_effect = portfolio_var - benchmark_var
            beta_effect = (portfolio_beta - 1.0) * benchmark_returns.mean()
            
            total_risk_effect = volatility_effect + var_effect + beta_effect
            
            return {
                'volatility_effect': float(volatility_effect),
                'var_effect': float(var_effect),
                'beta_effect': float(beta_effect),
                'total_risk_effect': float(total_risk_effect),
                'portfolio_volatility': float(portfolio_volatility),
                'benchmark_volatility': float(benchmark_volatility),
                'portfolio_var': float(portfolio_var),
                'benchmark_var': float(benchmark_var),
                'portfolio_beta': float(portfolio_beta),
                'attribution_method': 'risk'
            }
            
        except Exception as e:
            logger.error(f"Error in risk attribution: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk.
        """
        try:
            return float(np.percentile(returns, (1 - confidence_level) * 100))
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return 0.0
    
    def _calculate_beta(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate portfolio beta.
        """
        try:
            if len(portfolio_returns) != len(benchmark_returns):
                return 1.0
            
            # Calculate covariance and variance
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            
            if benchmark_variance > 0:
                return float(covariance / benchmark_variance)
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error calculating beta: {str(e)}")
            return 1.0

class TimingAttribution:
    """Timing Attribution Analysis."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        logger.info("Timing Attribution initialized")
    
    async def attribute_timing(self, portfolio_returns: pd.Series, 
                             benchmark_returns: pd.Series,
                             portfolio_weights: Dict[str, float],
                             benchmark_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Perform timing attribution analysis.
        """
        try:
            # Calculate timing effects
            market_timing_effect = self._calculate_market_timing(portfolio_returns, benchmark_returns)
            sector_timing_effect = self._calculate_sector_timing(portfolio_weights, benchmark_weights)
            style_timing_effect = self._calculate_style_timing(portfolio_weights, benchmark_weights)
            
            total_timing_effect = market_timing_effect + sector_timing_effect + style_timing_effect
            
            return {
                'market_timing_effect': float(market_timing_effect),
                'sector_timing_effect': float(sector_timing_effect),
                'style_timing_effect': float(style_timing_effect),
                'total_timing_effect': float(total_timing_effect),
                'attribution_method': 'timing'
            }
            
        except Exception as e:
            logger.error(f"Error in timing attribution: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_market_timing(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate market timing effect.
        """
        try:
            # Market timing: ability to increase beta in up markets and decrease in down markets
            up_market_mask = benchmark_returns > 0
            down_market_mask = benchmark_returns < 0
            
            if up_market_mask.any() and down_market_mask.any():
                up_market_timing = portfolio_returns[up_market_mask].mean() - benchmark_returns[up_market_mask].mean()
                down_market_timing = portfolio_returns[down_market_mask].mean() - benchmark_returns[down_market_mask].mean()
                return float(up_market_timing + down_market_timing)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating market timing: {str(e)}")
            return 0.0
    
    def _calculate_sector_timing(self, portfolio_weights: Dict[str, float], 
                               benchmark_weights: Dict[str, float]) -> float:
        """
        Calculate sector timing effect.
        """
        try:
            # Placeholder for sector timing calculation
            # In production, this would analyze sector allocation timing
            
            # Mock sector timing effect
            return 0.02
            
        except Exception as e:
            logger.error(f"Error calculating sector timing: {str(e)}")
            return 0.0
    
    def _calculate_style_timing(self, portfolio_weights: Dict[str, float], 
                              benchmark_weights: Dict[str, float]) -> float:
        """
        Calculate style timing effect.
        """
        try:
            # Placeholder for style timing calculation
            # In production, this would analyze style allocation timing
            
            # Mock style timing effect
            return 0.01
            
        except Exception as e:
            logger.error(f"Error calculating style timing: {str(e)}")
            return 0.0
    
    def get_attribution_summary(self) -> Dict[str, Any]:
        """Get comprehensive attribution summary."""
        try:
            summary = {
                'attribution_engine_status': 'active',
                'available_methods': list(self.attribution_methods.keys()),
                'total_attributions': 0,
                'last_attribution': None,
                'attribution_period': self.attribution_period
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting attribution summary: {str(e)}")
            return {'error': str(e)} 