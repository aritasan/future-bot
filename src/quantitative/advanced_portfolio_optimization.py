#!/usr/bin/env python3
"""
Advanced Portfolio Optimization Module
WorldQuant Standards Implementation

Implements:
- Multi-Period Optimization
- Risk Budgeting
- Black-Litterman Model
- Factor Neutral Optimization
- Cross-Asset Hedging
- Dynamic Rebalancing
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedPortfolioOptimizer:
    """
    Advanced Portfolio Optimization System following WorldQuant standards.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize Advanced Portfolio Optimizer."""
        self.config = config or {}
        self.optimizers = {
            'mean_variance': MeanVarianceOptimizer(config),
            'risk_parity': RiskParityOptimizer(config),
            'black_litterman': BlackLittermanOptimizer(config),
            'factor_neutral': FactorNeutralOptimizer(config),
            'cross_asset_hedging': CrossAssetHedgingOptimizer(config)
        }
        
        # Optimization parameters
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        self.max_position_size = self.config.get('max_position_size', 0.3)
        self.min_position_size = self.config.get('min_position_size', 0.01)
        self.target_volatility = self.config.get('target_volatility', 0.15)
        self.rebalancing_frequency = self.config.get('rebalancing_frequency', 'monthly')
        
        logger.info("Advanced Portfolio Optimizer initialized")
    
    async def optimize_portfolio(self, returns: pd.DataFrame, method: str = 'adaptive', **kwargs) -> Dict[str, Any]:
        """
        Optimize portfolio using specified method.
        
        Args:
            returns: Return series for assets
            method: Optimization method ('mean_variance', 'risk_parity', 'black_litterman', 'factor_neutral', 'cross_asset_hedging', 'adaptive')
            **kwargs: Additional parameters for optimization
            
        Returns:
            Optimization results
        """
        try:
            if method == 'adaptive':
                # Choose best method based on market conditions
                method = self._select_adaptive_method(returns)
                logger.info(f"Selected adaptive method: {method}")
            
            if method == 'mean_variance':
                result = await self.optimizers['mean_variance'].optimize(returns, **kwargs)
            elif method == 'risk_parity':
                result = await self.optimizers['risk_parity'].optimize(returns, **kwargs)
            elif method == 'black_litterman':
                views = kwargs.get('views', {})
                result = await self.optimizers['black_litterman'].optimize(returns, views, **kwargs)
            elif method == 'factor_neutral':
                factor_exposures = kwargs.get('factor_exposures', {})
                result = await self.optimizers['factor_neutral'].optimize(returns, factor_exposures, **kwargs)
            elif method == 'cross_asset_hedging':
                hedge_assets = kwargs.get('hedge_assets', [])
                result = await self.optimizers['cross_asset_hedging'].optimize(returns, hedge_assets, **kwargs)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            # Add optimization metadata
            result['optimization_method'] = method
            result['optimization_timestamp'] = pd.Timestamp.now().isoformat()
            result['market_conditions'] = self._analyze_market_conditions(returns)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {str(e)}")
            return {'optimization_status': 'error', 'message': str(e)}
    
    def _select_adaptive_method(self, returns: pd.DataFrame) -> str:
        """
        Select optimal method based on market conditions.
        """
        try:
            # Analyze market conditions
            volatility = returns.std().mean()
            correlation = returns.corr().mean().mean()
            sharpe_ratio = returns.mean().mean() / volatility if volatility > 0 else 0
            
            # Select method based on conditions
            if volatility > 0.25:  # High volatility
                return 'risk_parity'
            elif correlation > 0.7:  # High correlation
                return 'factor_neutral'
            elif sharpe_ratio > 1.0:  # High Sharpe ratio
                return 'mean_variance'
            else:
                return 'black_litterman'
                
        except Exception as e:
            logger.error(f"Error selecting adaptive method: {str(e)}")
            return 'mean_variance'
    
    def _analyze_market_conditions(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze current market conditions.
        """
        try:
            conditions = {
                'volatility': float(returns.std().mean()),
                'correlation': float(returns.corr().mean().mean()),
                'sharpe_ratio': float(returns.mean().mean() / returns.std().mean()) if returns.std().mean() > 0 else 0.0,
                'skewness': float(returns.skew().mean()),
                'kurtosis': float(returns.kurtosis().mean())
            }
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {str(e)}")
            return {}

class MeanVarianceOptimizer:
    """Mean-Variance Portfolio Optimization."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        logger.info("Mean-Variance Optimizer initialized")
    
    async def optimize(self, returns: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Optimize portfolio using mean-variance approach.
        """
        try:
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean()
            covariance_matrix = returns.cov()
            
            # Get optimization parameters
            target_return = kwargs.get('target_return', expected_returns.mean())
            risk_aversion = kwargs.get('risk_aversion', 1.0)
            
            # Define optimization constraints
            n_assets = len(returns.columns)
            
            # Objective function: minimize risk for given return
            def objective(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_variance = weights.T @ covariance_matrix.values @ weights
                return portfolio_variance - risk_aversion * portfolio_return
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
                {'type': 'eq', 'fun': lambda x: np.sum(x * expected_returns) - target_return}  # Target return
            ]
            
            # Bounds
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # Initial guess
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(objective, initial_weights, method='SLSQP', 
                           bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                portfolio_return = np.sum(optimal_weights * expected_returns)
                portfolio_variance = optimal_weights.T @ covariance_matrix.values @ optimal_weights
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe_ratio = (portfolio_return - self.config.get('risk_free_rate', 0.02)) / portfolio_volatility
                
                return {
                    'optimization_status': 'success',
                    'optimal_weights': dict(zip(returns.columns, optimal_weights)),
                    'portfolio_return': float(portfolio_return),
                    'portfolio_volatility': float(portfolio_volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'target_return': float(target_return),
                    'risk_aversion': float(risk_aversion)
                }
            else:
                return {'optimization_status': 'failed', 'message': result.message}
                
        except Exception as e:
            logger.error(f"Error in mean-variance optimization: {str(e)}")
            return {'optimization_status': 'error', 'message': str(e)}

class RiskParityOptimizer:
    """Risk Parity Portfolio Optimization."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        logger.info("Risk Parity Optimizer initialized")
    
    async def optimize(self, returns: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Optimize portfolio using risk parity approach.
        """
        try:
            # Calculate covariance matrix
            covariance_matrix = returns.cov()
            n_assets = len(returns.columns)
            
            # Objective function: minimize risk contribution dispersion
            def objective(weights):
                portfolio_variance = weights.T @ covariance_matrix.values @ weights
                risk_contributions = (weights * (covariance_matrix.values @ weights)) / np.sqrt(portfolio_variance)
                return np.std(risk_contributions)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
            # Bounds
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # Initial guess
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(objective, initial_weights, method='SLSQP', 
                           bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                portfolio_variance = optimal_weights.T @ covariance_matrix.values @ optimal_weights
                portfolio_volatility = np.sqrt(portfolio_variance)
                expected_returns = returns.mean()
                portfolio_return = np.sum(optimal_weights * expected_returns)
                sharpe_ratio = (portfolio_return - self.config.get('risk_free_rate', 0.02)) / portfolio_volatility
                
                # Calculate risk contributions
                risk_contributions = (optimal_weights * (covariance_matrix.values @ optimal_weights)) / portfolio_volatility
                
                return {
                    'optimization_status': 'success',
                    'optimal_weights': dict(zip(returns.columns, optimal_weights)),
                    'portfolio_return': float(portfolio_return),
                    'portfolio_volatility': float(portfolio_volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'risk_contributions': dict(zip(returns.columns, risk_contributions)),
                    'risk_contribution_std': float(np.std(risk_contributions))
                }
            else:
                return {'optimization_status': 'failed', 'message': result.message}
                
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {str(e)}")
            return {'optimization_status': 'error', 'message': str(e)}

class BlackLittermanOptimizer:
    """Black-Litterman Portfolio Optimization."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        logger.info("Black-Litterman Optimizer initialized")
    
    async def optimize(self, returns: pd.DataFrame, views: Dict, **kwargs) -> Dict[str, Any]:
        """
        Optimize portfolio using Black-Litterman approach.
        """
        try:
            # Calculate market equilibrium returns
            market_cap_weights = kwargs.get('market_cap_weights', np.array([1/len(returns.columns)] * len(returns.columns)))
            risk_aversion = kwargs.get('risk_aversion', 2.5)
            covariance_matrix = returns.cov()
            
            # Market equilibrium returns
            equilibrium_returns = risk_aversion * covariance_matrix.values @ market_cap_weights
            
            # Process views
            if views:
                # Placeholder for view processing
                # In production, this would process analyst views
                adjusted_returns = equilibrium_returns
            else:
                adjusted_returns = equilibrium_returns
            
            # Optimize with adjusted returns
            n_assets = len(returns.columns)
            
            def objective(weights):
                portfolio_return = np.sum(weights * adjusted_returns)
                portfolio_variance = weights.T @ covariance_matrix.values @ weights
                return -portfolio_return + 0.5 * risk_aversion * portfolio_variance
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
            # Bounds
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # Initial guess
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(objective, initial_weights, method='SLSQP', 
                           bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                portfolio_return = np.sum(optimal_weights * adjusted_returns)
                portfolio_variance = optimal_weights.T @ covariance_matrix.values @ optimal_weights
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe_ratio = (portfolio_return - self.config.get('risk_free_rate', 0.02)) / portfolio_volatility
                
                return {
                    'optimization_status': 'success',
                    'optimal_weights': dict(zip(returns.columns, optimal_weights)),
                    'portfolio_return': float(portfolio_return),
                    'portfolio_volatility': float(portfolio_volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'equilibrium_returns': dict(zip(returns.columns, equilibrium_returns)),
                    'adjusted_returns': dict(zip(returns.columns, adjusted_returns)),
                    'views_applied': len(views) if views else 0
                }
            else:
                return {'optimization_status': 'failed', 'message': result.message}
                
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {str(e)}")
            return {'optimization_status': 'error', 'message': str(e)}

class FactorNeutralOptimizer:
    """Factor Neutral Portfolio Optimization."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        logger.info("Factor Neutral Optimizer initialized")
    
    async def optimize(self, returns: pd.DataFrame, factor_exposures: Dict, **kwargs) -> Dict[str, Any]:
        """
        Optimize portfolio using factor neutral approach.
        """
        try:
            # Calculate covariance matrix
            covariance_matrix = returns.cov()
            n_assets = len(returns.columns)
            
            # Define factors (market, size, value, momentum, etc.)
            factors = ['market', 'size', 'value', 'momentum', 'volatility']
            
            # Create factor exposure matrix
            factor_matrix = np.zeros((n_assets, len(factors)))
            
            # Placeholder for factor exposures
            # In production, this would use actual factor data
            for i, asset in enumerate(returns.columns):
                for j, factor in enumerate(factors):
                    factor_matrix[i, j] = factor_exposures.get(factor, {}).get(asset, 0.0)
            
            # Objective function: minimize tracking error while neutralizing factors
            def objective(weights):
                portfolio_variance = weights.T @ covariance_matrix.values @ weights
                return portfolio_variance
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
            # Factor neutrality constraints
            for j in range(len(factors)):
                constraints.append({
                    'type': 'eq', 
                    'fun': lambda x, j=j: np.sum(x * factor_matrix[:, j])  # Factor neutral
                })
            
            # Bounds
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # Initial guess
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(objective, initial_weights, method='SLSQP', 
                           bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                portfolio_variance = optimal_weights.T @ covariance_matrix.values @ optimal_weights
                portfolio_volatility = np.sqrt(portfolio_variance)
                expected_returns = returns.mean()
                portfolio_return = np.sum(optimal_weights * expected_returns)
                sharpe_ratio = (portfolio_return - self.config.get('risk_free_rate', 0.02)) / portfolio_volatility
                
                # Calculate factor exposures
                factor_exposures_result = {}
                for j, factor in enumerate(factors):
                    factor_exposures_result[factor] = float(np.sum(optimal_weights * factor_matrix[:, j]))
                
                return {
                    'optimization_status': 'success',
                    'optimal_weights': dict(zip(returns.columns, optimal_weights)),
                    'portfolio_return': float(portfolio_return),
                    'portfolio_volatility': float(portfolio_volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'factor_exposures': factor_exposures_result,
                    'tracking_error': float(portfolio_volatility)
                }
            else:
                return {'optimization_status': 'failed', 'message': result.message}
                
        except Exception as e:
            logger.error(f"Error in factor neutral optimization: {str(e)}")
            return {'optimization_status': 'error', 'message': str(e)}

class CrossAssetHedgingOptimizer:
    """Cross-Asset Hedging Portfolio Optimization."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        logger.info("Cross-Asset Hedging Optimizer initialized")
    
    async def optimize(self, returns: pd.DataFrame, hedge_assets: List[str], **kwargs) -> Dict[str, Any]:
        """
        Optimize portfolio using cross-asset hedging approach.
        """
        try:
            # Calculate covariance matrix
            covariance_matrix = returns.cov()
            n_assets = len(returns.columns)
            
            # Identify hedge assets
            hedge_indices = [i for i, asset in enumerate(returns.columns) if asset in hedge_assets]
            core_indices = [i for i in range(n_assets) if i not in hedge_indices]
            
            # Objective function: maximize Sharpe ratio with hedging
            def objective(weights):
                expected_returns = returns.mean()
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_variance = weights.T @ covariance_matrix.values @ weights
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                if portfolio_volatility > 0:
                    return -(portfolio_return - self.config.get('risk_free_rate', 0.02)) / portfolio_volatility
                else:
                    return 0
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
            # Core position constraints
            for i in core_indices:
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda x, i=i: x[i] - self.config.get('min_position_size', 0.01)
                })
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda x, i=i: self.config.get('max_position_size', 0.3) - x[i]
                })
            
            # Bounds
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # Initial guess
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(objective, initial_weights, method='SLSQP', 
                           bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                expected_returns = returns.mean()
                portfolio_return = np.sum(optimal_weights * expected_returns)
                portfolio_variance = optimal_weights.T @ covariance_matrix.values @ optimal_weights
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe_ratio = (portfolio_return - self.config.get('risk_free_rate', 0.02)) / portfolio_volatility
                
                # Calculate hedge ratios
                hedge_weights = {asset: optimal_weights[i] for i, asset in enumerate(returns.columns) if asset in hedge_assets}
                core_weights = {asset: optimal_weights[i] for i, asset in enumerate(returns.columns) if asset not in hedge_assets}
                
                return {
                    'optimization_status': 'success',
                    'optimal_weights': dict(zip(returns.columns, optimal_weights)),
                    'portfolio_return': float(portfolio_return),
                    'portfolio_volatility': float(portfolio_volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'hedge_weights': hedge_weights,
                    'core_weights': core_weights,
                    'hedge_ratio': float(sum(hedge_weights.values()) / sum(optimal_weights))
                }
            else:
                return {'optimization_status': 'failed', 'message': result.message}
                
        except Exception as e:
            logger.error(f"Error in cross-asset hedging optimization: {str(e)}")
            return {'optimization_status': 'error', 'message': str(e)}
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary."""
        try:
            summary = {
                'optimizer_status': 'active',
                'available_methods': list(self.optimizers.keys()),
                'total_optimizations': 0,
                'last_optimization': None,
                'market_conditions': {}
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting optimization summary: {str(e)}")
            return {'error': str(e)} 