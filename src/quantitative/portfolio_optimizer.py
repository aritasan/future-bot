"""
WorldQuant Portfolio Optimizer Implementation
Advanced portfolio optimization with mean-variance, risk parity, factor neutral portfolios, and cross-asset hedging.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class WorldQuantPortfolioOptimizer:
    """
    WorldQuant-level portfolio optimizer with advanced optimization techniques.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize WorldQuant Portfolio Optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Optimization parameters
        self.optimization_params = {
            'mean_variance': {
                'risk_free_rate': 0.02,
                'target_return': 0.10,  # Reduced from 0.15
                'max_volatility': 0.25,
                'min_weight': 0.01,
                'max_weight': 0.40  # Increased from 0.30
            },
            'risk_parity': {
                'target_risk_contribution': 0.1,
                'min_weight': 0.01,
                'max_weight': 0.40,  # Increased from 0.30
                'risk_budget_method': 'equal'  # 'equal', 'volatility', 'custom'
            },
            'factor_neutral': {
                'factor_exposures': ['market', 'size', 'value', 'momentum', 'volatility', 'liquidity'],
                'max_factor_exposure': 0.2,  # Increased from 0.1
                'min_weight': 0.01,
                'max_weight': 0.40  # Increased from 0.30
            },
            'cross_asset_hedging': {
                'hedge_ratio_method': 'minimum_variance',  # 'minimum_variance', 'optimal_hedge'
                'correlation_threshold': 0.7,
                'hedge_cost_factor': 0.001
            }
        }
        
        # Portfolio constraints
        self.constraints = {
            'long_only': True,
            'leverage_limit': 1.0,
            'concentration_limit': 0.3,
            'sector_limit': 0.4,
            'geographic_limit': 0.5
        }
        
        # Performance tracking
        self.optimization_history = {}
        self.portfolio_metrics = {}
        self.rebalancing_dates = []
        
        logger.info("WorldQuantPortfolioOptimizer initialized")
    
    async def initialize(self) -> bool:
        """Initialize the portfolio optimizer."""
        try:
            logger.info("WorldQuantPortfolioOptimizer initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing WorldQuantPortfolioOptimizer: {str(e)}")
            return False
    
    async def optimize_mean_variance(self, returns: pd.DataFrame, 
                                   target_return: Optional[float] = None,
                                   max_volatility: Optional[float] = None) -> Dict[str, Any]:
        """
        Mean-variance optimization.
        
        Args:
            returns: DataFrame of asset returns
            target_return: Target portfolio return
            max_volatility: Maximum portfolio volatility
            
        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info("Performing mean-variance optimization...")
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Set parameters
            target_return = target_return or self.optimization_params['mean_variance']['target_return']
            max_volatility = max_volatility or self.optimization_params['mean_variance']['max_volatility']
            min_weight = self.optimization_params['mean_variance']['min_weight']
            max_weight = self.optimization_params['mean_variance']['max_weight']
            
            n_assets = len(expected_returns)
            
            # Objective function: minimize portfolio variance
            def objective(weights):
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                return portfolio_variance
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
                {'type': 'eq', 'fun': lambda x: np.dot(expected_returns, x) - target_return}  # Target return
            ]
            
            # Bounds
            bounds = [(min_weight, max_weight) for _ in range(n_assets)]
            
            # Initial weights (equal weight)
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective, 
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 2000, 'ftol': 1e-8, 'eps': 1e-8}  # Increased iterations and tolerance
            )
            
            if result.success:
                optimal_weights = result.x
                
                # Calculate portfolio metrics
                portfolio_return = np.dot(expected_returns, optimal_weights)
                portfolio_variance = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe_ratio = (portfolio_return - self.optimization_params['mean_variance']['risk_free_rate']) / portfolio_volatility
                
                # Risk decomposition
                risk_contributions = self._calculate_risk_contributions(optimal_weights, cov_matrix)
                
                optimization_result = {
                    'weights': dict(zip(returns.columns, optimal_weights)),
                    'portfolio_return': float(portfolio_return),
                    'portfolio_volatility': float(portfolio_volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'risk_contributions': risk_contributions,
                    'optimization_status': 'success',
                    'target_return': target_return,
                    'max_volatility': max_volatility
                }
                
                logger.info(f"Mean-variance optimization completed - Return: {portfolio_return:.4f}, Vol: {portfolio_volatility:.4f}, Sharpe: {sharpe_ratio:.3f}")
                return optimization_result
            else:
                logger.warning(f"Mean-variance optimization failed: {result.message}")
                return {'optimization_status': 'failed', 'message': result.message}
                
        except Exception as e:
            logger.error(f"Error in mean-variance optimization: {str(e)}")
            return {'optimization_status': 'error', 'message': str(e)}
    
    async def optimize_risk_parity(self, returns: pd.DataFrame, 
                                 target_risk_contribution: Optional[float] = None) -> Dict[str, Any]:
        """
        Risk parity optimization.
        
        Args:
            returns: DataFrame of asset returns
            target_risk_contribution: Target risk contribution per asset
            
        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info("Performing risk parity optimization...")
            
            # Calculate covariance matrix
            cov_matrix = returns.cov()
            n_assets = len(cov_matrix)
            
            # Set parameters
            target_risk_contribution = target_risk_contribution or self.optimization_params['risk_parity']['target_risk_contribution']
            min_weight = self.optimization_params['risk_parity']['min_weight']
            max_weight = self.optimization_params['risk_parity']['max_weight']
            
            # Objective function: minimize variance of risk contributions
            def objective(weights):
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                risk_contributions = weights * (np.dot(cov_matrix, weights)) / np.sqrt(portfolio_variance)
                risk_contribution_variance = np.var(risk_contributions)
                return risk_contribution_variance
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
            # Bounds
            bounds = [(min_weight, max_weight) for _ in range(n_assets)]
            
            # Initial weights (equal weight)
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective, 
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = result.x
                
                # Calculate portfolio metrics
                portfolio_variance = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                expected_returns = returns.mean()
                portfolio_return = np.dot(expected_returns, optimal_weights)
                
                # Risk contributions
                risk_contributions = self._calculate_risk_contributions(optimal_weights, cov_matrix)
                
                # Risk parity metrics
                risk_parity_score = 1 - np.std(list(risk_contributions.values()))
                
                optimization_result = {
                    'weights': dict(zip(returns.columns, optimal_weights)),
                    'portfolio_return': float(portfolio_return),
                    'portfolio_volatility': float(portfolio_volatility),
                    'risk_contributions': risk_contributions,
                    'risk_parity_score': float(risk_parity_score),
                    'optimization_status': 'success',
                    'target_risk_contribution': target_risk_contribution
                }
                
                logger.info(f"Risk parity optimization completed - Vol: {portfolio_volatility:.4f}, Parity Score: {risk_parity_score:.3f}")
                return optimization_result
            else:
                logger.warning(f"Risk parity optimization failed: {result.message}")
                return {'optimization_status': 'failed', 'message': result.message}
                
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {str(e)}")
            return {'optimization_status': 'error', 'message': str(e)}
    
    async def optimize_factor_neutral(self, returns: pd.DataFrame, 
                                    factor_exposures: Dict[str, Dict[str, float]],
                                    target_factors: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Factor neutral portfolio optimization.
        
        Args:
            returns: DataFrame of asset returns
            factor_exposures: Dictionary of factor exposures for each asset
            target_factors: List of factors to neutralize
            
        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info("Performing factor neutral optimization...")
            
            # Set parameters
            target_factors = target_factors or self.optimization_params['factor_neutral']['factor_exposures']
            max_factor_exposure = self.optimization_params['factor_neutral']['max_factor_exposure']
            min_weight = self.optimization_params['factor_neutral']['min_weight']
            max_weight = self.optimization_params['factor_neutral']['max_weight']
            
            n_assets = len(returns.columns)
            expected_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Objective function: maximize Sharpe ratio
            def objective(weights):
                portfolio_return = np.dot(expected_returns, weights)
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe_ratio = (portfolio_return - self.optimization_params['mean_variance']['risk_free_rate']) / portfolio_volatility
                return -sharpe_ratio  # Minimize negative Sharpe ratio
            
            # Factor neutrality constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
            # Add factor neutrality constraints
            for factor in target_factors:
                if factor in factor_exposures:
                    factor_exposures_factor = [factor_exposures[asset].get(factor, 0) for asset in returns.columns]
                    
                    # Constraint: factor exposure should be close to zero
                    constraints.append({
                        'type': 'ineq', 
                        'fun': lambda x, exposures=factor_exposures_factor: max_factor_exposure - abs(np.dot(x, exposures))
                    })
                    constraints.append({
                        'type': 'ineq', 
                        'fun': lambda x, exposures=factor_exposures_factor: max_factor_exposure + abs(np.dot(x, exposures))
                    })
            
            # Bounds
            bounds = [(min_weight, max_weight) for _ in range(n_assets)]
            
            # Initial weights (equal weight)
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective, 
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = result.x
                
                # Calculate portfolio metrics
                portfolio_return = np.dot(expected_returns, optimal_weights)
                portfolio_variance = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe_ratio = (portfolio_return - self.optimization_params['mean_variance']['risk_free_rate']) / portfolio_volatility
                
                # Factor exposures
                factor_exposures_portfolio = {}
                for factor in target_factors:
                    if factor in factor_exposures:
                        factor_exposures_factor = [factor_exposures[asset].get(factor, 0) for asset in returns.columns]
                        factor_exposure = np.dot(optimal_weights, factor_exposures_factor)
                        factor_exposures_portfolio[factor] = float(factor_exposure)
                
                # Risk contributions
                risk_contributions = self._calculate_risk_contributions(optimal_weights, cov_matrix)
                
                optimization_result = {
                    'weights': dict(zip(returns.columns, optimal_weights)),
                    'portfolio_return': float(portfolio_return),
                    'portfolio_volatility': float(portfolio_volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'factor_exposures': factor_exposures_portfolio,
                    'risk_contributions': risk_contributions,
                    'optimization_status': 'success',
                    'target_factors': target_factors
                }
                
                logger.info(f"Factor neutral optimization completed - Return: {portfolio_return:.4f}, Vol: {portfolio_volatility:.4f}, Sharpe: {sharpe_ratio:.3f}")
                return optimization_result
            else:
                logger.warning(f"Factor neutral optimization failed: {result.message}")
                return {'optimization_status': 'failed', 'message': result.message}
                
        except Exception as e:
            logger.error(f"Error in factor neutral optimization: {str(e)}")
            return {'optimization_status': 'error', 'message': str(e)}
    
    async def optimize_cross_asset_hedging(self, returns: pd.DataFrame, 
                                         hedge_assets: List[str],
                                         hedge_ratio_method: str = 'minimum_variance') -> Dict[str, Any]:
        """
        Cross-asset hedging optimization.
        
        Args:
            returns: DataFrame of asset returns
            hedge_assets: List of hedge assets
            hedge_ratio_method: Method for calculating hedge ratios
            
        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info("Performing cross-asset hedging optimization...")
            
            # Separate core assets and hedge assets
            core_assets = [col for col in returns.columns if col not in hedge_assets]
            core_returns = returns[core_assets]
            hedge_returns = returns[hedge_assets]
            
            # Calculate hedge ratios
            hedge_ratios = {}
            for core_asset in core_assets:
                for hedge_asset in hedge_assets:
                    if hedge_ratio_method == 'minimum_variance':
                        # Minimum variance hedge ratio
                        covariance = np.cov(returns[core_asset], returns[hedge_asset])[0, 1]
                        hedge_variance = np.var(returns[hedge_asset])
                        hedge_ratio = -covariance / hedge_variance if hedge_variance > 0 else 0
                    elif hedge_ratio_method == 'optimal_hedge':
                        # Optimal hedge ratio using regression
                        correlation = np.corrcoef(returns[core_asset], returns[hedge_asset])[0, 1]
                        volatility_ratio = np.std(returns[core_asset]) / np.std(returns[hedge_asset])
                        hedge_ratio = -correlation * volatility_ratio
                    else:
                        hedge_ratio = 0
                    
                    hedge_ratios[(core_asset, hedge_asset)] = hedge_ratio
            
            # Create hedged portfolio
            hedged_returns = core_returns.copy()
            for core_asset in core_assets:
                for hedge_asset in hedge_assets:
                    hedge_ratio = hedge_ratios.get((core_asset, hedge_asset), 0)
                    hedged_returns[core_asset] += hedge_ratio * hedge_returns[hedge_asset]
            
            # Optimize hedged portfolio
            expected_returns = hedged_returns.mean()
            cov_matrix = hedged_returns.cov()
            
            # Objective function: maximize Sharpe ratio
            def objective(weights):
                portfolio_return = np.dot(expected_returns, weights)
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe_ratio = (portfolio_return - self.optimization_params['mean_variance']['risk_free_rate']) / portfolio_volatility
                return -sharpe_ratio
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
            # Bounds
            n_core_assets = len(core_assets)
            min_weight = self.optimization_params['mean_variance']['min_weight']
            max_weight = self.optimization_params['mean_variance']['max_weight']
            bounds = [(min_weight, max_weight) for _ in range(n_core_assets)]
            
            # Initial weights
            initial_weights = np.array([1/n_core_assets] * n_core_assets)
            
            # Optimize
            result = minimize(
                objective, 
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = result.x
                
                # Calculate portfolio metrics
                portfolio_return = np.dot(expected_returns, optimal_weights)
                portfolio_variance = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe_ratio = (portfolio_return - self.optimization_params['mean_variance']['risk_free_rate']) / portfolio_volatility
                
                # Calculate hedge effectiveness
                original_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(core_returns.cov(), optimal_weights)))
                hedge_effectiveness = (original_volatility - portfolio_volatility) / original_volatility
                
                # Risk contributions
                risk_contributions = self._calculate_risk_contributions(optimal_weights, cov_matrix)
                
                optimization_result = {
                    'weights': dict(zip(core_assets, optimal_weights)),
                    'hedge_ratios': hedge_ratios,
                    'portfolio_return': float(portfolio_return),
                    'portfolio_volatility': float(portfolio_volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'hedge_effectiveness': float(hedge_effectiveness),
                    'risk_contributions': risk_contributions,
                    'optimization_status': 'success',
                    'hedge_ratio_method': hedge_ratio_method
                }
                
                logger.info(f"Cross-asset hedging optimization completed - Return: {portfolio_return:.4f}, Vol: {portfolio_volatility:.4f}, Hedge Effectiveness: {hedge_effectiveness:.3f}")
                return optimization_result
            else:
                logger.warning(f"Cross-asset hedging optimization failed: {result.message}")
                return {'optimization_status': 'failed', 'message': result.message}
                
        except Exception as e:
            logger.error(f"Error in cross-asset hedging optimization: {str(e)}")
            return {'optimization_status': 'error', 'message': str(e)}
    
    def _calculate_risk_contributions(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk contributions for each asset."""
        try:
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            risk_contributions = {}
            for i, asset in enumerate(cov_matrix.columns):
                marginal_risk = np.dot(cov_matrix.iloc[i], weights) / portfolio_volatility
                risk_contribution = weights[i] * marginal_risk
                risk_contributions[asset] = float(risk_contribution)
            
            return risk_contributions
            
        except Exception as e:
            logger.error(f"Error calculating risk contributions: {str(e)}")
            return {}
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio optimization summary."""
        try:
            summary = {
                'optimization_methods': ['mean_variance', 'risk_parity', 'factor_neutral', 'cross_asset_hedging'],
                'optimization_history': len(self.optimization_history),
                'portfolio_metrics': len(self.portfolio_metrics),
                'rebalancing_dates': len(self.rebalancing_dates)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {str(e)}")
            return {} 