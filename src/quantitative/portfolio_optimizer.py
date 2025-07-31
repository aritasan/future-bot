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
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict, deque
import json
from .performance_tracker import WorldQuantPerformanceTracker

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
        
        # Performance tracking - WorldQuant Standards
        self.performance_tracker = WorldQuantPerformanceTracker()
        
        # Portfolio state tracking
        self.portfolio_state = {
            'current_weights': {},
            'target_weights': {},
            'last_rebalancing': None,
            'total_trades': 0,
            'total_volume': 0.0,
            'rebalancing_count': 0
        }
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_task = None
        
        logger.info("WorldQuantPortfolioOptimizer initialized")
    
    async def initialize(self) -> bool:
        """Initialize the portfolio optimizer."""
        try:
            # Initialize performance tracker
            await self.performance_tracker.initialize()
            
            # Start real-time monitoring
            await self.start_performance_monitoring()
            
            logger.info("WorldQuantPortfolioOptimizer initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing WorldQuantPortfolioOptimizer: {str(e)}")
            return False
    
    async def start_performance_monitoring(self) -> None:
        """Start real-time performance monitoring."""
        try:
            if not self.monitoring_active:
                self.monitoring_active = True
                self.monitoring_task = asyncio.create_task(self._monitor_performance_loop())
                logger.info("Real-time performance monitoring started")
        except Exception as e:
            logger.error(f"Error starting performance monitoring: {str(e)}")
    
    async def stop_performance_monitoring(self) -> None:
        """Stop real-time performance monitoring."""
        try:
            if self.monitoring_active:
                self.monitoring_active = False
                if self.monitoring_task:
                    self.monitoring_task.cancel()
                    try:
                        await self.monitoring_task
                    except asyncio.CancelledError:
                        pass
                logger.info("Real-time performance monitoring stopped")
        except Exception as e:
            logger.error(f"Error stopping performance monitoring: {str(e)}")
    
    async def _monitor_performance_loop(self) -> None:
        """Real-time performance monitoring loop."""
        try:
            while self.monitoring_active:
                try:
                    # Update performance metrics
                    await self.performance_tracker.update_metrics()
                    
                    # Check for alerts
                    alerts = await self.performance_tracker.check_alerts()
                    if alerts:
                        await self._handle_performance_alerts(alerts)
                    
                    # Log performance summary
                    await self._log_performance_summary()
                    
                    # Wait for next update (30 seconds)
                    await asyncio.sleep(30)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in performance monitoring loop: {str(e)}")
                    await asyncio.sleep(60)  # Wait longer on error
                    
        except Exception as e:
            logger.error(f"Fatal error in performance monitoring: {str(e)}")
    
    async def _handle_performance_alerts(self, alerts: List[Dict]) -> None:
        """Handle performance alerts."""
        try:
            for alert in alerts:
                alert_type = alert.get('type')
                alert_message = alert.get('message')
                alert_level = alert.get('level', 'warning')
                
                logger.warning(f"Performance Alert [{alert_level.upper()}]: {alert_type} - {alert_message}")
                
                # Handle specific alert types
                if alert_type == 'drawdown_exceeded':
                    await self._handle_drawdown_alert(alert)
                elif alert_type == 'volatility_spike':
                    await self._handle_volatility_alert(alert)
                elif alert_type == 'sharpe_decline':
                    await self._handle_sharpe_alert(alert)
                elif alert_type == 'rebalancing_needed':
                    await self._handle_rebalancing_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error handling performance alerts: {str(e)}")
    
    async def _handle_drawdown_alert(self, alert: Dict) -> None:
        """Handle drawdown alert."""
        try:
            current_drawdown = alert.get('current_drawdown', 0)
            threshold = alert.get('threshold', 0.1)
            
            logger.warning(f"Drawdown Alert: Current {current_drawdown:.2%}, Threshold {threshold:.2%}")
            
            # Implement risk reduction strategy
            if current_drawdown > threshold * 1.5:  # 50% above threshold
                await self._implement_risk_reduction()
                
        except Exception as e:
            logger.error(f"Error handling drawdown alert: {str(e)}")
    
    async def _handle_volatility_alert(self, alert: Dict) -> None:
        """Handle volatility spike alert."""
        try:
            current_volatility = alert.get('current_volatility', 0)
            threshold = alert.get('threshold', 0.25)
            
            logger.warning(f"Volatility Alert: Current {current_volatility:.2%}, Threshold {threshold:.2%}")
            
            # Implement volatility management
            if current_volatility > threshold * 1.2:  # 20% above threshold
                await self._implement_volatility_management()
                
        except Exception as e:
            logger.error(f"Error handling volatility alert: {str(e)}")
    
    async def _handle_sharpe_alert(self, alert: Dict) -> None:
        """Handle Sharpe ratio decline alert."""
        try:
            current_sharpe = alert.get('current_sharpe', 0)
            threshold = alert.get('threshold', 0.5)
            
            logger.warning(f"Sharpe Alert: Current {current_sharpe:.3f}, Threshold {threshold:.3f}")
            
            # Implement performance improvement strategy
            if current_sharpe < threshold * 0.8:  # 20% below threshold
                await self._implement_performance_improvement()
                
        except Exception as e:
            logger.error(f"Error handling Sharpe alert: {str(e)}")
    
    async def _handle_rebalancing_alert(self, alert: Dict) -> None:
        """Handle rebalancing needed alert."""
        try:
            weight_drift = alert.get('weight_drift', 0)
            threshold = alert.get('threshold', 0.1)
            
            logger.warning(f"Rebalancing Alert: Weight drift {weight_drift:.2%}, Threshold {threshold:.2%}")
            
            # Trigger rebalancing
            if weight_drift > threshold:
                await self._trigger_rebalancing()
                
        except Exception as e:
            logger.error(f"Error handling rebalancing alert: {str(e)}")
    
    async def _implement_risk_reduction(self) -> None:
        """Implement risk reduction strategy."""
        try:
            logger.info("Implementing risk reduction strategy...")
            
            # Reduce position sizes by 20%
            reduction_factor = 0.8
            
            # Update portfolio weights
            for asset, weight in self.portfolio_state['current_weights'].items():
                new_weight = weight * reduction_factor
                self.portfolio_state['current_weights'][asset] = new_weight
            
            # Log risk reduction
            await self.performance_tracker.log_risk_reduction(reduction_factor)
            
        except Exception as e:
            logger.error(f"Error implementing risk reduction: {str(e)}")
    
    async def _implement_volatility_management(self) -> None:
        """Implement volatility management strategy."""
        try:
            logger.info("Implementing volatility management strategy...")
            
            # Increase diversification
            # Reduce concentrated positions
            max_weight = 0.25  # Reduce from 0.40
            
            for asset, weight in self.portfolio_state['current_weights'].items():
                if weight > max_weight:
                    excess = weight - max_weight
                    self.portfolio_state['current_weights'][asset] = max_weight
                    # Redistribute excess to other assets
                    await self._redistribute_weight(excess, exclude_asset=asset)
            
            # Log volatility management
            await self.performance_tracker.log_volatility_management(max_weight)
            
        except Exception as e:
            logger.error(f"Error implementing volatility management: {str(e)}")
    
    async def _implement_performance_improvement(self) -> None:
        """Implement performance improvement strategy."""
        try:
            logger.info("Implementing performance improvement strategy...")
            
            # Analyze underperforming assets
            performance_metrics = await self.performance_tracker.get_asset_performance()
            
            # Reduce exposure to underperforming assets
            for asset, metrics in performance_metrics.items():
                if metrics.get('sharpe_ratio', 0) < 0:
                    current_weight = self.portfolio_state['current_weights'].get(asset, 0)
                    if current_weight > 0.05:  # Only reduce if significant position
                        new_weight = current_weight * 0.7  # Reduce by 30%
                        self.portfolio_state['current_weights'][asset] = new_weight
            
            # Log performance improvement
            await self.performance_tracker.log_performance_improvement()
            
        except Exception as e:
            logger.error(f"Error implementing performance improvement: {str(e)}")
    
    async def _trigger_rebalancing(self) -> None:
        """Trigger portfolio rebalancing."""
        try:
            logger.info("Triggering portfolio rebalancing...")
            
            # Update rebalancing count
            self.portfolio_state['rebalancing_count'] += 1
            self.portfolio_state['last_rebalancing'] = datetime.now()
            
            # Log rebalancing event
            await self.performance_tracker.log_rebalancing_event()
            
        except Exception as e:
            logger.error(f"Error triggering rebalancing: {str(e)}")
    
    async def _redistribute_weight(self, excess_weight: float, exclude_asset: str) -> None:
        """Redistribute excess weight to other assets."""
        try:
            # Get other assets
            other_assets = [asset for asset in self.portfolio_state['current_weights'].keys() 
                          if asset != exclude_asset]
            
            if other_assets:
                # Distribute equally
                weight_per_asset = excess_weight / len(other_assets)
                for asset in other_assets:
                    current_weight = self.portfolio_state['current_weights'].get(asset, 0)
                    self.portfolio_state['current_weights'][asset] = current_weight + weight_per_asset
                    
        except Exception as e:
            logger.error(f"Error redistributing weight: {str(e)}")
    
    async def _log_performance_summary(self) -> None:
        """Log performance summary."""
        try:
            summary = await self.performance_tracker.get_performance_summary()
            
            logger.info(f"Performance Summary - "
                       f"Return: {summary.get('total_return', 0):.4f}, "
                       f"Vol: {summary.get('volatility', 0):.4f}, "
                       f"Sharpe: {summary.get('sharpe_ratio', 0):.3f}, "
                       f"Drawdown: {summary.get('max_drawdown', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Error logging performance summary: {str(e)}")
    
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
    
    # ==================== ADVANCED PERFORMANCE MONITORING ====================
    
    async def get_real_time_portfolio_metrics(self) -> Dict[str, Any]:
        """Get real-time portfolio metrics with WorldQuant standards."""
        try:
            # Get current portfolio state
            current_weights = self.portfolio_state.get('current_weights', {})
            target_weights = self.portfolio_state.get('target_weights', {})
            
            # Calculate weight drift
            weight_drift = {}
            total_drift = 0.0
            for asset in set(current_weights.keys()) | set(target_weights.keys()):
                current = current_weights.get(asset, 0.0)
                target = target_weights.get(asset, 0.0)
                drift = abs(current - target)
                weight_drift[asset] = drift
                total_drift += drift
            
            # Calculate portfolio concentration
            concentration = {}
            if current_weights:
                sorted_weights = sorted(current_weights.items(), key=lambda x: x[1], reverse=True)
                top_5_assets = sorted_weights[:5]
                concentration = {
                    'top_5_weight': sum(weight for _, weight in top_5_assets),
                    'herfindahl_index': sum(weight**2 for _, weight in current_weights.items()),
                    'largest_position': max(current_weights.values()) if current_weights else 0.0,
                    'largest_asset': max(current_weights.items(), key=lambda x: x[1])[0] if current_weights else None
                }
            
            # Performance tracking
            performance_metrics = {
                'total_trades': self.portfolio_state.get('total_trades', 0),
                'total_volume': self.portfolio_state.get('total_volume', 0.0),
                'rebalancing_count': self.portfolio_state.get('rebalancing_count', 0),
                'last_rebalancing': self.portfolio_state.get('last_rebalancing'),
                'weight_drift': total_drift,
                'concentration': concentration,
                'active_positions': len([w for w in current_weights.values() if w > 0.01])
            }
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Error getting real-time portfolio metrics: {str(e)}")
            return {}
    
    async def get_performance_attribution(self) -> Dict[str, Any]:
        """Get performance attribution analysis."""
        try:
            attribution = {
                'factor_attribution': {},
                'sector_attribution': {},
                'asset_attribution': {},
                'timing_attribution': {},
                'selection_attribution': {}
            }
            
            # Factor attribution (simplified)
            if hasattr(self, 'performance_tracker'):
                factor_exposures = await self.performance_tracker.get_performance_summary()
                attribution['factor_attribution'] = {
                    'market_factor': factor_exposures.get('beta', 1.0),
                    'size_factor': 0.0,  # Placeholder
                    'value_factor': 0.0,  # Placeholder
                    'momentum_factor': 0.0,  # Placeholder
                    'volatility_factor': factor_exposures.get('volatility', 0.0)
                }
            
            # Asset attribution
            current_weights = self.portfolio_state.get('current_weights', {})
            for asset, weight in current_weights.items():
                if weight > 0.01:  # Only significant positions
                    attribution['asset_attribution'][asset] = {
                        'weight': weight,
                        'contribution': weight * 0.02,  # Placeholder return
                        'risk_contribution': weight * 0.01  # Placeholder risk
                    }
            
            return attribution
            
        except Exception as e:
            logger.error(f"Error getting performance attribution: {str(e)}")
            return {}
    
    async def get_risk_decomposition(self) -> Dict[str, Any]:
        """Get risk decomposition analysis."""
        try:
            risk_decomposition = {
                'total_risk': 0.0,
                'factor_risk': {},
                'idiosyncratic_risk': 0.0,
                'concentration_risk': 0.0,
                'liquidity_risk': 0.0,
                'tail_risk': 0.0
            }
            
            # Calculate concentration risk
            current_weights = self.portfolio_state.get('current_weights', {})
            if current_weights:
                herfindahl_index = sum(weight**2 for weight in current_weights.values())
                risk_decomposition['concentration_risk'] = herfindahl_index
                risk_decomposition['total_risk'] = herfindahl_index * 0.15  # Placeholder
            
            # Factor risk (simplified)
            risk_decomposition['factor_risk'] = {
                'market_risk': 0.08,
                'size_risk': 0.03,
                'value_risk': 0.02,
                'momentum_risk': 0.04,
                'volatility_risk': 0.05
            }
            
            return risk_decomposition
            
        except Exception as e:
            logger.error(f"Error getting risk decomposition: {str(e)}")
            return {}
    
    async def get_stress_test_results(self) -> Dict[str, Any]:
        """Get stress test results."""
        try:
            stress_scenarios = {
                'market_crash': {
                    'description': '30% market decline',
                    'impact': -0.25,
                    'probability': 0.05
                },
                'volatility_spike': {
                    'description': '3x volatility increase',
                    'impact': -0.15,
                    'probability': 0.10
                },
                'liquidity_crisis': {
                    'description': '50% liquidity reduction',
                    'impact': -0.20,
                    'probability': 0.03
                },
                'correlation_breakdown': {
                    'description': 'Correlation breakdown',
                    'impact': -0.10,
                    'probability': 0.15
                }
            }
            
            # Calculate expected shortfall
            expected_shortfall = sum(
                scenario['impact'] * scenario['probability'] 
                for scenario in stress_scenarios.values()
            )
            
            return {
                'stress_scenarios': stress_scenarios,
                'expected_shortfall': expected_shortfall,
                'worst_case_scenario': min(stress_scenarios.values(), key=lambda x: x['impact']),
                'most_likely_scenario': max(stress_scenarios.values(), key=lambda x: x['probability'])
            }
            
        except Exception as e:
            logger.error(f"Error getting stress test results: {str(e)}")
            return {}
    
    async def get_optimization_effectiveness(self) -> Dict[str, Any]:
        """Get optimization effectiveness metrics."""
        try:
            effectiveness = {
                'tracking_error': 0.02,  # Placeholder
                'information_ratio': 0.5,  # Placeholder
                'sharpe_ratio': 1.2,  # Placeholder
                'max_drawdown': -0.08,  # Placeholder
                'calmar_ratio': 0.8,  # Placeholder
                'sortino_ratio': 1.5,  # Placeholder
                'win_rate': 0.65,  # Placeholder
                'profit_factor': 1.8,  # Placeholder
                'recovery_factor': 0.6  # Placeholder
            }
            
            # Calculate effectiveness score (0-100)
            score = (
                min(max(effectiveness['sharpe_ratio'] * 20, 0), 30) +
                min(max(effectiveness['win_rate'] * 30, 0), 30) +
                min(max(effectiveness['information_ratio'] * 20, 0), 20) +
                min(max(abs(effectiveness['max_drawdown']) * -50, 0), 20)
            )
            
            effectiveness['effectiveness_score'] = score
            
            return effectiveness
            
        except Exception as e:
            logger.error(f"Error getting optimization effectiveness: {str(e)}")
            return {}
    
    async def get_comprehensive_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report with all metrics."""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_metrics': await self.get_real_time_portfolio_metrics(),
                'performance_attribution': await self.get_performance_attribution(),
                'risk_decomposition': await self.get_risk_decomposition(),
                'stress_test_results': await self.get_stress_test_results(),
                'optimization_effectiveness': await self.get_optimization_effectiveness(),
                'alerts': await self.performance_tracker.check_alerts() if hasattr(self, 'performance_tracker') else [],
                'monitoring_status': {
                    'active': self.monitoring_active,
                    'last_update': self.monitoring_state.get('last_update'),
                    'update_frequency': self.monitoring_state.get('update_frequency', 30)
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error getting comprehensive performance report: {str(e)}")
            return {}
    
    async def close(self) -> None:
        """Close the portfolio optimizer and cleanup resources."""
        try:
            logger.info("Closing WorldQuantPortfolioOptimizer...")
            
            # Stop performance monitoring
            if hasattr(self, 'monitoring_active') and self.monitoring_active:
                await self.stop_performance_monitoring()
            
            # Clear portfolio state
            if hasattr(self, 'portfolio_state'):
                self.portfolio_state.clear()
            
            # Clear monitoring state
            if hasattr(self, 'monitoring_state'):
                self.monitoring_state.clear()
            
            logger.info("WorldQuantPortfolioOptimizer closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing WorldQuantPortfolioOptimizer: {str(e)}")
            raise 