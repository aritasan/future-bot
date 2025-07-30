import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Advanced portfolio optimization using modern quantitative methods.
    Implements Markowitz, Black-Litterman, and Risk Parity approaches.
    """
    
    def __init__(self, risk_free_rate: float = 0.02, target_return: float = None):
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.optimization_history = []
        
    def optimize_portfolio(self, returns: pd.DataFrame, method: str = 'markowitz', 
                         constraints: Dict = None) -> Dict:
        """
        Optimize portfolio weights using specified method.
        
        Args:
            returns: DataFrame of asset returns
            method: Optimization method ('markowitz', 'black_litterman', 'risk_parity', 'max_sharpe')
            constraints: Additional constraints for optimization
            
        Returns:
            Dict: Optimization results with weights and metrics
        """
        try:
            # Convert returns dict to DataFrame if needed
            if isinstance(returns, dict):
                returns_df = pd.DataFrame(returns)
            else:
                returns_df = returns
            
            if method == 'markowitz':
                return self._markowitz_optimization(returns_df, constraints)
            elif method == 'black_litterman':
                return self._black_litterman_optimization(returns_df, constraints)
            elif method == 'risk_parity':
                return self._risk_parity_optimization(returns_df, constraints)
            elif method == 'max_sharpe':
                return self._max_sharpe_optimization(returns_df, constraints)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
                
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {str(e)}")
            return {'error': str(e)}
    
    def _markowitz_optimization(self, returns: pd.DataFrame, constraints: Dict = None) -> Dict:
        """Markowitz mean-variance optimization."""
        try:
            # Ensure returns is a DataFrame
            if not isinstance(returns, pd.DataFrame):
                if isinstance(returns, dict):
                    returns = pd.DataFrame(returns)
                elif isinstance(returns, list):
                    return {'error': 'Returns data is a list, expected DataFrame', 'optimization_success': False}
                else:
                    return {'error': f'Invalid returns data type: {type(returns)}', 'optimization_success': False}
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean() * 252  # Annualized
            cov_matrix = returns.cov() * 252  # Annualized
            
            # Check for insufficient data
            if returns.empty or returns.isnull().all().all():
                return {'error': 'No returns data available', 'optimization_success': False}
            
            if cov_matrix.empty or cov_matrix.isnull().all().all():
                return {'error': 'Insufficient data for optimization', 'optimization_success': False}
            
            # Check if we have enough data points
            if len(returns) < 30:  # Need at least 30 data points
                return {'error': 'Insufficient data points for optimization (need at least 30)', 'optimization_success': False}
            
            n_assets = len(expected_returns)
            
            # Define objective function (minimize portfolio variance)
            def objective(weights):
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                return portfolio_variance
            
            # Define constraints
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
            if self.target_return is not None:
                constraints_list.append({
                    'type': 'eq', 
                    'fun': lambda x: np.dot(x, expected_returns) - self.target_return
                })
            
            # Add custom constraints
            if constraints:
                if 'min_weight' in constraints:
                    constraints_list.append({
                        'type': 'ineq',
                        'fun': lambda x: x - constraints['min_weight']
                    })
                if 'max_weight' in constraints:
                    constraints_list.append({
                        'type': 'ineq',
                        'fun': lambda x: constraints['max_weight'] - x
                    })
            
            # Initial guess (equal weights)
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective, initial_weights, 
                method='SLSQP', 
                constraints=constraints_list,
                bounds=[(0, 1)] * n_assets
            )
            
            if result.success:
                optimal_weights = result.x
                portfolio_return = np.dot(optimal_weights, expected_returns)
                portfolio_variance = result.fun
                portfolio_volatility = float(np.sqrt(portfolio_variance))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
                
                results = {
                    'method': 'markowitz',
                    'weights': dict(zip(returns.columns, optimal_weights)),
                    'portfolio_return': portfolio_return,
                    'portfolio_volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'optimization_success': True
                }
                
                self._store_optimization_result(results)
                return results
            else:
                return {'error': 'Optimization failed', 'optimization_success': False}
                
        except Exception as e:
            logger.error(f"Error in Markowitz optimization: {str(e)}")
            return {'error': str(e), 'optimization_success': False}
    
    def _black_litterman_optimization(self, returns: pd.DataFrame, constraints: Dict = None) -> Dict:
        """Black-Litterman model optimization."""
        try:
            # Calculate market equilibrium returns
            market_caps = np.ones(len(returns.columns))  # Equal market caps for simplicity
            market_weights = market_caps / np.sum(market_caps)
            
            # Calculate equilibrium returns
            cov_matrix = returns.cov() * 252
            risk_aversion = 3.0  # Typical risk aversion parameter
            equilibrium_returns = risk_aversion * np.dot(cov_matrix, market_weights)
            
            # Define views (example: bullish view on first asset)
            views = np.array([0.05])  # 5% excess return expectation
            view_assets = np.array([0])  # First asset
            view_matrix = np.zeros((len(views), len(returns.columns)))
            view_matrix[0, view_assets[0]] = 1
            
            # Define view uncertainty
            tau = 0.05  # Scaling parameter
            omega = tau * np.dot(np.dot(view_matrix, cov_matrix), view_matrix.T)
            
            # Black-Litterman posterior returns
            bl_returns = self._calculate_black_litterman_returns(
                equilibrium_returns, cov_matrix, views, view_matrix, omega
            )
            
            # Optimize with posterior returns
            return self._markowitz_optimization_with_returns(returns, bl_returns, constraints)
            
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {str(e)}")
            return {'error': str(e), 'optimization_success': False}
    
    def _calculate_black_litterman_returns(self, equilibrium_returns, cov_matrix, views, view_matrix, omega):
        """Calculate Black-Litterman posterior returns."""
        try:
            tau = 0.05
            sigma = tau * cov_matrix
            
            # Posterior covariance
            posterior_cov = np.linalg.inv(np.linalg.inv(sigma) + np.dot(np.dot(view_matrix.T, np.linalg.inv(omega)), view_matrix))
            
            # Posterior returns
            posterior_returns = np.dot(posterior_cov, 
                np.dot(np.linalg.inv(sigma), equilibrium_returns) + 
                np.dot(np.dot(view_matrix.T, np.linalg.inv(omega)), views)
            )
            
            return posterior_returns
            
        except Exception as e:
            logger.error(f"Error calculating Black-Litterman returns: {str(e)}")
            return equilibrium_returns
    
    def _risk_parity_optimization(self, returns: pd.DataFrame, constraints: Dict = None) -> Dict:
        """Risk parity optimization."""
        try:
            cov_matrix = returns.cov() * 252
            n_assets = len(returns.columns)
            
            # Define objective function (minimize risk contribution dispersion)
            def objective(weights):
                portfolio_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
                risk_contributions = weights * np.dot(cov_matrix, weights) / portfolio_vol
                return np.sum((risk_contributions - np.mean(risk_contributions))**2)
            
            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            # Initial guess
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective, initial_weights,
                method='SLSQP',
                constraints=constraints_list,
                bounds=[(0, 1)] * n_assets
            )
            
            if result.success:
                optimal_weights = result.x
                expected_returns = returns.mean() * 252
                portfolio_return = np.dot(optimal_weights, expected_returns)
                portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                
                # Calculate risk contributions
                risk_contributions = optimal_weights * np.dot(cov_matrix, optimal_weights) / portfolio_volatility
                
                results = {
                    'method': 'risk_parity',
                    'weights': dict(zip(returns.columns, optimal_weights)),
                    'portfolio_return': portfolio_return,
                    'portfolio_volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'risk_contributions': dict(zip(returns.columns, risk_contributions)),
                    'optimization_success': True
                }
                
                self._store_optimization_result(results)
                return results
            else:
                return {'error': 'Risk parity optimization failed', 'optimization_success': False}
                
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {str(e)}")
            return {'error': str(e), 'optimization_success': False}
    
    def _max_sharpe_optimization(self, returns: pd.DataFrame, constraints: Dict = None) -> Dict:
        """Maximum Sharpe ratio optimization."""
        try:
            expected_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            n_assets = len(expected_returns)
            
            # Define objective function (negative Sharpe ratio to minimize)
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_volatility = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
                return -(portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            # Initial guess
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective, initial_weights,
                method='SLSQP',
                constraints=constraints_list,
                bounds=[(0, 1)] * n_assets
            )
            
            if result.success:
                optimal_weights = result.x
                portfolio_return = np.dot(optimal_weights, expected_returns)
                portfolio_volatility = float(np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))))
                sharpe_ratio = -result.fun  # Convert back to positive
                
                results = {
                    'method': 'max_sharpe',
                    'weights': dict(zip(returns.columns, optimal_weights)),
                    'portfolio_return': portfolio_return,
                    'portfolio_volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'optimization_success': True
                }
                
                self._store_optimization_result(results)
                return results
            else:
                return {'error': 'Max Sharpe optimization failed', 'optimization_success': False}
                
        except Exception as e:
            logger.error(f"Error in max Sharpe optimization: {str(e)}")
            return {'error': str(e), 'optimization_success': False}
    
    def _markowitz_optimization_with_returns(self, returns: pd.DataFrame, custom_returns: np.array, constraints: Dict = None) -> Dict:
        """Markowitz optimization with custom expected returns."""
        try:
            cov_matrix = returns.cov() * 252
            n_assets = len(returns.columns)
            
            def objective(weights):
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                return portfolio_variance
            
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            if self.target_return is not None:
                constraints_list.append({
                    'type': 'eq', 
                    'fun': lambda x: np.dot(x, custom_returns) - self.target_return
                })
            
            initial_weights = np.array([1/n_assets] * n_assets)
            
            result = minimize(
                objective, initial_weights,
                method='SLSQP',
                constraints=constraints_list,
                bounds=[(0, 1)] * n_assets
            )
            
            if result.success:
                optimal_weights = result.x
                portfolio_return = np.dot(optimal_weights, custom_returns)
                portfolio_variance = result.fun
                portfolio_volatility = float(np.sqrt(portfolio_variance))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                
                return {
                    'method': 'black_litterman',
                    'weights': dict(zip(returns.columns, optimal_weights)),
                    'portfolio_return': portfolio_return,
                    'portfolio_volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'optimization_success': True
                }
            else:
                return {'error': 'Black-Litterman optimization failed', 'optimization_success': False}
                
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {str(e)}")
            return {'error': str(e), 'optimization_success': False}
    
    def _store_optimization_result(self, result: Dict):
        """Store optimization result in history."""
        self.optimization_history.append({
            'timestamp': pd.Timestamp.now(),
            'result': result
        })
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of all optimization results."""
        if not self.optimization_history:
            return {'message': 'No optimization history available'}
        
        summary = {
            'total_optimizations': len(self.optimization_history),
            'methods_used': [],
            'average_sharpe_ratios': {},
            'best_performing_method': None,
            'recent_results': []
        }
        
        method_performance = {}
        
        for record in self.optimization_history:
            result = record['result']
            method = result.get('method', 'unknown')
            
            if method not in summary['methods_used']:
                summary['methods_used'].append(method)
                method_performance[method] = []
            
            if 'sharpe_ratio' in result:
                method_performance[method].append(result['sharpe_ratio'])
        
        # Calculate average Sharpe ratios
        for method, ratios in method_performance.items():
            summary['average_sharpe_ratios'][method] = np.mean(ratios)
        
        # Find best performing method
        if summary['average_sharpe_ratios']:
            best_method = max(summary['average_sharpe_ratios'], 
                            key=summary['average_sharpe_ratios'].get)
            summary['best_performing_method'] = best_method
        
        # Get recent results
        summary['recent_results'] = self.optimization_history[-5:]
        
        return summary 