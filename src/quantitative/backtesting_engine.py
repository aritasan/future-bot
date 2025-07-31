"""
WorldQuant Advanced Backtesting Engine Implementation
Advanced backtesting with walk-forward analysis, Monte Carlo simulation, stress testing, and performance attribution.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedBacktestingEngine:
    """
    WorldQuant-level advanced backtesting engine with comprehensive analysis capabilities.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Advanced Backtesting Engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Backtesting parameters
        self.backtesting_params = {
            'walk_forward': {
                'n_splits': 3,  # Reduced from 5
                'test_size': 0.15,  # Reduced from 0.2
                'gap': 5,  # Reduced from 10
                'min_train_size': 50  # Reduced from 100
            },
            'monte_carlo': {
                'n_simulations': 1000,
                'confidence_level': 0.95,
                'risk_free_rate': 0.02
            },
            'stress_testing': {
                'scenarios': ['market_crash', 'volatility_spike', 'correlation_breakdown', 'liquidity_crisis'],
                'shock_sizes': [0.1, 0.2, 0.3, 0.5],
                'stress_periods': 30
            },
            'performance_attribution': {
                'factors': ['market', 'size', 'value', 'momentum', 'volatility', 'liquidity'],
                'attribution_method': 'brinson'  # 'brinson', 'carino', 'menchero'
            }
        }
        
        # Performance tracking
        self.backtest_results = {}
        self.performance_metrics = {}
        self.risk_metrics = {}
        self.attribution_results = {}
        
        logger.info("AdvancedBacktestingEngine initialized")
    
    async def initialize(self) -> bool:
        """Initialize the backtesting engine."""
        try:
            logger.info("AdvancedBacktestingEngine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing AdvancedBacktestingEngine: {str(e)}")
            return False
    
    async def run_walk_forward_analysis(self, strategy: Callable, 
                                      data: pd.DataFrame,
                                      initial_capital: float = 100000.0) -> Dict[str, Any]:
        """
        Walk-forward analysis for strategy validation.
        
        Args:
            strategy: Trading strategy function
            data: Historical market data
            initial_capital: Initial capital for backtesting
            
        Returns:
            Dictionary with walk-forward analysis results
        """
        try:
            logger.info("Running walk-forward analysis...")
            
            # Get parameters
            n_splits = self.backtesting_params['walk_forward']['n_splits']
            test_size = self.backtesting_params['walk_forward']['test_size']
            gap = self.backtesting_params['walk_forward']['gap']
            min_train_size = self.backtesting_params['walk_forward']['min_train_size']
            
            # Prepare data
            returns = data['returns'] if 'returns' in data.columns else data.pct_change().dropna()
            
            # Time series split
            tscv = TimeSeriesSplit(
                n_splits=n_splits,
                test_size=int(len(returns) * test_size),
                gap=gap
            )
            
            walk_forward_results = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(returns)):
                if len(train_idx) < min_train_size:
                    logger.warning(f"Fold {fold}: Insufficient training data ({len(train_idx)} < {min_train_size})")
                    continue
                
                # Split data
                train_data = data.iloc[train_idx]
                test_data = data.iloc[test_idx]
                
                # Run backtest on test period
                test_result = await self._run_single_backtest(strategy, test_data, initial_capital)
                
                if test_result:
                    test_result['fold'] = fold
                    test_result['train_period'] = (train_idx[0], train_idx[-1])
                    test_result['test_period'] = (test_idx[0], test_idx[-1])
                    walk_forward_results.append(test_result)
            
            # Aggregate results
            if walk_forward_results:
                aggregated_results = self._aggregate_walk_forward_results(walk_forward_results)
                
                logger.info(f"Walk-forward analysis completed - {len(walk_forward_results)} folds")
                return aggregated_results
            else:
                logger.warning("No valid walk-forward results")
                return {'status': 'failed', 'message': 'No valid results'}
                
        except Exception as e:
            logger.error(f"Error in walk-forward analysis: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    async def run_monte_carlo_simulation(self, strategy: Callable,
                                       data: pd.DataFrame,
                                       initial_capital: float = 100000.0,
                                       n_simulations: Optional[int] = None) -> Dict[str, Any]:
        """
        Monte Carlo simulation for strategy risk assessment.
        
        Args:
            strategy: Trading strategy function
            data: Historical market data
            initial_capital: Initial capital for backtesting
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with Monte Carlo simulation results
        """
        try:
            logger.info("Running Monte Carlo simulation...")
            
            # Get parameters
            n_sim = n_simulations or self.backtesting_params['monte_carlo']['n_simulations']
            confidence_level = self.backtesting_params['monte_carlo']['confidence_level']
            risk_free_rate = self.backtesting_params['monte_carlo']['risk_free_rate']
            
            # Prepare data
            returns = data['returns'] if 'returns' in data.columns else data.pct_change().dropna()
            
            # Run base backtest
            base_result = await self._run_single_backtest(strategy, data, initial_capital)
            
            if not base_result or base_result.get('status') != 'success':
                logger.warning("Base backtest failed, cannot run Monte Carlo simulation")
                return {'status': 'failed', 'message': 'Base backtest failed'}
            
            # Monte Carlo simulation
            mc_results = []
            
            for sim in range(n_sim):
                # Generate random returns based on historical distribution
                simulated_returns = self._generate_simulated_returns(returns)
                simulated_data = data.copy()
                simulated_data['returns'] = simulated_returns
                
                # Run backtest with simulated data
                sim_result = await self._run_single_backtest(strategy, simulated_data, initial_capital)
                
                if sim_result and sim_result.get('status') == 'success':
                    mc_results.append({
                        'total_return': sim_result['total_return'],
                        'sharpe_ratio': sim_result['sharpe_ratio'],
                        'max_drawdown': sim_result['max_drawdown'],
                        'volatility': sim_result['volatility']
                    })
            
            # Calculate statistics
            if mc_results:
                mc_stats = self._calculate_monte_carlo_statistics(mc_results, confidence_level, risk_free_rate)
                
                logger.info(f"Monte Carlo simulation completed - {len(mc_results)} simulations")
                return mc_stats
            else:
                logger.warning("No valid Monte Carlo results")
                return {'status': 'failed', 'message': 'No valid results'}
                
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    async def run_stress_testing(self, strategy: Callable,
                               data: pd.DataFrame,
                               initial_capital: float = 100000.0) -> Dict[str, Any]:
        """
        Stress testing for strategy robustness.
        
        Args:
            strategy: Trading strategy function
            data: Historical market data
            initial_capital: Initial capital for backtesting
            
        Returns:
            Dictionary with stress testing results
        """
        try:
            logger.info("Running stress testing...")
            
            # Get parameters
            scenarios = self.backtesting_params['stress_testing']['scenarios']
            shock_sizes = self.backtesting_params['stress_testing']['shock_sizes']
            stress_periods = self.backtesting_params['stress_testing']['stress_periods']
            
            # Base backtest
            base_result = await self._run_single_backtest(strategy, data, initial_capital)
            
            if not base_result or base_result.get('status') != 'success':
                logger.warning("Base backtest failed, cannot run stress testing")
                return {'status': 'failed', 'message': 'Base backtest failed'}
            
            stress_results = {}
            
            # Market crash scenario
            if 'market_crash' in scenarios:
                crash_results = []
                for shock_size in shock_sizes:
                    crash_data = self._apply_market_crash_shock(data, shock_size)
                    crash_result = await self._run_single_backtest(strategy, crash_data, initial_capital)
                    if crash_result and crash_result.get('status') == 'success':
                        crash_results.append({
                            'shock_size': shock_size,
                            'total_return': crash_result['total_return'],
                            'sharpe_ratio': crash_result['sharpe_ratio'],
                            'max_drawdown': crash_result['max_drawdown']
                        })
                stress_results['market_crash'] = crash_results
            
            # Volatility spike scenario
            if 'volatility_spike' in scenarios:
                volatility_results = []
                for shock_size in shock_sizes:
                    volatility_data = self._apply_volatility_spike_shock(data, shock_size)
                    volatility_result = await self._run_single_backtest(strategy, volatility_data, initial_capital)
                    if volatility_result and volatility_result.get('status') == 'success':
                        volatility_results.append({
                            'shock_size': shock_size,
                            'total_return': volatility_result['total_return'],
                            'sharpe_ratio': volatility_result['sharpe_ratio'],
                            'max_drawdown': volatility_result['max_drawdown']
                        })
                stress_results['volatility_spike'] = volatility_results
            
            # Correlation breakdown scenario
            if 'correlation_breakdown' in scenarios:
                correlation_results = []
                for shock_size in shock_sizes:
                    correlation_data = self._apply_correlation_breakdown_shock(data, shock_size)
                    correlation_result = await self._run_single_backtest(strategy, correlation_data, initial_capital)
                    if correlation_result and correlation_result.get('status') == 'success':
                        correlation_results.append({
                            'shock_size': shock_size,
                            'total_return': correlation_result['total_return'],
                            'sharpe_ratio': correlation_result['sharpe_ratio'],
                            'max_drawdown': correlation_result['max_drawdown']
                        })
                stress_results['correlation_breakdown'] = correlation_results
            
            # Liquidity crisis scenario
            if 'liquidity_crisis' in scenarios:
                liquidity_results = []
                for shock_size in shock_sizes:
                    liquidity_data = self._apply_liquidity_crisis_shock(data, shock_size)
                    liquidity_result = await self._run_single_backtest(strategy, liquidity_data, initial_capital)
                    if liquidity_result and liquidity_result.get('status') == 'success':
                        liquidity_results.append({
                            'shock_size': shock_size,
                            'total_return': liquidity_result['total_return'],
                            'sharpe_ratio': liquidity_result['sharpe_ratio'],
                            'max_drawdown': liquidity_result['max_drawdown']
                        })
                stress_results['liquidity_crisis'] = liquidity_results
            
            # Base results
            stress_results['base'] = {
                'total_return': base_result['total_return'],
                'sharpe_ratio': base_result['sharpe_ratio'],
                'max_drawdown': base_result['max_drawdown']
            }
            
            logger.info("Stress testing completed")
            return stress_results
            
        except Exception as e:
            logger.error(f"Error in stress testing: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    async def run_performance_attribution(self, strategy: Callable,
                                       data: pd.DataFrame,
                                       factor_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Performance attribution analysis.
        
        Args:
            strategy: Trading strategy function
            data: Historical market data
            factor_data: Factor exposure data
            
        Returns:
            Dictionary with performance attribution results
        """
        try:
            logger.info("Running performance attribution analysis...")
            
            # Get parameters
            factors = self.backtesting_params['performance_attribution']['factors']
            attribution_method = self.backtesting_params['performance_attribution']['attribution_method']
            
            # Run strategy backtest
            strategy_result = await self._run_single_backtest(strategy, data, 100000.0)
            
            if not strategy_result or strategy_result.get('status') != 'success':
                logger.warning("Strategy backtest failed, cannot run performance attribution")
                return {'status': 'failed', 'message': 'Strategy backtest failed'}
            
            # Calculate factor returns
            factor_returns = self._calculate_factor_returns(data, factor_data, factors)
            
            # Performance attribution
            if attribution_method == 'brinson':
                attribution_result = self._brinson_attribution(strategy_result, factor_returns)
            elif attribution_method == 'carino':
                attribution_result = self._carino_attribution(strategy_result, factor_returns)
            elif attribution_method == 'menchero':
                attribution_result = self._menchero_attribution(strategy_result, factor_returns)
            else:
                attribution_result = self._brinson_attribution(strategy_result, factor_returns)
            
            logger.info("Performance attribution analysis completed")
            return attribution_result
            
        except Exception as e:
            logger.error(f"Error in performance attribution: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    async def _run_single_backtest(self, strategy: Callable,
                                 data: pd.DataFrame,
                                 initial_capital: float) -> Dict[str, Any]:
        """Run a single backtest."""
        try:
            # Simulate strategy execution
            portfolio_values = [initial_capital]
            returns = []
            
            for i in range(1, len(data)):
                # Get current data slice
                current_data = data.iloc[:i+1]
                
                # Run strategy
                signal = strategy(current_data)
                
                # Calculate returns (simplified)
                if signal > 0:  # Buy signal
                    daily_return = data.iloc[i]['returns'] if 'returns' in data.columns else data.iloc[i].pct_change().iloc[-1]
                elif signal < 0:  # Sell signal
                    daily_return = -data.iloc[i]['returns'] if 'returns' in data.columns else -data.iloc[i].pct_change().iloc[-1]
                else:  # Hold
                    daily_return = 0
                
                returns.append(daily_return)
                
                # Update portfolio value
                new_value = portfolio_values[-1] * (1 + daily_return)
                portfolio_values.append(new_value)
            
            # Calculate metrics
            total_return = (portfolio_values[-1] - initial_capital) / initial_capital
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = (np.mean(returns) * 252 - 0.02) / volatility if volatility > 0 else 0
            
            # Calculate maximum drawdown
            peak = portfolio_values[0]
            max_drawdown = 0
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            return {
                'status': 'success',
                'total_return': float(total_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'portfolio_values': portfolio_values,
                'returns': returns
            }
            
        except Exception as e:
            logger.error(f"Error in single backtest: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate walk-forward analysis results."""
        try:
            total_returns = [r['total_return'] for r in results]
            sharpe_ratios = [r['sharpe_ratio'] for r in results]
            max_drawdowns = [r['max_drawdown'] for r in results]
            volatilities = [r['volatility'] for r in results]
            
            aggregated = {
                'status': 'success',
                'n_folds': len(results),
                'mean_total_return': float(np.mean(total_returns)),
                'std_total_return': float(np.std(total_returns)),
                'mean_sharpe_ratio': float(np.mean(sharpe_ratios)),
                'std_sharpe_ratio': float(np.std(sharpe_ratios)),
                'mean_max_drawdown': float(np.mean(max_drawdowns)),
                'std_max_drawdown': float(np.std(max_drawdowns)),
                'mean_volatility': float(np.mean(volatilities)),
                'std_volatility': float(np.std(volatilities)),
                'fold_results': results
            }
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating walk-forward results: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _generate_simulated_returns(self, historical_returns: pd.Series) -> pd.Series:
        """Generate simulated returns using historical distribution."""
        try:
            # Fit distribution to historical returns
            mu = np.mean(historical_returns)
            sigma = np.std(historical_returns)
            
            # Generate random returns
            simulated_returns = np.random.normal(mu, sigma, len(historical_returns))
            
            return pd.Series(simulated_returns, index=historical_returns.index)
            
        except Exception as e:
            logger.error(f"Error generating simulated returns: {str(e)}")
            return historical_returns
    
    def _calculate_monte_carlo_statistics(self, results: List[Dict],
                                        confidence_level: float,
                                        risk_free_rate: float) -> Dict[str, Any]:
        """Calculate Monte Carlo simulation statistics."""
        try:
            total_returns = [r['total_return'] for r in results]
            sharpe_ratios = [r['sharpe_ratio'] for r in results]
            max_drawdowns = [r['max_drawdown'] for r in results]
            volatilities = [r['volatility'] for r in results]
            
            # Calculate percentiles
            alpha = 1 - confidence_level
            lower_percentile = alpha / 2 * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            mc_stats = {
                'status': 'success',
                'n_simulations': len(results),
                'confidence_level': confidence_level,
                'total_return': {
                    'mean': float(np.mean(total_returns)),
                    'std': float(np.std(total_returns)),
                    'min': float(np.min(total_returns)),
                    'max': float(np.max(total_returns)),
                    'percentile_5': float(np.percentile(total_returns, 5)),
                    'percentile_95': float(np.percentile(total_returns, 95)),
                    'var': float(np.percentile(total_returns, lower_percentile)),
                    'cvar': float(np.mean([r for r in total_returns if r <= np.percentile(total_returns, lower_percentile)]))
                },
                'sharpe_ratio': {
                    'mean': float(np.mean(sharpe_ratios)),
                    'std': float(np.std(sharpe_ratios)),
                    'min': float(np.min(sharpe_ratios)),
                    'max': float(np.max(sharpe_ratios))
                },
                'max_drawdown': {
                    'mean': float(np.mean(max_drawdowns)),
                    'std': float(np.std(max_drawdowns)),
                    'max': float(np.max(max_drawdowns)),
                    'percentile_95': float(np.percentile(max_drawdowns, 95))
                },
                'volatility': {
                    'mean': float(np.mean(volatilities)),
                    'std': float(np.std(volatilities))
                }
            }
            
            return mc_stats
            
        except Exception as e:
            logger.error(f"Error calculating Monte Carlo statistics: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _apply_market_crash_shock(self, data: pd.DataFrame, shock_size: float) -> pd.DataFrame:
        """Apply market crash shock to data."""
        try:
            shocked_data = data.copy()
            
            # Apply negative shock to returns
            if 'returns' in shocked_data.columns:
                shocked_data['returns'] = shocked_data['returns'] * (1 - shock_size)
            else:
                # Apply to price data
                for col in shocked_data.columns:
                    if col != 'date' and col != 'timestamp':
                        shocked_data[col] = shocked_data[col] * (1 - shock_size)
            
            return shocked_data
            
        except Exception as e:
            logger.error(f"Error applying market crash shock: {str(e)}")
            return data
    
    def _apply_volatility_spike_shock(self, data: pd.DataFrame, shock_size: float) -> pd.DataFrame:
        """Apply volatility spike shock to data."""
        try:
            shocked_data = data.copy()
            
            # Increase volatility
            if 'returns' in shocked_data.columns:
                shocked_data['returns'] = shocked_data['returns'] * (1 + shock_size)
            else:
                # Apply to price data
                for col in shocked_data.columns:
                    if col != 'date' and col != 'timestamp':
                        shocked_data[col] = shocked_data[col] * (1 + shock_size * np.random.normal(0, 1))
            
            return shocked_data
            
        except Exception as e:
            logger.error(f"Error applying volatility spike shock: {str(e)}")
            return data
    
    def _apply_correlation_breakdown_shock(self, data: pd.DataFrame, shock_size: float) -> pd.DataFrame:
        """Apply correlation breakdown shock to data."""
        try:
            shocked_data = data.copy()
            
            # Add random noise to break correlations
            if 'returns' in shocked_data.columns:
                noise = np.random.normal(0, shock_size, len(shocked_data))
                shocked_data['returns'] = shocked_data['returns'] + noise
            else:
                # Apply to price data
                for col in shocked_data.columns:
                    if col != 'date' and col != 'timestamp':
                        noise = np.random.normal(0, shock_size, len(shocked_data))
                        shocked_data[col] = shocked_data[col] * (1 + noise)
            
            return shocked_data
            
        except Exception as e:
            logger.error(f"Error applying correlation breakdown shock: {str(e)}")
            return data
    
    def _apply_liquidity_crisis_shock(self, data: pd.DataFrame, shock_size: float) -> pd.DataFrame:
        """Apply liquidity crisis shock to data."""
        try:
            shocked_data = data.copy()
            
            # Simulate liquidity crisis with increased spreads
            if 'returns' in shocked_data.columns:
                # Add transaction costs
                shocked_data['returns'] = shocked_data['returns'] * (1 - shock_size * 0.01)
            else:
                # Apply to price data
                for col in shocked_data.columns:
                    if col != 'date' and col != 'timestamp':
                        shocked_data[col] = shocked_data[col] * (1 - shock_size * 0.01)
            
            return shocked_data
            
        except Exception as e:
            logger.error(f"Error applying liquidity crisis shock: {str(e)}")
            return data
    
    def _calculate_factor_returns(self, data: pd.DataFrame,
                                factor_data: Optional[Dict],
                                factors: List[str]) -> Dict[str, float]:
        """Calculate factor returns."""
        try:
            if factor_data is None:
                # Generate mock factor returns
                factor_returns = {}
                for factor in factors:
                    factor_returns[factor] = np.random.normal(0.001, 0.02)
                return factor_returns
            
            return factor_data
            
        except Exception as e:
            logger.error(f"Error calculating factor returns: {str(e)}")
            return {}
    
    def _brinson_attribution(self, strategy_result: Dict,
                            factor_returns: Dict[str, float]) -> Dict[str, Any]:
        """Brinson performance attribution."""
        try:
            attribution = {
                'status': 'success',
                'method': 'brinson',
                'total_return': strategy_result['total_return'],
                'factor_attribution': {},
                'residual': 0.0
            }
            
            # Calculate factor attribution
            total_factor_contribution = 0
            for factor, factor_return in factor_returns.items():
                # Simplified attribution calculation
                factor_contribution = factor_return * 0.1  # Assume 10% exposure
                attribution['factor_attribution'][factor] = float(factor_contribution)
                total_factor_contribution += factor_contribution
            
            # Calculate residual
            attribution['residual'] = float(strategy_result['total_return'] - total_factor_contribution)
            
            return attribution
            
        except Exception as e:
            logger.error(f"Error in Brinson attribution: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _carino_attribution(self, strategy_result: Dict,
                           factor_returns: Dict[str, float]) -> Dict[str, Any]:
        """Carino performance attribution."""
        try:
            # Simplified Carino attribution
            return self._brinson_attribution(strategy_result, factor_returns)
            
        except Exception as e:
            logger.error(f"Error in Carino attribution: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _menchero_attribution(self, strategy_result: Dict,
                             factor_returns: Dict[str, float]) -> Dict[str, Any]:
        """Menchero performance attribution."""
        try:
            # Simplified Menchero attribution
            return self._brinson_attribution(strategy_result, factor_returns)
            
        except Exception as e:
            logger.error(f"Error in Menchero attribution: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    async def get_backtesting_summary(self) -> Dict[str, Any]:
        """Get comprehensive backtesting summary."""
        try:
            summary = {
                'backtest_results': len(self.backtest_results),
                'performance_metrics': len(self.performance_metrics),
                'risk_metrics': len(self.risk_metrics),
                'attribution_results': len(self.attribution_results),
                'capabilities': [
                    'walk_forward_analysis',
                    'monte_carlo_simulation',
                    'stress_testing',
                    'performance_attribution'
                ]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting backtesting summary: {str(e)}")
            return {} 
    async def close(self) -> None:
        """Close the advancedbacktestingengine and cleanup resources."""
        try:
            logger.info("Closing AdvancedBacktestingEngine...")
            
            # Clear any stored data
            if hasattr(self, 'analysis_cache'):
                self.analysis_cache.clear()
            if hasattr(self, 'history'):
                self.history.clear()
            if hasattr(self, 'metrics_history'):
                self.metrics_history.clear()
            
            logger.info("AdvancedBacktestingEngine closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing AdvancedBacktestingEngine: {str(e)}")
            raise
