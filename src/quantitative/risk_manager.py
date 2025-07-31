import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class VaRCalculator:
    """
    Value at Risk (VaR) calculator for quantitative risk management.
    Implements multiple VaR methodologies.
    """
    
    def __init__(self, confidence_level: float = 0.95, time_horizon: int = 1):
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
        self.var_history = []
        
    def calculate_var(self, returns: np.array, position_size: float, method: str = 'historical') -> Dict:
        """
        Calculate Value at Risk using specified method.
        
        Args:
            returns: Historical returns
            position_size: Current position size
            method: VaR method ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            Dict: VaR results with different methodologies
        """
        try:
            results = {}
            
            if method == 'historical' or method == 'all':
                results['historical_var'] = self._calculate_historical_var(returns, position_size)
                
            if method == 'parametric' or method == 'all':
                results['parametric_var'] = self._calculate_parametric_var(returns, position_size)
                
            if method == 'monte_carlo' or method == 'all':
                results['monte_carlo_var'] = self._calculate_monte_carlo_var(returns, position_size)
                
            if method == 'all':
                results['expected_shortfall'] = self._calculate_expected_shortfall(returns, results['historical_var'])
                results['conditional_var'] = self._calculate_conditional_var(returns, results['historical_var'])
                
            # Store VaR history
            self.var_history.append({
                'timestamp': pd.Timestamp.now(),
                'position_size': position_size,
                'var_results': results,
                'returns_volatility': np.std(returns),
                'returns_mean': np.mean(returns)
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_historical_var(self, returns: np.array, position_size: float) -> float:
        """Calculate historical VaR."""
        try:
            var_percentile = (1 - self.confidence_level) * 100
            var_return = np.percentile(returns, var_percentile)
            return abs(var_return * position_size)
        except Exception as e:
            logger.error(f"Error calculating historical VaR: {str(e)}")
            return 0.0
    
    def _calculate_parametric_var(self, returns: np.array, position_size: float) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        try:
            if returns is None or len(returns) < 2 or bool(np.all(np.isnan(returns))) or position_size == 0:
                return 0.0
            mean_return = np.nanmean(returns)
            std_return = np.nanstd(returns)
            z_score = norm.ppf(self.confidence_level)
            var_return = mean_return - z_score * std_return
            return abs(var_return * position_size)
        except Exception as e:
            logger.error(f"Error calculating parametric VaR: {str(e)}")
            return 0.0
    
    def _calculate_monte_carlo_var(self, returns: np.array, position_size: float, n_simulations: int = 10000) -> float:
        """Calculate Monte Carlo VaR."""
        try:
            if returns is None or len(returns) < 2 or bool(np.all(np.isnan(returns))) or position_size == 0:
                return 0.0
            mean_return = np.nanmean(returns)
            std_return = np.nanstd(returns)
            
            # Generate Monte Carlo simulations
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            
            # Calculate VaR from simulations
            var_percentile = (1 - self.confidence_level) * 100
            var_return = np.percentile(simulated_returns, var_percentile)
            if position_size == 0:
                return 0.0
            return abs(var_return * position_size)
        except Exception as e:
            logger.error(f"Error calculating Monte Carlo VaR: {str(e)}")
            return 0.0
    
    def _calculate_expected_shortfall(self, returns: np.array, var_threshold: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        try:
            if returns is None or len(returns) < 2 or bool(np.all(np.isnan(returns))):
                return 0.0
            var_return = -var_threshold  # Convert to return space
            tail_returns = returns[returns <= var_return]
            if len(tail_returns) == 0 or bool(np.all(np.isnan(tail_returns))):
                return 0.0
            mean_tail = np.nanmean(tail_returns)
            if np.isnan(mean_tail):
                return 0.0
            return abs(mean_tail)
        except Exception as e:
            logger.error(f"Error calculating Expected Shortfall: {str(e)}")
            return 0.0
    
    def _calculate_conditional_var(self, returns: np.array, var_threshold: float) -> float:
        """Calculate Conditional VaR."""
        return self._calculate_expected_shortfall(returns, var_threshold)
    
    def get_var_summary(self) -> Dict:
        """Get summary of VaR calculations."""
        try:
            if not self.var_history:
                return {'total_calculations': 0}
            
            summary = {
                'total_calculations': len(self.var_history),
                'avg_historical_var': np.mean([v['var_results'].get('historical_var', 0) for v in self.var_history]),
                'avg_parametric_var': np.mean([v['var_results'].get('parametric_var', 0) for v in self.var_history]),
                'avg_monte_carlo_var': np.mean([v['var_results'].get('monte_carlo_var', 0) for v in self.var_history]),
                'max_var': max([v['var_results'].get('historical_var', 0) for v in self.var_history]),
                'avg_position_size': np.mean([v['position_size'] for v in self.var_history])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting VaR summary: {str(e)}")
            return {'error': str(e)}

class DynamicPositionSizer:
    """
    Dynamic position sizing using Kelly Criterion and risk metrics.
    Implements quantitative position sizing methodologies.
    """
    
    def __init__(self, max_position_size: float = 0.02, risk_tolerance: float = 0.02):
        self.max_position_size = max_position_size
        self.risk_tolerance = risk_tolerance
        self.position_history = []
        
    def calculate_position_size(self, signal_strength: float, volatility: float, 
                              correlation: float, var_limit: float, 
                              win_rate: float = None, avg_win: float = None, 
                              avg_loss: float = None) -> Dict:
        """
        Calculate optimal position size using multiple methodologies.
        
        Args:
            signal_strength: Signal strength (0-1)
            volatility: Asset volatility
            correlation: Correlation with portfolio
            var_limit: VaR limit
            win_rate: Historical win rate
            avg_win: Average winning trade
            avg_loss: Average losing trade
            
        Returns:
            Dict: Position sizing results
        """
        try:
            results = {}
            
            # Kelly Criterion
            if win_rate and avg_win and avg_loss:
                kelly_size = self._calculate_kelly_position(win_rate, avg_win, avg_loss)
            else:
                kelly_size = self._estimate_kelly_position(signal_strength)
            
            # Volatility-adjusted position
            vol_adjusted_size = self._calculate_volatility_adjusted_position(signal_strength, volatility)
            
            # Correlation-adjusted position
            corr_adjusted_size = self._calculate_correlation_adjusted_position(signal_strength, correlation)
            
            # VaR-constrained position
            var_adjusted_size = self._calculate_var_adjusted_position(signal_strength, var_limit)
            
            # Risk parity position
            risk_parity_size = self._calculate_risk_parity_position(signal_strength, volatility)
            
            # Final position size (weighted average)
            final_size = self._calculate_final_position_size(
                kelly_size, vol_adjusted_size, corr_adjusted_size, 
                var_adjusted_size, risk_parity_size
            )
            
            # Apply maximum position size constraint
            final_size = min(final_size, self.max_position_size)
            
            results = {
                'kelly_position': kelly_size,
                'volatility_adjusted': vol_adjusted_size,
                'correlation_adjusted': corr_adjusted_size,
                'var_adjusted': var_adjusted_size,
                'risk_parity': risk_parity_size,
                'final_position': final_size,
                'risk_metrics': {
                    'volatility_contribution': final_size * volatility,
                    'correlation_penalty': 1 - abs(correlation),
                    'var_contribution': final_size * volatility * 1.65  # 95% VaR
                }
            }
            
            # Store position history
            self.position_history.append({
                'timestamp': pd.Timestamp.now(),
                'signal_strength': signal_strength,
                'volatility': volatility,
                'correlation': correlation,
                'position_size': final_size,
                'position_metrics': results
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_kelly_position(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate position size using Kelly Criterion."""
        try:
            if avg_win == 0:
                return 0.0
                
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            return max(0.0, min(kelly_fraction, self.max_position_size))
        except Exception as e:
            logger.error(f"Error calculating Kelly position: {str(e)}")
            return 0.0
    
    def _estimate_kelly_position(self, signal_strength: float) -> float:
        """Estimate Kelly position based on signal strength."""
        try:
            # Estimate win rate and payoffs based on signal strength
            estimated_win_rate = 0.5 + signal_strength * 0.3  # 50% to 80%
            estimated_avg_win = 0.02 + signal_strength * 0.03  # 2% to 5%
            estimated_avg_loss = 0.015 + (1 - signal_strength) * 0.01  # 1.5% to 2.5%
            
            return self._calculate_kelly_position(estimated_win_rate, estimated_avg_win, estimated_avg_loss)
        except Exception as e:
            logger.error(f"Error estimating Kelly position: {str(e)}")
            return 0.0
    
    def _calculate_volatility_adjusted_position(self, signal_strength: float, volatility: float) -> float:
        """Calculate volatility-adjusted position size."""
        try:
            # Inverse volatility weighting
            vol_adjustment = 1 / (1 + volatility * 10)  # Scale volatility
            return signal_strength * vol_adjustment * self.max_position_size
        except Exception as e:
            logger.error(f"Error calculating volatility-adjusted position: {str(e)}")
            return 0.0
    
    def _calculate_correlation_adjusted_position(self, signal_strength: float, correlation: float) -> float:
        """Calculate correlation-adjusted position size."""
        try:
            # Penalty for high correlation
            correlation_penalty = 1 - abs(correlation) * 0.5
            return signal_strength * correlation_penalty * self.max_position_size
        except Exception as e:
            logger.error(f"Error calculating correlation-adjusted position: {str(e)}")
            return 0.0
    
    def _calculate_var_adjusted_position(self, signal_strength: float, var_limit: float) -> float:
        """Calculate VaR-adjusted position size."""
        try:
            # Position size based on VaR limit
            var_adjustment = min(1.0, var_limit / (signal_strength * 0.02))  # Assume 2% base risk
            return signal_strength * var_adjustment * self.max_position_size
        except Exception as e:
            logger.error(f"Error calculating VaR-adjusted position: {str(e)}")
            return 0.0
    
    def _calculate_risk_parity_position(self, signal_strength: float, volatility: float) -> float:
        """Calculate risk parity position size."""
        try:
            # Equal risk contribution
            risk_contribution = self.risk_tolerance / volatility
            return signal_strength * risk_contribution
        except Exception as e:
            logger.error(f"Error calculating risk parity position: {str(e)}")
            return 0.0
    
    def _calculate_final_position_size(self, kelly: float, vol_adj: float, 
                                     corr_adj: float, var_adj: float, risk_parity: float) -> float:
        """Calculate final position size using weighted average."""
        try:
            # Weighted average of different methodologies
            weights = {
                'kelly': 0.3,
                'volatility': 0.2,
                'correlation': 0.2,
                'var': 0.2,
                'risk_parity': 0.1
            }
            
            final_size = (kelly * weights['kelly'] + 
                         vol_adj * weights['volatility'] + 
                         corr_adj * weights['correlation'] + 
                         var_adj * weights['var'] + 
                         risk_parity * weights['risk_parity'])
            
            return max(0.0, final_size)
        except Exception as e:
            logger.error(f"Error calculating final position size: {str(e)}")
            return 0.0
    
    def get_position_summary(self) -> Dict:
        """Get summary of position sizing history."""
        try:
            if not self.position_history:
                return {'total_positions': 0}
            
            summary = {
                'total_positions': len(self.position_history),
                'avg_position_size': np.mean([p['position_size'] for p in self.position_history]),
                'max_position_size': max([p['position_size'] for p in self.position_history]),
                'avg_signal_strength': np.mean([p['signal_strength'] for p in self.position_history]),
                'avg_volatility': np.mean([p['volatility'] for p in self.position_history]),
                'avg_correlation': np.mean([p['correlation'] for p in self.position_history])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting position summary: {str(e)}")
            return {'error': str(e)}

class RiskManager:
    """
    WorldQuant-level risk manager for comprehensive risk analysis and management.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Risk Manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.var_calculator = VaRCalculator(
            confidence_level=config.get('risk', {}).get('var_confidence_level', 0.95),
            time_horizon=config.get('risk', {}).get('var_time_horizon', 1)
        )
        
        self.position_sizer = DynamicPositionSizer(
            max_position_size=config.get('risk', {}).get('max_position_size', 0.02),
            risk_tolerance=config.get('risk', {}).get('risk_tolerance', 0.02)
        )
        
        # Risk metrics storage
        self.risk_metrics_history = []
        self.position_history = []
        self.var_history = []
        
        logger.info("RiskManager initialized")
    
    async def initialize(self) -> bool:
        """Initialize the risk manager."""
        try:
            # Initialize risk tracking
            self.risk_metrics_history = []
            self.position_history = []
            self.var_history = []
            
            logger.info("RiskManager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RiskManager: {str(e)}")
            return False
    
    def calculate_risk_metrics(self, returns: np.array, signal_data: Dict, 
                             position_size: float) -> Dict:
        """Calculate comprehensive risk metrics."""
        try:
            # Calculate VaR
            var_results = self.var_calculator.calculate_var(returns, position_size, 'all')
            
            # Calculate position size
            position_results = self.position_sizer.calculate_position_size(
                signal_data.get('strength', 0.5),
                signal_data.get('volatility', 0.02),
                signal_data.get('correlation', 0.0),
                var_results.get('historical_var', 0.0),
                signal_data.get('win_rate'),
                signal_data.get('avg_win'),
                signal_data.get('avg_loss')
            )
            
            # Combine results
            risk_metrics = {
                'var_metrics': var_results,
                'position_metrics': position_results,
                'overall_risk': {
                    'total_risk': var_results.get('historical_var', 0.0),
                    'position_risk': position_results.get('final_position', 0.0) * signal_data.get('volatility', 0.02),
                    'portfolio_risk': self._calculate_portfolio_risk(var_results, position_results)
                }
            }
            
            # Store risk history
            self.risk_history.append({
                'timestamp': pd.Timestamp.now(),
                'signal_data': signal_data,
                'risk_metrics': risk_metrics
            })
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_portfolio_risk(self, var_results: Dict, position_results: Dict) -> float:
        """Calculate portfolio-level risk."""
        try:
            var_risk = var_results.get('historical_var', 0.0)
            position_risk = position_results.get('final_position', 0.0)
            
            # Simple portfolio risk calculation
            portfolio_risk = float(np.sqrt(var_risk**2 + position_risk**2))
            return portfolio_risk
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {str(e)}")
            return 0.0
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary."""
        try:
            var_summary = self.var_calculator.get_var_summary()
            position_summary = self.position_sizer.get_position_summary()
            
            return {
                'var_summary': var_summary,
                'position_summary': position_summary,
                'total_risk_calculations': len(self.risk_history)
            }
        except Exception as e:
            logger.error(f"Error getting risk summary: {str(e)}")
            return {'error': str(e)}
    
    async def close(self) -> None:
        """Close the risk manager and cleanup resources."""
        try:
            logger.info("Closing RiskManager...")
            
            # Clear history
            self.risk_metrics_history.clear()
            self.position_history.clear()
            self.var_history.clear()
            
            logger.info("RiskManager closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing RiskManager: {str(e)}")
            raise 