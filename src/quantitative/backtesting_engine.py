import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AdvancedBacktestingEngine:
    """
    Advanced backtesting engine for quantitative trading strategies.
    Implements realistic market simulation, transaction costs, and risk management.
    """
    
    def __init__(self, initial_capital: float = 100000, commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005, risk_free_rate: float = 0.02):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.risk_free_rate = risk_free_rate
        self.backtest_results = []
        
    def run_backtest(self, strategy_function: Callable, historical_data: pd.DataFrame,
                    strategy_params: Dict = None, risk_management: Dict = None) -> Dict:
        """
        Run comprehensive backtest of a trading strategy.
        
        Args:
            strategy_function: Function that generates trading signals
            historical_data: DataFrame with OHLCV data
            strategy_params: Parameters for the strategy
            risk_management: Risk management parameters
            
        Returns:
            Dict: Comprehensive backtest results
        """
        try:
            # Initialize backtest state
            portfolio = self._initialize_portfolio()
            positions = {}
            trades = []
            daily_returns = []
            equity_curve = []
            
            # Process historical data
            for i, (timestamp, row) in enumerate(historical_data.iterrows()):
                # Generate signal
                signal_data = strategy_function(row, historical_data.iloc[:i+1], strategy_params)
                
                # Apply risk management
                if risk_management:
                    signal_data = self._apply_risk_management(signal_data, portfolio, risk_management)
                
                # Execute trades
                new_trades = self._execute_trades(signal_data, row, portfolio, positions)
                trades.extend(new_trades)
                
                # Update portfolio
                self._update_portfolio(portfolio, positions, row)
                
                # Calculate daily metrics
                daily_return = self._calculate_daily_return(portfolio)
                daily_returns.append(daily_return)
                equity_curve.append(portfolio['total_value'])
            
            # Calculate comprehensive results
            results = self._calculate_backtest_results(
                daily_returns, equity_curve, trades, portfolio, historical_data
            )
            
            self._store_backtest_result(results)
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            return {'error': str(e)}
    
    def _initialize_portfolio(self) -> Dict:
        """Initialize portfolio state."""
        return {
            'cash': self.initial_capital,
            'total_value': self.initial_capital,
            'positions': {},
            'unrealized_pnl': 0,
            'realized_pnl': 0,
            'total_commission': 0,
            'total_slippage': 0
        }
    
    def _apply_risk_management(self, signal_data: Dict, portfolio: Dict, risk_params: Dict) -> Dict:
        """Apply risk management rules to trading signals."""
        try:
            modified_signal = signal_data.copy()
            
            # Position sizing based on risk
            if 'max_position_size' in risk_params:
                max_size = risk_params['max_position_size']
                if 'position_size' in modified_signal:
                    modified_signal['position_size'] = min(
                        modified_signal['position_size'], max_size
                    )
            
            # VaR-based position sizing
            if 'var_limit' in risk_params and 'var_estimate' in modified_signal:
                var_limit = risk_params['var_limit']
                var_estimate = modified_signal['var_estimate']
                if var_estimate > var_limit:
                    # Reduce position size based on VaR
                    reduction_factor = var_limit / var_estimate
                    if 'position_size' in modified_signal:
                        modified_signal['position_size'] *= reduction_factor
            
            # Maximum drawdown protection
            if 'max_drawdown' in risk_params:
                current_drawdown = self._calculate_current_drawdown(portfolio)
                if current_drawdown > risk_params['max_drawdown']:
                    # Stop trading if drawdown exceeds limit
                    modified_signal['action'] = 'hold'
            
            # Correlation-based position limits
            if 'max_correlation' in risk_params and 'correlation' in modified_signal:
                if modified_signal['correlation'] > risk_params['max_correlation']:
                    modified_signal['position_size'] *= 0.5  # Reduce position size
            
            return modified_signal
            
        except Exception as e:
            logger.error(f"Error applying risk management: {str(e)}")
            return signal_data
    
    def _execute_trades(self, signal_data: Dict, market_data: pd.Series, 
                       portfolio: Dict, positions: Dict) -> List[Dict]:
        """Execute trades based on signals."""
        trades = []
        
        try:
            if 'action' not in signal_data or signal_data['action'] == 'hold':
                return trades
            
            symbol = signal_data.get('symbol', 'default')
            action = signal_data['action']
            size = signal_data.get('position_size', 0)
            
            if action == 'buy' and size > 0:
                # Calculate trade details
                price = market_data['close']
                slippage = price * self.slippage_rate
                execution_price = price + slippage
                
                # Calculate position size in currency
                position_value = size * portfolio['total_value']
                shares = position_value / execution_price
                
                # Calculate costs
                commission = position_value * self.commission_rate
                total_cost = position_value + commission
                
                # Check if we have enough cash
                if total_cost <= portfolio['cash']:
                    # Execute trade
                    if symbol not in positions:
                        positions[symbol] = {'shares': 0, 'avg_price': 0}
                    
                    # Update position
                    old_shares = positions[symbol]['shares']
                    old_avg_price = positions[symbol]['avg_price']
                    
                    new_shares = old_shares + shares
                    new_avg_price = ((old_shares * old_avg_price) + (shares * execution_price)) / new_shares
                    
                    positions[symbol] = {
                        'shares': new_shares,
                        'avg_price': new_avg_price
                    }
                    
                    # Update portfolio
                    portfolio['cash'] -= total_cost
                    portfolio['total_commission'] += commission
                    portfolio['total_slippage'] += slippage * shares
                    
                    # Record trade
                    trades.append({
                        'timestamp': market_data.name,
                        'symbol': symbol,
                        'action': 'buy',
                        'shares': shares,
                        'price': execution_price,
                        'value': position_value,
                        'commission': commission,
                        'slippage': slippage * shares
                    })
            
            elif action == 'sell' and symbol in positions:
                # Calculate trade details
                price = market_data['close']
                slippage = price * self.slippage_rate
                execution_price = price - slippage
                
                # Sell entire position or specified size
                shares_to_sell = positions[symbol]['shares']
                if 'position_size' in signal_data and signal_data['position_size'] < 1:
                    shares_to_sell *= signal_data['position_size']
                
                position_value = shares_to_sell * execution_price
                commission = position_value * self.commission_rate
                net_proceeds = position_value - commission
                
                # Update position
                remaining_shares = positions[symbol]['shares'] - shares_to_sell
                if remaining_shares <= 0:
                    # Close entire position
                    realized_pnl = (execution_price - positions[symbol]['avg_price']) * shares_to_sell
                    portfolio['realized_pnl'] += realized_pnl
                    del positions[symbol]
                else:
                    positions[symbol]['shares'] = remaining_shares
                
                # Update portfolio
                portfolio['cash'] += net_proceeds
                portfolio['total_commission'] += commission
                portfolio['total_slippage'] += slippage * shares_to_sell
                
                # Record trade
                trades.append({
                    'timestamp': market_data.name,
                    'symbol': symbol,
                    'action': 'sell',
                    'shares': shares_to_sell,
                    'price': execution_price,
                    'value': position_value,
                    'commission': commission,
                    'slippage': slippage * shares_to_sell
                })
        
        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}")
        
        return trades
    
    def _update_portfolio(self, portfolio: Dict, positions: Dict, market_data: pd.Series):
        """Update portfolio values based on current market prices."""
        try:
            total_position_value = 0
            unrealized_pnl = 0
            
            for symbol, position in positions.items():
                current_price = market_data['close']
                position_value = position['shares'] * current_price
                total_position_value += position_value
                
                # Calculate unrealized P&L
                position_pnl = (current_price - position['avg_price']) * position['shares']
                unrealized_pnl += position_pnl
            
            portfolio['positions'] = positions
            portfolio['unrealized_pnl'] = unrealized_pnl
            portfolio['total_value'] = portfolio['cash'] + total_position_value
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {str(e)}")
    
    def _calculate_daily_return(self, portfolio: Dict) -> float:
        """Calculate daily portfolio return."""
        try:
            if portfolio['total_value'] > 0:
                return (portfolio['total_value'] - self.initial_capital) / self.initial_capital
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating daily return: {str(e)}")
            return 0.0
    
    def _calculate_current_drawdown(self, portfolio: Dict) -> float:
        """Calculate current drawdown."""
        try:
            if portfolio['total_value'] > 0:
                return (self.initial_capital - portfolio['total_value']) / self.initial_capital
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating drawdown: {str(e)}")
            return 0.0
    
    def _calculate_backtest_results(self, daily_returns: List[float], equity_curve: List[float], trades: List[Dict], portfolio: Dict, historical_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive backtest results."""
        try:
            returns_series = pd.Series(daily_returns)
            equity_series = pd.Series(equity_curve)
            # Basic performance metrics
            if len(returns_series) < 2 or returns_series.isna().all():
                return {}
            total_return = (portfolio['total_value'] - self.initial_capital) / self.initial_capital if self.initial_capital != 0 else 0
            annualized_return = self._calculate_annualized_return(returns_series) if len(returns_series) > 1 else 0
            volatility = returns_series.std() * np.sqrt(252) if len(returns_series) > 1 else 0
            sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            # Risk metrics
            max_drawdown = self._calculate_max_drawdown(equity_series) if len(equity_series) > 1 else 0
            var_95 = np.percentile(returns_series, 5) if len(returns_series) > 1 else 0
            cvar_95 = returns_series[returns_series <= var_95].mean() if len(returns_series) > 1 else 0
            # Trading metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            # Cost analysis
            total_commission = portfolio['total_commission']
            total_slippage = portfolio['total_slippage']
            total_costs = total_commission + total_slippage
            # Calculate average trade metrics
            if trades:
                trade_pnls = [t.get('pnl', 0) for t in trades]
                avg_trade_pnl = np.mean(trade_pnls) if len(trade_pnls) > 0 else 0
                avg_winning_trade = np.mean([p for p in trade_pnls if p > 0]) if any(p > 0 for p in trade_pnls) else 0
                avg_losing_trade = np.mean([p for p in trade_pnls if p < 0]) if any(p < 0 for p in trade_pnls) else 0
            else:
                avg_trade_pnl = 0
                avg_winning_trade = 0
                avg_losing_trade = 0
            profit_factor = abs(avg_winning_trade / avg_losing_trade) if avg_losing_trade != 0 else float('inf')
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            cost_impact = total_costs / self.initial_capital if self.initial_capital != 0 else 0
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': self._calculate_sortino_ratio(returns_series) if len(returns_series) > 1 else 0,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_trade_pnl': avg_trade_pnl,
                'avg_winning_trade': avg_winning_trade,
                'avg_losing_trade': avg_losing_trade,
                'profit_factor': profit_factor,
                'total_commission': total_commission,
                'total_slippage': total_slippage,
                'total_costs': total_costs,
                'cost_impact': cost_impact,
                'final_portfolio_value': portfolio['total_value'],
                'equity_curve': equity_curve,
                'daily_returns': daily_returns,
                'trades': trades
            }
        except Exception as e:
            logger.error(f"Error calculating backtest results: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_annualized_return(self, returns_series: pd.Series) -> float:
        """Calculate annualized return."""
        try:
            total_return = (1 + returns_series).prod() - 1
            years = len(returns_series) / 252
            return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        except Exception as e:
            logger.error(f"Error calculating annualized return: {str(e)}")
            return 0.0
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak
            return float(abs(drawdown.min()))
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0
    
    def _calculate_sortino_ratio(self, returns_series: pd.Series) -> float:
        """Calculate Sortino ratio."""
        try:
            negative_returns = returns_series[returns_series < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252)
            annualized_return = self._calculate_annualized_return(returns_series)
            return (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0.0
    
    def _store_backtest_result(self, result: Dict):
        """Store backtest result in history."""
        self.backtest_results.append({
            'timestamp': pd.Timestamp.now(),
            'result': result
        })
    
    def get_backtest_summary(self) -> Dict:
        """Get summary of all backtest results."""
        if not self.backtest_results:
            return {'message': 'No backtest history available'}
        
        summary = {
            'total_backtests': len(self.backtest_results),
            'average_sharpe_ratios': [],
            'average_returns': [],
            'average_drawdowns': [],
            'best_performing_backtest': None,
            'recent_results': []
        }
        
        for record in self.backtest_results:
            result = record['result']
            
            if 'sharpe_ratio' in result:
                summary['average_sharpe_ratios'].append(result['sharpe_ratio'])
            
            if 'total_return' in result:
                summary['average_returns'].append(result['total_return'])
            
            if 'max_drawdown' in result:
                summary['average_drawdowns'].append(result['max_drawdown'])
        
        if summary['average_sharpe_ratios']:
            summary['avg_sharpe_ratio'] = np.mean(summary['average_sharpe_ratios'])
            summary['best_sharpe_ratio'] = max(summary['average_sharpe_ratios'])
        
        if summary['average_returns']:
            summary['avg_return'] = np.mean(summary['average_returns'])
            summary['best_return'] = max(summary['average_returns'])
        
        if summary['average_drawdowns']:
            summary['avg_drawdown'] = np.mean(summary['average_drawdowns'])
            summary['worst_drawdown'] = max(summary['average_drawdowns'])
        
        # Find best performing backtest
        if summary['average_sharpe_ratios']:
            best_index = summary['average_sharpe_ratios'].index(summary['best_sharpe_ratio'])
            summary['best_performing_backtest'] = self.backtest_results[best_index]
        
        # Get recent results
        summary['recent_results'] = self.backtest_results[-5:]
        
        return summary 