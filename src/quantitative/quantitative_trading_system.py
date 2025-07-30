"""
Quantitative Trading System
Advanced quantitative analysis and validation system.
"""

from .statistical_validator import StatisticalValidator
from .risk_manager import RiskManager
from .portfolio_optimizer import PortfolioOptimizer
from .market_microstructure import MarketMicrostructureAnalyzer
from .backtesting_engine import AdvancedBacktestingEngine
from .factor_model import WorldQuantFactorModel
from .ml_ensemble import WorldQuantMLEnsemble
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

class QuantitativeTradingSystem:
    """
    Advanced quantitative trading system with comprehensive analysis capabilities.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize quantitative trading system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.statistical_validator = StatisticalValidator(
            significance_level=config.get('trading', {}).get('statistical_significance_level', 0.05),
            min_sample_size=config.get('trading', {}).get('min_sample_size', 100)
        )
        
        self.risk_manager = RiskManager(config)
        self.portfolio_optimizer = PortfolioOptimizer(config)
        self.market_microstructure = MarketMicrostructureAnalyzer(config)
        self.backtesting_engine = AdvancedBacktestingEngine(config)
        self.factor_model = WorldQuantFactorModel(config)
        self.ml_ensemble = WorldQuantMLEnsemble(config)
        
        # Performance tracking
        self.performance_metrics = {}
        self.validation_history = {}
        
        logger.info("QuantitativeTradingSystem initialized")
    
    async def initialize(self) -> bool:
        """Initialize the quantitative trading system."""
        try:
            # Initialize all components
            await self.risk_manager.initialize()
            await self.portfolio_optimizer.initialize()
            await self.market_microstructure.initialize()
            await self.backtesting_engine.initialize()
            await self.factor_model.initialize()
            
            logger.info("QuantitativeTradingSystem initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing QuantitativeTradingSystem: {str(e)}")
            return False
        
    def analyze_trading_opportunity(self, market_data: Dict, 
                                  signal_data: Dict = None) -> Dict:
        """
        Comprehensive analysis of a trading opportunity.
        
        Args:
            market_data: Market data including prices, volumes, orderbook
            signal_data: Trading signal data
            
        Returns:
            Dict: Comprehensive analysis results
        """
        try:
            results = {
                'timestamp': pd.Timestamp.now(),
                'market_analysis': {},
                'risk_analysis': {},
                'statistical_validation': {},
                'portfolio_analysis': {},
                'factor_analysis': {},
                'trading_recommendation': {}
            }
            
            # Market microstructure analysis
            if 'orderbook' in market_data:
                market_analysis = self.market_microstructure.analyze_market_structure(
                    market_data['orderbook'],
                    market_data.get('trades')
                )
                results['market_analysis'] = market_analysis
            
            # Risk analysis
            if signal_data:
                risk_metrics = self.risk_manager.calculate_risk_metrics(
                    returns=np.array(market_data.get('returns', [0.01])),
                    signal_data=signal_data,
                    position_size=signal_data.get('position_size', 0.01)
                )
                results['risk_analysis'] = risk_metrics
            
            # Statistical validation
            if signal_data:
                validation_results = self.statistical_validator.validate_signal(
                    signal_data,
                    historical_returns=np.array(market_data.get('returns', [0.01]))
                )
                results['statistical_validation'] = validation_results
            
            # Portfolio optimization (if multiple assets available)
            if 'portfolio_data' in market_data:
                portfolio_results = self.portfolio_optimizer.optimize_portfolio(
                    returns=market_data['portfolio_data'],
                    method=self.config.get('optimization_method', 'markowitz')
                )
                results['portfolio_analysis'] = portfolio_results
            
            # Factor analysis
            if 'factor_data' in market_data:
                factor_results = self.factor_model.build_factor_model(
                    returns_data=market_data['factor_data']
                )
                results['factor_analysis'] = factor_results
            
            # Generate trading recommendation
            recommendation = self._generate_trading_recommendation(results)
            results['trading_recommendation'] = recommendation
            
            # Store analysis
            self.trading_history.append(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in trading opportunity analysis: {str(e)}")
            return {'error': str(e)}
    
    def _generate_trading_recommendation(self, analysis_results: Dict) -> Dict:
        """Generate trading recommendation based on comprehensive analysis."""
        try:
            recommendation = {
                'action': 'hold',
                'confidence': 0.0,
                'position_size': 0.0,
                'stop_loss': None,
                'take_profit': None,
                'reasoning': []
            }
            
            # Check if we have valid analysis results
            if not isinstance(analysis_results, dict):
                return recommendation
            
            confidence_score = 0.0
            reasoning = []
            
            # Market analysis contribution
            if 'market_analysis' in analysis_results and isinstance(analysis_results['market_analysis'], dict) and 'error' not in analysis_results['market_analysis']:
                market_analysis = analysis_results['market_analysis']
                
                # Spread analysis
                if 'bid_ask_spread' in market_analysis and isinstance(market_analysis['bid_ask_spread'], dict):
                    spread = market_analysis['bid_ask_spread']
                    if 'relative_spread_bps' in spread and isinstance(spread['relative_spread_bps'], (int, float)):
                        if spread['relative_spread_bps'] < 10:  # Tight spread
                            confidence_score += 0.2
                            reasoning.append('tight_bid_ask_spread')
                        elif spread['relative_spread_bps'] > 50:  # Wide spread
                            confidence_score -= 0.1
                            reasoning.append('wide_bid_ask_spread')
                
                # Market efficiency
                if 'market_efficiency' in market_analysis and isinstance(market_analysis['market_efficiency'], dict):
                    efficiency = market_analysis['market_efficiency']
                    if 'efficiency_score' in efficiency and isinstance(efficiency['efficiency_score'], (int, float)):
                        if efficiency['efficiency_score'] > 0.7:
                            confidence_score += 0.15
                            reasoning.append('high_market_efficiency')
                        elif efficiency['efficiency_score'] < 0.3:
                            confidence_score -= 0.1
                            reasoning.append('low_market_efficiency')
            
            # Risk analysis contribution
            if 'risk_analysis' in analysis_results and isinstance(analysis_results['risk_analysis'], dict) and 'error' not in analysis_results['risk_analysis']:
                risk_analysis = analysis_results['risk_analysis']
                
                # VaR analysis
                if 'var_results' in risk_analysis and isinstance(risk_analysis['var_results'], dict):
                    var_results = risk_analysis['var_results']
                    if 'historical_var' in var_results and isinstance(var_results['historical_var'], (int, float)):
                        var_value = var_results['historical_var']
                        if var_value < 0.02:  # Low risk
                            confidence_score += 0.2
                            reasoning.append('low_var_risk')
                        elif var_value > 0.05:  # High risk
                            confidence_score -= 0.2
                            reasoning.append('high_var_risk')
                
                # Position sizing
                if 'position_results' in risk_analysis and isinstance(risk_analysis['position_results'], dict):
                    position_results = risk_analysis['position_results']
                    if 'final_position_size' in position_results and isinstance(position_results['final_position_size'], (int, float)):
                        position_size = position_results['final_position_size']
                        if position_size > 0.01:  # Significant position
                            confidence_score += 0.1
                            reasoning.append('adequate_position_size')
            
            # Statistical validation contribution
            if 'statistical_validation' in analysis_results and isinstance(analysis_results['statistical_validation'], dict) and 'error' not in analysis_results['statistical_validation']:
                validation = analysis_results['statistical_validation']
                
                if validation.get('is_valid', False):
                    confidence_score += 0.3
                    reasoning.append('statistically_valid_signal')
                else:
                    confidence_score -= 0.3
                    reasoning.append('statistically_invalid_signal')
                
                # Sharpe ratio contribution
                if 'sharpe_ratio' in validation and isinstance(validation['sharpe_ratio'], (int, float)):
                    sharpe = validation['sharpe_ratio']
                    if sharpe > 1.0:
                        confidence_score += 0.2
                        reasoning.append('high_sharpe_ratio')
                    elif sharpe < 0.5:
                        confidence_score -= 0.1
                        reasoning.append('low_sharpe_ratio')
            
            # Factor analysis contribution
            if 'factor_analysis' in analysis_results and isinstance(analysis_results['factor_analysis'], dict) and 'error' not in analysis_results['factor_analysis']:
                factor_analysis = analysis_results['factor_analysis']
                
                if 'factor_timing' in factor_analysis and isinstance(factor_analysis['factor_timing'], dict):
                    timing_results = factor_analysis['factor_timing']
                    positive_factors = 0
                    total_factors = len(timing_results)
                    
                    for factor_name, timing_data in timing_results.items():
                        if isinstance(timing_data, dict) and 'error' not in timing_data:
                            if isinstance(timing_data.get('momentum_1m', 0), (int, float)) and timing_data.get('momentum_1m', 0) > 0:
                                positive_factors += 1
                    
                    if total_factors > 0:
                        factor_score = positive_factors / total_factors
                        if factor_score > 0.6:
                            confidence_score += 0.15
                            reasoning.append('positive_factor_momentum')
                        elif factor_score < 0.4:
                            confidence_score -= 0.1
                            reasoning.append('negative_factor_momentum')
            
            # Determine action based on confidence score
            if confidence_score > 0.5:
                recommendation['action'] = 'buy'
                recommendation['position_size'] = min(confidence_score, 0.1)  # Cap at 10%
            elif confidence_score < -0.3:
                recommendation['action'] = 'sell'
                recommendation['position_size'] = min(abs(confidence_score), 0.1)
            else:
                recommendation['action'] = 'hold'
                recommendation['position_size'] = 0.0
            
            recommendation['confidence'] = abs(confidence_score)
            recommendation['reasoning'] = reasoning
            
            # Set stop loss and take profit based on risk analysis
            if 'risk_analysis' in analysis_results and 'error' not in analysis_results['risk_analysis']:
                risk_analysis = analysis_results['risk_analysis']
                if 'var_results' in risk_analysis:
                    var_value = risk_analysis['var_results'].get('historical_var', 0.02)
                    recommendation['stop_loss'] = -var_value * 2  # 2x VaR
                    recommendation['take_profit'] = var_value * 3  # 3x VaR
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating trading recommendation: {str(e)}")
            return {'action': 'hold', 'confidence': 0.0, 'error': str(e)}
    
    def run_backtest(self, strategy_function: Callable, historical_data: pd.DataFrame,
                    strategy_params: Dict = None) -> Dict:
        """
        Run comprehensive backtest with quantitative analysis.
        
        Args:
            strategy_function: Function that generates trading signals
            historical_data: Historical market data
            strategy_params: Strategy parameters
            
        Returns:
            Dict: Backtest results with quantitative analysis
        """
        try:
            # Run backtest
            backtest_results = self.backtesting_engine.run_backtest(
                strategy_function, historical_data, strategy_params
            )
            
            # Add quantitative analysis to backtest results
            if 'error' not in backtest_results:
                # Perform factor analysis on backtest data
                if len(historical_data) > 100:  # Need sufficient data
                    factor_results = self.factor_model.build_factor_model(historical_data)
                    backtest_results['factor_analysis'] = factor_results
                
                # Add risk analysis
                returns_series = pd.Series(backtest_results.get('daily_returns', []))
                if len(returns_series) > 0:
                    risk_metrics = self.risk_manager.calculate_risk_metrics(
                        returns=returns_series.values,
                        signal_data={'position_size': 0.01},
                        position_size=0.01
                    )
                    backtest_results['risk_analysis'] = risk_metrics
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error in quantitative backtest: {str(e)}")
            return {'error': str(e)}
    
    def optimize_portfolio(self, returns_data: pd.DataFrame, 
                          method: str = 'markowitz') -> Dict:
        """
        Optimize portfolio with quantitative analysis.
        
        Args:
            returns_data: Asset returns data
            method: Optimization method
            
        Returns:
            Dict: Portfolio optimization results
        """
        try:
            # Perform portfolio optimization
            optimization_results = self.portfolio_optimizer.optimize_portfolio(
                returns_data, method
            )
            
            # Add factor analysis
            factor_results = self.factor_model.build_factor_model(returns_data)
            optimization_results['factor_analysis'] = factor_results
            
            # Add risk analysis
            if 'weights' in optimization_results:
                portfolio_returns = returns_data.dot(list(optimization_results['weights'].values()))
                risk_metrics = self.risk_manager.calculate_risk_metrics(
                    returns=portfolio_returns.values,
                    signal_data={'position_size': 1.0},
                    position_size=1.0
                )
                optimization_results['risk_analysis'] = risk_metrics
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {str(e)}")
            return {'error': str(e)}
    
    def get_system_summary(self) -> Dict:
        """Get comprehensive system summary."""
        try:
            summary = {
                'total_analyses': len(self.trading_history),
                'risk_management_summary': self.risk_manager.get_risk_summary(),
                'statistical_validation_summary': self.statistical_validator.get_validation_summary(),
                'portfolio_optimization_summary': self.portfolio_optimizer.get_optimization_summary(),
                'market_analysis_summary': self.market_microstructure.get_analysis_summary(),
                'backtesting_summary': self.backtesting_engine.get_backtest_summary(),
                'factor_analysis_summary': self.factor_model.get_factor_summary(),
                'recent_analyses': self.trading_history[-5:] if self.trading_history else []
            }
            
            # Calculate overall system performance
            if self.trading_history:
                recent_analyses = self.trading_history[-10:]
                successful_recommendations = 0
                total_recommendations = 0
                
                for analysis in recent_analyses:
                    if 'trading_recommendation' in analysis:
                        recommendation = analysis['trading_recommendation']
                        if recommendation.get('confidence', 0) > 0.5:
                            total_recommendations += 1
                            if recommendation.get('action') != 'hold':
                                successful_recommendations += 1
                
                if total_recommendations > 0:
                    summary['recommendation_success_rate'] = successful_recommendations / total_recommendations
                else:
                    summary['recommendation_success_rate'] = 0.0
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting system summary: {str(e)}")
            return {'error': str(e)}
    
    def get_performance_metrics(self) -> Dict:
        """
        Get comprehensive performance metrics for the quantitative system.
        
        Returns:
            Dict: Performance metrics including analysis counts, success rates, etc.
        """
        try:
            metrics = {
                'total_analyses': len(self.trading_history),
                'recent_analyses_count': len([h for h in self.trading_history 
                                            if (pd.Timestamp.now() - h['timestamp']).days <= 7]),
                'system_status': 'active',
                'risk_management_metrics': self.risk_manager.get_risk_summary(),
                'statistical_validation_metrics': self.statistical_validator.get_validation_summary(),
                'portfolio_optimization_metrics': self.portfolio_optimizer.get_optimization_summary(),
                'market_analysis_metrics': self.market_microstructure.get_analysis_summary(),
                'backtesting_metrics': self.backtesting_engine.get_backtest_summary(),
                'factor_analysis_metrics': self.factor_model.get_factor_summary()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {'error': str(e)}
    
    async def validate_signal(self, signal_data: Dict, market_data: Dict) -> Dict:
        """
        Validate trading signal using statistical methods.
        
        Args:
            signal_data: Dictionary containing signal information
            market_data: Market data including returns for validation
            
        Returns:
            Dict: Validation results with statistical metrics
        """
        try:
            # Extract returns from market data
            historical_returns = None
            if 'returns' in market_data and len(market_data['returns']) > 0:
                historical_returns = np.array(market_data['returns'])
            
            # Use the statistical validator to validate the signal
            validation_results = self.statistical_validator.validate_signal(
                signal_data, 
                historical_returns
            )
            
            # Add additional quantitative validation
            validation_results['quantitative_validation'] = {
                'signal_strength': signal_data.get('strength', 0.0),
                'confidence': signal_data.get('confidence', 0.0),
                'position_size': signal_data.get('position_size', 0.01),
                'risk_metrics': self.risk_manager.calculate_risk_metrics(
                    returns=historical_returns if historical_returns is not None else np.array([0.01]),
                    signal_data=signal_data,
                    position_size=signal_data.get('position_size', 0.01)
                ) if historical_returns is not None else {}
            }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating signal: {str(e)}")
            return {'is_valid': False, 'error': str(e)}
    
    async def get_recommendations(self, symbol: str) -> Dict:
        """
        Get trading recommendations for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict: Trading recommendations with analysis
        """
        try:
            recommendations = {
                'symbol': symbol,
                'timestamp': pd.Timestamp.now(),
                'market_analysis': {},
                'risk_assessment': {},
                'trading_recommendation': {
                    'action': 'hold',
                    'confidence': 0.0,
                    'position_size': 0.01,
                    'reasoning': []
                }
            }
            
            # Get market data for analysis
            # This would typically fetch real market data
            # For now, return a basic recommendation structure
            
            # Add basic market analysis
            recommendations['market_analysis'] = {
                'volatility': 0.02,
                'trend': 'neutral',
                'support_level': 0.0,
                'resistance_level': 0.0
            }
            
            # Add risk assessment
            recommendations['risk_assessment'] = {
                'var_95': -0.02,
                'max_drawdown': 0.05,
                'sharpe_ratio': 0.5,
                'risk_level': 'medium'
            }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def export_analysis_report(self, analysis_id: str = None) -> Dict:
        """Export comprehensive analysis report."""
        try:
            if analysis_id is None and self.trading_history:
                # Export latest analysis
                latest_analysis = self.trading_history[-1]
            else:
                # Find specific analysis
                latest_analysis = None
                for analysis in self.trading_history:
                    if analysis.get('timestamp') == analysis_id:
                        latest_analysis = analysis
                        break
            
            if latest_analysis is None:
                return {'error': 'Analysis not found'}
            
            report = {
                'analysis_timestamp': latest_analysis['timestamp'],
                'market_analysis': latest_analysis.get('market_analysis', {}),
                'risk_analysis': latest_analysis.get('risk_analysis', {}),
                'statistical_validation': latest_analysis.get('statistical_validation', {}),
                'portfolio_analysis': latest_analysis.get('portfolio_analysis', {}),
                'factor_analysis': latest_analysis.get('factor_analysis', {}),
                'trading_recommendation': latest_analysis.get('trading_recommendation', {}),
                'system_summary': self.get_system_summary()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error exporting analysis report: {str(e)}")
            return {'error': str(e)} 