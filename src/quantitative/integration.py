"""
Integration module for Quantitative Trading System with existing trading bot.
Provides seamless integration of quantitative analysis into the trading strategy.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .quantitative_trading_system import QuantitativeTradingSystem
from .risk_manager import RiskManager
from .statistical_validator import StatisticalSignalValidator
from .portfolio_optimizer import PortfolioOptimizer
from .market_microstructure import MarketMicrostructureAnalyzer
from .factor_model import FactorModel

logger = logging.getLogger(__name__)

class QuantitativeIntegration:
    """
    Integration layer for Quantitative Trading System with existing trading bot.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize quantitative integration.
        
        Args:
            config: Configuration for quantitative system
        """
        self.config = config or {}
        
        # Initialize quantitative trading system
        self.quantitative_system = QuantitativeTradingSystem(config)
        
        # Initialize individual components for direct access
        self.risk_manager = self.quantitative_system.risk_manager
        self.statistical_validator = self.quantitative_system.statistical_validator
        self.portfolio_optimizer = self.quantitative_system.portfolio_optimizer
        self.market_analyzer = self.quantitative_system.market_analyzer
        self.factor_model = self.quantitative_system.factor_model
        
        # Integration state
        self.integration_enabled = self.config.get('quantitative_integration_enabled', True)
        self.analysis_cache = {}
        self.last_analysis_time = {}
    
    async def initialize(self) -> bool:
        """
        Initialize the quantitative integration system.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing Quantitative Integration...")
            
            # Initialize quantitative trading system
            # Note: QuantitativeTradingSystem doesn't have an initialize method,
            # so we just verify the components are available
            if not hasattr(self, 'quantitative_system'):
                logger.error("Quantitative trading system not available")
                return False
            
            # Verify all components are available
            required_components = [
                'risk_manager',
                'statistical_validator', 
                'portfolio_optimizer',
                'market_analyzer',
                'factor_model'
            ]
            
            for component in required_components:
                if not hasattr(self.quantitative_system, component):
                    logger.error(f"Required component {component} not available")
                    return False
            
            logger.info("Quantitative Integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Quantitative Integration: {str(e)}")
            return False
        
    async def enhance_trading_signal(self, symbol: str, base_signal: Dict, 
                                   market_data: Dict) -> Dict:
        """
        Enhance trading signal with quantitative analysis.
        
        Args:
            symbol: Trading symbol
            base_signal: Original trading signal from strategy
            market_data: Market data for analysis
            
        Returns:
            Dict: Enhanced signal with quantitative analysis
        """
        try:
            if not self.integration_enabled:
                return base_signal
            
            # Prepare market data for quantitative analysis
            quantitative_market_data = self._prepare_market_data(market_data)
            
            # Run comprehensive quantitative analysis
            analysis_results = await self._run_quantitative_analysis(
                symbol, quantitative_market_data, base_signal
            )
            
            # Enhance signal with quantitative insights
            enhanced_signal = self._enhance_signal_with_analysis(
                base_signal, analysis_results
            )
            
            # Cache analysis results
            self._cache_analysis_results(symbol, analysis_results)
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error enhancing trading signal for {symbol}: {str(e)}")
            return base_signal
    
    async def validate_signal_quantitatively(self, symbol: str, signal: Dict, 
                                           market_data: Dict) -> Dict:
        """
        Validate trading signal using quantitative methods.
        
        Args:
            symbol: Trading symbol
            signal: Trading signal to validate
            market_data: Market data
            
        Returns:
            Dict: Validation results
        """
        try:
            # Prepare returns data for statistical validation
            returns_data = self._extract_returns_from_market_data(market_data)
            
            # Validate signal statistically
            validation_results = self.statistical_validator.validate_signal(
                signal, returns_data
            )
            
            # Add risk analysis
            risk_metrics = self.risk_manager.calculate_risk_metrics(
                returns=returns_data,
                signal_data=signal,
                position_size=signal.get('position_size', 0.01)
            )
            
            return {
                'statistical_validation': validation_results,
                'risk_analysis': risk_metrics,
                'is_valid': validation_results.get('is_valid', False),
                'confidence_score': self._calculate_confidence_score(validation_results, risk_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error validating signal for {symbol}: {str(e)}")
            return {'is_valid': False, 'error': str(e)}
    
    async def optimize_position_size(self, symbol: str, base_size: float, 
                                   market_data: Dict, signal: Dict) -> float:
        """
        Optimize position size using quantitative methods.
        
        Args:
            symbol: Trading symbol
            base_size: Base position size
            market_data: Market data
            signal: Trading signal
            
        Returns:
            float: Optimized position size
        """
        try:
            # Extract volatility and correlation data
            volatility = self._extract_volatility(market_data)
            correlation = self._extract_correlation(market_data)
            
            # Calculate VaR-based position sizing
            position_results = self.risk_manager.position_sizer.calculate_position_size(
                signal_strength=signal.get('signal_strength', 0.5),
                volatility=volatility,
                correlation=correlation,
                var_limit=self.config.get('var_limit', 0.02),
                win_rate=signal.get('win_rate', 0.5),
                avg_win=signal.get('avg_win', 0.02),
                avg_loss=signal.get('avg_loss', 0.01)
            )
            
            optimized_size = position_results.get('final_position', base_size)
            
            # Apply additional constraints
            max_size = self.config.get('max_position_size', 0.02)
            optimized_size = min(optimized_size, max_size)
            
            return optimized_size
            
        except Exception as e:
            logger.error(f"Error optimizing position size for {symbol}: {str(e)}")
            return base_size
    
    async def analyze_market_microstructure(self, symbol: str, 
                                          market_data: Dict) -> Dict:
        """
        Analyze market microstructure for trading decisions.
        
        Args:
            symbol: Trading symbol
            market_data: Market data including orderbook
            
        Returns:
            Dict: Microstructure analysis results
        """
        try:
            # Extract orderbook data
            orderbook_data = market_data.get('orderbook', {})
            trade_data = market_data.get('trades')
            
            if not orderbook_data:
                return {'error': 'No orderbook data available'}
            
            # Run microstructure analysis
            analysis_results = self.market_analyzer.analyze_market_structure(
                orderbook_data, trade_data
            )
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing market microstructure for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    async def get_portfolio_optimization(self, symbols: List[str], 
                                       returns_data: Dict) -> Dict:
        """
        Get portfolio optimization recommendations.
        
        Args:
            symbols: List of trading symbols
            returns_data: Returns data for all symbols
            
        Returns:
            Dict: Portfolio optimization results
        """
        try:
            # Prepare returns DataFrame
            returns_df = pd.DataFrame(returns_data)
            
            # Run portfolio optimization
            optimization_results = self.portfolio_optimizer.optimize_portfolio(
                returns_df, method=self.config.get('optimization_method', 'markowitz')
            )
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {str(e)}")
            return {'error': str(e)}
    
    async def analyze_factor_exposures(self, symbols: List[str], 
                                     returns_data: Dict) -> Dict:
        """
        Analyze factor exposures for portfolio.
        
        Args:
            symbols: List of trading symbols
            returns_data: Returns data for all symbols
            
        Returns:
            Dict: Factor analysis results
        """
        try:
            # Prepare returns DataFrame
            returns_df = pd.DataFrame(returns_data)
            
            # Run factor analysis
            factor_results = self.factor_model.build_factor_model(returns_df)
            
            return factor_results
            
        except Exception as e:
            logger.error(f"Error in factor analysis: {str(e)}")
            return {'error': str(e)}
    
    async def get_quantitative_recommendations(self, symbol: str, 
                                             market_data: Dict, 
                                             current_positions: List[Dict]) -> Dict:
        """
        Get comprehensive quantitative recommendations.
        
        Args:
            symbol: Trading symbol
            market_data: Market data
            current_positions: Current open positions
            
        Returns:
            Dict: Quantitative recommendations
        """
        try:
            # Prepare market data
            quantitative_market_data = self._prepare_market_data(market_data)
            
            # Create signal data based on current market conditions
            signal_data = self._create_signal_data(symbol, market_data, current_positions)
            
            # Run comprehensive analysis (not await since it's not async)
            analysis_results = self.quantitative_system.analyze_trading_opportunity(
                quantitative_market_data, signal_data
            )
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error getting quantitative recommendations for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def _prepare_market_data(self, market_data: Dict) -> Dict:
        """Prepare market data for quantitative analysis."""
        try:
            quantitative_data = {
                'returns': market_data.get('returns', []),
                'portfolio_data': market_data.get('portfolio_returns', {}),
                'factor_data': market_data.get('factor_returns', {})
            }
            
            # Add orderbook data if available
            if 'orderbook' in market_data:
                quantitative_data['orderbook'] = market_data['orderbook']
            
            # Add trade data if available
            if 'trades' in market_data:
                quantitative_data['trades'] = market_data['trades']
            
            return quantitative_data
            
        except Exception as e:
            logger.error(f"Error preparing market data: {str(e)}")
            return {}
    
    def _extract_returns_from_market_data(self, market_data: Dict) -> np.array:
        """Extract returns data from market data."""
        try:
            returns = market_data.get('returns', [])
            if isinstance(returns, list):
                return np.array(returns)
            elif isinstance(returns, pd.Series):
                return returns.values
            else:
                return np.array([0.01])  # Default return
        except Exception as e:
            logger.error(f"Error extracting returns: {str(e)}")
            return np.array([0.01])
    
    def _extract_volatility(self, market_data: Dict) -> float:
        """Extract volatility from market data."""
        try:
            returns = self._extract_returns_from_market_data(market_data)
            return np.std(returns) if len(returns) > 1 else 0.02
        except Exception as e:
            logger.error(f"Error extracting volatility: {str(e)}")
            return 0.02
    
    def _extract_correlation(self, market_data: Dict) -> float:
        """Extract correlation from market data."""
        try:
            # This would typically come from correlation analysis
            return market_data.get('correlation', 0.3)
        except Exception as e:
            logger.error(f"Error extracting correlation: {str(e)}")
            return 0.3
    
    def _enhance_signal_with_analysis(self, base_signal: Dict, 
                                    analysis_results: Dict) -> Dict:
        """Enhance base signal with quantitative analysis."""
        try:
            enhanced_signal = base_signal.copy()
            
            # Add quantitative confidence
            if 'trading_recommendation' in analysis_results:
                recommendation = analysis_results['trading_recommendation']
                enhanced_signal['quantitative_confidence'] = recommendation.get('confidence', 0.0)
                enhanced_signal['quantitative_action'] = recommendation.get('action', 'hold')
                enhanced_signal['quantitative_reasoning'] = recommendation.get('reasoning', [])
            
            # Add risk metrics
            if 'risk_analysis' in analysis_results:
                risk_analysis = analysis_results['risk_analysis']
                enhanced_signal['var_estimate'] = risk_analysis.get('var_metrics', {}).get('historical_var', 0.0)
                enhanced_signal['position_risk'] = risk_analysis.get('position_metrics', {}).get('final_position', 0.0)
            
            # Add statistical validation
            if 'statistical_validation' in analysis_results:
                validation = analysis_results['statistical_validation']
                enhanced_signal['statistical_valid'] = validation.get('is_valid', False)
                enhanced_signal['sharpe_ratio'] = validation.get('sharpe_ratio', 0.0)
                enhanced_signal['p_value'] = validation.get('p_value', 1.0)
            
            # Add market microstructure insights
            if 'market_analysis' in analysis_results:
                market_analysis = analysis_results['market_analysis']
                if 'bid_ask_spread' in market_analysis:
                    spread = market_analysis['bid_ask_spread']
                    enhanced_signal['spread_bps'] = spread.get('relative_spread_bps', 0.0)
                    enhanced_signal['market_efficiency'] = market_analysis.get('market_efficiency', {}).get('efficiency_score', 0.0)
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error enhancing signal: {str(e)}")
            return base_signal
    
    def _calculate_confidence_score(self, validation_results: Dict, 
                                  risk_metrics: Dict) -> float:
        """Calculate overall confidence score."""
        try:
            confidence = 0.0
            
            # Statistical validation contribution
            if validation_results.get('is_valid', False):
                confidence += 0.3
            
            sharpe_ratio = validation_results.get('sharpe_ratio', 0.0)
            if sharpe_ratio > 1.0:
                confidence += 0.2
            elif sharpe_ratio > 0.5:
                confidence += 0.1
            
            # Risk metrics contribution
            var_estimate = risk_metrics.get('var_metrics', {}).get('historical_var', 0.02)
            if var_estimate < 0.01:
                confidence += 0.2
            elif var_estimate < 0.02:
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.0
    
    def _create_signal_data(self, symbol: str, market_data: Dict, 
                           current_positions: List[Dict]) -> Dict:
        """Create signal data for quantitative analysis."""
        try:
            # Find current position for this symbol
            current_position = None
            for position in current_positions:
                if position.get('symbol') == symbol:
                    current_position = position
                    break
            
            signal_data = {
                'symbol': symbol,
                'signal_strength': 0.5,  # Default
                'signal_type': 'quantitative',
                'position_size': current_position.get('size', 0.01) if current_position else 0.01,
                'confidence': 0.5,
                'current_price': market_data.get('current_price', 0.0),
                'volatility': self._extract_volatility(market_data),
                'correlation': self._extract_correlation(market_data)
            }
            
            return signal_data
            
        except Exception as e:
            logger.error(f"Error creating signal data: {str(e)}")
            return {'symbol': symbol, 'signal_strength': 0.5}
    
    def _cache_analysis_results(self, symbol: str, analysis_results: Dict) -> None:
        """Cache analysis results for future use."""
        try:
            self.analysis_cache[symbol] = {
                'timestamp': datetime.now(),
                'results': analysis_results
            }
            
            # Clean up old cache entries
            self._cleanup_cache()
            
        except Exception as e:
            logger.error(f"Error caching analysis results: {str(e)}")
    
    def _cleanup_cache(self) -> None:
        """Clean up old cache entries."""
        try:
            current_time = datetime.now()
            expired_keys = []
            
            for symbol, cache_entry in self.analysis_cache.items():
                if (current_time - cache_entry['timestamp']).seconds > 300:  # 5 minutes
                    expired_keys.append(symbol)
            
            for key in expired_keys:
                del self.analysis_cache[key]
                
        except Exception as e:
            logger.error(f"Error cleaning up cache: {str(e)}")
    
    async def _run_quantitative_analysis(self, symbol: str, market_data: Dict, 
                                       signal: Dict) -> Dict:
        """Run comprehensive quantitative analysis."""
        try:
            # Check cache first
            if symbol in self.analysis_cache:
                cache_entry = self.analysis_cache[symbol]
                if (datetime.now() - cache_entry['timestamp']).seconds < 60:  # 1 minute cache
                    return cache_entry['results']
            
            # Run analysis (not async)
            analysis_results = self.quantitative_system.analyze_trading_opportunity(
                market_data, signal
            )
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error running quantitative analysis: {str(e)}")
            return {'error': str(e)}
    
    def get_integration_status(self) -> Dict:
        """Get integration status and statistics."""
        try:
            return {
                'integration_enabled': self.integration_enabled,
                'cache_size': len(self.analysis_cache),
                'quantitative_system_status': 'active',
                'components_loaded': [
                    'risk_manager',
                    'statistical_validator', 
                    'portfolio_optimizer',
                    'market_analyzer',
                    'factor_model'
                ]
            }
        except Exception as e:
            logger.error(f"Error getting integration status: {str(e)}")
            return {'error': str(e)} 