"""
Quantitative Trading System
Integrates all quantitative components for advanced trading analysis.
"""

import logging
from typing import Dict, Optional, List, Any
import numpy as np
import pandas as pd

from .portfolio_optimizer import WorldQuantPortfolioOptimizer
from .risk_manager import RiskManager
from .statistical_validator import StatisticalValidator
from .market_microstructure import MarketMicrostructureAnalyzer
from .backtesting_engine import AdvancedBacktestingEngine
from .factor_model import WorldQuantFactorModel
from .ml_ensemble import WorldQuantMLEnsemble

logger = logging.getLogger(__name__)

class QuantitativeTradingSystem:
    """
    WorldQuant-level quantitative trading system that integrates all components.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Quantitative Trading System.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.portfolio_optimizer = WorldQuantPortfolioOptimizer(config)
        self.risk_manager = RiskManager(config)
        self.statistical_validator = StatisticalValidator(config)
        self.market_microstructure = MarketMicrostructureAnalyzer(config)
        self.backtesting_engine = AdvancedBacktestingEngine(config)
        self.factor_model = WorldQuantFactorModel(config)
        self.ml_ensemble = WorldQuantMLEnsemble(config)
        
        # Performance tracking
        self.analysis_history = {}
        self.optimization_results = {}
        self.risk_metrics = {}
        
        logger.info("QuantitativeTradingSystem initialized")
    
    async def initialize(self) -> bool:
        """Initialize all quantitative components."""
        try:
            # Initialize all components
            components = [
                self.portfolio_optimizer,
                self.risk_manager,
                self.statistical_validator,
                self.market_microstructure,
                self.backtesting_engine,
                self.factor_model,
                self.ml_ensemble
            ]
            
            for component in components:
                if hasattr(component, 'initialize'):
                    await component.initialize()
            
            logger.info("QuantitativeTradingSystem initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing QuantitativeTradingSystem: {str(e)}")
            return False
    
    async def analyze_portfolio(self, returns: pd.DataFrame, 
                              factor_exposures: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive portfolio analysis.
        
        Args:
            returns: DataFrame of asset returns
            factor_exposures: Optional factor exposures data
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        try:
            logger.info("Performing comprehensive portfolio analysis...")
            
            results = {}
            
            # 1. Portfolio Optimization
            logger.info("Running portfolio optimization...")
            optimization_results = {}
            
            # Mean-variance optimization
            mv_result = await self.portfolio_optimizer.optimize_mean_variance(returns)
            if mv_result.get('optimization_status') == 'success':
                optimization_results['mean_variance'] = mv_result
            
            # Risk parity optimization
            rp_result = await self.portfolio_optimizer.optimize_risk_parity(returns)
            if rp_result.get('optimization_status') == 'success':
                optimization_results['risk_parity'] = rp_result
            
            # Factor neutral optimization
            if factor_exposures:
                fn_result = await self.portfolio_optimizer.optimize_factor_neutral(returns, factor_exposures)
                if fn_result.get('optimization_status') == 'success':
                    optimization_results['factor_neutral'] = fn_result
            
            results['optimization'] = optimization_results
            
            # 2. Risk Analysis
            logger.info("Running risk analysis...")
            risk_analysis = await self.risk_manager.analyze_portfolio_risk(returns)
            results['risk_analysis'] = risk_analysis
            
            # 3. Statistical Validation
            logger.info("Running statistical validation...")
            validation_results = await self.statistical_validator.validate_portfolio(returns)
            results['statistical_validation'] = validation_results
            
            # 4. Market Microstructure Analysis
            logger.info("Running market microstructure analysis...")
            microstructure_results = await self.market_microstructure.analyze_market_impact(returns)
            results['market_microstructure'] = microstructure_results
            
            # 5. Factor Analysis
            logger.info("Running factor analysis...")
            factor_results = await self.factor_model.analyze_portfolio_factors(returns)
            results['factor_analysis'] = factor_results
            
            # 6. ML Analysis
            logger.info("Running ML analysis...")
            ml_results = await self.ml_ensemble.analyze_portfolio_ml(returns)
            results['ml_analysis'] = ml_results
            
            # Store results
            self.analysis_history[len(self.analysis_history)] = results
            
            logger.info("Comprehensive portfolio analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive portfolio analysis: {str(e)}")
            return {'error': str(e)}
    
    async def optimize_portfolio(self, returns: pd.DataFrame, 
                               method: str = 'mean_variance',
                               **kwargs) -> Dict[str, Any]:
        """
        Portfolio optimization with specified method.
        
        Args:
            returns: DataFrame of asset returns
            method: Optimization method ('mean_variance', 'risk_parity', 'factor_neutral', 'cross_asset_hedging')
            **kwargs: Additional parameters for optimization
            
        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info(f"Running {method} portfolio optimization...")
            
            if method == 'mean_variance':
                result = await self.portfolio_optimizer.optimize_mean_variance(returns, **kwargs)
            elif method == 'risk_parity':
                result = await self.portfolio_optimizer.optimize_risk_parity(returns, **kwargs)
            elif method == 'factor_neutral':
                factor_exposures = kwargs.get('factor_exposures', {})
                result = await self.portfolio_optimizer.optimize_factor_neutral(returns, factor_exposures, **kwargs)
            elif method == 'cross_asset_hedging':
                hedge_assets = kwargs.get('hedge_assets', [])
                result = await self.portfolio_optimizer.optimize_cross_asset_hedging(returns, hedge_assets, **kwargs)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            # Store optimization result
            self.optimization_results[method] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {str(e)}")
            return {'optimization_status': 'error', 'message': str(e)}
    
    async def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary."""
        try:
            summary = {
                'components': {
                    'portfolio_optimizer': 'WorldQuantPortfolioOptimizer',
                    'risk_manager': 'RiskManager',
                    'statistical_validator': 'StatisticalValidator',
                    'market_microstructure': 'MarketMicrostructureAnalyzer',
                    'backtesting_engine': 'AdvancedBacktestingEngine',
                    'factor_model': 'WorldQuantFactorModel',
                    'ml_ensemble': 'WorldQuantMLEnsemble'
                },
                'analysis_history': len(self.analysis_history),
                'optimization_results': len(self.optimization_results),
                'risk_metrics': len(self.risk_metrics)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting system summary: {str(e)}") 