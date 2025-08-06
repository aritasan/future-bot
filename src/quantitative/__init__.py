"""
Quantitative Trading Module

This module provides advanced quantitative trading tools including:
- Risk Management (VaR, Position Sizing)
- Statistical Validation
- Portfolio Optimization
- Market Microstructure Analysis
- Advanced Backtesting
- Factor Models
- Machine Learning Ensemble
- Advanced Risk Management (Phase 2)
- Statistical Arbitrage (Phase 2)
- Advanced ML Ensemble (Phase 2)
- Alternative Data Integration (Phase 2)
- Advanced Portfolio Optimization (Phase 2)
- Performance Attribution (Phase 2)
- Real-Time Risk Monitoring (Phase 2)
- High-Frequency Trading (Phase 3)
- Advanced Market Microstructure (Phase 3)
- Options-Based Strategies (Phase 3)
- On-Chain Analytics (Phase 3)
"""

from .risk_manager import RiskManager, VaRCalculator, DynamicPositionSizer
from .statistical_validator import StatisticalValidator
from .portfolio_optimizer import WorldQuantPortfolioOptimizer
from .market_microstructure import MarketMicrostructureAnalyzer
from .backtesting_engine import AdvancedBacktestingEngine
from .factor_model import WorldQuantFactorModel
from .ml_ensemble import WorldQuantMLEnsemble
from .quantitative_trading_system import QuantitativeTradingSystem

# Phase 2 Advanced Features
from .advanced_risk_management import DynamicRiskManager
from .statistical_arbitrage import StatisticalArbitrageEngine
from .advanced_ml_ensemble import AdvancedMLEnsemble
from .alternative_data_integration import AlternativeDataEngine
from .advanced_portfolio_optimization import AdvancedPortfolioOptimizer
from .performance_attribution import PerformanceAttribution
from .real_time_risk_monitoring import RealTimeRiskMonitor

# Phase 3 WorldQuant-Level Features
from .high_frequency_trading import HighFrequencyTradingEngine, TickData, OrderBookSnapshot
from .advanced_market_microstructure import AdvancedMarketMicrostructureAnalyzer
from .options_based_strategies import OptionsBasedStrategies, OptionContract, OptionsStrategy
from .on_chain_analytics import OnChainAnalytics, BlockchainTransaction, WalletProfile

# Additional Advanced Features
from .implied_volatility import ImpliedVolatilityEngine
from .worldquant_dca_trailing import WorldQuantDCA, WorldQuantTrailingStop
from .real_time_performance_monitor import WorldQuantRealTimePerformanceMonitor

__all__ = [
    # Core Quantitative System
    'RiskManager',
    'VaRCalculator', 
    'DynamicPositionSizer',
    'StatisticalValidator',
    'WorldQuantPortfolioOptimizer',
    'MarketMicrostructureAnalyzer',
    'AdvancedBacktestingEngine',
    'WorldQuantFactorModel',
    'WorldQuantMLEnsemble',
    'QuantitativeTradingSystem',
    
    # Phase 2 Advanced Features
    'DynamicRiskManager',
    'StatisticalArbitrageEngine',
    'AdvancedMLEnsemble',
    'AlternativeDataEngine',
    'AdvancedPortfolioOptimizer',
    'PerformanceAttribution',
    'RealTimeRiskMonitor',
    
    # Phase 3 WorldQuant-Level Features
    'HighFrequencyTradingEngine',
    'TickData',
    'OrderBookSnapshot',
    'AdvancedMarketMicrostructureAnalyzer',
    'OptionsBasedStrategies',
    'OptionContract',
    'OptionsStrategy',
    'OnChainAnalytics',
    'BlockchainTransaction',
    'WalletProfile',
    
    # Additional Advanced Features
    'ImpliedVolatilityEngine',
    'WorldQuantDCA',
    'WorldQuantTrailingStop',
    'WorldQuantRealTimePerformanceMonitor'
] 