#!/usr/bin/env python3
"""
Test script for Quantitative Trading System
Tests all components of the quantitative trading system
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from quantitative import (
    QuantitativeTradingSystem,
    RiskManager,
    StatisticalSignalValidator,
    PortfolioOptimizer,
    MarketMicrostructureAnalyzer,
    AdvancedBacktestingEngine,
    FactorModel
)

def generate_test_data():
    """Generate test data for quantitative analysis."""
    
    # Generate historical price data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate multiple asset returns
    n_assets = 5
    returns_data = pd.DataFrame(
        np.random.normal(0.001, 0.02, (len(dates), n_assets)),
        index=dates,
        columns=[f'Asset_{i+1}' for i in range(n_assets)]
    )
    
    # Generate price data
    price_data = (1 + returns_data).cumprod() * 100
    
    # Generate orderbook data
    orderbook_data = {
        'bids': [
            {'price': 100.0, 'size': 1000},
            {'price': 99.9, 'size': 1500},
            {'price': 99.8, 'size': 2000},
            {'price': 99.7, 'size': 2500},
            {'price': 99.6, 'size': 3000}
        ],
        'asks': [
            {'price': 100.1, 'size': 1000},
            {'price': 100.2, 'size': 1500},
            {'price': 100.3, 'size': 2000},
            {'price': 100.4, 'size': 2500},
            {'price': 100.5, 'size': 3000}
        ]
    }
    
    # Generate trade data
    trade_data = pd.DataFrame({
        'timestamp': dates[:100],
        'price': np.random.normal(100, 1, 100),
        'size': np.random.randint(100, 1000, 100),
        'side': np.random.choice(['buy', 'sell'], 100)
    })
    
    return {
        'returns_data': returns_data,
        'price_data': price_data,
        'orderbook_data': orderbook_data,
        'trade_data': trade_data
    }

def test_risk_manager():
    """Test Risk Manager component."""
    print("Testing Risk Manager...")
    
    risk_manager = RiskManager(confidence_level=0.95, max_position_size=0.02)
    
    # Generate test returns
    returns = np.random.normal(0.001, 0.02, 252)
    
    # Test VaR calculation
    var_results = risk_manager.var_calculator.calculate_var(
        returns=returns,
        position_size=10000,
        method='all'
    )
    
    print(f"VaR Results: {var_results}")
    
    # Test position sizing
    position_results = risk_manager.position_sizer.calculate_position_size(
        signal_strength=0.8,
        volatility=0.02,
        correlation=0.3,
        var_limit=0.02,
        win_rate=0.6,
        avg_win=0.02,
        avg_loss=0.01
    )
    
    print(f"Position Sizing Results: {position_results}")
    
    # Test risk metrics
    risk_metrics = risk_manager.calculate_risk_metrics(
        returns=returns,
        signal_data={'signal_strength': 0.8},
        position_size=0.01
    )
    
    print(f"Risk Metrics: {risk_metrics}")
    print("Risk Manager test completed successfully!\n")

def test_statistical_validator():
    """Test Statistical Validator component."""
    print("Testing Statistical Validator...")
    
    validator = StatisticalSignalValidator(min_p_value=0.05, min_t_stat=2.0)
    
    # Generate test signal data
    signal_data = {
        'signal_strength': 0.8,
        'signal_type': 'momentum',
        'confidence': 0.7
    }
    
    # Generate test returns
    returns = np.random.normal(0.001, 0.02, 252)
    
    # Test signal validation
    validation_results = validator.validate_signal(signal_data, returns)
    
    print(f"Validation Results: {validation_results}")
    
    # Test hypothesis testing
    hypothesis_results = validator.perform_hypothesis_test(returns, null_hypothesis=0.0)
    
    print(f"Hypothesis Test Results: {hypothesis_results}")
    print("Statistical Validator test completed successfully!\n")

def test_portfolio_optimizer():
    """Test Portfolio Optimizer component."""
    print("Testing Portfolio Optimizer...")
    
    optimizer = PortfolioOptimizer(risk_free_rate=0.02, target_return=0.10)
    
    # Generate test returns data
    np.random.seed(42)
    returns_data = pd.DataFrame(
        np.random.normal(0.001, 0.02, (252, 5)),
        columns=[f'Asset_{i+1}' for i in range(5)]
    )
    
    # Test different optimization methods
    methods = ['markowitz', 'risk_parity', 'max_sharpe']
    
    for method in methods:
        print(f"Testing {method} optimization...")
        results = optimizer.optimize_portfolio(returns_data, method=method)
        print(f"{method.capitalize()} Results: {results}")
    
    print("Portfolio Optimizer test completed successfully!\n")

def test_market_microstructure():
    """Test Market Microstructure Analyzer component."""
    print("Testing Market Microstructure Analyzer...")
    
    analyzer = MarketMicrostructureAnalyzer(min_tick_size=0.0001)
    
    # Generate test orderbook data
    orderbook_data = {
        'bids': [
            {'price': 100.0, 'size': 1000},
            {'price': 99.9, 'size': 1500},
            {'price': 99.8, 'size': 2000}
        ],
        'asks': [
            {'price': 100.1, 'size': 1000},
            {'price': 100.2, 'size': 1500},
            {'price': 100.3, 'size': 2000}
        ]
    }
    
    # Generate test trade data
    trade_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='D'),
        'price': np.random.normal(100, 1, 100),
        'size': np.random.randint(100, 1000, 100)
    })
    
    # Test market structure analysis
    analysis_results = analyzer.analyze_market_structure(orderbook_data, trade_data)
    
    print(f"Market Analysis Results: {analysis_results}")
    print("Market Microstructure Analyzer test completed successfully!\n")

def test_backtesting_engine():
    """Test Advanced Backtesting Engine component."""
    print("Testing Advanced Backtesting Engine...")
    
    engine = AdvancedBacktestingEngine(
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005
    )
    
    # Define a simple test strategy
    def test_strategy(row, historical_data, params):
        """Simple test strategy."""
        if len(historical_data) < 20:
            return {'action': 'hold', 'position_size': 0}
        
        # Simple moving average crossover
        ma_short = historical_data['close'].rolling(10).mean().iloc[-1]
        ma_long = historical_data['close'].rolling(20).mean().iloc[-1]
        
        if ma_short > ma_long:
            return {'action': 'buy', 'position_size': 0.1}
        elif ma_short < ma_long:
            return {'action': 'sell', 'position_size': 0.1}
        else:
            return {'action': 'hold', 'position_size': 0}
    
    # Generate test historical data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    np.random.seed(42)
    
    historical_data = pd.DataFrame({
        'open': np.random.normal(100, 1, 252),
        'high': np.random.normal(101, 1, 252),
        'low': np.random.normal(99, 1, 252),
        'close': np.random.normal(100, 1, 252),
        'volume': np.random.randint(1000, 10000, 252)
    }, index=dates)
    
    # Run backtest
    backtest_results = engine.run_backtest(
        strategy_function=test_strategy,
        historical_data=historical_data,
        strategy_params={'param1': 0.1}
    )
    
    print(f"Backtest Results: {backtest_results}")
    print("Advanced Backtesting Engine test completed successfully!\n")

def test_factor_model():
    """Test Factor Model component."""
    print("Testing Factor Model...")
    
    factor_model = FactorModel(n_factors=3, min_eigenvalue=1.0)
    
    # Generate test returns data
    np.random.seed(42)
    returns_data = pd.DataFrame(
        np.random.normal(0.001, 0.02, (252, 10)),
        columns=[f'Asset_{i+1}' for i in range(10)]
    )
    
    # Test factor model building
    factor_results = factor_model.build_factor_model(returns_data)
    
    print(f"Factor Model Results: {factor_results}")
    
    # Test factor timing analysis
    if 'pca_factors' in factor_results and 'factor_returns' in factor_results['pca_factors']:
        timing_results = factor_results['factor_timing']
        print(f"Factor Timing Results: {timing_results}")
    
    print("Factor Model test completed successfully!\n")

def test_quantitative_trading_system():
    """Test the integrated Quantitative Trading System."""
    print("Testing Quantitative Trading System...")
    
    # Initialize system
    config = {
        'confidence_level': 0.95,
        'max_position_size': 0.02,
        'risk_free_rate': 0.02,
        'optimization_method': 'markowitz',
        'n_factors': 3
    }
    
    system = QuantitativeTradingSystem(config)
    
    # Generate test data
    test_data = generate_test_data()
    
    # Test trading opportunity analysis
    market_data = {
        'orderbook': test_data['orderbook_data'],
        'trades': test_data['trade_data'],
        'returns': test_data['returns_data'].iloc[-1].values,
        'portfolio_data': test_data['returns_data'],
        'factor_data': test_data['returns_data']
    }
    
    signal_data = {
        'signal_strength': 0.8,
        'signal_type': 'momentum',
        'position_size': 0.01,
        'confidence': 0.7
    }
    
    # Run comprehensive analysis
    analysis_results = system.analyze_trading_opportunity(market_data, signal_data)
    
    print(f"Trading Analysis Results: {analysis_results}")
    
    # Test portfolio optimization
    portfolio_results = system.optimize_portfolio(test_data['returns_data'])
    print(f"Portfolio Optimization Results: {portfolio_results}")
    
    # Test backtest
    def test_strategy(row, historical_data, params):
        return {'action': 'buy', 'position_size': 0.1}
    
    backtest_results = system.run_backtest(
        test_strategy, 
        test_data['returns_data']
    )
    print(f"Backtest Results: {backtest_results}")
    
    # Get system summary
    summary = system.get_system_summary()
    print(f"System Summary: {summary}")
    
    print("Quantitative Trading System test completed successfully!\n")

def main():
    """Run all tests."""
    print("Starting Quantitative Trading System Tests...\n")
    
    try:
        test_risk_manager()
        test_statistical_validator()
        test_portfolio_optimizer()
        test_market_microstructure()
        test_backtesting_engine()
        test_factor_model()
        test_quantitative_trading_system()
        
        print("All tests completed successfully!")
        print("Quantitative Trading System is ready for use.")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 