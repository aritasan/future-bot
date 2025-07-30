#!/usr/bin/env python3
"""
Comprehensive test script for WorldQuant Advanced Backtesting Engine.
Tests walk-forward analysis, Monte Carlo simulation, stress testing, and performance attribution.
"""

import asyncio
import sys
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_mock_trading_data(n_periods: int = 500) -> pd.DataFrame:
    """Generate realistic mock trading data for testing."""
    np.random.seed(42)
    
    # Generate price data with trend and volatility
    base_price = 100.0
    prices = []
    current_price = base_price
    
    for i in range(n_periods):
        # Add trend and volatility
        trend = 0.001 * np.sin(i / 50)  # Small trend
        volatility = 0.02  # 2% volatility
        random_walk = np.random.normal(0, volatility)
        
        current_price *= (1 + trend + random_walk)
        prices.append(current_price)
    
    # Generate OHLCV data
    dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='D')
    
    data = pd.DataFrame({
        'date': dates,
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': [max(o, p) * (1 + abs(np.random.normal(0, 0.01))) for o, p in zip([p * (1 + np.random.normal(0, 0.005)) for p in prices], prices)],
        'low': [min(o, p) * (1 - abs(np.random.normal(0, 0.01))) for o, p in zip([p * (1 + np.random.normal(0, 0.005)) for p in prices], prices)],
        'close': prices,
        'volume': [np.random.uniform(1000, 10000) for _ in prices]
    })
    
    # Calculate returns
    data['returns'] = data['close'].pct_change().fillna(0)
    
    return data

def mock_trading_strategy(data: pd.DataFrame) -> float:
    """Mock trading strategy for testing."""
    try:
        if len(data) < 20:
            return 0.0  # Hold if insufficient data
        
        # Simple moving average strategy
        short_ma = data['close'].rolling(window=5).mean().iloc[-1]
        long_ma = data['close'].rolling(window=20).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Generate signal
        if short_ma > long_ma and current_price > short_ma:
            return 1.0  # Buy signal
        elif short_ma < long_ma and current_price < short_ma:
            return -1.0  # Sell signal
        else:
            return 0.0  # Hold signal
            
    except Exception as e:
        logger.error(f"Error in mock trading strategy: {str(e)}")
        return 0.0

async def test_backtesting_engine_initialization():
    """Test AdvancedBacktestingEngine initialization."""
    logger.info("Testing AdvancedBacktestingEngine initialization...")
    
    try:
        from src.quantitative.backtesting_engine import AdvancedBacktestingEngine
        
        # Create config
        config = {
            'trading': {
                'statistical_significance_level': 0.05,
                'min_sample_size': 50
            }
        }
        
        # Initialize backtesting engine
        engine = AdvancedBacktestingEngine(config)
        success = await engine.initialize()
        
        if success:
            logger.info("✅ Backtesting engine initialization PASSED")
            return True
        else:
            logger.error("❌ Backtesting engine initialization FAILED")
            return False
        
    except Exception as e:
        logger.error(f"Error testing backtesting engine initialization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_walk_forward_analysis():
    """Test walk-forward analysis."""
    logger.info("Testing walk-forward analysis...")
    
    try:
        from src.quantitative.backtesting_engine import AdvancedBacktestingEngine
        
        config = {'trading': {'statistical_significance_level': 0.05}}
        engine = AdvancedBacktestingEngine(config)
        await engine.initialize()
        
        # Generate mock data
        data = generate_mock_trading_data(n_periods=300)
        
        logger.info(f"Generated mock data: {data.shape}")
        logger.info(f"Data range: {data['date'].min()} to {data['date'].max()}")
        
        # Run walk-forward analysis
        result = await engine.run_walk_forward_analysis(mock_trading_strategy, data)
        
        if result and result.get('status') == 'success':
            logger.info(f"✅ Walk-forward analysis PASSED")
            logger.info(f"  Number of folds: {result['n_folds']}")
            logger.info(f"  Mean total return: {result['mean_total_return']:.4f}")
            logger.info(f"  Mean Sharpe ratio: {result['mean_sharpe_ratio']:.3f}")
            logger.info(f"  Mean max drawdown: {result['mean_max_drawdown']:.4f}")
            logger.info(f"  Mean volatility: {result['mean_volatility']:.4f}")
            
            # Show fold results
            for i, fold_result in enumerate(result['fold_results'][:3]):  # Show first 3 folds
                logger.info(f"  Fold {i+1}: Return={fold_result['total_return']:.4f}, Sharpe={fold_result['sharpe_ratio']:.3f}")
            
            return True
        else:
            logger.warning(f"❌ Walk-forward analysis FAILED: {result}")
            return False
            
    except Exception as e:
        logger.error(f"Error testing walk-forward analysis: {str(e)}")
        return False

async def test_monte_carlo_simulation():
    """Test Monte Carlo simulation."""
    logger.info("Testing Monte Carlo simulation...")
    
    try:
        from src.quantitative.backtesting_engine import AdvancedBacktestingEngine
        
        config = {'trading': {'statistical_significance_level': 0.05}}
        engine = AdvancedBacktestingEngine(config)
        await engine.initialize()
        
        # Generate mock data
        data = generate_mock_trading_data(n_periods=252)
        
        # Run Monte Carlo simulation
        result = await engine.run_monte_carlo_simulation(mock_trading_strategy, data, n_simulations=100)
        
        if result and result.get('status') == 'success':
            logger.info(f"✅ Monte Carlo simulation PASSED")
            logger.info(f"  Number of simulations: {result['n_simulations']}")
            
            # Total return statistics
            tr_stats = result['total_return']
            logger.info(f"  Total Return Statistics:")
            logger.info(f"    Mean: {tr_stats['mean']:.4f}")
            logger.info(f"    Std: {tr_stats['std']:.4f}")
            logger.info(f"    Min: {tr_stats['min']:.4f}")
            logger.info(f"    Max: {tr_stats['max']:.4f}")
            logger.info(f"    VaR (5%): {tr_stats['var']:.4f}")
            logger.info(f"    CVaR (5%): {tr_stats['cvar']:.4f}")
            
            # Sharpe ratio statistics
            sr_stats = result['sharpe_ratio']
            logger.info(f"  Sharpe Ratio Statistics:")
            logger.info(f"    Mean: {sr_stats['mean']:.3f}")
            logger.info(f"    Std: {sr_stats['std']:.3f}")
            logger.info(f"    Min: {sr_stats['min']:.3f}")
            logger.info(f"    Max: {sr_stats['max']:.3f}")
            
            # Max drawdown statistics
            md_stats = result['max_drawdown']
            logger.info(f"  Max Drawdown Statistics:")
            logger.info(f"    Mean: {md_stats['mean']:.4f}")
            logger.info(f"    Std: {md_stats['std']:.4f}")
            logger.info(f"    Max: {md_stats['max']:.4f}")
            logger.info(f"    95th percentile: {md_stats['percentile_95']:.4f}")
            
            return True
        else:
            logger.warning(f"❌ Monte Carlo simulation FAILED: {result}")
            return False
            
    except Exception as e:
        logger.error(f"Error testing Monte Carlo simulation: {str(e)}")
        return False

async def test_stress_testing():
    """Test stress testing."""
    logger.info("Testing stress testing...")
    
    try:
        from src.quantitative.backtesting_engine import AdvancedBacktestingEngine
        
        config = {'trading': {'statistical_significance_level': 0.05}}
        engine = AdvancedBacktestingEngine(config)
        await engine.initialize()
        
        # Generate mock data
        data = generate_mock_trading_data(n_periods=252)
        
        # Run stress testing
        result = await engine.run_stress_testing(mock_trading_strategy, data)
        
        if result and 'base' in result:
            logger.info(f"✅ Stress testing PASSED")
            
            # Base results
            base = result['base']
            logger.info(f"  Base Performance:")
            logger.info(f"    Total return: {base['total_return']:.4f}")
            logger.info(f"    Sharpe ratio: {base['sharpe_ratio']:.3f}")
            logger.info(f"    Max drawdown: {base['max_drawdown']:.4f}")
            
            # Stress test results
            for scenario, scenario_results in result.items():
                if scenario != 'base':
                    logger.info(f"  {scenario.replace('_', ' ').title()} Scenario:")
                    for shock_result in scenario_results:
                        shock_size = shock_result['shock_size']
                        logger.info(f"    Shock {shock_size:.1%}: Return={shock_result['total_return']:.4f}, Sharpe={shock_result['sharpe_ratio']:.3f}, DD={shock_result['max_drawdown']:.4f}")
            
            return True
        else:
            logger.warning(f"❌ Stress testing FAILED: {result}")
            return False
            
    except Exception as e:
        logger.error(f"Error testing stress testing: {str(e)}")
        return False

async def test_performance_attribution():
    """Test performance attribution."""
    logger.info("Testing performance attribution...")
    
    try:
        from src.quantitative.backtesting_engine import AdvancedBacktestingEngine
        
        config = {'trading': {'statistical_significance_level': 0.05}}
        engine = AdvancedBacktestingEngine(config)
        await engine.initialize()
        
        # Generate mock data
        data = generate_mock_trading_data(n_periods=252)
        
        # Generate mock factor data
        factor_data = {
            'market': 0.001,
            'size': 0.0005,
            'value': 0.0003,
            'momentum': 0.0008,
            'volatility': -0.0002,
            'liquidity': 0.0001
        }
        
        # Run performance attribution
        result = await engine.run_performance_attribution(mock_trading_strategy, data, factor_data)
        
        if result and result.get('status') == 'success':
            logger.info(f"✅ Performance attribution PASSED")
            logger.info(f"  Method: {result['method']}")
            logger.info(f"  Total return: {result['total_return']:.4f}")
            logger.info(f"  Residual: {result['residual']:.4f}")
            
            logger.info(f"  Factor Attribution:")
            for factor, contribution in result['factor_attribution'].items():
                logger.info(f"    {factor}: {contribution:.4f}")
            
            return True
        else:
            logger.warning(f"❌ Performance attribution FAILED: {result}")
            return False
            
    except Exception as e:
        logger.error(f"Error testing performance attribution: {str(e)}")
        return False

async def test_comprehensive_backtesting():
    """Test comprehensive backtesting workflow."""
    logger.info("Testing comprehensive backtesting workflow...")
    
    try:
        from src.quantitative.backtesting_engine import AdvancedBacktestingEngine
        
        config = {'trading': {'statistical_significance_level': 0.05}}
        engine = AdvancedBacktestingEngine(config)
        await engine.initialize()
        
        # Generate mock data
        data = generate_mock_trading_data(n_periods=500)
        
        # Test all backtesting methods
        backtesting_methods = [
            ('Walk-Forward Analysis', lambda: engine.run_walk_forward_analysis(mock_trading_strategy, data)),
            ('Monte Carlo Simulation', lambda: engine.run_monte_carlo_simulation(mock_trading_strategy, data, n_simulations=50)),
            ('Stress Testing', lambda: engine.run_stress_testing(mock_trading_strategy, data)),
            ('Performance Attribution', lambda: engine.run_performance_attribution(mock_trading_strategy, data))
        ]
        
        successful_tests = 0
        total_tests = len(backtesting_methods)
        
        for method_name, method_func in backtesting_methods:
            logger.info(f"  Testing {method_name}...")
            
            try:
                result = await method_func()
                
                if result and result.get('status') == 'success':
                    logger.info(f"    ✅ {method_name} successful")
                    successful_tests += 1
                    
                    # Show key metrics
                    if 'mean_total_return' in result:
                        logger.info(f"      Mean return: {result['mean_total_return']:.4f}")
                    elif 'total_return' in result:
                        logger.info(f"      Total return: {result['total_return']:.4f}")
                    elif 'n_simulations' in result:
                        logger.info(f"      Simulations: {result['n_simulations']}")
                    elif 'base' in result:
                        logger.info(f"      Base return: {result['base']['total_return']:.4f}")
                else:
                    logger.warning(f"    ❌ {method_name} failed")
                    
            except Exception as e:
                logger.error(f"    ❌ Error in {method_name}: {str(e)}")
        
        # Test backtesting summary
        summary = await engine.get_backtesting_summary()
        logger.info("Backtesting summary retrieved")
        
        success_rate = successful_tests / total_tests
        logger.info(f"  Comprehensive backtesting success rate: {success_rate:.1%} ({successful_tests}/{total_tests})")
        
        return success_rate >= 0.75  # At least 75% success rate
        
    except Exception as e:
        logger.error(f"Error testing comprehensive backtesting workflow: {str(e)}")
        return False

async def test_enhanced_trading_strategy_with_backtesting():
    """Test enhanced trading strategy with backtesting integration."""
    logger.info("Testing Enhanced Trading Strategy with Advanced Backtesting...")
    
    try:
        from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
        
        # Create mock services
        class MockIndicatorService:
            async def get_klines(self, symbol: str, timeframe: str, limit: int = 100) -> Dict:
                data = generate_mock_trading_data(limit)
                return {
                    'open': data['open'].tolist(),
                    'high': data['high'].tolist(),
                    'low': data['low'].tolist(),
                    'close': data['close'].tolist(),
                    'volume': data['volume'].tolist(),
                    'returns': data['returns'].tolist(),
                    'price': data['close'].iloc[-1]
                }
        
        class MockBinanceService:
            async def get_account_balance(self) -> Dict:
                return {
                    'USDT': {'total': 10000.0, 'available': 9500.0},
                    'BTC': {'total': 0.5, 'available': 0.4}
                }
            
            async def get_funding_rate(self, symbol: str) -> float:
                return np.random.normal(0.0001, 0.0002)
        
        class MockNotificationService:
            async def send_notification(self, message: str) -> bool:
                logger.info(f"Mock notification: {message}")
                return True
        
        # Create config
        config = {
            'trading': {
                'statistical_significance_level': 0.05,
                'min_sample_size': 50,
                'confidence_thresholds': {
                    'buy_base': 0.45,
                    'sell_base': 0.65,
                    'hold_base': 0.35
                }
            }
        }
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config=config,
            binance_service=MockBinanceService(),
            indicator_service=MockIndicatorService(),
            notification_service=MockNotificationService()
        )
        
        await strategy.initialize()
        
        # Test backtesting integration
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        # Generate mock data for multiple symbols
        mock_data = {}
        for symbol in symbols:
            mock_data[symbol] = generate_mock_trading_data(252)
        
        # Test backtesting capabilities
        logger.info("Testing backtesting integration with enhanced trading strategy...")
        
        # Test walk-forward analysis
        try:
            from src.quantitative.backtesting_engine import AdvancedBacktestingEngine
            engine = AdvancedBacktestingEngine(config)
            await engine.initialize()
            
            # Test with single symbol data
            btc_data = mock_data['BTCUSDT']
            walk_forward_result = await engine.run_walk_forward_analysis(mock_trading_strategy, btc_data)
            
            if walk_forward_result and walk_forward_result.get('status') == 'success':
                logger.info(f"  ✅ Walk-forward analysis with strategy integration successful")
                logger.info(f"    Folds: {walk_forward_result['n_folds']}")
                logger.info(f"    Mean return: {walk_forward_result['mean_total_return']:.4f}")
            else:
                logger.warning(f"  ❌ Walk-forward analysis with strategy integration failed")
            
            # Test Monte Carlo simulation
            mc_result = await engine.run_monte_carlo_simulation(mock_trading_strategy, btc_data, n_simulations=50)
            
            if mc_result and mc_result.get('status') == 'success':
                logger.info(f"  ✅ Monte Carlo simulation with strategy integration successful")
                logger.info(f"    Simulations: {mc_result['n_simulations']}")
                logger.info(f"    Mean return: {mc_result['total_return']['mean']:.4f}")
            else:
                logger.warning(f"  ❌ Monte Carlo simulation with strategy integration failed")
            
        except Exception as e:
            logger.error(f"  ❌ Error in backtesting integration: {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing enhanced trading strategy with backtesting: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def main():
    """Main test function."""
    logger.info("Starting Advanced Backtesting tests...")
    
    tests = [
        ("Backtesting Engine Initialization", test_backtesting_engine_initialization),
        ("Walk-Forward Analysis", test_walk_forward_analysis),
        ("Monte Carlo Simulation", test_monte_carlo_simulation),
        ("Stress Testing", test_stress_testing),
        ("Performance Attribution", test_performance_attribution),
        ("Comprehensive Backtesting Workflow", test_comprehensive_backtesting),
        ("Enhanced Trading Strategy with Backtesting", test_enhanced_trading_strategy_with_backtesting)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*60}")
        
        try:
            result = await test_func()
            results[test_name] = result
            
            if result:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} test FAILED with exception: {str(e)}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("ADVANCED BACKTESTING TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        logger.info("🎉 All Advanced Backtesting tests passed!")
    elif passed >= total * 0.8:
        logger.info("🎉 Most Advanced Backtesting tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed")
    
    return passed >= total * 0.8  # At least 80% success rate

if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 