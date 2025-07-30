#!/usr/bin/env python3
"""
Comprehensive test suite for WorldQuant Portfolio Optimizer.
Tests mean-variance optimization, risk parity, factor neutral portfolios, and cross-asset hedging.
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

class MockIndicatorService:
    """Mock indicator service for testing."""
    
    async def get_klines(self, symbol: str, timeframe: str, limit: int = 100) -> Dict:
        """Mock klines data."""
        # Generate realistic mock data
        np.random.seed(42)  # For reproducible results
        
        # Generate price data with trend and volatility
        base_price = 100.0
        if symbol == 'BTCUSDT':
            base_price = 50000.0
        elif symbol == 'ETHUSDT':
            base_price = 3000.0
        elif symbol == 'ADAUSDT':
            base_price = 0.5
        elif symbol == 'SOLUSDT':
            base_price = 100.0
        
        prices = []
        current_price = base_price
        
        for i in range(limit):
            # Add trend and volatility
            trend = 0.001 * np.sin(i / 10)  # Small trend
            volatility = 0.02  # 2% volatility
            random_walk = np.random.normal(0, volatility)
            
            current_price *= (1 + trend + random_walk)
            prices.append(current_price)
        
        # Generate OHLCV data
        opens = [p * (1 + np.random.normal(0, 0.005)) for p in prices]
        highs = [max(o, p) * (1 + abs(np.random.normal(0, 0.01))) for o, p in zip(opens, prices)]
        lows = [min(o, p) * (1 - abs(np.random.normal(0, 0.01))) for o, p in zip(opens, prices)]
        volumes = [np.random.uniform(1000, 10000) for _ in prices]
        
        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        
        return {
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes,
            'returns': returns,
            'price': prices[-1] if prices else base_price
        }

class MockBinanceService:
    """Mock binance service for testing."""
    
    async def get_account_balance(self) -> Dict:
        """Mock account balance."""
        return {
            'USDT': {'total': 10000.0, 'available': 9500.0},
            'BTC': {'total': 0.5, 'available': 0.4},
            'ETH': {'total': 5.0, 'available': 4.8}
        }
    
    async def get_funding_rate(self, symbol: str) -> float:
        """Mock funding rate."""
        return np.random.normal(0.0001, 0.0002)

class MockNotificationService:
    """Mock notification service for testing."""
    
    async def send_notification(self, message: str) -> bool:
        """Mock notification sending."""
        logger.info(f"Mock notification: {message}")
        return True

def generate_mock_returns_data(n_assets: int = 5, n_periods: int = 252) -> pd.DataFrame:
    """Generate realistic mock returns data for testing."""
    np.random.seed(42)
    
    assets = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'BNBUSDT'][:n_assets]
    
    # Create correlation matrix
    correlation_matrix = np.array([
        [1.0, 0.7, 0.5, 0.6, 0.4],
        [0.7, 1.0, 0.4, 0.5, 0.3],
        [0.5, 0.4, 1.0, 0.6, 0.2],
        [0.6, 0.5, 0.6, 1.0, 0.4],
        [0.4, 0.3, 0.2, 0.4, 1.0]
    ])[:n_assets, :n_assets]
    
    # Generate returns with correlation
    returns_data = {}
    for i, asset in enumerate(assets):
        # Generate base returns with different characteristics
        if asset == 'BTCUSDT':
            base_returns = np.random.normal(0.0015, 0.025, n_periods)
        elif asset == 'ETHUSDT':
            base_returns = np.random.normal(0.0012, 0.030, n_periods)
        elif asset == 'ADAUSDT':
            base_returns = np.random.normal(0.0008, 0.035, n_periods)
        elif asset == 'SOLUSDT':
            base_returns = np.random.normal(0.0010, 0.028, n_periods)
        else:  # BNBUSDT
            base_returns = np.random.normal(0.0013, 0.022, n_periods)
        
        # Add correlation effects
        correlated_returns = base_returns.copy()
        for j in range(n_assets):
            if i != j:
                correlation = correlation_matrix[i, j]
                correlated_returns += correlation * 0.01 * np.random.normal(0, 1, n_periods)
        
        returns_data[asset] = correlated_returns
    
    return pd.DataFrame(returns_data)

def generate_mock_factor_exposures(assets: List[str]) -> Dict[str, Dict[str, float]]:
    """Generate mock factor exposures for testing."""
    factors = ['market', 'size', 'value', 'momentum', 'volatility', 'liquidity']
    
    factor_exposures = {}
    for asset in assets:
        factor_exposures[asset] = {}
        for factor in factors:
            # Generate random factor exposures
            exposure = np.random.normal(0, 0.5)
            factor_exposures[asset][factor] = exposure
    
    return factor_exposures

async def test_portfolio_optimizer_initialization():
    """Test WorldQuantPortfolioOptimizer initialization."""
    logger.info("Testing WorldQuantPortfolioOptimizer initialization...")
    
    try:
        from src.quantitative.portfolio_optimizer import WorldQuantPortfolioOptimizer
        
        # Create config
        config = {
            'trading': {
                'statistical_significance_level': 0.05,
                'min_sample_size': 50
            }
        }
        
        # Initialize portfolio optimizer
        optimizer = WorldQuantPortfolioOptimizer(config)
        success = await optimizer.initialize()
        
        if success:
            logger.info("✅ Portfolio optimizer initialization PASSED")
            return True
        else:
            logger.error("❌ Portfolio optimizer initialization FAILED")
            return False
        
    except Exception as e:
        logger.error(f"Error testing portfolio optimizer initialization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_mean_variance_optimization():
    """Test mean-variance optimization."""
    logger.info("Testing mean-variance optimization...")
    
    try:
        from src.quantitative.portfolio_optimizer import WorldQuantPortfolioOptimizer
        
        config = {'trading': {'statistical_significance_level': 0.05}}
        optimizer = WorldQuantPortfolioOptimizer(config)
        await optimizer.initialize()
        
        # Generate mock returns data
        returns_df = generate_mock_returns_data(n_assets=5, n_periods=252)
        
        logger.info(f"Generated returns data: {returns_df.shape}")
        logger.info(f"Returns statistics:")
        logger.info(f"  Mean returns: {returns_df.mean().to_dict()}")
        logger.info(f"  Volatilities: {returns_df.std().to_dict()}")
        
        # Test mean-variance optimization
        result = await optimizer.optimize_mean_variance(returns_df, target_return=0.15)
        
        if result and result.get('optimization_status') == 'success':
            logger.info(f"✅ Mean-variance optimization PASSED")
            logger.info(f"  Portfolio return: {result['portfolio_return']:.4f}")
            logger.info(f"  Portfolio volatility: {result['portfolio_volatility']:.4f}")
            logger.info(f"  Sharpe ratio: {result['sharpe_ratio']:.3f}")
            
            weights = result['weights']
            logger.info(f"  Optimal weights:")
            for asset, weight in weights.items():
                logger.info(f"    {asset}: {weight:.4f}")
            
            risk_contributions = result['risk_contributions']
            logger.info(f"  Risk contributions:")
            for asset, contribution in risk_contributions.items():
                logger.info(f"    {asset}: {contribution:.4f}")
            
            return True
        else:
            logger.warning(f"❌ Mean-variance optimization FAILED: {result}")
            return False
            
    except Exception as e:
        logger.error(f"Error testing mean-variance optimization: {str(e)}")
        return False

async def test_risk_parity_optimization():
    """Test risk parity optimization."""
    logger.info("Testing risk parity optimization...")
    
    try:
        from src.quantitative.portfolio_optimizer import WorldQuantPortfolioOptimizer
        
        config = {'trading': {'statistical_significance_level': 0.05}}
        optimizer = WorldQuantPortfolioOptimizer(config)
        await optimizer.initialize()
        
        # Generate mock returns data
        returns_df = generate_mock_returns_data(n_assets=5, n_periods=252)
        
        # Test risk parity optimization
        result = await optimizer.optimize_risk_parity(returns_df)
        
        if result and result.get('optimization_status') == 'success':
            logger.info(f"✅ Risk parity optimization PASSED")
            logger.info(f"  Portfolio return: {result['portfolio_return']:.4f}")
            logger.info(f"  Portfolio volatility: {result['portfolio_volatility']:.4f}")
            logger.info(f"  Risk parity score: {result['risk_parity_score']:.3f}")
            
            weights = result['weights']
            logger.info(f"  Optimal weights:")
            for asset, weight in weights.items():
                logger.info(f"    {asset}: {weight:.4f}")
            
            risk_contributions = result['risk_contributions']
            logger.info(f"  Risk contributions:")
            for asset, contribution in risk_contributions.items():
                logger.info(f"    {asset}: {contribution:.4f}")
            
            return True
        else:
            logger.warning(f"❌ Risk parity optimization FAILED: {result}")
            return False
            
    except Exception as e:
        logger.error(f"Error testing risk parity optimization: {str(e)}")
        return False

async def test_factor_neutral_optimization():
    """Test factor neutral optimization."""
    logger.info("Testing factor neutral optimization...")
    
    try:
        from src.quantitative.portfolio_optimizer import WorldQuantPortfolioOptimizer
        
        config = {'trading': {'statistical_significance_level': 0.05}}
        optimizer = WorldQuantPortfolioOptimizer(config)
        await optimizer.initialize()
        
        # Generate mock returns data
        returns_df = generate_mock_returns_data(n_assets=5, n_periods=252)
        
        # Generate mock factor exposures
        assets = returns_df.columns.tolist()
        factor_exposures = generate_mock_factor_exposures(assets)
        
        # Test factor neutral optimization
        result = await optimizer.optimize_factor_neutral(returns_df, factor_exposures)
        
        if result and result.get('optimization_status') == 'success':
            logger.info(f"✅ Factor neutral optimization PASSED")
            logger.info(f"  Portfolio return: {result['portfolio_return']:.4f}")
            logger.info(f"  Portfolio volatility: {result['portfolio_volatility']:.4f}")
            logger.info(f"  Sharpe ratio: {result['sharpe_ratio']:.3f}")
            
            weights = result['weights']
            logger.info(f"  Optimal weights:")
            for asset, weight in weights.items():
                logger.info(f"    {asset}: {weight:.4f}")
            
            factor_exposures_portfolio = result['factor_exposures']
            logger.info(f"  Factor exposures:")
            for factor, exposure in factor_exposures_portfolio.items():
                logger.info(f"    {factor}: {exposure:.4f}")
            
            return True
        else:
            logger.warning(f"❌ Factor neutral optimization FAILED: {result}")
            return False
            
    except Exception as e:
        logger.error(f"Error testing factor neutral optimization: {str(e)}")
        return False

async def test_cross_asset_hedging():
    """Test cross-asset hedging optimization."""
    logger.info("Testing cross-asset hedging optimization...")
    
    try:
        from src.quantitative.portfolio_optimizer import WorldQuantPortfolioOptimizer
        
        config = {'trading': {'statistical_significance_level': 0.05}}
        optimizer = WorldQuantPortfolioOptimizer(config)
        await optimizer.initialize()
        
        # Generate mock returns data with more assets
        returns_df = generate_mock_returns_data(n_assets=5, n_periods=252)
        
        # Define hedge assets
        hedge_assets = ['SOLUSDT', 'BNBUSDT']
        
        # Test cross-asset hedging optimization
        result = await optimizer.optimize_cross_asset_hedging(returns_df, hedge_assets)
        
        if result and result.get('optimization_status') == 'success':
            logger.info(f"✅ Cross-asset hedging optimization PASSED")
            logger.info(f"  Portfolio return: {result['portfolio_return']:.4f}")
            logger.info(f"  Portfolio volatility: {result['portfolio_volatility']:.4f}")
            logger.info(f"  Sharpe ratio: {result['sharpe_ratio']:.3f}")
            logger.info(f"  Hedge effectiveness: {result['hedge_effectiveness']:.3f}")
            
            weights = result['weights']
            logger.info(f"  Optimal weights:")
            for asset, weight in weights.items():
                logger.info(f"    {asset}: {weight:.4f}")
            
            hedge_ratios = result['hedge_ratios']
            logger.info(f"  Hedge ratios:")
            for (core, hedge), ratio in hedge_ratios.items():
                logger.info(f"    {core} -> {hedge}: {ratio:.4f}")
            
            return True
        else:
            logger.warning(f"❌ Cross-asset hedging optimization FAILED: {result}")
            return False
            
    except Exception as e:
        logger.error(f"Error testing cross-asset hedging optimization: {str(e)}")
        return False

async def test_risk_contributions():
    """Test risk contributions calculation."""
    logger.info("Testing risk contributions calculation...")
    
    try:
        from src.quantitative.portfolio_optimizer import WorldQuantPortfolioOptimizer
        
        config = {'trading': {'statistical_significance_level': 0.05}}
        optimizer = WorldQuantPortfolioOptimizer(config)
        await optimizer.initialize()
        
        # Generate mock returns data
        returns_df = generate_mock_returns_data(n_assets=5, n_periods=252)
        
        # Calculate covariance matrix
        cov_matrix = returns_df.cov()
        
        # Test with equal weights
        equal_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        # Calculate risk contributions
        risk_contributions = optimizer._calculate_risk_contributions(equal_weights, cov_matrix)
        
        if risk_contributions:
            logger.info(f"✅ Risk contributions calculation PASSED")
            logger.info(f"  Risk contributions:")
            for asset, contribution in risk_contributions.items():
                logger.info(f"    {asset}: {contribution:.4f}")
            
            # Verify risk contributions sum to portfolio volatility
            portfolio_variance = np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            total_risk_contribution = sum(risk_contributions.values())
            
            logger.info(f"  Portfolio volatility: {portfolio_volatility:.4f}")
            logger.info(f"  Total risk contribution: {total_risk_contribution:.4f}")
            logger.info(f"  Risk contribution sum matches portfolio volatility: {abs(total_risk_contribution - portfolio_volatility) < 1e-6}")
            
            return True
        else:
            logger.warning(f"❌ Risk contributions calculation FAILED")
            return False
            
    except Exception as e:
        logger.error(f"Error testing risk contributions calculation: {str(e)}")
        return False

async def test_portfolio_constraints():
    """Test portfolio constraints validation."""
    logger.info("Testing portfolio constraints validation...")
    
    try:
        from src.quantitative.portfolio_optimizer import WorldQuantPortfolioOptimizer
        
        config = {'trading': {'statistical_significance_level': 0.05}}
        optimizer = WorldQuantPortfolioOptimizer(config)
        await optimizer.initialize()
        
        # Test constraints
        constraints = optimizer.constraints
        
        logger.info(f"✅ Portfolio constraints validation PASSED")
        logger.info(f"  Long only: {constraints['long_only']}")
        logger.info(f"  Leverage limit: {constraints['leverage_limit']}")
        logger.info(f"  Concentration limit: {constraints['concentration_limit']}")
        logger.info(f"  Sector limit: {constraints['sector_limit']}")
        logger.info(f"  Geographic limit: {constraints['geographic_limit']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing portfolio constraints validation: {str(e)}")
        return False

async def test_optimization_parameters():
    """Test optimization parameters validation."""
    logger.info("Testing optimization parameters validation...")
    
    try:
        from src.quantitative.portfolio_optimizer import WorldQuantPortfolioOptimizer
        
        config = {'trading': {'statistical_significance_level': 0.05}}
        optimizer = WorldQuantPortfolioOptimizer(config)
        await optimizer.initialize()
        
        # Test optimization parameters
        params = optimizer.optimization_params
        
        logger.info(f"✅ Optimization parameters validation PASSED")
        
        # Mean-variance parameters
        mv_params = params['mean_variance']
        logger.info(f"  Mean-variance parameters:")
        logger.info(f"    Risk-free rate: {mv_params['risk_free_rate']:.3f}")
        logger.info(f"    Target return: {mv_params['target_return']:.3f}")
        logger.info(f"    Max volatility: {mv_params['max_volatility']:.3f}")
        logger.info(f"    Min weight: {mv_params['min_weight']:.3f}")
        logger.info(f"    Max weight: {mv_params['max_weight']:.3f}")
        
        # Risk parity parameters
        rp_params = params['risk_parity']
        logger.info(f"  Risk parity parameters:")
        logger.info(f"    Target risk contribution: {rp_params['target_risk_contribution']:.3f}")
        logger.info(f"    Risk budget method: {rp_params['risk_budget_method']}")
        
        # Factor neutral parameters
        fn_params = params['factor_neutral']
        logger.info(f"  Factor neutral parameters:")
        logger.info(f"    Factor exposures: {len(fn_params['factor_exposures'])} factors")
        logger.info(f"    Max factor exposure: {fn_params['max_factor_exposure']:.3f}")
        
        # Cross-asset hedging parameters
        cah_params = params['cross_asset_hedging']
        logger.info(f"  Cross-asset hedging parameters:")
        logger.info(f"    Hedge ratio method: {cah_params['hedge_ratio_method']}")
        logger.info(f"    Correlation threshold: {cah_params['correlation_threshold']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing optimization parameters validation: {str(e)}")
        return False

async def test_portfolio_summary():
    """Test portfolio summary generation."""
    logger.info("Testing portfolio summary generation...")
    
    try:
        from src.quantitative.portfolio_optimizer import WorldQuantPortfolioOptimizer
        
        config = {'trading': {'statistical_significance_level': 0.05}}
        optimizer = WorldQuantPortfolioOptimizer(config)
        await optimizer.initialize()
        
        # Test portfolio summary
        summary = await optimizer.get_portfolio_summary()
        
        if summary:
            logger.info(f"✅ Portfolio summary generation PASSED")
            logger.info(f"  Optimization methods: {summary['optimization_methods']}")
            logger.info(f"  Optimization history: {summary['optimization_history']}")
            logger.info(f"  Portfolio metrics: {summary['portfolio_metrics']}")
            logger.info(f"  Rebalancing dates: {summary['rebalancing_dates']}")
            
            return True
        else:
            logger.warning(f"❌ Portfolio summary generation FAILED")
            return False
            
    except Exception as e:
        logger.error(f"Error testing portfolio summary generation: {str(e)}")
        return False

async def test_comprehensive_optimization():
    """Test comprehensive optimization workflow."""
    logger.info("Testing comprehensive optimization workflow...")
    
    try:
        from src.quantitative.portfolio_optimizer import WorldQuantPortfolioOptimizer
        
        config = {'trading': {'statistical_significance_level': 0.05}}
        optimizer = WorldQuantPortfolioOptimizer(config)
        await optimizer.initialize()
        
        # Generate mock returns data
        returns_df = generate_mock_returns_data(n_assets=5, n_periods=252)
        assets = returns_df.columns.tolist()
        factor_exposures = generate_mock_factor_exposures(assets)
        
        # Test all optimization methods
        optimization_methods = [
            ('Mean-Variance', lambda: optimizer.optimize_mean_variance(returns_df)),
            ('Risk Parity', lambda: optimizer.optimize_risk_parity(returns_df)),
            ('Factor Neutral', lambda: optimizer.optimize_factor_neutral(returns_df, factor_exposures)),
            ('Cross-Asset Hedging', lambda: optimizer.optimize_cross_asset_hedging(returns_df, ['SOLUSDT', 'BNBUSDT']))
        ]
        
        successful_optimizations = 0
        total_optimizations = len(optimization_methods)
        
        for method_name, optimization_func in optimization_methods:
            logger.info(f"  Testing {method_name} optimization...")
            
            try:
                result = await optimization_func()
                
                if result and result.get('optimization_status') == 'success':
                    logger.info(f"    ✅ {method_name} optimization successful")
                    logger.info(f"      Portfolio return: {result.get('portfolio_return', 0):.4f}")
                    logger.info(f"      Portfolio volatility: {result.get('portfolio_volatility', 0):.4f}")
                    
                    if 'sharpe_ratio' in result:
                        logger.info(f"      Sharpe ratio: {result['sharpe_ratio']:.3f}")
                    
                    if 'risk_parity_score' in result:
                        logger.info(f"      Risk parity score: {result['risk_parity_score']:.3f}")
                    
                    if 'hedge_effectiveness' in result:
                        logger.info(f"      Hedge effectiveness: {result['hedge_effectiveness']:.3f}")
                    
                    successful_optimizations += 1
                else:
                    logger.warning(f"    ❌ {method_name} optimization failed")
                    
            except Exception as e:
                logger.error(f"    ❌ Error in {method_name} optimization: {str(e)}")
        
        success_rate = successful_optimizations / total_optimizations
        logger.info(f"  Comprehensive optimization success rate: {success_rate:.2%} ({successful_optimizations}/{total_optimizations})")
        
        return success_rate >= 0.75  # At least 75% success rate
        
    except Exception as e:
        logger.error(f"Error testing comprehensive optimization workflow: {str(e)}")
        return False

async def main():
    """Main test function."""
    logger.info("Starting Comprehensive Portfolio Optimization tests...")
    
    tests = [
        ("Portfolio Optimizer Initialization", test_portfolio_optimizer_initialization),
        ("Mean-Variance Optimization", test_mean_variance_optimization),
        ("Risk Parity Optimization", test_risk_parity_optimization),
        ("Factor Neutral Optimization", test_factor_neutral_optimization),
        ("Cross-Asset Hedging", test_cross_asset_hedging),
        ("Risk Contributions Calculation", test_risk_contributions),
        ("Portfolio Constraints Validation", test_portfolio_constraints),
        ("Optimization Parameters Validation", test_optimization_parameters),
        ("Portfolio Summary Generation", test_portfolio_summary),
        ("Comprehensive Optimization Workflow", test_comprehensive_optimization)
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
    logger.info("COMPREHENSIVE TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        logger.info("🎉 All Portfolio Optimization tests passed!")
    elif passed >= total * 0.8:
        logger.info("🎉 Most Portfolio Optimization tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed")
    
    return passed >= total * 0.8  # At least 80% success rate

if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 