#!/usr/bin/env python3
"""
Test script for Statistical Framework implementation.
Tests hypothesis testing, bootstrap confidence intervals, and statistical validation.
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
        
        return {
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
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
    
    async def get_klines(self, symbol: str, timeframe: str, limit: int = 100) -> List:
        """Mock klines data."""
        # Generate realistic mock data
        np.random.seed(42)
        
        base_price = 100.0
        if symbol == 'BTCUSDT':
            base_price = 50000.0
        elif symbol == 'ETHUSDT':
            base_price = 3000.0
        
        klines = []
        current_price = base_price
        
        for i in range(limit):
            # Add trend and volatility
            trend = 0.001 * np.sin(i / 10)
            volatility = 0.02
            random_walk = np.random.normal(0, volatility)
            
            current_price *= (1 + trend + random_walk)
            
            # Generate OHLCV
            open_price = current_price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, current_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, current_price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.uniform(1000, 10000)
            
            klines.append([
                1640995200000 + i * 3600000,  # timestamp
                open_price,  # open
                high_price,  # high
                low_price,   # low
                current_price,  # close
                volume       # volume
            ])
        
        return klines

class MockNotificationService:
    """Mock notification service for testing."""
    
    async def send_notification(self, message: str) -> bool:
        """Mock notification sending."""
        logger.info(f"Mock notification: {message}")
        return True

async def test_statistical_validator():
    """Test StatisticalValidator class."""
    logger.info("Testing StatisticalValidator...")
    
    try:
        from src.quantitative.statistical_validator import StatisticalValidator
        
        # Initialize validator
        validator = StatisticalValidator(significance_level=0.05, min_sample_size=50)
        
        # Generate mock signal history
        np.random.seed(42)
        signal_history = []
        
        for i in range(100):
            # Generate realistic signal returns
            signal_return = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% std
            signal_history.append({
                'timestamp': f'2024-01-{i+1:02d}',
                'return': signal_return,
                'action': 'buy' if signal_return > 0 else 'sell',
                'confidence': abs(signal_return) * 10
            })
        
        # Generate benchmark returns
        benchmark_returns = np.random.normal(0.0005, 0.015, 100)  # 0.05% mean, 1.5% std
        
        # Test signal significance
        significance_result = validator.test_signal_significance(signal_history, benchmark_returns)
        
        logger.info(f"Signal significance test result:")
        logger.info(f"  Significant: {significance_result['significant']}")
        logger.info(f"  P-value: {significance_result['p_value']:.4f}")
        logger.info(f"  T-statistic: {significance_result['t_statistic']:.4f}")
        logger.info(f"  Effect size: {significance_result['effect_size']:.4f}")
        logger.info(f"  Confidence interval: {significance_result['confidence_interval']}")
        
        # Test signal quality validation
        test_signal = {
            'action': 'buy',
            'strength': 0.6,
            'confidence': 0.8,
            'current_price': 100.0,
            'reasons': ['trend_following', 'volume_support', 'momentum_positive']
        }
        
        quality_result = validator.validate_signal_quality(test_signal)
        
        logger.info(f"Signal quality validation result:")
        logger.info(f"  Is valid: {quality_result['is_valid']}")
        logger.info(f"  Confidence score: {quality_result['confidence_score']:.3f}")
        logger.info(f"  Warnings: {quality_result['warnings']}")
        
        # Test market regime stability
        returns = np.random.normal(0.001, 0.02, 200)
        regime_result = validator.validate_market_regime_stability(returns)
        
        logger.info(f"Market regime stability result:")
        logger.info(f"  Is stable: {regime_result['is_stable']}")
        logger.info(f"  Stability score: {regime_result['stability_score']:.3f}")
        logger.info(f"  Regime changes: {regime_result['regime_changes']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing StatisticalValidator: {str(e)}")
        return False

async def test_enhanced_trading_strategy():
    """Test enhanced trading strategy with statistical validation."""
    logger.info("Testing Enhanced Trading Strategy with Statistical Validation...")
    
    try:
        from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
        
        # Create mock services
        indicator_service = MockIndicatorService()
        binance_service = MockBinanceService()
        notification_service = MockNotificationService()
        
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
            binance_service=binance_service,
            indicator_service=indicator_service,
            notification_service=notification_service
        )
        
        # Initialize strategy
        await strategy.initialize()
        
        # Test signal generation with statistical validation
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        for symbol in symbols:
            logger.info(f"Testing signal generation for {symbol}...")
            
            # Generate signal
            signal = await strategy.generate_signals(symbol, indicator_service)
            
            if signal:
                logger.info(f"Signal generated for {symbol}:")
                logger.info(f"  Action: {signal.get('action', 'unknown')}")
                logger.info(f"  Strength: {signal.get('strength', 0):.3f}")
                logger.info(f"  Confidence: {signal.get('confidence', 0):.3f}")
                logger.info(f"  Current price: {signal.get('current_price', 0):.2f}")
                
                # Check statistical validation
                if 'statistical_validation' in signal:
                    stat_val = signal['statistical_validation']
                    logger.info(f"  Statistical validation:")
                    logger.info(f"    Quality valid: {stat_val.get('quality_validation', {}).get('is_valid', False)}")
                    logger.info(f"    Significant: {stat_val.get('significance_test', {}).get('significant', False)}")
                    logger.info(f"    Confidence score: {stat_val.get('quality_validation', {}).get('confidence_score', 0):.3f}")
                else:
                    logger.warning(f"  No statistical validation found")
            else:
                logger.warning(f"No signal generated for {symbol}")
        
        # Test walk-forward analysis
        logger.info("Testing walk-forward analysis...")
        walk_forward_result = await strategy.perform_walk_forward_analysis(symbols)
        
        if walk_forward_result['success']:
            logger.info("Walk-forward analysis completed successfully")
            aggregated = walk_forward_result.get('aggregated_results', {})
            logger.info(f"  Total return: {aggregated.get('total_return', 0):.4f}")
            logger.info(f"  Mean Sharpe ratio: {aggregated.get('mean_sharpe_ratio', 0):.3f}")
            logger.info(f"  Mean hit rate: {aggregated.get('mean_hit_rate', 0):.3f}")
        else:
            logger.warning(f"Walk-forward analysis failed: {walk_forward_result.get('error', 'Unknown error')}")
        
        # Test statistical validation summary
        summary = strategy.get_statistical_validation_summary()
        logger.info("Statistical validation summary:")
        logger.info(f"  Total validations: {summary.get('total_validations', 0)}")
        logger.info(f"  Successful validations: {summary.get('successful_validations', 0)}")
        logger.info(f"  Average confidence score: {summary.get('average_confidence_score', 0):.3f}")
        logger.info(f"  Statistical significance rate: {summary.get('statistical_significance_rate', 0):.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing enhanced trading strategy: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_bootstrap_confidence_intervals():
    """Test bootstrap confidence intervals."""
    logger.info("Testing bootstrap confidence intervals...")
    
    try:
        from src.quantitative.statistical_validator import StatisticalValidator
        
        validator = StatisticalValidator()
        
        # Generate test data
        np.random.seed(42)
        test_data = np.random.normal(0.001, 0.02, 1000)  # 1000 returns
        
        # Calculate bootstrap confidence interval
        ci_lower, ci_upper = validator._bootstrap_confidence_interval(test_data)
        
        logger.info(f"Bootstrap confidence interval (95%):")
        logger.info(f"  Lower bound: {ci_lower:.6f}")
        logger.info(f"  Upper bound: {ci_upper:.6f}")
        logger.info(f"  Mean: {np.mean(test_data):.6f}")
        logger.info(f"  Std: {np.std(test_data):.6f}")
        
        # Verify that mean is within confidence interval
        mean = np.mean(test_data)
        if ci_lower <= mean <= ci_upper:
            logger.info("✅ Mean is within confidence interval")
        else:
            logger.warning("❌ Mean is outside confidence interval")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing bootstrap confidence intervals: {str(e)}")
        return False

async def test_hypothesis_testing():
    """Test hypothesis testing functionality."""
    logger.info("Testing hypothesis testing...")
    
    try:
        from src.quantitative.statistical_validator import StatisticalValidator
        
        validator = StatisticalValidator()
        
        # Generate two samples with different means
        np.random.seed(42)
        sample1 = np.random.normal(0.002, 0.02, 100)  # Higher mean
        sample2 = np.random.normal(0.0005, 0.02, 100)  # Lower mean
        
        # Test significance using t-test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(sample1, sample2)
        
        # Calculate effect size
        effect_size = validator._calculate_effect_size(sample1, sample2)
        
        logger.info(f"Hypothesis testing result:")
        logger.info(f"  Sample 1 mean: {np.mean(sample1):.6f}")
        logger.info(f"  Sample 2 mean: {np.mean(sample2):.6f}")
        logger.info(f"  T-statistic: {t_stat:.4f}")
        logger.info(f"  P-value: {p_value:.4f}")
        logger.info(f"  Effect size: {effect_size:.4f}")
        
        # Test with same distribution
        sample3 = np.random.normal(0.001, 0.02, 100)
        sample4 = np.random.normal(0.001, 0.02, 100)
        
        effect_size_same = validator._calculate_effect_size(sample3, sample4)
        logger.info(f"  Effect size (same distribution): {effect_size_same:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing hypothesis testing: {str(e)}")
        return False

async def main():
    """Main test function."""
    logger.info("Starting Statistical Framework tests...")
    
    tests = [
        ("StatisticalValidator", test_statistical_validator),
        ("Enhanced Trading Strategy", test_enhanced_trading_strategy),
        ("Bootstrap Confidence Intervals", test_bootstrap_confidence_intervals),
        ("Hypothesis Testing", test_hypothesis_testing)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
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
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Statistical Framework tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed")
    
    return passed == total

if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 