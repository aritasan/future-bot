#!/usr/bin/env python3
"""
Comprehensive test script to verify all numpy array comparison fixes.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import asyncio

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_enhanced_trading_strategy_numpy_fixes():
    """Test numpy fixes in enhanced trading strategy."""
    try:
        from strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
        
        # Mock dependencies
        config = {'test': True}
        binance_service = None
        indicator_service = None
        notification_service = None
        cache_service = None
        
        strategy = EnhancedTradingStrategyWithQuantitative(
            config, binance_service, indicator_service, notification_service, cache_service
        )
        
        # Test _calculate_max_drawdown
        returns = np.random.normal(0.001, 0.02, 100)
        max_dd = strategy._calculate_max_drawdown(returns.tolist())
        print("✅ _calculate_max_drawdown - No numpy errors")
        
        # Test _analyze_mean_reversion
        mean_rev = strategy._analyze_mean_reversion(returns.tolist())
        print("✅ _analyze_mean_reversion - No numpy errors")
        
        # Test _optimize_position_size_advanced
        market_data = {'returns': returns.tolist()}
        signal = {'portfolio_correlation': 0.1}
        pos_size = strategy._optimize_position_size_advanced('BTCUSDT', 0.01, market_data, signal)
        print("✅ _optimize_position_size_advanced - No numpy errors")
        
        # Test _apply_volatility_regime_analysis
        vol_analysis = strategy._apply_volatility_regime_analysis('BTCUSDT', signal, market_data)
        print("✅ _apply_volatility_regime_analysis - No numpy errors")
        
        # Test _calculate_risk_metrics
        market_df = pd.DataFrame({
            'close': np.cumprod(1 + returns) * 100
        })
        risk_metrics = strategy._calculate_risk_metrics(market_df)
        print("✅ _calculate_risk_metrics - No numpy errors")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced trading strategy numpy test failed: {str(e)}")
        return False

def test_statistical_validator_numpy_fixes():
    """Test numpy fixes in statistical validator."""
    try:
        from quantitative.statistical_validator import StatisticalSignalValidator
        
        validator = StatisticalSignalValidator()
        
        # Test _calculate_sharpe_ratio
        returns = np.random.normal(0.001, 0.02, 100)
        sharpe = validator._calculate_sharpe_ratio(returns)
        print("✅ _calculate_sharpe_ratio - No numpy errors")
        
        # Test _calculate_information_ratio
        benchmark = np.random.normal(0.0005, 0.015, 100)
        info_ratio = validator._calculate_information_ratio(returns, benchmark)
        print("✅ _calculate_information_ratio - No numpy errors")
        
        # Test _calculate_sortino_ratio
        sortino = validator._calculate_sortino_ratio(returns)
        print("✅ _calculate_sortino_ratio - No numpy errors")
        
        # Test _calculate_calmar_ratio
        calmar = validator._calculate_calmar_ratio(returns)
        print("✅ _calculate_calmar_ratio - No numpy errors")
        
        # Test _calculate_max_drawdown
        max_dd = validator._calculate_max_drawdown(returns)
        print("✅ _calculate_max_drawdown - No numpy errors")
        
        return True
        
    except Exception as e:
        print(f"❌ Statistical validator numpy test failed: {str(e)}")
        return False

def test_risk_manager_numpy_fixes():
    """Test numpy fixes in risk manager."""
    try:
        from quantitative.risk_manager import VaRCalculator
        
        var_calc = VaRCalculator()
        
        # Test _calculate_parametric_var
        returns = np.random.normal(0.001, 0.02, 100)
        parametric_var = var_calc._calculate_parametric_var(returns, 0.01)
        print("✅ _calculate_parametric_var - No numpy errors")
        
        # Test _calculate_monte_carlo_var
        monte_carlo_var = var_calc._calculate_monte_carlo_var(returns, 0.01)
        print("✅ _calculate_monte_carlo_var - No numpy errors")
        
        # Test _calculate_expected_shortfall
        expected_shortfall = var_calc._calculate_expected_shortfall(returns, -0.02)
        print("✅ _calculate_expected_shortfall - No numpy errors")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk manager numpy test failed: {str(e)}")
        return False

def test_edge_cases():
    """Test edge cases that could cause numpy errors."""
    try:
        from quantitative.statistical_validator import StatisticalSignalValidator
        from quantitative.risk_manager import VaRCalculator
        
        validator = StatisticalSignalValidator()
        var_calc = VaRCalculator()
        
        # Test with NaN values
        returns_with_nan = np.array([0.01, 0.02, np.nan, 0.03, 0.04])
        sharpe = validator._calculate_sharpe_ratio(returns_with_nan)
        print("✅ NaN handling - No numpy errors")
        
        # Test with all NaN values
        all_nan = np.array([np.nan, np.nan, np.nan])
        sharpe_all_nan = validator._calculate_sharpe_ratio(all_nan)
        print("✅ All NaN handling - No numpy errors")
        
        # Test with empty array
        empty_returns = np.array([])
        sharpe_empty = validator._calculate_sharpe_ratio(empty_returns)
        print("✅ Empty array handling - No numpy errors")
        
        # Test with single value
        single_return = np.array([0.01])
        sharpe_single = validator._calculate_sharpe_ratio(single_return)
        print("✅ Single value handling - No numpy errors")
        
        # Test with zero standard deviation
        zero_std_returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        sharpe_zero_std = validator._calculate_sharpe_ratio(zero_std_returns)
        print("✅ Zero std dev handling - No numpy errors")
        
        return True
        
    except Exception as e:
        print(f"❌ Edge cases test failed: {str(e)}")
        return False

def test_array_comparisons():
    """Test array comparison operations."""
    try:
        # Test various array comparison patterns
        returns = np.random.normal(0.001, 0.02, 100)
        
        # Test boolean operations
        positive_returns = returns > 0
        negative_returns = returns < 0
        
        # Test with np.any and np.all
        has_positive = np.any(positive_returns)
        all_positive = np.all(positive_returns)
        
        # Test with explicit boolean conversion
        has_positive_bool = bool(np.any(positive_returns))
        all_positive_bool = bool(np.all(positive_returns))
        
        # Test with float conversion
        mean_positive = float(np.mean(returns[positive_returns])) if np.any(positive_returns) else 0.0
        std_returns = float(np.std(returns))
        
        print("✅ Array comparison operations - No numpy errors")
        return True
        
    except Exception as e:
        print(f"❌ Array comparison test failed: {str(e)}")
        return False

def main():
    """Run all numpy fix tests."""
    print("🔧 Testing Numpy Array Comparison Fixes")
    print("=" * 50)
    
    tests = [
        ("Enhanced Trading Strategy", test_enhanced_trading_strategy_numpy_fixes),
        ("Statistical Validator", test_statistical_validator_numpy_fixes),
        ("Risk Manager", test_risk_manager_numpy_fixes),
        ("Edge Cases", test_edge_cases),
        ("Array Comparisons", test_array_comparisons)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All numpy array comparison fixes verified successfully!")
        print("✅ No more 'ambiguous truth value' errors should occur")
    else:
        print("⚠️ Some tests failed. Please review the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 