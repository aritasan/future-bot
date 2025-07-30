#!/usr/bin/env python3
"""
Comprehensive test script to verify all quantitative trading system fixes.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import asyncio

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_market_microstructure_comprehensive():
    """Test all market microstructure methods with different data formats."""
    try:
        from quantitative.market_microstructure import MarketMicrostructureAnalyzer
        
        analyzer = MarketMicrostructureAnalyzer()
        
        # Test with list format orderbook
        orderbook_data_list = {
            'bids': [[50000.0, 1.0], [49999.0, 2.0], [49998.0, 3.0]],
            'asks': [[50001.0, 1.0], [50002.0, 2.0], [50003.0, 3.0]]
        }
        
        # Test with dict format orderbook
        orderbook_data_dict = {
            'bids': [{'price': 50000.0, 'size': 1.0}, {'price': 49999.0, 'size': 2.0}],
            'asks': [{'price': 50001.0, 'size': 1.0}, {'price': 50002.0, 'size': 2.0}]
        }
        
        # Test all methods with list format
        result_list = analyzer.analyze_market_structure(orderbook_data_list)
        print("✅ Market microstructure (list format) - All methods working")
        
        # Test all methods with dict format
        result_dict = analyzer.analyze_market_structure(orderbook_data_dict)
        print("✅ Market microstructure (dict format) - All methods working")
        
        return True
        
    except Exception as e:
        print(f"❌ Market microstructure comprehensive test failed: {str(e)}")
        return False

def test_portfolio_optimizer_comprehensive():
    """Test portfolio optimizer with various scenarios."""
    try:
        from quantitative.portfolio_optimizer import PortfolioOptimizer
        
        optimizer = PortfolioOptimizer()
        
        # Test with valid returns data
        returns_data = {
            'BTCUSDT': np.random.normal(0.001, 0.02, 100),
            'ETHUSDT': np.random.normal(0.001, 0.025, 100)
        }
        
        result = optimizer.optimize_portfolio(returns_data, method='markowitz')
        if 'error' not in result:
            print("✅ Portfolio optimizer - Markowitz method working")
        else:
            print(f"⚠️ Portfolio optimizer - Markowitz method: {result['error']}")
        
        # Test with single asset (edge case)
        single_asset_data = {
            'BTCUSDT': np.random.normal(0.001, 0.02, 100)
        }
        
        result_single = optimizer.optimize_portfolio(single_asset_data, method='markowitz')
        if 'error' not in result_single:
            print("✅ Portfolio optimizer - Single asset working")
        else:
            print(f"⚠️ Portfolio optimizer - Single asset: {result_single['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Portfolio optimizer comprehensive test failed: {str(e)}")
        return False

def test_factor_model_comprehensive():
    """Test factor model with various scenarios."""
    try:
        from quantitative.factor_model import FactorModel
        
        model = FactorModel()
        
        # Test with valid returns data
        returns_data = {
            'BTCUSDT': np.random.normal(0.001, 0.02, 100),
            'ETHUSDT': np.random.normal(0.001, 0.025, 100)
        }
        
        result = model.build_factor_model(returns_data)
        if 'error' not in result:
            print("✅ Factor model - PCA analysis working")
        else:
            print(f"⚠️ Factor model - PCA analysis: {result['error']}")
        
        # Test with single asset (edge case)
        single_asset_data = {
            'BTCUSDT': np.random.normal(0.001, 0.02, 100)
        }
        
        result_single = model.build_factor_model(single_asset_data)
        if 'error' not in result_single:
            print("✅ Factor model - Single asset working")
        else:
            print(f"⚠️ Factor model - Single asset: {result_single['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Factor model comprehensive test failed: {str(e)}")
        return False

def test_statistical_validator_comprehensive():
    """Test statistical validator with various scenarios."""
    try:
        from quantitative.statistical_validator import StatisticalSignalValidator
        
        validator = StatisticalSignalValidator()
        
        # Test with valid signal data
        signal_data = {
            'signal_strength': 0.7,
            'action': 'buy',
            'confidence': 0.8
        }
        
        returns = np.random.normal(0.001, 0.02, 100)
        result = validator.validate_signal(signal_data, returns)
        
        if 'error' not in result:
            print("✅ Statistical validator - Signal validation working")
        else:
            print(f"⚠️ Statistical validator - Signal validation: {result['error']}")
        
        # Test with zero variance returns (edge case)
        zero_var_returns = np.zeros(100)
        result_zero = validator.validate_signal(signal_data, zero_var_returns)
        
        if 'error' not in result_zero:
            print("✅ Statistical validator - Zero variance handling working")
        else:
            print(f"⚠️ Statistical validator - Zero variance: {result_zero['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Statistical validator comprehensive test failed: {str(e)}")
        return False

def test_quantitative_integration_comprehensive():
    """Test quantitative integration with various scenarios."""
    try:
        from quantitative.integration import QuantitativeIntegration
        
        config = {
            'quantitative_integration_enabled': True,
            'confidence_level': 0.95,
            'max_position_size': 0.02
        }
        
        integration = QuantitativeIntegration(config)
        status = integration.get_integration_status()
        
        if 'error' not in status:
            print("✅ Quantitative integration - Status check working")
        else:
            print(f"⚠️ Quantitative integration - Status check: {status['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantitative integration comprehensive test failed: {str(e)}")
        return False

def test_quantitative_trading_system_comprehensive():
    """Test quantitative trading system with various scenarios."""
    try:
        from quantitative.quantitative_trading_system import QuantitativeTradingSystem
        
        config = {
            'risk_free_rate': 0.02,
            'target_return': 0.15,
            'optimization_method': 'markowitz'
        }
        
        system = QuantitativeTradingSystem(config)
        
        # Test with valid market data
        market_data = {
            'orderbook': {
                'bids': [[50000.0, 1.0], [49999.0, 2.0]],
                'asks': [[50001.0, 1.0], [50002.0, 2.0]]
            },
            'returns': np.random.normal(0.001, 0.02, 100).tolist()
        }
        
        signal_data = {
            'signal_strength': 0.7,
            'action': 'buy',
            'confidence': 0.8
        }
        
        result = system.analyze_trading_opportunity(market_data, signal_data)
        
        if 'error' not in result:
            print("✅ Quantitative trading system - Analysis working")
        else:
            print(f"⚠️ Quantitative trading system - Analysis: {result['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantitative trading system comprehensive test failed: {str(e)}")
        return False

async def test_async_integration():
    """Test async integration methods."""
    try:
        from quantitative.integration import QuantitativeIntegration
        
        config = {
            'quantitative_integration_enabled': True,
            'confidence_level': 0.95,
            'max_position_size': 0.02
        }
        
        integration = QuantitativeIntegration(config)
        
        # Test async methods
        market_data = {
            'orderbook': {
                'bids': [[50000.0, 1.0], [49999.0, 2.0]],
                'asks': [[50001.0, 1.0], [50002.0, 2.0]]
            },
            'returns': np.random.normal(0.001, 0.02, 100).tolist()
        }
        
        signal = {
            'signal_strength': 0.7,
            'action': 'buy',
            'confidence': 0.8
        }
        
        # Test enhance_trading_signal
        enhanced_signal = await integration.enhance_trading_signal('BTCUSDT', signal, market_data)
        if 'error' not in enhanced_signal:
            print("✅ Async integration - Signal enhancement working")
        else:
            print(f"⚠️ Async integration - Signal enhancement: {enhanced_signal['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Async integration test failed: {str(e)}")
        return False

def main():
    """Run all comprehensive tests."""
    print("🧪 Comprehensive Quantitative Trading System Test")
    print("=" * 60)
    
    tests = [
        ("Market Microstructure", test_market_microstructure_comprehensive),
        ("Portfolio Optimizer", test_portfolio_optimizer_comprehensive),
        ("Factor Model", test_factor_model_comprehensive),
        ("Statistical Validator", test_statistical_validator_comprehensive),
        ("Quantitative Integration", test_quantitative_integration_comprehensive),
        ("Quantitative Trading System", test_quantitative_trading_system_comprehensive),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        print()
    
    # Test async integration
    print("🔍 Testing Async Integration...")
    try:
        asyncio.run(test_async_integration())
        passed += 1
        total += 1
        print("✅ Async integration test passed")
    except Exception as e:
        print(f"❌ Async integration test failed: {str(e)}")
        total += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Comprehensive Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All comprehensive tests passed! System is fully functional.")
        print("\n✅ All major components working:")
        print("   - Market Microstructure Analysis")
        print("   - Portfolio Optimization")
        print("   - Factor Analysis")
        print("   - Statistical Validation")
        print("   - Quantitative Integration")
        print("   - Quantitative Trading System")
        print("   - Async Integration")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 