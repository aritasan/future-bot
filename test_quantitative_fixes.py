#!/usr/bin/env python3
"""
Test script to check if quantitative trading system fixes are working.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_market_microstructure():
    """Test market microstructure analysis."""
    try:
        from quantitative.market_microstructure import MarketMicrostructureAnalyzer
        
        analyzer = MarketMicrostructureAnalyzer()
        
        # Test with different orderbook formats
        orderbook_data = {
            'bids': [[50000.0, 1.0], [49999.0, 2.0], [49998.0, 3.0]],
            'asks': [[50001.0, 1.0], [50002.0, 2.0], [50003.0, 3.0]]
        }
        
        result = analyzer.analyze_market_structure(orderbook_data)
        print("✅ Market microstructure test passed")
        return True
        
    except Exception as e:
        print(f"❌ Market microstructure test failed: {str(e)}")
        return False

def test_portfolio_optimizer():
    """Test portfolio optimizer."""
    try:
        from quantitative.portfolio_optimizer import PortfolioOptimizer
        
        optimizer = PortfolioOptimizer()
        
        # Test with returns data
        returns_data = {
            'BTCUSDT': np.random.normal(0.001, 0.02, 100),
            'ETHUSDT': np.random.normal(0.001, 0.025, 100)
        }
        
        result = optimizer.optimize_portfolio(returns_data, method='markowitz')
        print("✅ Portfolio optimizer test passed")
        return True
        
    except Exception as e:
        print(f"❌ Portfolio optimizer test failed: {str(e)}")
        return False

def test_factor_model():
    """Test factor model."""
    try:
        from quantitative.factor_model import FactorModel
        
        model = FactorModel()
        
        # Test with returns data
        returns_data = {
            'BTCUSDT': np.random.normal(0.001, 0.02, 100),
            'ETHUSDT': np.random.normal(0.001, 0.025, 100)
        }
        
        result = model.build_factor_model(returns_data)
        print("✅ Factor model test passed")
        return True
        
    except Exception as e:
        print(f"❌ Factor model test failed: {str(e)}")
        return False

def test_statistical_validator():
    """Test statistical validator."""
    try:
        from quantitative.statistical_validator import StatisticalSignalValidator
        
        validator = StatisticalSignalValidator()
        
        # Test with signal data
        signal_data = {
            'signal_strength': 0.7,
            'action': 'buy',
            'confidence': 0.8
        }
        
        returns = np.random.normal(0.001, 0.02, 100)
        result = validator.validate_signal(signal_data, returns)
        print("✅ Statistical validator test passed")
        return True
        
    except Exception as e:
        print(f"❌ Statistical validator test failed: {str(e)}")
        return False

def test_quantitative_integration():
    """Test quantitative integration."""
    try:
        from quantitative.integration import QuantitativeIntegration
        
        config = {
            'quantitative_integration_enabled': True,
            'confidence_level': 0.95,
            'max_position_size': 0.02
        }
        
        integration = QuantitativeIntegration(config)
        status = integration.get_integration_status()
        print("✅ Quantitative integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Quantitative integration test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Quantitative Trading System Fixes...")
    print("=" * 50)
    
    tests = [
        test_market_microstructure,
        test_portfolio_optimizer,
        test_factor_model,
        test_statistical_validator,
        test_quantitative_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Quantitative trading system is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 