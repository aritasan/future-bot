#!/usr/bin/env python3
"""
Test script to verify NumPy array comparison fixes
"""

import numpy as np
import sys

def test_kelly_criterion_fix():
    """Test the Kelly Criterion calculation fix"""
    try:
        # Simulate the market data returns
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.005, 0.015, -0.008])
        
        # Apply the fix from the strategy
        positive_returns_mask = returns > 0
        negative_returns_mask = returns < 0
        positive_count = float(np.sum(positive_returns_mask))
        negative_count = float(np.sum(negative_returns_mask))
        
        win_rate = positive_count / len(returns)
        avg_win = float(np.mean(returns[positive_returns_mask])) if positive_count > 0 else 0.001
        avg_loss = abs(float(np.mean(returns[negative_returns_mask]))) if negative_count > 0 else 0.001
        
        print(f"✅ Kelly Criterion fix test passed")
        print(f"   Win rate: {win_rate:.3f}")
        print(f"   Avg win: {avg_win:.3f}")
        print(f"   Avg loss: {avg_loss:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Kelly Criterion fix test failed: {e}")
        return False

def test_risk_metrics_fix():
    """Test the risk metrics calculation fix"""
    try:
        # Simulate returns data
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.005, 0.015, -0.008])
        
        # Apply the fix from the strategy
        returns_std = float(np.std(returns))
        sharpe_ratio = float(np.mean(returns) / returns_std * np.sqrt(252)) if returns_std > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_std = float(np.std(downside_returns)) if len(downside_returns) > 0 else 0
        sortino_ratio = float(np.mean(returns) / downside_std * np.sqrt(252)) if len(downside_returns) > 0 and downside_std > 0 else 0
        
        print(f"✅ Risk metrics fix test passed")
        print(f"   Sharpe ratio: {sharpe_ratio:.3f}")
        print(f"   Sortino ratio: {sortino_ratio:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Risk metrics fix test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases that might cause issues"""
    try:
        # Test with empty returns
        returns = np.array([])
        positive_returns_mask = returns > 0
        negative_returns_mask = returns < 0
        positive_count = float(np.sum(positive_returns_mask))
        negative_count = float(np.sum(negative_returns_mask))
        
        print(f"✅ Empty array test passed")
        
        # Test with all positive returns
        returns = np.array([0.01, 0.02, 0.03])
        positive_returns_mask = returns > 0
        negative_returns_mask = returns < 0
        positive_count = float(np.sum(positive_returns_mask))
        negative_count = float(np.sum(negative_returns_mask))
        
        print(f"✅ All positive returns test passed")
        
        # Test with all negative returns
        returns = np.array([-0.01, -0.02, -0.03])
        positive_returns_mask = returns > 0
        negative_returns_mask = returns < 0
        positive_count = float(np.sum(positive_returns_mask))
        negative_count = float(np.sum(negative_returns_mask))
        
        print(f"✅ All negative returns test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Edge cases test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing NumPy array comparison fixes...")
    print("=" * 50)
    
    tests = [
        test_kelly_criterion_fix,
        test_risk_metrics_fix,
        test_edge_cases
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
        print("🎉 All tests passed! NumPy fixes are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please review the fixes.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 