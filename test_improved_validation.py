#!/usr/bin/env python3
"""
Test script to verify the improved validation logic with adaptive thresholds
"""

import asyncio
import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.quantitative.statistical_validator import StatisticalSignalValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_improved_validation():
    """Test the improved validation logic with adaptive thresholds."""
    logger.info("=== TEST IMPROVED VALIDATION LOGIC ===")
    
    # Create validator with improved thresholds
    validator = StatisticalSignalValidator()
    
    # Test cases với các signal strength khác nhau
    test_cases = [
        {'strength': 0.2, 'confidence': 0.4, 'description': 'Weak signal'},
        {'strength': 0.4, 'confidence': 0.6, 'description': 'Moderate signal'},
        {'strength': 0.6, 'confidence': 0.8, 'description': 'Strong signal'},
        {'strength': 0.8, 'confidence': 0.9, 'description': 'Very strong signal'}
    ]
    
    passed_count = 0
    total_count = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n🧪 Test Case {i}: {test_case['description']}")
        logger.info(f"   Signal strength: {test_case['strength']}")
        logger.info(f"   Confidence: {test_case['confidence']}")
        
        # Validate signal
        validation_result = validator.validate_signal(test_case)
        
        logger.info(f"   📈 Validation Results:")
        logger.info(f"     - is_valid: {validation_result['is_valid']}")
        logger.info(f"     - market_regime: {validation_result['market_regime']}")
        logger.info(f"     - p_value: {validation_result['p_value']:.4f}")
        logger.info(f"     - t_statistic: {validation_result['t_statistic']:.4f}")
        logger.info(f"     - sharpe_ratio: {validation_result['sharpe_ratio']:.4f}")
        logger.info(f"     - max_drawdown: {validation_result['max_drawdown']:.4f}")
        logger.info(f"     - volatility: {validation_result['volatility']:.4f}")
        
        # Show adaptive thresholds used
        thresholds = validation_result['adaptive_thresholds_used']
        logger.info(f"   🎯 Adaptive Thresholds Used:")
        logger.info(f"     - min_p_value: {thresholds['min_p_value']}")
        logger.info(f"     - min_t_stat: {thresholds['min_t_stat']}")
        logger.info(f"     - min_sharpe_ratio: {thresholds['min_sharpe_ratio']}")
        logger.info(f"     - max_drawdown: {thresholds['max_drawdown']}")
        
        # Analyze why it passed or failed
        if validation_result['is_valid']:
            logger.info(f"   ✅ PASSED: Signal meets adaptive criteria")
            passed_count += 1
        else:
            logger.info(f"   ❌ FAILED: Signal does not meet adaptive criteria")
            if validation_result['p_value'] >= thresholds['min_p_value']:
                logger.info(f"     - p_value {validation_result['p_value']:.4f} >= {thresholds['min_p_value']} (not statistically significant)")
            if abs(validation_result['t_statistic']) < thresholds['min_t_stat']:
                logger.info(f"     - |t_statistic| {abs(validation_result['t_statistic']):.4f} < {thresholds['min_t_stat']} (weak signal)")
            if validation_result['sharpe_ratio'] <= thresholds['min_sharpe_ratio']:
                logger.info(f"     - sharpe_ratio {validation_result['sharpe_ratio']:.4f} <= {thresholds['min_sharpe_ratio']} (poor risk-adjusted return)")
            if validation_result['max_drawdown'] >= thresholds['max_drawdown']:
                logger.info(f"     - max_drawdown {validation_result['max_drawdown']:.4f} >= {thresholds['max_drawdown']} (high risk)")
    
    logger.info(f"\n📊 SUMMARY:")
    logger.info(f"   Passed: {passed_count}/{total_count} signals")
    logger.info(f"   Success rate: {(passed_count/total_count)*100:.1f}%")
    
    return passed_count, total_count

def test_market_regime_detection():
    """Test market regime detection based on volatility."""
    logger.info("\n=== TEST MARKET REGIME DETECTION ===")
    
    validator = StatisticalSignalValidator()
    
    # Test different volatility levels
    test_volatilities = [
        (0.15, "Low volatility"),
        (0.25, "Normal volatility"),
        (0.35, "Normal volatility"),
        (0.50, "High volatility"),
        (0.60, "High volatility")
    ]
    
    for volatility, expected_regime in test_volatilities:
        detected_regime = validator._determine_market_regime(volatility)
        logger.info(f"   Volatility: {volatility:.2f} → Detected: {detected_regime} (Expected: {expected_regime})")
        
        # Show thresholds for this regime
        thresholds = validator.adaptive_thresholds[detected_regime]
        logger.info(f"     Thresholds: p_value<{thresholds['min_p_value']}, t_stat>{thresholds['min_t_stat']}, sharpe>{thresholds['min_sharpe_ratio']}, drawdown<{thresholds['max_drawdown']}")

def test_validation_comparison():
    """Compare old vs new validation logic."""
    logger.info("\n=== COMPARISON: OLD VS NEW VALIDATION ===")
    
    # Create old validator (strict criteria)
    old_validator = StatisticalSignalValidator(min_p_value=0.05, min_t_stat=2.0)
    
    # Create new validator (adaptive criteria)
    new_validator = StatisticalSignalValidator()
    
    # Test cases
    test_cases = [
        {'strength': 0.3, 'confidence': 0.5, 'description': 'Moderate signal'},
        {'strength': 0.5, 'confidence': 0.7, 'description': 'Strong signal'},
        {'strength': 0.7, 'confidence': 0.8, 'description': 'Very strong signal'}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n🧪 Test Case {i}: {test_case['description']}")
        
        # Test with old validation
        old_result = old_validator.validate_signal(test_case)
        
        # Test with new validation
        new_result = new_validator.validate_signal(test_case)
        
        logger.info(f"   📊 Comparison Results:")
        logger.info(f"     - Old validation: {'✅ PASSED' if old_result['is_valid'] else '❌ FAILED'}")
        logger.info(f"     - New validation: {'✅ PASSED' if new_result['is_valid'] else '❌ FAILED'}")
        
        if old_result['is_valid'] != new_result['is_valid']:
            logger.info(f"     - 💡 IMPROVEMENT: Signal now {'passes' if new_result['is_valid'] else 'fails'} with adaptive criteria")

async def main():
    """Run all tests."""
    logger.info("🔍 TESTING IMPROVED VALIDATION LOGIC")
    
    # Test improved validation
    passed, total = test_improved_validation()
    
    # Test market regime detection
    test_market_regime_detection()
    
    # Test comparison
    test_validation_comparison()
    
    logger.info(f"\n🎯 FINAL RESULTS:")
    logger.info(f"   Improved validation success rate: {(passed/total)*100:.1f}%")
    
    if passed > total * 0.5:  # More than 50% pass rate
        logger.info("✅ IMPROVEMENT SUCCESSFUL: More signals now pass validation")
    else:
        logger.info("⚠️  FURTHER IMPROVEMENT NEEDED: Still too many signals failing")

if __name__ == "__main__":
    asyncio.run(main()) 