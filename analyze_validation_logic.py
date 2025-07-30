#!/usr/bin/env python3
"""
Script to analyze the validation logic and explain why most pairs fail quantitative validation
"""

import asyncio
import sys
import os
import logging
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from typing import Dict, Optional

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.quantitative.statistical_validator import StatisticalSignalValidator
from src.quantitative.quantitative_trading_system import QuantitativeTradingSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_validation_criteria():
    """Analyze the validation criteria and explain the logic."""
    logger.info("=== PHÂN TÍCH LOGIC VALIDATION ===")
    
    # Các tiêu chí validation hiện tại
    criteria = {
        'min_p_value': 0.05,  # p-value phải < 0.05 (statistically significant)
        'min_t_stat': 2.0,    # t-statistic phải > 2.0 (strong signal)
        'min_sharpe_ratio': 0.5,  # Sharpe ratio phải > 0.5 (good risk-adjusted return)
        'max_drawdown': 0.15   # Max drawdown phải < 15% (risk control)
    }
    
    logger.info("📊 CÁC TIÊU CHÍ VALIDATION HIỆN TẠI:")
    for criterion, value in criteria.items():
        logger.info(f"  - {criterion}: {value}")
    
    logger.info("\n🔍 LOGIC VALIDATION:")
    logger.info("Signal được coi là VALID khi TẤT CẢ các điều kiện sau đều thỏa mãn:")
    logger.info("  1. p_value < 0.05 (statistically significant)")
    logger.info("  2. |t_statistic| > 2.0 (strong signal)")
    logger.info("  3. sharpe_ratio > 0.5 (good risk-adjusted return)")
    logger.info("  4. max_drawdown < 0.15 (risk control)")
    
    return criteria

def test_validation_with_different_signals():
    """Test validation with different signal strengths."""
    logger.info("\n=== TEST VALIDATION VỚI CÁC SIGNAL KHÁC NHAU ===")
    
    validator = StatisticalSignalValidator()
    
    # Test cases với các signal strength khác nhau
    test_cases = [
        {'strength': 0.1, 'confidence': 0.3, 'description': 'Weak signal'},
        {'strength': 0.3, 'confidence': 0.5, 'description': 'Moderate signal'},
        {'strength': 0.5, 'confidence': 0.7, 'description': 'Strong signal'},
        {'strength': 0.7, 'confidence': 0.8, 'description': 'Very strong signal'},
        {'strength': 0.9, 'confidence': 0.9, 'description': 'Extremely strong signal'}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n🧪 Test Case {i}: {test_case['description']}")
        logger.info(f"   Signal strength: {test_case['strength']}")
        logger.info(f"   Confidence: {test_case['confidence']}")
        
        # Validate signal
        validation_result = validator.validate_signal(test_case)
        
        logger.info(f"   📈 Validation Results:")
        logger.info(f"     - is_valid: {validation_result['is_valid']}")
        logger.info(f"     - p_value: {validation_result['p_value']:.4f}")
        logger.info(f"     - t_statistic: {validation_result['t_statistic']:.4f}")
        logger.info(f"     - sharpe_ratio: {validation_result['sharpe_ratio']:.4f}")
        logger.info(f"     - max_drawdown: {validation_result['max_drawdown']:.4f}")
        
        # Analyze why it passed or failed
        if validation_result['is_valid']:
            logger.info(f"   ✅ PASSED: Signal meets all criteria")
        else:
            logger.info(f"   ❌ FAILED: Signal does not meet all criteria")
            if validation_result['p_value'] >= 0.05:
                logger.info(f"     - p_value {validation_result['p_value']:.4f} >= 0.05 (not statistically significant)")
            if abs(validation_result['t_statistic']) < 2.0:
                logger.info(f"     - |t_statistic| {abs(validation_result['t_statistic']):.4f} < 2.0 (weak signal)")
            if validation_result['sharpe_ratio'] <= 0.5:
                logger.info(f"     - sharpe_ratio {validation_result['sharpe_ratio']:.4f} <= 0.5 (poor risk-adjusted return)")
            if validation_result['max_drawdown'] >= 0.15:
                logger.info(f"     - max_drawdown {validation_result['max_drawdown']:.4f} >= 0.15 (high risk)")

def analyze_synthetic_returns_generation():
    """Analyze how synthetic returns are generated."""
    logger.info("\n=== PHÂN TÍCH SINH DỮ LIỆU SYNTHETIC RETURNS ===")
    
    validator = StatisticalSignalValidator()
    
    # Test với các signal khác nhau
    test_signals = [
        {'strength': 0.1, 'confidence': 0.3},
        {'strength': 0.5, 'confidence': 0.7},
        {'strength': 0.9, 'confidence': 0.9}
    ]
    
    for i, signal in enumerate(test_signals, 1):
        logger.info(f"\n📊 Test {i}: Signal {signal}")
        
        # Generate synthetic returns
        returns = validator._generate_synthetic_returns(signal)
        
        logger.info(f"   Generated returns statistics:")
        logger.info(f"     - Mean: {np.mean(returns):.6f}")
        logger.info(f"     - Std: {np.std(returns):.6f}")
        logger.info(f"     - Min: {np.min(returns):.6f}")
        logger.info(f"     - Max: {np.max(returns):.6f}")
        
        # Calculate expected return based on formula
        expected_return = signal['strength'] * signal['confidence'] * 0.02
        logger.info(f"   Expected return (formula): {expected_return:.6f}")
        logger.info(f"   Actual mean return: {np.mean(returns):.6f}")

def suggest_validation_improvements():
    """Suggest improvements to make validation more realistic."""
    logger.info("\n=== ĐỀ XUẤT CẢI THIỆN VALIDATION ===")
    
    logger.info("🔧 CÁC VẤN ĐỀ HIỆN TẠI:")
    logger.info("1. Tiêu chí quá nghiêm ngặt cho trading bot thực tế")
    logger.info("2. Synthetic returns không phản ánh đúng thị trường thực")
    logger.info("3. Thiếu context về market conditions")
    logger.info("4. Không có adaptive thresholds")
    
    logger.info("\n💡 ĐỀ XUẤT CẢI THIỆN:")
    logger.info("1. Giảm bớt tiêu chí validation:")
    logger.info("   - min_p_value: 0.05 → 0.1 (less strict)")
    logger.info("   - min_t_stat: 2.0 → 1.5 (more realistic)")
    logger.info("   - min_sharpe_ratio: 0.5 → 0.2 (more achievable)")
    logger.info("   - max_drawdown: 0.15 → 0.25 (more realistic)")
    
    logger.info("\n2. Thêm adaptive validation:")
    logger.info("   - Điều chỉnh thresholds dựa trên market volatility")
    logger.info("   - Sử dụng real market data thay vì synthetic")
    logger.info("   - Thêm market regime detection")
    
    logger.info("\n3. Cải thiện signal generation:")
    logger.info("   - Sử dụng real historical data")
    logger.info("   - Thêm market microstructure analysis")
    logger.info("   - Implement proper backtesting")

def test_relaxed_validation():
    """Test with relaxed validation criteria."""
    logger.info("\n=== TEST VỚI TIÊU CHÍ VALIDATION NỚI LỎNG ===")
    
    # Tạo validator với tiêu chí nới lỏng
    relaxed_validator = StatisticalSignalValidator(
        min_p_value=0.1,  # Nới lỏng từ 0.05
        min_t_stat=1.5    # Nới lỏng từ 2.0
    )
    
    # Override validation logic để nới lỏng thêm
    def relaxed_validate_signal(self, signal_data, historical_returns=None):
        try:
            results = {
                'is_valid': False,
                'p_value': None,
                't_statistic': None,
                'sharpe_ratio': None,
                'information_ratio': None,
                'calmar_ratio': None,
                'sortino_ratio': None,
                'max_drawdown': None,
                'volatility': None,
                'skewness': None,
                'kurtosis': None
            }
            
            if historical_returns is None:
                historical_returns = self._generate_synthetic_returns(signal_data)
            
            # Perform statistical tests
            t_stat, p_value = stats.ttest_1samp(historical_returns, 0)
            
            # Calculate risk-adjusted metrics
            sharpe_ratio = self._calculate_sharpe_ratio(historical_returns)
            information_ratio = self._calculate_information_ratio(historical_returns)
            sortino_ratio = self._calculate_sortino_ratio(historical_returns)
            calmar_ratio = self._calculate_calmar_ratio(historical_returns)
            
            # Calculate additional metrics
            max_drawdown = self._calculate_max_drawdown(historical_returns)
            volatility = float(np.std(historical_returns) * np.sqrt(252))
            skewness = stats.skew(historical_returns)
            kurtosis = stats.kurtosis(historical_returns)
            
            # RELAXED CRITERIA
            is_valid = (p_value < 0.1 and  # Nới lỏng từ 0.05
                       abs(t_stat) > 1.5 and  # Nới lỏng từ 2.0
                       sharpe_ratio > 0.2 and  # Nới lỏng từ 0.5
                       max_drawdown < 0.25)    # Nới lỏng từ 0.15
            
            results.update({
                'is_valid': is_valid,
                'p_value': p_value,
                't_statistic': t_stat,
                'sharpe_ratio': sharpe_ratio,
                'information_ratio': information_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'skewness': skewness,
                'kurtosis': kurtosis
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in relaxed signal validation: {str(e)}")
            return {'is_valid': False, 'error': str(e)}
    
    # Test với relaxed criteria
    test_signals = [
        {'strength': 0.3, 'confidence': 0.5, 'description': 'Moderate signal'},
        {'strength': 0.5, 'confidence': 0.7, 'description': 'Strong signal'},
        {'strength': 0.7, 'confidence': 0.8, 'description': 'Very strong signal'}
    ]
    
    for i, test_case in enumerate(test_signals, 1):
        logger.info(f"\n🧪 Relaxed Test {i}: {test_case['description']}")
        
        # Sử dụng logic validation nới lỏng
        validation_result = relaxed_validate_signal(relaxed_validator, test_case)
        
        logger.info(f"   📈 Relaxed Validation Results:")
        logger.info(f"     - is_valid: {validation_result['is_valid']}")
        logger.info(f"     - p_value: {validation_result['p_value']:.4f} (threshold: 0.1)")
        logger.info(f"     - t_statistic: {validation_result['t_statistic']:.4f} (threshold: 1.5)")
        logger.info(f"     - sharpe_ratio: {validation_result['sharpe_ratio']:.4f} (threshold: 0.2)")
        logger.info(f"     - max_drawdown: {validation_result['max_drawdown']:.4f} (threshold: 0.25)")
        
        if validation_result['is_valid']:
            logger.info(f"   ✅ PASSED with relaxed criteria")
        else:
            logger.info(f"   ❌ Still FAILED even with relaxed criteria")

async def main():
    """Run the analysis."""
    logger.info("🔍 PHÂN TÍCH LOGIC VALIDATION VÀ LÝ DO FAILED")
    
    # Phân tích tiêu chí validation
    analyze_validation_criteria()
    
    # Test với các signal khác nhau
    test_validation_with_different_signals()
    
    # Phân tích synthetic returns
    analyze_synthetic_returns_generation()
    
    # Đề xuất cải thiện
    suggest_validation_improvements()
    
    # Test với tiêu chí nới lỏng
    test_relaxed_validation()
    
    logger.info("\n🎯 KẾT LUẬN:")
    logger.info("Hầu hết các pairs failed validation vì:")
    logger.info("1. Tiêu chí validation quá nghiêm ngặt cho trading bot thực tế")
    logger.info("2. Synthetic returns không phản ánh đúng market conditions")
    logger.info("3. Thiếu adaptive thresholds cho các market conditions khác nhau")
    logger.info("4. Cần cải thiện signal generation và validation logic")

if __name__ == "__main__":
    asyncio.run(main()) 