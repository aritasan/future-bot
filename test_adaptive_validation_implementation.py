#!/usr/bin/env python3
"""
Test script to verify adaptive validation implementation.
"""

import asyncio
import logging
import sys
import os
from typing import Dict, Any
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_adaptive_validation_implementation():
    """Test the adaptive validation implementation."""
    logger.info("Testing adaptive validation implementation...")
    
    try:
        # Import the required modules
        from src.quantitative.worldquant_validation_system import WorldQuantValidationSystem, ValidationResult
        from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
        
        # Test 1: WorldQuant Validation System with relaxed thresholds
        logger.info("Test 1: WorldQuant Validation System with relaxed thresholds")
        
        config = {
            'validation': {
                'min_confidence': 0.60,
                'max_risk': 0.25,
                'min_statistical_significance': 0.10
            }
        }
        
        validator = WorldQuantValidationSystem(config)
        
        # Check if thresholds are relaxed
        assert validator.worldquant_thresholds['min_confidence'] == 0.60, f"Expected 0.60, got {validator.worldquant_thresholds['min_confidence']}"
        assert validator.worldquant_thresholds['max_risk'] == 0.25, f"Expected 0.25, got {validator.worldquant_thresholds['max_risk']}"
        assert validator.worldquant_thresholds['min_statistical_significance'] == 0.10, f"Expected 0.10, got {validator.worldquant_thresholds['min_statistical_significance']}"
        
        logger.info("✅ Test 1 PASSED: Relaxed thresholds implemented correctly")
        
        # Test 2: Adaptive thresholds
        logger.info("Test 2: Adaptive thresholds")
        
        # Test high volatility market
        high_volatility_market = {
            'volatility': 0.08,
            'trend_strength': 0.3,
            'correlation': 0.6
        }
        
        adaptive_thresholds = validator.get_adaptive_thresholds(high_volatility_market)
        
        # Check if thresholds are relaxed for high volatility
        assert adaptive_thresholds['min_confidence'] < 0.60, f"Expected relaxed confidence for high volatility, got {adaptive_thresholds['min_confidence']}"
        assert adaptive_thresholds['max_risk'] > 0.25, f"Expected increased risk tolerance for high volatility, got {adaptive_thresholds['max_risk']}"
        
        logger.info("✅ Test 2 PASSED: Adaptive thresholds working correctly")
        
        # Test 3: Signal quality scoring
        logger.info("Test 3: Signal quality scoring")
        
        test_signal = {
            'confidence': 0.65,
            'p_value': 0.08,
            'risk_score': 0.20,
            'factor_exposure': 0.35
        }
        
        quality_score = validator.calculate_signal_quality_score(test_signal, high_volatility_market)
        
        assert 0 <= quality_score <= 1, f"Quality score should be between 0 and 1, got {quality_score}"
        
        logger.info(f"✅ Test 3 PASSED: Signal quality score: {quality_score:.2f}")
        
        # Test 4: WorldQuant validation with test signal
        logger.info("Test 4: WorldQuant validation with test signal")
        
        validation_result = await validator.validate_signal_worldquant(test_signal, high_volatility_market)
        
        assert isinstance(validation_result, ValidationResult), f"Expected ValidationResult, got {type(validation_result)}"
        
        logger.info(f"✅ Test 4 PASSED: Validation result - compliance: {validation_result.worldquant_compliance}, confidence: {validation_result.confidence_score:.2f}")
        
        # Test 5: Enhanced trading strategy integration
        logger.info("Test 5: Enhanced trading strategy integration")
        
        # Mock services for testing
        class MockBinanceService:
            async def get_market_data(self, symbol):
                return {'price': 100.0, 'volume': 1000.0}
        
        class MockIndicatorService:
            async def get_indicators(self, symbol):
                return {'rsi': 50.0, 'macd': 0.0}
        
        class MockNotificationService:
            async def send_notification(self, message):
                pass
        
        mock_binance = MockBinanceService()
        mock_indicator = MockIndicatorService()
        mock_notification = MockNotificationService()
        
        strategy = EnhancedTradingStrategyWithQuantitative(
            config, mock_binance, mock_indicator, mock_notification
        )
        
        # Check if WorldQuant validator is initialized
        assert hasattr(strategy, 'worldquant_validator'), "WorldQuant validator not initialized"
        assert isinstance(strategy.worldquant_validator, WorldQuantValidationSystem), "Invalid WorldQuant validator type"
        
        logger.info("✅ Test 5 PASSED: Enhanced trading strategy integration successful")
        
        # Test 6: Signal preparation for WorldQuant validation
        logger.info("Test 6: Signal preparation for WorldQuant validation")
        
        test_signal_basic = {
            'action': 'buy',
            'confidence': 0.65,
            'symbol': 'BTCUSDT'
        }
        
        test_market_data = {
            'volatility': 0.02,
            'trend_strength': 0.5,
            'correlation': 0.5,
            'returns': [0.01, -0.02, 0.03, -0.01, 0.02]
        }
        
        enhanced_signal = await strategy._prepare_signal_for_worldquant_validation(test_signal_basic, test_market_data)
        
        # Check if required metrics are added
        required_metrics = ['confidence', 'risk_score', 'p_value', 't_statistic', 'factor_exposure']
        for metric in required_metrics:
            assert metric in enhanced_signal, f"Missing required metric: {metric}"
        
        logger.info("✅ Test 6 PASSED: Signal preparation working correctly")
        
        logger.info("🎉 ALL TESTS PASSED! Adaptive validation implementation is working correctly.")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = asyncio.run(test_adaptive_validation_implementation())
    if success:
        print("\n🎯 Adaptive validation implementation is ready for production!")
    else:
        print("\n❌ Adaptive validation implementation needs fixes.")
        sys.exit(1)
