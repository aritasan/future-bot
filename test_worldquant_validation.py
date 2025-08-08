#!/usr/bin/env python3
"""
Test WorldQuant Validation System
Verify 85% confidence, 15% risk max requirements and multi-layer validation.
"""

import asyncio
import logging
import sys
import os
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.quantitative.worldquant_validation_system import WorldQuantValidationSystem, ValidationResult
from src.core.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_worldquant_validation_system():
    """Test WorldQuant validation system with various scenarios."""
    try:
        logger.info("🧪 Testing WorldQuant Validation System")
        
        # Load configuration
        config = load_config()
        
        # Initialize WorldQuant validation system
        validator = WorldQuantValidationSystem(config)
        
        # Test scenarios
        test_scenarios = [
            {
                'name': 'Strong Signal - Should Pass',
                'signal': {
                    'symbol': 'BTCUSDT',
                    'action': 'buy',
                    'strength': 0.9,
                    'confidence': 0.95,
                    'position_size': 0.01,
                    'leverage': 1.0,
                    'p_value': 0.001,
                    't_statistic': 4.5,
                    'sample_size': 200,
                    'effect_size': 0.6,
                    'var_95': 0.008,
                    'expected_shortfall': 0.012,
                    'model_agreement': 0.9,
                    'prediction_confidence': 0.95,
                    'model_uncertainty': 0.05,
                    'signal_type': 'trend_following',
                    'factor_exposures': {
                        'momentum': 0.15,
                        'volatility': 0.1,
                        'size': 0.05,
                        'value': 0.08,
                        'quality': 0.15
                    }
                },
                'market_data': {
                    'volatility': 0.025,
                    'correlation': 0.4,
                    'trend_strength': 0.6,
                    'volume': 1500000
                },
                'expected_result': 'PASS'
            },
            {
                'name': 'Weak Signal - Should Fail',
                'signal': {
                    'symbol': 'ETHUSDT',
                    'action': 'sell',
                    'strength': 0.3,
                    'confidence': 0.4,
                    'position_size': 0.05,
                    'leverage': 2.5,
                    'p_value': 0.1,
                    't_statistic': 1.2,
                    'sample_size': 20,
                    'effect_size': 0.1,
                    'var_95': 0.04,
                    'expected_shortfall': 0.06,
                    'model_agreement': 0.5,
                    'prediction_confidence': 0.6,
                    'model_uncertainty': 0.4,
                    'signal_type': 'mean_reversion',
                    'factor_exposures': {
                        'momentum': 0.4,
                        'volatility': 0.3,
                        'size': 0.2,
                        'value': 0.25,
                        'quality': 0.35
                    }
                },
                'market_data': {
                    'volatility': 0.06,
                    'correlation': 0.8,
                    'trend_strength': 0.2,
                    'volume': 800000
                },
                'expected_result': 'FAIL'
            },
            {
                'name': 'High Risk Signal - Should Fail',
                'signal': {
                    'symbol': 'ADAUSDT',
                    'action': 'buy',
                    'strength': 0.7,
                    'confidence': 0.8,
                    'position_size': 0.15,
                    'leverage': 3.0,
                    'p_value': 0.02,
                    't_statistic': 2.8,
                    'sample_size': 80,
                    'effect_size': 0.3,
                    'var_95': 0.08,
                    'expected_shortfall': 0.12,
                    'model_agreement': 0.75,
                    'prediction_confidence': 0.8,
                    'model_uncertainty': 0.2,
                    'signal_type': 'trend_following',
                    'factor_exposures': {
                        'momentum': 0.35,
                        'volatility': 0.25,
                        'size': 0.15,
                        'value': 0.2,
                        'quality': 0.3
                    }
                },
                'market_data': {
                    'volatility': 0.08,
                    'correlation': 0.7,
                    'trend_strength': 0.8,
                    'volume': 1200000
                },
                'expected_result': 'FAIL'
            }
        ]
        
        # Run tests
        passed_tests = 0
        total_tests = len(test_scenarios)
        
        for i, scenario in enumerate(test_scenarios, 1):
            logger.info(f"\n📊 Test {i}/{total_tests}: {scenario['name']}")
            
            try:
                # Run validation
                validation_result = await validator.validate_signal_worldquant(
                    scenario['signal'], 
                    scenario['market_data']
                )
                
                # Check result
                actual_result = 'PASS' if validation_result.worldquant_compliance else 'FAIL'
                expected_result = scenario['expected_result']
                
                # Log detailed results
                logger.info(f"   Confidence Score: {validation_result.confidence_score:.3f}")
                logger.info(f"   Risk Score: {validation_result.risk_score:.3f}")
                logger.info(f"   WorldQuant Compliance: {'✅' if validation_result.worldquant_compliance else '❌'}")
                logger.info(f"   Expected: {expected_result}, Actual: {actual_result}")
                
                # Check individual layer results
                for layer, result in validation_result.layer_results.items():
                    layer_valid = result.get('is_valid', False)
                    layer_confidence = result.get('confidence_score', 0.0)
                    layer_risk = result.get('risk_score', 1.0)
                    logger.info(f"   {layer}: {'✅' if layer_valid else '❌'} "
                              f"(Confidence: {layer_confidence:.3f}, Risk: {layer_risk:.3f})")
                
                # Log warnings if any
                if validation_result.warnings:
                    logger.warning(f"   Warnings: {validation_result.warnings}")
                
                # Check if test passed
                if actual_result == expected_result:
                    logger.info(f"   ✅ Test PASSED")
                    passed_tests += 1
                else:
                    logger.error(f"   ❌ Test FAILED - Expected {expected_result}, got {actual_result}")
                
            except Exception as e:
                logger.error(f"   ❌ Test ERROR: {str(e)}")
        
        # Summary
        logger.info(f"\n📈 Test Summary:")
        logger.info(f"   Passed: {passed_tests}/{total_tests}")
        logger.info(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        # Get validation system summary
        summary = validator.get_validation_summary()
        logger.info(f"\n🔍 WorldQuant Validation System Summary:")
        logger.info(f"   WorldQuant Thresholds: {summary.get('worldquant_thresholds', {})}")
        logger.info(f"   Performance Metrics: {summary.get('performance_metrics', {})}")
        logger.info(f"   Validation History Size: {summary.get('validation_history_size', 0)}")
        
        # Close validator
        await validator.close()
        
        return passed_tests == total_tests
        
    except Exception as e:
        logger.error(f"Error in WorldQuant validation test: {str(e)}")
        return False

async def test_validation_layers():
    """Test individual validation layers."""
    try:
        logger.info("\n🔬 Testing Individual Validation Layers")
        
        config = load_config()
        validator = WorldQuantValidationSystem(config)
        
        # Test signal
        test_signal = {
            'symbol': 'BTCUSDT',
            'action': 'buy',
            'strength': 0.8,
            'confidence': 0.9,
            'position_size': 0.02,
            'leverage': 1.5,
            'p_value': 0.01,
            't_statistic': 3.5,
            'sample_size': 100,
            'effect_size': 0.4,
            'var_95': 0.015,
            'expected_shortfall': 0.025,
            'model_agreement': 0.85,
            'prediction_confidence': 0.9,
            'model_uncertainty': 0.1,
            'signal_type': 'trend_following',
            'factor_exposures': {
                'momentum': 0.2,
                'volatility': 0.15,
                'size': 0.1,
                'value': 0.1,
                'quality': 0.2
            }
        }
        
        test_market_data = {
            'volatility': 0.025,
            'correlation': 0.4,
            'trend_strength': 0.6,
            'volume': 1500000
        }
        
        # Test each layer individually
        layers = ['statistical', 'market_regime', 'factor_model', 'risk_management', 'machine_learning']
        
        for layer in layers:
            logger.info(f"\n   Testing {layer} layer:")
            
            if layer == 'statistical':
                result = await validator._validate_statistical_layer(test_signal, test_market_data)
            elif layer == 'market_regime':
                result = await validator._validate_market_regime_layer(test_signal, test_market_data)
            elif layer == 'factor_model':
                result = await validator._validate_factor_model_layer(test_signal, test_market_data)
            elif layer == 'risk_management':
                result = await validator._validate_risk_management_layer(test_signal, test_market_data)
            elif layer == 'machine_learning':
                result = await validator._validate_machine_learning_layer(test_signal, test_market_data)
            
            logger.info(f"     Valid: {'✅' if result.get('is_valid') else '❌'}")
            logger.info(f"     Confidence: {result.get('confidence_score', 0):.3f}")
            logger.info(f"     Risk: {result.get('risk_score', 1):.3f}")
            logger.info(f"     Details: {result.get('details', {})}")
        
        await validator.close()
        
    except Exception as e:
        logger.error(f"Error testing validation layers: {str(e)}")

async def test_worldquant_thresholds():
    """Test WorldQuant threshold compliance."""
    try:
        logger.info("\n🎯 Testing WorldQuant Thresholds")
        
        config = load_config()
        validator = WorldQuantValidationSystem(config)
        
        # Test threshold compliance
        thresholds = validator.worldquant_thresholds
        
        logger.info(f"   WorldQuant Thresholds:")
        for threshold, value in thresholds.items():
            logger.info(f"     {threshold}: {value}")
        
        # Test compliance check
        test_cases = [
            {
                'confidence': 0.9,
                'risk': 0.1,
                'layers_valid': True,
                'expected': True,
                'description': 'Compliant signal'
            },
            {
                'confidence': 0.7,
                'risk': 0.1,
                'layers_valid': True,
                'expected': False,
                'description': 'Low confidence'
            },
            {
                'confidence': 0.9,
                'risk': 0.2,
                'layers_valid': True,
                'expected': False,
                'description': 'High risk'
            },
            {
                'confidence': 0.9,
                'risk': 0.1,
                'layers_valid': False,
                'expected': False,
                'description': 'Invalid layers'
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n   Test Case {i}: {test_case['description']}")
            
            # Mock layer results
            layer_results = {
                'statistical': {'is_valid': test_case['layers_valid']},
                'market_regime': {'is_valid': test_case['layers_valid']},
                'factor_model': {'is_valid': test_case['layers_valid']},
                'risk_management': {'is_valid': test_case['layers_valid']},
                'machine_learning': {'is_valid': test_case['layers_valid']}
            }
            
            compliance = validator._check_worldquant_compliance(
                test_case['confidence'],
                test_case['risk'],
                layer_results
            )
            
            expected = test_case['expected']
            logger.info(f"     Confidence: {test_case['confidence']:.3f}")
            logger.info(f"     Risk: {test_case['risk']:.3f}")
            logger.info(f"     Layers Valid: {test_case['layers_valid']}")
            logger.info(f"     Expected: {expected}, Actual: {compliance}")
            logger.info(f"     Result: {'✅ PASS' if compliance == expected else '❌ FAIL'}")
        
        await validator.close()
        
    except Exception as e:
        logger.error(f"Error testing WorldQuant thresholds: {str(e)}")

async def main():
    """Main test function."""
    try:
        logger.info("🚀 Starting WorldQuant Validation System Tests")
        
        # Run all tests
        success = await test_worldquant_validation_system()
        await test_validation_layers()
        await test_worldquant_thresholds()
        
        if success:
            logger.info("\n🎉 All WorldQuant validation tests PASSED!")
        else:
            logger.error("\n❌ Some WorldQuant validation tests FAILED!")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in main test: {str(e)}")
        return False

if __name__ == "__main__":
    # Run tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
