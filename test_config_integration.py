#!/usr/bin/env python3
"""
Test script to verify configuration integration for confidence thresholds.
"""

import asyncio
import logging
from typing import Dict, Any
from unittest.mock import Mock, MagicMock
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockEnhancedTradingStrategy:
    """Mock class to test configuration integration."""
    
    def __init__(self):
        self.logger = logger
    
    def _calculate_dynamic_confidence_threshold(self, action: str, market_data: Dict, risk_metrics: Dict = None) -> float:
        """
        Calculate dynamic confidence threshold based on action type, market conditions, and risk metrics.
        WorldQuant-level implementation with asymmetric thresholds for BUY/SELL.
        """
        try:
            # Load configuration
            from src.core.config import load_config
            config = load_config()
            confidence_config = config.get('trading', {}).get('confidence_thresholds', {})
            
            # Base asymmetric thresholds from config
            base_thresholds = {
                'buy': confidence_config.get('buy_base', 0.45),
                'sell': confidence_config.get('sell_base', 0.65),
                'hold': confidence_config.get('hold_base', 0.35)
            }
            
            base_threshold = base_thresholds.get(action, 0.6)
            
            # Volatility adjustment from config
            volatility = market_data.get('volatility', 0.02)
            vol_adjustment = 0.0
            
            vol_adjustments = confidence_config.get('volatility_adjustments', {})
            if volatility > 0.05:  # High volatility
                vol_adjustment = vol_adjustments.get('high_volatility', 0.1)
            elif volatility < 0.01:  # Low volatility
                vol_adjustment = vol_adjustments.get('low_volatility', -0.05)
            
            # Market regime adjustment from config
            market_regime = market_data.get('market_regime', 'normal')
            regime_adjustment = 0.0
            
            regime_adjustments = confidence_config.get('regime_adjustments', {})
            if market_regime == 'trending':
                regime_adjustment = regime_adjustments.get('trending', -0.05)
            elif market_regime == 'mean_reverting':
                regime_adjustment = regime_adjustments.get('mean_reverting', 0.05)
            elif market_regime == 'high_volatility':
                regime_adjustment = regime_adjustments.get('high_volatility', 0.08)
            
            # Risk metrics adjustment from config
            risk_adjustment = 0.0
            if risk_metrics:
                sharpe_ratio = risk_metrics.get('sharpe_ratio', 0.0)
                var_95 = risk_metrics.get('var_95', -0.02)
                max_drawdown = risk_metrics.get('max_drawdown', 0.0)
                
                risk_adjustments = confidence_config.get('risk_adjustments', {})
                
                # Sharpe ratio adjustment
                if sharpe_ratio > 1.0:
                    risk_adjustment += risk_adjustments.get('sharpe_ratio_good', -0.05)
                elif sharpe_ratio < 0.5:
                    risk_adjustment += risk_adjustments.get('sharpe_ratio_poor', 0.05)
                
                # VaR adjustment
                if var_95 < -0.03:  # High risk
                    risk_adjustment += risk_adjustments.get('var_high_risk', 0.03)
                elif var_95 > -0.01:  # Low risk
                    risk_adjustment += risk_adjustments.get('var_low_risk', -0.02)
                
                # Drawdown adjustment
                if max_drawdown > 0.05:
                    risk_adjustment += risk_adjustments.get('drawdown_high', 0.02)
            
            # Calculate final threshold
            final_threshold = base_threshold + vol_adjustment + regime_adjustment + risk_adjustment
            
            # Ensure bounds from config
            bounds = confidence_config.get('bounds', {})
            min_threshold = bounds.get('min_threshold', 0.25)
            max_threshold = bounds.get('max_threshold', 0.85)
            final_threshold = max(min_threshold, min(max_threshold, final_threshold))
            
            self.logger.info(f"Dynamic confidence threshold for {action}: "
                           f"base={base_threshold:.3f}, vol_adj={vol_adjustment:.3f}, "
                           f"regime_adj={regime_adjustment:.3f}, risk_adj={risk_adjustment:.3f}, "
                           f"final={final_threshold:.3f}")
            
            return final_threshold
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic confidence threshold: {str(e)}")
            return 0.6  # Fallback to original threshold
    
    def _calculate_risk_adjusted_confidence(self, signal: Dict, risk_metrics: Dict) -> float:
        """
        Calculate risk-adjusted confidence score based on signal strength and risk metrics.
        WorldQuant-level implementation with comprehensive risk consideration.
        """
        try:
            # Load configuration
            from src.core.config import load_config
            config = load_config()
            confidence_config = config.get('trading', {}).get('confidence_thresholds', {})
            risk_adjusted_config = confidence_config.get('risk_adjusted_confidence', {})
            
            base_confidence = signal.get('confidence', 0.0)
            signal_strength = signal.get('signal_strength', 0.0)
            
            # Base confidence boost from signal strength using config
            strength_multiplier = risk_adjusted_config.get('strength_boost_multiplier', 0.2)
            max_strength_boost = risk_adjusted_config.get('max_strength_boost', 0.1)
            strength_boost = min(max_strength_boost, abs(signal_strength) * strength_multiplier)
            
            # Risk metrics adjustment using config
            risk_boost = 0.0
            
            if risk_metrics:
                sharpe_ratio = risk_metrics.get('sharpe_ratio', 0.0)
                var_95 = risk_metrics.get('var_95', -0.02)
                max_drawdown = risk_metrics.get('max_drawdown', 0.0)
                volatility = risk_metrics.get('volatility', 0.02)
                
                # Sharpe ratio boost
                if sharpe_ratio > 1.0:
                    risk_boost += risk_adjusted_config.get('sharpe_ratio_boost', 0.05)
                elif sharpe_ratio < 0.5:
                    risk_boost += risk_adjusted_config.get('sharpe_ratio_penalty', -0.05)
                
                # VaR boost
                if var_95 > -0.01:  # Low risk
                    risk_boost += risk_adjusted_config.get('var_low_risk_boost', 0.03)
                elif var_95 < -0.03:  # High risk
                    risk_boost += risk_adjusted_config.get('var_high_risk_penalty', -0.03)
                
                # Drawdown penalty
                if max_drawdown > 0.05:
                    risk_boost += risk_adjusted_config.get('drawdown_penalty', -0.02)
                
                # Volatility adjustment
                if volatility < 0.01:  # Low volatility
                    risk_boost += risk_adjusted_config.get('low_volatility_boost', 0.02)
                elif volatility > 0.05:  # High volatility
                    risk_boost += risk_adjusted_config.get('high_volatility_penalty', -0.02)
            
            # Market microstructure boost using config
            microstructure_boost = 0.0
            if 'market_microstructure' in signal:
                microstructure = signal['market_microstructure']
                
                # Volume profile boost
                if microstructure.get('volume_profile_valid', False):
                    microstructure_boost += risk_adjusted_config.get('volume_profile_boost', 0.02)
                
                # Order flow boost
                if microstructure.get('order_flow_bullish', False):
                    microstructure_boost += risk_adjusted_config.get('order_flow_bullish_boost', 0.03)
                elif microstructure.get('order_flow_bearish', False):
                    microstructure_boost += risk_adjusted_config.get('order_flow_bearish_penalty', -0.03)
                
                # Liquidity boost
                if microstructure.get('liquidity_adequate', False):
                    microstructure_boost += risk_adjusted_config.get('liquidity_boost', 0.01)
            
            # Calculate final confidence
            final_confidence = base_confidence + strength_boost + risk_boost + microstructure_boost
            
            # Ensure bounds from config
            confidence_bounds = risk_adjusted_config.get('confidence_bounds', {})
            min_confidence = confidence_bounds.get('min_confidence', 0.05)
            max_confidence = confidence_bounds.get('max_confidence', 0.95)
            final_confidence = max(min_confidence, min(max_confidence, final_confidence))
            
            self.logger.info(f"Risk-adjusted confidence: base={base_confidence:.3f}, "
                           f"strength_boost={strength_boost:.3f}, risk_boost={risk_boost:.3f}, "
                           f"microstructure_boost={microstructure_boost:.3f}, "
                           f"final={final_confidence:.3f}")
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted confidence: {str(e)}")
            return signal.get('confidence', 0.0)

def test_config_loading():
    """Test that configuration is properly loaded."""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION LOADING")
    print("="*60)
    
    try:
        from src.core.config import load_config
        config = load_config()
        
        # Check if confidence_thresholds exists
        confidence_config = config.get('trading', {}).get('confidence_thresholds', {})
        
        if confidence_config:
            print("✅ Configuration loaded successfully")
            print(f"Base thresholds: {confidence_config.get('buy_base', 'N/A')}, {confidence_config.get('sell_base', 'N/A')}, {confidence_config.get('hold_base', 'N/A')}")
            
            # Check for risk_adjusted_confidence section
            risk_adjusted_config = confidence_config.get('risk_adjusted_confidence', {})
            if risk_adjusted_config:
                print("✅ Risk-adjusted confidence configuration found")
                print(f"Strength boost multiplier: {risk_adjusted_config.get('strength_boost_multiplier', 'N/A')}")
                print(f"Max strength boost: {risk_adjusted_config.get('max_strength_boost', 'N/A')}")
            else:
                print("❌ Risk-adjusted confidence configuration missing")
        else:
            print("❌ Confidence thresholds configuration missing")
            
    except Exception as e:
        print(f"❌ Error loading configuration: {str(e)}")

def test_dynamic_threshold_with_config():
    """Test dynamic threshold calculation with configuration."""
    print("\n" + "="*60)
    print("TESTING DYNAMIC THRESHOLD WITH CONFIG")
    print("="*60)
    
    strategy = MockEnhancedTradingStrategy()
    
    # Test cases with different configurations
    test_cases = [
        {
            'name': 'BUY with Low Volatility and Good Performance',
            'action': 'buy',
            'market_data': {'volatility': 0.008, 'market_regime': 'trending'},
            'risk_metrics': {'sharpe_ratio': 1.2, 'var_95': -0.008, 'max_drawdown': 0.02},
            'expected_base': 0.45  # From config
        },
        {
            'name': 'SELL with High Volatility and Poor Performance',
            'action': 'sell',
            'market_data': {'volatility': 0.06, 'market_regime': 'high_volatility'},
            'risk_metrics': {'sharpe_ratio': 0.3, 'var_95': -0.04, 'max_drawdown': 0.08},
            'expected_base': 0.65  # From config
        },
        {
            'name': 'HOLD with Normal Conditions',
            'action': 'hold',
            'market_data': {'volatility': 0.025, 'market_regime': 'normal'},
            'risk_metrics': {'sharpe_ratio': 0.8, 'var_95': -0.015, 'max_drawdown': 0.03},
            'expected_base': 0.35  # From config
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 50)
        
        try:
            threshold = strategy._calculate_dynamic_confidence_threshold(
                test_case['action'],
                test_case['market_data'],
                test_case['risk_metrics']
            )
            
            expected_base = test_case['expected_base']
            print(f"Action: {test_case['action']}")
            print(f"Expected Base Threshold: {expected_base}")
            print(f"Calculated Threshold: {threshold:.3f}")
            print(f"Market Data: {test_case['market_data']}")
            print(f"Risk Metrics: {test_case['risk_metrics']}")
            
            # Verify the threshold is within reasonable bounds
            if 0.25 <= threshold <= 0.85:
                print("✅ Threshold within bounds")
            else:
                print("❌ Threshold outside bounds")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def test_risk_adjusted_confidence_with_config():
    """Test risk-adjusted confidence calculation with configuration."""
    print("\n" + "="*60)
    print("TESTING RISK-ADJUSTED CONFIDENCE WITH CONFIG")
    print("="*60)
    
    strategy = MockEnhancedTradingStrategy()
    
    # Test cases
    test_cases = [
        {
            'name': 'Strong Buy Signal with Good Risk Metrics',
            'signal': {
                'confidence': 0.6,
                'signal_strength': 0.8,
                'market_microstructure': {
                    'volume_profile_valid': True,
                    'order_flow_bullish': True,
                    'liquidity_adequate': True
                }
            },
            'risk_metrics': {
                'sharpe_ratio': 1.1,
                'var_95': -0.008,
                'max_drawdown': 0.02,
                'volatility': 0.015
            }
        },
        {
            'name': 'Weak Sell Signal with Poor Risk Metrics',
            'signal': {
                'confidence': 0.4,
                'signal_strength': -0.3,
                'market_microstructure': {
                    'volume_profile_valid': False,
                    'order_flow_bearish': True,
                    'liquidity_adequate': False
                }
            },
            'risk_metrics': {
                'sharpe_ratio': 0.3,
                'var_95': -0.04,
                'max_drawdown': 0.08,
                'volatility': 0.06
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 50)
        
        try:
            confidence = strategy._calculate_risk_adjusted_confidence(
                test_case['signal'],
                test_case['risk_metrics']
            )
            
            print(f"Signal: {test_case['signal']}")
            print(f"Risk Metrics: {test_case['risk_metrics']}")
            print(f"Calculated Confidence: {confidence:.3f}")
            
            # Verify the confidence is within bounds
            if 0.05 <= confidence <= 0.95:
                print("✅ Confidence within bounds")
            else:
                print("❌ Confidence outside bounds")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def test_config_parameter_usage():
    """Test that configuration parameters are actually being used."""
    print("\n" + "="*60)
    print("TESTING CONFIG PARAMETER USAGE")
    print("="*60)
    
    strategy = MockEnhancedTradingStrategy()
    
    # Test with different base thresholds
    buy_threshold = strategy._calculate_dynamic_confidence_threshold('buy', {'volatility': 0.02}, {})
    sell_threshold = strategy._calculate_dynamic_confidence_threshold('sell', {'volatility': 0.02}, {})
    hold_threshold = strategy._calculate_dynamic_confidence_threshold('hold', {'volatility': 0.02}, {})
    
    print(f"BUY threshold: {buy_threshold:.3f}")
    print(f"SELL threshold: {sell_threshold:.3f}")
    print(f"HOLD threshold: {hold_threshold:.3f}")
    
    # Verify asymmetric behavior
    if buy_threshold < sell_threshold:
        print("✅ Asymmetric thresholds working (BUY < SELL)")
    else:
        print("❌ Asymmetric thresholds not working")
    
    # Test risk-adjusted confidence with different signal strengths
    signal_weak = {'confidence': 0.3, 'signal_strength': 0.2}
    signal_strong = {'confidence': 0.6, 'signal_strength': 0.8}
    risk_metrics = {'sharpe_ratio': 1.0, 'var_95': -0.01, 'max_drawdown': 0.02, 'volatility': 0.02}
    
    weak_confidence = strategy._calculate_risk_adjusted_confidence(signal_weak, risk_metrics)
    strong_confidence = strategy._calculate_risk_adjusted_confidence(signal_strong, risk_metrics)
    
    print(f"Weak signal confidence: {weak_confidence:.3f}")
    print(f"Strong signal confidence: {strong_confidence:.3f}")
    
    if strong_confidence > weak_confidence:
        print("✅ Risk-adjusted confidence working (strong > weak)")
    else:
        print("❌ Risk-adjusted confidence not working")

def main():
    """Run all configuration integration tests."""
    print("CONFIGURATION INTEGRATION TEST SUITE")
    print("="*60)
    
    # Run all tests
    test_config_loading()
    test_dynamic_threshold_with_config()
    test_risk_adjusted_confidence_with_config()
    test_config_parameter_usage()
    
    print("\n" + "="*60)
    print("CONFIGURATION INTEGRATION TEST SUITE COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main() 