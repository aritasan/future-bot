#!/usr/bin/env python3
"""
Test script for WorldQuant-level confidence thresholds implementation.
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
    """Mock class to test confidence threshold methods."""
    
    def __init__(self):
        self.logger = logger
    
    def _calculate_dynamic_confidence_threshold(self, action: str, market_data: Dict, risk_metrics: Dict = None) -> float:
        """
        Calculate dynamic confidence threshold based on action type, market conditions, and risk metrics.
        WorldQuant-level implementation with asymmetric thresholds for BUY/SELL.
        """
        try:
            # Base asymmetric thresholds
            base_thresholds = {
                'buy': 0.45,      # Lower for BUY (more opportunities)
                'sell': 0.65,     # Higher for SELL (riskier)
                'hold': 0.35      # Neutral zone
            }
            
            base_threshold = base_thresholds.get(action, 0.6)
            
            # Volatility adjustment
            volatility = market_data.get('volatility', 0.02)
            vol_adjustment = 0.0
            
            if volatility > 0.05:  # High volatility
                vol_adjustment = 0.1
            elif volatility < 0.01:  # Low volatility
                vol_adjustment = -0.05
            
            # Market regime adjustment
            market_regime = market_data.get('market_regime', 'normal')
            regime_adjustment = 0.0
            
            if market_regime == 'trending':
                regime_adjustment = -0.05  # Lower threshold in trending markets
            elif market_regime == 'mean_reverting':
                regime_adjustment = 0.05   # Higher threshold in mean-reverting markets
            elif market_regime == 'high_volatility':
                regime_adjustment = 0.08   # Much higher threshold in high volatility
            
            # Risk metrics adjustment
            risk_adjustment = 0.0
            if risk_metrics:
                sharpe_ratio = risk_metrics.get('sharpe_ratio', 0.0)
                var_95 = risk_metrics.get('var_95', -0.02)
                max_drawdown = risk_metrics.get('max_drawdown', 0.0)
                
                # Sharpe ratio adjustment
                if sharpe_ratio > 1.0:
                    risk_adjustment -= 0.05  # Lower threshold for good performance
                elif sharpe_ratio < 0.5:
                    risk_adjustment += 0.05  # Higher threshold for poor performance
                
                # VaR adjustment
                if var_95 < -0.03:  # High risk
                    risk_adjustment += 0.03
                elif var_95 > -0.01:  # Low risk
                    risk_adjustment -= 0.02
                
                # Drawdown adjustment
                if max_drawdown > 0.05:
                    risk_adjustment += 0.02
            
            # Calculate final threshold
            final_threshold = base_threshold + vol_adjustment + regime_adjustment + risk_adjustment
            
            # Ensure reasonable bounds
            final_threshold = max(0.25, min(0.85, final_threshold))
            
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
            base_confidence = signal.get('confidence', 0.0)
            signal_strength = signal.get('signal_strength', 0.0)
            
            # Base confidence boost from signal strength
            strength_boost = min(0.1, abs(signal_strength) * 0.2)
            
            # Risk metrics adjustment
            risk_boost = 0.0
            
            if risk_metrics:
                sharpe_ratio = risk_metrics.get('sharpe_ratio', 0.0)
                var_95 = risk_metrics.get('var_95', -0.02)
                max_drawdown = risk_metrics.get('max_drawdown', 0.0)
                volatility = risk_metrics.get('volatility', 0.02)
                
                # Sharpe ratio boost
                if sharpe_ratio > 1.0:
                    risk_boost += 0.05
                elif sharpe_ratio < 0.5:
                    risk_boost -= 0.05
                
                # VaR boost
                if var_95 > -0.01:  # Low risk
                    risk_boost += 0.03
                elif var_95 < -0.03:  # High risk
                    risk_boost -= 0.03
                
                # Drawdown penalty
                if max_drawdown > 0.05:
                    risk_boost -= 0.02
                
                # Volatility adjustment
                if volatility < 0.01:  # Low volatility
                    risk_boost += 0.02
                elif volatility > 0.05:  # High volatility
                    risk_boost -= 0.02
            
            # Market microstructure boost
            microstructure_boost = 0.0
            if 'market_microstructure' in signal:
                microstructure = signal['market_microstructure']
                
                # Volume profile boost
                if microstructure.get('volume_profile_valid', False):
                    microstructure_boost += 0.02
                
                # Order flow boost
                if microstructure.get('order_flow_bullish', False):
                    microstructure_boost += 0.03
                elif microstructure.get('order_flow_bearish', False):
                    microstructure_boost -= 0.03
                
                # Liquidity boost
                if microstructure.get('liquidity_adequate', False):
                    microstructure_boost += 0.01
            
            # Calculate final confidence
            final_confidence = base_confidence + strength_boost + risk_boost + microstructure_boost
            
            # Ensure bounds
            final_confidence = max(0.05, min(0.95, final_confidence))
            
            self.logger.info(f"Risk-adjusted confidence: base={base_confidence:.3f}, "
                           f"strength_boost={strength_boost:.3f}, risk_boost={risk_boost:.3f}, "
                           f"microstructure_boost={microstructure_boost:.3f}, "
                           f"final={final_confidence:.3f}")
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted confidence: {str(e)}")
            return signal.get('confidence', 0.0)

def test_dynamic_confidence_thresholds():
    """Test dynamic confidence threshold calculation."""
    print("\n" + "="*60)
    print("TESTING DYNAMIC CONFIDENCE THRESHOLDS")
    print("="*60)
    
    strategy = MockEnhancedTradingStrategy()
    
    # Test cases
    test_cases = [
        {
            'name': 'Low Volatility, Good Performance, Trending Market',
            'action': 'buy',
            'market_data': {'volatility': 0.008, 'market_regime': 'trending'},
            'risk_metrics': {'sharpe_ratio': 1.2, 'var_95': -0.008, 'max_drawdown': 0.02},
            'expected_range': (0.25, 0.35)
        },
        {
            'name': 'High Volatility, Poor Performance, High Volatility Regime',
            'action': 'sell',
            'market_data': {'volatility': 0.06, 'market_regime': 'high_volatility'},
            'risk_metrics': {'sharpe_ratio': 0.3, 'var_95': -0.04, 'max_drawdown': 0.08},
            'expected_range': (0.75, 0.85)
        },
        {
            'name': 'Normal Volatility, Mean Reverting Market',
            'action': 'buy',
            'market_data': {'volatility': 0.025, 'market_regime': 'mean_reverting'},
            'risk_metrics': {'sharpe_ratio': 0.8, 'var_95': -0.015, 'max_drawdown': 0.03},
            'expected_range': (0.40, 0.55)
        },
        {
            'name': 'Hold Action Test',
            'action': 'hold',
            'market_data': {'volatility': 0.02, 'market_regime': 'normal'},
            'risk_metrics': {'sharpe_ratio': 0.6, 'var_95': -0.02, 'max_drawdown': 0.04},
            'expected_range': (0.30, 0.40)
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 50)
        
        threshold = strategy._calculate_dynamic_confidence_threshold(
            test_case['action'],
            test_case['market_data'],
            test_case['risk_metrics']
        )
        
        expected_min, expected_max = test_case['expected_range']
        passed = expected_min <= threshold <= expected_max
        
        print(f"Action: {test_case['action']}")
        print(f"Market Data: {test_case['market_data']}")
        print(f"Risk Metrics: {test_case['risk_metrics']}")
        print(f"Calculated Threshold: {threshold:.3f}")
        print(f"Expected Range: {expected_min:.3f} - {expected_max:.3f}")
        print(f"Status: {'✅ PASS' if passed else '❌ FAIL'}")
        
        if not passed:
            print(f"❌ Threshold {threshold:.3f} outside expected range [{expected_min:.3f}, {expected_max:.3f}]")

def test_risk_adjusted_confidence():
    """Test risk-adjusted confidence calculation."""
    print("\n" + "="*60)
    print("TESTING RISK-ADJUSTED CONFIDENCE")
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
            },
            'expected_range': (0.70, 0.85)
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
            },
            'expected_range': (0.25, 0.45)
        },
        {
            'name': 'Neutral Signal with Average Risk Metrics',
            'signal': {
                'confidence': 0.5,
                'signal_strength': 0.1,
                'market_microstructure': {
                    'volume_profile_valid': True,
                    'order_flow_bullish': False,
                    'order_flow_bearish': False,
                    'liquidity_adequate': True
                }
            },
            'risk_metrics': {
                'sharpe_ratio': 0.7,
                'var_95': -0.02,
                'max_drawdown': 0.04,
                'volatility': 0.025
            },
            'expected_range': (0.50, 0.65)
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 50)
        
        confidence = strategy._calculate_risk_adjusted_confidence(
            test_case['signal'],
            test_case['risk_metrics']
        )
        
        expected_min, expected_max = test_case['expected_range']
        passed = expected_min <= confidence <= expected_max
        
        print(f"Signal: {test_case['signal']}")
        print(f"Risk Metrics: {test_case['risk_metrics']}")
        print(f"Calculated Confidence: {confidence:.3f}")
        print(f"Expected Range: {expected_min:.3f} - {expected_max:.3f}")
        print(f"Status: {'✅ PASS' if passed else '❌ FAIL'}")
        
        if not passed:
            print(f"❌ Confidence {confidence:.3f} outside expected range [{expected_min:.3f}, {expected_max:.3f}]")

def test_asymmetric_thresholds():
    """Test asymmetric threshold behavior."""
    print("\n" + "="*60)
    print("TESTING ASYMMETRIC THRESHOLDS")
    print("="*60)
    
    strategy = MockEnhancedTradingStrategy()
    
    # Same market conditions, different actions
    market_data = {'volatility': 0.025, 'market_regime': 'normal'}
    risk_metrics = {'sharpe_ratio': 0.8, 'var_95': -0.015, 'max_drawdown': 0.03}
    
    buy_threshold = strategy._calculate_dynamic_confidence_threshold('buy', market_data, risk_metrics)
    sell_threshold = strategy._calculate_dynamic_confidence_threshold('sell', market_data, risk_metrics)
    hold_threshold = strategy._calculate_dynamic_confidence_threshold('hold', market_data, risk_metrics)
    
    print(f"Market Data: {market_data}")
    print(f"Risk Metrics: {risk_metrics}")
    print(f"BUY Threshold: {buy_threshold:.3f}")
    print(f"SELL Threshold: {sell_threshold:.3f}")
    print(f"HOLD Threshold: {hold_threshold:.3f}")
    
    # Verify asymmetric behavior
    buy_lower_than_sell = buy_threshold < sell_threshold
    hold_lower_than_sell = hold_threshold < sell_threshold
    buy_higher_than_hold = buy_threshold > hold_threshold
    
    print(f"\nAsymmetric Behavior Verification:")
    print(f"BUY < SELL: {'✅ PASS' if buy_lower_than_sell else '❌ FAIL'}")
    print(f"HOLD < SELL: {'✅ PASS' if hold_lower_than_sell else '❌ FAIL'}")
    print(f"BUY > HOLD: {'✅ PASS' if buy_higher_than_hold else '❌ FAIL'}")
    
    # Calculate threshold differences
    sell_buy_diff = sell_threshold - buy_threshold
    sell_hold_diff = sell_threshold - hold_threshold
    
    print(f"\nThreshold Differences:")
    print(f"SELL - BUY: {sell_buy_diff:.3f}")
    print(f"SELL - HOLD: {sell_hold_diff:.3f}")
    
    # Expected: SELL threshold should be significantly higher than BUY
    expected_sell_buy_diff = 0.10  # Expected difference (adjusted for realistic values)
    passed = sell_buy_diff >= expected_sell_buy_diff
    
    print(f"Expected SELL-BUY difference >= {expected_sell_buy_diff:.3f}: {'✅ PASS' if passed else '❌ FAIL'}")

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("TESTING EDGE CASES")
    print("="*60)
    
    strategy = MockEnhancedTradingStrategy()
    
    # Test cases with extreme values
    edge_cases = [
        {
            'name': 'Extreme High Volatility',
            'market_data': {'volatility': 0.1, 'market_regime': 'high_volatility'},
            'risk_metrics': {'sharpe_ratio': 0.1, 'var_95': -0.08, 'max_drawdown': 0.15},
            'action': 'sell'
        },
        {
            'name': 'Extreme Low Volatility',
            'market_data': {'volatility': 0.005, 'market_regime': 'trending'},
            'risk_metrics': {'sharpe_ratio': 2.0, 'var_95': -0.005, 'max_drawdown': 0.01},
            'action': 'buy'
        },
        {
            'name': 'Missing Risk Metrics',
            'market_data': {'volatility': 0.02, 'market_regime': 'normal'},
            'risk_metrics': None,
            'action': 'buy'
        },
        {
            'name': 'Invalid Action',
            'market_data': {'volatility': 0.02, 'market_regime': 'normal'},
            'risk_metrics': {'sharpe_ratio': 0.8, 'var_95': -0.02, 'max_drawdown': 0.03},
            'action': 'invalid'
        }
    ]
    
    for i, test_case in enumerate(edge_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 50)
        
        try:
            threshold = strategy._calculate_dynamic_confidence_threshold(
                test_case['action'],
                test_case['market_data'],
                test_case['risk_metrics']
            )
            
            # Check bounds
            in_bounds = 0.25 <= threshold <= 0.85
            print(f"Action: {test_case['action']}")
            print(f"Market Data: {test_case['market_data']}")
            print(f"Risk Metrics: {test_case['risk_metrics']}")
            print(f"Calculated Threshold: {threshold:.3f}")
            print(f"Within Bounds [0.25, 0.85]: {'✅ PASS' if in_bounds else '❌ FAIL'}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def main():
    """Run all tests."""
    print("WORLDQUANT CONFIDENCE THRESHOLDS TEST SUITE")
    print("="*60)
    
    # Run all tests
    test_dynamic_confidence_thresholds()
    test_risk_adjusted_confidence()
    test_asymmetric_thresholds()
    test_edge_cases()
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main() 