import asyncio
import sys
import os
import time
import logging
import numpy as np
import pandas as pd

# Set event loop policy for Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.quantitative.statistical_validator import StatisticalSignalValidator
from src.quantitative.risk_manager import VaRCalculator, DynamicPositionSizer, RiskManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_quantitative_components():
    """Test all quantitative trading components."""
    try:
        print("🔬 Testing Quantitative Trading Components")
        print("=" * 60)
        
        # Test 1: Statistical Signal Validator
        print("\n📊 Test 1: Statistical Signal Validator")
        validator = StatisticalSignalValidator()
        
        # Create sample signals
        sample_signals = [
            {'strength': 0.8, 'confidence': 0.9, 'type': 'trend_following'},
            {'strength': 0.6, 'confidence': 0.7, 'type': 'breakout'},
            {'strength': 0.4, 'confidence': 0.5, 'type': 'mean_reversion'},
            {'strength': 0.9, 'confidence': 0.95, 'type': 'strong_trend'}
        ]
        
        for i, signal in enumerate(sample_signals):
            print(f"  Testing signal {i+1}: {signal['type']}")
            validation = validator.validate_signal(signal)
            
            print(f"    Valid: {validation['is_valid']}")
            print(f"    P-value: {validation['p_value']:.4f}")
            print(f"    T-statistic: {validation['t_statistic']:.4f}")
            print(f"    Sharpe Ratio: {validation['sharpe_ratio']:.4f}")
            print(f"    Information Ratio: {validation['information_ratio']:.4f}")
            print(f"    Max Drawdown: {validation['max_drawdown']:.4f}")
            print()
        
        # Test ensemble validation
        ensemble_validation = validator.validate_signal_ensemble(sample_signals)
        print(f"Ensemble Validation: {ensemble_validation}")
        
        # Get validation summary
        validation_summary = validator.get_validation_summary()
        print(f"Validation Summary: {validation_summary}")
        
        # Test 2: VaR Calculator
        print("\n📊 Test 2: VaR Calculator")
        var_calculator = VaRCalculator()
        
        # Generate sample returns
        np.random.seed(42)
        sample_returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
        position_size = 10000  # $10,000 position
        
        var_results = var_calculator.calculate_var(sample_returns, position_size, 'all')
        print(f"Historical VaR: ${var_results['historical_var']:.2f}")
        print(f"Parametric VaR: ${var_results['parametric_var']:.2f}")
        print(f"Monte Carlo VaR: ${var_results['monte_carlo_var']:.2f}")
        print(f"Expected Shortfall: ${var_results['expected_shortfall']:.2f}")
        
        # Get VaR summary
        var_summary = var_calculator.get_var_summary()
        print(f"VaR Summary: {var_summary}")
        
        # Test 3: Dynamic Position Sizer
        print("\n📊 Test 3: Dynamic Position Sizer")
        position_sizer = DynamicPositionSizer()
        
        # Test different scenarios
        scenarios = [
            {'signal_strength': 0.8, 'volatility': 0.02, 'correlation': 0.1, 'var_limit': 500},
            {'signal_strength': 0.6, 'volatility': 0.03, 'correlation': 0.5, 'var_limit': 300},
            {'signal_strength': 0.9, 'volatility': 0.015, 'correlation': -0.2, 'var_limit': 800},
            {'signal_strength': 0.4, 'volatility': 0.04, 'correlation': 0.8, 'var_limit': 200}
        ]
        
        for i, scenario in enumerate(scenarios):
            print(f"  Testing scenario {i+1}:")
            print(f"    Signal Strength: {scenario['signal_strength']}")
            print(f"    Volatility: {scenario['volatility']}")
            print(f"    Correlation: {scenario['correlation']}")
            print(f"    VaR Limit: ${scenario['var_limit']}")
            
            position_results = position_sizer.calculate_position_size(
                scenario['signal_strength'],
                scenario['volatility'],
                scenario['correlation'],
                scenario['var_limit']
            )
            
            print(f"    Kelly Position: {position_results['kelly_position']:.4f}")
            print(f"    Volatility Adjusted: {position_results['volatility_adjusted']:.4f}")
            print(f"    Correlation Adjusted: {position_results['correlation_adjusted']:.4f}")
            print(f"    VaR Adjusted: {position_results['var_adjusted']:.4f}")
            print(f"    Risk Parity: {position_results['risk_parity']:.4f}")
            print(f"    Final Position: {position_results['final_position']:.4f}")
            print()
        
        # Get position summary
        position_summary = position_sizer.get_position_summary()
        print(f"Position Summary: {position_summary}")
        
        # Test 4: Risk Manager (Combined)
        print("\n📊 Test 4: Risk Manager (Combined)")
        risk_manager = RiskManager()
        
        # Test comprehensive risk calculation
        signal_data = {
            'strength': 0.75,
            'confidence': 0.8,
            'volatility': 0.025,
            'correlation': 0.3,
            'win_rate': 0.65,
            'avg_win': 0.03,
            'avg_loss': 0.02
        }
        
        risk_metrics = risk_manager.calculate_risk_metrics(
            sample_returns, signal_data, position_size
        )
        
        print("Risk Metrics:")
        print(f"  VaR Metrics: {risk_metrics['var_metrics']}")
        print(f"  Position Metrics: {risk_metrics['position_metrics']}")
        print(f"  Overall Risk: {risk_metrics['overall_risk']}")
        
        # Get risk summary
        risk_summary = risk_manager.get_risk_summary()
        print(f"Risk Summary: {risk_summary}")
        
        # Test 5: Performance Comparison
        print("\n📊 Test 5: Performance Comparison")
        
        # Compare different signal qualities
        signal_qualities = [
            {'name': 'High Quality', 'strength': 0.9, 'confidence': 0.95},
            {'name': 'Medium Quality', 'strength': 0.6, 'confidence': 0.7},
            {'name': 'Low Quality', 'strength': 0.3, 'confidence': 0.4}
        ]
        
        performance_results = []
        
        for quality in signal_qualities:
            print(f"  Testing {quality['name']} signal:")
            
            # Validate signal
            validation = validator.validate_signal(quality)
            
            # Calculate position size
            position_results = position_sizer.calculate_position_size(
                quality['strength'], 0.025, 0.2, 500
            )
            
            # Calculate risk metrics
            risk_metrics = risk_manager.calculate_risk_metrics(
                sample_returns, quality, position_size
            )
            
            performance_results.append({
                'quality': quality['name'],
                'validation': validation,
                'position': position_results,
                'risk': risk_metrics
            })
            
            print(f"    Valid: {validation['is_valid']}")
            print(f"    Sharpe: {validation['sharpe_ratio']:.4f}")
            print(f"    Position Size: {position_results['final_position']:.4f}")
            print(f"    VaR: ${risk_metrics['var_metrics']['historical_var']:.2f}")
            print()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 QUANTITATIVE COMPONENTS TEST SUMMARY")
        print("=" * 60)
        
        # Validation summary
        print(f"✅ Statistical Validation:")
        print(f"   - Total Validations: {validation_summary['total_validations']}")
        print(f"   - Valid Signals: {validation_summary['valid_signals']}")
        print(f"   - Validation Rate: {validation_summary['validation_rate']:.2%}")
        print(f"   - Avg Sharpe Ratio: {validation_summary['avg_sharpe_ratio']:.4f}")
        
        # VaR summary
        print(f"\n✅ Risk Management:")
        print(f"   - Total VaR Calculations: {var_summary['total_calculations']}")
        print(f"   - Avg Historical VaR: ${var_summary['avg_historical_var']:.2f}")
        print(f"   - Max VaR: ${var_summary['max_var']:.2f}")
        
        # Position sizing summary
        print(f"\n✅ Position Sizing:")
        print(f"   - Total Positions: {position_summary['total_positions']}")
        print(f"   - Avg Position Size: {position_summary['avg_position_size']:.4f}")
        print(f"   - Max Position Size: {position_summary['max_position_size']:.4f}")
        
        # Performance analysis
        print(f"\n✅ Performance Analysis:")
        for result in performance_results:
            print(f"   {result['quality']}:")
            print(f"     - Valid: {result['validation']['is_valid']}")
            print(f"     - Sharpe: {result['validation']['sharpe_ratio']:.4f}")
            print(f"     - Position: {result['position']['final_position']:.4f}")
            print(f"     - VaR: ${result['risk']['var_metrics']['historical_var']:.2f}")
        
        print("\n🎯 RECOMMENDATIONS:")
        
        # High quality signals
        high_quality_count = sum(1 for r in performance_results if r['validation']['is_valid'])
        if high_quality_count > 0:
            print(f"✅ {high_quality_count} high-quality signals identified")
            print("   - Consider increasing position sizes for these signals")
            print("   - Monitor for alpha decay over time")
        
        # Risk management
        avg_var = np.mean([r['risk']['var_metrics']['historical_var'] for r in performance_results])
        if avg_var > 1000:
            print("⚠️  High average VaR detected")
            print("   - Consider reducing position sizes")
            print("   - Implement stricter risk limits")
        else:
            print("✅ Risk levels are within acceptable range")
        
        # Position sizing
        avg_position = np.mean([r['position']['final_position'] for r in performance_results])
        if avg_position < 0.01:
            print("⚠️  Low position sizes detected")
            print("   - Consider relaxing position size constraints")
            print("   - Review signal quality thresholds")
        else:
            print("✅ Position sizes are appropriately sized")
        
        print("\n✅ Quantitative components test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in quantitative components test: {str(e)}")
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_quantitative_components()) 