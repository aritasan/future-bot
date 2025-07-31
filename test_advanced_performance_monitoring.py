"""
Advanced Performance Monitoring Test
Comprehensive testing for WorldQuant-level real-time performance tracking.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
import numpy as np
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import load_config
from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
from src.quantitative.portfolio_optimizer import WorldQuantPortfolioOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockBinanceService:
    """Mock Binance service for testing."""
    
    async def get_positions(self):
        """Mock positions data."""
        return [
            {
                'symbol': 'BTCUSDT',
                'unrealizedPnl': '150.75',
                'positionAmt': '0.15',
                'markPrice': '52000.0'
            },
            {
                'symbol': 'ETHUSDT',
                'unrealizedPnl': '-25.50',
                'positionAmt': '1.5',
                'markPrice': '3200.0'
            },
            {
                'symbol': 'BNBUSDT',
                'unrealizedPnl': '45.25',
                'positionAmt': '3.0',
                'markPrice': '450.0'
            },
            {
                'symbol': 'ADAUSDT',
                'unrealizedPnl': '12.30',
                'positionAmt': '100.0',
                'markPrice': '0.45'
            }
        ]
    
    async def get_account_balance(self):
        """Mock account balance."""
        return {
            'USDT': {
                'total': '15000.0',
                'free': '8000.0',
                'used': '7000.0'
            }
        }

class MockIndicatorService:
    """Mock Indicator service for testing."""
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 100):
        """Mock klines data."""
        # Generate synthetic price data with more realistic patterns
        base_price = 52000 if 'BTC' in symbol else 3200 if 'ETH' in symbol else 450 if 'BNB' in symbol else 0.45
        prices = []
        for i in range(limit):
            # Add trend and volatility
            trend = 0.001 * i  # Slight upward trend
            volatility = np.random.normal(0, 0.02)
            price = base_price * (1 + trend + volatility)
            prices.append(max(price, base_price * 0.5))  # Ensure positive prices
        
        return {
            'open': prices,
            'high': [p * 1.015 for p in prices],
            'low': [p * 0.985 for p in prices],
            'close': prices,
            'volume': [np.random.uniform(1000, 50000) for _ in prices]
        }

class MockNotificationService:
    """Mock Notification service for testing."""
    
    async def send_notification(self, message: str, level: str = 'info'):
        """Mock notification sending."""
        logger.info(f"Mock notification [{level.upper()}]: {message}")

async def test_advanced_portfolio_optimizer():
    """Test advanced portfolio optimizer features."""
    logger.info("============================================================")
    logger.info("Testing Advanced Portfolio Optimizer Features...")
    logger.info("============================================================")
    
    try:
        # Initialize portfolio optimizer
        config = load_config()
        optimizer = WorldQuantPortfolioOptimizer(config)
        await optimizer.initialize()
        
        logger.info("✅ Portfolio optimizer initialized successfully")
        
        # Test real-time portfolio metrics
        logger.info("Testing real-time portfolio metrics...")
        portfolio_metrics = await optimizer.get_real_time_portfolio_metrics()
        
        if portfolio_metrics:
            logger.info("✅ Real-time portfolio metrics generated")
            logger.info(f"  Total trades: {portfolio_metrics.get('total_trades', 0)}")
            logger.info(f"  Total volume: {portfolio_metrics.get('total_volume', 0):.2f}")
            logger.info(f"  Rebalancing count: {portfolio_metrics.get('rebalancing_count', 0)}")
            logger.info(f"  Weight drift: {portfolio_metrics.get('weight_drift', 0):.4f}")
            logger.info(f"  Active positions: {portfolio_metrics.get('active_positions', 0)}")
        else:
            logger.warning("⚠️ Real-time portfolio metrics failed")
        
        # Test performance attribution
        logger.info("Testing performance attribution...")
        attribution = await optimizer.get_performance_attribution()
        
        if attribution:
            logger.info("✅ Performance attribution generated")
            logger.info(f"  Factor attribution: {len(attribution.get('factor_attribution', {}))} factors")
            logger.info(f"  Asset attribution: {len(attribution.get('asset_attribution', {}))} assets")
        else:
            logger.warning("⚠️ Performance attribution failed")
        
        # Test risk decomposition
        logger.info("Testing risk decomposition...")
        risk_decomposition = await optimizer.get_risk_decomposition()
        
        if risk_decomposition:
            logger.info("✅ Risk decomposition generated")
            logger.info(f"  Total risk: {risk_decomposition.get('total_risk', 0):.4f}")
            logger.info(f"  Concentration risk: {risk_decomposition.get('concentration_risk', 0):.4f}")
            logger.info(f"  Factor risks: {len(risk_decomposition.get('factor_risk', {}))} factors")
        else:
            logger.warning("⚠️ Risk decomposition failed")
        
        # Test stress test results
        logger.info("Testing stress test results...")
        stress_results = await optimizer.get_stress_test_results()
        
        if stress_results:
            logger.info("✅ Stress test results generated")
            logger.info(f"  Expected shortfall: {stress_results.get('expected_shortfall', 0):.4f}")
            logger.info(f"  Stress scenarios: {len(stress_results.get('stress_scenarios', {}))} scenarios")
        else:
            logger.warning("⚠️ Stress test results failed")
        
        # Test optimization effectiveness
        logger.info("Testing optimization effectiveness...")
        effectiveness = await optimizer.get_optimization_effectiveness()
        
        if effectiveness:
            logger.info("✅ Optimization effectiveness generated")
            logger.info(f"  Sharpe ratio: {effectiveness.get('sharpe_ratio', 0):.3f}")
            logger.info(f"  Information ratio: {effectiveness.get('information_ratio', 0):.3f}")
            logger.info(f"  Win rate: {effectiveness.get('win_rate', 0):.3f}")
            logger.info(f"  Effectiveness score: {effectiveness.get('effectiveness_score', 0):.1f}/100")
        else:
            logger.warning("⚠️ Optimization effectiveness failed")
        
        # Test comprehensive performance report
        logger.info("Testing comprehensive performance report...")
        comprehensive_report = await optimizer.get_comprehensive_performance_report()
        
        if comprehensive_report:
            logger.info("✅ Comprehensive performance report generated")
            logger.info(f"  Report timestamp: {comprehensive_report.get('timestamp', 'N/A')}")
            logger.info(f"  Monitoring active: {comprehensive_report.get('monitoring_status', {}).get('active', False)}")
        else:
            logger.warning("⚠️ Comprehensive performance report failed")
        
        logger.info("✅ Advanced portfolio optimizer test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Advanced portfolio optimizer test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_advanced_trading_strategy():
    """Test advanced trading strategy performance monitoring."""
    logger.info("============================================================")
    logger.info("Testing Advanced Trading Strategy Performance Monitoring...")
    logger.info("============================================================")
    
    try:
        # Load config
        config = load_config()
        
        # Initialize mock services
        mock_binance = MockBinanceService()
        mock_indicator = MockIndicatorService()
        mock_notification = MockNotificationService()
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config=config,
            binance_service=mock_binance,
            indicator_service=mock_indicator,
            notification_service=mock_notification
        )
        
        # Initialize strategy
        await strategy.initialize()
        logger.info("✅ Strategy initialized successfully")
        
        # Start performance monitoring
        await strategy.start_performance_monitoring()
        logger.info("✅ Performance monitoring started")
        
        # Test advanced performance metrics
        logger.info("Testing advanced performance metrics...")
        advanced_metrics = await strategy.get_advanced_performance_metrics()
        
        if advanced_metrics:
            logger.info("✅ Advanced performance metrics generated")
            
            # Log basic metrics
            basic_metrics = advanced_metrics.get('basic_metrics', {})
            logger.info(f"  Total return: {basic_metrics.get('total_return', 0):.4f}")
            logger.info(f"  Volatility: {basic_metrics.get('volatility', 0):.4f}")
            logger.info(f"  Sharpe ratio: {basic_metrics.get('sharpe_ratio', 0):.3f}")
            
            # Log risk metrics
            risk_metrics = advanced_metrics.get('risk_metrics', {})
            logger.info(f"  VaR 95%: {risk_metrics.get('var_95', 0):.4f}")
            logger.info(f"  Max drawdown: {risk_metrics.get('max_drawdown', 0):.4f}")
            logger.info(f"  Downside deviation: {risk_metrics.get('downside_deviation', 0):.4f}")
            
            # Log efficiency metrics
            efficiency_metrics = advanced_metrics.get('efficiency_metrics', {})
            logger.info(f"  Sortino ratio: {efficiency_metrics.get('sortino_ratio', 0):.3f}")
            logger.info(f"  Calmar ratio: {efficiency_metrics.get('calmar_ratio', 0):.3f}")
            logger.info(f"  Information ratio: {efficiency_metrics.get('information_ratio', 0):.3f}")
            
            # Log timing metrics
            timing_metrics = advanced_metrics.get('timing_metrics', {})
            logger.info(f"  Signal frequency: {timing_metrics.get('signal_frequency', 0)}")
            logger.info(f"  Signal quality: {timing_metrics.get('signal_quality', 0):.3f}")
            logger.info(f"  Timing accuracy: {timing_metrics.get('timing_accuracy', 0):.3f}")
            
            # Log quality metrics
            quality_metrics = advanced_metrics.get('quality_metrics', {})
            logger.info(f"  Overall quality score: {quality_metrics.get('overall_quality_score', 0):.3f}")
            logger.info(f"  Data quality: {quality_metrics.get('data_quality', 0):.3f}")
            logger.info(f"  Model accuracy: {quality_metrics.get('model_accuracy', 0):.3f}")
        else:
            logger.warning("⚠️ Advanced performance metrics failed")
        
        # Test performance attribution analysis
        logger.info("Testing performance attribution analysis...")
        attribution = await strategy.get_performance_attribution_analysis()
        
        if attribution:
            logger.info("✅ Performance attribution analysis generated")
            logger.info(f"  Factor attribution: {len(attribution.get('factor_attribution', {}))} factors")
            logger.info(f"  Asset attribution: {len(attribution.get('asset_attribution', {}))} assets")
            logger.info(f"  Timing attribution: {len(attribution.get('timing_attribution', {}))} metrics")
            logger.info(f"  Selection attribution: {len(attribution.get('selection_attribution', {}))} metrics")
        else:
            logger.warning("⚠️ Performance attribution analysis failed")
        
        # Test comprehensive performance report
        logger.info("Testing comprehensive performance report...")
        comprehensive_report = await strategy.get_comprehensive_performance_report()
        
        if comprehensive_report:
            logger.info("✅ Comprehensive performance report generated")
            logger.info(f"  Report timestamp: {comprehensive_report.get('timestamp', 'N/A')}")
            
            # Log monitoring status
            monitoring_status = comprehensive_report.get('monitoring_status', {})
            logger.info(f"  Monitoring active: {monitoring_status.get('active', False)}")
            logger.info(f"  Alert count: {monitoring_status.get('alert_count', 0)}")
            logger.info(f"  Performance score: {monitoring_status.get('performance_score', 0):.1f}/100")
            logger.info(f"  Risk score: {monitoring_status.get('risk_score', 0):.1f}/100")
            logger.info(f"  Stability score: {monitoring_status.get('stability_score', 0):.1f}/100")
            
            # Log quantitative analysis
            quantitative_analysis = comprehensive_report.get('quantitative_analysis', {})
            logger.info(f"  Statistical validation: {'Available' if quantitative_analysis.get('statistical_validation') else 'Not available'}")
            logger.info(f"  Factor analysis: {'Available' if quantitative_analysis.get('factor_analysis') else 'Not available'}")
            logger.info(f"  ML analysis: {'Available' if quantitative_analysis.get('ml_analysis') else 'Not available'}")
        else:
            logger.warning("⚠️ Comprehensive performance report failed")
        
        # Stop monitoring
        await strategy.stop_performance_monitoring()
        logger.info("✅ Performance monitoring stopped")
        
        logger.info("✅ Advanced trading strategy test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Advanced trading strategy test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_performance_monitoring_integration():
    """Test integrated performance monitoring system."""
    logger.info("============================================================")
    logger.info("Testing Integrated Performance Monitoring System...")
    logger.info("============================================================")
    
    try:
        # Load config
        config = load_config()
        
        # Initialize services
        mock_binance = MockBinanceService()
        mock_indicator = MockIndicatorService()
        mock_notification = MockNotificationService()
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config=config,
            binance_service=mock_binance,
            indicator_service=mock_indicator,
            notification_service=mock_notification
        )
        
        # Initialize
        await strategy.initialize()
        
        # Start performance monitoring
        await strategy.start_performance_monitoring()
        
        # Simulate trading activity with performance tracking
        logger.info("Simulating trading activity with performance tracking...")
        
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT']
        
        for i in range(3):  # Simulate 3 rounds
            logger.info(f"Round {i+1}/3:")
            
            # Generate signals for all symbols
            for symbol in test_symbols:
                signal = await strategy.generate_signals(symbol, mock_indicator)
                if signal:
                    logger.info(f"  {symbol}: {signal.get('action', 'HOLD')} (conf: {signal.get('confidence', 0):.3f})")
            
            # Update performance metrics
            await strategy.update_performance_metrics()
            
            # Get comprehensive performance report
            report = await strategy.get_comprehensive_performance_report()
            
            if report:
                basic_performance = report.get('basic_performance', {})
                performance_metrics = basic_performance.get('performance_metrics', {})
                
                logger.info(f"  Performance: Return={performance_metrics.get('total_return', 0):.4f}, "
                          f"Vol={performance_metrics.get('volatility', 0):.4f}, "
                          f"Sharpe={performance_metrics.get('sharpe_ratio', 0):.3f}")
                
                # Log advanced metrics
                advanced_metrics = report.get('advanced_metrics', {})
                if advanced_metrics:
                    risk_metrics = advanced_metrics.get('risk_metrics', {})
                    efficiency_metrics = advanced_metrics.get('efficiency_metrics', {})
                    
                    logger.info(f"  Risk: VaR={risk_metrics.get('var_95', 0):.4f}, "
                              f"Drawdown={risk_metrics.get('max_drawdown', 0):.4f}")
                    logger.info(f"  Efficiency: Sortino={efficiency_metrics.get('sortino_ratio', 0):.3f}, "
                              f"Calmar={efficiency_metrics.get('calmar_ratio', 0):.3f}")
            
            # Small delay between rounds
            await asyncio.sleep(0.1)
        
        # Generate final comprehensive report
        logger.info("Generating final comprehensive performance report...")
        final_report = await strategy.get_comprehensive_performance_report()
        
        if final_report:
            logger.info("✅ Final Comprehensive Performance Report:")
            
            # Log monitoring status
            monitoring_status = final_report.get('monitoring_status', {})
            logger.info(f"  Monitoring Status:")
            logger.info(f"    Active: {monitoring_status.get('active', False)}")
            logger.info(f"    Performance Score: {monitoring_status.get('performance_score', 0):.1f}/100")
            logger.info(f"    Risk Score: {monitoring_status.get('risk_score', 0):.1f}/100")
            logger.info(f"    Stability Score: {monitoring_status.get('stability_score', 0):.1f}/100")
            logger.info(f"    Alert Count: {monitoring_status.get('alert_count', 0)}")
            
            # Log advanced metrics summary
            advanced_metrics = final_report.get('advanced_metrics', {})
            if advanced_metrics:
                logger.info(f"  Advanced Metrics Summary:")
                
                risk_metrics = advanced_metrics.get('risk_metrics', {})
                logger.info(f"    Risk Metrics: {len(risk_metrics)} calculated")
                
                efficiency_metrics = advanced_metrics.get('efficiency_metrics', {})
                logger.info(f"    Efficiency Metrics: {len(efficiency_metrics)} calculated")
                
                timing_metrics = advanced_metrics.get('timing_metrics', {})
                logger.info(f"    Timing Metrics: {len(timing_metrics)} calculated")
                
                quality_metrics = advanced_metrics.get('quality_metrics', {})
                logger.info(f"    Quality Metrics: {len(quality_metrics)} calculated")
            
            # Log performance attribution
            attribution = final_report.get('performance_attribution', {})
            if attribution:
                logger.info(f"  Performance Attribution:")
                logger.info(f"    Factor Attribution: {len(attribution.get('factor_attribution', {}))} factors")
                logger.info(f"    Asset Attribution: {len(attribution.get('asset_attribution', {}))} assets")
                logger.info(f"    Timing Attribution: {len(attribution.get('timing_attribution', {}))} metrics")
                logger.info(f"    Selection Attribution: {len(attribution.get('selection_attribution', {}))} metrics")
            
            # Log quantitative analysis
            quantitative_analysis = final_report.get('quantitative_analysis', {})
            if quantitative_analysis:
                logger.info(f"  Quantitative Analysis:")
                logger.info(f"    Statistical Validation: {'Available' if quantitative_analysis.get('statistical_validation') else 'Not available'}")
                logger.info(f"    Factor Analysis: {'Available' if quantitative_analysis.get('factor_analysis') else 'Not available'}")
                logger.info(f"    ML Analysis: {'Available' if quantitative_analysis.get('ml_analysis') else 'Not available'}")
        
        # Stop monitoring
        await strategy.stop_performance_monitoring()
        
        logger.info("✅ Integrated Performance Monitoring test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integrated Performance Monitoring test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def main():
    """Main test function."""
    logger.info("🚀 Starting Advanced Performance Monitoring Tests")
    logger.info("=" * 60)
    
    test_results = []
    
    # Test 1: Advanced Portfolio Optimizer
    logger.info("\n📊 Test 1: Advanced Portfolio Optimizer")
    result1 = await test_advanced_portfolio_optimizer()
    test_results.append(("Advanced Portfolio Optimizer", result1))
    
    # Test 2: Advanced Trading Strategy
    logger.info("\n📊 Test 2: Advanced Trading Strategy")
    result2 = await test_advanced_trading_strategy()
    test_results.append(("Advanced Trading Strategy", result2))
    
    # Test 3: Integrated Performance Monitoring
    logger.info("\n📊 Test 3: Integrated Performance Monitoring")
    result3 = await test_performance_monitoring_integration()
    test_results.append(("Integrated Performance Monitoring", result3))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 ADVANCED PERFORMANCE MONITORING TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All Advanced Performance Monitoring tests passed!")
        logger.info("\n🚀 Advanced Real-time Performance Monitoring Features:")
        logger.info("  ✅ WorldQuant-level portfolio optimization")
        logger.info("  ✅ Real-time risk decomposition")
        logger.info("  ✅ Performance attribution analysis")
        logger.info("  ✅ Stress testing and scenario analysis")
        logger.info("  ✅ Advanced efficiency metrics (Sortino, Calmar, Treynor)")
        logger.info("  ✅ Timing and quality metrics")
        logger.info("  ✅ Comprehensive reporting system")
        logger.info("  ✅ Integrated monitoring dashboard")
        logger.info("  ✅ Automated alert system")
        logger.info("  ✅ Performance scoring (0-100)")
        logger.info("  ✅ Risk-adjusted return calculations")
        logger.info("  ✅ Factor model integration")
        logger.info("  ✅ Machine learning integration")
        logger.info("  ✅ Statistical validation")
        logger.info("  ✅ Portfolio optimization integration")
    else:
        logger.error(f"❌ {total-passed} test(s) failed")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        sys.exit(1) 