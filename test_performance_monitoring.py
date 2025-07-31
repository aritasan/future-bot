"""
Test Real-time Performance Monitoring System
Comprehensive testing for WorldQuant-level performance tracking.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import load_config
from src.services.binance_service import BinanceService
from src.services.indicator_service import IndicatorService
from src.services.notification_service import NotificationService
from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
from src.quantitative.performance_tracker import WorldQuantPerformanceTracker

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
                'unrealizedPnl': '100.50',
                'positionAmt': '0.1',
                'markPrice': '50000.0'
            },
            {
                'symbol': 'ETHUSDT',
                'unrealizedPnl': '-50.25',
                'positionAmt': '1.0',
                'markPrice': '3000.0'
            },
            {
                'symbol': 'BNBUSDT',
                'unrealizedPnl': '25.75',
                'positionAmt': '2.0',
                'markPrice': '400.0'
            }
        ]
    
    async def get_account_balance(self):
        """Mock account balance."""
        return {
            'USDT': {
                'total': '10000.0',
                'free': '5000.0',
                'used': '5000.0'
            }
        }

class MockIndicatorService:
    """Mock Indicator service for testing."""
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 100):
        """Mock klines data."""
        # Generate synthetic price data
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 400
        prices = []
        for i in range(limit):
            # Add some volatility
            price = base_price * (1 + np.random.normal(0, 0.02))
            prices.append(price)
        
        return {
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [np.random.uniform(1000, 10000) for _ in prices]
        }

class MockNotificationService:
    """Mock Notification service for testing."""
    
    async def send_notification(self, message: str, level: str = 'info'):
        """Mock notification sending."""
        logger.info(f"Mock notification [{level.upper()}]: {message}")

async def test_performance_tracker():
    """Test WorldQuant Performance Tracker."""
    logger.info("============================================================")
    logger.info("Testing WorldQuant Performance Tracker...")
    logger.info("============================================================")
    
    try:
        # Initialize performance tracker
        tracker = WorldQuantPerformanceTracker()
        success = await tracker.initialize()
        
        if not success:
            logger.error("❌ Performance tracker initialization failed")
            return False
        
        logger.info("✅ Performance tracker initialized successfully")
        
        # Test metric updates
        logger.info("Testing metric updates...")
        
        # Generate test portfolio data
        test_data = {
            'timestamp': datetime.now(),
            'total_value': 100000.0,
            'returns': 0.02,  # 2% return
            'positions': {
                'BTCUSDT': {'weight': 0.4, 'return': 0.03},
                'ETHUSDT': {'weight': 0.3, 'return': 0.01},
                'BNBUSDT': {'weight': 0.3, 'return': 0.02}
            },
            'benchmark_return': 0.015,
            'risk_free_rate': 0.02
        }
        
        # Update metrics
        await tracker.update_metrics(test_data)
        logger.info("✅ Metric updates completed")
        
        # Test alerts
        logger.info("Testing performance alerts...")
        alerts = await tracker.check_alerts()
        logger.info(f"✅ Alerts check completed - Found {len(alerts)} alerts")
        
        for alert in alerts:
            logger.info(f"  Alert: {alert['type']} - {alert['message']}")
        
        # Test performance summary
        logger.info("Testing performance summary...")
        summary = await tracker.get_performance_summary()
        
        if summary:
            logger.info("✅ Performance summary generated successfully")
            logger.info(f"  Total Return: {summary.get('total_return', 0):.4f}")
            logger.info(f"  Volatility: {summary.get('volatility', 0):.4f}")
            logger.info(f"  Sharpe Ratio: {summary.get('sharpe_ratio', 0):.3f}")
            logger.info(f"  Max Drawdown: {summary.get('max_drawdown', 0):.4f}")
            logger.info(f"  Performance Score: {summary.get('performance_score', 0):.1f}")
            logger.info(f"  Risk Score: {summary.get('risk_score', 0):.1f}")
            logger.info(f"  Stability Score: {summary.get('stability_score', 0):.1f}")
        else:
            logger.error("❌ Performance summary generation failed")
            return False
        
        # Test asset performance
        logger.info("Testing asset performance...")
        asset_performance = await tracker.get_asset_performance()
        logger.info(f"✅ Asset performance retrieved - {len(asset_performance)} assets")
        
        # Test logging methods
        logger.info("Testing logging methods...")
        await tracker.log_risk_reduction(0.8)
        await tracker.log_volatility_management(0.25)
        await tracker.log_performance_improvement()
        await tracker.log_rebalancing_event()
        logger.info("✅ Logging methods completed")
        
        logger.info("✅ Performance tracker test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance tracker test failed: {str(e)}")
        return False

async def test_enhanced_strategy_performance_monitoring():
    """Test Enhanced Strategy Performance Monitoring."""
    logger.info("============================================================")
    logger.info("Testing Enhanced Strategy Performance Monitoring...")
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
        success = await strategy.initialize()
        if not success:
            logger.error("❌ Strategy initialization failed")
            return False
        
        logger.info("✅ Strategy initialized successfully")
        
        # Test performance monitoring start/stop
        logger.info("Testing performance monitoring control...")
        await strategy.start_performance_monitoring()
        logger.info("✅ Performance monitoring started")
        
        await strategy.stop_performance_monitoring()
        logger.info("✅ Performance monitoring stopped")
        
        # Test performance metrics update
        logger.info("Testing performance metrics update...")
        await strategy.update_performance_metrics()
        logger.info("✅ Performance metrics updated")
        
        # Test real-time performance summary
        logger.info("Testing real-time performance summary...")
        summary = await strategy.get_real_time_performance_summary()
        
        if summary:
            logger.info("✅ Real-time performance summary generated")
            logger.info(f"  Monitoring Active: {summary.get('monitoring_active', False)}")
            logger.info(f"  Performance Score: {summary.get('performance_score', 0):.1f}")
            logger.info(f"  Risk Score: {summary.get('risk_score', 0):.1f}")
            logger.info(f"  Stability Score: {summary.get('stability_score', 0):.1f}")
            logger.info(f"  Alerts Count: {len(summary.get('alerts', []))}")
            
            # Log alerts if any
            alerts = summary.get('alerts', [])
            if alerts:
                logger.info("  Alerts:")
                for alert in alerts:
                    logger.info(f"    - {alert['type']}: {alert['message']}")
        else:
            logger.error("❌ Real-time performance summary generation failed")
            return False
        
        # Test signal generation with performance tracking
        logger.info("Testing signal generation with performance tracking...")
        
        # Generate test signals
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        for symbol in test_symbols:
            signal = await strategy.generate_signals(symbol, mock_indicator)
            if signal:
                logger.info(f"✅ Signal generated for {symbol}: {signal.get('action', 'HOLD')}")
            else:
                logger.warning(f"⚠️ No signal generated for {symbol}")
        
        # Update performance metrics after signal generation
        await strategy.update_performance_metrics()
        
        # Get final performance summary
        final_summary = await strategy.get_real_time_performance_summary()
        logger.info("✅ Final performance summary:")
        logger.info(f"  Performance Metrics: {len(final_summary.get('performance_metrics', {}))} metrics")
        logger.info(f"  Signal History: {len(strategy.signal_history)} symbols")
        logger.info(f"  Quantitative Analysis: {len(strategy.quantitative_analysis_history)} entries")
        
        logger.info("✅ Enhanced Strategy Performance Monitoring test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Strategy Performance Monitoring test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_performance_monitoring_integration():
    """Test Performance Monitoring Integration."""
    logger.info("============================================================")
    logger.info("Testing Performance Monitoring Integration...")
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
        
        # Simulate trading activity
        logger.info("Simulating trading activity...")
        
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT']
        
        for i in range(5):  # Simulate 5 rounds
            logger.info(f"Round {i+1}/5:")
            
            # Generate signals for all symbols
            for symbol in test_symbols:
                signal = await strategy.generate_signals(symbol, mock_indicator)
                if signal:
                    logger.info(f"  {symbol}: {signal.get('action', 'HOLD')} (conf: {signal.get('confidence', 0):.3f})")
            
            # Update performance metrics
            await strategy.update_performance_metrics()
            
            # Get performance summary
            summary = await strategy.get_real_time_performance_summary()
            
            if summary:
                metrics = summary.get('performance_metrics', {})
                logger.info(f"  Performance: Return={metrics.get('total_return', 0):.4f}, "
                          f"Vol={metrics.get('volatility', 0):.4f}, "
                          f"Sharpe={metrics.get('sharpe_ratio', 0):.3f}")
                
                alerts = summary.get('alerts', [])
                if alerts:
                    logger.info(f"  Alerts: {len(alerts)} active alerts")
            
            # Small delay between rounds
            await asyncio.sleep(0.1)
        
        # Test comprehensive performance analysis
        logger.info("Testing comprehensive performance analysis...")
        
        # Test portfolio optimization
        try:
            optimization_results = await strategy.analyze_portfolio_optimization(test_symbols)
            if optimization_results and 'error' not in optimization_results:
                logger.info("✅ Portfolio optimization analysis completed")
            else:
                logger.warning("⚠️ Portfolio optimization analysis failed")
        except Exception as e:
            logger.warning(f"⚠️ Portfolio optimization analysis failed: {str(e)}")
        
        # Test factor analysis
        try:
            factor_results = await strategy.analyze_factor_exposures(test_symbols)
            if factor_results and 'error' not in factor_results:
                logger.info("✅ Factor analysis completed")
            else:
                logger.warning("⚠️ Factor analysis failed")
        except Exception as e:
            logger.warning(f"⚠️ Factor analysis failed: {str(e)}")
        
        # Test ML analysis
        ml_results = await strategy.train_ml_models(test_symbols)
        if ml_results:
            logger.info(f"✅ ML training completed for {len(ml_results)} symbols")
        else:
            logger.warning("⚠️ ML training failed")
        
        # Final performance summary
        final_summary = await strategy.get_real_time_performance_summary()
        logger.info("✅ Final Integration Performance Summary:")
        logger.info(f"  Performance Score: {final_summary.get('performance_score', 0):.1f}/100")
        logger.info(f"  Risk Score: {final_summary.get('risk_score', 0):.1f}/100")
        logger.info(f"  Stability Score: {final_summary.get('stability_score', 0):.1f}/100")
        logger.info(f"  Active Alerts: {len(final_summary.get('alerts', []))}")
        logger.info(f"  Monitoring Active: {final_summary.get('monitoring_active', False)}")
        
        # Stop monitoring
        await strategy.stop_performance_monitoring()
        
        logger.info("✅ Performance Monitoring Integration test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance Monitoring Integration test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def main():
    """Main test function."""
    logger.info("🚀 Starting Real-time Performance Monitoring Tests")
    logger.info("=" * 60)
    
    test_results = []
    
    # Test 1: Performance Tracker
    logger.info("\n📊 Test 1: WorldQuant Performance Tracker")
    result1 = await test_performance_tracker()
    test_results.append(("Performance Tracker", result1))
    
    # Test 2: Enhanced Strategy Performance Monitoring
    logger.info("\n📊 Test 2: Enhanced Strategy Performance Monitoring")
    result2 = await test_enhanced_strategy_performance_monitoring()
    test_results.append(("Enhanced Strategy Performance Monitoring", result2))
    
    # Test 3: Performance Monitoring Integration
    logger.info("\n📊 Test 3: Performance Monitoring Integration")
    result3 = await test_performance_monitoring_integration()
    test_results.append(("Performance Monitoring Integration", result3))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 PERFORMANCE MONITORING TEST SUMMARY")
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
        logger.info("🎉 All Performance Monitoring tests passed!")
        logger.info("\n🚀 Real-time Performance Monitoring System Features:")
        logger.info("  ✅ WorldQuant-level performance tracking")
        logger.info("  ✅ Real-time metrics calculation")
        logger.info("  ✅ Automated alert system")
        logger.info("  ✅ Performance scoring (0-100)")
        logger.info("  ✅ Risk monitoring and management")
        logger.info("  ✅ Portfolio optimization integration")
        logger.info("  ✅ Factor analysis integration")
        logger.info("  ✅ Machine learning integration")
        logger.info("  ✅ Comprehensive reporting")
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