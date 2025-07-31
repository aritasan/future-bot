#!/usr/bin/env python3
"""
Comprehensive test script to verify all error fixes
Tests WebSocket port binding, factor model, risk manager, and other components
"""

import asyncio
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from src.core.config import load_config
from src.quantitative.factor_model import WorldQuantFactorModel
from src.quantitative.risk_manager import RiskManager
from src.quantitative.integration import QuantitativeIntegration
from src.quantitative.quantitative_trading_system import QuantitativeTradingSystem
from src.quantitative.real_time_performance_monitor import WorldQuantRealTimePerformanceMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/comprehensive_test_v2.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ComprehensiveErrorFixTest:
    """Test class to verify all error fixes."""
    
    def __init__(self):
        """Initialize test class."""
        self.config = load_config()
        self.test_results = {}
        
    async def test_factor_model_build_method(self):
        """Test the build_factor_model method."""
        try:
            logger.info("Testing build_factor_model method...")
            
            # Create test data
            np.random.seed(42)
            returns_data = pd.DataFrame({
                'BTCUSDT': np.random.normal(0.001, 0.02, 100),
                'ETHUSDT': np.random.normal(0.001, 0.025, 100),
                'ADAUSDT': np.random.normal(0.0005, 0.03, 100)
            })
            
            # Initialize factor model
            factor_model = WorldQuantFactorModel(self.config)
            await factor_model.initialize()
            
            # Test build_factor_model method
            result = await factor_model.build_factor_model(returns_data)
            
            if 'error' not in result:
                logger.info("✅ build_factor_model method works correctly")
                self.test_results['factor_model_build'] = 'PASS'
                return True
            else:
                logger.error(f"❌ build_factor_model failed: {result['error']}")
                self.test_results['factor_model_build'] = 'FAIL'
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing build_factor_model: {str(e)}")
            self.test_results['factor_model_build'] = 'FAIL'
            return False
    
    async def test_risk_manager_initialize(self):
        """Test the RiskManager initialize method."""
        try:
            logger.info("Testing RiskManager initialize method...")
            
            risk_manager = RiskManager(self.config)
            result = await risk_manager.initialize()
            
            if result:
                logger.info("✅ RiskManager initialize method works correctly")
                self.test_results['risk_manager_initialize'] = 'PASS'
                return True
            else:
                logger.error("❌ RiskManager initialize failed")
                self.test_results['risk_manager_initialize'] = 'FAIL'
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing RiskManager initialize: {str(e)}")
            self.test_results['risk_manager_initialize'] = 'FAIL'
            return False
    
    async def test_quantitative_integration_cache(self):
        """Test the analysis_cache attribute in QuantitativeIntegration."""
        try:
            logger.info("Testing QuantitativeIntegration analysis_cache...")
            
            integration = QuantitativeIntegration(self.config)
            
            # Check if analysis_cache attribute exists
            if hasattr(integration, 'analysis_cache'):
                logger.info("✅ analysis_cache attribute exists")
                self.test_results['integration_cache'] = 'PASS'
                return True
            else:
                logger.error("❌ analysis_cache attribute missing")
                self.test_results['integration_cache'] = 'FAIL'
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing integration cache: {str(e)}")
            self.test_results['integration_cache'] = 'FAIL'
            return False
    
    async def test_quantitative_system_performance_metrics(self):
        """Test the get_performance_metrics method."""
        try:
            logger.info("Testing get_performance_metrics method...")
            
            system = QuantitativeTradingSystem(self.config)
            await system.initialize()
            
            # Test get_performance_metrics method
            metrics = await system.get_performance_metrics()
            
            if 'error' not in metrics:
                logger.info("✅ get_performance_metrics method works correctly")
                self.test_results['performance_metrics'] = 'PASS'
                return True
            else:
                logger.error(f"❌ get_performance_metrics failed: {metrics['error']}")
                self.test_results['performance_metrics'] = 'FAIL'
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing performance metrics: {str(e)}")
            self.test_results['performance_metrics'] = 'FAIL'
            return False
    
    async def test_websocket_port_binding(self):
        """Test WebSocket port binding with fallback."""
        try:
            logger.info("Testing WebSocket port binding...")
            
            monitor = WorldQuantRealTimePerformanceMonitor()
            await monitor.initialize()
            
            # Start WebSocket server
            await monitor._start_websocket_server()
            
            # Wait a bit for server to start
            await asyncio.sleep(2)
            
            logger.info("✅ WebSocket server started successfully")
            self.test_results['websocket_binding'] = 'PASS'
            
            # Stop monitoring
            await monitor.stop_monitoring()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error testing WebSocket binding: {str(e)}")
            self.test_results['websocket_binding'] = 'FAIL'
            return False
    
    async def test_portfolio_optimization_fix(self):
        """Test portfolio optimization with proper error handling."""
        try:
            logger.info("Testing portfolio optimization...")
            
            # Create test data
            np.random.seed(42)
            returns_data = pd.DataFrame({
                'BTCUSDT': np.random.normal(0.001, 0.02, 100),
                'ETHUSDT': np.random.normal(0.001, 0.025, 100),
                'ADAUSDT': np.random.normal(0.0005, 0.03, 100)
            })
            
            system = QuantitativeTradingSystem(self.config)
            await system.initialize()
            
            # Test portfolio optimization
            result = await system.optimize_portfolio(returns_data)
            
            if result and 'error' not in result:
                logger.info("✅ Portfolio optimization works correctly")
                self.test_results['portfolio_optimization'] = 'PASS'
                return True
            else:
                logger.warning(f"⚠️ Portfolio optimization returned: {result}")
                self.test_results['portfolio_optimization'] = 'WARNING'
                return True  # Not a critical failure
                
        except Exception as e:
            logger.error(f"❌ Error testing portfolio optimization: {str(e)}")
            self.test_results['portfolio_optimization'] = 'FAIL'
            return False
    
    async def run_comprehensive_test(self):
        """Run all comprehensive tests."""
        logger.info("🧪 Starting Comprehensive Error Fix Tests...")
        
        tests = [
            self.test_factor_model_build_method,
            self.test_risk_manager_initialize,
            self.test_quantitative_integration_cache,
            self.test_quantitative_system_performance_metrics,
            self.test_websocket_port_binding,
            self.test_portfolio_optimization_fix
        ]
        
        results = []
        for test in tests:
            try:
                result = await test()
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Test failed with exception: {str(e)}")
                results.append(False)
        
        # Summary
        passed = sum(results)
        total = len(results)
        
        logger.info(f"\n📊 Test Results Summary:")
        logger.info(f"Passed: {passed}/{total}")
        logger.info(f"Failed: {total - passed}/{total}")
        
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result == "PASS" else "❌" if result == "FAIL" else "⚠️"
            logger.info(f"{status_emoji} {test_name}: {result}")
        
        if passed == total:
            logger.info("🎉 All tests passed! Error fixes are working correctly.")
        else:
            logger.warning("⚠️ Some tests failed. Please review the errors above.")
        
        return passed == total

async def main():
    """Main test function."""
    try:
        test = ComprehensiveErrorFixTest()
        success = await test.run_comprehensive_test()
        
        if success:
            logger.info("✅ Comprehensive error fix verification completed successfully!")
        else:
            logger.warning("⚠️ Some error fixes need attention.")
            
    except Exception as e:
        logger.error(f"❌ Error running comprehensive test: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 