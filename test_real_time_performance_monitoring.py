"""
Test Real-Time Performance Monitoring System
Comprehensive test for WorldQuant real-time performance monitoring.
"""

import asyncio
import logging
import json
import time
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from quantitative.real_time_performance_monitor import WorldQuantRealTimePerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealTimePerformanceTest:
    """Test class for real-time performance monitoring."""
    
    def __init__(self):
        """Initialize the test."""
        self.monitor = WorldQuantRealTimePerformanceMonitor()
        self.test_duration = 60  # 60 seconds test
        self.test_results = {
            'performance_metrics': [],
            'system_metrics': [],
            'alerts': [],
            'websocket_connections': 0
        }
    
    async def run_comprehensive_test(self):
        """Run comprehensive real-time performance monitoring test."""
        try:
            logger.info("🚀 Starting Real-Time Performance Monitoring Test")
            logger.info("=" * 60)
            
            # Initialize monitor
            logger.info("📊 Initializing WorldQuant Real-Time Performance Monitor...")
            success = await self.monitor.initialize()
            if not success:
                logger.error("❌ Failed to initialize performance monitor")
                return False
            
            logger.info("✅ Performance monitor initialized successfully")
            
            # Start monitoring
            logger.info("🔄 Starting real-time monitoring...")
            await self.monitor._start_real_time_monitoring()
            
            # Run test for specified duration
            logger.info(f"⏱️  Running test for {self.test_duration} seconds...")
            start_time = datetime.now()
            
            while (datetime.now() - start_time).seconds < self.test_duration:
                try:
                    # Get real-time summary
                    summary = await self.monitor.get_real_time_summary()
                    
                    # Log performance metrics
                    await self._log_performance_metrics(summary)
                    
                    # Check for alerts
                    alerts = await self.monitor.check_alerts()
                    if alerts:
                        await self._log_alerts(alerts)
                    
                    # Store test results
                    self._store_test_results(summary, alerts)
                    
                    # Wait for next update
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    logger.error(f"Error during test: {str(e)}")
                    await asyncio.sleep(5)
            
            # Stop monitoring
            logger.info("🛑 Stopping real-time monitoring...")
            await self.monitor.stop_monitoring()
            
            # Generate test report
            await self._generate_test_report()
            
            logger.info("✅ Real-Time Performance Monitoring Test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test failed: {str(e)}")
            return False
    
    async def _log_performance_metrics(self, summary: dict):
        """Log performance metrics."""
        try:
            performance_metrics = summary.get('performance_metrics', {})
            system_metrics = summary.get('system_metrics', {})
            
            logger.info("📈 Performance Metrics:")
            logger.info(f"  - Performance Score: {summary.get('performance_score', 0):.2f}")
            logger.info(f"  - Risk Score: {summary.get('risk_score', 0):.2f}")
            logger.info(f"  - Stability Score: {summary.get('stability_score', 0):.2f}")
            logger.info(f"  - Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.4f}")
            logger.info(f"  - Volatility: {performance_metrics.get('volatility', 0):.4f}")
            logger.info(f"  - Max Drawdown: {performance_metrics.get('max_drawdown', 0):.4f}")
            
            logger.info("💻 System Metrics:")
            logger.info(f"  - CPU Usage: {system_metrics.get('cpu_usage', 0):.1f}%")
            logger.info(f"  - Memory Usage: {system_metrics.get('memory_usage', 0):.1f}%")
            logger.info(f"  - API Response Time: {system_metrics.get('api_response_time', 0):.3f}s")
            logger.info(f"  - Error Rate: {system_metrics.get('error_rate', 0):.1f}%")
            logger.info(f"  - Cache Hit Rate: {system_metrics.get('cache_hit_rate', 0):.1f}%")
            
        except Exception as e:
            logger.error(f"Error logging performance metrics: {str(e)}")
    
    async def _log_alerts(self, alerts: list):
        """Log performance alerts."""
        try:
            logger.warning("🚨 Performance Alerts:")
            for alert in alerts:
                level_emoji = {
                    'critical': '🔴',
                    'warning': '🟡',
                    'info': '🔵'
                }.get(alert.get('level', 'info'), '⚪')
                
                logger.warning(f"  {level_emoji} {alert.get('message', 'Unknown alert')}")
                logger.warning(f"     Level: {alert.get('level', 'unknown')}")
                logger.warning(f"     Current Value: {alert.get('current_value', 0):.4f}")
                logger.warning(f"     Threshold: {alert.get('threshold', 0):.4f}")
                
        except Exception as e:
            logger.error(f"Error logging alerts: {str(e)}")
    
    def _store_test_results(self, summary: dict, alerts: list):
        """Store test results for reporting."""
        try:
            self.test_results['performance_metrics'].append(summary.get('performance_metrics', {}))
            self.test_results['system_metrics'].append(summary.get('system_metrics', {}))
            self.test_results['alerts'].extend(alerts)
            
        except Exception as e:
            logger.error(f"Error storing test results: {str(e)}")
    
    async def _generate_test_report(self):
        """Generate comprehensive test report."""
        try:
            logger.info("📊 Generating Test Report")
            logger.info("=" * 60)
            
            # Calculate statistics
            performance_scores = [m.get('performance_score', 0) for m in self.test_results['performance_metrics']]
            risk_scores = [m.get('risk_score', 0) for m in self.test_results['performance_metrics']]
            stability_scores = [m.get('stability_score', 0) for m in self.test_results['performance_metrics']]
            
            cpu_usage = [m.get('cpu_usage', 0) for m in self.test_results['system_metrics']]
            memory_usage = [m.get('memory_usage', 0) for m in self.test_results['system_metrics']]
            api_response_times = [m.get('api_response_time', 0) for m in self.test_results['system_metrics']]
            
            # Performance Statistics
            logger.info("📈 Performance Statistics:")
            if performance_scores:
                logger.info(f"  - Avg Performance Score: {np.mean(performance_scores):.2f}")
                logger.info(f"  - Min Performance Score: {np.min(performance_scores):.2f}")
                logger.info(f"  - Max Performance Score: {np.max(performance_scores):.2f}")
                logger.info(f"  - Std Performance Score: {np.std(performance_scores):.2f}")
            
            if risk_scores:
                logger.info(f"  - Avg Risk Score: {np.mean(risk_scores):.2f}")
                logger.info(f"  - Min Risk Score: {np.min(risk_scores):.2f}")
                logger.info(f"  - Max Risk Score: {np.max(risk_scores):.2f}")
            
            if stability_scores:
                logger.info(f"  - Avg Stability Score: {np.mean(stability_scores):.2f}")
                logger.info(f"  - Min Stability Score: {np.min(stability_scores):.2f}")
                logger.info(f"  - Max Stability Score: {np.max(stability_scores):.2f}")
            
            # System Statistics
            logger.info("💻 System Statistics:")
            if cpu_usage:
                logger.info(f"  - Avg CPU Usage: {np.mean(cpu_usage):.1f}%")
                logger.info(f"  - Max CPU Usage: {np.max(cpu_usage):.1f}%")
                logger.info(f"  - CPU Usage > 80%: {sum(1 for x in cpu_usage if x > 80)} times")
            
            if memory_usage:
                logger.info(f"  - Avg Memory Usage: {np.mean(memory_usage):.1f}%")
                logger.info(f"  - Max Memory Usage: {np.max(memory_usage):.1f}%")
                logger.info(f"  - Memory Usage > 85%: {sum(1 for x in memory_usage if x > 85)} times")
            
            if api_response_times:
                logger.info(f"  - Avg API Response Time: {np.mean(api_response_times):.3f}s")
                logger.info(f"  - Max API Response Time: {np.max(api_response_times):.3f}s")
                logger.info(f"  - API Response Time > 2s: {sum(1 for x in api_response_times if x > 2)} times")
            
            # Alert Statistics
            logger.info("🚨 Alert Statistics:")
            total_alerts = len(self.test_results['alerts'])
            critical_alerts = len([a for a in self.test_results['alerts'] if a.get('level') == 'critical'])
            warning_alerts = len([a for a in self.test_results['alerts'] if a.get('level') == 'warning'])
            info_alerts = len([a for a in self.test_results['alerts'] if a.get('level') == 'info'])
            
            logger.info(f"  - Total Alerts: {total_alerts}")
            logger.info(f"  - Critical Alerts: {critical_alerts}")
            logger.info(f"  - Warning Alerts: {warning_alerts}")
            logger.info(f"  - Info Alerts: {info_alerts}")
            
            # Alert breakdown by type
            alert_types = {}
            for alert in self.test_results['alerts']:
                alert_type = alert.get('type', 'unknown')
                alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
            
            logger.info("  - Alert Breakdown by Type:")
            for alert_type, count in alert_types.items():
                logger.info(f"    * {alert_type}: {count}")
            
            # Performance Assessment
            logger.info("🎯 Performance Assessment:")
            
            # Performance Score Assessment
            avg_performance = np.mean(performance_scores) if performance_scores else 0
            if avg_performance >= 70:
                logger.info("  ✅ Performance Score: EXCELLENT (>70)")
            elif avg_performance >= 50:
                logger.info("  🟡 Performance Score: GOOD (50-70)")
            else:
                logger.info("  🔴 Performance Score: NEEDS IMPROVEMENT (<50)")
            
            # Risk Score Assessment
            avg_risk = np.mean(risk_scores) if risk_scores else 0
            if avg_risk <= 30:
                logger.info("  ✅ Risk Score: LOW (<30)")
            elif avg_risk <= 50:
                logger.info("  🟡 Risk Score: MODERATE (30-50)")
            else:
                logger.info("  🔴 Risk Score: HIGH (>50)")
            
            # System Health Assessment
            avg_cpu = np.mean(cpu_usage) if cpu_usage else 0
            avg_memory = np.mean(memory_usage) if memory_usage else 0
            avg_api_time = np.mean(api_response_times) if api_response_times else 0
            
            system_health_score = 0
            if avg_cpu < 80:
                system_health_score += 25
            if avg_memory < 85:
                system_health_score += 25
            if avg_api_time < 2:
                system_health_score += 25
            if total_alerts < 10:
                system_health_score += 25
            
            if system_health_score >= 80:
                logger.info("  ✅ System Health: EXCELLENT")
            elif system_health_score >= 60:
                logger.info("  🟡 System Health: GOOD")
            else:
                logger.info("  🔴 System Health: NEEDS ATTENTION")
            
            # Recommendations
            logger.info("💡 Recommendations:")
            
            if avg_performance < 50:
                logger.info("  - Focus on improving performance metrics")
                logger.info("  - Review trading strategy parameters")
                logger.info("  - Consider portfolio rebalancing")
            
            if avg_risk > 50:
                logger.info("  - Implement additional risk controls")
                logger.info("  - Review position sizing")
                logger.info("  - Consider hedging strategies")
            
            if avg_cpu > 80 or avg_memory > 85:
                logger.info("  - Optimize system resource usage")
                logger.info("  - Consider hardware upgrades")
                logger.info("  - Review application efficiency")
            
            if total_alerts > 20:
                logger.info("  - Review alert thresholds")
                logger.info("  - Investigate root causes of alerts")
                logger.info("  - Implement preventive measures")
            
            logger.info("=" * 60)
            logger.info("📋 Test Report Generated Successfully")
            
        except Exception as e:
            logger.error(f"Error generating test report: {str(e)}")
    
    async def test_websocket_integration(self):
        """Test WebSocket integration."""
        try:
            logger.info("🔌 Testing WebSocket Integration...")
            
            # Start monitor
            await self.monitor.initialize()
            
            # Simulate WebSocket client
            import websockets
            
            async def test_websocket_client():
                try:
                    async with websockets.connect("ws://localhost:8765") as websocket:
                        logger.info("✅ WebSocket client connected successfully")
                        
                        # Receive initial data
                        initial_message = await websocket.recv()
                        initial_data = json.loads(initial_message)
                        logger.info(f"📊 Received initial data: {len(str(initial_data))} chars")
                        
                        # Receive real-time updates
                        message_count = 0
                        start_time = time.time()
                        
                        while time.time() - start_time < 30:  # 30 seconds test
                            try:
                                message = await asyncio.wait_for(websocket.recv(), timeout=10)
                                data = json.loads(message)
                                message_count += 1
                                
                                if message_count % 5 == 0:  # Log every 5th message
                                    logger.info(f"📡 Received message #{message_count}: {len(str(data))} chars")
                                
                            except asyncio.TimeoutError:
                                logger.warning("⚠️  WebSocket timeout - no message received")
                                break
                        
                        logger.info(f"📈 WebSocket test completed: {message_count} messages received")
                        
                except Exception as e:
                    logger.error(f"❌ WebSocket test failed: {str(e)}")
            
            # Run WebSocket test
            await test_websocket_client()
            
        except Exception as e:
            logger.error(f"Error in WebSocket integration test: {str(e)}")
    
    async def test_alert_system(self):
        """Test the alert system."""
        try:
            logger.info("🚨 Testing Alert System...")
            
            # Start monitor
            await self.monitor.initialize()
            
            # Simulate different scenarios
            test_scenarios = [
                {
                    'name': 'High CPU Usage',
                    'data': {'cpu_usage': 85.0, 'memory_usage': 70.0, 'api_response_time': 1.5, 'error_rate': 2.0}
                },
                {
                    'name': 'High Memory Usage',
                    'data': {'cpu_usage': 60.0, 'memory_usage': 90.0, 'api_response_time': 1.0, 'error_rate': 1.0}
                },
                {
                    'name': 'Slow API Response',
                    'data': {'cpu_usage': 50.0, 'memory_usage': 60.0, 'api_response_time': 3.0, 'error_rate': 1.0}
                },
                {
                    'name': 'High Error Rate',
                    'data': {'cpu_usage': 40.0, 'memory_usage': 50.0, 'api_response_time': 0.5, 'error_rate': 8.0}
                }
            ]
            
            for scenario in test_scenarios:
                logger.info(f"🧪 Testing scenario: {scenario['name']}")
                
                # Update system metrics with test data
                for metric, value in scenario['data'].items():
                    if metric in self.monitor.system_metrics:
                        self.monitor.system_metrics[metric].append(value)
                
                # Check for alerts
                alerts = await self.monitor.check_alerts()
                
                if alerts:
                    logger.info(f"  🚨 Alerts triggered: {len(alerts)}")
                    for alert in alerts:
                        logger.info(f"    - {alert.get('message', 'Unknown alert')}")
                else:
                    logger.info("  ✅ No alerts triggered")
                
                # Wait before next scenario
                await asyncio.sleep(2)
            
            logger.info("✅ Alert system test completed")
            
        except Exception as e:
            logger.error(f"Error in alert system test: {str(e)}")

async def main():
    """Main test function."""
    try:
        logger.info("🎯 WorldQuant Real-Time Performance Monitoring Test Suite")
        logger.info("=" * 80)
        
        # Create test instance
        test = RealTimePerformanceTest()
        
        # Run comprehensive test
        logger.info("📊 Running Comprehensive Real-Time Performance Test...")
        success = await test.run_comprehensive_test()
        
        if success:
            logger.info("✅ All tests completed successfully")
        else:
            logger.error("❌ Some tests failed")
        
        # Run additional tests
        logger.info("\n🔌 Testing WebSocket Integration...")
        await test.test_websocket_integration()
        
        logger.info("\n🚨 Testing Alert System...")
        await test.test_alert_system()
        
        logger.info("\n🎉 Test Suite Completed!")
        
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")

if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main()) 