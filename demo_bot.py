#!/usr/bin/env python3
"""
Demo script để test Trading Bot với Real-time Performance Monitoring
"""
import asyncio
import sys
import logging
import time
from pathlib import Path

# Thêm thư mục src vào Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.quantitative.real_time_performance_monitor import WorldQuantRealTimePerformanceMonitor
from performance_dashboard_enhanced import EnhancedPerformanceDashboard

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/demo.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def demo_performance_monitoring():
    """Demo performance monitoring system"""
    try:
        logger.info("🎬 Starting Performance Monitoring Demo...")
        
        # Khởi tạo performance monitor
        monitor = WorldQuantRealTimePerformanceMonitor()
        await monitor.initialize()
        logger.info("✅ Performance Monitor initialized")
        
        # Khởi tạo dashboard
        dashboard = EnhancedPerformanceDashboard()
        logger.info("✅ Dashboard initialized")
        
        # Chạy demo trong 60 giây
        logger.info("⏱️  Running demo for 60 seconds...")
        logger.info("📊 Dashboard available at: http://localhost:8050")
        logger.info("📡 WebSocket available at: ws://localhost:8765")
        
        start_time = time.time()
        while time.time() - start_time < 60:
            # Lấy performance summary
            summary = await monitor.get_real_time_summary()
            
            if summary:
                metrics = summary.get('performance_metrics', {})
                system_metrics = summary.get('system_metrics', {})
                
                logger.info(f"📈 Performance: Return={metrics.get('total_return', 0):.4f}, "
                           f"Sharpe={metrics.get('sharpe_ratio', 0):.3f}, "
                           f"Drawdown={metrics.get('max_drawdown', 0):.4f}")
                
                logger.info(f"🖥️  System: CPU={system_metrics.get('cpu_usage', 0):.1f}%, "
                           f"Memory={system_metrics.get('memory_usage', 0):.1f}%, "
                           f"API={system_metrics.get('api_response_time', 0):.0f}ms")
            
            # Kiểm tra alerts
            alerts = await monitor.check_alerts()
            if alerts:
                logger.info(f"⚠️  Alerts: {len(alerts)} active alerts")
                for alert in alerts:
                    logger.info(f"   - {alert['type']}: {alert['message']}")
            
            await asyncio.sleep(5)  # Update every 5 seconds
        
        logger.info("✅ Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")

async def demo_dashboard():
    """Demo dashboard functionality"""
    try:
        logger.info("🎨 Starting Dashboard Demo...")
        
        dashboard = EnhancedPerformanceDashboard()
        logger.info("✅ Dashboard initialized")
        
        # Chạy dashboard
        await dashboard.run(host='localhost', port=8050, debug=False)
        
    except Exception as e:
        logger.error(f"❌ Dashboard demo failed: {e}")

def main():
    """Main demo function"""
    print("🎬 Trading Bot Performance Monitoring Demo")
    print("=" * 50)
    
    print("\n🎯 Demo options:")
    print("1. Performance Monitoring Demo (60 seconds)")
    print("2. Dashboard Demo")
    print("3. Full Demo (Both)")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting Performance Monitoring Demo...")
        asyncio.run(demo_performance_monitoring())
    elif choice == "2":
        print("\n🎨 Starting Dashboard Demo...")
        asyncio.run(demo_dashboard())
    elif choice == "3":
        print("\n🎬 Starting Full Demo...")
        # Chạy cả hai
        async def run_full_demo():
            await asyncio.gather(
                demo_performance_monitoring(),
                demo_dashboard()
            )
        asyncio.run(run_full_demo())
    else:
        print("❌ Invalid choice. Please run again.")

if __name__ == "__main__":
    main() 