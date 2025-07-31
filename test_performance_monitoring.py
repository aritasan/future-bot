#!/usr/bin/env python3
"""
Script để test Performance Monitoring System
"""
import asyncio
import sys
import logging
from pathlib import Path

# Thêm thư mục src vào Python path
sys.path.append(str(Path(__file__).parent / "src"))

from test_real_time_performance_monitoring import RealTimePerformanceTest

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/performance_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def run_performance_test():
    """Chạy test performance monitoring"""
    try:
        logger.info("🧪 Starting Performance Monitoring Test...")
        
        test = RealTimePerformanceTest()
        await test.run_comprehensive_test()
        
        logger.info("✅ Performance Monitoring Test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during performance test: {e}")

def main():
    """Main function để chạy test"""
    print("🧪 Performance Monitoring Test")
    print("=" * 50)
    
    print("\n📊 Testing components:")
    print("   • Real-time Performance Monitor")
    print("   • WebSocket Integration")
    print("   • Alert System")
    print("   • Financial Metrics")
    print("   • System Health Monitoring")
    
    print("\n🎯 Starting test in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        import time
        time.sleep(1)
    
    # Chạy test
    try:
        asyncio.run(run_performance_test())
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main() 