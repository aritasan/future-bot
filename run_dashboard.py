#!/usr/bin/env python3
"""
Script để chạy Performance Dashboard riêng biệt
"""
import asyncio
import sys
import logging
from pathlib import Path

# Thêm thư mục src vào Python path
sys.path.append(str(Path(__file__).parent / "src"))

from performance_dashboard_enhanced import EnhancedPerformanceDashboard

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dashboard.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def run_dashboard():
    """Chạy Performance Dashboard"""
    try:
        logger.info("🚀 Starting Performance Dashboard...")
        
        dashboard = EnhancedPerformanceDashboard()
        logger.info("✅ Performance Dashboard initialized")
        
        logger.info("📊 Dashboard features:")
        logger.info("   • Real-time Financial Metrics")
        logger.info("   • System Health Monitoring")
        logger.info("   • Performance Charts")
        logger.info("   • Alert Notifications")
        logger.info("   • Risk Analysis")
        
        logger.info("🌐 Dashboard will be available at: http://localhost:8050")
        logger.info("📡 WebSocket connection: ws://localhost:8765")
        
        # Chạy dashboard
        await dashboard.run(host='localhost', port=8050, debug=False)
        
    except Exception as e:
        logger.error(f"❌ Error starting Performance Dashboard: {e}")

def main():
    """Main function để chạy dashboard"""
    print("📊 Performance Dashboard")
    print("=" * 40)
    
    print("\n🎯 Starting dashboard in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        import time
        time.sleep(1)
    
    # Chạy dashboard
    try:
        asyncio.run(run_dashboard())
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main() 