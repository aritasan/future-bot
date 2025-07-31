#!/usr/bin/env python3
"""
Script để chạy Trading Bot với Real-time Performance Monitoring
"""
import asyncio
import os
import sys
import logging
from pathlib import Path

# Thêm thư mục src vào Python path
sys.path.append(str(Path(__file__).parent / "src"))

from main_with_quantitative import main
from performance_dashboard_enhanced import EnhancedPerformanceDashboard
from src.quantitative.real_time_performance_monitor import WorldQuantRealTimePerformanceMonitor

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def run_performance_dashboard():
    """Chạy Performance Dashboard"""
    try:
        dashboard = EnhancedPerformanceDashboard()
        logger.info("🚀 Starting Performance Dashboard...")
        
        # Chạy dashboard trong thread riêng vì Dash không async
        import threading
        def run_dashboard():
            dashboard.run(host='localhost', port=8050, debug=False)
        
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        
        logger.info("✅ Performance Dashboard started successfully")
        
        # Giữ task này chạy
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"❌ Error starting Performance Dashboard: {e}")

async def run_real_time_monitor():
    """Chạy Real-time Performance Monitor"""
    try:
        monitor = WorldQuantRealTimePerformanceMonitor()
        logger.info("📊 Starting Real-time Performance Monitor...")
        await monitor.initialize()
        logger.info("✅ Real-time Performance Monitor started successfully")
        
        # Giữ monitor chạy
        while True:
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"❌ Error starting Real-time Performance Monitor: {e}")

async def run_bot_with_monitoring():
    """Chạy bot với performance monitoring"""
    try:
        logger.info("🤖 Starting Trading Bot with Quantitative Analysis...")
        
        # Tạo tasks cho bot và monitoring
        tasks = [
            asyncio.create_task(main()),  # Bot chính
            asyncio.create_task(run_real_time_monitor()),  # Real-time monitor
            asyncio.create_task(run_performance_dashboard()),  # Performance dashboard
        ]
        
        # Chạy tất cả tasks
        await asyncio.gather(*tasks)
        
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Error running bot: {e}")

def check_environment():
    """Kiểm tra môi trường trước khi chạy"""
    logger.info("🔍 Checking environment...")
    
    # Kiểm tra file .env
    if not os.path.exists('.env'):
        logger.warning("⚠️  File .env not found. Please create it with your API keys.")
        logger.info("📝 See SETUP_GUIDE.md for configuration details.")
        return False
    
    # Kiểm tra thư mục logs
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Kiểm tra dependencies
    try:
        import pandas
        import numpy
        import asyncio
        import websockets
        import psutil
        logger.info("✅ All required dependencies are installed")
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("💡 Run: pip install -r requirements.txt")
        return False
    
    return True

def main_sync():
    """Main function để chạy bot"""
    print("🚀 Trading Bot with Real-time Performance Monitoring")
    print("=" * 60)
    
    # Kiểm tra môi trường
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        return
    
    print("\n✅ Environment check passed!")
    print("\n📊 Available monitoring features:")
    print("   • Real-time Performance Dashboard (http://localhost:8050)")
    print("   • WebSocket Performance Data (ws://localhost:8765)")
    print("   • Financial Metrics Tracking")
    print("   • System Health Monitoring")
    print("   • Risk Management Alerts")
    
    print("\n🎯 Starting bot in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        import time
        time.sleep(1)
    
    # Chạy bot
    try:
        asyncio.run(run_bot_with_monitoring())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main_sync() 