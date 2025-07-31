#!/usr/bin/env python3
"""
Run Complete Trading System
Script to run the main trading bot, API server, and dashboard together
"""

import asyncio
import logging
import sys
from pathlib import Path

# Setup logging configuration
from src.utils.logging_config import setup_logging
setup_logging()

# Disable werkzeug logs
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('dash').setLevel(logging.ERROR)
logging.getLogger('dash.dash').setLevel(logging.ERROR)
logging.getLogger('flask').setLevel(logging.ERROR)
logging.getLogger('aiohttp.access').setLevel(logging.ERROR)
logging.getLogger('websockets.server').setLevel(logging.WARNING)


import threading
import time
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_main_trading_bot():
    """Run the main trading bot."""
    try:
        from main_with_quantitative import main
        
        logger.info("🤖 Starting Main Trading Bot (main_with_quantitative.py)...")
        await main()
        
    except Exception as e:
        logger.error(f"❌ Error running main trading bot: {e}")

async def run_api_server():
    """Run the performance API server."""
    try:
        from performance_api_server import PerformanceAPIServer
        
        server = PerformanceAPIServer()
        logger.info("🚀 Starting Performance API Server...")
        await server.run(host='localhost', port=8000)
        
    except Exception as e:
        logger.error(f"❌ Error running API server: {e}")

def run_dashboard():
    """Run the HTTP polling dashboard."""
    try:
        from performance_dashboard_http_polling import start_http_polling_dashboard
        
        logger.info("🌐 Starting HTTP Polling Dashboard...")
        start_http_polling_dashboard(
            api_url="http://localhost:8000/api/performance",
            host='localhost',
            port=8050
        )
        
    except Exception as e:
        logger.error(f"❌ Error running dashboard: {e}")

async def run_complete_system():
    """Run the complete system with all components."""
    try:
        logger.info("🚀 Starting Complete Trading System")
        logger.info("=" * 60)
        logger.info("📊 Components:")
        logger.info("   • Main Trading Bot (main_with_quantitative.py)")
        logger.info("   • Performance API Server (port 8000)")
        logger.info("   • HTTP Polling Dashboard (port 8050)")
        logger.info("   • Quantitative Analysis")
        logger.info("   • Real-time Performance Monitoring")
        logger.info("=" * 60)
        
        # Start API server in background
        api_task = asyncio.create_task(run_api_server())
        
        # Wait for API server to start
        logger.info("⏳ Waiting for API server to start...")
        await asyncio.sleep(5)
        
        # Start dashboard in separate thread
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        
        # Wait for dashboard to start
        logger.info("⏳ Waiting for dashboard to start...")
        await asyncio.sleep(3)
        
        logger.info("✅ API Server and Dashboard started successfully!")
        logger.info("📊 Available endpoints:")
        logger.info("   • API Server: http://localhost:8000")
        logger.info("   • API Data: http://localhost:8000/api/performance")
        logger.info("   • Dashboard: http://localhost:8050")
        logger.info("   • Health Check: http://localhost:8000/api/health")
        
        # Start main trading bot
        logger.info("🤖 Starting Main Trading Bot...")
        bot_task = asyncio.create_task(run_main_trading_bot())
        
        # Wait for both API server and trading bot
        await asyncio.gather(api_task, bot_task)
        
    except KeyboardInterrupt:
        logger.info("🛑 System stopped by user")
    except Exception as e:
        logger.error(f"❌ Error running complete system: {e}")

def main():
    """Main function."""
    print("🚀 Complete Trading System")
    print("=" * 60)
    print("📊 System Components:")
    print("   • Main Trading Bot (main_with_quantitative.py)")
    print("   • Performance API Server (port 8000)")
    print("   • HTTP Polling Dashboard (port 8050)")
    print("   • Quantitative Analysis Integration")
    print("   • Real-time Performance Monitoring")
    print("   • Trading Signal Processing")
    print("   • Portfolio Optimization")
    print()
    
    print("🎯 Starting complete system in 5 seconds...")
    for i in range(5, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    try:
        asyncio.run(run_complete_system())
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main() 