#!/usr/bin/env python3
"""
Run HTTP Polling System
Script to run both API server and HTTP polling dashboard
"""

import asyncio
import threading
import time
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

async def run_system():
    """Run the complete HTTP polling system."""
    try:
        logger.info("🚀 Starting HTTP Polling System")
        logger.info("=" * 50)
        
        # Start API server in background
        api_task = asyncio.create_task(run_api_server())
        
        # Wait for API server to start
        logger.info("⏳ Waiting for API server to start...")
        await asyncio.sleep(3)
        
        # Start dashboard in separate thread
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        
        logger.info("✅ System started successfully!")
        logger.info("📊 Available endpoints:")
        logger.info("   • API Server: http://localhost:8000")
        logger.info("   • API Data: http://localhost:8000/api/performance")
        logger.info("   • Dashboard: http://localhost:8050")
        logger.info("   • Health Check: http://localhost:8000/api/health")
        
        # Keep system running
        await api_task
        
    except KeyboardInterrupt:
        logger.info("🛑 System stopped by user")
    except Exception as e:
        logger.error(f"❌ Error running system: {e}")

def main():
    """Main function."""
    print("🚀 HTTP Polling Performance Monitoring System")
    print("=" * 60)
    print("📊 Features:")
    print("   • HTTP API Server (port 8000)")
    print("   • HTTP Polling Dashboard (port 8050)")
    print("   • Real-time Performance Data")
    print("   • 5-second polling interval")
    print("   • No WebSocket dependencies")
    print()
    
    print("🎯 Starting system in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    try:
        asyncio.run(run_system())
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main() 