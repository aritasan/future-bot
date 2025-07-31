#!/usr/bin/env python3
"""
Performance API Server
Simple HTTP server to provide performance data for dashboard polling
"""

import asyncio
import json
import logging
from datetime import datetime
from aiohttp import web
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.quantitative.real_time_performance_monitor import WorldQuantRealTimePerformanceMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceAPIServer:
    def __init__(self):
        self.monitor = None
        self.app = web.Application()
        self.setup_routes()
        
    def setup_routes(self):
        """Setup API routes."""
        self.app.router.add_get('/', self.index)
        self.app.router.add_get('/api/performance', self.get_performance_data)
        self.app.router.add_get('/api/health', self.health_check)
        
    async def index(self, request):
        """Index page."""
        return web.Response(
            text="Performance API Server\nUse /api/performance to get data",
            content_type='text/plain'
        )
        
    async def get_performance_data(self, request):
        """Get performance data for dashboard."""
        try:
            if self.monitor is None:
                # Return default data if monitor not initialized
                default_data = {
                    'performance_score': 0.0,
                    'risk_score': 0.0,
                    'stability_score': 0.0,
                    'timestamp': datetime.now().isoformat(),
                    'alerts_count': 0,
                    'system_status': 'initializing',
                    'websocket_clients': 0,
                    'last_update': datetime.now().isoformat()
                }
                return web.json_response(default_data)
            
            # Get real-time data from monitor
            data = await self.monitor.get_real_time_summary()
            return web.json_response(data)
            
        except Exception as e:
            logger.error(f"Error getting performance data: {str(e)}")
            error_data = {
                'performance_score': 0.0,
                'risk_score': 0.0,
                'stability_score': 0.0,
                'timestamp': datetime.now().isoformat(),
                'alerts_count': 0,
                'system_status': 'error',
                'websocket_clients': 0,
                'last_update': datetime.now().isoformat(),
                'error': str(e)
            }
            return web.json_response(error_data)
            
    async def health_check(self, request):
        """Health check endpoint."""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'monitor_active': self.monitor is not None
        })
        
    async def initialize_monitor(self):
        """Initialize the performance monitor."""
        try:
            self.monitor = WorldQuantRealTimePerformanceMonitor()
            await self.monitor.initialize()
            logger.info("Performance monitor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing monitor: {str(e)}")
            
    async def run(self, host='localhost', port=8000):
        """Run the API server."""
        try:
            # Initialize monitor
            await self.initialize_monitor()
            
            # Start server
            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, host, port)
            
            logger.info(f"🚀 Performance API Server starting on http://{host}:{port}")
            await site.start()
            
            logger.info("✅ Performance API Server started successfully")
            logger.info(f"📊 API Endpoints:")
            logger.info(f"   • GET /api/performance - Get performance data")
            logger.info(f"   • GET /api/health - Health check")
            
            # Keep server running
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error running API server: {str(e)}")

async def main():
    """Main function."""
    server = PerformanceAPIServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main()) 