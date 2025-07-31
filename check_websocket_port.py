#!/usr/bin/env python3
"""
Script to check WebSocket port availability and find the correct port
"""

import asyncio
import websockets
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def check_websocket_ports():
    """Check which WebSocket ports are available and running."""
    ports_to_check = [8765, 8766, 8767, 8768, 8769]
    
    logger.info("🔍 Checking WebSocket ports...")
    
    for port in ports_to_check:
        try:
            # Try to connect to the WebSocket server
            uri = f"ws://localhost:{port}"
            logger.info(f"Testing connection to {uri}...")
            
            async with websockets.connect(uri) as websocket:
                logger.info(f"✅ SUCCESS: WebSocket server is running on port {port}")
                logger.info(f"🌐 Dashboard should connect to: ws://localhost:{port}")
                return port
                
        except websockets.exceptions.InvalidURI:
            logger.error(f"❌ Invalid URI: {uri}")
        except websockets.exceptions.InvalidHandshake:
            logger.warning(f"⚠️ Port {port} is in use but not a WebSocket server")
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"⚠️ Port {port} connection closed")
        except Exception as e:
            logger.warning(f"⚠️ Port {port} not available: {str(e)}")
    
    logger.error("❌ No WebSocket servers found on any port")
    return None

async def test_websocket_connection(port):
    """Test WebSocket connection and get data."""
    try:
        uri = f"ws://localhost:{port}"
        logger.info(f"🔗 Testing WebSocket connection to {uri}...")
        
        async with websockets.connect(uri) as websocket:
            logger.info("✅ Connected to WebSocket server")
            
            # Wait for initial data
            try:
                data = await asyncio.wait_for(websocket.recv(), timeout=10)
                logger.info(f"📊 Received data: {data[:200]}...")
                return True
            except asyncio.TimeoutError:
                logger.warning("⚠️ No data received within timeout")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error testing WebSocket connection: {str(e)}")
        return False

async def main():
    """Main function to check WebSocket ports."""
    logger.info("🚀 WebSocket Port Checker")
    logger.info("=" * 50)
    
    # Check which port is available
    port = await check_websocket_ports()
    
    if port:
        logger.info(f"🎯 Found WebSocket server on port {port}")
        
        # Test the connection
        success = await test_websocket_connection(port)
        
        if success:
            logger.info("✅ WebSocket connection test successful!")
            logger.info(f"📋 Dashboard Configuration:")
            logger.info(f"   WebSocket URL: ws://localhost:{port}")
            logger.info(f"   Dashboard URL: http://localhost:8050")
        else:
            logger.warning("⚠️ WebSocket connection test failed")
    else:
        logger.error("❌ No WebSocket servers found")
        logger.info("💡 Make sure to run the trading bot first:")
        logger.info("   python run_bot.py")

if __name__ == "__main__":
    asyncio.run(main()) 