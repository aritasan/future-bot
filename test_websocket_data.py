#!/usr/bin/env python3
"""
Simple WebSocket data test
"""

import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_websocket_data():
    """Test WebSocket data reception."""
    try:
        uri = "ws://localhost:8765"
        logger.info(f"Connecting to {uri}...")
        
        async with websockets.connect(uri) as websocket:
            logger.info("✅ Connected to WebSocket server")
            
            # Wait for initial data
            try:
                logger.info("Waiting for initial data...")
                data = await asyncio.wait_for(websocket.recv(), timeout=10)
                logger.info(f"📊 Received data: {data}")
                
                # Try to parse JSON
                try:
                    json_data = json.loads(data)
                    logger.info("✅ JSON parsed successfully")
                    logger.info(f"Performance Score: {json_data.get('performance_score', 'N/A')}")
                    logger.info(f"Risk Score: {json_data.get('risk_score', 'N/A')}")
                    logger.info(f"Stability Score: {json_data.get('stability_score', 'N/A')}")
                    return True
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON parsing error: {e}")
                    logger.error(f"Raw data: {data}")
                    return False
                    
            except asyncio.TimeoutError:
                logger.warning("⚠️ No data received within timeout")
                return False
                
    except Exception as e:
        logger.error(f"❌ WebSocket error: {str(e)}")
        return False

async def main():
    """Main function."""
    logger.info("🧪 Testing WebSocket Data Reception")
    logger.info("=" * 40)
    
    success = await test_websocket_data()
    
    if success:
        logger.info("🎉 WebSocket data test successful!")
    else:
        logger.error("❌ WebSocket data test failed")

if __name__ == "__main__":
    asyncio.run(main()) 