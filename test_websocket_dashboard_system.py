#!/usr/bin/env python3
"""
Comprehensive WebSocket and Dashboard System Test
"""

import asyncio
import websockets
import json
import logging
import time
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_websocket_connection():
    """Test WebSocket connection and data flow."""
    logger.info("🧪 Testing WebSocket Connection...")
    
    try:
        # Test connection to port 8765 (main bot)
        uri = "ws://localhost:8765"
        logger.info(f"Testing connection to {uri}...")
        
        async with websockets.connect(uri) as websocket:
            logger.info("✅ Connected to WebSocket server")
            
            # Wait for initial data
            try:
                data = await asyncio.wait_for(websocket.recv(), timeout=10)
                logger.info(f"📊 Received data: {data[:200]}...")
                
                # Parse JSON
                try:
                    json_data = json.loads(data)
                    logger.info("✅ JSON data parsed successfully")
                    return True
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON parsing error: {e}")
                    return False
                    
            except asyncio.TimeoutError:
                logger.warning("⚠️ No data received within timeout")
                return False
                
    except Exception as e:
        logger.error(f"❌ WebSocket connection error: {str(e)}")
        return False

def test_dashboard_connection():
    """Test dashboard HTTP connection."""
    logger.info("🧪 Testing Dashboard Connection...")
    
    try:
        response = requests.get("http://localhost:8050", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Dashboard is accessible")
            return True
        else:
            logger.error(f"❌ Dashboard returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Dashboard connection error: {str(e)}")
        return False

def test_simple_websocket():
    """Test simple WebSocket server."""
    logger.info("🧪 Testing Simple WebSocket Server...")
    
    try:
        response = requests.get("http://localhost:8051", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Simple dashboard is accessible")
            return True
        else:
            logger.error(f"❌ Simple dashboard returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Simple dashboard connection error: {str(e)}")
        return False

async def main():
    """Main test function."""
    logger.info("🚀 WebSocket and Dashboard System Test")
    logger.info("=" * 50)
    
    results = {}
    
    # Test 1: WebSocket Connection
    logger.info("\n📡 Test 1: WebSocket Connection")
    results['websocket'] = await test_websocket_connection()
    
    # Test 2: Dashboard Connection
    logger.info("\n🌐 Test 2: Dashboard Connection")
    results['dashboard'] = test_dashboard_connection()
    
    # Test 3: Simple WebSocket (if running)
    logger.info("\n🧪 Test 3: Simple WebSocket Test")
    results['simple_websocket'] = test_simple_websocket()
    
    # Summary
    logger.info("\n📊 Test Results Summary:")
    logger.info("=" * 30)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
    
    passed = sum(results.values())
    total = len(results)
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is working correctly.")
        logger.info("\n📋 System Status:")
        logger.info("   • WebSocket Server: ✅ Running on ws://localhost:8765")
        logger.info("   • Dashboard: ✅ Accessible at http://localhost:8050")
        logger.info("   • Data Flow: ✅ WebSocket data is being sent")
    else:
        logger.warning("⚠️ Some tests failed. Please check the issues above.")
        
        if not results.get('websocket'):
            logger.info("\n💡 WebSocket Issues:")
            logger.info("   • Make sure the trading bot is running: python run_bot.py")
            logger.info("   • Check if port 8765 is available")
            
        if not results.get('dashboard'):
            logger.info("\n💡 Dashboard Issues:")
            logger.info("   • Make sure dashboard is running: python run_dashboard.py")
            logger.info("   • Check if port 8050 is available")

if __name__ == "__main__":
    asyncio.run(main()) 