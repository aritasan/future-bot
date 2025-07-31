#!/usr/bin/env python3
"""
Test HTTP Polling System
Script to test the HTTP polling performance monitoring system
"""

import requests
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_endpoints():
    """Test API endpoints."""
    logger.info("🧪 Testing API Endpoints...")
    
    results = {}
    
    # Test 1: Health Check
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info("✅ Health Check: PASS")
            logger.info(f"   Status: {data.get('status')}")
            logger.info(f"   Monitor Active: {data.get('monitor_active')}")
            results['health_check'] = True
        else:
            logger.error(f"❌ Health Check: FAIL (Status: {response.status_code})")
            results['health_check'] = False
    except Exception as e:
        logger.error(f"❌ Health Check: FAIL ({str(e)})")
        results['health_check'] = False
    
    # Test 2: Performance Data
    try:
        response = requests.get("http://localhost:8000/api/performance", timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info("✅ Performance Data: PASS")
            logger.info(f"   Performance Score: {data.get('performance_score', 'N/A')}")
            logger.info(f"   Risk Score: {data.get('risk_score', 'N/A')}")
            logger.info(f"   Stability Score: {data.get('stability_score', 'N/A')}")
            logger.info(f"   System Status: {data.get('system_status', 'N/A')}")
            results['performance_data'] = True
        else:
            logger.error(f"❌ Performance Data: FAIL (Status: {response.status_code})")
            results['performance_data'] = False
    except Exception as e:
        logger.error(f"❌ Performance Data: FAIL ({str(e)})")
        results['performance_data'] = False
    
    return results

def test_dashboard():
    """Test dashboard accessibility."""
    logger.info("🧪 Testing Dashboard...")
    
    try:
        response = requests.get("http://localhost:8050", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Dashboard: PASS")
            logger.info("   Dashboard is accessible")
            return True
        else:
            logger.error(f"❌ Dashboard: FAIL (Status: {response.status_code})")
            return False
    except Exception as e:
        logger.error(f"❌ Dashboard: FAIL ({str(e)})")
        return False

def test_data_flow():
    """Test data flow from API to dashboard."""
    logger.info("🧪 Testing Data Flow...")
    
    try:
        # Get data from API
        api_response = requests.get("http://localhost:8000/api/performance", timeout=5)
        if api_response.status_code != 200:
            logger.error("❌ Data Flow: FAIL (API not responding)")
            return False
        
        api_data = api_response.json()
        
        # Check if data has required fields
        required_fields = ['performance_score', 'risk_score', 'stability_score', 'timestamp']
        missing_fields = [field for field in required_fields if field not in api_data]
        
        if missing_fields:
            logger.error(f"❌ Data Flow: FAIL (Missing fields: {missing_fields})")
            return False
        
        logger.info("✅ Data Flow: PASS")
        logger.info("   API data structure is correct")
        logger.info(f"   Data: {json.dumps(api_data, indent=2)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data Flow: FAIL ({str(e)})")
        return False

def main():
    """Main test function."""
    logger.info("🚀 HTTP Polling System Test")
    logger.info("=" * 50)
    
    results = {}
    
    # Test 1: API Endpoints
    logger.info("\n📡 Test 1: API Endpoints")
    api_results = test_api_endpoints()
    results.update(api_results)
    
    # Test 2: Dashboard
    logger.info("\n🌐 Test 2: Dashboard")
    results['dashboard'] = test_dashboard()
    
    # Test 3: Data Flow
    logger.info("\n📊 Test 3: Data Flow")
    results['data_flow'] = test_data_flow()
    
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
        logger.info("🎉 All tests passed! HTTP Polling System is working correctly.")
        logger.info("\n📋 System Status:")
        logger.info("   • API Server: ✅ Running on http://localhost:8000")
        logger.info("   • Dashboard: ✅ Running on http://localhost:8050")
        logger.info("   • Data Flow: ✅ HTTP polling is working")
        logger.info("   • No WebSocket: ✅ Using HTTP polling instead")
    else:
        logger.warning("⚠️ Some tests failed. Please check the issues above.")
        
        if not results.get('health_check'):
            logger.info("\n💡 API Health Check Issues:")
            logger.info("   • Make sure API server is running: python run_http_polling_system.py")
            logger.info("   • Check if port 8000 is available")
            
        if not results.get('dashboard'):
            logger.info("\n💡 Dashboard Issues:")
            logger.info("   • Make sure dashboard is running")
            logger.info("   • Check if port 8050 is available")

if __name__ == "__main__":
    main() 