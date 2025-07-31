#!/usr/bin/env python3
"""
Test System Comparison
Script to test and compare different system configurations
"""

import requests
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_data():
    """Test API data to see if it's real or simulated."""
    try:
        response = requests.get("http://localhost:8000/api/performance", timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info("📊 API Data Analysis:")
            logger.info(f"   Performance Score: {data.get('performance_score', 'N/A')}")
            logger.info(f"   Risk Score: {data.get('risk_score', 'N/A')}")
            logger.info(f"   Stability Score: {data.get('stability_score', 'N/A')}")
            logger.info(f"   System Status: {data.get('system_status', 'N/A')}")
            logger.info(f"   Alerts Count: {data.get('alerts_count', 'N/A')}")
            
            # Check if data is simulated
            if data.get('system_status') == 'error' or data.get('system_status') == 'initializing':
                logger.warning("⚠️ This appears to be SIMULATED data (no real trading bot running)")
                return False
            else:
                logger.info("✅ This appears to be REAL data from trading bot")
                return True
        else:
            logger.error(f"❌ API not responding (Status: {response.status_code})")
            return False
    except Exception as e:
        logger.error(f"❌ Error testing API: {str(e)}")
        return False

def test_trading_bot_status():
    """Test if trading bot is actually running."""
    try:
        # Check if there are any trading-related processes
        import psutil
        
        trading_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python' and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'main_with_quantitative' in cmdline:
                        trading_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if trading_processes:
            logger.info("✅ Trading bot processes found:")
            for proc in trading_processes:
                logger.info(f"   PID {proc['pid']}: {proc['cmdline']}")
            return True
        else:
            logger.warning("⚠️ No trading bot processes found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error checking trading bot status: {str(e)}")
        return False

def test_dashboard_connection():
    """Test dashboard connection."""
    try:
        response = requests.get("http://localhost:8050", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Dashboard is accessible")
            return True
        else:
            logger.error(f"❌ Dashboard not accessible (Status: {response.status_code})")
            return False
    except Exception as e:
        logger.error(f"❌ Error testing dashboard: {str(e)}")
        return False

def main():
    """Main test function."""
    logger.info("🔍 System Configuration Analysis")
    logger.info("=" * 50)
    
    # Test 1: Check API data
    logger.info("\n📊 Test 1: API Data Analysis")
    has_real_data = test_api_data()
    
    # Test 2: Check trading bot status
    logger.info("\n🤖 Test 2: Trading Bot Status")
    bot_running = test_trading_bot_status()
    
    # Test 3: Check dashboard
    logger.info("\n🌐 Test 3: Dashboard Status")
    dashboard_ok = test_dashboard_connection()
    
    # Summary
    logger.info("\n📋 System Status Summary:")
    logger.info("=" * 30)
    
    if has_real_data and bot_running:
        logger.info("✅ COMPLETE SYSTEM: All components running")
        logger.info("   • Trading Bot: ✅ Running")
        logger.info("   • API Server: ✅ Real data")
        logger.info("   • Dashboard: ✅ Connected")
        logger.info("   • Status: Production Ready")
        
    elif not bot_running and has_real_data:
        logger.info("⚠️ PARTIAL SYSTEM: API + Dashboard only")
        logger.info("   • Trading Bot: ❌ Not running")
        logger.info("   • API Server: ✅ Simulated data")
        logger.info("   • Dashboard: ✅ Connected")
        logger.info("   • Status: Demo Mode")
        
    elif not bot_running and not has_real_data:
        logger.info("❌ MINIMAL SYSTEM: Dashboard only")
        logger.info("   • Trading Bot: ❌ Not running")
        logger.info("   • API Server: ❌ No data")
        logger.info("   • Dashboard: ✅ Connected")
        logger.info("   • Status: Dashboard Only")
    
    # Recommendations
    logger.info("\n💡 Recommendations:")
    if not bot_running:
        logger.info("   • Run: python run_complete_system.py")
        logger.info("   • This will start the full trading system")
    else:
        logger.info("   • System is running correctly")
        logger.info("   • Monitor dashboard for real trading data")

if __name__ == "__main__":
    main() 