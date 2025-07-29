"""
Simple test script for IP monitoring.
"""

import asyncio
import logging
from src.services.ip_monitor_service import IPMonitorService
from src.core.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def simple_ip_test():
    """Simple IP monitoring test."""
    try:
        # Load config
        config = load_config()
        
        # Create IP monitor without notification callback
        ip_monitor = IPMonitorService(config)
        
        # Initialize
        if await ip_monitor.initialize():
            logger.info("✅ IP monitor initialized successfully")
            
            # Get current IP
            current_ip = await ip_monitor.get_current_ip()
            logger.info(f"🌐 Current IP: {current_ip}")
            
            # Test IP validation
            test_ips = ["192.168.1.1", "256.256.256.256", "invalid", "8.8.8.8"]
            for test_ip in test_ips:
                is_valid = ip_monitor._is_valid_ip(test_ip)
                logger.info(f"IP {test_ip}: {'✅ Valid' if is_valid else '❌ Invalid'}")
            
            # Test message creation
            old_ip = "192.168.1.100"
            new_ip = "192.168.1.101"
            message = ip_monitor._create_ip_change_message(old_ip, new_ip)
            logger.info("📝 Sample notification message:")
            print(message)
            
        else:
            logger.error("❌ Failed to initialize IP monitor")
            
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        
    finally:
        if 'ip_monitor' in locals():
            await ip_monitor.close()

if __name__ == "__main__":
    print("🧪 Testing IP Monitoring...")
    asyncio.run(simple_ip_test())
    print("✅ Test completed!") 