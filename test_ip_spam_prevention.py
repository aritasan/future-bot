"""
Test script for IP spam prevention logic.
"""

import asyncio
import logging
import time
from src.services.ip_monitor_service import IPMonitorService
from src.core.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_spam_prevention():
    """Test spam prevention logic."""
    try:
        # Load config
        config = load_config()
        
        # Create IP monitor without notification callback
        ip_monitor = IPMonitorService(config)
        
        # Initialize
        if await ip_monitor.initialize():
            logger.info("✅ IP monitor initialized successfully")
            
            # Test spam prevention logic
            current_time = time.time()
            
            # Test 1: First notification (should always notify)
            logger.info("\n🧪 Test 1: First notification")
            should_notify = await ip_monitor._should_send_notification("192.168.1.100", current_time)
            logger.info(f"Should notify for first IP: {should_notify}")
            
            # Test 2: Same IP again (should not notify due to cooldown)
            logger.info("\n🧪 Test 2: Same IP again (cooldown)")
            # First, simulate that we just notified about this IP
            ip_monitor._last_notified_ip = "192.168.1.100"
            ip_monitor._last_notification_time = current_time
            should_notify = await ip_monitor._should_send_notification("192.168.1.100", current_time)
            logger.info(f"Should notify for same IP: {should_notify}")
            
            # Test 3: Different IP (should notify)
            logger.info("\n🧪 Test 3: Different IP")
            should_notify = await ip_monitor._should_send_notification("192.168.1.101", current_time)
            logger.info(f"Should notify for different IP: {should_notify}")
            
            # Test 4: Same IP after cooldown (should notify)
            logger.info("\n🧪 Test 4: Same IP after cooldown")
            future_time = current_time + ip_monitor._notification_cooldown + 60  # After cooldown
            should_notify = await ip_monitor._should_send_notification("192.168.1.100", future_time)
            logger.info(f"Should notify for same IP after cooldown: {should_notify}")
            
            # Test 5: Simulate IP change sequence
            logger.info("\n🧪 Test 5: IP change sequence simulation")
            
            # Reset tracking variables
            ip_monitor._last_notified_ip = None
            ip_monitor._last_notification_time = 0
            
            # Simulate IP A -> B -> A -> C
            test_sequence = [
                ("192.168.1.100", "IP A"),
                ("192.168.1.101", "IP B"), 
                ("192.168.1.100", "IP A again"),
                ("192.168.1.102", "IP C")
            ]
            
            for ip, description in test_sequence:
                should_notify = await ip_monitor._should_send_notification(ip, current_time)
                logger.info(f"{description} ({ip}): {'✅ Notify' if should_notify else '❌ Skip'}")
                
                if should_notify:
                    # Update the tracking variables as if notification was sent
                    ip_monitor._last_notified_ip = ip
                    ip_monitor._last_notification_time = current_time
            
        else:
            logger.error("❌ Failed to initialize IP monitor")
            
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        
    finally:
        if 'ip_monitor' in locals():
            await ip_monitor.close()

async def test_binance_ip_error_logic():
    """Test Binance IP error notification logic."""
    try:
        from src.services.binance_service import BinanceService
        
        # Load config
        config = load_config()
        
        # Create Binance service without notification callback
        binance_service = BinanceService(config)
        
        # Test IP error notification logic
        current_time = time.time()
        
        # Test 1: First IP error (should notify)
        logger.info("\n🧪 Test 1: First IP error")
        should_notify = await binance_service._should_send_ip_error_notification("192.168.1.100", current_time)
        logger.info(f"Should notify for first IP error: {should_notify}")
        
        # Test 2: Same IP error again (should not notify)
        logger.info("\n🧪 Test 2: Same IP error again")
        # First, simulate that we just notified about this IP error
        binance_service._last_notified_ip_error = "192.168.1.100"
        binance_service._last_ip_error_time = current_time
        should_notify = await binance_service._should_send_ip_error_notification("192.168.1.100", current_time)
        logger.info(f"Should notify for same IP error: {should_notify}")
        
        # Test 3: Different IP error (should notify)
        logger.info("\n🧪 Test 3: Different IP error")
        should_notify = await binance_service._should_send_ip_error_notification("192.168.1.101", current_time)
        logger.info(f"Should notify for different IP error: {should_notify}")
        
        # Test 4: Same IP after cooldown (should notify)
        logger.info("\n🧪 Test 4: Same IP error after cooldown")
        future_time = current_time + binance_service._ip_error_cooldown + 60  # After cooldown
        should_notify = await binance_service._should_send_ip_error_notification("192.168.1.100", future_time)
        logger.info(f"Should notify for same IP error after cooldown: {should_notify}")
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("🧪 Testing IP Spam Prevention Logic...")
    
    # Test 1: IP Monitor spam prevention
    print("\n=== Test 1: IP Monitor Spam Prevention ===")
    asyncio.run(test_spam_prevention())
    
    # Test 2: Binance IP Error spam prevention
    print("\n=== Test 2: Binance IP Error Spam Prevention ===")
    asyncio.run(test_binance_ip_error_logic())
    
    print("\n✅ All tests completed!") 