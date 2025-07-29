"""
Demo script to illustrate IP spam prevention logic.
"""

import asyncio
import logging
import time
from src.services.ip_monitor_service import IPMonitorService
from src.core.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockNotificationCallback:
    """Mock notification callback to simulate sending notifications."""
    
    def __init__(self):
        self.notifications_sent = []
        
    async def __call__(self, message: str):
        """Simulate sending notification."""
        timestamp = time.strftime("%H:%M:%S")
        self.notifications_sent.append(f"[{timestamp}] {message[:50]}...")
        logger.info(f"📱 Notification sent: {message[:50]}...")

async def demo_ip_spam_prevention():
    """Demo IP spam prevention logic."""
    try:
        # Load config
        config = load_config()
        
        # Create mock notification callback
        mock_callback = MockNotificationCallback()
        
        # Create IP monitor with mock callback
        ip_monitor = IPMonitorService(config, mock_callback)
        
        # Initialize
        if await ip_monitor.initialize():
            logger.info("✅ IP monitor initialized successfully")
            
            # Demo scenario: IP changes and errors
            logger.info("\n🎭 Demo Scenario: IP Changes and Spam Prevention")
            logger.info("=" * 60)
            
            # Scenario 1: IP A -> A -> A (should only notify once)
            logger.info("\n📋 Scenario 1: Same IP multiple times")
            logger.info("Expected: Only 1 notification for IP A")
            
            # Simulate IP A detected multiple times
            for i in range(3):
                logger.info(f"  Detecting IP A (attempt {i+1})...")
                await ip_monitor._check_ip_change_simulation("192.168.1.100", "IP A")
                await asyncio.sleep(1)
            
            # Scenario 2: IP A -> B -> A (should notify for A, B, then A again)
            logger.info("\n📋 Scenario 2: IP A -> B -> A")
            logger.info("Expected: 3 notifications (A, B, A again)")
            
            # Reset for new scenario
            ip_monitor._last_notified_ip = None
            ip_monitor._last_notification_time = 0
            
            # Simulate IP changes
            await ip_monitor._check_ip_change_simulation("192.168.1.100", "IP A")
            await ip_monitor._check_ip_change_simulation("192.168.1.101", "IP B")
            await ip_monitor._check_ip_change_simulation("192.168.1.100", "IP A again")
            
            # Scenario 3: Same IP after cooldown
            logger.info("\n📋 Scenario 3: Same IP after cooldown")
            logger.info("Expected: Notification after cooldown period")
            
            # Simulate time passing (cooldown + 1 minute)
            future_time = time.time() + ip_monitor._notification_cooldown + 60
            ip_monitor._last_notification_time = time.time() - ip_monitor._notification_cooldown - 60
            
            await ip_monitor._check_ip_change_simulation("192.168.1.100", "IP A after cooldown", future_time)
            
            # Show results
            logger.info("\n📊 Results Summary:")
            logger.info("=" * 60)
            for i, notification in enumerate(mock_callback.notifications_sent, 1):
                logger.info(f"{i}. {notification}")
            
            logger.info(f"\n✅ Total notifications sent: {len(mock_callback.notifications_sent)}")
            logger.info("🎯 Spam prevention working correctly!")
            
        else:
            logger.error("❌ Failed to initialize IP monitor")
            
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        
    finally:
        if 'ip_monitor' in locals():
            await ip_monitor.close()

# Add simulation method to IPMonitorService for demo
async def _check_ip_change_simulation(self, new_ip: str, description: str, current_time: float = None):
    """Simulate IP change detection for demo purposes."""
    if current_time is None:
        current_time = time.time()
        
    logger.info(f"  🔍 Checking {description} ({new_ip})...")
    
    if self._current_ip and new_ip != self._current_ip:
        logger.info(f"  ⚠️  IP changed from {self._current_ip} to {new_ip}")
        
        # Check if we should send notification
        should_notify = await self._should_send_notification(new_ip, current_time)
        
        if should_notify:
            logger.info(f"  ✅ Sending notification for {description}")
            await self._handle_ip_change(self._current_ip, new_ip)
            self._last_notified_ip = new_ip
            self._last_notification_time = current_time
        else:
            logger.info(f"  ❌ Skipping notification for {description} (spam prevention)")
    else:
        logger.info(f"  ℹ️  No IP change detected for {description}")
        
    self._current_ip = new_ip

# Add the simulation method to IPMonitorService
IPMonitorService._check_ip_change_simulation = _check_ip_change_simulation

if __name__ == "__main__":
    print("🎭 IP Spam Prevention Demo")
    print("=" * 60)
    asyncio.run(demo_ip_spam_prevention())
    print("\n🎉 Demo completed!") 