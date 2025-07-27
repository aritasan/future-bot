"""
IP Monitor Service for detecting IP address changes.
"""

import logging
import asyncio
import aiohttp
import time
from typing import Dict, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

class IPMonitorService:
    """Service for monitoring IP address changes."""
    
    def __init__(self, config: Dict, notification_callback: Optional[Callable] = None):
        """Initialize the IP monitor service.
        
        Args:
            config: Configuration dictionary
            notification_callback: Callback function to send notifications
        """
        self.config = config
        self.notification_callback = notification_callback
        self._current_ip = None
        self._last_check_time = 0
        self._check_interval = config.get('ip_monitor', {}).get('check_interval', 300)  # 5 minutes
        self._ip_services = [
            'https://api.ipify.org',
            'https://httpbin.org/ip',
            'https://ipinfo.io/ip',
            'https://icanhazip.com'
        ]
        self._is_running = False
        self._monitor_task = None
        
        # Track IP change notifications to avoid spam
        self._last_notified_ip = None
        self._last_notification_time = 0
        self._notification_cooldown = config.get('ip_monitor', {}).get('notification_cooldown', 300)  # 5 minutes
        
    async def initialize(self) -> bool:
        """Initialize the IP monitor service."""
        try:
            # Get initial IP
            initial_ip = await self._get_current_ip()
            if initial_ip:
                self._current_ip = initial_ip
                logger.info(f"IP Monitor initialized with IP: {initial_ip}")
                return True
            else:
                logger.error("Failed to get initial IP address")
                return False
        except Exception as e:
            logger.error(f"Error initializing IP monitor: {str(e)}")
            return False
            
    async def start_monitoring(self) -> None:
        """Start IP monitoring in background."""
        if self._is_running:
            logger.warning("IP monitoring is already running")
            return
            
        self._is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("IP monitoring started")
        
    async def stop_monitoring(self) -> None:
        """Stop IP monitoring."""
        self._is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("IP monitoring stopped")
        
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._is_running:
            try:
                await self._check_ip_change()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in IP monitor loop: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
                
    async def _check_ip_change(self) -> None:
        """Check if IP has changed."""
        try:
            current_time = time.time()
            if current_time - self._last_check_time < self._check_interval:
                return
                
            new_ip = await self._get_current_ip()
            if not new_ip:
                logger.warning("Failed to get current IP address")
                return
                
            self._last_check_time = current_time
            
            # Only send notification if IP actually changed and we haven't notified for this IP recently
            if self._current_ip and new_ip != self._current_ip:
                logger.warning(f"IP address changed from {self._current_ip} to {new_ip}")
                
                # Check if we should send notification (avoid spam)
                should_notify = await self._should_send_notification(new_ip, current_time)
                
                if should_notify:
                    await self._handle_ip_change(self._current_ip, new_ip)
                    self._last_notified_ip = new_ip
                    self._last_notification_time = current_time
                else:
                    logger.info(f"IP changed to {new_ip} but notification skipped (already notified or cooldown active)")
                
            self._current_ip = new_ip
            
        except Exception as e:
            logger.error(f"Error checking IP change: {str(e)}")
            
    async def _get_current_ip(self) -> Optional[str]:
        """Get current public IP address."""
        for service_url in self._ip_services:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.get(service_url) as response:
                        if response.status == 200:
                            content = await response.text()
                            # Extract IP from response
                            ip = content.strip()
                            # Validate IP format (basic validation)
                            if self._is_valid_ip(ip):
                                return ip
            except Exception as e:
                logger.debug(f"Failed to get IP from {service_url}: {str(e)}")
                continue
                
        return None
        
    def _is_valid_ip(self, ip: str) -> bool:
        """Basic IP validation."""
        try:
            parts = ip.split('.')
            if len(parts) != 4:
                return False
            for part in parts:
                if not part.isdigit() or not 0 <= int(part) <= 255:
                    return False
            return True
        except:
            return False
            
    async def _should_send_notification(self, new_ip: str, current_time: float) -> bool:
        """Check if we should send notification for this IP change.
        
        Args:
            new_ip: The new IP address
            current_time: Current timestamp
            
        Returns:
            bool: True if notification should be sent, False otherwise
        """
        # If this is the first time we're seeing this IP, always notify
        if self._last_notified_ip is None:
            return True
            
        # If this is a different IP than the last one we notified about, notify
        if new_ip != self._last_notified_ip:
            return True
            
        # If it's the same IP but enough time has passed since last notification, notify
        if current_time - self._last_notification_time > self._notification_cooldown:
            return True
            
        # Otherwise, don't notify (avoid spam)
        return False
        
    async def _handle_ip_change(self, old_ip: str, new_ip: str) -> None:
        """Handle IP address change."""
        try:
            message = self._create_ip_change_message(old_ip, new_ip)
            
            if self.notification_callback:
                await self.notification_callback(message)
            else:
                logger.warning(f"IP changed but no notification callback available: {message}")
                
        except Exception as e:
            logger.error(f"Error handling IP change: {str(e)}")
            
    def _create_ip_change_message(self, old_ip: str, new_ip: str) -> str:
        """Create IP change notification message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"""
🚨 **IP ADDRESS CHANGE DETECTED** 🚨

**Time:** {timestamp}
**Old IP:** {old_ip}
**New IP:** {new_ip}

⚠️ **Action Required:**
1. Log into your Binance account
2. Go to API Management
3. Add the new IP address to whitelist: **{new_ip}**
4. Remove old IP if no longer needed: **{old_ip}**

🔗 **Quick Links:**
• Binance API Management: https://www.binance.com/en/my/settings/api-management
• IP Check: https://whatismyipaddress.com/

The bot will resume trading once the new IP is whitelisted.
"""
        return message.strip()
        
    async def get_current_ip(self) -> Optional[str]:
        """Get current IP address."""
        return self._current_ip
        
    async def force_ip_check(self) -> Optional[str]:
        """Force an IP check and return current IP."""
        try:
            new_ip = await self._get_current_ip()
            if new_ip and new_ip != self._current_ip:
                await self._handle_ip_change(self._current_ip, new_ip)
                self._current_ip = new_ip
            return new_ip
        except Exception as e:
            logger.error(f"Error in force IP check: {str(e)}")
            return None
            
    def is_monitoring(self) -> bool:
        """Check if monitoring is active."""
        return self._is_running
        
    async def close(self) -> None:
        """Close the IP monitor service."""
        await self.stop_monitoring() 