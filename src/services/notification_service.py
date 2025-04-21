"""
Service for handling notifications and alerts.
"""
import logging
from typing import Dict
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class NotificationService:
    """Service for handling notifications and alerts."""
    
    def __init__(self, config: Dict):
        """Initialize the notification service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._is_initialized = False
        self._is_closed = False
        self._notification_queue = asyncio.Queue()
        self._processing_task = None
        self._last_notification = {}
        self._notification_cooldown = 60  # Minimum 60 seconds between notifications
        self._max_notifications_per_hour = 10  # Maximum 10 notifications per hour
        
    async def initialize(self) -> bool:
        """Initialize the notification service.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            if self._is_initialized:
                logger.warning("Notification service already initialized")
                return True
                
            # Start notification processing task
            self._processing_task = asyncio.create_task(self._process_notifications())
            
            self._is_initialized = True
            logger.info("Notification service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing notification service: {str(e)}")
            return False
            
    async def _process_notifications(self) -> None:
        """Process notifications from the queue."""
        try:
            while not self._is_closed:
                try:
                    # Get notification from queue
                    notification = await self._notification_queue.get()
                    
                    # Check notification cooldown
                    if not await self._check_notification_cooldown(notification['type']):
                        logger.warning(f"Notification cooldown active for {notification['type']}")
                        continue
                        
                    # Process notification based on type
                    if notification['type'] == 'error':
                        await self._handle_error_notification(notification)
                    elif notification['type'] == 'trade':
                        await self._handle_trade_notification(notification)
                    elif notification['type'] == 'status':
                        await self._handle_status_notification(notification)
                    else:
                        logger.warning(f"Unknown notification type: {notification['type']}")
                        
                    # Mark task as done
                    self._notification_queue.task_done()
                    
                except asyncio.CancelledError:
                    logger.info("Notification processing task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error processing notification: {str(e)}")
                    await asyncio.sleep(1)  # Wait before retrying
                    
        except Exception as e:
            logger.error(f"Error in notification processing task: {str(e)}")
            
    async def _check_notification_cooldown(self, notification_type: str) -> bool:
        """Check if notification can be sent based on cooldown.
        
        Args:
            notification_type: Type of notification
            
        Returns:
            bool: True if notification can be sent
        """
        try:
            current_time = datetime.now()
            
            # Initialize notification history if not exists
            if notification_type not in self._last_notification:
                self._last_notification[notification_type] = []
                
            # Remove notifications older than 1 hour
            self._last_notification[notification_type] = [
                time for time in self._last_notification[notification_type]
                if (current_time - time).total_seconds() < 3600
            ]
            
            # Check if we have reached max notifications per hour
            if len(self._last_notification[notification_type]) >= self._max_notifications_per_hour:
                logger.warning(f"Maximum notifications per hour reached for {notification_type}")
                return False
                
            # Check minimum interval since last notification
            if self._last_notification[notification_type]:
                last_notification = self._last_notification[notification_type][-1]
                if (current_time - last_notification).total_seconds() < self._notification_cooldown:
                    logger.warning(f"Notification cooldown not met for {notification_type}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking notification cooldown: {str(e)}")
            return False
            
    async def _handle_error_notification(self, notification: Dict) -> None:
        """Handle error notification.
        
        Args:
            notification: Notification details
        """
        try:
            error_message = notification.get('message', 'Unknown error')
            error_details = notification.get('details', {})
            
            # Log error
            logger.error(f"Error notification: {error_message}")
            if error_details:
                logger.error(f"Error details: {error_details}")
                
            # Update notification history
            self._last_notification['error'].append(datetime.now())
            
        except Exception as e:
            logger.error(f"Error handling error notification: {str(e)}")
            
    async def _handle_trade_notification(self, notification: Dict) -> None:
        """Handle trade notification.
        
        Args:
            notification: Notification details
        """
        try:
            symbol = notification.get('symbol')
            side = notification.get('side')
            amount = notification.get('amount')
            price = notification.get('price')
            
            # Log trade
            logger.info(f"Trade notification: {symbol} {side} {amount} @ {price}")
            
            # Update notification history
            self._last_notification['trade'].append(datetime.now())
            
        except Exception as e:
            logger.error(f"Error handling trade notification: {str(e)}")
            
    async def _handle_status_notification(self, notification: Dict) -> None:
        """Handle status notification.
        
        Args:
            notification: Notification details
        """
        try:
            status = notification.get('status')
            details = notification.get('details', {})
            
            # Log status
            logger.info(f"Status notification: {status}")
            if details:
                logger.info(f"Status details: {details}")
                
            # Update notification history
            self._last_notification['status'].append(datetime.now())
            
        except Exception as e:
            logger.error(f"Error handling status notification: {str(e)}")
            
    async def send_notification(self, notification_type: str, message: str, details: Dict = None) -> bool:
        """Send a notification.
        
        Args:
            notification_type: Type of notification
            message: Notification message
            details: Additional notification details
            
        Returns:
            bool: True if notification sent successfully
        """
        try:
            if not self._is_initialized:
                logger.error("Notification service not initialized")
                return False
                
            # Create notification
            notification = {
                'type': notification_type,
                'message': message,
                'details': details or {},
                'timestamp': datetime.now()
            }
            
            # Add to queue
            await self._notification_queue.put(notification)
            return True
            
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
            return False
            
    async def close(self) -> None:
        """Close the notification service."""
        try:
            if not self._is_initialized:
                return
                
            self._is_closed = True
            
            # Cancel processing task
            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    pass
                    
            # Clear notification history
            self._last_notification.clear()
            
            self._is_initialized = False
            logger.info("Notification service closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing notification service: {str(e)}")
            raise 