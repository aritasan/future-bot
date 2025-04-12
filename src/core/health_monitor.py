"""
Service for monitoring bot health and performance.
"""
import logging
from typing import Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class HealthMonitor:
    """Service for monitoring bot health and performance."""
    
    def __init__(self, config: Dict):
        """Initialize the service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._is_initialized = False
        self._is_closed = False
        self.error_count = 0
        self.max_errors = config.get('health_monitor', {}).get('max_errors', 10)
        self.error_window = timedelta(minutes=5)
        self.error_timestamps = []
        self.performance_metrics = {}
        self.last_check = None
        
    async def initialize(self) -> bool:
        """Initialize the health monitor.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            if self._is_initialized:
                logger.warning("Health monitor already initialized")
                return True
                
            # Initialize metrics
            self.performance_metrics = {
                'total_orders': 0,
                'successful_orders': 0,
                'failed_orders': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0
            }
            
            self._is_initialized = True
            logger.info("Health monitor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize health monitor: {str(e)}")
            return False
            
    async def check_health(self) -> bool:
        """Check if the bot is healthy.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            if not self._is_initialized:
                logger.error("Health monitor not initialized")
                return False
                
            if self._is_closed:
                logger.error("Health monitor is closed")
                return False
                
            # Clean up old error timestamps
            current_time = datetime.now()
            self.error_timestamps = [
                ts for ts in self.error_timestamps
                if current_time - ts < self.error_window
            ]
            
            # Check error count
            if len(self.error_timestamps) >= self.max_errors:
                logger.error(f"Too many errors in the last {self.error_window}")
                return False
                
            self.last_check = current_time
            return True
            
        except Exception as e:
            logger.error(f"Error checking health: {str(e)}")
            return False
            
    def record_error(self):
        """Record an error occurrence."""
        try:
            if not self._is_initialized:
                logger.error("Health monitor not initialized")
                return
                
            if self._is_closed:
                logger.error("Health monitor is closed")
                return
                
            self.error_timestamps.append(datetime.now())
            self.error_count += 1
            
        except Exception as e:
            logger.error(f"Error recording error: {str(e)}")
            
    def record_order(self, success: bool):
        """Record an order execution.
        
        Args:
            success: Whether the order was successful
        """
        try:
            if not self._is_initialized:
                logger.error("Health monitor not initialized")
                return
                
            if self._is_closed:
                logger.error("Health monitor is closed")
                return
                
            self.performance_metrics['total_orders'] += 1
            if success:
                self.performance_metrics['successful_orders'] += 1
            else:
                self.performance_metrics['failed_orders'] += 1
                
        except Exception as e:
            logger.error(f"Error recording order: {str(e)}")
            
    def record_trade(self, pnl: float):
        """Record a trade execution.
        
        Args:
            pnl: Profit and loss of the trade
        """
        try:
            if not self._is_initialized:
                logger.error("Health monitor not initialized")
                return
                
            if self._is_closed:
                logger.error("Health monitor is closed")
                return
                
            self.performance_metrics['total_trades'] += 1
            if pnl > 0:
                self.performance_metrics['winning_trades'] += 1
            else:
                self.performance_metrics['losing_trades'] += 1
                
            self.performance_metrics['total_pnl'] += pnl
            
        except Exception as e:
            logger.error(f"Error recording trade: {str(e)}")
            
    def get_metrics(self) -> Dict:
        """Get current performance metrics.
        
        Returns:
            Dict: Performance metrics
        """
        try:
            if not self._is_initialized:
                logger.error("Health monitor not initialized")
                return {}
                
            if self._is_closed:
                logger.error("Health monitor is closed")
                return {}
                
            return self.performance_metrics.copy()
            
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return {}
            
    async def close(self):
        """Close the health monitor."""
        try:
            if not self._is_initialized:
                logger.warning("Health monitor was not initialized")
                return
                
            if self._is_closed:
                logger.warning("Health monitor already closed")
                return
                
            self._is_closed = True
            logger.info("Health monitor closed")
            
        except Exception as e:
            logger.error(f"Error closing health monitor: {str(e)}") 