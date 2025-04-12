"""
Circuit breaker utility for handling API failures.
"""

import time
from typing import Optional

class CircuitBreaker:
    """
    Circuit breaker pattern implementation for handling API failures.
    """
    
    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 60):
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of consecutive failures before opening the circuit
            reset_timeout: Time in seconds to wait before attempting to reset the circuit
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.is_open = False
        
    def record_failure(self) -> None:
        """Record a failure and update circuit state."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            
    def record_success(self) -> None:
        """Record a success and reset failure count."""
        self.failure_count = 0
        self.is_open = False
        
    def can_execute(self) -> bool:
        """
        Check if the circuit is closed or can be reset.
        
        Returns:
            bool: True if the circuit is closed or can be reset, False otherwise
        """
        if not self.is_open:
            return True
            
        if self.last_failure_time is None:
            return True
            
        if time.time() - self.last_failure_time >= self.reset_timeout:
            self.is_open = False
            self.failure_count = 0
            return True
            
        return False 