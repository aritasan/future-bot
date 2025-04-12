"""
API Manager with circuit breaker pattern implementation.
"""
import logging
import time
from typing import Dict, Optional, List
import aiohttp
import asyncio

logger = logging.getLogger(__name__)

class APIManager:
    """API Manager with circuit breaker pattern."""
    
    def __init__(self, config: Dict):
        """Initialize API Manager.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Circuit breaker settings
        self.failure_threshold = config.get('circuit_breaker', {}).get('failure_threshold', 5)
        self.reset_timeout = config.get('circuit_breaker', {}).get('reset_timeout', 60)
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        
        # Rate limiting settings
        self.rate_limit = config.get('rate_limit', {}).get('requests_per_second', 10)
        self.rate_limit_window = config.get('rate_limit', {}).get('window_seconds', 1)
        self.request_times: List[float] = []
        
        # Session management
        self.session = None
        self.session_timeout = config.get('session_timeout', 30)
        self.last_session_time = None
        
    async def initialize(self):
        """Initialize API manager."""
        try:
            # Create aiohttp session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.session_timeout)
            )
            self.last_session_time = time.time()
            self.logger.info("API Manager initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing API Manager: {str(e)}")
            raise
            
    async def close(self):
        """Close API manager."""
        try:
            if self.session:
                await self.session.close()
            self.logger.info("API Manager closed")
            
        except Exception as e:
            self.logger.error(f"Error closing API Manager: {str(e)}")
            
    async def _check_circuit_breaker(self) -> bool:
        """Check circuit breaker state.
        
        Returns:
            bool: True if request can proceed, False otherwise
        """
        try:
            current_time = time.time()
            
            # Check if circuit breaker is open
            if self.state == "OPEN":
                # Check if reset timeout has passed
                if (self.last_failure_time and 
                    current_time - self.last_failure_time >= self.reset_timeout):
                    self.state = "HALF-OPEN"
                    self.logger.info("Circuit breaker moved to HALF-OPEN state")
                else:
                    return False
                    
            # Check if circuit breaker is half-open
            if self.state == "HALF-OPEN":
                # Allow one request to test
                self.state = "CLOSED"
                self.logger.info("Circuit breaker moved to CLOSED state")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking circuit breaker: {str(e)}")
            return False
            
    async def _update_circuit_breaker(self, success: bool):
        """Update circuit breaker state.
        
        Args:
            success: Whether the request was successful
        """
        try:
            if success:
                # Reset failure count on success
                self.failure_count = 0
                self.state = "CLOSED"
            else:
                # Increment failure count
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                # Check if threshold reached
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    self.logger.warning("Circuit breaker opened due to too many failures")
                    
        except Exception as e:
            self.logger.error(f"Error updating circuit breaker: {str(e)}")
            
    async def _check_rate_limit(self) -> bool:
        """Check rate limit.
        
        Returns:
            bool: True if request can proceed, False otherwise
        """
        try:
            current_time = time.time()
            
            # Remove old request times
            self.request_times = [t for t in self.request_times 
                                if current_time - t < self.rate_limit_window]
            
            # Check if rate limit reached
            if len(self.request_times) >= self.rate_limit:
                return False
                
            # Add current request time
            self.request_times.append(current_time)
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking rate limit: {str(e)}")
            return False
            
    async def _check_session(self):
        """Check and refresh session if needed."""
        try:
            current_time = time.time()
            
            # Check if session needs refresh
            if (self.last_session_time and 
                current_time - self.last_session_time >= self.session_timeout):
                await self.close()
                await self.initialize()
                
        except Exception as e:
            self.logger.error(f"Error checking session: {str(e)}")
            
    async def get(self, url: str, headers: Optional[Dict] = None, 
                 params: Optional[Dict] = None) -> Optional[Dict]:
        """Make GET request.
        
        Args:
            url: Request URL
            headers: Request headers
            params: Request parameters
            
        Returns:
            Optional[Dict]: Response data
        """
        try:
            # Check circuit breaker
            if not await self._check_circuit_breaker():
                self.logger.warning("Request blocked by circuit breaker")
                return None
                
            # Check rate limit
            if not await self._check_rate_limit():
                self.logger.warning("Request blocked by rate limit")
                await asyncio.sleep(1)  # Wait before retrying
                return await self.get(url, headers, params)
                
            # Check session
            await self._check_session()
            
            # Make request
            async with self.session.get(url, headers=headers, params=params) as response:
                success = response.status == 200
                await self._update_circuit_breaker(success)
                
                if success:
                    return await response.json()
                else:
                    self.logger.error(f"GET request failed: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error in GET request: {str(e)}")
            await self._update_circuit_breaker(False)
            return None
            
    async def post(self, url: str, headers: Optional[Dict] = None, 
                  data: Optional[Dict] = None) -> Optional[Dict]:
        """Make POST request.
        
        Args:
            url: Request URL
            headers: Request headers
            data: Request data
            
        Returns:
            Optional[Dict]: Response data
        """
        try:
            # Check circuit breaker
            if not await self._check_circuit_breaker():
                self.logger.warning("Request blocked by circuit breaker")
                return None
                
            # Check rate limit
            if not await self._check_rate_limit():
                self.logger.warning("Request blocked by rate limit")
                await asyncio.sleep(1)  # Wait before retrying
                return await self.post(url, headers, data)
                
            # Check session
            await self._check_session()
            
            # Make request
            async with self.session.post(url, headers=headers, json=data) as response:
                success = response.status == 200
                await self._update_circuit_breaker(success)
                
                if success:
                    return await response.json()
                else:
                    self.logger.error(f"POST request failed: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error in POST request: {str(e)}")
            await self._update_circuit_breaker(False)
            return None
            
    async def put(self, url: str, headers: Optional[Dict] = None, 
                 data: Optional[Dict] = None) -> Optional[Dict]:
        """Make PUT request.
        
        Args:
            url: Request URL
            headers: Request headers
            data: Request data
            
        Returns:
            Optional[Dict]: Response data
        """
        try:
            # Check circuit breaker
            if not await self._check_circuit_breaker():
                self.logger.warning("Request blocked by circuit breaker")
                return None
                
            # Check rate limit
            if not await self._check_rate_limit():
                self.logger.warning("Request blocked by rate limit")
                await asyncio.sleep(1)  # Wait before retrying
                return await self.put(url, headers, data)
                
            # Check session
            await self._check_session()
            
            # Make request
            async with self.session.put(url, headers=headers, json=data) as response:
                success = response.status == 200
                await self._update_circuit_breaker(success)
                
                if success:
                    return await response.json()
                else:
                    self.logger.error(f"PUT request failed: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error in PUT request: {str(e)}")
            await self._update_circuit_breaker(False)
            return None
            
    async def delete(self, url: str, headers: Optional[Dict] = None) -> bool:
        """Make DELETE request.
        
        Args:
            url: Request URL
            headers: Request headers
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check circuit breaker
            if not await self._check_circuit_breaker():
                self.logger.warning("Request blocked by circuit breaker")
                return False
                
            # Check rate limit
            if not await self._check_rate_limit():
                self.logger.warning("Request blocked by rate limit")
                await asyncio.sleep(1)  # Wait before retrying
                return await self.delete(url, headers)
                
            # Check session
            await self._check_session()
            
            # Make request
            async with self.session.delete(url, headers=headers) as response:
                success = response.status == 200
                await self._update_circuit_breaker(success)
                
                if success:
                    return True
                else:
                    self.logger.error(f"DELETE request failed: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error in DELETE request: {str(e)}")
            await self._update_circuit_breaker(False)
            return False
            
    def get_state(self) -> Dict:
        """Get current state of API manager.
        
        Returns:
            Dict: Current state
        """
        return {
            "circuit_breaker": {
                "state": self.state,
                "failure_count": self.failure_count,
                "last_failure_time": self.last_failure_time
            },
            "rate_limit": {
                "current_requests": len(self.request_times),
                "limit": self.rate_limit
            },
            "session": {
                "active": self.session is not None,
                "last_activity": self.last_session_time
            }
        } 