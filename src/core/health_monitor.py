import logging
import psutil
import asyncio
from typing import Dict

class HealthMonitor:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.error_count = 0
        self.max_errors = 10
        self.error_rate = 0.0
        self.window_size = 100
        self.error_window = []
        self.is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize the health monitor."""
        try:
            # Test memory check
            memory_usage = self.check_memory_usage()
            if not memory_usage:
                raise Exception("Failed to check memory usage")
                
            self.is_initialized = True
            self.logger.info("Health monitor initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing health monitor: {e}")
            raise
            
    async def check_health(self) -> bool:
        """Check overall system health."""
        try:
            if not self.is_initialized:
                return False
                
            # Check API health
            api_health = await self.check_api_health()
            if not api_health:
                self.logger.warning("API health check failed")
                return False
                
            # Check WebSocket health
            ws_health = await self.check_websocket_health()
            if not ws_health:
                self.logger.warning("WebSocket health check failed")
                return False
                
            # Check memory usage
            memory_usage = self.check_memory_usage()
            if not memory_usage:
                self.logger.warning("Memory check failed")
                return False
                
            # Check against thresholds
            if memory_usage['percent'] > self.config['health']['memory_threshold']:
                self.logger.warning(f"Memory usage {memory_usage['percent']}% exceeds threshold")
                return False
                
            return True
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.logger.error(f"Health check error: {e}")
            return False
            
    async def check_api_health(self) -> bool:
        """Kiểm tra sức khỏe của API"""
        try:
            # Thêm logic kiểm tra API ở đây
            return True
        except Exception as e:
            self.logger.error(f"API health check failed: {e}")
            return False
            
    async def check_websocket_health(self) -> bool:
        """Kiểm tra sức khỏe của WebSocket"""
        try:
            # Thêm logic kiểm tra WebSocket ở đây
            return True
        except Exception as e:
            self.logger.error(f"WebSocket health check failed: {e}")
            return False
            
    def check_memory_usage(self) -> Dict:
        """Kiểm tra sử dụng bộ nhớ"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss": memory_info.rss / 1024 / 1024,  # MB
                "vms": memory_info.vms / 1024 / 1024,  # MB
                "percent": process.memory_percent()
            }
        except Exception as e:
            self.logger.error(f"Memory check failed: {e}")
            return {}
            
    def update_error_rate(self, success: bool):
        """Cập nhật tỷ lệ lỗi"""
        self.error_window.append(not success)
        if len(self.error_window) > self.window_size:
            self.error_window.pop(0)
        self.error_rate = sum(self.error_window) / len(self.error_window)
        
    async def run_health_check(self):
        """Chạy kiểm tra sức khỏe định kỳ"""
        while True:
            try:
                api_health = await self.check_api_health()
                ws_health = await self.check_websocket_health()
                memory_usage = self.check_memory_usage()
                
                if not api_health or not ws_health:
                    self.error_count += 1
                    if self.error_count >= self.max_errors:
                        self.logger.error("Too many health check failures, stopping bot")
                        # Thêm logic dừng bot ở đây
                        break
                        
                self.update_error_rate(api_health and ws_health)
                
                # Log health metrics
                self.logger.info(f"Health check: API={api_health}, WS={ws_health}, Memory={memory_usage}")
                
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                
            await asyncio.sleep(60)  # Kiểm tra mỗi phút 