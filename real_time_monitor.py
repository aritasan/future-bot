#!/usr/bin/env python3
"""
Real-time monitoring script for the trading bot.
"""

import asyncio
import logging
import time
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeMonitor:
    """Real-time monitoring for the trading bot."""
    
    def __init__(self):
        self.last_strategy_log = None
        self.last_main_log = None
        self.error_count = 0
        self.health_checks = []
        
    async def monitor_logs(self):
        """Monitor log files in real-time."""
        log_file = "logs/trading_bot_quantitative_20250802.log"
        
        if not os.path.exists(log_file):
            logger.error(f"Log file not found: {log_file}")
            return
        
        # Get initial file size
        last_size = os.path.getsize(log_file)
        
        while True:
            try:
                # Check for new content
                current_size = os.path.getsize(log_file)
                
                if current_size > last_size:
                    # Read new content
                    with open(log_file, 'r', encoding='utf-8') as f:
                        f.seek(last_size)
                        new_content = f.read()
                    
                    # Process new lines
                    for line in new_content.strip().split('\n'):
                        if line.strip():
                            await self.analyze_log_line(line)
                    
                    last_size = current_size
                
                # Check for inactivity
                current_time = time.time()
                if self.last_strategy_log and current_time - self.last_strategy_log > 300:
                    logger.warning("⚠️ No strategy logs for 5 minutes!")
                    
                if self.last_main_log and current_time - self.last_main_log > 300:
                    logger.warning("⚠️ No main logs for 5 minutes!")
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error monitoring logs: {str(e)}")
                await asyncio.sleep(30)
    
    async def analyze_log_line(self, line: str):
        """Analyze a log line for important events."""
        try:
            if ' - ' in line:
                parts = line.split(' - ', 2)
                if len(parts) >= 3:
                    timestamp_str = parts[0]
                    logger_name = parts[1]
                    message = parts[2]
                    
                    # Track strategy logs
                    if 'src.strategies.enhanced_trading_strategy_with_quantitative' in logger_name:
                        self.last_strategy_log = time.time()
                        logger.info(f"📊 Strategy: {message[:100]}...")
                    
                    # Track main logs
                    if '__main__' in logger_name:
                        self.last_main_log = time.time()
                        if 'Starting cycle' in message:
                            logger.info(f"🔄 Cycle started: {message}")
                        elif 'Completed cycle' in message:
                            logger.info(f"✅ Cycle completed: {message}")
                    
                    # Track errors
                    if 'ERROR' in line:
                        self.error_count += 1
                        logger.error(f"❌ Error #{self.error_count}: {message}")
                    
                    # Track health checks
                    if 'Health check' in message:
                        self.health_checks.append({
                            'timestamp': time.time(),
                            'message': message
                        })
                        
                        # Keep only last 10 health checks
                        if len(self.health_checks) > 10:
                            self.health_checks.pop(0)
                    
        except Exception as e:
            logger.error(f"Error analyzing log line: {str(e)}")

async def main():
    """Main monitoring function."""
    logger.info("🚀 Starting real-time monitoring...")
    
    monitor = RealTimeMonitor()
    
    try:
        await monitor.monitor_logs()
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Monitoring error: {str(e)}")
    
    logger.info("📋 Monitoring Summary:")
    logger.info(f"✅ Error count: {monitor.error_count}")
    logger.info(f"✅ Health checks: {len(monitor.health_checks)}")

if __name__ == "__main__":
    asyncio.run(main())
