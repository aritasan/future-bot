#!/usr/bin/env python3
"""
Monitor strategy process to identify why it stopped.
"""

import asyncio
import logging
import time
import os
import signal
from typing import Dict, Any, Optional, List
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyMonitor:
    """Monitor strategy process and identify issues."""
    
    def __init__(self):
        self.last_strategy_log = None
        self.last_main_log = None
        self.error_count = 0
        self.memory_usage = []
        
    async def monitor_log_file(self, log_file: str):
        """Monitor log file for strategy activity."""
        logger.info(f"🔍 Monitoring log file: {log_file}")
        
        if not os.path.exists(log_file):
            logger.error(f"Log file not found: {log_file}")
            return
        
        # Get initial file size
        initial_size = os.path.getsize(log_file)
        logger.info(f"Initial log file size: {initial_size} bytes")
        
        # Monitor for changes
        last_size = initial_size
        last_check = time.time()
        
        while True:
            try:
                # Check if file exists and has changed
                if not os.path.exists(log_file):
                    logger.error(f"Log file disappeared: {log_file}")
                    break
                
                current_size = os.path.getsize(log_file)
                current_time = time.time()
                
                # Check for new content
                if current_size > last_size:
                    # Read new content
                    with open(log_file, 'r', encoding='utf-8') as f:
                        f.seek(last_size)
                        new_content = f.read()
                        
                    # Parse new lines
                    new_lines = new_content.strip().split('\n')
                    for line in new_lines:
                        if line.strip():
                            await self.analyze_log_line(line)
                    
                    last_size = current_size
                    last_check = current_time
                
                # Check for inactivity (no new logs for 5 minutes)
                if current_time - last_check > 300:  # 5 minutes
                    logger.warning(f"⚠️ No new logs for 5 minutes! Last activity: {time.strftime('%H:%M:%S', time.localtime(last_check))}")
                    
                    # Check if strategy is still active
                    if self.last_strategy_log:
                        time_since_strategy = current_time - self.last_strategy_log
                        logger.warning(f"⚠️ Strategy log inactive for {time_since_strategy:.1f} seconds")
                    
                    if self.last_main_log:
                        time_since_main = current_time - self.last_main_log
                        logger.warning(f"⚠️ Main log inactive for {time_since_main:.1f} seconds")
                
                # Check memory usage
                await self.check_memory_usage()
                
                # Sleep for 10 seconds
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error monitoring log file: {str(e)}")
                await asyncio.sleep(30)
    
    async def analyze_log_line(self, line: str):
        """Analyze a log line for important events."""
        try:
            # Extract timestamp and content
            if ' - ' in line:
                parts = line.split(' - ', 2)
                if len(parts) >= 3:
                    timestamp_str = parts[0]
                    logger_name = parts[1]
                    message = parts[2]
                    
                    # Parse timestamp
                    try:
                        timestamp = time.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        timestamp_float = time.mktime(timestamp)
                    except:
                        return
                    
                    # Track strategy logs
                    if 'src.strategies.enhanced_trading_strategy_with_quantitative' in logger_name:
                        self.last_strategy_log = timestamp_float
                        logger.info(f"📊 Strategy log: {message[:100]}...")
                    
                    # Track main logs
                    if '__main__' in logger_name:
                        self.last_main_log = timestamp_float
                        if 'Starting cycle' in message:
                            logger.info(f"🔄 Cycle started: {message}")
                        elif 'Completed cycle' in message:
                            logger.info(f"✅ Cycle completed: {message}")
                    
                    # Track errors
                    if 'ERROR' in line:
                        self.error_count += 1
                        logger.error(f"❌ Error #{self.error_count}: {message}")
                        
                        # Check for specific error patterns
                        if 'Exception' in message or 'Traceback' in message:
                            logger.error(f"🚨 Exception detected: {message}")
                        
                        if 'Failed to place' in message:
                            logger.warning(f"⚠️ Order placement failed: {message}")
                        
                        if 'Insufficient margin' in message:
                            logger.warning(f"⚠️ Margin issue: {message}")
                    
                    # Track warnings
                    if 'WARNING' in line:
                        if 'Memory' in message:
                            logger.warning(f"⚠️ Memory warning: {message}")
                        elif 'Cache' in message:
                            logger.warning(f"⚠️ Cache warning: {message}")
                        elif 'Performance' in message:
                            logger.warning(f"⚠️ Performance warning: {message}")
                    
        except Exception as e:
            logger.error(f"Error analyzing log line: {str(e)}")
    
    async def check_memory_usage(self):
        """Check memory usage of the process."""
        try:
            import psutil
            
            # Find Python processes
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                if 'python' in proc.info['name'].lower():
                    try:
                        memory_mb = proc.info['memory_info'].rss / 1024 / 1024
                        self.memory_usage.append(memory_mb)
                        
                        # Keep only last 10 measurements
                        if len(self.memory_usage) > 10:
                            self.memory_usage.pop(0)
                        
                        # Check for high memory usage
                        if memory_mb > 500:  # 500MB
                            logger.warning(f"⚠️ High memory usage: {memory_mb:.1f} MB")
                        
                        # Check for memory growth
                        if len(self.memory_usage) >= 2:
                            growth = self.memory_usage[-1] - self.memory_usage[0]
                            if growth > 100:  # 100MB growth
                                logger.warning(f"⚠️ Memory growth detected: {growth:.1f} MB")
                        
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                        
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
        except Exception as e:
            logger.error(f"Error checking memory usage: {str(e)}")
    
    async def check_process_health(self):
        """Check if the main process is still running."""
        try:
            import psutil
            
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if 'python' in proc.info['name'].lower():
                    cmdline = proc.info['cmdline']
                    if cmdline and any('main_with_quantitative' in arg for arg in cmdline):
                        python_processes.append(proc)
            
            if not python_processes:
                logger.error("❌ No main_with_quantitative.py processes found!")
                return False
            
            logger.info(f"✅ Found {len(python_processes)} main_with_quantitative.py processes")
            
            for proc in python_processes:
                try:
                    # Check if process is responsive
                    if proc.is_running():
                        logger.info(f"✅ Process {proc.pid} is running")
                    else:
                        logger.error(f"❌ Process {proc.pid} is not running")
                        return False
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    logger.error(f"❌ Cannot access process {proc.pid}")
                    return False
            
            return True
            
        except ImportError:
            logger.warning("psutil not available for process monitoring")
            return True
        except Exception as e:
            logger.error(f"Error checking process health: {str(e)}")
            return False

async def main():
    """Main monitoring function."""
    logger.info("🚀 Starting strategy process monitor...")
    
    # Create monitor
    monitor = StrategyMonitor()
    
    # Monitor log file
    log_file = "logs/trading_bot_quantitative_20250802.log"
    
    # Start monitoring tasks
    tasks = [
        monitor.monitor_log_file(log_file),
        monitor.check_process_health()
    ]
    
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Monitoring error: {str(e)}")
    
    logger.info("📋 Monitoring Summary:")
    logger.info(f"✅ Strategy logs tracked")
    logger.info(f"✅ Error count: {monitor.error_count}")
    logger.info(f"✅ Memory usage monitored")
    logger.info(f"✅ Process health checked")

if __name__ == "__main__":
    asyncio.run(main()) 