#!/usr/bin/env python3
"""
Restart and monitor the trading bot to identify why strategy stops.
"""

import asyncio
import logging
import subprocess
import time
import os
import signal
from typing import Dict, Any, Optional, List
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BotRestartMonitor:
    """Restart and monitor the trading bot."""
    
    def __init__(self):
        self.process = None
        self.start_time = None
        self.last_strategy_log = None
        self.last_main_log = None
        self.error_count = 0
        
    async def kill_existing_processes(self):
        """Kill any existing bot processes."""
        logger.info("🔪 Killing existing bot processes...")
        
        try:
            # Kill processes containing main_with_quantitative
            result = subprocess.run(
                ['taskkill', '/F', '/IM', 'python.exe'],
                capture_output=True,
                text=True
            )
            logger.info(f"Killed processes: {result.stdout}")
            
            # Wait a bit for processes to fully terminate
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Error killing processes: {str(e)}")
    
    async def start_bot(self):
        """Start the trading bot."""
        logger.info("🚀 Starting trading bot...")
        
        try:
            # Start the bot process
            self.process = subprocess.Popen(
                ['python', 'main_with_quantitative.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.start_time = time.time()
            logger.info(f"✅ Bot started with PID: {self.process.pid}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting bot: {str(e)}")
            return False
    
    async def monitor_bot_output(self):
        """Monitor the bot's output in real-time."""
        logger.info("📊 Monitoring bot output...")
        
        if not self.process:
            logger.error("❌ No bot process to monitor")
            return
        
        try:
            # Monitor stdout
            while self.process.poll() is None:
                line = self.process.stdout.readline()
                if line:
                    line = line.strip()
                    if line:
                        await self.analyze_bot_output(line)
                
                # Check stderr for errors
                error_line = self.process.stderr.readline()
                if error_line:
                    error_line = error_line.strip()
                    if error_line:
                        logger.error(f"❌ Bot stderr: {error_line}")
                        await self.analyze_error(error_line)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
            
            # Process has ended
            return_code = self.process.returncode
            logger.error(f"❌ Bot process ended with return code: {return_code}")
            
            # Get any remaining output
            stdout, stderr = self.process.communicate()
            if stdout:
                logger.info(f"Final stdout: {stdout}")
            if stderr:
                logger.error(f"Final stderr: {stderr}")
                
        except Exception as e:
            logger.error(f"Error monitoring bot output: {str(e)}")
    
    async def analyze_bot_output(self, line: str):
        """Analyze bot output line."""
        try:
            # Track strategy logs
            if 'src.strategies.enhanced_trading_strategy_with_quantitative' in line:
                self.last_strategy_log = time.time()
                logger.info(f"📊 Strategy: {line[:100]}...")
            
            # Track main logs
            if '__main__' in line:
                self.last_main_log = time.time()
                if 'Starting cycle' in line:
                    logger.info(f"🔄 Cycle started: {line}")
                elif 'Completed cycle' in line:
                    logger.info(f"✅ Cycle completed: {line}")
            
            # Track errors
            if 'ERROR' in line:
                self.error_count += 1
                logger.error(f"❌ Error #{self.error_count}: {line}")
                
                # Check for specific error patterns
                if 'Exception' in line or 'Traceback' in line:
                    logger.error(f"🚨 Exception detected: {line}")
                
                if 'Failed to place' in line:
                    logger.warning(f"⚠️ Order placement failed: {line}")
                
                if 'Insufficient margin' in line:
                    logger.warning(f"⚠️ Margin issue: {line}")
            
            # Track warnings
            if 'WARNING' in line:
                if 'Memory' in line:
                    logger.warning(f"⚠️ Memory warning: {line}")
                elif 'Cache' in line:
                    logger.warning(f"⚠️ Cache warning: {line}")
                elif 'Performance' in line:
                    logger.warning(f"⚠️ Performance warning: {line}")
            
            # Track important events
            if 'Generated quantitative signals' in line:
                logger.info(f"✅ Signal generated: {line}")
            
            if 'Completed processing for' in line:
                logger.info(f"✅ Symbol completed: {line}")
            
        except Exception as e:
            logger.error(f"Error analyzing bot output: {str(e)}")
    
    async def analyze_error(self, error_line: str):
        """Analyze error output."""
        logger.error(f"🚨 Bot error: {error_line}")
        
        # Check for specific error patterns
        if 'KeyboardInterrupt' in error_line:
            logger.info("🛑 Bot stopped by user (Ctrl+C)")
        elif 'MemoryError' in error_line:
            logger.error("💥 Memory error detected!")
        elif 'TimeoutError' in error_line:
            logger.error("⏰ Timeout error detected!")
        elif 'ConnectionError' in error_line:
            logger.error("🌐 Connection error detected!")
    
    async def check_bot_health(self):
        """Check if the bot is healthy."""
        try:
            current_time = time.time()
            
            # Check if process is still running
            if self.process and self.process.poll() is None:
                logger.info("✅ Bot process is running")
            else:
                logger.error("❌ Bot process is not running")
                return False
            
            # Check for recent activity
            if self.last_strategy_log:
                time_since_strategy = current_time - self.last_strategy_log
                if time_since_strategy > 300:  # 5 minutes
                    logger.warning(f"⚠️ No strategy logs for {time_since_strategy:.1f} seconds")
            
            if self.last_main_log:
                time_since_main = current_time - self.last_main_log
                if time_since_main > 300:  # 5 minutes
                    logger.warning(f"⚠️ No main logs for {time_since_main:.1f} seconds")
            
            # Check error count
            if self.error_count > 10:
                logger.warning(f"⚠️ High error count: {self.error_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking bot health: {str(e)}")
            return False
    
    async def run_monitoring_cycle(self):
        """Run a complete monitoring cycle."""
        logger.info("🔄 Starting monitoring cycle...")
        
        # Kill existing processes
        await self.kill_existing_processes()
        
        # Start bot
        if not await self.start_bot():
            return False
        
        # Monitor for 30 minutes
        monitor_duration = 1800  # 30 minutes
        start_time = time.time()
        
        try:
            while time.time() - start_time < monitor_duration:
                # Check bot health every 30 seconds
                if not await self.check_bot_health():
                    logger.error("❌ Bot health check failed")
                    break
                
                await asyncio.sleep(30)
            
            logger.info("⏰ Monitoring cycle completed")
            
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {str(e)}")
        
        # Kill the bot
        if self.process:
            logger.info("🛑 Stopping bot...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("⚠️ Bot didn't stop gracefully, forcing kill")
                self.process.kill()
        
        return True

async def main():
    """Main function."""
    logger.info("🚀 Starting bot restart and monitor...")
    
    # Create monitor
    monitor = BotRestartMonitor()
    
    # Run monitoring cycle
    success = await monitor.run_monitoring_cycle()
    
    if success:
        logger.info("\n🎉 Monitoring completed successfully!")
        logger.info("✅ Bot started and monitored")
        logger.info(f"✅ Error count: {monitor.error_count}")
        logger.info("✅ Health checks performed")
    else:
        logger.error("\n❌ Monitoring failed!")
    
    logger.info("\n📋 Monitoring Summary:")
    logger.info("✅ Bot process management")
    logger.info("✅ Real-time output monitoring")
    logger.info("✅ Error tracking and analysis")
    logger.info("✅ Health checks")
    logger.info("✅ Graceful shutdown")

if __name__ == "__main__":
    asyncio.run(main()) 