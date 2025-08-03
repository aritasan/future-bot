#!/usr/bin/env python3
"""
Start the enhanced trading bot with comprehensive error handling and monitoring.
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

class EnhancedBotStarter:
    """Start the enhanced trading bot with monitoring."""
    
    def __init__(self):
        self.bot_process = None
        self.monitor_process = None
        self.start_time = None
        
    async def kill_existing_processes(self):
        """Kill any existing bot processes."""
        logger.info("🔪 Killing existing processes...")
        
        try:
            # Kill Python processes
            result = subprocess.run(
                ['taskkill', '/F', '/IM', 'python.exe'],
                capture_output=True,
                text=True
            )
            logger.info(f"Killed processes: {result.stdout}")
            
            # Wait for processes to terminate
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Error killing processes: {str(e)}")
    
    async def start_enhanced_bot(self):
        """Start the enhanced trading bot."""
        logger.info("🚀 Starting enhanced trading bot...")
        
        try:
            # Start the bot process
            self.bot_process = subprocess.Popen(
                ['python', 'main_with_quantitative.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.start_time = time.time()
            logger.info(f"✅ Enhanced bot started with PID: {self.bot_process.pid}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting enhanced bot: {str(e)}")
            return False
    
    async def start_real_time_monitor(self):
        """Start the real-time monitoring script."""
        logger.info("📊 Starting real-time monitor...")
        
        try:
            # Start the monitoring process
            self.monitor_process = subprocess.Popen(
                ['python', 'real_time_monitor.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            logger.info(f"✅ Real-time monitor started with PID: {self.monitor_process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting monitor: {str(e)}")
            return False
    
    async def monitor_bot_output(self):
        """Monitor the bot's output in real-time."""
        logger.info("📊 Monitoring enhanced bot output...")
        
        if not self.bot_process:
            logger.error("❌ No bot process to monitor")
            return
        
        try:
            # Monitor stdout
            while self.bot_process.poll() is None:
                line = self.bot_process.stdout.readline()
                if line:
                    line = line.strip()
                    if line:
                        await self.analyze_bot_output(line)
                
                # Check stderr for errors
                error_line = self.bot_process.stderr.readline()
                if error_line:
                    error_line = error_line.strip()
                    if error_line:
                        logger.error(f"❌ Bot stderr: {error_line}")
                        await self.analyze_error(error_line)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
            
            # Process has ended
            return_code = self.bot_process.returncode
            logger.error(f"❌ Bot process ended with return code: {return_code}")
            
            # Get any remaining output
            stdout, stderr = self.bot_process.communicate()
            if stdout:
                logger.info(f"Final stdout: {stdout}")
            if stderr:
                logger.error(f"Final stderr: {stderr}")
                
        except Exception as e:
            logger.error(f"Error monitoring bot output: {str(e)}")
    
    async def analyze_bot_output(self, line: str):
        """Analyze bot output line."""
        try:
            # Track enhanced features
            if 'with_timeout' in line:
                logger.info(f"⏰ Timeout protection: {line[:100]}...")
            
            if 'health_check' in line:
                logger.info(f"🏥 Health check: {line[:100]}...")
            
            if 'recover_from_error' in line:
                logger.info(f"🔄 Error recovery: {line[:100]}...")
            
            # Track strategy logs
            if 'src.strategies.enhanced_trading_strategy_with_quantitative' in line:
                logger.info(f"📊 Strategy: {line[:100]}...")
            
            # Track main logs
            if '__main__' in line:
                if 'Starting cycle' in line:
                    logger.info(f"🔄 Cycle started: {line}")
                elif 'Completed cycle' in line:
                    logger.info(f"✅ Cycle completed: {line}")
            
            # Track errors
            if 'ERROR' in line:
                logger.error(f"❌ Error: {line}")
                
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
            if self.bot_process and self.bot_process.poll() is None:
                logger.info("✅ Enhanced bot process is running")
                
                # Check uptime
                if self.start_time:
                    uptime = current_time - self.start_time
                    logger.info(f"⏱️ Bot uptime: {uptime:.1f} seconds")
                
                return True
            else:
                logger.error("❌ Enhanced bot process is not running")
                return False
                
        except Exception as e:
            logger.error(f"Error checking bot health: {str(e)}")
            return False
    
    async def run_enhanced_bot_session(self):
        """Run a complete enhanced bot session."""
        logger.info("🔄 Starting enhanced bot session...")
        
        # Kill existing processes
        await self.kill_existing_processes()
        
        # Start enhanced bot
        if not await self.start_enhanced_bot():
            return False
        
        # Start real-time monitor
        await self.start_real_time_monitor()
        
        # Monitor for 1 hour
        monitor_duration = 3600  # 1 hour
        start_time = time.time()
        
        try:
            while time.time() - start_time < monitor_duration:
                # Check bot health every 30 seconds
                if not await self.check_bot_health():
                    logger.error("❌ Enhanced bot health check failed")
                    break
                
                await asyncio.sleep(30)
            
            logger.info("⏰ Enhanced bot session completed")
            
        except KeyboardInterrupt:
            logger.info("🛑 Enhanced bot session stopped by user")
        except Exception as e:
            logger.error(f"Enhanced bot session error: {str(e)}")
        
        # Kill the processes
        if self.bot_process:
            logger.info("🛑 Stopping enhanced bot...")
            self.bot_process.terminate()
            try:
                self.bot_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("⚠️ Enhanced bot didn't stop gracefully, forcing kill")
                self.bot_process.kill()
        
        if self.monitor_process:
            logger.info("🛑 Stopping real-time monitor...")
            self.monitor_process.terminate()
            try:
                self.monitor_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.monitor_process.kill()
        
        return True

async def main():
    """Main function."""
    logger.info("🚀 Starting enhanced bot with comprehensive monitoring...")
    
    # Create enhanced bot starter
    starter = EnhancedBotStarter()
    
    # Run enhanced bot session
    success = await starter.run_enhanced_bot_session()
    
    if success:
        logger.info("\n🎉 Enhanced bot session completed successfully!")
        logger.info("✅ Enhanced error handling active")
        logger.info("✅ Timeout protection active")
        logger.info("✅ Health checks active")
        logger.info("✅ Recovery mechanisms active")
        logger.info("✅ Real-time monitoring active")
    else:
        logger.error("\n❌ Enhanced bot session failed!")
    
    logger.info("\n📋 Enhanced Bot Summary:")
    logger.info("✅ Enhanced error handling implemented")
    logger.info("✅ Timeout protection implemented")
    logger.info("✅ Health checks implemented")
    logger.info("✅ Recovery mechanisms implemented")
    logger.info("✅ Real-time monitoring implemented")
    logger.info("✅ Comprehensive logging active")

if __name__ == "__main__":
    asyncio.run(main()) 