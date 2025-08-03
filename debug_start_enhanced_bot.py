#!/usr/bin/env python3
"""
Debug script to identify why the enhanced bot starter is stopping.
"""

import asyncio
import logging
import subprocess
import time
import os
import signal
import psutil
from typing import Dict, Any, Optional, List
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebugEnhancedBotStarter:
    """Debug version of EnhancedBotStarter."""
    
    def __init__(self):
        self.bot_process = None
        self.monitor_process = None
        self.start_time = None
        self.current_pid = os.getpid()
        
    async def debug_kill_existing_processes(self):
        """Debug version of kill existing processes."""
        logger.info("🔪 DEBUG: Starting kill existing processes...")
        
        try:
            logger.info(f"🔪 DEBUG: Current process PID: {self.current_pid}")
            
            # Check if we're on Windows
            if os.name == 'nt':
                logger.info("🔪 DEBUG: Running on Windows, using psutil...")
                
                # Use psutil to find and kill other Python processes
                killed_count = 0
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['name'] == 'python.exe' and proc.pid != self.current_pid:
                            cmdline = proc.info.get('cmdline', [])
                            if cmdline and any('main_with_quantitative.py' in arg for arg in cmdline):
                                logger.info(f"🔪 DEBUG: Killing process {proc.pid} with cmdline: {cmdline}")
                                proc.terminate()
                                killed_count += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        pass
                
                logger.info(f"🔪 DEBUG: Killed {killed_count} trading bot processes")
                
            else:
                logger.info("🔪 DEBUG: Running on Unix-like system...")
                # Use psutil for Unix-like systems too
                killed_count = 0
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['name'] == 'python' and proc.pid != self.current_pid:
                            cmdline = proc.info.get('cmdline', [])
                            if cmdline and any('main_with_quantitative.py' in arg for arg in cmdline):
                                logger.info(f"🔪 DEBUG: Killing process {proc.pid} with cmdline: {cmdline}")
                                proc.terminate()
                                killed_count += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        pass
                
                logger.info(f"🔪 DEBUG: Killed {killed_count} trading bot processes")
            
            # Wait for processes to terminate
            logger.info("🔪 DEBUG: Waiting 3 seconds for processes to terminate...")
            await asyncio.sleep(3)
            
            logger.info("🔪 DEBUG: Kill existing processes completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"🔪 DEBUG: Error killing processes: {str(e)}")
            logger.error(f"🔪 DEBUG: Traceback: {traceback.format_exc()}")
            return False
    
    async def debug_start_enhanced_bot(self):
        """Debug version of start enhanced bot."""
        logger.info("🚀 DEBUG: Starting enhanced bot...")
        
        try:
            logger.info("🚀 DEBUG: About to start main_with_quantitative.py...")
            
            # Check if the file exists
            if not os.path.exists('main_with_quantitative.py'):
                logger.error("🚀 DEBUG: main_with_quantitative.py not found!")
                return False
            
            logger.info("🚀 DEBUG: main_with_quantitative.py exists, starting process...")
            
            # Start the bot process
            self.bot_process = subprocess.Popen(
                ['python', 'main_with_quantitative.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            logger.info(f"🚀 DEBUG: Process created with PID: {self.bot_process.pid}")
            
            # Wait a moment to see if it starts successfully
            await asyncio.sleep(2)
            
            # Check if process is still running
            if self.bot_process.poll() is None:
                logger.info("🚀 DEBUG: Process is running successfully!")
                self.start_time = time.time()
                return True
            else:
                return_code = self.bot_process.returncode
                logger.error(f"🚀 DEBUG: Process failed to start, return code: {return_code}")
                
                # Get any error output
                stdout, stderr = self.bot_process.communicate()
                if stdout:
                    logger.info(f"🚀 DEBUG: Process stdout: {stdout}")
                if stderr:
                    logger.error(f"🚀 DEBUG: Process stderr: {stderr}")
                
                return False
            
        except Exception as e:
            logger.error(f"🚀 DEBUG: Error starting enhanced bot: {str(e)}")
            logger.error(f"🚀 DEBUG: Traceback: {traceback.format_exc()}")
            return False
    
    async def debug_start_real_time_monitor(self):
        """Debug version of start real-time monitor."""
        logger.info("📊 DEBUG: Starting real-time monitor...")
        
        try:
            # Check if the file exists
            if not os.path.exists('real_time_monitor.py'):
                logger.error("📊 DEBUG: real_time_monitor.py not found!")
                return False
            
            logger.info("📊 DEBUG: real_time_monitor.py exists, starting process...")
            
            # Start the monitoring process
            self.monitor_process = subprocess.Popen(
                ['python', 'real_time_monitor.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            logger.info(f"📊 DEBUG: Monitor process created with PID: {self.monitor_process.pid}")
            
            # Wait a moment to see if it starts successfully
            await asyncio.sleep(2)
            
            # Check if process is still running
            if self.monitor_process.poll() is None:
                logger.info("📊 DEBUG: Monitor process is running successfully!")
                return True
            else:
                return_code = self.monitor_process.returncode
                logger.error(f"📊 DEBUG: Monitor process failed to start, return code: {return_code}")
                
                # Get any error output
                stdout, stderr = self.monitor_process.communicate()
                if stdout:
                    logger.info(f"📊 DEBUG: Monitor stdout: {stdout}")
                if stderr:
                    logger.error(f"📊 DEBUG: Monitor stderr: {stderr}")
                
                return False
            
        except Exception as e:
            logger.error(f"📊 DEBUG: Error starting monitor: {str(e)}")
            logger.error(f"📊 DEBUG: Traceback: {traceback.format_exc()}")
            return False
    
    async def debug_run_enhanced_bot_session(self):
        """Debug version of run enhanced bot session."""
        logger.info("🔄 DEBUG: Starting enhanced bot session...")
        
        # Step 1: Kill existing processes
        logger.info("🔄 DEBUG: Step 1 - Killing existing processes...")
        kill_success = await self.debug_kill_existing_processes()
        if not kill_success:
            logger.error("🔄 DEBUG: Failed to kill existing processes!")
            return False
        
        logger.info("🔄 DEBUG: Step 1 completed successfully!")
        
        # Step 2: Start enhanced bot
        logger.info("🔄 DEBUG: Step 2 - Starting enhanced bot...")
        bot_success = await self.debug_start_enhanced_bot()
        if not bot_success:
            logger.error("🔄 DEBUG: Failed to start enhanced bot!")
            return False
        
        logger.info("🔄 DEBUG: Step 2 completed successfully!")
        
        # Step 3: Start real-time monitor
        logger.info("🔄 DEBUG: Step 3 - Starting real-time monitor...")
        monitor_success = await self.debug_start_real_time_monitor()
        if not monitor_success:
            logger.warning("🔄 DEBUG: Failed to start real-time monitor, but continuing...")
        
        logger.info("🔄 DEBUG: Step 3 completed!")
        
        # Step 4: Monitor for a short time
        logger.info("🔄 DEBUG: Step 4 - Monitoring for 30 seconds...")
        monitor_duration = 30  # 30 seconds for testing
        start_time = time.time()
        
        try:
            while time.time() - start_time < monitor_duration:
                # Check bot health every 5 seconds
                if self.bot_process and self.bot_process.poll() is None:
                    logger.info("🔄 DEBUG: Bot process is still running...")
                else:
                    logger.error("🔄 DEBUG: Bot process has stopped!")
                    break
                
                await asyncio.sleep(5)
            
            logger.info("🔄 DEBUG: Monitoring completed!")
            
        except KeyboardInterrupt:
            logger.info("🔄 DEBUG: Session stopped by user")
        except Exception as e:
            logger.error(f"🔄 DEBUG: Session error: {str(e)}")
        
        # Cleanup
        logger.info("🔄 DEBUG: Cleaning up processes...")
        if self.bot_process:
            logger.info("🔄 DEBUG: Terminating bot process...")
            self.bot_process.terminate()
            try:
                self.bot_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("🔄 DEBUG: Bot didn't stop gracefully, forcing kill")
                self.bot_process.kill()
        
        if self.monitor_process:
            logger.info("🔄 DEBUG: Terminating monitor process...")
            self.monitor_process.terminate()
            try:
                self.monitor_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.monitor_process.kill()
        
        logger.info("🔄 DEBUG: Enhanced bot session completed!")
        return True

async def main():
    """Main debug function."""
    logger.info("🚀 DEBUG: Starting debug enhanced bot...")
    
    # Create debug enhanced bot starter
    starter = DebugEnhancedBotStarter()
    
    # Run debug enhanced bot session
    success = await starter.debug_run_enhanced_bot_session()
    
    if success:
        logger.info("\n🎉 DEBUG: Enhanced bot session completed successfully!")
    else:
        logger.error("\n❌ DEBUG: Enhanced bot session failed!")
    
    logger.info("\n📋 DEBUG: Debug session completed!")

if __name__ == "__main__":
    asyncio.run(main()) 