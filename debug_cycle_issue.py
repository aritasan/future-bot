#!/usr/bin/env python3
"""
Debug script to identify why the bot stops after completing cycle 1.
"""

import asyncio
import logging
import time
import signal
import traceback
from typing import List, Set, Optional, Dict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
is_running = True
tasks: List[asyncio.Task] = []
shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global is_running
    logger.info(f"Received signal {signum}")
    logger.info(f'Frame: {frame}')
    is_running = False
    if shutdown_event:
        shutdown_event.set()

async def test_cycle_processing():
    """Test cycle processing to identify the issue."""
    logger.info("🧪 Testing cycle processing...")
    
    try:
        # Simulate the main loop
        cycle_count = 0
        while is_running and not shutdown_event.is_set():
            cycle_count += 1
            logger.info(f"=== Starting cycle {cycle_count} ===")
            
            # Simulate processing
            logger.info("Processing symbols...")
            await asyncio.sleep(2)  # Simulate processing time
            
            logger.info(f"=== Completed cycle {cycle_count} ===")
            
            # Wait before starting next cycle (5 minutes)
            logger.info("Waiting 5 minutes before starting next cycle...")
            
            try:
                # Test the sleep with timeout and shutdown check
                start_time = time.time()
                while time.time() - start_time < 300:  # 5 minutes
                    if shutdown_event.is_set():
                        logger.info("Shutdown event detected during sleep")
                        break
                    
                    # Sleep in smaller chunks to check shutdown event
                    await asyncio.sleep(10)  # Check every 10 seconds
                    
                    # Log progress
                    elapsed = time.time() - start_time
                    if elapsed % 60 < 10:  # Log every minute
                        logger.info(f"Sleep progress: {elapsed:.0f}s / 300s")
                
                if shutdown_event.is_set():
                    logger.info("Shutdown detected, breaking cycle loop")
                    break
                    
                logger.info("Sleep completed, continuing to next cycle")
                
            except asyncio.CancelledError:
                logger.info("Sleep cancelled")
                raise
            except Exception as e:
                logger.error(f"Error during sleep: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                break
        
        logger.info(f"Cycle processing completed. Total cycles: {cycle_count}")
        
    except Exception as e:
        logger.error(f"Error in test_cycle_processing: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

async def test_signal_handling():
    """Test signal handling."""
    logger.info("🧪 Testing signal handling...")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await test_cycle_processing()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    except Exception as e:
        logger.error(f"Error in test_signal_handling: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

async def test_main_loop_simulation():
    """Simulate the exact main loop from main_with_quantitative.py."""
    logger.info("🧪 Testing main loop simulation...")
    
    try:
        # Simulate the main loop structure
        cycle_count = 0
        while is_running and not shutdown_event.is_set():
            cycle_count += 1
            logger.info(f"=== Starting cycle {cycle_count} ===")
            
            # Clear previous tasks
            tasks.clear()
            
            # Simulate processing tasks
            logger.info("Creating processing tasks...")
            for i in range(3):  # Simulate 3 batches
                task = asyncio.create_task(asyncio.sleep(1))  # Simulate processing
                tasks.append(task)
            
            # Wait for all tasks to complete
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.info(f"=== Completed cycle {cycle_count} ===")
                
                # Wait before starting next cycle (5 minutes)
                logger.info("Waiting 5 minutes before starting next cycle...")
                
                # Test different sleep approaches
                logger.info("Testing sleep with shutdown event check...")
                
                # Approach 1: Simple sleep
                try:
                    await asyncio.sleep(300)  # 5 minutes
                    logger.info("Simple sleep completed")
                except asyncio.CancelledError:
                    logger.info("Simple sleep cancelled")
                    raise
                except Exception as e:
                    logger.error(f"Error in simple sleep: {str(e)}")
                    break
                
            except asyncio.CancelledError:
                logger.info("Main task gathering cancelled")
                raise
            except Exception as e:
                logger.error(f"Error in cycle {cycle_count}: {str(e)}")
                await asyncio.sleep(60)  # 1 minute
        
        logger.info(f"Main loop simulation completed. Total cycles: {cycle_count}")
        
    except Exception as e:
        logger.error(f"Error in test_main_loop_simulation: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

async def test_shutdown_event():
    """Test shutdown event behavior."""
    logger.info("🧪 Testing shutdown event...")
    
    try:
        # Test shutdown event setting
        logger.info("Setting shutdown event...")
        shutdown_event.set()
        
        logger.info(f"Shutdown event is set: {shutdown_event.is_set()}")
        
        # Test sleep with shutdown event
        logger.info("Testing sleep with shutdown event set...")
        try:
            await asyncio.sleep(10)
            logger.info("Sleep completed despite shutdown event")
        except asyncio.CancelledError:
            logger.info("Sleep cancelled due to shutdown event")
        
    except Exception as e:
        logger.error(f"Error in test_shutdown_event: {str(e)}")

async def main():
    """Main debug function."""
    logger.info("🚀 Starting cycle issue debug...")
    
    try:
        # Test 1: Signal handling
        logger.info("\n🧪 Test 1: Signal handling")
        await test_signal_handling()
        
        # Reset for next test
        global is_running, shutdown_event
        is_running = True
        shutdown_event = asyncio.Event()
        
        # Test 2: Main loop simulation
        logger.info("\n🧪 Test 2: Main loop simulation")
        await test_main_loop_simulation()
        
        # Test 3: Shutdown event
        logger.info("\n🧪 Test 3: Shutdown event")
        await test_shutdown_event()
        
    except KeyboardInterrupt:
        logger.info("Debug stopped by user")
    except Exception as e:
        logger.error(f"Debug error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    logger.info("📋 Cycle issue debug completed!")

if __name__ == "__main__":
    asyncio.run(main()) 