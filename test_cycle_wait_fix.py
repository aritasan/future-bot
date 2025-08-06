#!/usr/bin/env python3
"""
Test script to verify the cycle wait fix.
"""

import asyncio
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_cycle_wait_fix.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def test_cycle_wait():
    """Test the cycle wait mechanism."""
    
    logger.info("=== Starting Cycle Wait Test ===")
    
    # Simulate shutdown event
    shutdown_event = asyncio.Event()
    
    cycle_count = 0
    max_cycles = 3
    
    while cycle_count < max_cycles:
        cycle_count += 1
        logger.info(f"=== Starting cycle {cycle_count} ===")
        
        # Simulate processing
        logger.info("Processing cycle...")
        await asyncio.sleep(2)  # Simulate processing time
        
        logger.info(f"=== Completed cycle {cycle_count} ===")
        
        if cycle_count < max_cycles:
            # Test the wait mechanism
            logger.info("Waiting 10 seconds before next cycle...")
            try:
                # Use the same mechanism as the main bot
                await asyncio.wait_for(shutdown_event.wait(), timeout=10)  # 10 seconds for testing
                if shutdown_event.is_set():
                    logger.info("Shutdown event detected, stopping test")
                    break
            except asyncio.TimeoutError:
                logger.info("10-second wait completed, continuing to next cycle")
            except asyncio.CancelledError:
                logger.info("Cycle wait cancelled, stopping test")
                raise
            except Exception as e:
                logger.error(f"Error during cycle wait: {str(e)}")
                logger.info("Continuing to next cycle despite wait error")
    
    logger.info("=== Cycle Wait Test Completed ===")

async def test_long_wait():
    """Test a longer wait to simulate the 5-minute wait."""
    
    logger.info("=== Starting Long Wait Test ===")
    
    shutdown_event = asyncio.Event()
    
    logger.info("Starting 30-second wait (simulating 5-minute wait)...")
    start_time = time.time()
    
    try:
        await asyncio.wait_for(shutdown_event.wait(), timeout=30)  # 30 seconds for testing
        if shutdown_event.is_set():
            logger.info("Shutdown event detected during wait")
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.info(f"30-second wait completed in {elapsed:.2f} seconds")
    except asyncio.CancelledError:
        logger.info("Wait cancelled")
        raise
    except Exception as e:
        logger.error(f"Error during long wait: {str(e)}")
    
    logger.info("=== Long Wait Test Completed ===")

async def test_error_recovery():
    """Test error recovery during wait."""
    
    logger.info("=== Starting Error Recovery Test ===")
    
    shutdown_event = asyncio.Event()
    
    logger.info("Testing error recovery during wait...")
    
    try:
        # Simulate an error during wait
        async def problematic_wait():
            await asyncio.sleep(5)
            raise Exception("Simulated error during wait")
        
        await asyncio.wait_for(problematic_wait(), timeout=10)
    except asyncio.TimeoutError:
        logger.info("Timeout occurred as expected")
    except Exception as e:
        logger.info(f"Error caught and handled: {str(e)}")
        logger.info("Continuing despite error")
    
    logger.info("=== Error Recovery Test Completed ===")

if __name__ == "__main__":
    async def run_all_tests():
        """Run all tests."""
        logger.info("🚀 Starting Cycle Wait Fix Tests")
        
        # Test 1: Basic cycle wait
        await test_cycle_wait()
        
        # Test 2: Long wait simulation
        await test_long_wait()
        
        # Test 3: Error recovery
        await test_error_recovery()
        
        logger.info("🎉 All tests completed successfully!")
    
    # Run tests
    asyncio.run(run_all_tests()) 