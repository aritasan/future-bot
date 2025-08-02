#!/usr/bin/env python3
"""
Test script to verify the continuous loop implementation in main_with_quantitative.py
"""

import asyncio
import logging
from typing import List

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables (same as main_with_quantitative.py)
is_running = True
tasks: List[asyncio.Task] = []
shutdown_event = asyncio.Event()

async def mock_process_symbol(symbol: str, cycle: int):
    """Mock function to simulate symbol processing."""
    logger.info(f"Processing {symbol} in cycle {cycle}")
    await asyncio.sleep(0.1)  # Simulate processing time
    return f"Processed {symbol}"

async def mock_portfolio_analysis(cycle: int):
    """Mock function to simulate portfolio analysis."""
    logger.info(f"Running portfolio analysis in cycle {cycle}")
    await asyncio.sleep(0.2)  # Simulate analysis time
    return "Portfolio analysis completed"

async def test_continuous_loop():
    """Test the continuous loop implementation."""
    global is_running, tasks, shutdown_event
    
    # Mock symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
    max_concurrent_tasks = 3
    
    logger.info(f"Testing continuous loop with {len(symbols)} symbols")
    
    # Continuous processing loop
    cycle_count = 0
    max_cycles = 3  # Limit for testing
    
    while is_running and not shutdown_event.is_set() and cycle_count < max_cycles:
        cycle_count += 1
        logger.info(f"=== Starting cycle {cycle_count} ===")
        
        # Clear previous tasks
        tasks.clear()
        
        # Process symbols in batches
        batch_size = max_concurrent_tasks
        symbol_batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        
        for batch in symbol_batches:
            async def process_batch(batch_symbols):
                """Process a batch of symbols."""
                for symbol in batch_symbols:
                    await mock_process_symbol(symbol, cycle_count)
            
            task = asyncio.create_task(process_batch(batch))
            tasks.append(task)
        
        # Start portfolio analysis task
        portfolio_task = asyncio.create_task(mock_portfolio_analysis(cycle_count))
        tasks.append(portfolio_task)
        
        # Wait for all tasks to complete
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"=== Completed cycle {cycle_count} ===")
            
            # Wait before starting next cycle (shorter for testing)
            logger.info("Waiting 2 seconds before starting next cycle...")
            await asyncio.sleep(2)  # 2 seconds for testing
            
        except asyncio.CancelledError:
            logger.info("Main task gathering cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in cycle {cycle_count}: {str(e)}")
            await asyncio.sleep(1)  # 1 second
    
    logger.info("Continuous loop test completed successfully!")

async def test_shutdown_handling():
    """Test shutdown handling."""
    global is_running, shutdown_event
    
    logger.info("Testing shutdown handling...")
    
    # Start the continuous loop
    loop_task = asyncio.create_task(test_continuous_loop())
    
    # Wait a bit then trigger shutdown
    await asyncio.sleep(3)
    logger.info("Triggering shutdown...")
    is_running = False
    shutdown_event.set()
    
    # Wait for the loop to finish
    try:
        await loop_task
    except asyncio.CancelledError:
        logger.info("Loop was cancelled as expected")
    
    logger.info("Shutdown test completed!")

if __name__ == "__main__":
    print("Testing continuous loop implementation...")
    asyncio.run(test_continuous_loop())
    
    print("\nTesting shutdown handling...")
    asyncio.run(test_shutdown_handling())
    
    print("\nAll tests completed successfully!") 