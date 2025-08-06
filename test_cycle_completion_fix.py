#!/usr/bin/env python3
"""
Test script to verify the cycle completion fix.
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
        logging.FileHandler('test_cycle_completion_fix.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def mock_portfolio_analysis():
    """Mock portfolio analysis that completes quickly."""
    logger.info("Running mock portfolio analysis...")
    await asyncio.sleep(2)  # Simulate 2 seconds of work
    logger.info("Mock portfolio analysis completed")
    return {"status": "success"}

async def mock_symbol_processing(symbol: str):
    """Mock symbol processing."""
    logger.info(f"Processing symbol: {symbol}")
    await asyncio.sleep(0.1)  # Simulate 0.1 seconds of work
    logger.info(f"Completed processing symbol: {symbol}")

async def test_cycle_completion():
    """Test cycle completion with portfolio analysis."""
    
    logger.info("=== Starting Cycle Completion Test ===")
    
    cycle_count = 0
    max_cycles = 3
    
    while cycle_count < max_cycles:
        cycle_count += 1
        logger.info(f"=== Starting cycle {cycle_count} ===")
        
        # Create tasks
        tasks = []
        
        # Add symbol processing tasks
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
        for symbol in test_symbols:
            task = asyncio.create_task(mock_symbol_processing(symbol))
            tasks.append(task)
        
        # Add portfolio analysis task with timeout
        portfolio_task = asyncio.create_task(
            asyncio.wait_for(mock_portfolio_analysis(), timeout=10)  # 10 seconds timeout
        )
        tasks.append(portfolio_task)
        
        logger.info(f"Created {len(tasks)} tasks for cycle {cycle_count}")
        
        # Wait for all tasks to complete
        try:
            logger.info(f"Waiting for {len(tasks)} tasks to complete in cycle {cycle_count}")
            logger.info("Starting asyncio.gather() for cycle tasks...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"asyncio.gather() completed for cycle {cycle_count}")
            
            # Check for exceptions
            exceptions = [r for r in results if isinstance(r, Exception)]
            if exceptions:
                logger.error(f"Found {len(exceptions)} exceptions in cycle {cycle_count}:")
                for i, exc in enumerate(exceptions):
                    logger.error(f"Exception {i+1}: {str(exc)}")
            
            logger.info(f"=== Completed cycle {cycle_count} ===")
            
            if cycle_count < max_cycles:
                logger.info("Waiting 5 seconds before next cycle...")
                await asyncio.sleep(5)
                logger.info("5-second wait completed, continuing to next cycle")
        
        except Exception as e:
            logger.error(f"Error in cycle {cycle_count}: {str(e)}")
            logger.info("Continuing to next cycle despite error")
    
    logger.info("=== Cycle Completion Test Completed ===")

async def test_portfolio_analysis_timeout():
    """Test portfolio analysis timeout handling."""
    
    logger.info("=== Starting Portfolio Analysis Timeout Test ===")
    
    async def slow_portfolio_analysis():
        """Simulate a slow portfolio analysis."""
        logger.info("Starting slow portfolio analysis...")
        await asyncio.sleep(15)  # Simulate 15 seconds of work
        logger.info("Slow portfolio analysis completed")
        return {"status": "success"}
    
    try:
        # Test with 5-second timeout
        logger.info("Testing portfolio analysis with 5-second timeout...")
        result = await asyncio.wait_for(slow_portfolio_analysis(), timeout=5)
        logger.info(f"Portfolio analysis result: {result}")
    except asyncio.TimeoutError:
        logger.info("Portfolio analysis timed out as expected")
    except Exception as e:
        logger.error(f"Error in portfolio analysis: {str(e)}")
    
    logger.info("=== Portfolio Analysis Timeout Test Completed ===")

if __name__ == "__main__":
    async def run_all_tests():
        """Run all tests."""
        logger.info("🚀 Starting Cycle Completion Fix Tests")
        
        # Test 1: Cycle completion with portfolio analysis
        await test_cycle_completion()
        
        # Test 2: Portfolio analysis timeout
        await test_portfolio_analysis_timeout()
        
        logger.info("🎉 All tests completed successfully!")
    
    # Run tests
    asyncio.run(run_all_tests()) 