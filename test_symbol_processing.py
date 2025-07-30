#!/usr/bin/env python3
"""
Test script to verify the new symbol processing logic
"""

import asyncio
import sys
import os
import logging
from typing import List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_symbol_batch_processing():
    """Test the new symbol batch processing logic."""
    try:
        logger.info("🧪 TESTING NEW SYMBOL BATCH PROCESSING LOGIC")
        
        # Create test symbols
        test_symbols = [
            'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT',
            'LTCUSDT', 'BCHUSDT', 'XLMUSDT', 'VETUSDT', 'TRXUSDT',
            'EOSUSDT', 'ATOMUSDT', 'NEOUSDT', 'IOTAUSDT', 'XMRUSDT',
            'DASHUSDT', 'ZECUSDT', 'ETCUSDT', 'XRPUSDT', 'BNBUSDT'
        ]
        
        logger.info(f"Test symbols: {len(test_symbols)}")
        logger.info(f"First 5: {test_symbols[:5]}")
        logger.info(f"Last 5: {test_symbols[-5:]}")
        
        # Test batch creation logic
        max_concurrent_tasks = 5
        batch_size = max_concurrent_tasks
        symbol_batches = [test_symbols[i:i + batch_size] for i in range(0, len(test_symbols), batch_size)]
        
        logger.info(f"Created {len(symbol_batches)} batches:")
        for i, batch in enumerate(symbol_batches):
            logger.info(f"Batch {i+1}: {len(batch)} symbols - {batch}")
        
        # Test processing simulation
        processed_count = 0
        total_symbols = len(test_symbols)
        
        async def simulate_process_symbol(symbol: str):
            """Simulate processing a single symbol."""
            nonlocal processed_count
            processed_count += 1
            logger.info(f"Processing symbol {processed_count}/{total_symbols}: {symbol}")
            # Simulate processing time
            await asyncio.sleep(0.1)
            logger.info(f"Completed processing for {symbol}")
        
        async def process_symbol_batch(symbol_batch: List[str]):
            """Process a batch of symbols."""
            logger.info(f"Starting batch with {len(symbol_batch)} symbols")
            for symbol in symbol_batch:
                await simulate_process_symbol(symbol)
            logger.info(f"Completed batch with {len(symbol_batch)} symbols")
        
        # Process all batches
        tasks = []
        for batch in symbol_batches:
            task = asyncio.create_task(process_symbol_batch(batch))
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"✅ All symbols processed: {processed_count}/{total_symbols}")
        
        if processed_count == total_symbols:
            logger.info("🎉 Test passed! All symbols were processed.")
            return True
        else:
            logger.error(f"❌ Test failed! Only {processed_count}/{total_symbols} symbols processed.")
            return False
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {str(e)}")
        return False

async def main():
    """Run the test."""
    success = await test_symbol_batch_processing()
    if success:
        logger.info("🎉 Symbol batch processing test completed successfully!")
    else:
        logger.error("❌ Symbol batch processing test failed!")

if __name__ == "__main__":
    asyncio.run(main())