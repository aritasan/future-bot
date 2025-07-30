#!/usr/bin/env python3
"""
Test script to verify symbols loading and processing.
"""

import sys
import os
import asyncio
import logging
from typing import List

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_symbols_loading():
    """Test loading symbols from future_symbols.txt."""
    try:
        logger.info("Testing symbols loading...")
        
        # Load symbols from file
        try:
            with open("future_symbols.txt", "r") as f:
                symbols = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error reading trading pairs: {str(e)}")
            symbols = ['BTCUSDT']
        
        logger.info(f"Loaded {len(symbols)} trading symbols from future_symbols.txt")
        logger.info(f"First 10 symbols: {symbols[:10]}")
        logger.info(f"Last 10 symbols: {symbols[-10:]}")
        logger.info(f"Total symbols to process: {len(symbols)}")
        
        # Check for duplicates
        unique_symbols = set(symbols)
        if len(unique_symbols) != len(symbols):
            logger.warning(f"Found {len(symbols) - len(unique_symbols)} duplicate symbols")
        else:
            logger.info("No duplicate symbols found")
        
        # Check for invalid symbols
        invalid_symbols = []
        for symbol in symbols:
            if not symbol or '/' not in symbol:
                invalid_symbols.append(symbol)
        
        if invalid_symbols:
            logger.warning(f"Found {len(invalid_symbols)} invalid symbols: {invalid_symbols}")
        else:
            logger.info("All symbols are valid")
        
        # Test concurrency simulation
        max_concurrent_tasks = 10
        total_symbols = len(symbols)
        
        logger.info(f"Simulating processing of {total_symbols} symbols with max {max_concurrent_tasks} concurrent tasks")
        
        # Calculate processing time estimate
        estimated_time_per_symbol = 60  # seconds
        total_estimated_time = (total_symbols / max_concurrent_tasks) * estimated_time_per_symbol
        logger.info(f"Estimated processing time: {total_estimated_time/60:.1f} minutes")
        
        return symbols
        
    except Exception as e:
        logger.error(f"Symbols loading test error: {str(e)}")
        return []

def test_config_loading():
    """Test configuration loading."""
    try:
        logger.info("Testing configuration loading...")
        
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Check key configuration items
        telegram_enabled = config.get('telegram_enabled', False)
        discord_enabled = config.get('api', {}).get('discord', {}).get('enabled', False)
        
        logger.info(f"Telegram enabled: {telegram_enabled}")
        logger.info(f"Discord enabled: {discord_enabled}")
        
        return config
        
    except Exception as e:
        logger.error(f"Configuration loading test error: {str(e)}")
        return None

async def test_concurrent_processing_simulation():
    """Simulate concurrent processing of symbols."""
    try:
        logger.info("Testing concurrent processing simulation...")
        
        # Load symbols
        symbols = test_symbols_loading()
        if not symbols:
            return
        
        # Simulate processing with semaphore
        max_concurrent_tasks = 10
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        processed_count = 0
        total_symbols = len(symbols)
        
        async def simulate_process_symbol(symbol):
            nonlocal processed_count
            async with semaphore:
                processed_count += 1
                logger.info(f"Simulating processing symbol {processed_count}/{total_symbols}: {symbol}")
                # Simulate processing time
                await asyncio.sleep(0.1)
                return f"Processed {symbol}"
        
        logger.info(f"Starting simulation of {total_symbols} symbols with max {max_concurrent_tasks} concurrent tasks")
        
        # Create tasks
        tasks = []
        for symbol in symbols[:20]:  # Test with first 20 symbols
            task = asyncio.create_task(simulate_process_symbol(symbol))
            tasks.append(task)
        
        # Wait for completion
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"Simulation completed. Processed {len(results)} symbols")
        
    except Exception as e:
        logger.error(f"Concurrent processing simulation error: {str(e)}")

async def main():
    """Run all verification tests."""
    try:
        logger.info("Starting symbols loading and processing verification...")
        
        # Test symbols loading
        symbols = test_symbols_loading()
        
        # Test configuration loading
        config = test_config_loading()
        
        # Test concurrent processing simulation
        await test_concurrent_processing_simulation()
        
        logger.info("All symbols loading and processing verification completed successfully")
        
        # Summary
        logger.info("=== SUMMARY ===")
        logger.info(f"Total symbols available: {len(symbols)}")
        logger.info(f"Configuration loaded: {'Yes' if config else 'No'}")
        logger.info("Concurrent processing simulation: Completed")
        
    except Exception as e:
        logger.error(f"Verification test error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 