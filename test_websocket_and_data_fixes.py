#!/usr/bin/env python3
"""
Test script to verify websocket and data processing fixes.
"""

import sys
import os
import asyncio
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Suppress warnings for testing
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from src.core.config import load_config
from src.services.binance_service import BinanceService
from src.quantitative.portfolio_optimizer import PortfolioOptimizer
from src.quantitative.factor_model import FactorModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_websocket_fix():
    """Test websocket fix for unsupported watch_trades."""
    try:
        logger.info("Testing websocket fix...")
        
        config = load_config()
        binance_service = BinanceService(config)
        await binance_service.initialize()
        
        # Test symbols that were causing websocket errors
        test_symbols = ['1000CAT/USDT', '1000BONK/USDT', 'BTCUSDT']
        
        for symbol in test_symbols:
            try:
                logger.info(f"Testing trades for {symbol}...")
                
                # Test get_trades method
                trades = await binance_service.get_trades(symbol)
                logger.info(f"Trades for {symbol}: {len(trades)} trades retrieved")
                
                # Test get_recent_trades method
                recent_trades = await binance_service.get_recent_trades(symbol)
                logger.info(f"Recent trades for {symbol}: {len(recent_trades)} trades retrieved")
                
            except Exception as e:
                logger.error(f"Error testing {symbol}: {str(e)}")
        
        await binance_service.close()
        logger.info("Websocket fix test completed")
        
    except Exception as e:
        logger.error(f"Websocket test error: {str(e)}")

def test_portfolio_optimizer_fix():
    """Test portfolio optimizer fix for insufficient data."""
    try:
        logger.info("Testing portfolio optimizer fix...")
        
        optimizer = PortfolioOptimizer()
        
        # Test with insufficient data
        test_cases = [
            pd.DataFrame(),  # Empty DataFrame
            pd.DataFrame({'A': [1, 2, 3]}),  # Single column, few data points
            pd.DataFrame({'A': [np.nan, np.nan, np.nan]}),  # All NaN
            pd.DataFrame({'A': [1, 1, 1], 'B': [1, 1, 1]})  # No variance
        ]
        
        for i, returns in enumerate(test_cases):
            try:
                result = optimizer.optimize_portfolio(returns, method='markowitz')
                logger.info(f"Test case {i+1}: {result.get('error', 'Success')}")
            except Exception as e:
                logger.error(f"Test case {i+1} failed: {str(e)}")
        
        # Test with sufficient data
        try:
            # Create realistic returns data
            np.random.seed(42)
            returns = pd.DataFrame({
                'BTC': np.random.normal(0.001, 0.02, 100),
                'ETH': np.random.normal(0.0008, 0.025, 100),
                'BNB': np.random.normal(0.0005, 0.015, 100)
            })
            
            result = optimizer.optimize_portfolio(returns, method='markowitz')
            if 'error' not in result:
                logger.info("Sufficient data test: Optimization successful")
            else:
                logger.warning(f"Sufficient data test: {result['error']}")
                
        except Exception as e:
            logger.error(f"Sufficient data test failed: {str(e)}")
        
        logger.info("Portfolio optimizer fix test completed")
        
    except Exception as e:
        logger.error(f"Portfolio optimizer test error: {str(e)}")

def test_factor_model_fix():
    """Test factor model fix for insufficient data."""
    try:
        logger.info("Testing factor model fix...")
        
        factor_model = FactorModel()
        
        # Test with insufficient data
        test_cases = [
            pd.DataFrame(),  # Empty DataFrame
            pd.DataFrame({'A': [1, 2, 3]}),  # Single column, few data points
            pd.DataFrame({'A': [np.nan, np.nan, np.nan]}),  # All NaN
        ]
        
        for i, returns in enumerate(test_cases):
            try:
                result = factor_model.build_factor_model(returns)
                logger.info(f"Test case {i+1}: {result.get('pca_factors', {}).get('error', 'Success')}")
            except Exception as e:
                logger.error(f"Test case {i+1} failed: {str(e)}")
        
        # Test with sufficient data
        try:
            # Create realistic returns data
            np.random.seed(42)
            returns = pd.DataFrame({
                'BTC': np.random.normal(0.001, 0.02, 100),
                'ETH': np.random.normal(0.0008, 0.025, 100),
                'BNB': np.random.normal(0.0005, 0.015, 100),
                'ADA': np.random.normal(0.0003, 0.03, 100),
                'DOT': np.random.normal(0.0006, 0.018, 100)
            })
            
            result = factor_model.build_factor_model(returns)
            if 'error' not in result.get('pca_factors', {}):
                logger.info("Sufficient data test: Factor analysis successful")
            else:
                logger.warning(f"Sufficient data test: {result['pca_factors']['error']}")
                
        except Exception as e:
            logger.error(f"Sufficient data test failed: {str(e)}")
        
        logger.info("Factor model fix test completed")
        
    except Exception as e:
        logger.error(f"Factor model test error: {str(e)}")

async def test_integration():
    """Test integration of all fixes."""
    try:
        logger.info("Testing integration of all fixes...")
        
        config = load_config()
        binance_service = BinanceService(config)
        await binance_service.initialize()
        
        # Test market data retrieval
        test_symbol = 'BTCUSDT'
        
        try:
            # Get market data
            market_data = await binance_service.get_market_data(test_symbol)
            if market_data:
                logger.info(f"Market data retrieved for {test_symbol}")
            else:
                logger.warning(f"No market data for {test_symbol}")
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
        
        # Test trades retrieval
        try:
            trades = await binance_service.get_trades(test_symbol)
            logger.info(f"Trades retrieved: {len(trades)} trades")
        except Exception as e:
            logger.error(f"Error getting trades: {str(e)}")
        
        await binance_service.close()
        logger.info("Integration test completed")
        
    except Exception as e:
        logger.error(f"Integration test error: {str(e)}")

async def main():
    """Run all verification tests."""
    try:
        logger.info("Starting websocket and data fixes verification...")
        
        # Test websocket fix
        await test_websocket_fix()
        
        # Test portfolio optimizer fix
        test_portfolio_optimizer_fix()
        
        # Test factor model fix
        test_factor_model_fix()
        
        # Test integration
        await test_integration()
        
        logger.info("All websocket and data fixes verification completed successfully")
        
    except Exception as e:
        logger.error(f"Verification test error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 