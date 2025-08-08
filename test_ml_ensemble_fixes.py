#!/usr/bin/env python3
"""
Test script to verify ML ensemble error fixes
"""

import asyncio
import logging
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.quantitative.advanced_ml_ensemble import AdvancedMLEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_market_data():
    """Create test market data with missing columns"""
    # Create data with missing 'close' column
    data = {
        'price': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        'timestamp': range(10)
    }
    return pd.DataFrame(data)

def create_test_market_data_with_close():
    """Create test market data with all required columns"""
    data = {
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
        'open': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        'timestamp': range(10)
    }
    return pd.DataFrame(data)

def test_engineer_features_with_missing_columns():
    """Test feature engineering with missing columns"""
    logger.info("Testing feature engineering with missing columns...")
    
    # Create ML ensemble
    config = {
        'lstm_params': {'units': 50, 'layers': 2, 'dropout': 0.2, 'lookback': 60},
        'transformer_params': {'d_model': 64, 'n_heads': 8, 'n_layers': 4, 'dropout': 0.1}
    }
    
    ml_ensemble = AdvancedMLEnsemble(config)
    
    # Test with missing columns
    market_data = create_test_market_data()
    
    try:
        features = ml_ensemble._engineer_features(market_data)
        logger.info(f"✅ Feature engineering with missing columns completed successfully")
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Features columns: {list(features.columns)}")
        return True
    except Exception as e:
        logger.error(f"❌ Error in feature engineering with missing columns: {str(e)}")
        return False

def test_engineer_features_with_all_columns():
    """Test feature engineering with all required columns"""
    logger.info("Testing feature engineering with all columns...")
    
    # Create ML ensemble
    config = {
        'lstm_params': {'units': 50, 'layers': 2, 'dropout': 0.2, 'lookback': 60},
        'transformer_params': {'d_model': 64, 'n_heads': 8, 'n_layers': 4, 'dropout': 0.1}
    }
    
    ml_ensemble = AdvancedMLEnsemble(config)
    
    # Test with all columns
    market_data = create_test_market_data_with_close()
    
    try:
        features = ml_ensemble._engineer_features(market_data)
        logger.info(f"✅ Feature engineering with all columns completed successfully")
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Features columns: {list(features.columns)}")
        return True
    except Exception as e:
        logger.error(f"❌ Error in feature engineering with all columns: {str(e)}")
        return False

def test_create_sequences_with_missing_close():
    """Test sequence creation with missing 'close' column"""
    logger.info("Testing sequence creation with missing 'close' column...")
    
    # Create ML ensemble
    config = {
        'lstm_params': {'units': 50, 'layers': 2, 'dropout': 0.2, 'lookback': 60},
        'transformer_params': {'d_model': 64, 'n_heads': 8, 'n_layers': 4, 'dropout': 0.1}
    }
    
    ml_ensemble = AdvancedMLEnsemble(config)
    
    # Test with missing 'close' column
    market_data = create_test_market_data()
    features = ml_ensemble._engineer_features(market_data)
    
    try:
        X, y = ml_ensemble._create_sequences(features, lookback=3)
        logger.info(f"✅ Sequence creation with missing 'close' column completed successfully")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        return True
    except Exception as e:
        logger.error(f"❌ Error in sequence creation with missing 'close' column: {str(e)}")
        return False

def test_create_sequences_with_close():
    """Test sequence creation with 'close' column"""
    logger.info("Testing sequence creation with 'close' column...")
    
    # Create ML ensemble
    config = {
        'lstm_params': {'units': 50, 'layers': 2, 'dropout': 0.2, 'lookback': 60},
        'transformer_params': {'d_model': 64, 'n_heads': 8, 'n_layers': 4, 'dropout': 0.1}
    }
    
    ml_ensemble = AdvancedMLEnsemble(config)
    
    # Test with 'close' column
    market_data = create_test_market_data_with_close()
    features = ml_ensemble._engineer_features(market_data)
    
    try:
        X, y = ml_ensemble._create_sequences(features, lookback=3)
        logger.info(f"✅ Sequence creation with 'close' column completed successfully")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        return True
    except Exception as e:
        logger.error(f"❌ Error in sequence creation with 'close' column: {str(e)}")
        return False

def test_prepare_data():
    """Test the complete data preparation pipeline"""
    logger.info("Testing complete data preparation pipeline...")
    
    # Create ML ensemble
    config = {
        'lstm_params': {'units': 50, 'layers': 2, 'dropout': 0.2, 'lookback': 60},
        'transformer_params': {'d_model': 64, 'n_heads': 8, 'n_layers': 4, 'dropout': 0.1}
    }
    
    ml_ensemble = AdvancedMLEnsemble(config)
    
    # Test with missing columns
    market_data = create_test_market_data()
    
    try:
        X, y = ml_ensemble.prepare_data(market_data)
        logger.info(f"✅ Complete data preparation pipeline completed successfully")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        return True
    except Exception as e:
        logger.error(f"❌ Error in complete data preparation pipeline: {str(e)}")
        return False

async def main():
    """Main test function"""
    logger.info("Starting ML ensemble error fixes test...")
    
    tests = [
        test_engineer_features_with_missing_columns,
        test_engineer_features_with_all_columns,
        test_create_sequences_with_missing_close,
        test_create_sequences_with_close,
        test_prepare_data
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {str(e)}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"TEST SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        logger.info("✅ All tests passed! ML ensemble errors have been fixed.")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
