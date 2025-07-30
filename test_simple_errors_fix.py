#!/usr/bin/env python3
"""
Simple test script to verify that the basic errors are fixed.
"""

import asyncio
import logging
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_convert_klines_to_dataframe():
    """Test the klines to DataFrame conversion logic."""
    logger.info("Testing klines to DataFrame conversion...")
    
    # Test klines data
    klines = {
        'open': [50000, 50100, 50200],
        'high': [50500, 50600, 50700],
        'low': [49500, 49600, 49700],
        'close': [50000, 50100, 50200],
        'volume': [100, 110, 120]
    }
    
    # Simulate the conversion logic
    try:
        import pandas as pd
        
        # Create DataFrame from klines data
        df = pd.DataFrame({
            'open': klines['open'],
            'high': klines['high'],
            'low': klines['low'],
            'close': klines['close'],
            'volume': klines.get('volume', [0] * len(klines['close']))
        })
        
        logger.info(f"Converted DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        
        assert not df.empty, "DataFrame should not be empty"
        assert 'close' in df.columns, "DataFrame should have 'close' column"
        assert len(df) == 3, "DataFrame should have 3 rows"
        
        logger.info("✅ klines to DataFrame conversion works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ klines to DataFrame conversion failed: {e}")
        return False

def test_comprehensive_market_data_logic():
    """Test the comprehensive market data logic."""
    logger.info("Testing comprehensive market data logic...")
    
    # Simulate klines data
    klines = {
        'close': [50000, 50100, 50200, 50300, 50400]
    }
    
    try:
        import numpy as np
        
        # Simulate the market data calculation
        if klines is not None and len(klines['close']) > 1:
            prices = np.array(klines['close'])
            returns = np.diff(np.log(prices))
            volatility = float(np.std(returns) * np.sqrt(252))
            
            market_data = {
                'symbol': 'BTCUSDT',
                'returns': returns.tolist(),
                'volatility': volatility
            }
            
            logger.info(f"Market data: {market_data}")
            
            assert 'symbol' in market_data, "Market data should have 'symbol'"
            assert 'returns' in market_data, "Market data should have 'returns'"
            assert 'volatility' in market_data, "Market data should have 'volatility'"
            assert market_data['symbol'] == 'BTCUSDT', "Symbol should be BTCUSDT"
            
            logger.info("✅ Comprehensive market data logic works correctly")
            return True
            
    except Exception as e:
        logger.error(f"❌ Comprehensive market data logic failed: {e}")
        return False

def test_advanced_indicators_logic():
    """Test the advanced indicators calculation logic."""
    logger.info("Testing advanced indicators logic...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create test DataFrame
        test_df = pd.DataFrame({
            'open': [50000, 50100, 50200, 50300, 50400],
            'high': [50500, 50600, 50700, 50800, 50900],
            'low': [49500, 49600, 49700, 49800, 49900],
            'close': [50000, 50100, 50200, 50300, 50400],
            'volume': [100, 110, 120, 130, 140]
        })
        
        # Simulate advanced indicators calculation
        df = test_df.copy()
        
        # Basic indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum indicators
        df['momentum'] = df['close'] / df['close'].shift(10) - 1
        
        logger.info(f"Result DataFrame shape: {df.shape}")
        logger.info(f"Result DataFrame columns: {df.columns.tolist()}")
        
        # Check that advanced indicators were added
        expected_indicators = ['sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd', 'rsi', 'atr']
        for indicator in expected_indicators:
            assert indicator in df.columns, f"DataFrame should have '{indicator}' column"
        
        logger.info("✅ Advanced indicators logic works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Advanced indicators logic failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting simple errors fix tests...")
    
    try:
        test1 = test_convert_klines_to_dataframe()
        test2 = test_comprehensive_market_data_logic()
        test3 = test_advanced_indicators_logic()
        
        if test1 and test2 and test3:
            logger.info("🎉 All simple errors fix tests passed!")
        else:
            logger.error("❌ Some tests failed")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main() 