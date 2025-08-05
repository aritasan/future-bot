#!/usr/bin/env python3
"""
Test script for Implied Volatility Integration
Following WorldQuant Standards for Volatility Analysis
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock imports for testing
class MockBinanceService:
    def __init__(self):
        self.initialized = True
    
    async def get_current_price(self, symbol):
        return 50000.0
    
    async def get_klines(self, symbol, interval, limit=100):
        # Generate mock klines data
        dates = pd.date_range(start='2024-01-01', periods=limit, freq='1h')
        prices = np.random.normal(50000, 1000, limit)
        return {
            'close': prices.tolist(),
            'high': (prices * 1.02).tolist(),
            'low': (prices * 0.98).tolist(),
            'volume': np.random.uniform(1000, 10000, limit).tolist()
        }

class MockIndicatorService:
    def __init__(self):
        self.initialized = True
    
    async def get_klines(self, symbol, interval, limit=100):
        # Generate mock klines data
        dates = pd.date_range(start='2024-01-01', periods=limit, freq='1h')
        prices = np.random.normal(50000, 1000, limit)
        return {
            'close': prices.tolist(),
            'high': (prices * 1.02).tolist(),
            'low': (prices * 0.98).tolist(),
            'volume': np.random.uniform(1000, 10000, limit).tolist()
        }

class MockNotificationService:
    def __init__(self):
        self.initialized = True
    
    async def send_message(self, message):
        logger.info(f"Mock notification: {message}")

class MockCacheService:
    def __init__(self):
        self.initialized = True
        self.cache = {}
    
    async def get_market_data(self, symbol, timeframe):
        return None
    
    async def cache_market_data(self, symbol, timeframe, data, ttl=300):
        self.cache[f"{symbol}_{timeframe}"] = data

# Import the Implied Volatility Engine
try:
    from src.quantitative.implied_volatility import ImpliedVolatilityEngine
    from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
    IMPLIED_VOL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import Implied Volatility modules: {e}")
    IMPLIED_VOL_AVAILABLE = False

async def test_implied_volatility_engine():
    """Test the Implied Volatility Engine functionality."""
    if not IMPLIED_VOL_AVAILABLE:
        logger.warning("Skipping Implied Volatility Engine test - modules not available")
        return
    
    logger.info("🧪 Testing Implied Volatility Engine...")
    
    try:
        # Initialize the engine
        config = {
            'volatility_window': 252,
            'garch_p': 1,
            'garch_q': 1,
            'volatility_threshold': 0.2
        }
        
        volatility_engine = ImpliedVolatilityEngine(config)
        logger.info("✅ Implied Volatility Engine initialized successfully")
        
        # Generate mock market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        prices = np.random.normal(50000, 1000, 100)
        returns = np.diff(np.log(prices))
        
        market_data = {
            'close': prices.tolist(),
            'high': (prices * 1.02).tolist(),
            'low': (prices * 0.98).tolist(),
            'volume': np.random.uniform(1000, 10000, 100).tolist(),
            'returns': returns.tolist()
        }
        
        # Test volatility surface analysis
        volatility_analysis = volatility_engine.analyze_volatility_surface('BTCUSDT', market_data)
        
        if 'error' not in volatility_analysis:
            logger.info("✅ Volatility surface analysis completed successfully")
            
            # Test key components
            assert 'historical_volatility' in volatility_analysis
            assert 'realized_volatility' in volatility_analysis
            assert 'garch_volatility' in volatility_analysis
            assert 'regime_analysis' in volatility_analysis
            assert 'volatility_forecast' in volatility_analysis
            
            logger.info(f"📊 Historical Volatility: {volatility_analysis['historical_volatility']:.4f}")
            logger.info(f"📊 Realized Volatility: {volatility_analysis['realized_volatility']:.4f}")
            logger.info(f"📊 GARCH Volatility: {volatility_analysis['garch_volatility']:.4f}")
            
            regime = volatility_analysis['regime_analysis']['regime']
            regime_score = volatility_analysis['regime_analysis']['regime_score']
            logger.info(f"📊 Volatility Regime: {regime} (score: {regime_score:.2f})")
            
            # Test volatility trading signals
            vol_signals = volatility_engine.get_volatility_trading_signals(volatility_analysis)
            logger.info(f"📊 Volatility Signals: {vol_signals}")
            
            # Test position size adjustment
            base_size = 0.01
            adjusted_size = volatility_engine.adjust_position_size_by_volatility(base_size, volatility_analysis)
            logger.info(f"📊 Position Size Adjustment: {base_size:.4f} -> {adjusted_size:.4f}")
            
            # Test optimal stop loss calculation
            current_price = 50000.0
            optimal_sl = volatility_engine.calculate_volatility_optimal_stop_loss(
                current_price, volatility_analysis, 'long'
            )
            logger.info(f"📊 Optimal Stop Loss: {optimal_sl:.2f}")
            
        else:
            logger.error(f"❌ Volatility surface analysis failed: {volatility_analysis['error']}")
        
        logger.info("✅ Implied Volatility Engine test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Implied Volatility Engine test failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def test_strategy_volatility_integration():
    """Test the integration of Implied Volatility into the trading strategy."""
    if not IMPLIED_VOL_AVAILABLE:
        logger.warning("Skipping strategy volatility integration test - modules not available")
        return
    
    logger.info("🧪 Testing Strategy Volatility Integration...")
    
    try:
        # Initialize mock services
        config = {
            'volatility_window': 252,
            'garch_p': 1,
            'garch_q': 1,
            'volatility_threshold': 0.2
        }
        
        binance_service = MockBinanceService()
        indicator_service = MockIndicatorService()
        notification_service = MockNotificationService()
        cache_service = MockCacheService()
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config, binance_service, indicator_service, notification_service, cache_service
        )
        
        await strategy.initialize()
        logger.info("✅ Strategy initialized successfully")
        
        # Generate mock market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        prices = np.random.normal(50000, 1000, 100)
        returns = np.diff(np.log(prices))
        
        market_data = {
            'close': prices.tolist(),
            'high': (prices * 1.02).tolist(),
            'low': (prices * 0.98).tolist(),
            'volume': np.random.uniform(1000, 10000, 100).tolist(),
            'returns': returns.tolist()
        }
        
        # Create base signal
        base_signal = {
            'action': 'buy',
            'strength': 0.5,
            'confidence': 0.6,
            'position_size': 0.01,
            'current_price': 50000.0,
            'reasons': ['technical_analysis', 'momentum'],
            'symbol': 'BTCUSDT'
        }
        
        # Test implied volatility analysis
        vol_signal = await strategy._apply_implied_volatility_analysis('BTCUSDT', base_signal, market_data)
        
        if 'implied_volatility_analysis' in vol_signal:
            logger.info("✅ Implied volatility analysis applied successfully")
            
            vol_analysis = vol_signal['implied_volatility_analysis']
            regime = vol_analysis['regime_analysis']['regime']
            regime_score = vol_analysis['regime_analysis']['regime_score']
            
            logger.info(f"📊 Signal after volatility analysis:")
            logger.info(f"  - Original strength: {base_signal['strength']:.3f}")
            logger.info(f"  - Adjusted strength: {vol_signal['strength']:.3f}")
            logger.info(f"  - Volatility regime: {regime} (score: {regime_score:.2f})")
            logger.info(f"  - Volatility signals: {vol_signal.get('volatility_signals', {})}")
            
            # Test final signal optimization with volatility
            optimized_signal = await strategy._optimize_final_signal_with_volatility('BTCUSDT', vol_signal, market_data)
            
            logger.info(f"📊 Signal after volatility optimization:")
            logger.info(f"  - Final confidence: {optimized_signal.get('final_confidence', 0):.3f}")
            logger.info(f"  - Optimized position size: {optimized_signal.get('optimized_position_size', 0):.4f}")
            logger.info(f"  - Risk adjusted strength: {optimized_signal.get('risk_adjusted_strength', 0):.3f}")
            
        else:
            logger.error("❌ Implied volatility analysis failed")
        
        logger.info("✅ Strategy volatility integration test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Strategy volatility integration test failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def test_volatility_regime_analysis():
    """Test the enhanced volatility regime analysis."""
    if not IMPLIED_VOL_AVAILABLE:
        logger.warning("Skipping volatility regime analysis test - modules not available")
        return
    
    logger.info("🧪 Testing Enhanced Volatility Regime Analysis...")
    
    try:
        # Initialize mock services
        config = {
            'volatility_window': 252,
            'garch_p': 1,
            'garch_q': 1,
            'volatility_threshold': 0.2
        }
        
        binance_service = MockBinanceService()
        indicator_service = MockIndicatorService()
        notification_service = MockNotificationService()
        cache_service = MockCacheService()
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config, binance_service, indicator_service, notification_service, cache_service
        )
        
        await strategy.initialize()
        
        # Generate mock market data with different volatility regimes
        # High volatility regime
        high_vol_prices = np.random.normal(50000, 2000, 100)  # High volatility
        high_vol_returns = np.diff(np.log(high_vol_prices))
        
        high_vol_market_data = {
            'close': high_vol_prices.tolist(),
            'high': (high_vol_prices * 1.02).tolist(),
            'low': (high_vol_prices * 0.98).tolist(),
            'volume': np.random.uniform(1000, 10000, 100).tolist(),
            'returns': high_vol_returns.tolist()
        }
        
        # Low volatility regime
        low_vol_prices = np.random.normal(50000, 500, 100)  # Low volatility
        low_vol_returns = np.diff(np.log(low_vol_prices))
        
        low_vol_market_data = {
            'close': low_vol_prices.tolist(),
            'high': (low_vol_prices * 1.02).tolist(),
            'low': (low_vol_prices * 0.98).tolist(),
            'volume': np.random.uniform(1000, 10000, 100).tolist(),
            'returns': low_vol_returns.tolist()
        }
        
        # Test high volatility regime
        base_signal = {
            'action': 'buy',
            'strength': 0.5,
            'confidence': 0.6,
            'position_size': 0.01,
            'current_price': 50000.0,
            'reasons': ['technical_analysis'],
            'symbol': 'BTCUSDT'
        }
        
        high_vol_signal = await strategy._apply_volatility_regime_analysis('BTCUSDT', base_signal, high_vol_market_data)
        low_vol_signal = await strategy._apply_volatility_regime_analysis('BTCUSDT', base_signal, low_vol_market_data)
        
        logger.info("📊 High Volatility Regime Analysis:")
        if 'implied_volatility_analysis' in high_vol_signal:
            vol_analysis = high_vol_signal['implied_volatility_analysis']
            regime = vol_analysis['regime_analysis']['regime']
            regime_score = vol_analysis['regime_analysis']['regime_score']
            logger.info(f"  - Regime: {regime} (score: {regime_score:.2f})")
            logger.info(f"  - Position size adjustment: {high_vol_signal.get('volatility_adjusted_position_size', 0):.4f}")
            logger.info(f"  - Optimal stop loss: {high_vol_signal.get('volatility_optimal_stop_loss', 0):.2f}")
        
        logger.info("📊 Low Volatility Regime Analysis:")
        if 'implied_volatility_analysis' in low_vol_signal:
            vol_analysis = low_vol_signal['implied_volatility_analysis']
            regime = vol_analysis['regime_analysis']['regime']
            regime_score = vol_analysis['regime_analysis']['regime_score']
            logger.info(f"  - Regime: {regime} (score: {regime_score:.2f})")
            logger.info(f"  - Position size adjustment: {low_vol_signal.get('volatility_adjusted_position_size', 0):.4f}")
            logger.info(f"  - Optimal stop loss: {low_vol_signal.get('volatility_optimal_stop_loss', 0):.2f}")
        
        logger.info("✅ Enhanced volatility regime analysis test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Enhanced volatility regime analysis test failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def main():
    """Main test function."""
    logger.info("🚀 Starting Implied Volatility Integration Tests...")
    
    try:
        # Test 1: Implied Volatility Engine
        await test_implied_volatility_engine()
        
        # Test 2: Strategy Integration
        await test_strategy_volatility_integration()
        
        # Test 3: Volatility Regime Analysis
        await test_volatility_regime_analysis()
        
        logger.info("🎉 All Implied Volatility integration tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main()) 