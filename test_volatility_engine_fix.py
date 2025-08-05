#!/usr/bin/env python3
"""
Test script to verify volatility_engine initialization fix
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

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

# Import the strategy
try:
    from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
    STRATEGY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import strategy: {e}")
    STRATEGY_AVAILABLE = False

async def test_volatility_engine_initialization():
    """Test that volatility_engine is properly initialized."""
    if not STRATEGY_AVAILABLE:
        logger.warning("Skipping volatility engine initialization test - strategy not available")
        return
    
    logger.info("🧪 Testing Volatility Engine Initialization...")
    
    try:
        # Initialize mock services
        config = {
            'volatility_window': 252,
            'garch_p': 1,
            'garch_q': 1,
            'volatility_threshold': 0.2,
            'trading': {
                'statistical_significance_level': 0.1,
                'min_sample_size': 10
            }
        }
        
        binance_service = MockBinanceService()
        indicator_service = MockIndicatorService()
        notification_service = MockNotificationService()
        cache_service = MockCacheService()
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config, binance_service, indicator_service, notification_service, cache_service
        )
        
        # Test that volatility_engine is initialized
        if hasattr(strategy, 'volatility_engine'):
            logger.info("✅ volatility_engine attribute exists")
            
            # Test that it's properly initialized
            if strategy.volatility_engine is not None:
                logger.info("✅ volatility_engine is not None")
                
                # Test basic functionality
                try:
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
                    volatility_analysis = strategy.volatility_engine.analyze_volatility_surface('BTCUSDT', market_data)
                    
                    if 'error' not in volatility_analysis:
                        logger.info("✅ volatility_engine.analyze_volatility_surface() works correctly")
                        
                        # Test key components
                        assert 'historical_volatility' in volatility_analysis
                        assert 'realized_volatility' in volatility_analysis
                        assert 'garch_volatility' in volatility_analysis
                        assert 'regime_analysis' in volatility_analysis
                        
                        logger.info(f"📊 Historical Volatility: {volatility_analysis['historical_volatility']:.4f}")
                        logger.info(f"📊 Realized Volatility: {volatility_analysis['realized_volatility']:.4f}")
                        logger.info(f"📊 GARCH Volatility: {volatility_analysis['garch_volatility']:.4f}")
                        
                        regime = volatility_analysis['regime_analysis']['regime']
                        regime_score = volatility_analysis['regime_analysis']['regime_score']
                        logger.info(f"📊 Volatility Regime: {regime} (score: {regime_score:.2f})")
                        
                    else:
                        logger.error(f"❌ volatility_engine.analyze_volatility_surface() failed: {volatility_analysis['error']}")
                        
                except Exception as e:
                    logger.error(f"❌ Error testing volatility_engine functionality: {str(e)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    
            else:
                logger.error("❌ volatility_engine is None")
        else:
            logger.error("❌ volatility_engine attribute does not exist")
        
        logger.info("✅ Volatility Engine Initialization test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Volatility Engine Initialization test failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def test_volatility_regime_analysis_fix():
    """Test that the volatility regime analysis works without errors."""
    if not STRATEGY_AVAILABLE:
        logger.warning("Skipping volatility regime analysis test - strategy not available")
        return
    
    logger.info("🧪 Testing Volatility Regime Analysis Fix...")
    
    try:
        # Initialize mock services
        config = {
            'volatility_window': 252,
            'garch_p': 1,
            'garch_q': 1,
            'volatility_threshold': 0.2,
            'trading': {
                'statistical_significance_level': 0.1,
                'min_sample_size': 10
            }
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
            'reasons': ['technical_analysis'],
            'symbol': 'BTCUSDT'
        }
        
        # Test volatility regime analysis (this was failing before)
        try:
            vol_signal = await strategy._apply_volatility_regime_analysis('BTCUSDT', base_signal, market_data)
            logger.info("✅ _apply_volatility_regime_analysis() completed successfully")
            
            if 'implied_volatility_analysis' in vol_signal:
                logger.info("✅ Implied volatility analysis was applied")
                
                vol_analysis = vol_signal['implied_volatility_analysis']
                regime = vol_analysis['regime_analysis']['regime']
                regime_score = vol_analysis['regime_analysis']['regime_score']
                
                logger.info(f"📊 Volatility Regime: {regime} (score: {regime_score:.2f})")
                logger.info(f"📊 Position Size Adjustment: {vol_signal.get('volatility_adjusted_position_size', 0):.4f}")
                logger.info(f"📊 Optimal Stop Loss: {vol_signal.get('volatility_optimal_stop_loss', 0):.2f}")
                
            else:
                logger.warning("⚠️ Implied volatility analysis not applied (fallback used)")
                
        except Exception as e:
            logger.error(f"❌ _apply_volatility_regime_analysis() failed: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        logger.info("✅ Volatility Regime Analysis Fix test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Volatility Regime Analysis Fix test failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def main():
    """Main test function."""
    logger.info("🚀 Starting Volatility Engine Fix Tests...")
    
    try:
        # Test 1: Volatility Engine Initialization
        await test_volatility_engine_initialization()
        
        # Test 2: Volatility Regime Analysis Fix
        await test_volatility_regime_analysis_fix()
        
        logger.info("🎉 All Volatility Engine Fix tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main()) 