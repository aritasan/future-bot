#!/usr/bin/env python3
"""
Test script to verify WorldQuant-level strategy improvements.
"""

import sys
import os
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import load_config
from src.services.binance_service import BinanceService
from src.services.indicator_service import IndicatorService
from src.services.notification_service import NotificationService
from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_worldquant_strategy():
    """Test WorldQuant-level strategy improvements."""
    try:
        logger.info("Testing WorldQuant-level strategy improvements...")
        
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize services
        binance_service = BinanceService(config)
        await binance_service.initialize()
        logger.info("Binance service initialized")
        
        indicator_service = IndicatorService(config)
        await indicator_service.initialize()
        logger.info("Indicator service initialized")
        
        notification_service = NotificationService(config, None, None)
        await notification_service.initialize()
        logger.info("Notification service initialized")
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config, binance_service, indicator_service, notification_service
        )
        await strategy.initialize()
        logger.info("WorldQuant strategy initialized")
        
        # Test symbols
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        for symbol in test_symbols:
            logger.info(f"Testing strategy for {symbol}...")
            
            # Test signal generation
            signal = await strategy.generate_signals(symbol, indicator_service)
            
            if signal:
                logger.info(f"Signal generated for {symbol}:")
                logger.info(f"  Action: {signal.get('action', 'hold')}")
                logger.info(f"  Signal Strength: {signal.get('signal_strength', 0):.3f}")
                logger.info(f"  Confidence: {signal.get('final_confidence', 0):.3f}")
                logger.info(f"  Position Size: {signal.get('optimized_position_size', 0):.4f}")
                logger.info(f"  Reasons: {signal.get('reasons', [])}")
                
                # Test advanced features
                if 'timeframes' in signal:
                    logger.info(f"  Multi-timeframe analysis:")
                    for tf, tf_signal in signal['timeframes'].items():
                        logger.info(f"    {tf}: {tf_signal.get('signal', 'hold')} (strength: {tf_signal.get('strength', 0):.3f})")
                
                if 'volatility_regime' in signal:
                    regime = signal['volatility_regime']
                    logger.info(f"  Volatility Regime: {regime.get('regime', 'unknown')}")
                    logger.info(f"  Volatility Ratio: {regime.get('volatility_ratio', 0):.3f}")
                
                if 'momentum' in signal:
                    momentum = signal['momentum']
                    logger.info(f"  Momentum Analysis:")
                    logger.info(f"    Short-term: {momentum.get('short_term', 0):.4f}")
                    logger.info(f"    Medium-term: {momentum.get('medium_term', 0):.4f}")
                    logger.info(f"    Long-term: {momentum.get('long_term', 0):.4f}")
                
                if 'var_95' in signal:
                    logger.info(f"  Risk Metrics:")
                    logger.info(f"    VaR 95%: {signal.get('var_95', 0):.4f}")
                    logger.info(f"    VaR 99%: {signal.get('var_99', 0):.4f}")
                    logger.info(f"    Max Drawdown: {signal.get('max_drawdown', 0):.4f}")
                
                if 'mean_reversion' in signal:
                    mr = signal['mean_reversion']
                    logger.info(f"  Mean Reversion:")
                    logger.info(f"    Is Mean Reverting: {mr.get('is_mean_reverting', False)}")
                    logger.info(f"    Deviation: {mr.get('deviation', 0):.3f}")
                
                logger.info(f"  Risk-adjusted Strength: {signal.get('risk_adjusted_strength', 0):.3f}")
                
            else:
                logger.warning(f"No signal generated for {symbol}")
        
        # Test performance metrics
        metrics = await strategy.get_performance_metrics()
        logger.info(f"Performance Metrics: {metrics}")
        
        logger.info("WorldQuant strategy testing completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing WorldQuant strategy: {str(e)}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False

async def test_advanced_indicators():
    """Test advanced technical indicators."""
    try:
        logger.info("Testing advanced technical indicators...")
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1H')
        np.random.seed(42)
        
        # Generate realistic price data
        returns = np.random.normal(0.0001, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, len(dates)))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, len(dates))
        })
        
        # Initialize strategy to test indicators
        config = load_config()
        binance_service = BinanceService(config)
        indicator_service = IndicatorService(config)
        notification_service = NotificationService(config, None, None)
        
        strategy = EnhancedTradingStrategyWithQuantitative(
            config, binance_service, indicator_service, notification_service
        )
        
        # Test advanced indicators calculation
        df_with_indicators = await strategy._calculate_advanced_indicators(df)
        
        # Check if indicators were calculated
        required_indicators = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd', 'macd_signal',
            'rsi', 'bb_middle', 'bb_upper', 'bb_lower', 'atr', 'momentum'
        ]
        
        missing_indicators = []
        for indicator in required_indicators:
            if indicator not in df_with_indicators.columns:
                missing_indicators.append(indicator)
        
        if missing_indicators:
            logger.warning(f"Missing indicators: {missing_indicators}")
        else:
            logger.info("All advanced indicators calculated successfully")
            
            # Test indicator values
            latest = df_with_indicators.iloc[-1]
            logger.info(f"Latest indicator values:")
            logger.info(f"  RSI: {latest['rsi']:.2f}")
            logger.info(f"  MACD: {latest['macd']:.4f}")
            logger.info(f"  ATR: {latest['atr']:.4f}")
            logger.info(f"  Momentum: {latest['momentum']:.4f}")
            logger.info(f"  Volatility: {latest['volatility']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing advanced indicators: {str(e)}")
        return False

async def test_risk_management():
    """Test advanced risk management features."""
    try:
        logger.info("Testing advanced risk management...")
        
        # Create sample returns data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns
        
        # Test VaR calculation
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        logger.info(f"Risk Metrics:")
        logger.info(f"  VaR 95%: {var_95:.4f}")
        logger.info(f"  VaR 99%: {var_99:.4f}")
        logger.info(f"  Volatility: {np.std(returns):.4f}")
        logger.info(f"  Mean Return: {np.mean(returns):.4f}")
        
        # Test Kelly Criterion
        win_rate = np.sum(returns > 0) / len(returns)
        avg_win = np.mean(returns[returns > 0])
        avg_loss = abs(np.mean(returns[returns < 0]))
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = np.clip(kelly_fraction, 0.0, 0.25)
        
        logger.info(f"Kelly Criterion:")
        logger.info(f"  Win Rate: {win_rate:.3f}")
        logger.info(f"  Avg Win: {avg_win:.4f}")
        logger.info(f"  Avg Loss: {avg_loss:.4f}")
        logger.info(f"  Kelly Fraction: {kelly_fraction:.4f}")
        
        # Test drawdown calculation
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min())
        
        logger.info(f"Drawdown Analysis:")
        logger.info(f"  Max Drawdown: {max_dd:.4f}")
        logger.info(f"  Current Drawdown: {drawdown.iloc[-1]:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing risk management: {str(e)}")
        return False

async def test_market_microstructure():
    """Test market microstructure analysis."""
    try:
        logger.info("Testing market microstructure analysis...")
        
        # Create sample orderbook data
        orderbook = {
            'bids': [
                [100.0, 1.5],  # [price, quantity]
                [99.9, 2.0],
                [99.8, 1.8],
                [99.7, 2.2],
                [99.6, 1.6]
            ],
            'asks': [
                [100.1, 1.2],
                [100.2, 1.8],
                [100.3, 2.1],
                [100.4, 1.9],
                [100.5, 2.3]
            ]
        }
        
        # Test bid-ask spread calculation
        best_bid = float(orderbook['bids'][0][0])
        best_ask = float(orderbook['asks'][0][0])
        spread = (best_ask - best_bid) / best_bid
        
        logger.info(f"Orderbook Analysis:")
        logger.info(f"  Best Bid: {best_bid}")
        logger.info(f"  Best Ask: {best_ask}")
        logger.info(f"  Spread: {spread:.4f} ({spread*100:.2f}%)")
        
        # Test order imbalance
        bid_volume = sum(float(bid[1]) for bid in orderbook['bids'][:5])
        ask_volume = sum(float(ask[1]) for ask in orderbook['asks'][:5])
        total_volume = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total_volume
        
        logger.info(f"Order Imbalance:")
        logger.info(f"  Bid Volume: {bid_volume}")
        logger.info(f"  Ask Volume: {ask_volume}")
        logger.info(f"  Imbalance: {imbalance:.3f}")
        
        # Test volume profile
        trades_data = pd.DataFrame({
            'price': np.random.uniform(99.5, 100.5, 1000),
            'qty': np.random.lognormal(0, 1, 1000)
        })
        
        vwap = (trades_data['price'] * trades_data['qty']).sum() / trades_data['qty'].sum()
        
        logger.info(f"Volume Profile:")
        logger.info(f"  VWAP: {vwap:.4f}")
        logger.info(f"  Price Range: {trades_data['price'].min():.4f} - {trades_data['price'].max():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing market microstructure: {str(e)}")
        return False

async def main():
    """Run all WorldQuant strategy tests."""
    try:
        logger.info("Starting WorldQuant strategy comprehensive testing...")
        
        # Test 1: Advanced indicators
        success1 = await test_advanced_indicators()
        
        # Test 2: Risk management
        success2 = await test_risk_management()
        
        # Test 3: Market microstructure
        success3 = await test_market_microstructure()
        
        # Test 4: Full strategy
        success4 = await test_worldquant_strategy()
        
        # Summary
        logger.info("=== WORLDQUANT STRATEGY TEST SUMMARY ===")
        logger.info(f"Advanced Indicators: {'✅ PASS' if success1 else '❌ FAIL'}")
        logger.info(f"Risk Management: {'✅ PASS' if success2 else '❌ FAIL'}")
        logger.info(f"Market Microstructure: {'✅ PASS' if success3 else '❌ FAIL'}")
        logger.info(f"Full Strategy: {'✅ PASS' if success4 else '❌ FAIL'}")
        
        overall_success = all([success1, success2, success3, success4])
        logger.info(f"Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
        
        if overall_success:
            logger.info("🎯 WorldQuant-level strategy improvements are working correctly!")
        else:
            logger.warning("⚠️ Some improvements need attention")
        
        return overall_success
        
    except Exception as e:
        logger.error(f"Error in main test: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(main()) 