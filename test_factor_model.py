#!/usr/bin/env python3
"""
Test script for WorldQuant Factor Model implementation.
Tests multi-factor model, factor exposure calculation, risk attribution analysis, and sector/geographic risk exposure.
"""

import asyncio
import sys
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockIndicatorService:
    """Mock indicator service for testing."""
    
    async def get_klines(self, symbol: str, timeframe: str, limit: int = 100) -> Dict:
        """Mock klines data."""
        # Generate realistic mock data
        np.random.seed(42)  # For reproducible results
        
        # Generate price data with trend and volatility
        base_price = 100.0
        if symbol == 'BTCUSDT':
            base_price = 50000.0
        elif symbol == 'ETHUSDT':
            base_price = 3000.0
        elif symbol == 'ADAUSDT':
            base_price = 0.5
        elif symbol == 'SOLUSDT':
            base_price = 100.0
        
        prices = []
        current_price = base_price
        
        for i in range(limit):
            # Add trend and volatility
            trend = 0.001 * np.sin(i / 10)  # Small trend
            volatility = 0.02  # 2% volatility
            random_walk = np.random.normal(0, volatility)
            
            current_price *= (1 + trend + random_walk)
            prices.append(current_price)
        
        # Generate OHLCV data
        opens = [p * (1 + np.random.normal(0, 0.005)) for p in prices]
        highs = [max(o, p) * (1 + abs(np.random.normal(0, 0.01))) for o, p in zip(opens, prices)]
        lows = [min(o, p) * (1 - abs(np.random.normal(0, 0.01))) for o, p in zip(opens, prices)]
        volumes = [np.random.uniform(1000, 10000) for _ in prices]
        
        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        
        return {
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes,
            'returns': returns,
            'price': prices[-1] if prices else base_price
        }

class MockBinanceService:
    """Mock binance service for testing."""
    
    async def get_account_balance(self) -> Dict:
        """Mock account balance."""
        return {
            'USDT': {'total': 10000.0, 'available': 9500.0},
            'BTC': {'total': 0.5, 'available': 0.4},
            'ETH': {'total': 5.0, 'available': 4.8}
        }
    
    async def get_funding_rate(self, symbol: str) -> float:
        """Mock funding rate."""
        return np.random.normal(0.0001, 0.0002)

class MockNotificationService:
    """Mock notification service for testing."""
    
    async def send_notification(self, message: str) -> bool:
        """Mock notification sending."""
        logger.info(f"Mock notification: {message}")
        return True

async def test_factor_model():
    """Test WorldQuantFactorModel class."""
    logger.info("Testing WorldQuantFactorModel...")
    
    try:
        from src.quantitative.factor_model import WorldQuantFactorModel
        
        # Create config
        config = {
            'trading': {
                'statistical_significance_level': 0.05,
                'min_sample_size': 50
            }
        }
        
        # Initialize factor model
        factor_model = WorldQuantFactorModel(config)
        await factor_model.initialize()
        
        # Generate mock market data
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT']
        market_data = {}
        
        for symbol in symbols:
            # Generate mock data
            np.random.seed(42)
            prices = []
            volumes = []
            returns = []
            
            base_price = 100.0
            if symbol == 'BTCUSDT':
                base_price = 50000.0
            elif symbol == 'ETHUSDT':
                base_price = 3000.0
            elif symbol == 'ADAUSDT':
                base_price = 0.5
            elif symbol == 'SOLUSDT':
                base_price = 100.0
            
            current_price = base_price
            for i in range(100):
                trend = 0.001 * np.sin(i / 10)
                volatility = 0.02
                random_walk = np.random.normal(0, volatility)
                
                current_price *= (1 + trend + random_walk)
                prices.append(current_price)
                
                volume = np.random.uniform(1000, 10000)
                volumes.append(volume)
                
                if i > 0:
                    ret = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(ret)
            
            market_data[symbol] = {
                'price': prices[-1],
                'volume': volumes[-1],
                'returns': returns,
                'prices': prices
            }
        
        # Test factor calculations
        logger.info("Testing factor calculations...")
        
        # Test market factor
        market_factor = await factor_model.calculate_market_factor(symbols, market_data)
        logger.info(f"Market factor: {len(market_factor)} symbols calculated")
        
        # Test size factor
        size_factor = await factor_model.calculate_size_factor(symbols, market_data)
        logger.info(f"Size factor: {len(size_factor)} symbols calculated")
        
        # Test value factor
        value_factor = await factor_model.calculate_value_factor(symbols, market_data)
        logger.info(f"Value factor: {len(value_factor)} symbols calculated")
        
        # Test momentum factor
        momentum_factor = await factor_model.calculate_momentum_factor(symbols, market_data)
        logger.info(f"Momentum factor: {len(momentum_factor)} symbols calculated")
        
        # Test volatility factor
        volatility_factor = await factor_model.calculate_volatility_factor(symbols, market_data)
        logger.info(f"Volatility factor: {len(volatility_factor)} symbols calculated")
        
        # Test liquidity factor
        liquidity_factor = await factor_model.calculate_liquidity_factor(symbols, market_data)
        logger.info(f"Liquidity factor: {len(liquidity_factor)} symbols calculated")
        
        # Test all factors
        all_factors = await factor_model.calculate_all_factors(symbols, market_data)
        logger.info(f"All factors: {len(all_factors)} factor types calculated")
        
        # Test factor exposures
        factor_exposures = await factor_model.calculate_factor_exposures(symbols, market_data)
        logger.info(f"Factor exposures: {len(factor_exposures)} symbols calculated")
        
        # Test risk attribution analysis
        risk_attribution = await factor_model.perform_risk_attribution_analysis(symbols, market_data)
        logger.info("Risk attribution analysis completed")
        
        # Test sector analysis
        sector_analysis = await factor_model.analyze_sector_risk_exposure(symbols)
        logger.info("Sector risk exposure analysis completed")
        
        # Test geographic analysis
        geographic_analysis = await factor_model.analyze_geographic_risk_exposure(symbols)
        logger.info("Geographic risk exposure analysis completed")
        
        # Test factor summary
        factor_summary = await factor_model.get_factor_summary()
        logger.info("Factor summary generated")
        
        # Log results
        logger.info(f"Factor model test results:")
        logger.info(f"  Total factors: {factor_summary.get('total_factors', 0)}")
        logger.info(f"  Factor data points: {factor_summary.get('factor_data_points', 0)}")
        logger.info(f"  Diversification score: {risk_attribution.get('diversification_score', 0):.3f}")
        logger.info(f"  Total factor risk: {risk_attribution.get('total_factor_risk', 0):.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing WorldQuantFactorModel: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_enhanced_trading_strategy_with_factors():
    """Test enhanced trading strategy with factor model integration."""
    logger.info("Testing Enhanced Trading Strategy with Factor Model...")
    
    try:
        from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
        
        # Create mock services
        indicator_service = MockIndicatorService()
        binance_service = MockBinanceService()
        notification_service = MockNotificationService()
        
        # Create config
        config = {
            'trading': {
                'statistical_significance_level': 0.05,
                'min_sample_size': 50,
                'confidence_thresholds': {
                    'buy_base': 0.45,
                    'sell_base': 0.65,
                    'hold_base': 0.35
                }
            }
        }
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config=config,
            binance_service=binance_service,
            indicator_service=indicator_service,
            notification_service=notification_service
        )
        
        # Initialize strategy
        await strategy.initialize()
        
        # Test signal generation with factor analysis
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        for symbol in symbols:
            logger.info(f"Testing signal generation with factor analysis for {symbol}...")
            
            # Generate signal
            signal = await strategy.generate_signals(symbol, indicator_service)
            
            if signal:
                logger.info(f"Signal generated for {symbol}:")
                logger.info(f"  Action: {signal.get('action', 'unknown')}")
                logger.info(f"  Confidence: {signal.get('confidence', 0):.3f}")
                
                # Check factor analysis
                if 'factor_exposures' in signal:
                    factor_exposures = signal['factor_exposures']
                    logger.info(f"  Factor exposures: {len(factor_exposures)} factors")
                    for factor, exposure in factor_exposures.items():
                        logger.info(f"    {factor}: {exposure:.4f}")
                
                if 'factor_adjusted_confidence' in signal:
                    logger.info(f"  Factor-adjusted confidence: {signal['factor_adjusted_confidence']:.3f}")
                
                if 'factor_analysis' in signal:
                    factor_analysis = signal['factor_analysis']
                    logger.info(f"  Factor-adjusted action: {factor_analysis.get('factor_adjusted_action', 'unknown')}")
            else:
                logger.warning(f"No signal generated for {symbol}")
        
        # Test portfolio factor analysis
        logger.info("Testing portfolio factor analysis...")
        portfolio_analysis = await strategy.analyze_portfolio_factor_exposures(symbols)
        
        if portfolio_analysis:
            summary = portfolio_analysis.get('summary', {})
            logger.info(f"Portfolio analysis results:")
            logger.info(f"  Total symbols: {summary.get('total_symbols', 0)}")
            logger.info(f"  Total factors: {summary.get('total_factors', 0)}")
            logger.info(f"  Diversification score: {summary.get('diversification_score', 0):.3f}")
            logger.info(f"  Total factor risk: {summary.get('total_factor_risk', 0):.3f}")
        
        # Test factor model summary
        factor_summary = await strategy.get_factor_model_summary()
        logger.info("Factor model summary retrieved")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing enhanced trading strategy with factors: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_factor_exposure_calculation():
    """Test factor exposure calculation."""
    logger.info("Testing factor exposure calculation...")
    
    try:
        from src.quantitative.factor_model import WorldQuantFactorModel
        
        config = {'trading': {'statistical_significance_level': 0.05}}
        factor_model = WorldQuantFactorModel(config)
        await factor_model.initialize()
        
        # Generate test data
        symbols = ['BTCUSDT', 'ETHUSDT']
        market_data = {}
        
        for symbol in symbols:
            np.random.seed(42)
            prices = [100.0 + i * 0.1 + np.random.normal(0, 0.5) for i in range(100)]
            volumes = [np.random.uniform(1000, 10000) for _ in range(100)]
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            
            market_data[symbol] = {
                'price': prices[-1],
                'volume': volumes[-1],
                'returns': returns,
                'prices': prices
            }
        
        # Calculate factor exposures
        factor_exposures = await factor_model.calculate_factor_exposures(symbols, market_data)
        
        logger.info(f"Factor exposure calculation results:")
        for symbol, exposures in factor_exposures.items():
            logger.info(f"  {symbol}:")
            for factor, exposure in exposures.items():
                logger.info(f"    {factor}: {exposure:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing factor exposure calculation: {str(e)}")
        return False

async def test_risk_attribution_analysis():
    """Test risk attribution analysis."""
    logger.info("Testing risk attribution analysis...")
    
    try:
        from src.quantitative.factor_model import WorldQuantFactorModel
        
        config = {'trading': {'statistical_significance_level': 0.05}}
        factor_model = WorldQuantFactorModel(config)
        await factor_model.initialize()
        
        # Generate test data
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        market_data = {}
        
        for symbol in symbols:
            np.random.seed(42)
            prices = [100.0 + i * 0.1 + np.random.normal(0, 0.5) for i in range(100)]
            volumes = [np.random.uniform(1000, 10000) for _ in range(100)]
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            
            market_data[symbol] = {
                'price': prices[-1],
                'volume': volumes[-1],
                'returns': returns,
                'prices': prices
            }
        
        # Perform risk attribution analysis
        risk_attribution = await factor_model.perform_risk_attribution_analysis(symbols, market_data)
        
        logger.info(f"Risk attribution analysis results:")
        logger.info(f"  Total factor risk: {risk_attribution.get('total_factor_risk', 0):.4f}")
        logger.info(f"  Diversification score: {risk_attribution.get('diversification_score', 0):.3f}")
        logger.info(f"  VaR 95%: {risk_attribution.get('var_95', 0):.4f}")
        logger.info(f"  VaR 99%: {risk_attribution.get('var_99', 0):.4f}")
        
        factor_concentrations = risk_attribution.get('factor_concentrations', {})
        logger.info(f"  Factor concentrations:")
        for factor, concentration in factor_concentrations.items():
            logger.info(f"    {factor}: {concentration:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing risk attribution analysis: {str(e)}")
        return False

async def main():
    """Main test function."""
    logger.info("Starting Factor Model tests...")
    
    tests = [
        ("WorldQuantFactorModel", test_factor_model),
        ("Enhanced Trading Strategy with Factors", test_enhanced_trading_strategy_with_factors),
        ("Factor Exposure Calculation", test_factor_exposure_calculation),
        ("Risk Attribution Analysis", test_risk_attribution_analysis)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results[test_name] = result
            
            if result:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} test FAILED with exception: {str(e)}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Factor Model tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed")
    
    return passed == total

if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 