#!/usr/bin/env python3
"""
Test script for WorldQuant ML Ensemble implementation.
Tests ensemble ML, feature engineering, cross-validation, and model interpretability.
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

async def test_ml_ensemble():
    """Test WorldQuantMLEnsemble class."""
    logger.info("Testing WorldQuantMLEnsemble...")
    
    try:
        from src.quantitative.ml_ensemble import WorldQuantMLEnsemble
        
        # Create config
        config = {
            'trading': {
                'statistical_significance_level': 0.05,
                'min_sample_size': 50
            }
        }
        
        # Initialize ML ensemble
        ml_ensemble = WorldQuantMLEnsemble(config)
        await ml_ensemble.initialize()
        
        # Generate mock market data
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT']
        
        for symbol in symbols:
            logger.info(f"Testing ML ensemble for {symbol}...")
            
            # Generate mock data
            np.random.seed(42)
            prices = []
            volumes = []
            
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
            for i in range(200):  # More data for ML training
                trend = 0.001 * np.sin(i / 10)
                volatility = 0.02
                random_walk = np.random.normal(0, volatility)
                
                current_price *= (1 + trend + random_walk)
                prices.append(current_price)
                
                volume = np.random.uniform(1000, 10000)
                volumes.append(volume)
            
            # Create DataFrame
            df = pd.DataFrame({
                'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'close': prices,
                'volume': volumes
            })
            
            # Engineer features
            df_features = ml_ensemble.engineer_features(df)
            
            if not df_features.empty:
                logger.info(f"Feature engineering completed: {df_features.shape[1]} features")
                
                # Train ensemble
                training_results = await ml_ensemble.train_ensemble(df_features)
                
                if training_results:
                    logger.info(f"ML training completed for {symbol}")
                    
                    # Test predictions
                    predictions = await ml_ensemble.predict_ensemble(df_features.tail(10))
                    
                    if predictions:
                        logger.info(f"ML predictions generated for {symbol}")
                        logger.info(f"  Ensemble prediction: {predictions['ensemble_prediction'][-1]:.4f}")
                        logger.info(f"  Confidence: {predictions['confidence'][-1]:.3f}")
                        
                        # Test interpretability
                        interpretability = ml_ensemble.get_model_interpretability(df_features, 'random_forest')
                        
                        if interpretability:
                            logger.info(f"Model interpretability completed for {symbol}")
                            top_features = list(interpretability['feature_importance'].keys())[:5]
                            logger.info(f"  Top 5 features: {top_features}")
                    else:
                        logger.warning(f"No ML predictions for {symbol}")
                else:
                    logger.warning(f"ML training failed for {symbol}")
            else:
                logger.warning(f"No features available for {symbol}")
        
        # Test ML summary
        ml_summary = await ml_ensemble.get_ml_summary()
        logger.info("ML ensemble summary generated")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing WorldQuantMLEnsemble: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_enhanced_trading_strategy_with_ml():
    """Test enhanced trading strategy with ML ensemble integration."""
    logger.info("Testing Enhanced Trading Strategy with ML Ensemble...")
    
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
        
        # Test signal generation with ML analysis
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        for symbol in symbols:
            logger.info(f"Testing signal generation with ML analysis for {symbol}...")
            
            # Generate signal
            signal = await strategy.generate_signals(symbol, indicator_service)
            
            if signal:
                logger.info(f"Signal generated for {symbol}:")
                logger.info(f"  Action: {signal.get('action', 'unknown')}")
                logger.info(f"  Confidence: {signal.get('confidence', 0):.3f}")
                
                # Check ML analysis
                if 'ml_predictions' in signal:
                    ml_predictions = signal['ml_predictions']
                    logger.info(f"  ML ensemble prediction: {ml_predictions.get('ensemble_prediction', 0):.4f}")
                    logger.info(f"  ML confidence: {ml_predictions.get('confidence', 0):.3f}")
                    
                    individual_predictions = ml_predictions.get('individual_predictions', {})
                    logger.info(f"  Individual model predictions: {len(individual_predictions)} models")
                    for model, pred in individual_predictions.items():
                        logger.info(f"    {model}: {pred:.4f}")
                
                if 'ml_analysis' in signal:
                    ml_analysis = signal['ml_analysis']
                    logger.info(f"  ML-adjusted action: {ml_analysis.get('ml_adjusted_action', 'unknown')}")
                    logger.info(f"  ML-adjusted confidence: {ml_analysis.get('ml_adjusted_confidence', 0):.3f}")
                    logger.info(f"  Prediction strength: {ml_analysis.get('prediction_strength', 0):.4f}")
                    logger.info(f"  Model agreement: {ml_analysis.get('model_agreement', 0):.3f}")
            else:
                logger.warning(f"No signal generated for {symbol}")
        
        # Test ML model training
        logger.info("Testing ML model training...")
        training_results = await strategy.train_ml_models(symbols)
        
        if training_results:
            logger.info(f"ML training completed for {len(training_results)} symbols")
            for symbol, results in training_results.items():
                performance = results.get('performance_metrics', {})
                logger.info(f"  {symbol}: {len(performance)} models trained")
        else:
            logger.warning("ML training failed")
        
        # Test ML interpretability
        logger.info("Testing ML model interpretability...")
        for symbol in symbols[:2]:  # Test first 2 symbols
            interpretability = await strategy.get_ml_model_interpretability(symbol, 'random_forest')
            
            if interpretability:
                feature_importance = interpretability.get('feature_importance', {})
                logger.info(f"  {symbol}: {len(feature_importance)} features analyzed")
                top_features = list(feature_importance.keys())[:3]
                logger.info(f"    Top features: {top_features}")
            else:
                logger.warning(f"No interpretability results for {symbol}")
        
        # Test ML summary
        ml_summary = await strategy.get_ml_summary()
        logger.info("ML summary retrieved")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing enhanced trading strategy with ML: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_feature_engineering():
    """Test feature engineering capabilities."""
    logger.info("Testing feature engineering...")
    
    try:
        from src.quantitative.ml_ensemble import WorldQuantMLEnsemble
        
        config = {'trading': {'statistical_significance_level': 0.05}}
        ml_ensemble = WorldQuantMLEnsemble(config)
        await ml_ensemble.initialize()
        
        # Generate test data
        np.random.seed(42)
        n_samples = 100
        
        df = pd.DataFrame({
            'open': np.random.uniform(100, 200, n_samples),
            'high': np.random.uniform(200, 300, n_samples),
            'low': np.random.uniform(50, 100, n_samples),
            'close': np.random.uniform(100, 200, n_samples),
            'volume': np.random.uniform(1000, 10000, n_samples)
        })
        
        # Engineer features
        df_features = ml_ensemble.engineer_features(df)
        
        logger.info(f"Feature engineering results:")
        logger.info(f"  Original features: {df.shape[1]}")
        logger.info(f"  Engineered features: {df_features.shape[1]}")
        logger.info(f"  Feature types: {list(df_features.columns)}")
        
        # Check for specific feature types
        feature_types = {
            'price_features': ['returns', 'log_returns', 'price_level'],
            'volume_features': ['volume_ma', 'volume_ratio', 'volume_momentum'],
            'volatility_features': ['volatility_20', 'high_low_ratio_20'],
            'momentum_features': ['rsi', 'macd', 'stoch_k'],
            'technical_indicators': ['bb_position', 'williams_r', 'cci']
        }
        
        for feature_type, expected_features in feature_types.items():
            found_features = [f for f in expected_features if f in df_features.columns]
            logger.info(f"  {feature_type}: {len(found_features)}/{len(expected_features)} features found")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing feature engineering: {str(e)}")
        return False

async def test_cross_validation():
    """Test time-series cross-validation."""
    logger.info("Testing time-series cross-validation...")
    
    try:
        from src.quantitative.ml_ensemble import WorldQuantMLEnsemble
        
        config = {'trading': {'statistical_significance_level': 0.05}}
        ml_ensemble = WorldQuantMLEnsemble(config)
        await ml_ensemble.initialize()
        
        # Generate test data
        np.random.seed(42)
        n_samples = 200
        
        df = pd.DataFrame({
            'open': np.random.uniform(100, 200, n_samples),
            'high': np.random.uniform(200, 300, n_samples),
            'low': np.random.uniform(50, 100, n_samples),
            'close': np.random.uniform(100, 200, n_samples),
            'volume': np.random.uniform(1000, 10000, n_samples)
        })
        
        # Engineer features
        df_features = ml_ensemble.engineer_features(df)
        
        if not df_features.empty:
            # Test training with cross-validation
            training_results = await ml_ensemble.train_ensemble(df_features)
            
            if training_results:
                logger.info(f"Cross-validation results:")
                
                for model_name, results in training_results.items():
                    train_scores = results.get('train_scores', [])
                    val_scores = results.get('val_scores', [])
                    
                    if train_scores and val_scores:
                        mean_train = np.mean(train_scores)
                        mean_val = np.mean(val_scores)
                        std_val = np.std(val_scores)
                        
                        logger.info(f"  {model_name}:")
                        logger.info(f"    Mean train score: {mean_train:.4f}")
                        logger.info(f"    Mean val score: {mean_val:.4f}")
                        logger.info(f"    Val score std: {std_val:.4f}")
                        logger.info(f"    CV folds: {len(val_scores)}")
                
                return True
            else:
                logger.warning("Training results not available")
                return False
        else:
            logger.warning("No features available for cross-validation")
            return False
            
    except Exception as e:
        logger.error(f"Error testing cross-validation: {str(e)}")
        return False

async def test_model_interpretability():
    """Test model interpretability with SHAP values."""
    logger.info("Testing model interpretability...")
    
    try:
        from src.quantitative.ml_ensemble import WorldQuantMLEnsemble
        
        config = {'trading': {'statistical_significance_level': 0.05}}
        ml_ensemble = WorldQuantMLEnsemble(config)
        await ml_ensemble.initialize()
        
        # Generate test data
        np.random.seed(42)
        n_samples = 100
        
        df = pd.DataFrame({
            'open': np.random.uniform(100, 200, n_samples),
            'high': np.random.uniform(200, 300, n_samples),
            'low': np.random.uniform(50, 100, n_samples),
            'close': np.random.uniform(100, 200, n_samples),
            'volume': np.random.uniform(1000, 10000, n_samples)
        })
        
        # Engineer features
        df_features = ml_ensemble.engineer_features(df)
        
        if not df_features.empty:
            # Test interpretability for different models
            models_to_test = ['random_forest', 'gradient_boosting']
            
            for model_name in models_to_test:
                logger.info(f"Testing interpretability for {model_name}...")
                
                interpretability = ml_ensemble.get_model_interpretability(df_features, model_name)
                
                if interpretability:
                    feature_importance = interpretability.get('feature_importance', {})
                    
                    logger.info(f"  {model_name} interpretability results:")
                    logger.info(f"    Total features: {len(feature_importance)}")
                    
                    # Show top 5 features
                    top_features = list(feature_importance.items())[:5]
                    for feature, importance in top_features:
                        logger.info(f"      {feature}: {importance:.4f}")
                    
                    if 'shap_values' in interpretability:
                        shap_values = interpretability['shap_values']
                        logger.info(f"    SHAP values shape: {shap_values.shape}")
                else:
                    logger.warning(f"No interpretability results for {model_name}")
            
            return True
        else:
            logger.warning("No features available for interpretability")
            return False
            
    except Exception as e:
        logger.error(f"Error testing model interpretability: {str(e)}")
        return False

async def main():
    """Main test function."""
    logger.info("Starting ML Ensemble tests...")
    
    tests = [
        ("WorldQuantMLEnsemble", test_ml_ensemble),
        ("Enhanced Trading Strategy with ML", test_enhanced_trading_strategy_with_ml),
        ("Feature Engineering", test_feature_engineering),
        ("Cross-Validation", test_cross_validation),
        ("Model Interpretability", test_model_interpretability)
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
        logger.info("🎉 All ML Ensemble tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed")
    
    return passed == total

if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 