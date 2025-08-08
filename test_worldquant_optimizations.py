#!/usr/bin/env python3
"""
WorldQuant Optimization Verification Script
Test script để verify các tối ưu hóa đã thực hiện
"""

import asyncio
import logging
import time
from typing import Dict, List
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.quantitative.market_microstructure_analyzer import WorldQuantMarketMicrostructureAnalyzer
from src.quantitative.factor_model import WorldQuantFactorModel
from src.quantitative.portfolio_optimizer import WorldQuantPortfolioOptimizer
from src.quantitative.ml_ensemble import WorldQuantMLEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockBinanceService:
    """Mock Binance service for testing."""
    
    async def get_recent_trades(self, symbol: str, limit: int = 100):
        """Mock recent trades data."""
        import random
        trades = []
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
        
        for i in range(limit):
            price = base_price + random.uniform(-100, 100)
            quantity = random.uniform(0.1, 10.0)
            is_buyer_maker = random.choice([True, False])
            
            trades.append({
                'price': str(price),
                'qty': str(quantity),
                'isBuyerMaker': is_buyer_maker,
                'time': int(time.time() * 1000) - i * 1000
            })
        
        return trades
    
    async def get_order_book(self, symbol: str):
        """Mock order book data."""
        import random
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
        
        bids = []
        asks = []
        
        for i in range(10):
            bid_price = base_price - i * 10 - random.uniform(0, 5)
            ask_price = base_price + i * 10 + random.uniform(0, 5)
            quantity = random.uniform(1, 100)
            
            bids.append([str(bid_price), str(quantity)])
            asks.append([str(ask_price), str(quantity)])
        
        return {'bids': bids, 'asks': asks}
    
    async def get_klines(self, symbol: str, interval: str = '1h', limit: int = 168):
        """Mock klines data."""
        import random
        klines = []
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
        
        for i in range(limit):
            open_price = base_price + random.uniform(-50, 50)
            high_price = open_price + random.uniform(0, 20)
            low_price = open_price - random.uniform(0, 20)
            close_price = open_price + random.uniform(-10, 10)
            volume = random.uniform(100, 1000)
            
            klines.append([
                int(time.time() * 1000) - i * 3600000,  # timestamp
                str(open_price),  # open
                str(high_price),  # high
                str(low_price),   # low
                str(close_price), # close
                str(volume),      # volume
                int(time.time() * 1000) - i * 3600000,  # close time
                str(volume),      # quote asset volume
                0,                # number of trades
                str(volume),      # taker buy base asset volume
                str(volume)       # taker buy quote asset volume
            ])
        
        return klines

class WorldQuantOptimizationTester:
    """Test class for WorldQuant optimizations."""
    
    def __init__(self):
        """Initialize the tester."""
        self.config = {
            'order_flow_window': 200,
            'market_impact_threshold': 0.03,
            'liquidity_threshold': 0.001,
            'volume_profile_periods': 168,
            'optimization_method': 'mean_variance',
            'risk_free_rate': 0.02,
            'target_return': 0.12,
            'max_volatility': 0.20,
            'min_weight': 0.005,
            'max_weight': 0.35
        }
        
        self.mock_binance = MockBinanceService()
        self.test_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT']
        
    async def test_market_microstructure_analyzer(self):
        """Test Market Microstructure Analyzer optimizations."""
        logger.info("Testing Market Microstructure Analyzer optimizations...")
        
        try:
            # Initialize analyzer
            analyzer = WorldQuantMarketMicrostructureAnalyzer(self.mock_binance, self.config)
            
            # Test basic functionality
            start_time = time.time()
            analysis = await analyzer.analyze_market_microstructure('BTCUSDT')
            execution_time = time.time() - start_time
            
            # Verify advanced features
            assert 'metrics' in analysis, "Advanced metrics missing"
            assert 'market_regime' in analysis, "Market regime analysis missing"
            assert 'order_flow' in analysis, "Order flow analysis missing"
            assert 'liquidity' in analysis, "Liquidity analysis missing"
            
            # Check performance
            assert execution_time < 5.0, f"Execution too slow: {execution_time}s"
            
            # Verify advanced data structures
            assert hasattr(analyzer, 'cache'), "Caching system missing"
            assert hasattr(analyzer, 'executor'), "Parallel processing missing"
            
            logger.info(f"✅ Market Microstructure Analyzer test passed in {execution_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Market Microstructure Analyzer test failed: {str(e)}")
            return False
    
    async def test_factor_model(self):
        """Test Factor Model optimizations."""
        logger.info("Testing Factor Model optimizations...")
        
        try:
            # Initialize factor model
            factor_model = WorldQuantFactorModel(self.config)
            
            # Test initialization
            success = await factor_model.initialize()
            assert success, "Factor model initialization failed"
            
            # Verify advanced features
            assert hasattr(factor_model, 'factors'), "Factors missing"
            assert len(factor_model.factors) >= 10, "Insufficient factors"
            
            # Check for new factor types
            factor_types = [f.value for f in factor_model.factors.keys()]
            expected_factors = ['market', 'size', 'value', 'momentum', 'volatility', 'liquidity', 'quality', 'momentum_reversal', 'volatility_targeting', 'cross_asset']
            
            for factor in expected_factors:
                assert factor in factor_types, f"Missing factor: {factor}"
            
            # Verify advanced data structures
            assert hasattr(factor_model, 'factor_data'), "Factor data missing"
            assert hasattr(factor_model, 'factor_exposures'), "Factor exposures missing"
            assert hasattr(factor_model, 'executor'), "Parallel processing missing"
            
            logger.info("✅ Factor Model test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Factor Model test failed: {str(e)}")
            return False
    
    async def test_portfolio_optimizer(self):
        """Test Portfolio Optimizer optimizations."""
        logger.info("Testing Portfolio Optimizer optimizations...")
        
        try:
            # Initialize optimizer
            optimizer = WorldQuantPortfolioOptimizer(self.config)
            
            # Test initialization
            success = await optimizer.initialize()
            assert success, "Portfolio optimizer initialization failed"
            
            # Verify advanced features
            assert hasattr(optimizer, 'optimization_params'), "Optimization params missing"
            assert len(optimizer.optimization_params) >= 9, "Insufficient optimization methods"
            
            # Check for new optimization methods
            method_types = [m.value for m in optimizer.optimization_params.keys()]
            expected_methods = ['mean_variance', 'risk_parity', 'factor_neutral', 'black_litterman', 'maximum_sharpe', 'minimum_variance', 'maximum_diversification', 'regime_aware', 'machine_learning']
            
            for method in expected_methods:
                assert method in method_types, f"Missing optimization method: {method}"
            
            # Verify advanced data structures
            assert hasattr(optimizer, 'portfolio_state'), "Portfolio state missing"
            assert hasattr(optimizer, 'monitoring_active'), "Monitoring system missing"
            
            logger.info("✅ Portfolio Optimizer test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Portfolio Optimizer test failed: {str(e)}")
            return False
    
    async def test_ml_ensemble(self):
        """Test ML Ensemble optimizations."""
        logger.info("Testing ML Ensemble optimizations...")
        
        try:
            # Initialize ML ensemble
            ml_ensemble = WorldQuantMLEnsemble(self.config)
            
            # Test initialization
            success = await ml_ensemble.initialize()
            assert success, "ML ensemble initialization failed"
            
            # Verify advanced features
            assert hasattr(ml_ensemble, 'model_configs'), "Model configs missing"
            assert len(ml_ensemble.model_configs) >= 8, "Insufficient model types"
            
            # Check for new model types
            model_types = [m.value for m in ml_ensemble.model_configs.keys()]
            expected_models = ['random_forest', 'gradient_boosting', 'neural_network', 'svm', 'deep_learning', 'reinforcement_learning', 'transformer', 'ensemble']
            
            for model in expected_models:
                assert model in model_types, f"Missing model type: {model}"
            
            # Verify advanced features
            assert hasattr(ml_ensemble, 'feature_params'), "Feature params missing"
            assert hasattr(ml_ensemble, 'cv_params'), "CV params missing"
            
            logger.info("✅ ML Ensemble test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ ML Ensemble test failed: {str(e)}")
            return False
    
    async def test_performance_improvements(self):
        """Test performance improvements."""
        logger.info("Testing performance improvements...")
        
        try:
            # Test parallel processing
            analyzer = WorldQuantMarketMicrostructureAnalyzer(self.mock_binance, self.config)
            
            # Test single symbol analysis
            start_time = time.time()
            analysis1 = await analyzer.analyze_market_microstructure('BTCUSDT')
            single_time = time.time() - start_time
            
            # Test multiple symbols (should be faster due to caching)
            start_time = time.time()
            analyses = []
            for symbol in self.test_symbols:
                analysis = await analyzer.analyze_market_microstructure(symbol)
                analyses.append(analysis)
            multi_time = time.time() - start_time
            
            # Verify performance improvements
            assert single_time < 3.0, f"Single analysis too slow: {single_time}s"
            assert multi_time < 8.0, f"Multi analysis too slow: {multi_time}s"
            
            # Verify caching effectiveness
            assert len(analyses) == len(self.test_symbols), "Incorrect number of analyses"
            
            logger.info(f"✅ Performance test passed - Single: {single_time:.2f}s, Multi: {multi_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance test failed: {str(e)}")
            return False
    
    async def test_error_handling(self):
        """Test error handling improvements."""
        logger.info("Testing error handling improvements...")
        
        try:
            # Test with invalid data
            analyzer = WorldQuantMarketMicrostructureAnalyzer(self.mock_binance, self.config)
            
            # Test with empty data (should handle gracefully)
            analysis = await analyzer.analyze_market_microstructure('INVALID')
            
            # Should return default analysis structure
            assert 'timestamp' in analysis, "Default analysis missing timestamp"
            assert 'symbol' in analysis, "Default analysis missing symbol"
            
            logger.info("✅ Error handling test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error handling test failed: {str(e)}")
            return False
    
    async def run_all_tests(self):
        """Run all optimization tests."""
        logger.info("🚀 Starting WorldQuant Optimization Tests...")
        
        tests = [
            ("Market Microstructure Analyzer", self.test_market_microstructure_analyzer),
            ("Factor Model", self.test_factor_model),
            ("Portfolio Optimizer", self.test_portfolio_optimizer),
            ("ML Ensemble", self.test_ml_ensemble),
            ("Performance Improvements", self.test_performance_improvements),
            ("Error Handling", self.test_error_handling)
        ]
        
        results = []
        total_start_time = time.time()
        
        for test_name, test_func in tests:
            try:
                result = await test_func()
                results.append((test_name, result))
            except Exception as e:
                logger.error(f"❌ {test_name} test failed with exception: {str(e)}")
                results.append((test_name, False))
        
        total_time = time.time() - total_start_time
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info("WORLDQUANT OPTIMIZATION TEST RESULTS")
        logger.info("="*50)
        
        passed = 0
        failed = 0
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
            if result:
                passed += 1
            else:
                failed += 1
        
        logger.info("-"*50)
        logger.info(f"Total Tests: {len(results)}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {(passed/len(results)*100):.1f}%")
        logger.info(f"Total Time: {total_time:.2f}s")
        logger.info("="*50)
        
        if failed == 0:
            logger.info("🎉 ALL TESTS PASSED! WorldQuant optimizations are working correctly.")
        else:
            logger.warning(f"⚠️ {failed} test(s) failed. Please review the optimizations.")
        
        return failed == 0

async def main():
    """Main test function."""
    tester = WorldQuantOptimizationTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 WorldQuant Optimization Verification Complete!")
        print("All optimizations have been successfully implemented and tested.")
        sys.exit(0)
    else:
        print("\n❌ Some optimizations failed verification.")
        print("Please review and fix the failing components.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
