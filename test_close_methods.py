#!/usr/bin/env python3
"""
Test close methods for all quantitative components
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_quantitative_integration_close():
    """Test QuantitativeIntegration close method."""
    try:
        from src.quantitative.integration import QuantitativeIntegration
        
        config = {'quantitative_integration_enabled': True}
        integration = QuantitativeIntegration(config)
        await integration.initialize()
        
        logger.info("✅ Testing QuantitativeIntegration close method...")
        await integration.close()
        logger.info("✅ QuantitativeIntegration close method works")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing QuantitativeIntegration close: {str(e)}")
        return False

async def test_risk_manager_close():
    """Test RiskManager close method."""
    try:
        from src.quantitative.risk_manager import RiskManager
        
        config = {'risk': {'var_confidence_level': 0.95}}
        risk_manager = RiskManager(config)
        await risk_manager.initialize()
        
        logger.info("✅ Testing RiskManager close method...")
        await risk_manager.close()
        logger.info("✅ RiskManager close method works")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing RiskManager close: {str(e)}")
        return False

async def test_portfolio_optimizer_close():
    """Test WorldQuantPortfolioOptimizer close method."""
    try:
        from src.quantitative.portfolio_optimizer import WorldQuantPortfolioOptimizer
        
        config = {'portfolio': {'max_positions': 10}}
        optimizer = WorldQuantPortfolioOptimizer(config)
        await optimizer.initialize()
        
        logger.info("✅ Testing WorldQuantPortfolioOptimizer close method...")
        await optimizer.close()
        logger.info("✅ WorldQuantPortfolioOptimizer close method works")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing WorldQuantPortfolioOptimizer close: {str(e)}")
        return False

async def test_market_microstructure_close():
    """Test MarketMicrostructureAnalyzer close method."""
    try:
        from src.quantitative.market_microstructure import MarketMicrostructureAnalyzer
        
        config = {'market_microstructure': {'enabled': True}}
        analyzer = MarketMicrostructureAnalyzer(config)
        await analyzer.initialize()
        
        logger.info("✅ Testing MarketMicrostructureAnalyzer close method...")
        await analyzer.close()
        logger.info("✅ MarketMicrostructureAnalyzer close method works")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing MarketMicrostructureAnalyzer close: {str(e)}")
        return False

async def test_backtesting_engine_close():
    """Test AdvancedBacktestingEngine close method."""
    try:
        from src.quantitative.backtesting_engine import AdvancedBacktestingEngine
        
        config = {'backtesting': {'enabled': True}}
        engine = AdvancedBacktestingEngine(config)
        await engine.initialize()
        
        logger.info("✅ Testing AdvancedBacktestingEngine close method...")
        await engine.close()
        logger.info("✅ AdvancedBacktestingEngine close method works")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing AdvancedBacktestingEngine close: {str(e)}")
        return False

async def test_factor_model_close():
    """Test WorldQuantFactorModel close method."""
    try:
        from src.quantitative.factor_model import WorldQuantFactorModel
        
        config = {'factor_model': {'enabled': True}}
        model = WorldQuantFactorModel(config)
        await model.initialize()
        
        logger.info("✅ Testing WorldQuantFactorModel close method...")
        await model.close()
        logger.info("✅ WorldQuantFactorModel close method works")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing WorldQuantFactorModel close: {str(e)}")
        return False

async def main():
    """Test all close methods."""
    logger.info("🧪 Testing close methods for all quantitative components...")
    logger.info("=" * 60)
    
    tests = [
        test_quantitative_integration_close(),
        test_risk_manager_close(),
        test_portfolio_optimizer_close(),
        test_market_microstructure_close(),
        test_backtesting_engine_close(),
        test_factor_model_close(),
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    logger.info("\n📊 Test Results:")
    logger.info("=" * 30)
    
    test_names = [
        "QuantitativeIntegration",
        "RiskManager", 
        "WorldQuantPortfolioOptimizer",
        "MarketMicrostructureAnalyzer",
        "AdvancedBacktestingEngine",
        "WorldQuantFactorModel"
    ]
    
    passed = 0
    for i, (name, result) in enumerate(zip(test_names, results)):
        if isinstance(result, Exception):
            logger.error(f"❌ {name}: {str(result)}")
        elif result:
            logger.info(f"✅ {name}: PASS")
            passed += 1
        else:
            logger.error(f"❌ {name}: FAIL")
    
    logger.info(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 All close methods are working correctly!")
        logger.info("✅ The 'QuantitativeIntegration' object has no attribute 'close' error should be fixed!")
    else:
        logger.warning("⚠️ Some close methods are still failing")
    
    return passed == len(tests)

if __name__ == "__main__":
    asyncio.run(main()) 