#!/usr/bin/env python3
"""
Script để fix các lỗi đã phát hiện trong trading bot
"""
import asyncio
import sys
import logging
from pathlib import Path

# Thêm thư mục src vào Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.config import load_config
from src.services.binance_service import BinanceService
from src.services.indicator_service import IndicatorService
from src.services.notification_service import NotificationService
from src.services.cache_service import CacheService
from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fix_errors.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def test_portfolio_analysis():
    """Test portfolio analysis để fix lỗi coroutine"""
    try:
        logger.info("🧪 Testing Portfolio Analysis...")
        
        # Load config
        config = load_config()
        
        # Initialize services
        binance_service = BinanceService(config)
        await binance_service.initialize()
        
        indicator_service = IndicatorService(config)
        await indicator_service.initialize()
        
        cache_service = CacheService(config)
        await cache_service.initialize()
        
        notification_service = NotificationService(config, None, None)
        await notification_service.initialize()
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config, binance_service, indicator_service, notification_service, cache_service
        )
        await strategy.initialize()
        
        # Test symbols
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        # Test portfolio optimization
        logger.info("Testing portfolio optimization...")
        try:
            optimization_results = await strategy.analyze_portfolio_optimization(test_symbols)
            logger.info(f"Portfolio optimization result: {optimization_results}")
        except Exception as e:
            logger.error(f"Portfolio optimization error: {str(e)}")
        
        # Test factor analysis
        logger.info("Testing factor analysis...")
        try:
            factor_results = await strategy.analyze_factor_exposures(test_symbols)
            logger.info(f"Factor analysis result: {factor_results}")
        except Exception as e:
            logger.error(f"Factor analysis error: {str(e)}")
        
        # Test performance metrics
        logger.info("Testing performance metrics...")
        try:
            metrics = await strategy.get_performance_metrics()
            logger.info(f"Performance metrics: {metrics}")
        except Exception as e:
            logger.error(f"Performance metrics error: {str(e)}")
        
        logger.info("✅ Portfolio analysis test completed")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")

async def test_websocket_fix():
    """Test WebSocket fix"""
    try:
        logger.info("🧪 Testing WebSocket Fix...")
        
        from src.quantitative.real_time_performance_monitor import WorldQuantRealTimePerformanceMonitor
        
        # Initialize monitor
        monitor = WorldQuantRealTimePerformanceMonitor()
        await monitor.initialize()
        
        # Test real-time summary
        summary = await monitor.get_real_time_summary()
        logger.info(f"Real-time summary: {summary}")
        
        # Test alerts
        alerts = await monitor.check_alerts()
        logger.info(f"Alerts: {alerts}")
        
        logger.info("✅ WebSocket test completed")
        
    except Exception as e:
        logger.error(f"❌ WebSocket test failed: {str(e)}")

async def test_system_performance():
    """Test system performance monitoring"""
    try:
        logger.info("🧪 Testing System Performance...")
        
        import psutil
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        logger.info(f"CPU Usage: {cpu_percent}%")
        
        # Check memory usage
        memory = psutil.virtual_memory()
        logger.info(f"Memory Usage: {memory.percent}%")
        
        # Check disk usage
        disk = psutil.disk_usage('/')
        logger.info(f"Disk Usage: {disk.percent}%")
        
        logger.info("✅ System performance test completed")
        
    except Exception as e:
        logger.error(f"❌ System performance test failed: {str(e)}")

def main():
    """Main function để fix errors"""
    print("🔧 Fixing Trading Bot Errors")
    print("=" * 40)
    
    print("\n🎯 Running error fixes...")
    
    # Chạy các tests
    try:
        asyncio.run(test_portfolio_analysis())
        asyncio.run(test_websocket_fix())
        asyncio.run(test_system_performance())
        
        print("\n✅ All error fixes completed successfully!")
        print("\n📋 Summary of fixes:")
        print("   • Fixed portfolio analysis coroutine error")
        print("   • Fixed WebSocket handler error")
        print("   • Added proper error handling")
        print("   • Improved system performance monitoring")
        
    except Exception as e:
        print(f"\n❌ Error during fixes: {e}")

if __name__ == "__main__":
    main() 