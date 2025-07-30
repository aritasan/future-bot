import asyncio
import sys
import os
import time
import logging

# Set event loop policy for Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.core.config import load_config
from src.services.binance_service import BinanceService
from src.services.indicator_service import IndicatorService
from src.services.notification_service import NotificationService
from src.strategies.enhanced_trading_strategy import EnhancedTradingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_performance_optimization():
    """Test the performance optimizations in the trading strategy."""
    try:
        print("🚀 Testing Performance Optimizations")
        print("=" * 50)
        
        # Load configuration
        config = load_config()
        print("✅ Configuration loaded")
        
        # Initialize services
        binance_service = BinanceService(config)
        await binance_service.initialize()
        print("✅ Binance service initialized")
        
        indicator_service = IndicatorService(config)
        await indicator_service.initialize()
        print("✅ Indicator service initialized")
        
        notification_service = NotificationService(config)
        
        # Initialize strategy
        strategy = EnhancedTradingStrategy(config, binance_service, indicator_service, notification_service)
        await strategy.initialize()
        print("✅ Strategy initialized")
        
        # Test symbols
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT']
        
        print(f"\n📊 Testing with {len(test_symbols)} symbols...")
        
        # Test 1: Single symbol processing
        print("\n🔍 Test 1: Single Symbol Processing")
        start_time = time.time()
        
        for symbol in test_symbols[:3]:  # Test first 3 symbols
            signal = await strategy.generate_signals(symbol, indicator_service)
            if signal:
                print(f"✅ {symbol}: Signal generated")
            else:
                print(f"❌ {symbol}: No signal")
        
        single_time = time.time() - start_time
        print(f"⏱️  Single processing time: {single_time:.2f}s")
        
        # Test 2: Batch processing
        print("\n🔍 Test 2: Batch Processing")
        start_time = time.time()
        
        batch_results = await strategy._batch_process_symbols(test_symbols, indicator_service)
        batch_time = time.time() - start_time
        
        print(f"✅ Batch processing completed: {len(batch_results)} symbols")
        print(f"⏱️  Batch processing time: {batch_time:.2f}s")
        
        # Test 3: Cache performance
        print("\n🔍 Test 3: Cache Performance")
        
        # First run (cache miss)
        start_time = time.time()
        signal1 = await strategy.generate_signals('BTCUSDT', indicator_service)
        first_run_time = time.time() - start_time
        
        # Second run (cache hit)
        start_time = time.time()
        signal2 = await strategy.generate_signals('BTCUSDT', indicator_service)
        second_run_time = time.time() - start_time
        
        print(f"⏱️  First run (cache miss): {first_run_time:.2f}s")
        print(f"⏱️  Second run (cache hit): {second_run_time:.2f}s")
        print(f"🚀 Speed improvement: {((first_run_time - second_run_time) / first_run_time * 100):.1f}%")
        
        # Test 4: Performance metrics
        print("\n🔍 Test 4: Performance Metrics")
        metrics = await strategy.get_performance_metrics()
        
        print(f"📈 API Calls: {metrics.get('api_calls', 0)}")
        print(f"📈 Cache Hits: {metrics.get('cache_hits', 0)}")
        print(f"📈 Cache Misses: {metrics.get('cache_misses', 0)}")
        print(f"📈 Cache Hit Rate: {metrics.get('cache_hit_rate', 0):.1f}%")
        print(f"📈 Cache Size: {metrics.get('cache_size', 0)}")
        print(f"📈 Avg Processing Time: {metrics.get('avg_processing_time', 0):.2f}s")
        
        # Test 5: Memory optimization
        print("\n🔍 Test 5: Memory Optimization")
        await strategy._optimize_memory_usage()
        print("✅ Memory optimization completed")
        
        # Test 6: Cache optimization
        print("\n🔍 Test 6: Cache Optimization")
        await strategy.optimize_cache_settings()
        print("✅ Cache optimization completed")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 PERFORMANCE OPTIMIZATION SUMMARY")
        print("=" * 50)
        
        if batch_time < single_time:
            print(f"✅ Batch processing is {(single_time - batch_time) / single_time * 100:.1f}% faster")
        else:
            print(f"⚠️  Batch processing is {(batch_time - single_time) / single_time * 100:.1f}% slower")
        
        if second_run_time < first_run_time:
            print(f"✅ Caching provides {((first_run_time - second_run_time) / first_run_time * 100):.1f}% speed improvement")
        else:
            print("⚠️  Caching not providing expected improvement")
        
        cache_hit_rate = metrics.get('cache_hit_rate', 0)
        if cache_hit_rate > 50:
            print(f"✅ Good cache hit rate: {cache_hit_rate:.1f}%")
        else:
            print(f"⚠️  Low cache hit rate: {cache_hit_rate:.1f}%")
        
        avg_time = metrics.get('avg_processing_time', 0)
        if avg_time < 2.0:
            print(f"✅ Good average processing time: {avg_time:.2f}s")
        else:
            print(f"⚠️  High average processing time: {avg_time:.2f}s")
        
        print("\n🎯 RECOMMENDATIONS:")
        if cache_hit_rate < 50:
            print("- Consider increasing cache TTL for better hit rates")
        if avg_time > 2.0:
            print("- Consider reducing API calls or optimizing data processing")
        if batch_time > single_time:
            print("- Consider adjusting batch size for better performance")
        
        print("\n✅ Performance optimization test completed!")
        
    except Exception as e:
        logger.error(f"Error in performance test: {str(e)}")
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_performance_optimization()) 