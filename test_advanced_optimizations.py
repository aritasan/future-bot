import asyncio
import sys
import os
import time
import logging
import gc
import psutil

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

async def test_advanced_optimizations():
    """Test advanced optimizations including memory management and error handling."""
    try:
        print("🚀 Testing Advanced Optimizations")
        print("=" * 60)
        
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
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT']
        
        print(f"\n📊 Testing with {len(test_symbols)} symbols...")
        
        # Test 1: Memory Usage Monitoring
        print("\n🔍 Test 1: Memory Usage Monitoring")
        initial_memory = strategy._memory_monitor.get_memory_usage()
        print(f"Initial memory usage: {initial_memory['current_mb']:.1f}MB ({initial_memory['percent']:.1f}%)")
        
        # Test 2: LRU Cache Performance
        print("\n🔍 Test 2: LRU Cache Performance")
        cache_stats = strategy._cache.get_stats()
        print(f"Cache size: {cache_stats['size']}/{cache_stats['maxsize']} ({cache_stats['usage_percent']:.1f}%)")
        
        # Test 3: Error Handling
        print("\n🔍 Test 3: Error Handling")
        print(f"Initial error count: {strategy._error_count}")
        
        # Test 4: Performance with Increased TTL
        print("\n🔍 Test 4: Performance with Increased TTL")
        start_time = time.time()
        
        # Process symbols multiple times to test caching
        for i in range(3):
            print(f"  Round {i+1}: Processing {len(test_symbols)} symbols...")
            round_start = time.time()
            
            for symbol in test_symbols:
                signal = await strategy.generate_signals(symbol, indicator_service)
                if signal:
                    print(f"    ✅ {symbol}: Signal generated")
                else:
                    print(f"    ❌ {symbol}: No signal")
            
            round_time = time.time() - round_start
            print(f"  Round {i+1} time: {round_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f}s")
        
        # Test 5: Memory Usage After Processing
        print("\n🔍 Test 5: Memory Usage After Processing")
        final_memory = strategy._memory_monitor.get_memory_usage()
        print(f"Final memory usage: {final_memory['current_mb']:.1f}MB ({final_memory['percent']:.1f}%)")
        print(f"Memory increase: {final_memory['current_mb'] - initial_memory['current_mb']:.1f}MB")
        
        # Test 6: Cache Performance After Processing
        print("\n🔍 Test 6: Cache Performance After Processing")
        final_cache_stats = strategy._cache.get_stats()
        print(f"Final cache size: {final_cache_stats['size']}/{final_cache_stats['maxsize']} ({final_cache_stats['usage_percent']:.1f}%)")
        print(f"Average access count: {final_cache_stats['avg_access_count']:.1f}")
        
        # Test 7: Comprehensive Performance Metrics
        print("\n🔍 Test 7: Comprehensive Performance Metrics")
        metrics = await strategy.get_performance_metrics()
        
        print(f"📈 API Calls: {metrics.get('api_calls', 0)}")
        print(f"📈 API Errors: {metrics.get('api_errors', 0)}")
        print(f"📈 Error Rate: {metrics.get('error_rate', 0):.1f}%")
        print(f"📈 Cache Hits: {metrics.get('cache_hits', 0)}")
        print(f"📈 Cache Misses: {metrics.get('cache_misses', 0)}")
        print(f"📈 Cache Hit Rate: {metrics.get('cache_hit_rate', 0):.1f}%")
        print(f"📈 Cache Usage: {metrics.get('cache_usage_percent', 0):.1f}%")
        print(f"📈 Avg Processing Time: {metrics.get('avg_processing_time', 0):.2f}s")
        print(f"📈 Memory Usage: {metrics.get('memory_percent', 0):.1f}%")
        print(f"📈 Optimization Suggestions: {metrics.get('optimization_suggestions', 0)}")
        
        # Test 8: Memory Cleanup
        print("\n🔍 Test 8: Memory Cleanup")
        if strategy._memory_monitor.should_cleanup():
            print("🔄 Memory cleanup needed, performing cleanup...")
            await strategy._aggressive_memory_cleanup()
            
            after_cleanup_memory = strategy._memory_monitor.get_memory_usage()
            print(f"Memory after cleanup: {after_cleanup_memory['current_mb']:.1f}MB ({after_cleanup_memory['percent']:.1f}%)")
        else:
            print("✅ Memory usage is normal, no cleanup needed")
        
        # Test 9: Error Tracking
        print("\n🔍 Test 9: Error Tracking")
        print(f"Total errors tracked: {strategy._error_count}")
        print(f"Error history count: {len(strategy._error_history)}")
        if strategy._error_history:
            recent_errors = strategy._error_history[-5:]  # Last 5 errors
            print("Recent errors:")
            for error in recent_errors:
                print(f"  - {error['type']}: {error['message'][:50]}...")
        
        # Test 10: Optimization Suggestions
        print("\n🔍 Test 10: Optimization Suggestions")
        suggestions = strategy._adaptive_cache_optimizer.get_optimization_suggestions()
        if suggestions:
            print(f"Found {len(suggestions)} optimization suggestions:")
            for suggestion in suggestions[:3]:  # Show first 3
                print(f"  - {suggestion['cache_key']}: {suggestion['reason']}")
        else:
            print("✅ No optimization suggestions needed")
        
        # Test 11: Performance Alerts
        print("\n🔍 Test 11: Performance Alerts")
        alerts = metrics.get('alerts', [])
        if alerts:
            print("🚨 Performance alerts detected:")
            for alert in alerts:
                print(f"  - {alert}")
        else:
            print("✅ No performance alerts - All systems normal")
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 ADVANCED OPTIMIZATION SUMMARY")
        print("=" * 60)
        
        # Memory efficiency
        memory_efficiency = "✅ Good" if final_memory['percent'] < 80 else "⚠️ High"
        print(f"Memory Efficiency: {memory_efficiency} ({final_memory['percent']:.1f}%)")
        
        # Cache efficiency
        cache_efficiency = "✅ Good" if metrics.get('cache_hit_rate', 0) > 70 else "⚠️ Low"
        print(f"Cache Efficiency: {cache_efficiency} ({metrics.get('cache_hit_rate', 0):.1f}%)")
        
        # Error handling
        error_efficiency = "✅ Good" if metrics.get('error_rate', 0) < 5 else "⚠️ High"
        print(f"Error Handling: {error_efficiency} ({metrics.get('error_rate', 0):.1f}%)")
        
        # Processing speed
        speed_efficiency = "✅ Good" if metrics.get('avg_processing_time', 0) < 0.5 else "⚠️ Slow"
        print(f"Processing Speed: {speed_efficiency} ({metrics.get('avg_processing_time', 0):.2f}s)")
        
        # Optimization status
        optimization_status = "✅ Optimized" if metrics.get('optimization_suggestions', 0) == 0 else "🔧 Needs Optimization"
        print(f"Optimization Status: {optimization_status}")
        
        print("\n🎯 RECOMMENDATIONS:")
        
        if final_memory['percent'] > 80:
            print("- Consider reducing cache size or implementing more aggressive cleanup")
        if metrics.get('cache_hit_rate', 0) < 70:
            print("- Consider increasing cache TTL for better hit rates")
        if metrics.get('error_rate', 0) > 5:
            print("- Review error handling and API rate limiting")
        if metrics.get('avg_processing_time', 0) > 0.5:
            print("- Consider optimizing data processing or reducing API calls")
        if metrics.get('optimization_suggestions', 0) > 0:
            print("- Apply suggested cache optimizations")
        
        print("\n✅ Advanced optimization test completed!")
        
        # Cleanup
        await strategy.close()
        await binance_service.close()
        await indicator_service.close()
        
    except Exception as e:
        logger.error(f"Error in advanced optimization test: {str(e)}")
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_advanced_optimizations()) 