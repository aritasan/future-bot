import asyncio
import sys
import os
import time
import json
from typing import Dict, Any

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

async def optimize_cache_configuration():
    """Optimize cache configuration based on performance analysis."""
    try:
        print("🔧 Cache Configuration Optimizer")
        print("=" * 50)
        
        # Load configuration
        config = load_config()
        
        # Initialize services
        binance_service = BinanceService(config)
        await binance_service.initialize()
        
        indicator_service = IndicatorService(config)
        await indicator_service.initialize()
        print("✅ Indicator service initialized")
        
        notification_service = NotificationService(config)
        
        # Initialize strategy
        strategy = EnhancedTradingStrategy(config, binance_service, indicator_service, notification_service)
        await strategy.initialize()
        
        # Test symbols for optimization
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT']
        
        print(f"📊 Testing with {len(test_symbols)} symbols...")
        
        # Test different cache configurations
        cache_configs = {
            'conservative': {
                'timeframe_data': 30,      # 30 seconds
                'market_structure': 180,    # 3 minutes
                'volume_profile': 300,      # 5 minutes
                'funding_rate': 180,        # 3 minutes
                'open_interest': 180,       # 3 minutes
                'order_book': 15,           # 15 seconds
                'btc_volatility': 180,      # 3 minutes
                'altcoin_correlation': 300, # 5 minutes
                'sentiment': 180,           # 3 minutes
                'signal_score': 30,         # 30 seconds
                'market_conditions': 60     # 1 minute
            },
            'balanced': {
                'timeframe_data': 60,      # 1 minute
                'market_structure': 300,    # 5 minutes
                'volume_profile': 600,      # 10 minutes
                'funding_rate': 300,        # 5 minutes
                'open_interest': 300,       # 5 minutes
                'order_book': 30,           # 30 seconds
                'btc_volatility': 300,      # 5 minutes
                'altcoin_correlation': 600, # 10 minutes
                'sentiment': 300,           # 5 minutes
                'signal_score': 60,         # 1 minute
                'market_conditions': 120    # 2 minutes
            },
            'aggressive': {
                'timeframe_data': 120,     # 2 minutes
                'market_structure': 600,    # 10 minutes
                'volume_profile': 1200,     # 20 minutes
                'funding_rate': 600,        # 10 minutes
                'open_interest': 600,       # 10 minutes
                'order_book': 60,           # 1 minute
                'btc_volatility': 600,      # 10 minutes
                'altcoin_correlation': 1200, # 20 minutes
                'sentiment': 600,           # 10 minutes
                'signal_score': 120,        # 2 minutes
                'market_conditions': 300    # 5 minutes
            }
        }
        
        results = {}
        
        for config_name, cache_ttl in cache_configs.items():
            print(f"\n🔍 Testing {config_name} configuration...")
            
            # Apply cache configuration
            strategy._cache_ttl = cache_ttl.copy()
            strategy._cache.clear()  # Clear existing cache
            
            # Test performance
            start_time = time.time()
            
            # Process all symbols
            for symbol in test_symbols:
                await strategy.generate_signals(symbol, indicator_service)
            
            processing_time = time.time() - start_time
            
            # Get metrics
            metrics = await strategy.get_performance_metrics()
            
            results[config_name] = {
                'processing_time': processing_time,
                'api_calls': metrics.get('api_calls', 0),
                'cache_hits': metrics.get('cache_hits', 0),
                'cache_misses': metrics.get('cache_misses', 0),
                'cache_hit_rate': metrics.get('cache_hit_rate', 0),
                'avg_processing_time': metrics.get('avg_processing_time', 0)
            }
            
            print(f"⏱️  Processing time: {processing_time:.2f}s")
            print(f"📈 Cache hit rate: {metrics.get('cache_hit_rate', 0):.1f}%")
            print(f"📈 API calls: {metrics.get('api_calls', 0)}")
        
        # Analyze results and recommend optimal configuration
        print("\n" + "=" * 50)
        print("📊 CONFIGURATION ANALYSIS")
        print("=" * 50)
        
        best_config = None
        best_score = float('inf')
        
        for config_name, result in results.items():
            # Calculate score (lower is better)
            # Weight: processing_time (40%), cache_hit_rate (30%), api_calls (30%)
            score = (
                result['processing_time'] * 0.4 +
                (100 - result['cache_hit_rate']) * 0.3 +
                (result['api_calls'] / 100) * 0.3
            )
            
            print(f"\n{config_name.upper()}:")
            print(f"  ⏱️  Processing time: {result['processing_time']:.2f}s")
            print(f"  📈 Cache hit rate: {result['cache_hit_rate']:.1f}%")
            print(f"  📈 API calls: {result['api_calls']}")
            print(f"  🎯 Score: {score:.2f}")
            
            if score < best_score:
                best_score = score
                best_config = config_name
        
        print(f"\n🏆 RECOMMENDED CONFIGURATION: {best_config.upper()}")
        
        # Generate optimized configuration
        optimal_config = cache_configs[best_config].copy()
        
        # Fine-tune based on results
        if results[best_config]['cache_hit_rate'] < 70:
            # Increase TTL for better hit rate
            for key in optimal_config:
                optimal_config[key] = int(optimal_config[key] * 1.2)
            print("🔧 Increased TTL for better cache hit rate")
        
        if results[best_config]['processing_time'] > 10:
            # Decrease TTL for faster processing
            for key in optimal_config:
                optimal_config[key] = int(optimal_config[key] * 0.8)
            print("🔧 Decreased TTL for faster processing")
        
        # Save optimized configuration
        config_file = 'optimized_cache_config.json'
        with open(config_file, 'w') as f:
            json.dump({
                'cache_ttl': optimal_config,
                'recommendation': best_config,
                'analysis_results': results
            }, f, indent=2)
        
        print(f"\n💾 Optimized configuration saved to {config_file}")
        
        # Apply optimal configuration
        strategy._cache_ttl = optimal_config
        print("✅ Optimal configuration applied")
        
        # Final test with optimal configuration
        print("\n🔍 Final test with optimal configuration...")
        strategy._cache.clear()
        
        start_time = time.time()
        for symbol in test_symbols:
            await strategy.generate_signals(symbol, indicator_service)
        final_time = time.time() - start_time
        
        final_metrics = await strategy.get_performance_metrics()
        
        print(f"⏱️  Final processing time: {final_time:.2f}s")
        print(f"📈 Final cache hit rate: {final_metrics.get('cache_hit_rate', 0):.1f}%")
        print(f"📈 Final API calls: {final_metrics.get('api_calls', 0)}")
        
        print("\n✅ Cache optimization completed!")
        
    except Exception as e:
        print(f"❌ Optimization failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(optimize_cache_configuration()) 