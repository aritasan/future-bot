"""
Cache Service for Trading Bot.
Integrates with Advanced Cache Manager for distributed caching.
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.utils.advanced_cache_manager import AdvancedCacheManager

logger = logging.getLogger(__name__)

class CacheService:
    """
    Cache Service for Trading Bot.
    Provides caching capabilities for market data, signals, and analysis results.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Cache Service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Get cache configuration
        self.cache_config = {
            'default_ttl': int(os.getenv('CACHE_TTL', 3600)),
            'max_size': int(os.getenv('CACHE_MAX_SIZE', 1000)),
            'compression_enabled': True,
            'distributed_cache_enabled': True,
            'cache_layers': ['memory', 'redis', 'distributed']
        }
        
        # Initialize cache manager
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        nats_url = os.getenv('NATS_URL', 'nats://localhost:4222')
        
        self.cache_manager = AdvancedCacheManager(
            redis_url=redis_url,
            nats_url=nats_url,
            cache_config=self.cache_config
        )
        
        # Cache keys patterns
        self.cache_patterns = {
            'market_data': 'market_data:{symbol}:{timeframe}',
            'signals': 'signals:{symbol}:{timestamp}',
            'analysis': 'analysis:{symbol}:{type}',
            'indicators': 'indicators:{symbol}:{timeframe}',
            'portfolio': 'portfolio:{account_id}',
            'risk_metrics': 'risk:{symbol}:{metric}',
            'ml_predictions': 'ml:{symbol}:{model}',
            'order_book': 'orderbook:{symbol}',
            'trades': 'trades:{symbol}:{limit}'
        }
        
        logger.info("Cache Service initialized")
    
    async def initialize(self) -> bool:
        """Initialize cache service."""
        try:
            # Initialize cache manager
            success = await self.cache_manager.initialize()
            if success:
                logger.info("Cache Service initialization completed")
                return True
            else:
                logger.error("Failed to initialize cache manager")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing cache service: {str(e)}")
            return False
    
    async def cache_market_data(self, symbol: str, timeframe: str, data: Any, ttl: int = None) -> bool:
        """
        Cache market data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (1h, 4h, 1d, etc.)
            data: Market data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            key = self.cache_patterns['market_data'].format(symbol=symbol, timeframe=timeframe)
            return await self.cache_manager.set(key, data, ttl)
            
        except Exception as e:
            logger.error(f"Error caching market data for {symbol}: {str(e)}")
            return False
    
    async def get_market_data(self, symbol: str, timeframe: str) -> Optional[Any]:
        """
        Get cached market data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Cached market data or None
        """
        try:
            key = self.cache_patterns['market_data'].format(symbol=symbol, timeframe=timeframe)
            return await self.cache_manager.get(key)
            
        except Exception as e:
            logger.error(f"Error getting cached market data for {symbol}: {str(e)}")
            return None
    
    async def cache_signal(self, symbol: str, signal: Dict, ttl: int = None) -> bool:
        """
        Cache trading signal.
        
        Args:
            symbol: Trading symbol
            signal: Signal data
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            key = self.cache_patterns['signals'].format(symbol=symbol, timestamp=timestamp)
            return await self.cache_manager.set(key, signal, ttl)
            
        except Exception as e:
            logger.error(f"Error caching signal for {symbol}: {str(e)}")
            return False
    
    async def get_latest_signal(self, symbol: str) -> Optional[Dict]:
        """
        Get latest cached signal for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Latest signal or None
        """
        try:
            # Get all signal keys for symbol
            pattern = self.cache_patterns['signals'].format(symbol=symbol, timestamp='*')
            # This would require Redis SCAN command implementation
            # For now, return None
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest signal for {symbol}: {str(e)}")
            return None
    
    async def cache_analysis(self, symbol: str, analysis_type: str, data: Any, ttl: int = None) -> bool:
        """
        Cache analysis results.
        
        Args:
            symbol: Trading symbol
            analysis_type: Type of analysis
            data: Analysis data
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            key = self.cache_patterns['analysis'].format(symbol=symbol, type=analysis_type)
            return await self.cache_manager.set(key, data, ttl)
            
        except Exception as e:
            logger.error(f"Error caching analysis for {symbol}: {str(e)}")
            return False
    
    async def get_analysis(self, symbol: str, analysis_type: str) -> Optional[Any]:
        """
        Get cached analysis.
        
        Args:
            symbol: Trading symbol
            analysis_type: Type of analysis
            
        Returns:
            Cached analysis or None
        """
        try:
            key = self.cache_patterns['analysis'].format(symbol=symbol, type=analysis_type)
            return await self.cache_manager.get(key)
            
        except Exception as e:
            logger.error(f"Error getting cached analysis for {symbol}: {str(e)}")
            return None
    
    async def cache_indicators(self, symbol: str, timeframe: str, indicators: Dict, ttl: int = None) -> bool:
        """
        Cache technical indicators.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            indicators: Indicators data
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            key = self.cache_patterns['indicators'].format(symbol=symbol, timeframe=timeframe)
            return await self.cache_manager.set(key, indicators, ttl)
            
        except Exception as e:
            logger.error(f"Error caching indicators for {symbol}: {str(e)}")
            return False
    
    async def get_indicators(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """
        Get cached indicators.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Cached indicators or None
        """
        try:
            key = self.cache_patterns['indicators'].format(symbol=symbol, timeframe=timeframe)
            return await self.cache_manager.get(key)
            
        except Exception as e:
            logger.error(f"Error getting cached indicators for {symbol}: {str(e)}")
            return None
    
    async def cache_portfolio(self, account_id: str, portfolio_data: Dict, ttl: int = None) -> bool:
        """
        Cache portfolio data.
        
        Args:
            account_id: Account ID
            portfolio_data: Portfolio data
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            key = self.cache_patterns['portfolio'].format(account_id=account_id)
            return await self.cache_manager.set(key, portfolio_data, ttl)
            
        except Exception as e:
            logger.error(f"Error caching portfolio for {account_id}: {str(e)}")
            return False
    
    async def get_portfolio(self, account_id: str) -> Optional[Dict]:
        """
        Get cached portfolio data.
        
        Args:
            account_id: Account ID
            
        Returns:
            Cached portfolio data or None
        """
        try:
            key = self.cache_patterns['portfolio'].format(account_id=account_id)
            return await self.cache_manager.get(key)
            
        except Exception as e:
            logger.error(f"Error getting cached portfolio for {account_id}: {str(e)}")
            return None
    
    async def cache_risk_metrics(self, symbol: str, metric: str, data: Any, ttl: int = None) -> bool:
        """
        Cache risk metrics.
        
        Args:
            symbol: Trading symbol
            metric: Risk metric name
            data: Risk metric data
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            key = self.cache_patterns['risk_metrics'].format(symbol=symbol, metric=metric)
            return await self.cache_manager.set(key, data, ttl)
            
        except Exception as e:
            logger.error(f"Error caching risk metrics for {symbol}: {str(e)}")
            return False
    
    async def get_risk_metrics(self, symbol: str, metric: str) -> Optional[Any]:
        """
        Get cached risk metrics.
        
        Args:
            symbol: Trading symbol
            metric: Risk metric name
            
        Returns:
            Cached risk metrics or None
        """
        try:
            key = self.cache_patterns['risk_metrics'].format(symbol=symbol, metric=metric)
            return await self.cache_manager.get(key)
            
        except Exception as e:
            logger.error(f"Error getting cached risk metrics for {symbol}: {str(e)}")
            return None
    
    async def cache_ml_prediction(self, symbol: str, model: str, prediction: Dict, ttl: int = None) -> bool:
        """
        Cache ML prediction.
        
        Args:
            symbol: Trading symbol
            model: ML model name
            prediction: Prediction data
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            key = self.cache_patterns['ml_predictions'].format(symbol=symbol, model=model)
            return await self.cache_manager.set(key, prediction, ttl)
            
        except Exception as e:
            logger.error(f"Error caching ML prediction for {symbol}: {str(e)}")
            return False
    
    async def get_ml_prediction(self, symbol: str, model: str) -> Optional[Dict]:
        """
        Get cached ML prediction.
        
        Args:
            symbol: Trading symbol
            model: ML model name
            
        Returns:
            Cached ML prediction or None
        """
        try:
            key = self.cache_patterns['ml_predictions'].format(symbol=symbol, model=model)
            return await self.cache_manager.get(key)
            
        except Exception as e:
            logger.error(f"Error getting cached ML prediction for {symbol}: {str(e)}")
            return None
    
    async def cache_order_book(self, symbol: str, order_book: Dict, ttl: int = None) -> bool:
        """
        Cache order book data.
        
        Args:
            symbol: Trading symbol
            order_book: Order book data
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            key = self.cache_patterns['order_book'].format(symbol=symbol)
            return await self.cache_manager.set(key, order_book, ttl)
            
        except Exception as e:
            logger.error(f"Error caching order book for {symbol}: {str(e)}")
            return False
    
    async def get_order_book(self, symbol: str) -> Optional[Dict]:
        """
        Get cached order book.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Cached order book or None
        """
        try:
            key = self.cache_patterns['order_book'].format(symbol=symbol)
            return await self.cache_manager.get(key)
            
        except Exception as e:
            logger.error(f"Error getting cached order book for {symbol}: {str(e)}")
            return None
    
    async def invalidate_symbol_cache(self, symbol: str) -> bool:
        """
        Invalidate all cache entries for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if successful
        """
        try:
            pattern = f"*{symbol}*"
            return await self.cache_manager.invalidate_pattern(pattern)
            
        except Exception as e:
            logger.error(f"Error invalidating cache for {symbol}: {str(e)}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        try:
            return await self.cache_manager.get_stats()
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {}
    
    async def clear_all_cache(self) -> bool:
        """
        Clear all cache.
        
        Returns:
            True if successful
        """
        try:
            # Clear all cache patterns
            for pattern_name, pattern in self.cache_patterns.items():
                # Extract base pattern without placeholders
                base_pattern = pattern.split('{')[0]
                await self.cache_manager.invalidate_pattern(f"{base_pattern}*")
            
            logger.info("All cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing all cache: {str(e)}")
            return False
    
    async def cache_portfolio_analysis(self, analysis_type: str, data: Any, ttl: int = None) -> bool:
        """
        Cache portfolio analysis data.
        
        Args:
            analysis_type: Type of analysis (optimization, factors, etc.)
            data: Analysis data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            key = f"portfolio_analysis:{analysis_type}"
            return await self.cache_manager.set(key, data, ttl)
            
        except Exception as e:
            logger.error(f"Error caching portfolio analysis {analysis_type}: {str(e)}")
            return False
    
    async def get_portfolio_analysis(self, analysis_type: str) -> Optional[Any]:
        """
        Get cached portfolio analysis data.
        
        Args:
            analysis_type: Type of analysis (optimization, factors, etc.)
            
        Returns:
            Cached analysis data or None
        """
        try:
            key = f"portfolio_analysis:{analysis_type}"
            return await self.cache_manager.get(key)
            
        except Exception as e:
            logger.error(f"Error getting portfolio analysis {analysis_type}: {str(e)}")
            return None
    
    async def cache_performance_metrics(self, metrics: Dict, ttl: int = None) -> bool:
        """
        Cache performance metrics.
        
        Args:
            metrics: Performance metrics to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            key = "performance_metrics"
            return await self.cache_manager.set(key, metrics, ttl)
            
        except Exception as e:
            logger.error(f"Error caching performance metrics: {str(e)}")
            return False
    
    async def get_performance_metrics(self) -> Optional[Dict]:
        """
        Get cached performance metrics.
        
        Returns:
            Cached performance metrics or None
        """
        try:
            key = "performance_metrics"
            return await self.cache_manager.get(key)
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return None
    
    async def close(self):
        """Close cache service."""
        try:
            await self.cache_manager.close()
            logger.info("Cache Service closed")
            
        except Exception as e:
            logger.error(f"Error closing cache service: {str(e)}") 