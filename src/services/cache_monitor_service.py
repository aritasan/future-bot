"""
Cache Monitor Service for Trading Bot.
Provides monitoring and dashboard for cache performance.
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import psutil

import redis.asyncio as redis
import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg

logger = logging.getLogger(__name__)

class CacheMonitorService:
    """
    Cache Monitor Service for Trading Bot.
    Provides monitoring, alerting, and dashboard for cache performance.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Cache Monitor Service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Get connection URLs
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.nats_url = os.getenv('NATS_URL', 'nats://localhost:4222')
        
        # Initialize connections
        self.redis_client = None
        self.nats_client = None
        
        # Monitoring data
        self.monitoring_data = {
            'cache_stats': {},
            'performance_metrics': [],
            'alerts': [],
            'system_metrics': {},
            'cache_hit_rates': [],
            'response_times': [],
            'memory_usage': [],
            'error_rates': []
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'hit_rate_min': 0.7,  # 70% minimum hit rate
            'response_time_max': 0.1,  # 100ms maximum response time
            'memory_usage_max': 0.8,  # 80% maximum memory usage
            'error_rate_max': 0.05  # 5% maximum error rate
        }
        
        # Performance tracking
        self.performance_history = []
        self.max_history_size = 1000
        
        logger.info("Cache Monitor Service initialized")
    
    async def initialize(self) -> bool:
        """Initialize cache monitor service."""
        try:
            # Initialize Redis connection
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=False,
                max_connections=10
            )
            await self.redis_client.ping()
            logger.info("Redis connection established for monitoring")
            
            # Initialize NATS connection
            self.nats_client = NATS()
            await self.nats_client.connect(self.nats_url)
            logger.info("NATS connection established for monitoring")
            
            # Subscribe to monitoring events
            await self._subscribe_to_monitoring_events()
            
            # Start monitoring tasks
            asyncio.create_task(self._monitor_cache_performance())
            asyncio.create_task(self._monitor_system_metrics())
            asyncio.create_task(self._generate_alerts())
            asyncio.create_task(self._cleanup_old_data())
            
            logger.info("Cache Monitor Service initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing cache monitor service: {str(e)}")
            return False
    
    async def _subscribe_to_monitoring_events(self):
        """Subscribe to monitoring events."""
        try:
            # Subscribe to cache performance events
            await self.nats_client.subscribe(
                "cache.performance",
                cb=self._handle_performance_event
            )
            
            # Subscribe to cache statistics events
            await self.nats_client.subscribe(
                "cache.stats",
                cb=self._handle_stats_event
            )
            
            # Subscribe to cache error events
            await self.nats_client.subscribe(
                "cache.error",
                cb=self._handle_error_event
            )
            
            logger.info("Subscribed to monitoring events")
            
        except Exception as e:
            logger.error(f"Error subscribing to monitoring events: {str(e)}")
    
    async def _handle_performance_event(self, msg: Msg):
        """Handle performance monitoring events."""
        try:
            data = json.loads(msg.data.decode())
            
            # Store performance data
            self.monitoring_data['performance_metrics'].append({
                'timestamp': datetime.now().isoformat(),
                'avg_response_time': data.get('avg_response_time', 0),
                'cache_hit_rate': data.get('cache_hit_rate', 0),
                'memory_usage': data.get('memory_usage', 0)
            })
            
            # Update performance history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'response_time': data.get('avg_response_time', 0),
                'hit_rate': data.get('cache_hit_rate', 0),
                'memory_usage': data.get('memory_usage', 0)
            })
            
            # Keep history size manageable
            if len(self.performance_history) > self.max_history_size:
                self.performance_history = self.performance_history[-self.max_history_size:]
                
        except Exception as e:
            logger.error(f"Error handling performance event: {str(e)}")
    
    async def _handle_stats_event(self, msg: Msg):
        """Handle cache statistics events."""
        try:
            data = json.loads(msg.data.decode())
            
            # Update cache stats
            self.monitoring_data['cache_stats'].update(data)
            
            # Calculate hit rate
            hits = data.get('hits', 0)
            misses = data.get('misses', 0)
            total = hits + misses
            
            if total > 0:
                hit_rate = hits / total
                self.monitoring_data['cache_hit_rates'].append({
                    'timestamp': datetime.now().isoformat(),
                    'hit_rate': hit_rate
                })
                
        except Exception as e:
            logger.error(f"Error handling stats event: {str(e)}")
    
    async def _handle_error_event(self, msg: Msg):
        """Handle cache error events."""
        try:
            data = json.loads(msg.data.decode())
            
            # Store error data
            self.monitoring_data['error_rates'].append({
                'timestamp': datetime.now().isoformat(),
                'error_count': data.get('error_count', 0),
                'error_type': data.get('error_type', 'unknown')
            })
            
            # Generate alert if error rate is high
            if data.get('error_rate', 0) > self.alert_thresholds['error_rate_max']:
                await self._generate_alert('high_error_rate', {
                    'error_rate': data.get('error_rate', 0),
                    'threshold': self.alert_thresholds['error_rate_max']
                })
                
        except Exception as e:
            logger.error(f"Error handling error event: {str(e)}")
    
    async def _monitor_cache_performance(self):
        """Monitor cache performance metrics."""
        while True:
            try:
                # Get Redis info
                if self.redis_client:
                    info = await self.redis_client.info()
                    
                    # Extract relevant metrics
                    metrics = {
                        'used_memory': info.get('used_memory', 0),
                        'connected_clients': info.get('connected_clients', 0),
                        'total_commands_processed': info.get('total_commands_processed', 0),
                        'keyspace_hits': info.get('keyspace_hits', 0),
                        'keyspace_misses': info.get('keyspace_misses', 0),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Calculate hit rate
                    hits = metrics['keyspace_hits']
                    misses = metrics['keyspace_misses']
                    total = hits + misses
                    
                    if total > 0:
                        hit_rate = hits / total
                        metrics['hit_rate'] = hit_rate
                        
                        # Check hit rate threshold
                        if hit_rate < self.alert_thresholds['hit_rate_min']:
                            await self._generate_alert('low_hit_rate', {
                                'hit_rate': hit_rate,
                                'threshold': self.alert_thresholds['hit_rate_min']
                            })
                    
                    # Store metrics
                    self.monitoring_data['system_metrics'] = metrics
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring cache performance: {str(e)}")
                await asyncio.sleep(30)
    
    async def _monitor_system_metrics(self):
        """Monitor system metrics."""
        while True:
            try:
                # Get system memory usage
                memory = psutil.virtual_memory()
                cpu = psutil.cpu_percent()
                
                system_metrics = {
                    'memory_usage_percent': memory.percent,
                    'memory_available': memory.available,
                    'cpu_usage_percent': cpu,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Check memory usage threshold
                if memory.percent > self.alert_thresholds['memory_usage_max'] * 100:
                    await self._generate_alert('high_memory_usage', {
                        'memory_usage': memory.percent,
                        'threshold': self.alert_thresholds['memory_usage_max'] * 100
                    })
                
                # Store system metrics
                self.monitoring_data['system_metrics'].update(system_metrics)
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error monitoring system metrics: {str(e)}")
                await asyncio.sleep(60)
    
    async def _generate_alerts(self):
        """Generate alerts based on thresholds."""
        while True:
            try:
                # Check performance metrics
                if self.performance_history:
                    latest = self.performance_history[-1]
                    
                    # Check response time
                    if latest['response_time'] > self.alert_thresholds['response_time_max']:
                        await self._generate_alert('high_response_time', {
                            'response_time': latest['response_time'],
                            'threshold': self.alert_thresholds['response_time_max']
                        })
                    
                    # Check hit rate
                    if latest['hit_rate'] < self.alert_thresholds['hit_rate_min']:
                        await self._generate_alert('low_hit_rate', {
                            'hit_rate': latest['hit_rate'],
                            'threshold': self.alert_thresholds['hit_rate_min']
                        })
                
                await asyncio.sleep(60)  # Check alerts every minute
                
            except Exception as e:
                logger.error(f"Error generating alerts: {str(e)}")
                await asyncio.sleep(60)
    
    async def _generate_alert(self, alert_type: str, data: Dict):
        """Generate and store alert."""
        try:
            alert = {
                'type': alert_type,
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'severity': self._get_alert_severity(alert_type)
            }
            
            # Store alert
            self.monitoring_data['alerts'].append(alert)
            
            # Publish alert to NATS
            if self.nats_client:
                await self.nats_client.publish(
                    "cache.alert",
                    json.dumps(alert).encode()
                )
            
            logger.warning(f"Cache alert generated: {alert_type} - {data}")
            
        except Exception as e:
            logger.error(f"Error generating alert: {str(e)}")
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Get alert severity level."""
        severity_map = {
            'high_error_rate': 'critical',
            'high_memory_usage': 'warning',
            'high_response_time': 'warning',
            'low_hit_rate': 'warning'
        }
        return severity_map.get(alert_type, 'info')
    
    async def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        while True:
            try:
                # Keep only last 24 hours of data
                cutoff_time = datetime.now() - timedelta(hours=24)
                
                # Clean up performance metrics
                self.monitoring_data['performance_metrics'] = [
                    m for m in self.monitoring_data['performance_metrics']
                    if datetime.fromisoformat(m['timestamp']) > cutoff_time
                ]
                
                # Clean up alerts (keep last 100)
                self.monitoring_data['alerts'] = self.monitoring_data['alerts'][-100:]
                
                # Clean up performance history
                self.performance_history = [
                    p for p in self.performance_history
                    if p['timestamp'] > cutoff_time
                ]
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Error cleaning up old data: {str(e)}")
                await asyncio.sleep(3600)
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for monitoring."""
        try:
            # Calculate current metrics
            current_metrics = {}
            if self.performance_history:
                latest = self.performance_history[-1]
                current_metrics = {
                    'current_hit_rate': latest['hit_rate'],
                    'current_response_time': latest['response_time'],
                    'current_memory_usage': latest['memory_usage']
                }
            
            # Calculate averages
            if self.performance_history:
                avg_response_time = sum(p['response_time'] for p in self.performance_history) / len(self.performance_history)
                avg_hit_rate = sum(p['hit_rate'] for p in self.performance_history) / len(self.performance_history)
                avg_memory_usage = sum(p['memory_usage'] for p in self.performance_history) / len(self.performance_history)
            else:
                avg_response_time = avg_hit_rate = avg_memory_usage = 0
            
            # Get recent alerts
            recent_alerts = self.monitoring_data['alerts'][-10:]  # Last 10 alerts
            
            return {
                'current_metrics': current_metrics,
                'average_metrics': {
                    'avg_response_time': avg_response_time,
                    'avg_hit_rate': avg_hit_rate,
                    'avg_memory_usage': avg_memory_usage
                },
                'system_metrics': self.monitoring_data['system_metrics'],
                'recent_alerts': recent_alerts,
                'cache_stats': self.monitoring_data['cache_stats'],
                'performance_history': self.performance_history[-100:],  # Last 100 entries
                'alert_thresholds': self.alert_thresholds
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {str(e)}")
            return {}
    
    async def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance report for specified hours."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filter data for specified time period
            period_data = [
                p for p in self.performance_history
                if p['timestamp'] > cutoff_time
            ]
            
            if not period_data:
                return {'error': 'No data available for specified period'}
            
            # Calculate statistics
            response_times = [p['response_time'] for p in period_data]
            hit_rates = [p['hit_rate'] for p in period_data]
            memory_usage = [p['memory_usage'] for p in period_data]
            
            report = {
                'period_hours': hours,
                'data_points': len(period_data),
                'response_time': {
                    'min': min(response_times),
                    'max': max(response_times),
                    'avg': sum(response_times) / len(response_times),
                    'p95': sorted(response_times)[int(len(response_times) * 0.95)]
                },
                'hit_rate': {
                    'min': min(hit_rates),
                    'max': max(hit_rates),
                    'avg': sum(hit_rates) / len(hit_rates)
                },
                'memory_usage': {
                    'min': min(memory_usage),
                    'max': max(memory_usage),
                    'avg': sum(memory_usage) / len(memory_usage)
                },
                'alerts_in_period': len([
                    a for a in self.monitoring_data['alerts']
                    if datetime.fromisoformat(a['timestamp']) > cutoff_time
                ])
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error getting performance report: {str(e)}")
            return {'error': str(e)}
    
    async def update_alert_thresholds(self, thresholds: Dict[str, float]) -> bool:
        """Update alert thresholds."""
        try:
            self.alert_thresholds.update(thresholds)
            logger.info(f"Updated alert thresholds: {thresholds}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating alert thresholds: {str(e)}")
            return False
    
    async def close(self):
        """Close cache monitor service."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            if self.nats_client:
                await self.nats_client.close()
            
            logger.info("Cache Monitor Service closed")
            
        except Exception as e:
            logger.error(f"Error closing cache monitor service: {str(e)}") 