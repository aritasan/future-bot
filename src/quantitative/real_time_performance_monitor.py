"""
WorldQuant Real-Time Performance Monitor
Advanced real-time performance monitoring with WebSocket integration.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict, deque
import json
import warnings
import websockets
import threading
import time
import psutil
import os

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class WorldQuantRealTimePerformanceMonitor:
    """
    WorldQuant-level real-time performance monitor with WebSocket integration.
    """
    
    def __init__(self):
        """Initialize WorldQuant Real-Time Performance Monitor."""
        
        # Performance metrics storage
        self.performance_metrics = {
            'returns': deque(maxlen=1000),
            'volatility': deque(maxlen=1000),
            'sharpe_ratio': deque(maxlen=1000),
            'drawdown': deque(maxlen=1000),
            'var': deque(maxlen=1000),
            'cvar': deque(maxlen=1000),
            'beta': deque(maxlen=1000),
            'correlation': deque(maxlen=1000),
            'tracking_error': deque(maxlen=1000),
            'information_ratio': deque(maxlen=1000),
            'calmar_ratio': deque(maxlen=1000),
            'sortino_ratio': deque(maxlen=1000),
            'max_drawdown': deque(maxlen=1000),
            'win_rate': deque(maxlen=1000),
            'profit_factor': deque(maxlen=1000),
            'recovery_factor': deque(maxlen=1000),
            'risk_adjusted_return': deque(maxlen=1000),
        }
        
        # System performance metrics
        self.system_metrics = {
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'disk_usage': deque(maxlen=100),
            'network_io': deque(maxlen=100),
            'api_response_time': deque(maxlen=100),
            'cache_hit_rate': deque(maxlen=100),
            'error_rate': deque(maxlen=100),
            'processing_time': deque(maxlen=100)
        }
        
        # Real-time monitoring state
        self.monitoring_state = {
            'active': False,
            'last_update': None,
            'update_frequency': 5,  # 5 seconds for real-time
            'alert_count': 0,
            'performance_score': 0.0,
            'risk_score': 0.0,
            'stability_score': 0.0,
            'websocket_clients': set(),
            'monitoring_task': None,
            'websocket_port': None # Added for storing the port
        }
        
        # Alert thresholds - WorldQuant Standards
        self.alert_thresholds = {
            'drawdown_exceeded': {'threshold': 0.10, 'level': 'warning'},
            'volatility_spike': {'threshold': 0.25, 'level': 'warning'},
            'sharpe_decline': {'threshold': 0.5, 'level': 'warning'},
            'rebalancing_needed': {'threshold': 0.1, 'level': 'info'},
            'var_exceeded': {'threshold': 0.15, 'level': 'critical'},
            'correlation_spike': {'threshold': 0.8, 'level': 'warning'},
            'cpu_high': {'threshold': 80.0, 'level': 'warning'},
            'memory_high': {'threshold': 85.0, 'level': 'warning'},
            'api_slow': {'threshold': 2.0, 'level': 'warning'},
            'error_rate_high': {'threshold': 5.0, 'level': 'critical'}
        }
        
        # Performance history for trend analysis
        self.performance_history = {
            'hourly': deque(maxlen=24),
            'daily': deque(maxlen=30),
            'weekly': deque(maxlen=12)
        }
        
        logger.info("WorldQuantRealTimePerformanceMonitor initialized")
    
    async def initialize(self) -> bool:
        """Initialize the real-time performance monitor."""
        try:
            await self._initialize_default_metrics()
            await self._start_real_time_monitoring()
            await self._start_websocket_server()
            logger.info("WorldQuantRealTimePerformanceMonitor initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing WorldQuantRealTimePerformanceMonitor: {str(e)}")
            return False
    
    async def _initialize_default_metrics(self) -> None:
        """Initialize default performance metrics."""
        try:
            default_metrics = {
                'returns': 0.0, 'volatility': 0.0, 'sharpe_ratio': 0.0,
                'drawdown': 0.0, 'var': 0.0, 'cvar': 0.0, 'beta': 1.0,
                'correlation': 0.0, 'tracking_error': 0.0, 'information_ratio': 0.0,
                'calmar_ratio': 0.0, 'sortino_ratio': 0.0, 'max_drawdown': 0.0,
                'win_rate': 0.5, 'profit_factor': 1.0, 'recovery_factor': 0.0,
                'risk_adjusted_return': 0.0
            }
            
            for metric, value in default_metrics.items():
                self.performance_metrics[metric].append(value)
                
        except Exception as e:
            logger.error(f"Error initializing default metrics: {str(e)}")
    
    async def _start_real_time_monitoring(self) -> None:
        """Start real-time monitoring loop."""
        try:
            self.monitoring_state['active'] = True
            self.monitoring_state['last_update'] = datetime.now()
            
            # Start monitoring task
            self.monitoring_state['monitoring_task'] = asyncio.create_task(
                self._real_time_monitoring_loop()
            )
            
            logger.info("Real-time performance monitoring started")
        except Exception as e:
            logger.error(f"Error starting real-time monitoring: {str(e)}")
    
    async def _real_time_monitoring_loop(self) -> None:
        """Real-time monitoring loop with system metrics."""
        try:
            while self.monitoring_state['active']:
                try:
                    # Update system metrics
                    await self._update_system_metrics()
                    
                    # Update performance metrics
                    await self.update_metrics()
                    
                    # Check for alerts
                    alerts = await self.check_alerts()
                    if alerts:
                        await self._handle_real_time_alerts(alerts)
                    
                    # Broadcast to WebSocket clients
                    await self._broadcast_performance_data()
                    
                    # Log performance summary
                    await self._log_performance_summary()
                    
                    # Wait for next update
                    await asyncio.sleep(self.monitoring_state['update_frequency'])
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in real-time monitoring loop: {str(e)}")
                    await asyncio.sleep(10)  # Wait on error
                    
        except Exception as e:
            logger.error(f"Fatal error in real-time monitoring: {str(e)}")
    
    async def _update_system_metrics(self) -> None:
        """Update system performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_metrics['cpu_usage'].append(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.system_metrics['memory_usage'].append(memory_percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.system_metrics['disk_usage'].append(disk_percent)
            
            # Network I/O (simplified)
            network_io = 0.0  # Placeholder for network monitoring
            self.system_metrics['network_io'].append(network_io)
            
            # API response time (simulated)
            api_response_time = np.random.exponential(0.5)  # Simulated
            self.system_metrics['api_response_time'].append(api_response_time)
            
            # Cache hit rate (simulated)
            cache_hit_rate = np.random.uniform(70, 95)  # Simulated
            self.system_metrics['cache_hit_rate'].append(cache_hit_rate)
            
            # Error rate (simulated)
            error_rate = np.random.uniform(0, 3)  # Simulated
            self.system_metrics['error_rate'].append(error_rate)
            
            # Processing time (simulated)
            processing_time = np.random.exponential(0.3)  # Simulated
            self.system_metrics['processing_time'].append(processing_time)
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {str(e)}")
    
    async def _start_websocket_server(self) -> None:
        """Start WebSocket server for real-time data broadcasting."""
        try:
            # Start WebSocket server in background
            asyncio.create_task(self._websocket_server())
            logger.info("WebSocket server started for real-time monitoring")
        except Exception as e:
            logger.error(f"Error starting WebSocket server: {str(e)}")
    
    async def _websocket_server(self) -> None:
        """WebSocket server for real-time performance data."""
        try:
            async def websocket_handler(websocket, path):
                """Handle WebSocket connections."""
                try:
                    self.monitoring_state['websocket_clients'].add(websocket)
                    logger.info(f"WebSocket client connected: {websocket.remote_address}")
                    
                    # Send initial data
                    try:
                        logger.info("Preparing initial data...")
                        initial_data = await self.get_real_time_summary()
                        logger.info(f"Initial data: {initial_data}")
                        
                        # Ensure data is serializable
                        json_data = json.dumps(initial_data, default=str)
                        logger.info(f"JSON data: {json_data}")
                        
                        await websocket.send(json_data)
                        logger.info("Sent initial data successfully")
                    except Exception as e:
                        logger.error(f"Error sending initial data: {str(e)}")
                        # Send fallback data
                        fallback_data = {
                            'performance_score': 0.0,
                            'risk_score': 0.0,
                            'stability_score': 0.0,
                            'timestamp': datetime.now().isoformat(),
                            'status': 'fallback'
                        }
                        await websocket.send(json.dumps(fallback_data))
                        logger.info("Sent fallback data")
                    
                    # Keep connection alive
                    while True:
                        try:
                            await asyncio.sleep(5)  # Update every 5 seconds
                            
                            # Send performance data
                            data = await self.get_real_time_summary()
                            json_data = json.dumps(data, default=str)
                            await websocket.send(json_data)
                            
                        except websockets.exceptions.ConnectionClosed:
                            logger.info("WebSocket connection closed by client")
                            break
                        except Exception as e:
                            logger.error(f"WebSocket error: {str(e)}")
                            break
                            
                except Exception as e:
                    logger.error(f"WebSocket handler error: {str(e)}")
                finally:
                    self.monitoring_state['websocket_clients'].discard(websocket)
                    logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
            
            # Try different ports if the default port is busy
            ports_to_try = [8765, 8766, 8767, 8768, 8769]
            server = None
            selected_port = None
            
            for port in ports_to_try:
                try:
                    server = await websockets.serve(websocket_handler, "localhost", port)
                    selected_port = port
                    logger.info(f"WebSocket server running on ws://localhost:{port}")
                    break
                except OSError as e:
                    if "Address already in use" in str(e) or "Only one usage" in str(e):
                        logger.warning(f"Port {port} is busy, trying next port...")
                        continue
                    else:
                        raise e
            
            if server is None:
                logger.error("Could not start WebSocket server on any available port")
                return
            
            # Store the selected port for dashboard connection
            self.monitoring_state['websocket_port'] = selected_port
            
            # Keep server running
            await server.wait_closed()
            
        except Exception as e:
            logger.error(f"Error in WebSocket server: {str(e)}")
    
    async def _broadcast_performance_data(self) -> None:
        """Broadcast performance data to all WebSocket clients."""
        try:
            if not self.monitoring_state['websocket_clients']:
                return
            
            data = await self.get_real_time_summary()
            message = json.dumps(data, default=str)
            
            # Broadcast to all clients
            disconnected_clients = set()
            for client in self.monitoring_state['websocket_clients']:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
                except Exception as e:
                    logger.error(f"Error broadcasting to client: {str(e)}")
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.monitoring_state['websocket_clients'] -= disconnected_clients
            
        except Exception as e:
            logger.error(f"Error broadcasting performance data: {str(e)}")
    
    async def _handle_real_time_alerts(self, alerts: List[Dict]) -> None:
        """Handle real-time performance alerts."""
        try:
            for alert in alerts:
                # Log alert
                logger.warning(f"Performance Alert: {alert['message']}")
                
                # Send to WebSocket clients
                alert_data = {
                    'type': 'alert',
                    'data': alert,
                    'timestamp': datetime.now().isoformat()
                }
                
                message = json.dumps(alert_data, default=str)
                for client in self.monitoring_state['websocket_clients']:
                    try:
                        await client.send(message)
                    except Exception as e:
                        logger.error(f"Error sending alert to client: {str(e)}")
                
                # Handle critical alerts
                if alert['level'] == 'critical':
                    await self._handle_critical_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error handling real-time alerts: {str(e)}")
    
    async def _handle_critical_alert(self, alert: Dict) -> None:
        """Handle critical performance alerts."""
        try:
            logger.critical(f"CRITICAL ALERT: {alert['message']}")
            
            # Implement emergency actions
            if alert['type'] == 'var_exceeded':
                await self._emergency_risk_reduction()
            elif alert['type'] == 'error_rate_high':
                await self._emergency_system_check()
                
        except Exception as e:
            logger.error(f"Error handling critical alert: {str(e)}")
    
    async def _emergency_risk_reduction(self) -> None:
        """Emergency risk reduction actions."""
        try:
            logger.warning("Executing emergency risk reduction")
            # Implement emergency risk reduction logic
        except Exception as e:
            logger.error(f"Error in emergency risk reduction: {str(e)}")
    
    async def _emergency_system_check(self) -> None:
        """Emergency system health check."""
        try:
            logger.warning("Executing emergency system check")
            # Implement emergency system check logic
        except Exception as e:
            logger.error(f"Error in emergency system check: {str(e)}")
    
    async def update_metrics(self, portfolio_data: Optional[Dict] = None) -> None:
        """Update performance metrics with new data."""
        try:
            if portfolio_data is None:
                portfolio_data = await self._get_default_portfolio_data()
            
            metrics = await self._calculate_performance_metrics(portfolio_data)
            
            for metric_name, value in metrics.items():
                if metric_name in self.performance_metrics:
                    self.performance_metrics[metric_name].append(value)
            
            self.monitoring_state['last_update'] = datetime.now()
            await self._calculate_performance_scores()
            
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
    
    async def _get_default_portfolio_data(self) -> Dict:
        """Get default portfolio data for testing."""
        try:
            return {
                'timestamp': datetime.now(),
                'total_value': 100000.0,
                'returns': np.random.normal(0.001, 0.02),
                'positions': {
                    'BTCUSDT': {'weight': 0.3, 'return': np.random.normal(0.002, 0.03)},
                    'ETHUSDT': {'weight': 0.25, 'return': np.random.normal(0.0015, 0.025)},
                    'BNBUSDT': {'weight': 0.2, 'return': np.random.normal(0.001, 0.02)},
                    'ADAUSDT': {'weight': 0.15, 'return': np.random.normal(0.0005, 0.015)},
                    'DOTUSDT': {'weight': 0.1, 'return': np.random.normal(0.0008, 0.018)}
                },
                'benchmark_return': np.random.normal(0.0008, 0.015),
                'risk_free_rate': 0.02
            }
        except Exception as e:
            logger.error(f"Error getting default portfolio data: {str(e)}")
            return {}
    
    async def _calculate_performance_metrics(self, portfolio_data: Dict) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        try:
            metrics = {}
            total_return = portfolio_data.get('returns', 0.0)
            risk_free_rate = portfolio_data.get('risk_free_rate', 0.02)
            
            returns_list = list(self.performance_metrics['returns'])
            if len(returns_list) > 1:
                volatility = np.std(returns_list[-30:]) * np.sqrt(252) if len(returns_list) >= 30 else np.std(returns_list) * np.sqrt(252)
                excess_return = total_return - risk_free_rate / 252
                sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0
                
                cumulative_returns = np.cumprod(1 + np.array(returns_list))
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / running_max
                current_drawdown = drawdown[-1] if len(drawdown) > 0 else 0.0
                max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
                
                var_95 = np.percentile(returns_list, 5)
                cvar_95 = np.mean([r for r in returns_list if r <= var_95]) if var_95 < 0 else 0.0
                
                metrics.update({
                    'returns': total_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'drawdown': current_drawdown,
                    'max_drawdown': max_drawdown,
                    'var': var_95,
                    'cvar': cvar_95,
                    'beta': 1.0,
                    'correlation': 0.0,
                    'tracking_error': 0.0,
                    'information_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'calmar_ratio': 0.0,
                    'win_rate': 0.5,
                    'profit_factor': 1.0,
                    'recovery_factor': 0.0,
                    'risk_adjusted_return': total_return
                })
            else:
                metrics.update({
                    'returns': total_return,
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'drawdown': 0.0,
                    'max_drawdown': 0.0,
                    'var': 0.0,
                    'cvar': 0.0,
                    'beta': 1.0,
                    'correlation': 0.0,
                    'tracking_error': 0.0,
                    'information_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'calmar_ratio': 0.0,
                    'win_rate': 0.5,
                    'profit_factor': 1.0,
                    'recovery_factor': 0.0,
                    'risk_adjusted_return': total_return
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    async def _calculate_performance_scores(self) -> None:
        """Calculate performance, risk, and stability scores."""
        try:
            sharpe_ratio = self.performance_metrics['sharpe_ratio'][-1] if self.performance_metrics['sharpe_ratio'] else 0.0
            information_ratio = self.performance_metrics['information_ratio'][-1] if self.performance_metrics['information_ratio'] else 0.0
            win_rate = self.performance_metrics['win_rate'][-1] if self.performance_metrics['win_rate'] else 0.5
            
            performance_score = (
                min(max(sharpe_ratio * 20, 0), 40) +
                min(max(information_ratio * 15, 0), 30) +
                min(max(win_rate * 30, 0), 30)
            )
            
            volatility = self.performance_metrics['volatility'][-1] if self.performance_metrics['volatility'] else 0.0
            max_drawdown = abs(self.performance_metrics['max_drawdown'][-1] if self.performance_metrics['max_drawdown'] else 0.0)
            var = abs(self.performance_metrics['var'][-1] if self.performance_metrics['var'] else 0.0)
            
            risk_score = (
                min(max(volatility * 100, 0), 40) +
                min(max(max_drawdown * 100, 0), 30) +
                min(max(var * 100, 0), 30)
            )
            
            tracking_error = self.performance_metrics['tracking_error'][-1] if self.performance_metrics['tracking_error'] else 0.0
            correlation = self.performance_metrics['correlation'][-1] if self.performance_metrics['correlation'] else 0.0
            
            stability_score = (
                max(0, 50 - tracking_error * 100) +
                max(0, 50 - abs(correlation) * 50)
            )
            
            self.monitoring_state['performance_score'] = performance_score
            self.monitoring_state['risk_score'] = risk_score
            self.monitoring_state['stability_score'] = stability_score
            
        except Exception as e:
            logger.error(f"Error calculating performance scores: {str(e)}")
    
    async def check_alerts(self) -> List[Dict]:
        """Check for performance alerts including system metrics."""
        try:
            alerts = []
            
            # Performance metrics alerts
            current_metrics = {
                'drawdown': abs(self.performance_metrics['drawdown'][-1] if self.performance_metrics['drawdown'] else 0.0),
                'volatility': self.performance_metrics['volatility'][-1] if self.performance_metrics['volatility'] else 0.0,
                'sharpe_ratio': self.performance_metrics['sharpe_ratio'][-1] if self.performance_metrics['sharpe_ratio'] else 0.0,
                'var': abs(self.performance_metrics['var'][-1] if self.performance_metrics['var'] else 0.0),
                'correlation': abs(self.performance_metrics['correlation'][-1] if self.performance_metrics['correlation'] else 0.0)
            }
            
            # System metrics alerts
            system_metrics = {
                'cpu_high': self.system_metrics['cpu_usage'][-1] if self.system_metrics['cpu_usage'] else 0.0,
                'memory_high': self.system_metrics['memory_usage'][-1] if self.system_metrics['memory_usage'] else 0.0,
                'api_slow': self.system_metrics['api_response_time'][-1] if self.system_metrics['api_response_time'] else 0.0,
                'error_rate_high': self.system_metrics['error_rate'][-1] if self.system_metrics['error_rate'] else 0.0
            }
            
            # Check performance alerts
            for alert_type, threshold_config in self.alert_thresholds.items():
                if alert_type in ['cpu_high', 'memory_high', 'api_slow', 'error_rate_high']:
                    continue  # Handle system alerts separately
                
                threshold = threshold_config['threshold']
                level = threshold_config['level']
                
                metric_name = alert_type.split('_')[0]
                current_value = current_metrics.get(metric_name, 0.0)
                
                if current_value > threshold:
                    alert = {
                        'type': alert_type,
                        'level': level,
                        'current_value': current_value,
                        'threshold': threshold,
                        'message': f"{alert_type.replace('_', ' ').title()}: {current_value:.4f} > {threshold:.4f}",
                        'timestamp': datetime.now()
                    }
                    alerts.append(alert)
            
            # Check system alerts
            for alert_type, threshold_config in self.alert_thresholds.items():
                if alert_type in ['cpu_high', 'memory_high', 'api_slow', 'error_rate_high']:
                    threshold = threshold_config['threshold']
                    level = threshold_config['level']
                    
                    current_value = system_metrics.get(alert_type, 0.0)
                    
                    if current_value > threshold:
                        alert = {
                            'type': alert_type,
                            'level': level,
                            'current_value': current_value,
                            'threshold': threshold,
                            'message': f"{alert_type.replace('_', ' ').title()}: {current_value:.2f}% > {threshold:.2f}%",
                            'timestamp': datetime.now()
                        }
                        alerts.append(alert)
            
            self.monitoring_state['alert_count'] = len(alerts)
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking alerts: {str(e)}")
            return []
    
    async def get_real_time_summary(self) -> Dict[str, Any]:
        """Get simplified real-time summary for WebSocket."""
        try:
            # Create simple, serializable data structure
            summary = {
                'performance_score': float(self.monitoring_state.get('performance_score', 0.0)),
                'risk_score': float(self.monitoring_state.get('risk_score', 0.0)),
                'stability_score': float(self.monitoring_state.get('stability_score', 0.0)),
                'timestamp': datetime.now().isoformat(),
                'alerts_count': int(self.monitoring_state.get('alert_count', 0)),
                'system_status': 'active' if self.monitoring_state.get('active', False) else 'inactive',
                'websocket_clients': len(self.monitoring_state.get('websocket_clients', set())),
                'last_update': self.monitoring_state.get('last_update', datetime.now().isoformat())
            }
            
            # Ensure last_update is a string
            if isinstance(summary['last_update'], datetime):
                summary['last_update'] = summary['last_update'].isoformat()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting real-time summary: {str(e)}")
            return {
                'performance_score': 0.0,
                'risk_score': 0.0,
                'stability_score': 0.0,
                'timestamp': datetime.now().isoformat(),
                'alerts_count': 0,
                'system_status': 'error',
                'websocket_clients': 0,
                'last_update': datetime.now().isoformat(),
                'error': str(e)
            }
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            latest_metrics = {}
            for metric_name, metric_queue in self.performance_metrics.items():
                if metric_queue:
                    latest_metrics[metric_name] = metric_queue[-1]
            
            summary = {
                'total_return': latest_metrics.get('returns', 0.0),
                'volatility': latest_metrics.get('volatility', 0.0),
                'sharpe_ratio': latest_metrics.get('sharpe_ratio', 0.0),
                'max_drawdown': latest_metrics.get('max_drawdown', 0.0),
                'var_95': latest_metrics.get('var', 0.0),
                'cvar_95': latest_metrics.get('cvar', 0.0),
                'beta': latest_metrics.get('beta', 1.0),
                'information_ratio': latest_metrics.get('information_ratio', 0.0),
                'sortino_ratio': latest_metrics.get('sortino_ratio', 0.0),
                'calmar_ratio': latest_metrics.get('calmar_ratio', 0.0),
                'win_rate': latest_metrics.get('win_rate', 0.5),
                'profit_factor': latest_metrics.get('profit_factor', 1.0),
                'recovery_factor': latest_metrics.get('recovery_factor', 0.0),
                'risk_adjusted_return': latest_metrics.get('risk_adjusted_return', 0.0),
                'performance_score': self.monitoring_state['performance_score'],
                'risk_score': self.monitoring_state['risk_score'],
                'stability_score': self.monitoring_state['stability_score'],
                'alert_count': self.monitoring_state['alert_count'],
                'last_update': self.monitoring_state['last_update']
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {}
    
    async def _log_performance_summary(self) -> None:
        """Log performance summary for monitoring."""
        try:
            summary = await self.get_real_time_summary()
            logger.info(f"Performance Summary - Score: {summary.get('performance_score', 0):.2f}, "
                       f"Risk: {summary.get('risk_score', 0):.2f}, "
                       f"Stability: {summary.get('stability_score', 0):.2f}")
        except Exception as e:
            logger.error(f"Error logging performance summary: {str(e)}")
    
    async def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        try:
            self.monitoring_state['active'] = False
            
            if self.monitoring_state['monitoring_task']:
                self.monitoring_state['monitoring_task'].cancel()
            
            # Close WebSocket connections
            for client in self.monitoring_state['websocket_clients']:
                try:
                    await client.close()
                except Exception as e:
                    logger.error(f"Error closing WebSocket client: {str(e)}")
            
            self.monitoring_state['websocket_clients'].clear()
            
            logger.info("Real-time performance monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}") 