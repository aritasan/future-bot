"""
Enhanced Real-Time Performance Dashboard
Advanced dashboard with WebSocket integration and comprehensive monitoring.
"""

import dash
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import asyncio
import threading
import time
import json
import websockets
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPerformanceDashboard:
    """Enhanced real-time performance dashboard with WebSocket integration."""
    
    def __init__(self, websocket_url="ws://localhost:8765"):
        self.websocket_url = websocket_url
        self.metrics_history = []
        self.max_history_points = 1000
        self.websocket_client = None
        self.websocket_connected = False
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        self.setup_callbacks()
        
        # Start WebSocket client
        self.start_websocket_client()
        
    def setup_layout(self):
        """Setup the enhanced dashboard layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🚀 WorldQuant Real-Time Performance Dashboard", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Connection Status
            dbc.Row([
                dbc.Col([
                    dbc.Alert([
                        html.H4("🔌 Connection Status", className="alert-heading"),
                        html.P(id="connection-status", children="Connecting..."),
                    ], id="connection-alert", color="warning", className="mb-3")
                ])
            ]),
            
            # Key Performance Metrics Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Performance Score", className="card-title"),
                            html.H2(id="performance-score", children="0"),
                            html.P("Target: >70", className="text-muted")
                        ])
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Risk Score", className="card-title"),
                            html.H2(id="risk-score", children="0"),
                            html.P("Target: <30", className="text-muted")
                        ])
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Stability Score", className="card-title"),
                            html.H2(id="stability-score", children="0"),
                            html.P("Target: >80", className="text-muted")
                        ])
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Active Alerts", className="card-title"),
                            html.H2(id="alert-count", children="0"),
                            html.P("Critical: 0", className="text-muted")
                        ])
                    ], className="text-center")
                ], width=3)
            ], className="mb-4"),
            
            # System Performance Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("CPU Usage", className="card-title"),
                            html.H2(id="cpu-usage", children="0%"),
                            html.P("Target: <80%", className="text-muted")
                        ])
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Memory Usage", className="card-title"),
                            html.H2(id="memory-usage", children="0%"),
                            html.P("Target: <85%", className="text-muted")
                        ])
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("API Response Time", className="card-title"),
                            html.H2(id="api-response-time", children="0.00s"),
                            html.P("Target: <2.0s", className="text-muted")
                        ])
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Error Rate", className="card-title"),
                            html.H2(id="error-rate", children="0%"),
                            html.P("Target: <5%", className="text-muted")
                        ])
                    ], className="text-center")
                ], width=3)
            ], className="mb-4"),
            
            # Performance Charts Row
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="performance-chart", style={'height': '400px'})
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="system-chart", style={'height': '400px'})
                ], width=6)
            ], className="mb-4"),
            
            # Risk Metrics Charts
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="risk-chart", style={'height': '300px'})
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="volatility-chart", style={'height': '300px'})
                ], width=6)
            ], className="mb-4"),
            
            # Real-Time Alerts and System Status
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🚨 Real-Time Alerts"),
                        dbc.CardBody(id="alerts-content")
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 System Status"),
                        dbc.CardBody(id="system-status-content")
                    ])
                ], width=6)
            ]),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=2*1000,  # 2 seconds for real-time
                n_intervals=0
            ),
            
            # Store for real-time data
            dcc.Store(id='real-time-data-store')
        ], fluid=True)
        
    def setup_callbacks(self):
        """Setup dashboard callbacks for real-time updates."""
        
        @self.app.callback(
            [Output("connection-status", "children"),
             Output("connection-alert", "color")],
            [Input("interval-component", "n_intervals")]
        )
        def update_connection_status(n):
            """Update connection status."""
            if self.websocket_connected:
                return "🟢 Connected to WebSocket Server", "success"
            else:
                return "🔴 Disconnected - Retrying...", "danger"
        
        @self.app.callback(
            [Output("performance-score", "children"),
             Output("risk-score", "children"),
             Output("stability-score", "children"),
             Output("alert-count", "children")],
            [Input("real-time-data-store", "data")]
        )
        def update_performance_scores(data):
            """Update performance score cards."""
            if not data:
                return "0", "0", "0", "0"
            
            try:
                performance_score = data.get('performance_score', 0)
                risk_score = data.get('risk_score', 0)
                stability_score = data.get('stability_score', 0)
                alert_count = data.get('alert_count', 0)
                
                return f"{performance_score:.1f}", f"{risk_score:.1f}", f"{stability_score:.1f}", str(alert_count)
                
            except Exception as e:
                logger.error(f"Error updating performance scores: {str(e)}")
                return "0", "0", "0", "0"
        
        @self.app.callback(
            [Output("cpu-usage", "children"),
             Output("memory-usage", "children"),
             Output("api-response-time", "children"),
             Output("error-rate", "children")],
            [Input("real-time-data-store", "data")]
        )
        def update_system_metrics(data):
            """Update system metrics cards."""
            if not data:
                return "0%", "0%", "0.00s", "0%"
            
            try:
                system_metrics = data.get('system_metrics', {})
                
                cpu_usage = f"{system_metrics.get('cpu_usage', 0):.1f}%"
                memory_usage = f"{system_metrics.get('memory_usage', 0):.1f}%"
                api_response_time = f"{system_metrics.get('api_response_time', 0):.2f}s"
                error_rate = f"{system_metrics.get('error_rate', 0):.1f}%"
                
                return cpu_usage, memory_usage, api_response_time, error_rate
                
            except Exception as e:
                logger.error(f"Error updating system metrics: {str(e)}")
                return "0%", "0%", "0.00s", "0%"
        
        @self.app.callback(
            Output("performance-chart", "figure"),
            [Input("real-time-data-store", "data")]
        )
        def update_performance_chart(data):
            """Update performance chart."""
            if not data:
                return self._create_empty_chart("Performance Metrics")
            
            try:
                self._add_metrics_to_history(data)
                
                df = pd.DataFrame(self.metrics_history)
                if df.empty:
                    return self._create_empty_chart("Performance Metrics")
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Performance Score", "Risk Score"),
                    vertical_spacing=0.1
                )
                
                # Performance score
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['performance_score'],
                        name="Performance Score",
                        line=dict(color='green')
                    ),
                    row=1, col=1
                )
                
                # Risk score
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['risk_score'],
                        name="Risk Score",
                        line=dict(color='red')
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    title="Performance Metrics Over Time"
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating performance chart: {str(e)}")
                return self._create_empty_chart("Performance Metrics")
        
        @self.app.callback(
            Output("system-chart", "figure"),
            [Input("real-time-data-store", "data")]
        )
        def update_system_chart(data):
            """Update system metrics chart."""
            if not data:
                return self._create_empty_chart("System Metrics")
            
            try:
                system_metrics = data.get('system_metrics', {})
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("CPU & Memory Usage", "API Response Time"),
                    vertical_spacing=0.1
                )
                
                # CPU and Memory usage
                fig.add_trace(
                    go.Scatter(
                        x=[datetime.now()],
                        y=[system_metrics.get('cpu_usage', 0)],
                        name="CPU Usage",
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[datetime.now()],
                        y=[system_metrics.get('memory_usage', 0)],
                        name="Memory Usage",
                        line=dict(color='orange')
                    ),
                    row=1, col=1
                )
                
                # API response time
                fig.add_trace(
                    go.Scatter(
                        x=[datetime.now()],
                        y=[system_metrics.get('api_response_time', 0)],
                        name="API Response Time",
                        line=dict(color='purple')
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    title="System Metrics"
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating system chart: {str(e)}")
                return self._create_empty_chart("System Metrics")
        
        @self.app.callback(
            Output("risk-chart", "figure"),
            [Input("real-time-data-store", "data")]
        )
        def update_risk_chart(data):
            """Update risk metrics chart."""
            if not data:
                return self._create_empty_chart("Risk Metrics")
            
            try:
                performance_metrics = data.get('performance_metrics', {})
                
                # Risk metrics
                risk_data = {
                    'Metric': ['Volatility', 'Max Drawdown', 'VaR (95%)', 'CVaR (95%)'],
                    'Value': [
                        performance_metrics.get('volatility', 0) * 100,
                        abs(performance_metrics.get('max_drawdown', 0)) * 100,
                        abs(performance_metrics.get('var', 0)) * 100,
                        abs(performance_metrics.get('cvar', 0)) * 100
                    ]
                }
                
                df = pd.DataFrame(risk_data)
                
                fig = px.bar(df, x='Metric', y='Value', 
                           title="Risk Metrics (%)",
                           color='Value',
                           color_continuous_scale='Reds')
                
                fig.update_layout(height=300)
                return fig
                
            except Exception as e:
                logger.error(f"Error updating risk chart: {str(e)}")
                return self._create_empty_chart("Risk Metrics")
        
        @self.app.callback(
            Output("volatility-chart", "figure"),
            [Input("real-time-data-store", "data")]
        )
        def update_volatility_chart(data):
            """Update volatility chart."""
            if not data:
                return self._create_empty_chart("Volatility Analysis")
            
            try:
                performance_metrics = data.get('performance_metrics', {})
                
                # Volatility gauge
                volatility = performance_metrics.get('volatility', 0) * 100
                
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=volatility,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Portfolio Volatility (%)"},
                    delta={'reference': 20},
                    gauge={
                        'axis': {'range': [None, 50]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 15], 'color': "lightgreen"},
                            {'range': [15, 30], 'color': "yellow"},
                            {'range': [30, 50], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 30
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                return fig
                
            except Exception as e:
                logger.error(f"Error updating volatility chart: {str(e)}")
                return self._create_empty_chart("Volatility Analysis")
        
        @self.app.callback(
            Output("alerts-content", "children"),
            [Input("real-time-data-store", "data")]
        )
        def update_alerts(data):
            """Update alerts content."""
            if not data:
                return html.P("No data available")
            
            try:
                alerts = data.get('alerts', [])
                
                if not alerts:
                    return html.P("✅ No alerts - All systems normal", className="text-success")
                
                alert_items = []
                for alert in alerts:
                    level_color = {
                        'critical': 'danger',
                        'warning': 'warning',
                        'info': 'info'
                    }.get(alert.get('level', 'info'), 'info')
                    
                    alert_items.append(
                        dbc.Alert([
                            html.H6(alert.get('message', 'Unknown alert'), className="alert-heading"),
                            html.Small(f"Level: {alert.get('level', 'unknown')} | Time: {alert.get('timestamp', 'unknown')}")
                        ], color=level_color, className="mb-2")
                    )
                
                return html.Div(alert_items)
                
            except Exception as e:
                logger.error(f"Error updating alerts: {str(e)}")
                return html.P("Error loading alerts", className="text-danger")
        
        @self.app.callback(
            Output("system-status-content", "children"),
            [Input("real-time-data-store", "data")]
        )
        def update_system_status(data):
            """Update system status content."""
            if not data:
                return html.P("No data available")
            
            try:
                system_metrics = data.get('system_metrics', {})
                performance_metrics = data.get('performance_metrics', {})
                
                status_items = []
                
                # System health indicators
                cpu_usage = system_metrics.get('cpu_usage', 0)
                memory_usage = system_metrics.get('memory_usage', 0)
                api_response_time = system_metrics.get('api_response_time', 0)
                error_rate = system_metrics.get('error_rate', 0)
                
                # Performance indicators
                sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
                volatility = performance_metrics.get('volatility', 0)
                drawdown = performance_metrics.get('drawdown', 0)
                
                # System status
                status_items.append(html.H6("System Health", className="mb-2"))
                status_items.append(html.P(f"CPU: {cpu_usage:.1f}% {'🟢' if cpu_usage < 80 else '🔴'}"))
                status_items.append(html.P(f"Memory: {memory_usage:.1f}% {'🟢' if memory_usage < 85 else '🔴'}"))
                status_items.append(html.P(f"API Response: {api_response_time:.2f}s {'🟢' if api_response_time < 2 else '🔴'}"))
                status_items.append(html.P(f"Error Rate: {error_rate:.1f}% {'🟢' if error_rate < 5 else '🔴'}"))
                
                status_items.append(html.Hr())
                status_items.append(html.H6("Performance Health", className="mb-2"))
                status_items.append(html.P(f"Sharpe Ratio: {sharpe_ratio:.2f} {'🟢' if sharpe_ratio > 0.5 else '🔴'}"))
                status_items.append(html.P(f"Volatility: {volatility:.2%} {'🟢' if volatility < 0.25 else '🔴'}"))
                status_items.append(html.P(f"Drawdown: {drawdown:.2%} {'🟢' if abs(drawdown) < 0.1 else '🔴'}"))
                
                return html.Div(status_items)
                
            except Exception as e:
                logger.error(f"Error updating system status: {str(e)}")
                return html.P("Error loading system status", className="text-danger")
        
        @self.app.callback(
            Output("real-time-data-store", "data"),
            [Input("interval-component", "n_intervals")]
        )
        def update_real_time_data(n):
            """Update real-time data store."""
            # This will be updated by WebSocket data
            return self.latest_data if hasattr(self, 'latest_data') else {}
    
    def start_websocket_client(self):
        """Start WebSocket client for real-time data."""
        def websocket_client():
            async def connect():
                try:
                    async with websockets.connect(self.websocket_url) as websocket:
                        self.websocket_client = websocket
                        self.websocket_connected = True
                        logger.info("Connected to WebSocket server")
                        
                        async for message in websocket:
                            try:
                                data = json.loads(message)
                                self.latest_data = data
                                logger.debug(f"Received WebSocket data: {len(str(data))} chars")
                            except json.JSONDecodeError as e:
                                logger.error(f"Error decoding WebSocket message: {str(e)}")
                            except Exception as e:
                                logger.error(f"Error processing WebSocket message: {str(e)}")
                                
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                except Exception as e:
                    logger.error(f"WebSocket connection error: {str(e)}")
                finally:
                    self.websocket_connected = False
                    self.websocket_client = None
            
            # Run WebSocket client in asyncio loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            while True:
                try:
                    loop.run_until_complete(connect())
                except Exception as e:
                    logger.error(f"WebSocket client error: {str(e)}")
                
                # Wait before reconnecting
                time.sleep(5)
        
        # Start WebSocket client in separate thread
        websocket_thread = threading.Thread(target=websocket_client, daemon=True)
        websocket_thread.start()
    
    def _add_metrics_to_history(self, data: dict):
        """Add metrics to history for charting."""
        data['timestamp'] = datetime.now()
        self.metrics_history.append(data)
        
        # Keep only recent history
        if len(self.metrics_history) > self.max_history_points:
            self.metrics_history = self.metrics_history[-self.max_history_points:]
    
    def _create_empty_chart(self, title: str):
        """Create an empty chart when no data is available."""
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(
            title=title,
            height=400
        )
        return fig
    
    def run(self, host='localhost', port=8050, debug=False):
        """Run the enhanced dashboard."""
        logger.info(f"Starting enhanced performance dashboard on http://{host}:{port}")
        logger.info(f"WebSocket server expected at {self.websocket_url}")
        self.app.run_server(host=host, port=port, debug=debug)

def start_enhanced_dashboard(websocket_url="ws://localhost:8765", host='localhost', port=8050):
    """Start the enhanced performance dashboard."""
    dashboard = EnhancedPerformanceDashboard(websocket_url)
    dashboard.run(host=host, port=port)

if __name__ == "__main__":
    # Example usage
    start_enhanced_dashboard() 