#!/usr/bin/env python3
"""
Performance Dashboard with HTTP Polling
Dashboard that polls performance data from HTTP API instead of WebSocket
"""

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import requests
import json
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceDashboardHTTP:
    """Performance dashboard using HTTP polling."""
    
    def __init__(self, api_url="http://localhost:8000/api/performance"):
        self.api_url = api_url
        self.metrics_history = []
        self.max_history_points = 1000
        self.api_connected = False
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🚀 WorldQuant Performance Dashboard (HTTP Polling)", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Connection Status
            dbc.Row([
                dbc.Col([
                    dbc.Alert([
                        html.H4("🔌 API Connection Status", className="alert-heading"),
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
                ], width=3),
            ]),
            
            # System Metrics Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("System Status", className="card-title"),
                            html.H2(id="system-status", children="Initializing"),
                            html.P("API: Active", className="text-muted")
                        ])
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Last Update", className="card-title"),
                            html.H2(id="last-update", children="--"),
                            html.P("Real-time", className="text-muted")
                        ])
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Data Points", className="card-title"),
                            html.H2(id="data-points", children="0"),
                            html.P("History", className="text-muted")
                        ])
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("API Response", className="card-title"),
                            html.H2(id="api-response", children="--"),
                            html.P("Status", className="text-muted")
                        ])
                    ], className="text-center")
                ], width=3),
            ]),
            
            # Performance Charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Performance Metrics Over Time", className="card-title"),
                            dcc.Graph(id="performance-chart", style={'height': '400px'})
                        ])
                    ])
                ], width=12),
            ], className="mt-4"),
            
            # System Status and Alerts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("System Status & Alerts", className="card-title"),
                            html.Div(id="system-status-content", children="No alerts")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Raw API Data", className="card-title"),
                            html.Pre(id="raw-data", children="No data received", 
                                   style={'max-height': '300px', 'overflow': 'auto'})
                        ])
                    ])
                ], width=6),
            ], className="mt-4"),
            
            # Hidden div for storing data
            html.Div(id="data-store", style={"display": "none"}),
            
            # Interval component for polling
            dcc.Interval(
                id="interval-component",
                interval=5*1000,  # 5 seconds
                n_intervals=0
            )
        ])
        
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output("connection-status", "children"),
             Output("connection-alert", "color")],
            [Input("interval-component", "n_intervals")]
        )
        def update_connection_status(n):
            if self.api_connected:
                return "Connected to API", "success"
            else:
                return "Disconnected - Retrying...", "danger"
        
        @self.app.callback(
            [Output("performance-score", "children"),
             Output("risk-score", "children"),
             Output("stability-score", "children"),
             Output("alert-count", "children")],
            [Input("data-store", "data")]
        )
        def update_performance_scores(data):
            if data and isinstance(data, dict):
                return (
                    f"{data.get('performance_score', 0):.1f}",
                    f"{data.get('risk_score', 0):.1f}",
                    f"{data.get('stability_score', 0):.1f}",
                    f"{data.get('alerts_count', 0)}"
                )
            return "0", "0", "0", "0"
        
        @self.app.callback(
            [Output("system-status", "children"),
             Output("last-update", "children"),
             Output("data-points", "children"),
             Output("api-response", "children")],
            [Input("data-store", "data")]
        )
        def update_system_metrics(data):
            if data and isinstance(data, dict):
                status = data.get('system_status', 'unknown')
                last_update = data.get('last_update', '--')
                data_points = len(self.metrics_history)
                api_response = "OK" if self.api_connected else "Error"
                
                return status, last_update, data_points, api_response
            return "Unknown", "--", "0", "Error"
        
        @self.app.callback(
            Output("performance-chart", "figure"),
            [Input("data-store", "data")]
        )
        def update_performance_chart(data):
            if not self.metrics_history:
                return self._create_empty_chart("Performance Metrics")
            
            df = pd.DataFrame(self.metrics_history)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Performance Score', 'Risk Score', 
                              'Stability Score', 'Alerts Count'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            if 'performance_score' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=df['performance_score'],
                              name='Performance Score', line=dict(color='green')),
                    row=1, col=1
                )
            
            if 'risk_score' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=df['risk_score'],
                              name='Risk Score', line=dict(color='red')),
                    row=1, col=2
                )
            
            if 'stability_score' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=df['stability_score'],
                              name='Stability Score', line=dict(color='blue')),
                    row=2, col=1
                )
            
            if 'alerts_count' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=df['alerts_count'],
                              name='Alerts Count', line=dict(color='orange')),
                    row=2, col=2
                )
            
            fig.update_layout(height=400, showlegend=True)
            return fig
        
        @self.app.callback(
            Output("system-status-content", "children"),
            [Input("data-store", "data")]
        )
        def update_system_status(data):
            if not data or not isinstance(data, dict):
                return "No data available"
            
            status_items = []
            
            # System status
            system_status = data.get('system_status', 'unknown')
            status_color = 'success' if system_status == 'active' else 'warning'
            status_items.append(
                dbc.Alert(f"System Status: {system_status}", color=status_color, className="mb-2")
            )
            
            # Alerts
            alerts_count = data.get('alerts_count', 0)
            if alerts_count > 0:
                status_items.append(
                    dbc.Alert(f"Active Alerts: {alerts_count}", color="danger", className="mb-2")
                )
            else:
                status_items.append(
                    dbc.Alert("No active alerts", color="success", className="mb-2")
                )
            
            # Timestamp
            timestamp = data.get('timestamp', 'Unknown')
            status_items.append(
                html.P(f"Last Update: {timestamp}", className="text-muted")
            )
            
            return status_items
        
        @self.app.callback(
            Output("raw-data", "children"),
            [Input("data-store", "data")]
        )
        def update_raw_data(data):
            if data:
                return json.dumps(data, indent=2)
            return "No data received"
        
        @self.app.callback(
            Output("data-store", "data"),
            [Input("interval-component", "n_intervals")]
        )
        def poll_performance_data(n):
            """Poll performance data from API."""
            try:
                response = requests.get(self.api_url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    self.api_connected = True
                    
                    # Add timestamp if not present
                    if 'timestamp' not in data:
                        data['timestamp'] = datetime.now().isoformat()
                    
                    # Add to history
                    self._add_metrics_to_history(data)
                    
                    return data
                else:
                    self.api_connected = False
                    logger.error(f"API returned status code: {response.status_code}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                self.api_connected = False
                logger.error(f"Error polling API: {str(e)}")
                return None
            except Exception as e:
                self.api_connected = False
                logger.error(f"Unexpected error: {str(e)}")
                return None
    
    def _add_metrics_to_history(self, data: dict):
        """Add metrics to history."""
        if isinstance(data, dict):
            # Add timestamp if not present
            if 'timestamp' not in data:
                data['timestamp'] = datetime.now().isoformat()
            
            self.metrics_history.append(data)
            
            # Keep only recent data
            if len(self.metrics_history) > self.max_history_points:
                self.metrics_history = self.metrics_history[-self.max_history_points:]
    
    def _create_empty_chart(self, title: str):
        """Create empty chart."""
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title=title, height=400)
        return fig
    
    def run(self, host='localhost', port=8050, debug=False):
        """Run the dashboard."""
        logger.info(f"🚀 Starting Performance Dashboard (HTTP Polling) on http://{host}:{port}")
        logger.info(f"📊 Polling API: {self.api_url}")
        self.app.run(host=host, port=port, debug=debug)

def start_http_polling_dashboard(api_url="http://localhost:8000/api/performance", 
                                host='localhost', port=8050):
    """Start the HTTP polling dashboard."""
    dashboard = PerformanceDashboardHTTP(api_url)
    dashboard.run(host=host, port=port)

if __name__ == "__main__":
    start_http_polling_dashboard() 