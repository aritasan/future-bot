import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import asyncio
import threading
import time
import json
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceDashboard:
    """Real-time performance dashboard for trading strategy monitoring."""
    
    def __init__(self, strategy_instance=None):
        self.strategy = strategy_instance
        self.metrics_history = []
        self.max_history_points = 1000
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🚀 Trading Strategy Performance Dashboard", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Key Metrics Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Cache Hit Rate", className="card-title"),
                            html.H2(id="cache-hit-rate", children="0%"),
                            html.P("Target: >70%", className="text-muted")
                        ])
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Processing Time", className="card-title"),
                            html.H2(id="processing-time", children="0.00s"),
                            html.P("Target: <0.5s", className="text-muted")
                        ])
                    ], className="text-center")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Memory Usage", className="card-title"),
                            html.H2(id="memory-usage", children="0%"),
                            html.P("Target: <80%", className="text-muted")
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
            
            # Charts Row
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="performance-chart", style={'height': '400px'})
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="memory-chart", style={'height': '400px'})
                ], width=6)
            ], className="mb-4"),
            
            # Cache and API Metrics
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="cache-chart", style={'height': '300px'})
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="api-chart", style={'height': '300px'})
                ], width=6)
            ], className="mb-4"),
            
            # Alerts and Optimization Suggestions
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🚨 Performance Alerts"),
                        dbc.CardBody(id="alerts-content")
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔧 Optimization Suggestions"),
                        dbc.CardBody(id="optimization-content")
                    ])
                ], width=6)
            ]),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # 5 seconds
                n_intervals=0
            )
        ], fluid=True)
        
    def setup_callbacks(self):
        """Setup dashboard callbacks for real-time updates."""
        
        @self.app.callback(
            [Output("cache-hit-rate", "children"),
             Output("processing-time", "children"),
             Output("memory-usage", "children"),
             Output("error-rate", "children")],
            [Input("interval-component", "n_intervals")]
        )
        def update_key_metrics(n):
            """Update key metrics cards."""
            if not self.strategy:
                return "0%", "0.00s", "0%", "0%"
                
            try:
                metrics = asyncio.run(self.strategy.get_performance_metrics())
                
                cache_hit_rate = f"{metrics.get('cache_hit_rate', 0):.1f}%"
                processing_time = f"{metrics.get('avg_processing_time', 0):.2f}s"
                memory_usage = f"{metrics.get('memory_percent', 0):.1f}%"
                error_rate = f"{metrics.get('error_rate', 0):.1f}%"
                
                return cache_hit_rate, processing_time, memory_usage, error_rate
                
            except Exception as e:
                logger.error(f"Error updating key metrics: {str(e)}")
                return "0%", "0.00s", "0%", "0%"
        
        @self.app.callback(
            Output("performance-chart", "figure"),
            [Input("interval-component", "n_intervals")]
        )
        def update_performance_chart(n):
            """Update performance chart."""
            if not self.strategy:
                return self._create_empty_chart("Performance Metrics")
                
            try:
                metrics = asyncio.run(self.strategy.get_performance_metrics())
                self._add_metrics_to_history(metrics)
                
                df = pd.DataFrame(self.metrics_history)
                if df.empty:
                    return self._create_empty_chart("Performance Metrics")
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Cache Hit Rate", "Processing Time"),
                    vertical_spacing=0.1
                )
                
                # Cache hit rate
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['cache_hit_rate'],
                        name="Cache Hit Rate",
                        line=dict(color='green')
                    ),
                    row=1, col=1
                )
                
                # Processing time
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['avg_processing_time'],
                        name="Processing Time (s)",
                        line=dict(color='blue')
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
            Output("memory-chart", "figure"),
            [Input("interval-component", "n_intervals")]
        )
        def update_memory_chart(n):
            """Update memory usage chart."""
            if not self.strategy:
                return self._create_empty_chart("Memory Usage")
                
            try:
                metrics = asyncio.run(self.strategy.get_performance_metrics())
                
                fig = go.Figure()
                
                # Memory usage gauge
                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=metrics.get('memory_percent', 0),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Memory Usage (%)"},
                    delta={'reference': 80},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))
                
                fig.update_layout(height=400)
                return fig
                
            except Exception as e:
                logger.error(f"Error updating memory chart: {str(e)}")
                return self._create_empty_chart("Memory Usage")
        
        @self.app.callback(
            Output("cache-chart", "figure"),
            [Input("interval-component", "n_intervals")]
        )
        def update_cache_chart(n):
            """Update cache statistics chart."""
            if not self.strategy:
                return self._create_empty_chart("Cache Statistics")
                
            try:
                metrics = asyncio.run(self.strategy.get_performance_metrics())
                
                # Cache statistics
                cache_data = {
                    'Metric': ['Cache Size', 'Cache Usage', 'Cache Hits', 'Cache Misses'],
                    'Value': [
                        metrics.get('cache_size', 0),
                        metrics.get('cache_usage_percent', 0),
                        metrics.get('cache_hits', 0),
                        metrics.get('cache_misses', 0)
                    ]
                }
                
                df = pd.DataFrame(cache_data)
                
                fig = px.bar(df, x='Metric', y='Value', 
                           title="Cache Statistics",
                           color='Value',
                           color_continuous_scale='viridis')
                
                fig.update_layout(height=300)
                return fig
                
            except Exception as e:
                logger.error(f"Error updating cache chart: {str(e)}")
                return self._create_empty_chart("Cache Statistics")
        
        @self.app.callback(
            Output("api-chart", "figure"),
            [Input("interval-component", "n_intervals")]
        )
        def update_api_chart(n):
            """Update API statistics chart."""
            if not self.strategy:
                return self._create_empty_chart("API Statistics")
                
            try:
                metrics = asyncio.run(self.strategy.get_performance_metrics())
                
                # API statistics
                api_data = {
                    'Metric': ['API Calls', 'API Errors', 'Error Rate'],
                    'Value': [
                        metrics.get('api_calls', 0),
                        metrics.get('api_errors', 0),
                        metrics.get('error_rate', 0)
                    ]
                }
                
                df = pd.DataFrame(api_data)
                
                fig = px.bar(df, x='Metric', y='Value',
                           title="API Statistics",
                           color='Value',
                           color_continuous_scale='plasma')
                
                fig.update_layout(height=300)
                return fig
                
            except Exception as e:
                logger.error(f"Error updating API chart: {str(e)}")
                return self._create_empty_chart("API Statistics")
        
        @self.app.callback(
            Output("alerts-content", "children"),
            [Input("interval-component", "n_intervals")]
        )
        def update_alerts(n):
            """Update alerts content."""
            if not self.strategy:
                return html.P("No strategy instance available")
                
            try:
                metrics = asyncio.run(self.strategy.get_performance_metrics())
                alerts = metrics.get('alerts', [])
                
                if not alerts:
                    return html.P("✅ No alerts - All systems normal", className="text-success")
                
                alert_items = []
                for alert in alerts:
                    alert_items.append(html.Li(alert, className="text-danger"))
                
                return html.Ul(alert_items)
                
            except Exception as e:
                logger.error(f"Error updating alerts: {str(e)}")
                return html.P("Error loading alerts", className="text-danger")
        
        @self.app.callback(
            Output("optimization-content", "children"),
            [Input("interval-component", "n_intervals")]
        )
        def update_optimization_suggestions(n):
            """Update optimization suggestions."""
            if not self.strategy:
                return html.P("No strategy instance available")
                
            try:
                metrics = asyncio.run(self.strategy.get_performance_metrics())
                suggestions_count = metrics.get('optimization_suggestions', 0)
                
                if suggestions_count == 0:
                    return html.P("✅ No optimization suggestions needed", className="text-success")
                else:
                    return html.P(f"🔧 {suggestions_count} optimization suggestions available", 
                                className="text-warning")
                
            except Exception as e:
                logger.error(f"Error updating optimization suggestions: {str(e)}")
                return html.P("Error loading suggestions", className="text-danger")
    
    def _add_metrics_to_history(self, metrics: dict):
        """Add metrics to history for charting."""
        metrics['timestamp'] = datetime.now()
        self.metrics_history.append(metrics)
        
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
        """Run the dashboard."""
        logger.info(f"Starting performance dashboard on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

def start_dashboard(strategy_instance=None, host='localhost', port=8050):
    """Start the performance dashboard."""
    dashboard = PerformanceDashboard(strategy_instance)
    dashboard.run(host=host, port=port)

if __name__ == "__main__":
    # Example usage
    start_dashboard() 