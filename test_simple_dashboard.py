#!/usr/bin/env python3
"""
Simple dashboard test with WebSocket connection
"""

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import asyncio
import websockets
import json
import threading
import time

class SimpleDashboard:
    def __init__(self, websocket_url="ws://localhost:8766"):
        self.websocket_url = websocket_url
        self.websocket_connected = False
        self.latest_data = {
            "performance_score": 0,
            "risk_score": 0,
            "stability_score": 0
        }
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        self.setup_callbacks()
        
        # Start WebSocket client
        self.start_websocket_client()
        
    def setup_layout(self):
        """Setup the simple dashboard layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🧪 Simple WebSocket Test Dashboard", 
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
            
            # Performance Metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Performance Score", className="card-title"),
                            html.H2(id="performance-score", children="0"),
                        ])
                    ], className="text-center")
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Risk Score", className="card-title"),
                            html.H2(id="risk-score", children="0"),
                        ])
                    ], className="text-center")
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Stability Score", className="card-title"),
                            html.H2(id="stability-score", children="0"),
                        ])
                    ], className="text-center")
                ], width=4),
            ]),
            
            # Data Display
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Latest Data", className="card-title"),
                            html.Pre(id="data-display", children="No data received")
                        ])
                    ])
                ])
            ]),
            
            # Hidden div for storing data
            html.Div(id="data-store", style={"display": "none"}),
            
            # Interval component for updates
            dcc.Interval(
                id="interval-component",
                interval=1*1000,  # 1 second
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
            if self.websocket_connected:
                return "Connected", "success"
            else:
                return "Disconnected - Retrying...", "danger"
        
        @self.app.callback(
            [Output("performance-score", "children"),
             Output("risk-score", "children"),
             Output("stability-score", "children")],
            [Input("data-store", "data")]
        )
        def update_metrics(data):
            if data:
                return (
                    f"{data.get('performance_score', 0):.1f}",
                    f"{data.get('risk_score', 0):.1f}",
                    f"{data.get('stability_score', 0):.1f}"
                )
            return "0", "0", "0"
        
        @self.app.callback(
            Output("data-display", "children"),
            [Input("data-store", "data")]
        )
        def update_data_display(data):
            if data:
                return json.dumps(data, indent=2)
            return "No data received"
        
        @self.app.callback(
            Output("data-store", "data"),
            [Input("interval-component", "n_intervals")]
        )
        def update_data_store(n):
            return self.latest_data
        
    def start_websocket_client(self):
        """Start WebSocket client in a separate thread."""
        def websocket_client():
            async def connect():
                while True:
                    try:
                        async with websockets.connect(self.websocket_url) as websocket:
                            print(f"✅ Connected to WebSocket server: {self.websocket_url}")
                            self.websocket_connected = True
                            
                            async for message in websocket:
                                try:
                                    data = json.loads(message)
                                    print(f"📊 Received data: {data}")
                                    self.latest_data = data
                                except json.JSONDecodeError as e:
                                    print(f"❌ Error parsing JSON: {e}")
                                    
                    except Exception as e:
                        print(f"❌ WebSocket error: {e}")
                        self.websocket_connected = False
                        await asyncio.sleep(5)  # Wait before retrying
            
            asyncio.run(connect())
        
        # Start WebSocket client in a separate thread
        thread = threading.Thread(target=websocket_client, daemon=True)
        thread.start()
        
    def run(self, host='localhost', port=8051, debug=False):
        """Run the dashboard."""
        print(f"🚀 Starting Simple Dashboard on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

def main():
    """Main function."""
    dashboard = SimpleDashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 